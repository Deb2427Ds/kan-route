"""
dataset.py — COCO-grounded synthetic dataset generation for KAN-Route.

Pipeline:
    1. Load COCO 2017 Val annotations → image_id → object list
    2. For each image, query a teacher LLM to generate routing queries
    3. Parse structured JSON output with multi-level fallback
    4. Save raw dataset JSON for embedding in the next stage

Supports: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct (and compatible models)
"""

import os
import json
import time
import random
from pathlib import Path
from tqdm.auto import tqdm

from .tools import TOOLS, TOOL2ID, TOOL_DESCRIPTIONS, TOOL_BATCHES


# ── COCO loading ──────────────────────────────────────────────────────────────

def load_coco_metadata(annotations_path: str) -> tuple:
    """
    Load COCO 2017 Val annotations and return image metadata.

    Args:
        annotations_path : Path to instances_val2017.json

    Returns:
        (img_ids, img_meta) where img_meta maps img_id → {file_name, objects}
    """
    from pycocotools.coco import COCO

    print("Loading COCO annotations...")
    coco = COCO(annotations_path)

    cat_id2name = {cat["id"]: cat["name"] for cat in coco.loadCats(coco.getCatIds())}
    img_meta = {}

    for img_id in coco.getImgIds():
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns    = coco.loadAnns(ann_ids)
        cats    = list({cat_id2name[a["category_id"]] for a in anns})
        img_info = coco.loadImgs(img_id)[0]

        # Only keep images with at least 2 distinct object categories
        if len(cats) >= 2:
            img_meta[img_id] = {
                "file_name": img_info["file_name"],
                "objects":   cats,
            }

    img_ids = list(img_meta.keys())
    print(f"✅ {len(img_ids)} usable images (≥2 distinct categories)")
    return img_ids, img_meta


# ── JSON parsing ──────────────────────────────────────────────────────────────

def parse_json_output(raw_text: str, tool_batch: list) -> list:
    """
    Parse LLM JSON output, validate tool names against current batch.
    Handles markdown fences, partial JSON, and malformed arrays.

    Returns list of {query, tool, tool_id} dicts.
    """
    import re

    try:
        raw = raw_text.strip()
        # Strip markdown fences
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        # Find JSON array bounds
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        data = json.loads(raw[start:end])
        results = []
        for item in data:
            tool  = item.get("tool",  "").strip()
            query = item.get("query", "").strip()
            if tool in tool_batch and tool in TOOL2ID and query:
                results.append({
                    "query"  : query,
                    "tool"   : tool,
                    "tool_id": TOOL2ID[tool],
                })
        return results
    except Exception:
        return []


# ── Prompt building ───────────────────────────────────────────────────────────

def build_chat_prompt(objects: list, tool_batch: list, tokenizer) -> str:
    """Build a chat-formatted prompt for LLM query generation."""
    tool_str = "\n".join(
        f"  {i}. {t} — {TOOL_DESCRIPTIONS[t]}"
        for i, t in enumerate(tool_batch)
    )
    obj_str = ", ".join(objects[:6]) if objects else "various objects"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise JSON generator for an AI benchmark dataset. "
                "You ALWAYS output valid JSON and nothing else. "
                "No markdown, no explanation, no code blocks."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Image contains: {obj_str}\n\n"
                f"Available tools:\n{tool_str}\n\n"
                "Pick the 2 most relevant tools and write one realistic user query per tool. "
                "Use EXACT tool names.\n\n"
                'Output ONLY this JSON:\n'
                '[{"query": "...", "tool": "exact_tool_name"}, '
                '{"query": "...", "tool": "exact_tool_name"}]'
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ── Batched generation ────────────────────────────────────────────────────────

def generate_queries_batched(
    objects_list: list,
    tool_batch: list,
    teacher,
    tokenizer,
    device: str = "cuda",
    batch_size: int = 8,
    max_new_tokens: int = 256,
) -> list:
    """
    Run batched LLM generation for a list of images and one tool batch.

    Returns list of lists — one inner list of results per image.
    """
    import torch

    all_results = [[] for _ in range(len(objects_list))]
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [build_chat_prompt(objs, tool_batch, tokenizer) for objs in objects_list]

    for start in range(0, len(prompts), batch_size):
        batch_prompts  = prompts[start : start + batch_size]
        batch_indices  = list(range(start, min(start + batch_size, len(prompts))))

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            output_ids = teacher.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        for out, orig_idx in zip(output_ids, batch_indices):
            raw_text = tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
            parsed   = parse_json_output(raw_text, tool_batch)
            all_results[orig_idx].extend(parsed)

    return all_results


def generate_dataset(
    img_ids: list,
    img_meta: dict,
    clip_features: dict,
    teacher,
    tokenizer,
    device: str = "cuda",
    batch_size: int = 8,
    cache_path: str = None,
    max_images: int = None,
) -> list:
    """
    Full dataset generation pipeline: iterate all tool batches for all images.

    Args:
        img_ids       : List of COCO image IDs to process
        img_meta      : Dict from load_coco_metadata
        clip_features : Dict of img_id → CLIP vector (used to filter valid IDs)
        teacher       : Loaded LLM (Llama or Qwen)
        tokenizer     : Corresponding tokenizer
        device        : CUDA or CPU
        batch_size    : Images per generation batch
        cache_path    : If provided, load from cache if it exists
        max_images    : Cap number of images processed (None = all)

    Returns:
        List of {img_id, query, tool, tool_id, objects} records
    """
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached dataset from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    valid_ids = [iid for iid in img_ids if iid in clip_features]
    if max_images:
        valid_ids = valid_ids[:max_images]

    print(f"Generating queries for {len(valid_ids)} images across {len(TOOL_BATCHES)} tool batches...")
    t0 = time.time()

    objects_list = [img_meta[iid]["objects"] for iid in valid_ids]
    per_image_results = [[] for _ in range(len(valid_ids))]

    for batch_idx, tool_batch in enumerate(TOOL_BATCHES):
        print(f"  Tool batch {batch_idx + 1}/{len(TOOL_BATCHES)}...")
        batch_results = generate_queries_batched(
            objects_list, tool_batch, teacher, tokenizer,
            device=device, batch_size=batch_size,
        )
        for i, res in enumerate(batch_results):
            per_image_results[i].extend(res)

    records = []
    for iid, results in zip(valid_ids, per_image_results):
        for p in results:
            records.append({
                "img_id" : iid,
                "query"  : p["query"],
                "tool"   : p["tool"],
                "tool_id": p["tool_id"],
                "objects": img_meta[iid]["objects"],
            })

    elapsed = time.time() - t0
    print(f"✅ {len(records)} records generated in {elapsed / 60:.1f} min")

    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(records, f)
        print(f"Saved to {cache_path}")

    return records
