"""
embeddings.py — Multimodal embedding pipeline for KAN-Route.

Produces 896-dim fused vectors:
    text query  → all-MiniLM-L6-v2 (384-dim, frozen)
    image context → CLIP ViT-B/32 text encoder (512-dim, frozen)
    fused = concat(text_emb, clip_emb) → 896-dim
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm

TEXT_DIM  = 384
CLIP_DIM  = 512
INPUT_DIM = TEXT_DIM + CLIP_DIM  # 896


def load_sbert(device: str = "cuda"):
    """Load frozen all-MiniLM-L6-v2 sentence transformer."""
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    for p in sbert.parameters():
        p.requires_grad = False
    print("Loaded all-MiniLM-L6-v2 (frozen)")
    return sbert


def load_clip(device: str = "cuda"):
    """Load frozen CLIP ViT-B/32 model and preprocessor."""
    import open_clip
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_model = clip_model.to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    print("Loaded CLIP ViT-B/32 (frozen)")
    return clip_model, clip_preprocess


def embed_queries(
    queries: list,
    sbert,
    cache_path: str = None,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Encode a list of text queries with MiniLM.

    Args:
        queries    : List of query strings
        sbert      : Loaded SentenceTransformer
        cache_path : If provided, load from cache if exists, else save after encoding
        batch_size : Encoding batch size

    Returns:
        np.ndarray of shape (N, 384), L2-normalized
    """
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached text embeddings from {cache_path}")
        return np.load(cache_path)

    print(f"Encoding {len(queries)} queries with MiniLM...")
    embeddings = sbert.encode(
        queries,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    if cache_path:
        np.save(cache_path, embeddings)
        print(f"Saved to {cache_path}")

    return embeddings


def embed_clip_from_objects(
    img_ids: list,
    img_meta: dict,
    clip_model,
    device: str = "cuda",
    cache_path: str = None,
    batch_size: int = 256,
) -> dict:
    """
    Encode image contexts (object label lists) using CLIP text encoder.
    No pixel loading required — uses textual descriptions of COCO objects.

    Format: 'a photo with: [obj1], [obj2], ...'

    Args:
        img_ids    : List of COCO image IDs to encode
        img_meta   : Dict mapping img_id → {'objects': [...]}
        clip_model : Loaded CLIP model (frozen)
        device     : CUDA or CPU
        cache_path : Optional .npz cache file path
        batch_size : Batch size for CLIP encoding

    Returns:
        Dict mapping img_id → 512-dim numpy vector
    """
    import open_clip

    if cache_path and Path(cache_path).exists():
        print(f"Loading cached CLIP features from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return {int(k): v for k, v in data.items()}

    print(f"Encoding {len(img_ids)} image contexts with CLIP...")
    features = {}

    for start in tqdm(range(0, len(img_ids), batch_size)):
        batch_ids = img_ids[start : start + batch_size]
        descriptions = [
            "a photo with: " + ", ".join(img_meta[iid]["objects"][:6])
            for iid in batch_ids
        ]
        tokens = open_clip.tokenize(descriptions).to(device)
        with torch.no_grad():
            embs = clip_model.encode_text(tokens)
            embs = F.normalize(embs, dim=-1).cpu().numpy()
        for iid, emb in zip(batch_ids, embs):
            features[iid] = emb

    if cache_path:
        np.savez(cache_path, **{str(k): v for k, v in features.items()})
        print(f"Saved to {cache_path}")

    return features


def build_feature_matrix(
    dataset_path: str,
    sbert,
    clip_features: dict,
    text_cache_path: str = None,
) -> tuple:
    """
    Build the final (N, 896) feature matrix and label array from a dataset JSON.

    Args:
        dataset_path    : Path to dataset JSON (list of {query, tool_id, img_id, ...})
        sbert           : Loaded SentenceTransformer
        clip_features   : Dict of img_id → 512-dim CLIP vectors
        text_cache_path : Optional cache for text embeddings

    Returns:
        X : np.ndarray of shape (N, 896), float32
        y : np.ndarray of shape (N,), int64
    """
    import pandas as pd

    with open(dataset_path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    text_embs = embed_queries(
        df["query"].tolist(), sbert, cache_path=text_cache_path
    )
    clip_vecs = np.stack([clip_features[int(iid)] for iid in df["img_id"]])

    X = np.concatenate([text_embs, clip_vecs], axis=1).astype(np.float32)
    y = df["tool_id"].values.astype(np.int64)

    print(f"Feature matrix: X={X.shape}, y={y.shape}")
    return X, y
