"""
agent.py — Full KAN-Route agentic pipeline.

KANRouteAgent  : Wraps router + encoders, provides route() and run() methods
ToolExecutionEngine : Maps tool names to HuggingFace Inference API endpoints
"""

import os
import time
import base64
import requests
import numpy as np
import torch
import torch.nn.functional as F
from io import BytesIO
from pathlib import Path
from PIL import Image

from .tools import TOOLS, NUM_TOOLS


# ── Query alias system (production-only, excluded from paper metrics) ─────────
QUERY_ALIASES = {
    # Background removal
    "remove the background": "background_removal",
    "cut out the background": "background_removal",
    "isolate the foreground": "background_removal",
    "erase background": "background_removal",
    "transparent background": "background_removal",
    "remove background": "background_removal",
    # Counting
    "how many": "object_counting",
    "count the": "object_counting",
    "number of": "object_counting",
    "count how many": "object_counting",
    # Depth
    "how far": "depth_estimation",
    "distance to": "depth_estimation",
    "depth of": "depth_estimation",
    "depth map": "depth_estimation",
    # OCR
    "read the text": "text_detection_ocr",
    "what does it say": "text_detection_ocr",
    "extract text": "text_detection_ocr",
    "ocr": "text_detection_ocr",
    # Captioning
    "describe this image": "image_captioning",
    "describe what you see": "image_captioning",
    "what is in this image": "image_captioning",
    "caption this": "image_captioning",
    "generate a caption": "image_captioning",
    # Emotion
    "what emotion": "facial_expression_recognition",
    "facial expression": "facial_expression_recognition",
    "how is the person feeling": "facial_expression_recognition",
    # Pose
    "body pose": "pose_estimation",
    "skeleton": "pose_estimation",
    "keypoints": "pose_estimation",
    # Style transfer
    "apply style": "style_transfer",
    "artistic style": "style_transfer",
    "painting style": "style_transfer",
    # Super resolution
    "enhance the image": "super_resolution",
    "increase resolution": "super_resolution",
    "upscale": "super_resolution",
    # Detection
    "detect all objects": "object_detection",
    "draw boxes": "object_detection",
    "bounding box": "object_detection",
    # Segmentation
    "segment the": "semantic_segmentation",
    "label every pixel": "semantic_segmentation",
    # Age / gender
    "how old": "age_gender_estimation",
    "age of": "age_gender_estimation",
    "gender of": "age_gender_estimation",
    # Colorization
    "colorize": "image_colorization",
    "add color": "image_colorization",
    "black and white to color": "image_colorization",
}


def normalize_query(query: str) -> tuple:
    """Match query against aliases. Longest match wins (most specific)."""
    q_lower = query.lower()
    matches = [(phrase, tool) for phrase, tool in QUERY_ALIASES.items() if phrase in q_lower]
    if not matches:
        return query, None
    best_phrase, best_tool = max(matches, key=lambda x: len(x[0]))
    return query, best_tool


# ── Tool Execution Engine ─────────────────────────────────────────────────────

class ToolExecutionEngine:
    """
    Executes the tool selected by KAN-Route via HuggingFace Inference API.

    Set HF_TOKEN environment variable for authenticated access.
    """

    # Maps tool_name → (hf_model_id, task_type)
    TOOL_REGISTRY = {
        "object_detection":               ("facebook/detr-resnet-50",                         "object-detection"),
        "object_counting":                ("facebook/detr-resnet-50",                         "object-detection"),
        "person_detection":               ("facebook/detr-resnet-50",                         "object-detection"),
        "face_detection":                 ("microsoft/resnet-50",                             "image-classification"),
        "vehicle_detection":              ("facebook/detr-resnet-50",                         "object-detection"),
        "animal_detection":               ("facebook/detr-resnet-50",                         "object-detection"),
        "text_detection_ocr":             ("microsoft/trocr-base-printed",                    "image-to-text"),
        "logo_detection":                 ("google/vit-base-patch16-224",                     "image-classification"),
        "anomaly_detection":              ("google/vit-base-patch16-224",                     "image-classification"),
        "semantic_segmentation":          ("nvidia/segformer-b0-finetuned-ade-512-512",        "image-segmentation"),
        "instance_segmentation":          ("facebook/maskformer-swin-base-ade",               "image-segmentation"),
        "panoptic_segmentation":          ("facebook/maskformer-swin-base-ade",               "image-segmentation"),
        "background_removal":             ("briaai/RMBG-1.4",                                 "image-segmentation"),
        "salient_object_segmentation":    ("nvidia/segformer-b0-finetuned-ade-512-512",        "image-segmentation"),
        "matting":                        ("briaai/RMBG-1.4",                                 "image-segmentation"),
        "text_to_image_generation":       ("stabilityai/stable-diffusion-2-1",                "text-to-image"),
        "image_inpainting":               ("stabilityai/stable-diffusion-2-inpainting",       "inpainting"),
        "style_transfer":                 ("stabilityai/stable-diffusion-2-1",                "text-to-image"),
        "super_resolution":               ("caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",   "image-to-image"),
        "image_colorization":             ("google/vit-base-patch16-224",                     "image-to-image"),
        "image_denoising":                ("caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",   "image-to-image"),
        "image_deblurring":               ("caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",   "image-to-image"),
        "depth_estimation":               ("Intel/dpt-large",                                 "depth-estimation"),
        "surface_normal_estimation":      ("Intel/dpt-large",                                 "depth-estimation"),
        "3d_reconstruction":              ("Intel/dpt-large",                                 "depth-estimation"),
        "point_cloud_generation":         ("Intel/dpt-large",                                 "depth-estimation"),
        "stereo_matching":                ("Intel/dpt-large",                                 "depth-estimation"),
        "image_classification":           ("google/vit-base-patch16-224",                     "image-classification"),
        "fine_grained_classification":    ("google/vit-base-patch16-224",                     "image-classification"),
        "scene_recognition":              ("google/vit-base-patch16-224",                     "image-classification"),
        "facial_expression_recognition":  ("trpakov/vit-face-expression",                     "image-classification"),
        "age_gender_estimation":          ("rizvandwiki/gender-classification",                "image-classification"),
        "action_recognition":             ("google/vit-base-patch16-224",                     "image-classification"),
        "pose_estimation":                ("lllyasviel/ControlNet",                           "image-to-image"),
        "gaze_estimation":                ("google/vit-base-patch16-224",                     "image-classification"),
        "visual_question_answering":      ("dandelin/vilt-b32-finetuned-vqa",                 "visual-question-answering"),
        "image_captioning":               ("Salesforce/blip-image-captioning-base",           "image-to-text"),
        "visual_reasoning":               ("dandelin/vilt-b32-finetuned-vqa",                 "visual-question-answering"),
        "visual_grounding":               ("dandelin/vilt-b32-finetuned-vqa",                 "visual-question-answering"),
        "relation_detection":             ("dandelin/vilt-b32-finetuned-vqa",                 "visual-question-answering"),
        "attribute_recognition":          ("dandelin/vilt-b32-finetuned-vqa",                 "visual-question-answering"),
        "change_detection":               ("google/vit-base-patch16-224",                     "image-classification"),
        "video_object_tracking":          ("google/vit-base-patch16-224",                     "image-classification"),
        "optical_flow_estimation":        ("google/vit-base-patch16-224",                     "image-classification"),
        "video_captioning":               ("Salesforce/blip-image-captioning-base",           "image-to-text"),
        "temporal_action_localization":   ("google/vit-base-patch16-224",                     "image-classification"),
        "medical_image_segmentation":     ("nvidia/segformer-b0-finetuned-ade-512-512",        "image-segmentation"),
        "document_layout_analysis":       ("microsoft/layoutlm-base-uncased",                 "image-classification"),
        "satellite_image_analysis":       ("google/vit-base-patch16-224",                     "image-classification"),
    }

    def __init__(self, hf_token: str = None):
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self.hf_api   = "https://api-inference.huggingface.co/models"
        self.headers  = {"Authorization": f"Bearer {self.hf_token}"}

    def _image_to_bytes(self, image_path: str) -> bytes:
        img = Image.open(image_path).convert("RGB")
        if max(img.size) > 1024:
            img.thumbnail((1024, 1024), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def execute(self, tool_name: str, image_path: str, query: str = "", max_retries: int = 3) -> dict:
        """Execute the selected tool via HuggingFace Inference API."""
        if tool_name not in self.TOOL_REGISTRY:
            return {"error": f"Tool '{tool_name}' not in registry", "tool": tool_name}

        model_id, task_type = self.TOOL_REGISTRY[tool_name]
        api_url   = f"{self.hf_api}/{model_id}"
        img_bytes = self._image_to_bytes(image_path)

        for attempt in range(max_retries):
            try:
                if task_type == "visual-question-answering":
                    payload  = {"inputs": {"image": base64.b64encode(img_bytes).decode(), "question": query or "What do you see?"}}
                    response = requests.post(api_url, headers=self.headers, json=payload, timeout=30)
                elif task_type == "text-to-image":
                    response = requests.post(api_url, headers=self.headers, json={"inputs": query or "high quality image"}, timeout=60)
                else:
                    response = requests.post(api_url, headers=self.headers, data=img_bytes, timeout=30)

                if response.status_code == 200:
                    is_image = task_type == "text-to-image" or response.headers.get("content-type", "").startswith("image")
                    return {
                        "tool": tool_name, "model": model_id, "task": task_type,
                        "status": "success",
                        "result_type": "image" if is_image else "json",
                        "result": response.content if is_image else response.json(),
                    }
                elif response.status_code == 503:
                    wait = response.json().get("estimated_time", 20)
                    time.sleep(min(wait, 30))
                else:
                    return {"tool": tool_name, "status": "error", "code": response.status_code, "error": response.text[:200]}

            except requests.exceptions.Timeout:
                print(f"  Timeout on attempt {attempt + 1}")
            except Exception as e:
                return {"tool": tool_name, "status": "error", "error": str(e)}

        return {"tool": tool_name, "status": "error", "error": "Max retries exceeded"}


# ── KANRouteAgent ─────────────────────────────────────────────────────────────

class KANRouteAgent:
    """
    End-to-end visual agent using KAN-Route for tool selection.

    Usage:
        agent = KANRouteAgent(kan_model, sbert, clip_model, clip_preprocess)
        result = agent.run("How many people are there?", "image.jpg", execute=False)
    """

    def __init__(self, kan_model, sbert, clip_model, clip_preprocess,
                 hf_token: str = None, device: str = "cuda"):
        self.router     = kan_model
        self.sbert      = sbert
        self.clip       = clip_model.float()
        self.clip_prep  = clip_preprocess
        self.engine     = ToolExecutionEngine(hf_token=hf_token)
        self.device     = device
        self.router.eval()
        self.clip.eval()
        self._warmup()

    def _warmup(self, n: int = 10):
        """Pre-warm all models to eliminate first-call cold-start latency."""
        print("Warming up models...")
        dummy_img    = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        dummy_tensor = self.clip_prep(dummy_img).unsqueeze(0).float().to(self.device)
        for _ in range(n):
            with torch.no_grad():
                self.sbert.encode(["warmup query"], normalize_embeddings=True)
                self.clip.encode_image(dummy_tensor)
                _ = self.router(torch.randn(1, 896, dtype=torch.float32).to(self.device))
        torch.cuda.synchronize()
        print("✅ Warmup complete.")

    def route(self, query: str, image_path: str, top_k: int = 3) -> dict:
        """
        Route a query+image to the most appropriate tool.
        Note: For paper metrics, call this directly (bypasses alias system).
        """
        text_emb   = self.sbert.encode([query], normalize_embeddings=True)
        img        = Image.open(image_path).convert("RGB")
        img_tensor = self.clip_prep(img).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            img_emb = F.normalize(self.clip.encode_image(img_tensor), dim=-1).cpu().numpy()

        x = torch.tensor(
            np.concatenate([text_emb, img_emb], axis=1), dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            logits = self.router(x)
            probs  = F.softmax(logits, dim=-1).cpu().numpy()[0]

        top_k_idx = probs.argsort()[::-1][:top_k]
        top_conf  = float(probs[top_k_idx[0]])

        return {
            "selected_tool": TOOLS[top_k_idx[0]],
            "confidence":    top_conf,
            "uncertain":     top_conf < 0.15,
            "top_k": [
                {"tool": TOOLS[i], "confidence": float(probs[i])}
                for i in top_k_idx
            ],
        }

    def run(self, query: str, image_path: str, execute: bool = True, top_k: int = 3) -> dict:
        """
        Full agentic pipeline: route query → (optionally) execute tool.

        Args:
            query      : Natural language user request
            image_path : Path to input image
            execute    : If True, call HuggingFace Inference API after routing
            top_k      : Number of top tool candidates to return

        Returns:
            Dict with routing result, timing, and (optionally) execution result
        """
        t0 = time.perf_counter()

        # Check alias layer first (production use; excluded from paper metrics)
        _, forced_tool = normalize_query(query)
        if forced_tool:
            routing = self.route(query, image_path, top_k=top_k)
            routing["selected_tool"] = forced_tool
            routing["alias_override"] = True
        else:
            routing = self.route(query, image_path, top_k=top_k)
            routing["alias_override"] = False

        routing_ms = (time.perf_counter() - t0) * 1000

        print(f"\n{'═'*58}")
        print(f"  Query  : {query}")
        print(f"  Image  : {Path(image_path).name}")
        print(f"{'─'*58}")
        print(f"  ✅ Selected : [{routing['selected_tool']}]")
        print(f"  Confidence : {routing['confidence'] * 100:.1f}%  {'⚠️ uncertain' if routing['uncertain'] else '✅ confident'}")
        print(f"  Routing    : {routing_ms:.2f} ms")

        for i, t in enumerate(routing["top_k"]):
            bar    = "█" * int(t["confidence"] * 30)
            marker = " ←" if i == 0 else ""
            print(f"     {i+1}. {t['tool']:<40s} {t['confidence']*100:5.1f}%  {bar}{marker}")

        if not execute:
            return {"routing": routing, "result": None, "timing": {"routing_ms": routing_ms}}

        print(f"\n  🔧 Executing: {routing['selected_tool']} ...")
        t1     = time.perf_counter()
        result = self.engine.execute(routing["selected_tool"], image_path, query=query)
        exec_ms  = (time.perf_counter() - t1) * 1000
        total_ms = (time.perf_counter() - t0) * 1000

        icon = "✅" if result.get("status") == "success" else "❌"
        print(f"  {icon} Status : {result.get('status', 'unknown')}")
        print(f"  Model  : {result.get('model', 'N/A')}")
        print(f"  Timing : route={routing_ms:.1f}ms + exec={exec_ms:.0f}ms = {total_ms:.0f}ms total")

        return {
            "query"  : query,
            "image"  : image_path,
            "routing": routing,
            "result" : result,
            "timing" : {"routing_ms": routing_ms, "exec_ms": exec_ms, "total_ms": total_ms},
        }
