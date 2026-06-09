"""
evaluate.py — Evaluation utilities for KAN-Route.

Metrics  : Top-1 accuracy, Top-3 accuracy, Macro F1
Latency  : CUDA-event timing over 200 single-query forward passes
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score


def evaluate_model(
    model,
    test_loader,
    model_name: str = "model",
    device: str = "cuda",
) -> dict:
    """
    Evaluate model on a held-out test set.

    Returns:
        dict with keys: params, top1, top3, macro_f1, all_preds, all_labels
    """
    model.eval()
    all_preds, all_labels = [], []
    correct = top3_correct = total = 0

    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.float().to(device), y_b.to(device)
            logits        = model(X_b)
            preds         = logits.argmax(-1)
            top3          = logits.topk(3, dim=-1).indices

            correct      += (preds == y_b).sum().item()
            top3_correct += (top3 == y_b.unsqueeze(1)).any(1).sum().item()
            total        += len(y_b)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_b.cpu().numpy())

    top1     = correct / total
    top3_acc = top3_correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    params   = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n[{model_name}]")
    print(f"  Top-1    : {top1 * 100:.2f}%")
    print(f"  Top-3    : {top3_acc * 100:.2f}%")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print(f"  Params   : {params:,}")

    return {
        "params"     : params,
        "top1"       : top1,
        "top3"       : top3_acc,
        "macro_f1"   : macro_f1,
        "all_preds"  : all_preds,
        "all_labels" : all_labels,
    }


def measure_latency(
    model,
    input_dim: int = 896,
    n_warmup: int = 100,
    n_measure: int = 200,
    device: str = "cuda",
) -> tuple:
    """
    Measure single-query routing head latency using CUDA events.
    Embedding computation (MiniLM + CLIP) is excluded.

    Args:
        model     : Trained routing model
        input_dim : Input dimensionality (default 896)
        n_warmup  : Warmup iterations (not timed)
        n_measure : Timed iterations
        device    : Must be 'cuda' for CUDA event timing

    Returns:
        (mean_ms, std_ms, median_ms)
    """
    model.eval()
    dummy = torch.randn(1, input_dim, dtype=torch.float32).to(device)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(dummy)
    torch.cuda.synchronize()

    # Timed runs
    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)
    times = []

    for _ in range(n_measure):
        start_event.record()
        with torch.no_grad():
            _ = model(dummy)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    times = np.array(times)
    # Remove top/bottom 5% outliers
    times = times[
        (times >= np.percentile(times, 5)) &
        (times <= np.percentile(times, 95))
    ]
    return float(np.mean(times)), float(np.std(times)), float(np.median(times))


def print_classification_report(results: dict, tools: list):
    """Print per-class precision/recall/F1 for all tool classes."""
    print(classification_report(
        results["all_labels"],
        results["all_preds"],
        target_names=tools,
        digits=3,
        zero_division=0,
    ))
