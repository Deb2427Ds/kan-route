"""
benchmark_latency.py — Reproduce routing latency numbers from Table 1.

Measures mean ± std latency over 200 single-query CUDA forward passes for
KANRoute and MLPBaseline, isolating the routing head from embedding computation.

Usage
-----
    python scripts/benchmark_latency.py \
        --kan-checkpoint  /path/to/KAN_Route_best.pt \
        --mlp-checkpoint  /path/to/MLP_Baseline_best.pt \
        --num-tools 50

Example output
--------------
    KANRoute   — mean: 0.640 ms  std: 0.029 ms  params: 66,962
    MLPBaseline — mean: 0.481 ms  std: 0.027 ms  params: 96,434
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import KANRoute, MLPBaseline


def load_checkpoint(model, path: str, device: str):
    state = torch.load(path, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model.eval()
    return model


def benchmark(model, device: str, n_warmup: int = 100, n_measure: int = 200,
              input_dim: int = 896):
    """CUDA-event timing over n_measure single-query forward passes."""
    dummy = torch.randn(1, input_dim, dtype=torch.float32, device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    if device == "cuda":
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(n_measure):
            if device == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end   = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(dummy)
                end.record()
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))  # ms
            else:
                import time
                t0 = time.perf_counter()
                _ = model(dummy)
                latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    return float(arr.mean()), float(arr.std())


def main():
    parser = argparse.ArgumentParser(description="Routing head latency benchmark")
    parser.add_argument("--kan-checkpoint",  default=None)
    parser.add_argument("--mlp-checkpoint",  default=None)
    parser.add_argument("--num-tools",  type=int, default=50)
    parser.add_argument("--n-warmup",   type=int, default=100)
    parser.add_argument("--n-measure",  type=int, default=200)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("Warning: paper latencies were measured on CUDA (T4). CPU numbers will differ.")

    results = []

    if args.kan_checkpoint:
        kan = KANRoute(input_dim=896, proj_dim=48, hidden_dims=[48],
                       num_tools=args.num_tools, num_grids=4).float().to(device)
        kan = load_checkpoint(kan, args.kan_checkpoint, device)
        mean, std = benchmark(kan, device, args.n_warmup, args.n_measure)
        params = sum(p.numel() for p in kan.parameters() if p.requires_grad)
        results.append(("KANRoute (FastKAN)", mean, std, params))
        print(f"KANRoute   — mean: {mean:.3f} ms  std: {std:.3f} ms  params: {params:,}")

    if args.mlp_checkpoint:
        mlp = MLPBaseline(input_dim=896, proj_dim=96, hidden_dims=[64, 32],
                          num_tools=args.num_tools, dropout=0.3).float().to(device)
        mlp = load_checkpoint(mlp, args.mlp_checkpoint, device)
        mean, std = benchmark(mlp, device, args.n_warmup, args.n_measure)
        params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
        results.append(("MLPBaseline", mean, std, params))
        print(f"MLPBaseline — mean: {mean:.3f} ms  std: {std:.3f} ms  params: {params:,}")

    if not results:
        print("No checkpoints provided. Running with randomly-initialised weights for reference.")
        for name, model in [
            ("KANRoute",    KANRoute(896,48,[48],args.num_tools,4).float().to(device)),
            ("MLPBaseline", MLPBaseline(896,96,[64,32],args.num_tools,0.3).float().to(device)),
        ]:
            model.eval()
            mean, std = benchmark(model, device, args.n_warmup, args.n_measure)
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"{name:<15} — mean: {mean:.3f} ms  std: {std:.3f} ms  params: {params:,}")


if __name__ == "__main__":
    main()
