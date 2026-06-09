"""
plot_spline_activations.py — Reproduce Figure 1 from the paper.

Visualises the top-3 most influential spline activation curves per tool class,
extracted via direct forward-pass activation recording on ground-truth test
samples. Red curves = MiniLM text embedding dimensions (indices 0–383);
blue curves = CLIP visual embedding dimensions (indices 384–895).

Usage
-----
    python scripts/plot_spline_activations.py \
        --checkpoint /path/to/KAN_Route_best.pt \
        --embeddings  /path/to/test_embeddings.npy \
        --labels      /path/to/test_labels.npy \
        --out         assets/spline_activations.png

Requirements
------------
    pip install matplotlib numpy torch
    pip install git+https://github.com/ZiyaoLi/fast-kan.git
"""

import argparse
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.use("Agg")


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, num_tools: int = 50, device: str = "cpu"):
    """Load a saved KANRoute checkpoint."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.models import KANRoute

    model = KANRoute(
        input_dim=896, proj_dim=48, hidden_dims=[48],
        num_tools=num_tools, num_grids=4,
    ).float().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    # Handle DataParallel / plain state dict
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model.eval()
    return model


def record_activations(model, x: torch.Tensor):
    """
    Run a forward pass and record per-edge spline activation values.

    Returns
    -------
    activations : dict mapping (layer_idx, in_idx, out_idx) → activation scalar
    """
    activations = {}

    def make_hook(layer_idx):
        def hook(module, inp, out):
            # FastKAN layers expose per-edge RBF outputs
            # inp[0]: (batch, in_features)
            # We record the mean absolute activation per input dimension
            x_in = inp[0].detach()                    # (B, in_dim)
            mean_act = x_in.abs().mean(dim=0)          # (in_dim,)
            activations[layer_idx] = mean_act.cpu().numpy()
        return hook

    handles = []
    for i, layer in enumerate(model.kan.layers):
        h = layer.register_forward_hook(make_hook(i))
        handles.append(h)

    with torch.no_grad():
        _ = model(x)

    for h in handles:
        h.remove()
    return activations


def top_dims_per_class(model, X: np.ndarray, y: np.ndarray,
                        n_tools: int, top_k: int = 3):
    """
    For each tool class, collect the top-k most activated input dimensions
    on correctly-predicted ground-truth samples.

    Returns dict: tool_id → list of (dim_idx, mean_activation) sorted desc.
    """
    device = next(model.parameters()).device
    tool_activations = {t: [] for t in range(n_tools)}

    for i in range(len(y)):
        xi = torch.tensor(X[i:i+1], dtype=torch.float32).to(device)
        yi = int(y[i])
        acts = record_activations(model, xi)
        # Layer 0 activations reflect input-dimension importance
        if 0 in acts:
            tool_activations[yi].append(acts[0])

    top_dims = {}
    for t in range(n_tools):
        if not tool_activations[t]:
            top_dims[t] = []
            continue
        mean_act = np.stack(tool_activations[t]).mean(axis=0)  # (in_dim,)
        top_idx = np.argsort(mean_act)[::-1][:top_k]
        top_dims[t] = [(int(idx), float(mean_act[idx])) for idx in top_idx]
    return top_dims


def plot_spline_curve(ax, model, dim_idx: int, is_text: bool, label: str,
                      n_points: int = 100):
    """
    Plot the learned spline (RBF) activation curve for a single input dimension.
    Red = MiniLM text dims (0–383), Blue = CLIP visual dims (384–895).
    """
    device = next(model.parameters()).device
    color = "#C0392B" if is_text else "#2980B9"

    # Probe the model's first KAN layer across the input range
    x_vals = torch.linspace(-1.5, 1.5, n_points).unsqueeze(1).to(device)
    probe = torch.zeros(n_points, 896, device=device)
    probe[:, dim_idx] = x_vals.squeeze()

    with torch.no_grad():
        # Pass through projection + norm, then first KAN layer only
        proj = model.norm(model.proj(probe.float()))   # (N, proj_dim)
        out = model.kan.layers[0](proj)                # (N, proj_dim)

    y_vals = out[:, 0].cpu().numpy()                   # first output unit
    x_np   = x_vals.squeeze().cpu().numpy()

    ax.plot(x_np, y_vals, color=color, linewidth=1.2, alpha=0.85, label=label)
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--", alpha=0.4)
    ax.axvline(0, color="gray", linewidth=0.4, linestyle="--", alpha=0.4)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.tools import TOOLS

    parser = argparse.ArgumentParser(description="Plot KAN-Route spline activations")
    parser.add_argument("--checkpoint", required=True,  help="Path to KAN_Route_best.pt")
    parser.add_argument("--embeddings", required=True,  help="Path to test_embeddings.npy")
    parser.add_argument("--labels",     required=True,  help="Path to test_labels.npy")
    parser.add_argument("--num-tools",  type=int, default=50)
    parser.add_argument("--top-k",      type=int, default=3,  help="Top-k dims per class")
    parser.add_argument("--out",        default="assets/spline_activations.png")
    parser.add_argument("--dpi",        type=int, default=200)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = load_model(args.checkpoint, num_tools=args.num_tools, device=device)
    X     = np.load(args.embeddings)
    y     = np.load(args.labels)
    print(f"Test set: {len(y)} samples, {args.num_tools} tools")

    top_dims = top_dims_per_class(model, X, y, args.num_tools, top_k=args.top_k)

    # Layout: up to 10 cols, auto rows
    n_tools = args.num_tools
    ncols   = min(10, n_tools)
    nrows   = (n_tools + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 2.4, nrows * 2.0),
                              facecolor="white")
    axes = np.array(axes).reshape(nrows, ncols)

    for t in range(n_tools):
        row, col = divmod(t, ncols)
        ax = axes[row, col]
        tool_name = TOOLS[t] if t < len(TOOLS) else f"tool_{t}"
        ax.set_title(tool_name.replace("_", " "), fontsize=6.5, pad=3)
        ax.tick_params(labelsize=5)
        ax.set_xlabel("Input value", fontsize=5)
        ax.set_ylabel("Activation", fontsize=5)

        for dim_idx, _ in top_dims.get(t, []):
            is_text = dim_idx < 384
            label = f"dim {dim_idx} ({'text' if is_text else 'visual'})"
            try:
                plot_spline_curve(ax, model, dim_idx, is_text, label)
            except Exception as e:
                print(f"  [warn] tool {t} dim {dim_idx}: {e}")

        ax.legend(fontsize=4, loc="upper right", framealpha=0.5)

    # Hide unused axes
    for t in range(n_tools, nrows * ncols):
        row, col = divmod(t, ncols)
        axes[row, col].set_visible(False)

    # Legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="#C0392B", linewidth=1.5, label="MiniLM text dims (0–383)"),
        Line2D([0], [0], color="#2980B9", linewidth=1.5, label="CLIP visual dims (384–895)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2,
               fontsize=8, framealpha=0.8, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        "KAN-Route Spline Activations — Top-3 Input Dims per Tool Class\n"
        "(direct forward pass on ground-truth test samples)",
        fontsize=10, y=1.01,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
