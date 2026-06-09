"""
models.py — KANRoute and MLPBaseline architectures for visual tool routing.

KANRoute:    896 → proj(48) → FastKAN([48, 48, N]) — ~66,962 params
MLPBaseline: 896 → proj(96) → MLP([64, 32, N])    — ~96,434 params
"""

import torch
import torch.nn as nn

# FastKAN must be installed separately:
#   pip install git+https://github.com/ZiyaoLi/fast-kan.git
try:
    from fastkan import FastKAN
    FASTKAN_AVAILABLE = True
except ImportError:
    FASTKAN_AVAILABLE = False
    print("Warning: FastKAN not found. KANRoute will be unavailable.")
    print("Install via: pip install git+https://github.com/ZiyaoLi/fast-kan.git")


class KANRoute(nn.Module):
    """
    Lightweight KAN-based multimodal tool router.

    Architecture:
        Linear projection: input_dim → proj_dim
        LayerNorm
        FastKAN: proj_dim → hidden_dims → num_tools

    Default config (35-tool / 50-tool taxonomy):
        input_dim=896, proj_dim=48, hidden_dims=[48], num_tools=50
        → ~66,962 trainable parameters

    Args:
        input_dim   : Dimensionality of fused embedding (default 896 = 384+512)
        proj_dim    : Linear projection output dim before KAN (default 48)
        hidden_dims : List of hidden layer sizes in the KAN (default [48])
        num_tools   : Number of output classes / tools
        num_grids   : Number of grid points for FastKAN RBF (default 4)
    """

    def __init__(
        self,
        input_dim: int = 896,
        proj_dim: int = 48,
        hidden_dims: list = None,
        num_tools: int = 50,
        num_grids: int = 4,
    ):
        super().__init__()
        if not FASTKAN_AVAILABLE:
            raise ImportError("FastKAN is required. See instructions above.")
        if hidden_dims is None:
            hidden_dims = [48]

        self.proj = nn.Linear(input_dim, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)

        kan_layers = [proj_dim] + hidden_dims + [num_tools]
        self.kan = FastKAN(layers_hidden=kan_layers, num_grids=num_grids)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.proj(x.float()))
        return self.kan(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLPBaseline(nn.Module):
    """
    Parameter-matched MLP baseline for ablation comparison.

    Architecture:
        Linear projection: input_dim → proj_dim
        LayerNorm + GELU + Dropout
        Hidden layers with BatchNorm + GELU + Dropout
        Output: num_tools logits

    Default config (35-tool / 50-tool taxonomy):
        input_dim=896, proj_dim=96, hidden_dims=[64, 32], num_tools=50
        → ~96,434 trainable parameters

    Note: Unlike KANRoute, MLPBaseline provides no interpretability —
    spline activations cannot be visualized. This is the key trade-off.

    Args:
        input_dim   : Dimensionality of fused embedding (default 896)
        proj_dim    : Linear projection output dim (default 96)
        hidden_dims : List of hidden layer sizes (default [64, 32])
        num_tools   : Number of output classes / tools
        dropout     : Dropout rate (default 0.3)
    """

    def __init__(
        self,
        input_dim: int = 896,
        proj_dim: int = 96,
        hidden_dims: list = None,
        num_tools: int = 50,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = [
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        prev = proj_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_tools))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_kan_route(num_tools: int = 50, device: str = "cuda") -> KANRoute:
    """Instantiate default KANRoute model."""
    model = KANRoute(
        input_dim=896, proj_dim=48, hidden_dims=[48],
        num_tools=num_tools, num_grids=4,
    ).float().to(device)
    print(f"KANRoute — {model.count_parameters():,} parameters")
    return model


def build_mlp_baseline(num_tools: int = 50, device: str = "cuda") -> MLPBaseline:
    """Instantiate default MLPBaseline model."""
    model = MLPBaseline(
        input_dim=896, proj_dim=96, hidden_dims=[64, 32],
        num_tools=num_tools, dropout=0.3,
    ).float().to(device)
    print(f"MLPBaseline — {model.count_parameters():,} parameters")
    return model
