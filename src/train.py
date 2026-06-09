"""
train.py — Training loop for KAN-Route and MLP Baseline.

Optimizer : AdamW (lr=1e-3 for MLP, 5e-4 for KAN)
Scheduler : OneCycleLR
Loss      : CrossEntropyLoss with label smoothing (0.05)
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path


class ToolRoutingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    batch_size: int = 256,
    split: tuple = (0.70, 0.15, 0.15),
    num_workers: int = 2,
) -> tuple:
    """
    Split dataset and return (train_loader, val_loader, test_loader).

    Args:
        X          : Feature matrix (N, 896)
        y          : Label array (N,)
        seed       : Random seed for reproducibility
        batch_size : DataLoader batch size
        split      : (train, val, test) fractions — must sum to 1.0
        num_workers: DataLoader workers

    Returns:
        (train_loader, val_loader, test_loader, split_sizes)
    """
    assert abs(sum(split) - 1.0) < 1e-6, "Split fractions must sum to 1.0"
    dataset = ToolRoutingDataset(X, y)
    n = len(dataset)
    n_train = int(split[0] * n)
    n_val   = int(split[1] * n)
    n_test  = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    make_loader = lambda ds, shuffle: DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
    )
    return (
        make_loader(train_ds, True),
        make_loader(val_ds, False),
        make_loader(test_ds, False),
        (n_train, n_val, n_test),
    )


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str = "model",
    out_dir: str = ".",
    epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    is_kan: bool = False,
    device: str = "cuda",
) -> dict:
    """
    Train a routing model and return training history.

    KAN models use lr=5e-4 and 80 epochs by default (slower convergence
    due to B-spline grid fitting). MLPs converge faster (~5 epochs).

    Args:
        model       : KANRoute or MLPBaseline instance
        train_loader: Training DataLoader
        val_loader  : Validation DataLoader
        model_name  : Name prefix for checkpoint file
        out_dir     : Directory to save best checkpoint
        epochs      : Number of training epochs
        lr          : Base learning rate (overridden to 5e-4 for KAN)
        weight_decay: AdamW weight decay
        is_kan      : If True, use KAN-tuned hyperparameters
        device      : CUDA or CPU

    Returns:
        history dict with train_loss, val_loss, val_acc, val_top3_acc per epoch
    """
    if is_kan:
        lr     = 5e-4
        epochs = 80

    out_dir   = Path(out_dir)
    best_path = out_dir / f"{model_name}_best.pt"

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    history      = {"train_loss": [], "val_loss": [], "val_acc": [], "val_top3_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.float().to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * len(y_b)
        train_loss /= len(train_loader.dataset)

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        val_loss, correct, top3_correct, total = 0.0, 0, 0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.float().to(device), y_b.to(device)
                logits     = model(X_b)
                val_loss  += criterion(logits, y_b).item() * len(y_b)
                correct   += (logits.argmax(-1) == y_b).sum().item()
                top3       = logits.topk(3, dim=-1).indices
                top3_correct += (top3 == y_b.unsqueeze(1)).any(1).sum().item()
                total     += len(y_b)

        val_loss /= total
        val_acc   = correct / total
        top3_acc  = top3_correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_top3_acc"].append(top3_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[{model_name}] Ep {epoch:3d}/{epochs} | "
                f"TrLoss: {train_loss:.4f} | "
                f"ValLoss: {val_loss:.4f} | "
                f"Top-1: {val_acc:.4f} | "
                f"Top-3: {top3_acc:.4f}"
            )

    print(f"\n✅ [{model_name}] Best Val Top-1: {best_val_acc:.4f}")
    model.load_state_dict(torch.load(best_path))

    # Save final checkpoint and history
    torch.save(model.state_dict(), out_dir / f"{model_name}_final.pt")
    with open(out_dir / f"{model_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history
