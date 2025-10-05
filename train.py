import os
import sys
import time
import logging
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch import amp
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp.autocast")

from src.dataloader import create_dataloaders
from models.bloodflow_cnn import BloodFlowCNN2D

# =========================================================
# Utilities
# =========================================================

def set_seed(seed: int = 42) -> None:
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("bloodflow_regression_train")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
        fh = logging.FileHandler(os.path.join(output_dir, "training.log"))
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger

def plot_loss_curve(train_losses: List[float], val_losses: List[float], out_path: str) -> None:
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, marker="o", label="Train Loss")
    plt.plot(range(1, len(val_losses)+1), val_losses, marker="x", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# =========================================================
# Training / Validation
# =========================================================

def _fix_input_shape(xb: torch.Tensor) -> torch.Tensor:
    if xb.dim() == 5 and xb.size(2) == 1:
        xb = xb.squeeze(2)
    return xb

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    amp_scaler: Optional[GradScaler],
    grad_clip: float,
    logger: Optional[logging.Logger] = None,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    for xb, yb in tqdm(loader, desc="train", leave=False, dynamic_ncols=True):
        xb = _fix_input_shape(xb).to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)  # (B,1,H,W)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast("cuda", enabled=(amp_scaler is not None and device.type == "cuda")):
            preds = model(xb)
            loss = criterion(preds, yb)

        if amp_scaler:
            amp_scaler.scale(loss).backward()
            if grad_clip > 0:
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_loss += loss.item() * xb.size(0)
        total += xb.size(0)

    avg_loss = running_loss / total if total > 0 else float("nan")
    if logger:
        logger.info(f"Train Loss: {avg_loss:.6f}")
    return avg_loss

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0
    total = 0
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="val", leave=False, dynamic_ncols=True):
            xb = _fix_input_shape(xb).to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            preds = model(xb)
            loss = criterion(preds, yb)
            running_loss += loss.item() * xb.size(0)
            total += xb.size(0)

    avg_loss = running_loss / total if total > 0 else float("nan")
    return avg_loss

# =========================================================
# Main training loop
# =========================================================

def train(
    data_dir,
    output_dir,
    batch_size,
    stride,
    num_epochs,
    lr,
    weight_decay,
    num_workers,
    grad_clip,
    seed,
    augment_train,
    safe_mode=False,
    target_resolution=(128,128),
    channel_width_um=100.0
):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    logger = setup_logger(output_dir)
    logger.info("Preparing dataloaders...")

    train_loader, val_loader = create_dataloaders(
        data_folder=data_dir,
        batch_size=batch_size,
        stride=stride,
        num_workers=(0 if safe_mode else num_workers),
        target_resolution=target_resolution,
        channel_width_um=channel_width_um
    )

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BloodFlowCNN2D(input_channels=1, output_channels=1).to(device)
    logger.info(f"Model created and moved to {device}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    amp_scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    best_ckpt = os.path.join(checkpoints_dir, "best_model.pth")

    logger.info("Starting training loop...")
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, amp_scaler, grad_clip, logger)
        val_loss = validate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {time.time()-t0:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict()}, best_ckpt)
            logger.info(f"Saved best model to {best_ckpt}")

        scheduler.step(val_loss)  # reduce LR on plateau

    logger.info("Training complete. Loading best model for final evaluation.")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    final_val_loss = validate(model, val_loader, criterion, device)
    logger.info(f"Final Val MSE Loss: {final_val_loss:.6f}")

    plot_loss_curve(train_losses, val_losses, os.path.join(output_dir, "training_loss.png"))

# =========================================================
# CLI
# =========================================================

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Train BloodFlow 2D-CNN Regression")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augment_train", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        stride=args.stride,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        grad_clip=args.grad_clip,
        seed=args.seed,
        augment_train=args.augment_train
    )

if __name__ == "__main__":
    main()
