import argparse
import os
import sys
import platform
from typing import Optional, List

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure root path for imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.train import train
from src.dataloader import create_dataloaders
from models.bloodflow_cnn import BloodFlowCNN2D


def evaluate_model(
    checkpoint_path: str,
    data_dir: str,
    sequence_len: int = 1,
    batch_size: int = 8,
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
    num_workers: int = 0,
    safe_mode: bool = True,
) -> None:
    """Load a trained regression model and evaluate on validation data."""
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Evaluating on device: {device}")

    # Use DataLoader to get validation set
    train_loader, val_loader, _ = create_dataloaders(
        data_folder=data_dir,
        batch_size=batch_size,
        sequence_len=sequence_len,
        test_split=0.2,
        stride=1,
        num_workers=num_workers,
        normalize_mode="scale",
        augment_train=False,
        safe_mode=safe_mode,
    )

    if val_loader is None or len(val_loader) == 0:
        raise RuntimeError("Validation data not found or improperly formatted.")

    # Create model for regression (1 output)
    model = BloodFlowCNN2D(input_channels=1, output_channels=1).to(device)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError("Invalid checkpoint format.")

    model.eval()

    all_preds: List[float] = []
    all_targets: List[float] = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.squeeze(1).cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    # Compute regression metrics
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    print("\n[RESULTS]")
    print(f"MSE: {mse:.6f} | MAE: {mae:.6f}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame({"Target": all_targets, "Prediction": all_preds})
        df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

        # Optional: scatter plot of predictions vs targets
        plt.figure(figsize=(6,6))
        plt.scatter(targets, preds, alpha=0.6)
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
        plt.xlabel("True Flowrate")
        plt.ylabel("Predicted Flowrate")
        plt.title("Predicted vs True Flowrate")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "prediction_scatter.png"))
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blood Flow Regression")
    parser.add_argument("--mode", choices=["train", "evaluate"], required=True)
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint (.pth) for evaluation")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--sequence_len", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--augment_train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    safe_mode = False
    if platform.system() == "Windows":
        print("[INFO] Windows detected — enabling DataLoader safe mode (num_workers=0)")
        args.num_workers = 0
        safe_mode = True

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Running on device: {device}")

    outputs_dir = os.path.join(ROOT_DIR, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    try:
        if args.mode == "train":
            train(
                data_dir=args.data_dir,
                output_dir=outputs_dir,
                batch_size=args.batch_size,
                sequence_len=args.sequence_len,
                stride=1,
                num_epochs=args.num_epochs,
                lr=1e-3,
                weight_decay=1e-5,
                num_workers=args.num_workers,
                grad_clip=1.0,
                seed=args.seed,
                num_classes=1,  # regression
                augment_train=args.augment_train,
                safe_mode=safe_mode,
            )
        elif args.mode == "evaluate":
            if not args.checkpoint:
                raise ValueError("--checkpoint is required for evaluation")
            evaluate_model(
                checkpoint_path=args.checkpoint,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                sequence_len=args.sequence_len,
                device=args.device,
                output_dir=outputs_dir,
                num_workers=args.num_workers,
                safe_mode=safe_mode,
            )
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
