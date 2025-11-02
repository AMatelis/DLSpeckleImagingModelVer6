import argparse
import os
import sys
import platform
from typing import Optional, List, Tuple

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Set root path for imports
# ---------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "models"))
    sys.path.append(os.path.join(ROOT_DIR, "src"))

from src.train import train  # 3D train function
from src.dataloader import create_3d_dataloaders  # robust 3D dataloader
from models.bloodflow_cnn import BloodFlowCNN3D  # 3D CNN model


# =========================================================
# Evaluation
# =========================================================

def evaluate_model(
    checkpoint_path: str,
    data_dir: str,
    batch_size: int = 4,
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
    num_workers: int = 0,
    stride: int = 1,
    frame_depth: int = 16,
    target_resolution: Tuple[int,int] = (128,128),
    channel_width_um: float = 100.0,
) -> None:
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Evaluating on device: {device}")

    # Load dataloaders
    train_loader, val_loader = create_3d_dataloaders(
        data_folder=data_dir,
        batch_size=batch_size,
        stride=stride,
        frame_depth=frame_depth,
        target_resolution=target_resolution,
        channel_width_um=channel_width_um,
        num_workers=num_workers,
    )

    if val_loader is None or len(val_loader) == 0:
        raise RuntimeError("Validation data not found or improperly formatted.")

    # Load model
    model = BloodFlowCNN3D(input_channels=1, output_channels=1).to(device)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
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

    preds = np.array(all_preds)
    targets = np.array(all_targets)
    mse = np.mean((preds - targets)**2)
    mae = np.mean(np.abs(preds - targets))
    print("\n[RESULTS]")
    print(f"MSE: {mse:.6f} | MAE: {mae:.6f}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame({"Target": all_targets, "Prediction": all_preds})
        df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

        plt.figure(figsize=(6,6))
        plt.scatter(targets, preds, alpha=0.6)
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
        plt.xlabel("True Velocity (m/s)")
        plt.ylabel("Predicted Velocity (m/s)")
        plt.title("Predicted vs True Velocity")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "prediction_scatter.png"))
        plt.close()


# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D Blood Flow Regression")
    parser.add_argument("--mode", choices=["train", "evaluate"], required=True)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"C:\Users\andre\Downloads\SpeckleModelProject\SpeckleModelCodeVer6\data",
        help="Path to dataset folder (defaults to your local data folder)"
    )
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path for evaluation")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--frame_depth", type=int, default=16)
    parser.add_argument("--target_resolution", type=int, nargs=2, default=(128,128))
    parser.add_argument("--channel_width_um", type=float, default=100.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment_train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if platform.system() == "Windows":
        print("[INFO] Windows detected — enabling safe mode (num_workers=0)")
        args.num_workers = 0

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Running on device: {device}")

    outputs_dir = os.path.join(ROOT_DIR, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    if args.mode == "train":
        train(
            data_dir=args.data_dir,
            output_dir=outputs_dir,
            batch_size=args.batch_size,
            stride=args.stride,
            num_epochs=args.num_epochs,
            grad_clip=args.grad_clip,
            num_workers=args.num_workers,
            seed=args.seed,
            target_resolution=tuple(args.target_resolution),
            channel_width_um=args.channel_width_um,
        )
    elif args.mode == "evaluate":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for evaluation")
        evaluate_model(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=outputs_dir,
            num_workers=args.num_workers,
            stride=args.stride,
            frame_depth=args.frame_depth,
            target_resolution=tuple(args.target_resolution),
            channel_width_um=args.channel_width_um,
        )


if __name__ == "__main__":
    main()
