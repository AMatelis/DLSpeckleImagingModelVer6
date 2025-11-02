import os
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional, Union

# =========================================================
# Utility functions
# =========================================================

def extract_flowrate_from_filename(filename: str) -> Optional[float]:
    """Extract the first numeric flowrate value from a filename."""
    match = re.search(r"([\d.]+)", filename.lower())
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def load_video_frames(video_path: str, target_resolution: Tuple[int, int]) -> np.ndarray:
    """Load video frames as grayscale and resize to target resolution."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"[ERROR] Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"[ERROR] Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, target_resolution, interpolation=cv2.INTER_AREA)
        frames.append(gray.astype(np.float32))
    cap.release()

    if not frames:
        raise RuntimeError(f"[ERROR] No frames found in: {video_path}")

    return np.stack(frames, axis=0)

def volumetric_flow_to_velocity(flow: float, channel_width_um: float) -> float:
    """Convert volumetric flow (µL/min) to average velocity magnitude (m/s)."""
    Q = flow * 1e-9 / 60.0  # µL/min -> m^3/s
    R = channel_width_um / 2e6  # µm -> m
    A = np.pi * R**2
    return Q / A

# =========================================================
# Dataset Class
# =========================================================

class Speckle3DRegressionDataset(Dataset):
    """3D Speckle Dataset for volumetric regression tasks."""
    def __init__(self,
                 volumes: List[np.ndarray],
                 flowrates: List[float],
                 normalize: str = "zscore",
                 channel_width_um: float = 100.0):
        assert len(volumes) == len(flowrates), "[ERROR] Volume/label mismatch."
        self.volumes = volumes
        self.flowrates = flowrates
        self.normalize = normalize
        self.channel_width_um = channel_width_um

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        vol = self.volumes[idx]
        flow = self.flowrates[idx]

        # Normalize volume
        if self.normalize == "scale":
            vol = vol / 255.0
        elif self.normalize == "zscore":
            mu, sd = vol.mean(), vol.std()
            vol = (vol - mu) / (sd + 1e-8)

        # Convert to tensor: (C,D,H,W)
        vol_tensor = torch.from_numpy(vol[None, ...])
        target_tensor = torch.tensor([volumetric_flow_to_velocity(flow, self.channel_width_um)],
                                     dtype=torch.float32)
        return vol_tensor, target_tensor

# =========================================================
# Dataset Preparation
# =========================================================

def prepare_3d_dataset(
    data_folder: str = r"C:\Users\andre\Downloads\SpeckleModelProject\SpeckleModelCodeVer6\data",
    stride: int = 1,
    frame_depth: int = 16,
    target_resolution: Tuple[int, int] = (128, 128),
    video_extensions: Tuple[str,...] = (".avi", ".mp4")
) -> Tuple[List[np.ndarray], List[float]]:
    """Load videos, segment into 3D volumes, return (volumes, flowrates)."""
    all_volumes = []
    all_flowrates = []

    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"[ERROR] Data folder not found: {data_folder}")

    video_files = [f for f in sorted(os.listdir(data_folder)) if f.lower().endswith(video_extensions)]
    if not video_files:
        raise RuntimeError(f"[ERROR] No video files found in {data_folder}")

    for fname in video_files:
        flowrate = extract_flowrate_from_filename(fname)
        if flowrate is None:
            print(f"[WARN] Skipping {fname} (invalid flowrate).")
            continue

        video_path = os.path.join(data_folder, fname)
        try:
            frames = load_video_frames(video_path, target_resolution)
        except Exception as e:
            print(f"[ERROR] Failed to load {fname}: {e}")
            continue

        for i in range(0, len(frames) - frame_depth + 1, stride):
            all_volumes.append(frames[i:i+frame_depth])
            all_flowrates.append(flowrate)

    if not all_volumes:
        raise RuntimeError("[ERROR] No valid volumes extracted.")

    print(f"[INFO] Prepared {len(all_volumes)} volumetric samples from {len(video_files)} videos.")
    return all_volumes, all_flowrates

# =========================================================
# Dataloader Creation
# =========================================================

def create_3d_dataloaders(
    data_folder: str = r"C:\Users\andre\Downloads\SpeckleModelProject\SpeckleModelCodeVer6\data",
    batch_size: int = 4,
    test_split: float = 0.2,
    frame_depth: int = 16,
    stride: int = 4,
    target_resolution: Tuple[int, int] = (128, 128),
    channel_width_um: float = 100.0,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """Prepare train and validation dataloaders for 3D regression."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    volumes, flowrates = prepare_3d_dataset(
        data_folder,
        stride=stride,
        frame_depth=frame_depth,
        target_resolution=target_resolution
    )

    X_train, X_val, y_train, y_val = train_test_split(
        volumes, flowrates, test_size=test_split, random_state=seed, shuffle=True
    )

    train_set = Speckle3DRegressionDataset(X_train, y_train, channel_width_um=channel_width_um)
    val_set = Speckle3DRegressionDataset(X_val, y_val, channel_width_um=channel_width_um)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    print(f"[INFO] Train samples: {len(train_set)} | Validation samples: {len(val_set)}")
    return train_loader, val_loader
