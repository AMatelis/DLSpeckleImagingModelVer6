import os
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional

# =========================================================
# Utility functions
# =========================================================

def extract_flowrate_from_filename(filename: str) -> Optional[float]:
    """Extract the first floating-point number from filename."""
    match = re.search(r"([\d.]+)", filename.lower())
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def load_video_frames(video_path: str, target_resolution: Tuple[int, int]) -> List[np.ndarray]:
    """Load grayscale frames from video file and resize to target resolution."""
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

    return frames


def volumetric_flow_to_velocity_map(flow: float, frame_shape: Tuple[int, int], channel_width_um: float) -> np.ndarray:
    """Convert volumetric flow rate to linear velocity map (Poiseuille-like)."""
    H, W = frame_shape
    Q = flow * 1e-9 / 60.0  # µL/min -> m^3/s approx
    R = channel_width_um / 2.0
    V_max = (2 * Q) / (np.pi * R**2)
    x = np.linspace(-R, R, W)
    velocity_profile = V_max * (1 - (x / R) ** 2)
    velocity_map = np.tile(velocity_profile, (H, 1))
    return velocity_map.astype(np.float32)


# =========================================================
# Dataset class
# =========================================================

class SpeckleRegressionDataset(Dataset):
    """PyTorch Dataset for 2D CNN regression predicting velocity maps."""
    def __init__(
        self,
        frames: List[np.ndarray],
        flowrates: List[float],
        normalize: str = "zscore",
        channel_width_um: float = 100.0
    ):
        assert len(frames) == len(flowrates), "Frames and flowrates must match."
        self.frames = frames
        self.flowrates = flowrates
        self.normalize = normalize
        self.channel_width_um = channel_width_um

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        flow = self.flowrates[idx]

        # Normalize frame
        if self.normalize == "scale":
            frame = frame / 255.0
        elif self.normalize == "zscore":
            mu, sd = frame.mean(), frame.std()
            frame = (frame - mu) / (sd + 1e-8)

        frame_tensor = torch.from_numpy(frame[None, ...])  # (1,H,W)

        # Velocity map target
        vel_map = volumetric_flow_to_velocity_map(flow, frame.shape, self.channel_width_um)
        vel_tensor = torch.from_numpy(vel_map[None, ...])  # (1,H,W)

        return frame_tensor, vel_tensor


# =========================================================
# Dataset preparation
# =========================================================

def prepare_dataset(
    data_folder: str,
    stride: int = 1,
    target_resolution: Tuple[int, int] = (128, 128),
    channel_width_um: float = 100.0
) -> Tuple[List[np.ndarray], List[float]]:
    """Extract frames and corresponding flow rates from all videos in folder."""
    all_frames, all_flowrates = [], []

    for filename in sorted(os.listdir(data_folder)):
        if not filename.lower().endswith(".avi"):
            continue

        flowrate = extract_flowrate_from_filename(filename)
        if flowrate is None:
            print(f"[WARN] Skipping file with invalid flowrate: {filename}")
            continue

        video_path = os.path.join(data_folder, filename)
        try:
            frames = load_video_frames(video_path, target_resolution)
        except Exception as e:
            print(f"[ERROR] Failed to load {filename}: {e}")
            continue

        # Sample frames by stride
        for i in range(0, len(frames), stride):
            all_frames.append(frames[i])
            all_flowrates.append(flowrate)

    if not all_frames:
        raise RuntimeError("[ERROR] No valid frames extracted.")

    print(f"[INFO] Total frames prepared: {len(all_frames)}")
    return all_frames, all_flowrates


# =========================================================
# DataLoader creation
# =========================================================

def create_dataloaders(
    data_folder: str,
    batch_size: int = 8,
    test_split: float = 0.2,
    stride: int = 1,
    target_resolution: Tuple[int, int] = (128, 128),
    channel_width_um: float = 100.0,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """Prepare train/validation DataLoaders for linear regression."""
    frames, flowrates = prepare_dataset(
        data_folder, stride=stride, target_resolution=target_resolution, channel_width_um=channel_width_um
    )

    X_train, X_val, y_train, y_val = train_test_split(
        frames, flowrates, test_size=test_split, random_state=seed, shuffle=True
    )

    train_set = SpeckleRegressionDataset(X_train, y_train, channel_width_um=channel_width_um)
    val_set = SpeckleRegressionDataset(X_val, y_val, channel_width_um=channel_width_um)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader
