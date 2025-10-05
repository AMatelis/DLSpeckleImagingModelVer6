import os
import re
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def extract_flow_from_filename(fname: str) -> Optional[float]:
    """
    Extract numeric volumetric flow rate from a filename.
    Returns None if no valid number is found.
    """
    m = re.search(r"([0-9]+\.?[0-9]*)", fname)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def read_video_frames(video_path: str) -> List[np.ndarray]:
    """
    Read all frames from a video in grayscale.
    Returns a list of float32 numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[ERROR] Could not open video: {video_path}")

    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame.astype(np.float32))
    cap.release()
    return frames


def volumetric_flow_to_velocity_map(flow: float, frame_shape: Tuple[int, int], channel_width_um: float) -> np.ndarray:
    """
    Convert volumetric flow rate (µL/min) to a linear velocity map across the channel width
    using the Poiseuille profile. Assumes parabolic profile across width.
    Args:
        flow: volumetric flow rate (µL/min)
        frame_shape: (H, W) shape of frame in pixels
        channel_width_um: physical channel width in microns
    Returns:
        velocity_map: np.ndarray of shape (H, W)
    """
    H, W = frame_shape
    # Convert flow from µL/min to µm^3/s
    Q = flow * 1e-9 / 60.0  # µL/min → m^3/s → µm^3/s approximation
    R = channel_width_um / 2.0  # assume circular channel, radius in µm

    # Poiseuille max velocity
    V_max = (2 * Q) / (np.pi * R**2)  # simple approximation

    # Create parabolic profile across width
    x = np.linspace(-R, R, W)
    velocity_profile = V_max * (1 - (x / R)**2)
    velocity_map = np.tile(velocity_profile, (H, 1))  # same for all rows
    return velocity_map.astype(np.float32)


class Speckle2DRegressionDataset(Dataset):
    """
    2D Speckle flow dataset for LINEAR REGRESSION predicting velocity maps.

    Args:
        folder: path containing .avi videos
        stride: frame sampling step
        normalize: 'scale' or 'zscore'
        cache_videos: cache full videos in memory
        shuffle_frames: shuffle frames within videos
        channel_width_um: physical channel width in microns for velocity mapping
    """

    def __init__(
        self,
        folder: str,
        stride: int = 1,
        normalize: str = "scale",
        cache_videos: bool = False,
        shuffle_frames: bool = False,
        channel_width_um: float = 100.0,
    ):
        self.folder = folder
        self.stride = stride
        self.normalize = normalize
        self.cache_videos = cache_videos
        self.shuffle_frames = shuffle_frames
        self.channel_width_um = channel_width_um

        self.samples: List[Tuple[str, int, float]] = []
        self.cache: Dict[str, List[np.ndarray]] = {}

        if not os.path.isdir(folder):
            raise NotADirectoryError(f"[ERROR] Dataset folder not found: {folder}")

        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".avi")])
        if not files:
            raise RuntimeError(f"[ERROR] No .avi files found in {folder}")

        # Build samples
        for fn in files:
            flow = extract_flow_from_filename(fn)
            if flow is None:
                print(f"[WARN] Could not parse flow rate from filename: {fn}")
                continue

            path = os.path.join(folder, fn)
            frames = read_video_frames(path)
            if len(frames) == 0:
                print(f"[WARN] No frames found in video: {fn}")
                continue

            if cache_videos:
                self.cache[path] = frames

            indices = list(range(0, len(frames), stride))
            if shuffle_frames:
                np.random.shuffle(indices)

            for i in indices:
                self.samples.append((path, i, flow))

        if not self.samples:
            raise RuntimeError(f"[ERROR] No valid frames found in {folder}")

        print(f"[INFO] Prepared {len(self.samples)} regression samples from {len(files)} videos.")

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.normalize == "scale":
            frame = frame / 255.0
        elif self.normalize == "zscore":
            mu, sd = frame.mean(), frame.std()
            frame = (frame - mu) / (sd + 1e-8)
        return frame.astype(np.float32)

    def _get_frame(self, path: str, idx: int) -> torch.Tensor:
        if path in self.cache:
            frame = self.cache[path][idx]
        else:
            frames = read_video_frames(path)
            frame = frames[idx]
        frame = self._normalize_frame(frame)
        frame = frame[None, ...]  # (1, H, W)
        return torch.from_numpy(frame)

    def _get_velocity_map(self, flow: float, frame_shape: Tuple[int, int]) -> torch.Tensor:
        vel_map = volumetric_flow_to_velocity_map(flow, frame_shape, self.channel_width_um)
        return torch.from_numpy(vel_map[None, ...])  # (1, H, W)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, frame_idx, flow = self.samples[idx]
        x = self._get_frame(path, frame_idx)
        y = self._get_velocity_map(flow, x.shape[1:])  # velocity map matches frame size
        return x, y
