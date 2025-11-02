import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple
import matplotlib.pyplot as plt


class Linear3DRegressionDataset(Dataset):
    """
    3D Linear Regression Dataset.
    Each sample: (x1, x2, x3) → y
    where y = w1*x1 + w2*x2 + w3*x3 + b + noise.

    Parameters
    ----------
    csv_path : str, optional
        Path to CSV with columns ['x1', 'x2', 'x3', 'y'].
    synthetic : bool
        If True, generate synthetic data instead of loading from CSV.
    n_samples : int
        Number of samples to generate (for synthetic mode).
    noise_std : float
        Gaussian noise standard deviation added to y.
    normalize : {'scale', 'zscore', None}
        Normalization mode for X.
    seed : int
        Random seed for reproducibility.
    save_dir : str, optional
        Directory to save dataset stats, plots, and metadata.
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        synthetic: bool = False,
        n_samples: int = 1000,
        noise_std: float = 0.05,
        normalize: Optional[str] = "zscore",
        seed: int = 42,
        save_dir: Optional[str] = "dataset_info",
    ):
        super().__init__()
        self.synthetic = synthetic
        self.normalize = normalize
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        # Load or generate data
        if synthetic or csv_path is None:
            self._generate_synthetic(n_samples, noise_std)
        else:
            if not os.path.isfile(csv_path):
                raise FileNotFoundError(f"[ERROR] CSV not found: {csv_path}")
            self._load_from_csv(csv_path)

        # Normalize features
        if self.normalize:
            self._normalize()

        # Convert to tensors
        self.X = torch.tensor(self.data[["x1", "x2", "x3"]].values, dtype=torch.float32)
        self.y = torch.tensor(self.data[["y"]].values, dtype=torch.float32)

        # Log and save
        self._save_summary_plots()
        self._save_metadata()

        print(f"[INFO] Dataset prepared: {len(self.data)} samples "
              f"| Mode: {'synthetic' if synthetic else 'real'} "
              f"| Normalization: {normalize}")

    # ==========================================================
    # Private Methods
    # ==========================================================

    def _generate_synthetic(self, n_samples: int, noise_std: float):
        """Generate synthetic 3D regression data."""
        true_w = np.array([1.5, -2.3, 0.9], dtype=np.float32)
        true_b = 0.75
        X = np.random.randn(n_samples, 3).astype(np.float32)
        y = X @ true_w + true_b + np.random.randn(n_samples).astype(np.float32) * noise_std

        self.data = pd.DataFrame(X, columns=["x1", "x2", "x3"])
        self.data["y"] = y
        self.true_params = dict(weights=true_w.tolist(), bias=true_b)
        print(f"[INFO] Generated synthetic dataset (n={n_samples}, noise_std={noise_std})")

    def _load_from_csv(self, csv_path: str):
        """Load dataset from a CSV file."""
        df = pd.read_csv(csv_path)
        required_cols = {"x1", "x2", "x3", "y"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"[ERROR] CSV must contain columns: {required_cols}. Found: {df.columns.tolist()}")
        self.data = df
        print(f"[INFO] Loaded CSV dataset: {csv_path} ({len(df)} samples)")

    def _normalize(self):
        """Normalize features according to selected mode."""
        X = self.data[["x1", "x2", "x3"]].values
        if self.normalize == "scale":
            X_min, X_max = X.min(axis=0), X.max(axis=0)
            X_norm = (X - X_min) / (X_max - X_min + 1e-8)
            self.data[["x1", "x2", "x3"]] = X_norm
            print("[INFO] Applied min-max scaling [0, 1] to features.")
        elif self.normalize == "zscore":
            mean, std = X.mean(axis=0), X.std(axis=0)
            X_norm = (X - mean) / (std + 1e-8)
            self.data[["x1", "x2", "x3"]] = X_norm
            print("[INFO] Applied z-score normalization to features.")
        else:
            print("[INFO] No normalization applied to features.")

    # ==========================================================
    # Public Dataset Interface
    # ==========================================================

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    # ==========================================================
    # Publication-Ready Summaries and Exports
    # ==========================================================

    def _save_summary_plots(self):
        """Generate publication-ready scatter and histogram plots."""
        # Pairwise feature relationships
        pd.plotting.scatter_matrix(self.data[["x1", "x2", "x3", "y"]],
                                   figsize=(8, 8), diagonal="hist", alpha=0.6)
        plt.suptitle("Feature Relationships and Target Distribution", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "scatter_matrix.png"), dpi=300)
        plt.close()

        # Correlation heatmap
        corr = self.data.corr()
        plt.figure(figsize=(4, 4))
        plt.imshow(corr, cmap="coolwarm", interpolation="none")
        plt.colorbar(label="Correlation")
        plt.xticks(range(len(corr)), corr.columns, rotation=45)
        plt.yticks(range(len(corr)), corr.columns)
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "correlation_matrix.png"), dpi=300)
        plt.close()

        print("[INFO] Saved dataset summary plots.")

    def _save_metadata(self):
        """Save dataset metadata and stats."""
        meta = {
            "n_samples": len(self.data),
            "normalize": self.normalize,
            "synthetic": self.synthetic,
            "seed": self.seed,
            "columns": list(self.data.columns),
            "feature_means": self.data[["x1", "x2", "x3"]].mean().to_dict(),
            "feature_stds": self.data[["x1", "x2", "x3"]].std().to_dict(),
        }
        if hasattr(self, "true_params"):
            meta["true_params"] = self.true_params

        with open(os.path.join(self.save_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=4)
        print(f"[INFO] Metadata saved to {self.save_dir}/metadata.json")

    def save_to_csv(self, out_path: str):
        """Save dataset to CSV for reproducibility."""
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self.data.to_csv(out_path, index=False)
        print(f"[INFO] Dataset saved to {out_path}")

    def summary(self) -> dict:
        """Return summary stats for reporting."""
        desc = self.data.describe().to_dict()
        info = {
            "n_samples": len(self.data),
            "normalize": self.normalize,
            "synthetic": self.synthetic,
        }
        if hasattr(self, "true_params"):
            info.update(self.true_params)
        return {**info, **desc}


# ==========================================================
# Sanity Check
# ==========================================================

if __name__ == "__main__":
    dataset = Linear3DRegressionDataset(synthetic=True, n_samples=500, normalize="zscore")
    print(dataset.summary())
