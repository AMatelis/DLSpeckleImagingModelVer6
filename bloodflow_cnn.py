import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================================
# Utilities
# ==========================================================

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_synthetic_3d_data(num_samples=64, shape=(1, 16, 128, 128)):
    """Generate synthetic 3D data for regression."""
    x = torch.randn(num_samples, *shape)
    y = torch.randn(num_samples, 1) * 5 + 2
    return TensorDataset(x, y)

# ==========================================================
# 3D CNN Model
# ==========================================================

class BloodFlowCNN3D(nn.Module):
    """
    3D CNN for blood flow velocity regression.
    Input: (B, C=1, D, H, W)
    Output: (B, 1)
    """
    def __init__(self, input_channels=1, output_channels=1, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

# ==========================================================
# Evaluation & Plotting
# ==========================================================

def evaluate_model(model, loader, criterion, device):
    """Evaluate model and return metrics and predictions."""
    model.eval()
    preds, trues = [], []
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item() * x.size(0)
            preds.append(y_pred.cpu())
            trues.append(y.cpu())
    preds = torch.cat(preds)
    trues = torch.cat(trues)
    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    return total_loss / len(loader.dataset), mse, mae, r2, preds, trues

def plot_predictions(trues, preds, out_path, title="Predicted vs True"):
    """Publication-ready scatter plot."""
    plt.figure(figsize=(6,6))
    plt.scatter(trues, preds, s=30, alpha=0.6)
    lims = [min(trues.min(), preds.min()), max(trues.max(), preds.max())]
    plt.plot(lims, lims, 'r--', label="Perfect Fit")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_loss_curve(train_losses, val_losses, out_path):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(7,5))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ==========================================================
# Training Loop (3D CNN)
# ==========================================================

def train_3d_cnn(
    input_shape=(1, 16, 128, 128),
    epochs=50,
    batch_size=8,
    lr=1e-3,
    save_dir="results_3d_cnn"
):
    set_seed(42)
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loaders
    train_data = generate_synthetic_3d_data(80, shape=input_shape)
    val_data = generate_synthetic_3d_data(20, shape=input_shape)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    # Model, loss, optimizer
    model = BloodFlowCNN3D(input_channels=input_shape[0]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        val_loss, mse, mae, r2, preds, trues = evaluate_model(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Save predictions CSV & plot every epoch
        pd.DataFrame({"True": trues.squeeze().numpy(), "Predicted": preds.squeeze().numpy()})\
            .to_csv(os.path.join(save_dir, f"epoch_{epoch:03d}_predictions.csv"), index=False)
        plot_predictions(trues, preds, os.path.join(save_dir, f"epoch_{epoch:03d}_scatter.png"),
                         title=f"Epoch {epoch}: Predicted vs True (R²={r2:.3f})")

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"MSE: {mse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

    # Final loss curve
    plot_loss_curve(history["train_loss"], history["val_loss"], os.path.join(save_dir, "loss_curve.png"))
    print("\n✅ Training complete. Best model + plots saved in:", save_dir)

# ==========================================================
# Standalone Test
# ==========================================================

if __name__ == "__main__":
    train_3d_cnn()
