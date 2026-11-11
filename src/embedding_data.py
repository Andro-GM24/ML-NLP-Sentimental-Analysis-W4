"""Shared loaders for the latent-space analysis and the VAE augmentation.

The RoBERTa CLS embeddings (src/extract_embeddings.py) are indexed like the silver
layer, so Workshop #1's split indices select each split directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from autoencoder import AE, VAE
from config import LABELS, PROJECT_ROOT, SPLITS_PATH

EMB_PATH = PROJECT_ROOT / "data" / "gold" / "embeddings_bert.npy"
LABELS_PATH = PROJECT_ROOT / "data" / "gold" / "labels.npy"
SCALER_PATH = PROJECT_ROOT / "checkpoints" / "scaler.pkl"
W2_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "model_best.pt"
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"
RUNS_DIR = PROJECT_ROOT / "runs" / "workshop4"

# Same colour per class as the Workshop #2 figures (palette validated for colour blindness).
CLASS_COLORS = dict(zip(LABELS, ["#2a78d6", "#eb6834", "#1baf7a"]))


def load_splits() -> dict[str, dict[str, np.ndarray]]:
    """{split: {"X": raw CLS vectors, "y": label ids}} for train, validation and test."""
    if not EMB_PATH.exists():
        msg = f"{EMB_PATH} is missing: run src/extract_embeddings.py first."
        raise FileNotFoundError(msg)
    embeddings = np.load(EMB_PATH)
    labels = np.load(LABELS_PATH).astype(np.int64)
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    return {name: {"X": embeddings[splits[name]], "y": labels[splits[name]]}
            for name in ("train", "validation", "test")}


def load_scaler(train_X: np.ndarray) -> StandardScaler:
    """The StandardScaler fitted on train (notebooks/exploration.ipynb), fitted here if absent."""
    if SCALER_PATH.exists():
        return joblib.load(SCALER_PATH)
    scaler = StandardScaler().fit(train_X)
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    return scaler


def load_autoencoder(path: Path):
    """Rebuild an AE or VAE checkpoint written by src/train_ae.py; returns the model in eval mode."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    is_vae = any(key.startswith("encoder.fc_mu") for key in checkpoint["state_dict"])
    if is_vae:
        model = VAE(z_dim=config["z_dim"], dropout=config["dropout"],
                    input_dropout=config.get("input_dropout", 0.2))
    else:
        model = AE(z_dim=config["z_dim"], dropout=config["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


@torch.no_grad()
def encode(model, X_scaled: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Latent codes: z for the AE, µ for the VAE (as the brief asks)."""
    parts = [model.encode(torch.from_numpy(X_scaled[i:i + batch_size].astype(np.float32))).numpy()
             for i in range(0, len(X_scaled), batch_size)]
    return np.concatenate(parts)
