"""Paths, constants and default hyperparameters shared by every Workshop #2 script.

One place for every number the report must quote, so train.py, evaluate.py and
test_model.py cannot drift apart.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

# Anchored to the package, not the working directory, so notebooks and scripts
# resolve the same files.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SILVER_PATH = DATA_DIR / "silver" / "reviews.parquet"
SPLITS_PATH = DATA_DIR / "gold" / "splits.json"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BEST_CHECKPOINT = CHECKPOINT_DIR / "model_best.pt"
RUNS_DIR = PROJECT_ROOT / "runs"

# Workshop #1's frozen split. Its hash proves this repo trains and evaluates on
# exactly the rows the baselines used; a different file is refused, not used.
#the split SHA change
SPLITS_SHA256 = "ab3b8e6826127908e471fc8c1dc5f49ab619ac7780fc8e17847b3e26632267cb"

SILVER_ROWS = 42_632

RANDOM_STATE = 42
TEXT_COL = "Review_Text"
TARGET_COL = "sentiment"
# Same order as Workshop #1, so confusion matrices and reports line up.
LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


@dataclass
class TrainConfig:
    """Every hyperparameter of a run. Saved inside the checkpoint and the run log."""

    model_name: str = "bert-base-uncased"
    max_len: int = 256
    # Head+tail truncation: long reviews keep their opening and their verdict.
    # 64 + 190 content tokens + [CLS] + [SEP] = 256.
    head_tokens: int = 64
    batch_size: int = 32
    eval_batch_size: int = 64
    # Phase 1: encoder frozen, only the classifier head learns.
    frozen_epochs: int = 1
    head_lr: float = 1e-3
    # Phase 2: whole network, small learning rate.
    finetune_epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0
    # Stop phase 2 when validation macro-F1 has not improved for this many epochs.
    patience: int = 1
    bf16: bool = True
    seed: int = RANDOM_STATE
    # Debug knob: keep only the first N rows of each split (0 = all of them).
    limit: int = 0

    def to_dict(self) -> dict:
        """Plain-dict view, for checkpoints and logs."""
        return asdict(self)


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Seed Python, numpy and torch (CPU and GPU), as the brief requires."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(preferred: str | None = None) -> torch.device:
    """The GPU when present (ROCm shows up through the torch.cuda API), else CPU."""
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
