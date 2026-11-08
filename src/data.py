"""Data loading for the transformer: Workshop #1's split, tokenised once.

The silver layer and the frozen split come from Workshop #1 unchanged (see
README.md). Nothing here re-cleans or re-splits: the only new step is turning
review text into token ids.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from .config import (
    LABEL_TO_ID,
    LABELS,
    SILVER_PATH,
    SILVER_ROWS,
    SPLITS_PATH,
    SPLITS_SHA256,
    TARGET_COL,
    TEXT_COL,
    TrainConfig,
)

SPLIT_NAMES = ("train", "validation", "test")


def load_split_frames(limit: int = 0) -> dict[str, pd.DataFrame]:
    """Return {split: DataFrame} for Workshop #1's frozen split, after verifying it.

    The split file must hash to SPLITS_SHA256 and silver must have the expected
    row count; anything else means the data is not what the baselines saw, and
    any comparison with them would be meaningless.
    """
    for path in (SILVER_PATH, SPLITS_PATH):
        if not path.exists():
            msg = f"{path} is missing; see README.md, section 'Data'."
            raise FileNotFoundError(msg)

    digest = hashlib.sha256(SPLITS_PATH.read_bytes()).hexdigest()
    if digest != SPLITS_SHA256:
        msg = f"{SPLITS_PATH} hashes to {digest}, not Workshop #1's split {SPLITS_SHA256}."
        raise ValueError(msg)

    silver = pd.read_parquet(SILVER_PATH)
    if len(silver) != SILVER_ROWS:
        msg = f"{SILVER_PATH} has {len(silver):,} rows, expected {SILVER_ROWS:,}."
        raise ValueError(msg)

    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    frames = {}
    for name in SPLIT_NAMES:
        rows = silver.loc[splits[name], [TEXT_COL, TARGET_COL]]
        frames[name] = rows.head(limit) if limit else rows
    return frames


def encode_head_tail(texts, tokenizer, max_len: int, head_tokens: int) -> list[list[int]]:
    """Tokenise, keeping the first `head_tokens` and the last tokens of long reviews.

    Plain truncation keeps only the opening, but reviews tend to end with the
    verdict ("...overall, would not go back"). Keeping head and tail is the
    standard fix for long reviews and costs nothing extra.
    """
    # Every candidate model wraps a single sequence as <cls> ... <sep>: [CLS]/[SEP]
    # for BERT and DistilBERT, <s>/</s> for RoBERTa (exposed under the same names).
    budget = max_len - 2
    tail_tokens = budget - head_tokens
    encoded = tokenizer(
        list(texts), add_special_tokens=False, truncation=False, verbose=False
    )["input_ids"]

    sequences = []
    for ids in encoded:
        if len(ids) > budget:
            ids = ids[:head_tokens] + ids[-tail_tokens:]
        sequences.append([tokenizer.cls_token_id, *ids, tokenizer.sep_token_id])
    return sequences


class ReviewDataset(Dataset):
    """Pre-tokenised reviews and integer labels."""

    def __init__(self, input_ids: list[list[int]], labels: np.ndarray):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.input_ids[index], int(self.labels[index])


def make_collate(pad_token_id: int):
    """Pad each batch only to its own longest review (dynamic padding)."""

    def collate(batch):
        sequences, labels = zip(*batch)
        width = max(len(ids) for ids in sequences)
        input_ids = torch.full((len(batch), width), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, ids in enumerate(sequences):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, : len(ids)] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def encode_labels(series: pd.Series) -> np.ndarray:
    """Map label strings to ids in the fixed LABELS order."""
    return series.astype(str).map(LABEL_TO_ID).to_numpy()


def class_weights(y_train: np.ndarray) -> torch.Tensor:
    """Balanced weights from the TRAIN labels only, as in Workshop #1."""
    weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(LABELS)), y=y_train
    )
    return torch.tensor(weights, dtype=torch.float32)


def build_loaders(tokenizer, config: TrainConfig, splits=SPLIT_NAMES):
    """Return ({split: DataLoader}, train labels) for the requested splits."""
    frames = load_split_frames(config.limit)
    collate = make_collate(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(config.seed)

    loaders = {}
    for name in splits:
        frame = frames[name]
        dataset = ReviewDataset(
            encode_head_tail(
                frame[TEXT_COL], tokenizer, config.max_len, config.head_tokens
            ),
            encode_labels(frame[TARGET_COL]),
        )
        is_train = name == "train"
        loaders[name] = DataLoader(
            dataset,
            batch_size=config.batch_size if is_train else config.eval_batch_size,
            shuffle=is_train,
            collate_fn=collate,
            generator=generator if is_train else None,
        )
    return loaders, encode_labels(frames["train"][TARGET_COL])
