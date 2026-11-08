"""Extract BERT CLS embeddings for the full silver layer.

Runs once and saves:
    data/gold/embeddings_bert.npy   shape (42632, 768) float32
    data/gold/labels.npy            shape (42632,)     int8

Then splits.json indices remain valid: embed[splits["train"]] gives train vectors.

Usage:
    python src/extract_embeddings.py --checkpoint checkpoints/model_best.pt --batch_size 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from config import (
    LABEL_TO_ID,
    SILVER_PATH,
    SILVER_ROWS,
    SPLITS_PATH,
    SPLITS_SHA256,
    TARGET_COL,
    TEXT_COL,
    TrainConfig,
)

OUT_EMBEDDINGS = Path("data/gold/embeddings_bert.npy")
OUT_LABELS     = Path("data/gold/labels.npy")

_cfg        = TrainConfig()
MAX_LEN     = _cfg.max_len       # 256
HEAD_TOKENS = _cfg.head_tokens   # 64


# ── dataset ───────────────────────────────────────────────────────────────────
class FullSilverDataset(Dataset):
    """Tokenised full silver, in row order (no shuffle ever)."""

    def __init__(self, texts: list[str], tokenizer, max_len: int, head_tokens: int):
        budget = max_len - 2
        tail_tokens = budget - head_tokens
        encoded = tokenizer(
            texts, add_special_tokens=False, truncation=False, verbose=False
        )["input_ids"]

        self.sequences = []
        for ids in encoded:
            if len(ids) > budget:
                ids = ids[:head_tokens] + ids[-tail_tokens:]
            self.sequences.append(
                [tokenizer.cls_token_id, *ids, tokenizer.sep_token_id]
            )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_fn(pad_id: int):
    def _collate(batch):
        width = max(len(ids) for ids in batch)
        input_ids      = torch.full((len(batch), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, ids in enumerate(batch):
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :len(ids)] = 1
        return input_ids, attention_mask
    return _collate


# ── helpers ───────────────────────────────────────────────────────────────────
def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load checkpoint, detect model_name, return (model_name, state_dict)."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint no encontrado: {ckpt_path}\n"
            "Este script requiere el checkpoint de W2. No puede correr sin él."
        )

    state = torch.load(ckpt_path, map_location=device)

    # Intentar detectar model_name guardado por TrainConfig en el checkpoint
    model_name = None
    if isinstance(state, dict):
        cfg = state.get("config", {})
        if isinstance(cfg, dict):
            model_name = cfg.get("model_name", None)

    if model_name is None:
        # Fallback: usar el de TrainConfig por defecto
        model_name = _cfg.model_name
        print(f"model_name no encontrado en checkpoint — usando default: {model_name}")
    else:
        print(f"model_name detectado desde checkpoint: {model_name}")

    # El checkpoint de W2 guarda model.state_dict() donde el encoder
    # está bajo el prefijo "encoder." → hay que stripear ese prefijo
    # para que AutoModel pueda cargar los pesos directamente.
    raw = state.get("state_dict", state.get("model_state_dict", state))
    state_dict = {
        k[len("encoder."):]: v
        for k, v in raw.items()
        if k.startswith("encoder.")
    }
    if not state_dict:
        raise ValueError("No se encontraron pesos bajo el prefijo 'encoder.' en el checkpoint.")
    return model_name, state_dict


# ── main ──────────────────────────────────────────────────────────────────────
def main(checkpoint: str, batch_size: int):
    # 1. Verify splits hash
    digest = hashlib.sha256(SPLITS_PATH.read_bytes()).hexdigest()
    if digest != SPLITS_SHA256:
        raise ValueError(
            f"splits.json hash mismatch.\n"
            f"  Esperado : {SPLITS_SHA256}\n"
            f"  Obtenido : {digest}\n"
            "Copia el splits.json original del repositorio W1/W2."
        )

    # 2. Load full silver in order
    silver = pd.read_parquet(SILVER_PATH)
    if len(silver) != SILVER_ROWS:
        raise ValueError(f"Expected {SILVER_ROWS} rows, got {len(silver)}")

    texts  = silver[TEXT_COL].tolist()
    labels = silver[TARGET_COL].map(LABEL_TO_ID).to_numpy(dtype=np.int8)

    # 3. Load checkpoint → detect model_name → build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_name, state_dict = load_checkpoint(checkpoint, device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name)

    # Cargar solo los pesos del encoder (ignorar cabeza clasificadora si existe)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    encoder_missing = [k for k in missing if not k.startswith("classifier")]
    if encoder_missing:
        print(f"⚠ Pesos del encoder no cargados: {encoder_missing[:5]}")
    print(f"Checkpoint cargado correctamente desde: {checkpoint}")

    model.to(device)
    model.eval()

    # 4. Build dataloader (no shuffle, full silver)
    dataset = FullSilverDataset(texts, tokenizer, MAX_LEN, HEAD_TOKENS)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn(tokenizer.pad_token_id),
    )

    # 5. Extract CLS tokens
    all_cls = []
    total   = len(dataset)
    done    = 0

    with torch.no_grad():
        for input_ids, attention_mask in loader:
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # last_hidden_state[:, 0, :] es el CLS token
            cls = outputs.last_hidden_state[:, 0, :].cpu().float().numpy()
            all_cls.append(cls)

            done += len(cls)
            print(f"  {done}/{total}", end="\r", flush=True)

    embeddings = np.concatenate(all_cls, axis=0)
    print(f"\nEmbeddings shape: {embeddings.shape}")

    # 6. Save
    OUT_EMBEDDINGS.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_EMBEDDINGS, embeddings)
    np.save(OUT_LABELS, labels)
    print(f"Saved → {OUT_EMBEDDINGS}")
    print(f"Saved → {OUT_LABELS}")

    # 7. Sanity check con splits
    splits = json.loads(SPLITS_PATH.read_text())
    for name in ("train", "validation", "test"):
        idx = splits[name]
        print(f"  {name:>10}: {len(idx)} rows, embed shape {embeddings[idx].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="checkpoints/model_best.pt",
        help="Path al checkpoint de W2 (requerido)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size para inferencia"
    )
    args = parser.parse_args()
    main(args.checkpoint, args.batch_size)