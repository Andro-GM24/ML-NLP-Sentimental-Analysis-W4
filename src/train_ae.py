"""Train AE and VAE models on RoBERTa CLS embeddings.

Trains four models and saves the best checkpoint for each:
    checkpoints/ae_best.pt
    checkpoints/vae_b0.5_best.pt
    checkpoints/vae_b1.0_best.pt
    checkpoints/vae_b4.0_best.pt

Logs go to runs/workshop4/<run_name>/

Usage:
    poetry run python src/train_ae.py
    poetry run python src/train_ae.py --epochs 50 --z_dim 64 --batch_size 256
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from autoencoder import AE, VAE, ae_loss, vae_loss
from config import SPLITS_PATH, get_device, set_seed

CHECKPOINT_DIR = Path("checkpoints")
RUNS_DIR       = Path("runs/workshop4")
EMB_PATH       = Path("data/gold/embeddings_bert.npy")
LABELS_PATH    = Path("data/gold/labels.npy")
SCALER_PATH    = CHECKPOINT_DIR / "scaler.pkl"


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class AEConfig:
    z_dim:        int   = 64
    dropout:      float = 0.2
    input_dropout: float = 0.2
    epochs:       int   = 40
    batch_size:   int   = 256
    lr:           float = 1e-3
    weight_decay: float = 1e-5
    patience:     int   = 10      # early stopping on val loss
    kl_anneal_epochs: int = 10    # ramp KL weight from 0 to 1 over first N epochs
    seed:         int   = 42
    betas: tuple  = (0.5, 1.0, 4.0)

    def to_dict(self) -> dict:
        return asdict(self)


# ── data ──────────────────────────────────────────────────────────────────────

def load_data(config: AEConfig, device: torch.device):
    """Load embeddings, apply scaler, return DataLoaders and anomaly mask."""
    emb    = np.load(EMB_PATH)
    labels = np.load(LABELS_PATH)
    splits = json.loads(SPLITS_PATH.read_text())

    scaler = joblib.load(SCALER_PATH)

    X_train = scaler.transform(emb[splits["train"]]).astype(np.float32)
    X_val   = scaler.transform(emb[splits["validation"]]).astype(np.float32)
    X_test  = scaler.transform(emb[splits["test"]]).astype(np.float32)

    y_train = labels[splits["train"]]
    y_val   = labels[splits["validation"]]
    y_test  = labels[splits["test"]]

    # Anomaly detection: train AE only on positive class (label=2)
    pos_mask = y_train == 2
    X_train_pos = X_train[pos_mask]
    print(f"Train total: {len(X_train)} | positive only: {len(X_train_pos)}")

    def make_loader(X, shuffle=False):
        ds = TensorDataset(torch.from_numpy(X))
        return DataLoader(ds, batch_size=config.batch_size, shuffle=shuffle)

    loaders = {
        "train_pos": make_loader(X_train_pos, shuffle=True),   # for AE training
        "train_all": make_loader(X_train),                      # for latent analysis
        "val":       make_loader(X_val),
        "test":      make_loader(X_test),
    }
    data = {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
    }
    return loaders, data


# ── training loop ─────────────────────────────────────────────────────────────

def run_epoch_ae(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for (x,) in loader:
            x = x.to(device)
            x_hat, _ = model(x)
            loss = ae_loss(x, x_hat)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(x)
            n += len(x)
    return total_loss / n


def run_epoch_vae(model, loader, optimizer, device, beta, kl_weight, train=True):
    model.train() if train else model.eval()
    total, recon_sum, kl_sum, n = 0.0, 0.0, 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for (x,) in loader:
            x = x.to(device)
            x_hat, mu, log_var = model(x)
            loss, recon, kl = vae_loss(x, x_hat, mu, log_var, beta=beta, kl_weight=kl_weight)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total      += loss.item()  * len(x)
            recon_sum  += recon.item() * len(x)
            kl_sum     += kl.item()    * len(x)
            n          += len(x)
    return total / n, recon_sum / n, kl_sum / n


def train_ae(config: AEConfig, loaders, device, run_dir: Path):
    writer = SummaryWriter(run_dir / "ae")
    model  = AE(z_dim=config.z_dim, dropout=config.dropout).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=config.lr,
                              weight_decay=config.weight_decay)

    best_val, stale, best_state = float("inf"), 0, None
    ckpt_path = CHECKPOINT_DIR / "ae_best.pt"

    for epoch in range(1, config.epochs + 1):
        t_loss = run_epoch_ae(model, loaders["train_pos"], opt, device, train=True)
        v_loss = run_epoch_ae(model, loaders["val"],       opt, device, train=False)

        writer.add_scalars("loss", {"train": t_loss, "val": v_loss}, epoch)
        print(f"[AE] epoch {epoch:3d} | train {t_loss:.4f} | val {v_loss:.4f}", end="")

        if v_loss < best_val:
            best_val   = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            print(" ✓")
        else:
            stale += 1
            print(f" (stale {stale}/{config.patience})")
            if stale >= config.patience:
                print(f"  Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "config": config.to_dict(),
                "best_val_loss": best_val}, ckpt_path)
    print(f"AE saved → {ckpt_path}  (best val loss {best_val:.4f})")
    writer.close()
    return model


def train_vae(config: AEConfig, loaders, device, run_dir: Path, beta: float):
    tag    = f"vae_b{beta}"
    writer = SummaryWriter(run_dir / tag)
    model  = VAE(z_dim=config.z_dim, dropout=config.dropout, input_dropout=config.input_dropout).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=config.lr,
                              weight_decay=config.weight_decay)

    best_val, stale, best_state = float("inf"), 0, None
    ckpt_path = CHECKPOINT_DIR / f"{tag}_best.pt"

    for epoch in range(1, config.epochs + 1):
        # KL annealing: ramp kl_weight from 0 to 1 over first kl_anneal_epochs
        kl_weight = min(1.0, epoch / max(1, config.kl_anneal_epochs))

        t_total, t_recon, t_kl = run_epoch_vae(
            model, loaders["train_pos"], opt, device, beta, kl_weight, train=True)
        v_total, v_recon, v_kl = run_epoch_vae(
            model, loaders["val"],       opt, device, beta, kl_weight, train=False)

        writer.add_scalars("loss",  {"train": t_total, "val": v_total}, epoch)
        writer.add_scalars("recon", {"train": t_recon, "val": v_recon}, epoch)
        writer.add_scalars("kl",    {"train": t_kl,    "val": v_kl},    epoch)
        writer.add_scalar("kl_weight", kl_weight, epoch)

        print(f"[VAE β={beta}] epoch {epoch:3d} | "
              f"train {t_total:.4f} (recon {t_recon:.4f} kl {t_kl:.4f}) | "
              f"val {v_total:.4f} | kl_w {kl_weight:.2f}", end="")

        if v_total < best_val:
            best_val   = v_total
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            print(" done ")
        else:
            stale += 1
            print(f" (stale {stale}/{config.patience})")
            if stale >= config.patience:
                print(f"  Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "config": config.to_dict(),
                "beta": beta, "best_val_loss": best_val}, ckpt_path)
    print(f"VAE β={beta} saved → {ckpt_path}  (best val loss {best_val:.4f})")
    writer.close()
    return model


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--z_dim",      type=int,   default=64)
    # ~8% of 768-d RoBERTa embeddings; balances compression and reconstruction quality, fitting the 5-10% heuristic for text.
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()

    config = AEConfig(
        epochs=args.epochs,
        z_dim=args.z_dim,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    set_seed(config.seed)
    device   = get_device(args.device)
    run_name = f"w4_{int(time.time())}"
    run_dir  = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    print(f"Device  : {device}")
    print(f"z_dim   : {config.z_dim}")
    print(f"Epochs  : {config.epochs}")
    print(f"Run dir : {run_dir}\n")

    loaders, _ = load_data(config, device)

    # ── AE ────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Training AE")
    print("=" * 60)
    train_ae(config, loaders, device, run_dir)

    # ── VAE × 3 ───────────────────────────────────────────────────────────────
    for beta in config.betas:
        print("\n" + "=" * 60)
        print(f"Training VAE  β = {beta}")
        print("=" * 60)
        train_vae(config, loaders, device, run_dir, beta=beta)

    print("\nAll models trained. Run tensorboard with:")
    print(f"  poetry run tensorboard --logdir {run_dir}")


if __name__ == "__main__":
    main()