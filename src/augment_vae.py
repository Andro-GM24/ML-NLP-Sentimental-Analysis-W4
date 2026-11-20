"""Data augmentation with the VAE, evaluated on Workshop #2's classifier head.

The VAE lives in RoBERTa's 768-d embedding space, so it generates embedding vectors,
not text. RoBERTa itself cannot be retrained on them (it reads tokens); what can be
retrained is Workshop #2's classification head (dropout + Linear 768 → 3) on top of the
frozen encoder. That is the experiment:

1. Train a VAE on ALL training classes (the anomaly-detection VAEs see positives only and
   could not generate negative or neutral reviews) → checkpoints/vae_all_classes.pt.
2. Synthetic vector = decode(µ + s · σ ⊙ ε) of a real training review; it keeps that
   review's label (perturbing µ, rather than sampling the prior, preserves the class).
3. Retrain the head on real train vectors + 0 / 25 / 50 / 100 % synthetic ones, three
   seeds each; select the epoch on validation macro-F1; report test macro-F1.

Validation and test stay 100 % real. Sanity check: Workshop #2's own head, applied to the
extracted test vectors, must reproduce its test macro-F1 (0.713).

    python src/augment_vae.py
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn

from autoencoder import VAE, vae_loss
from config import LABELS, PROJECT_ROOT, RANDOM_STATE, set_seed
from embedding_data import FIGURES_DIR, RUNS_DIR, W2_CHECKPOINT, load_scaler, load_splits

VAE_PATH = PROJECT_ROOT / "checkpoints" / "vae_all_classes.pt"
FRACTIONS = (0.0, 0.25, 0.50, 1.00)
SEEDS = (42, 43, 44)


# ---------------------------------------------------------------------- VAE on all classes

def train_vae_all(X_train, X_val, z_dim=64, beta=1.0, epochs=40, anneal=10, lr=1e-3,
                  batch_size=256, seed=RANDOM_STATE) -> VAE:
    """Same architecture and loss as src/autoencoder.py, trained on every class.

    Model selection starts only once KL annealing is over, so the saved model is one
    trained under the full KL term (an earlier, barely regularised epoch would always
    have the lowest total loss).
    """
    set_seed(seed)
    model = VAE(z_dim=z_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    train = torch.from_numpy(X_train)
    val = torch.from_numpy(X_val)
    best, best_state = np.inf, None
    for epoch in range(1, epochs + 1):
        kl_weight = min(1.0, epoch / anneal)
        model.train()
        for index in torch.randperm(len(train)).split(batch_size):
            x = train[index]
            x_hat, mu, log_var = model(x)
            loss, _, _ = vae_loss(x, x_hat, mu, log_var, beta=beta, kl_weight=kl_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            x_hat, mu, log_var = model(val)
            total, recon, kl = vae_loss(val, x_hat, mu, log_var, beta=beta, kl_weight=1.0)
        print(f"  VAE epoch {epoch:>2}  val recon {recon:.4f}  kl {kl:.4f}  total {total:.4f}"
              f"{'  (annealing)' if epoch < anneal else ''}")
        if epoch >= anneal and total.item() < best:
            best = total.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "best_val_loss": best,
                "config": {"z_dim": z_dim, "dropout": 0.2, "input_dropout": 0.2, "beta": beta,
                           "epochs": epochs, "kl_anneal_epochs": anneal, "classes": "all"}},
               VAE_PATH)
    return model.eval()


@torch.no_grad()
def synthesise(model: VAE, X_scaled: np.ndarray, count: int, noise: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Decode µ + noise·σ·ε for `count` training rows drawn without replacement (with, past 100 %)."""
    index = rng.choice(len(X_scaled), size=count, replace=count > len(X_scaled))
    x = torch.from_numpy(X_scaled[index])
    mu, log_var = model.encoder(x)
    z = mu + noise * torch.exp(0.5 * log_var) * torch.from_numpy(
        rng.standard_normal(mu.shape).astype(np.float32))
    return model.decoder(z).numpy(), index


# ---------------------------------------------------------------------- Workshop #2 head

def w2_head() -> nn.Module:
    """Workshop #2's classifier: dropout 0.1 + Linear(768, 3) over the CLS vector."""
    return nn.Sequential(nn.Dropout(0.1), nn.Linear(768, len(LABELS)))


def scores(model, X, y) -> dict:
    with torch.no_grad():
        probabilities = torch.softmax(model.eval()(torch.from_numpy(X)), dim=-1).numpy()
    predictions = probabilities.argmax(1)
    return {"macro_f1": float(f1_score(y, predictions, average="macro")),
            "accuracy": float(accuracy_score(y, predictions)),
            "auc_roc": float(roc_auc_score(y, probabilities, multi_class="ovr", average="macro"))}


def train_head(X, y, X_val, y_val, class_weight, seed, epochs=20, lr=1e-3, batch_size=256):
    """Weighted cross-entropy (weights from the real train labels), best epoch on validation."""
    set_seed(seed)
    model = w2_head()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    X_t, y_t = torch.from_numpy(X), torch.from_numpy(y)
    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for index in torch.randperm(len(X_t)).split(batch_size):
            loss = criterion(model(X_t[index]), y_t[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        f1 = scores(model, X_val, y_val)["macro_f1"]
        if f1 > best:
            best, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------- experiment

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retrain-vae", action="store_true", help="ignore a saved vae_all_classes.pt")
    parser.add_argument("--noise", type=float, default=1.0, help="s in µ + s·σ·ε")
    args = parser.parse_args()
    torch.set_num_threads(4)

    data = load_splits()
    scaler = load_scaler(data["train"]["X"])
    scaled = {name: scaler.transform(split["X"]).astype(np.float32) for name, split in data.items()}
    raw = {name: split["X"].astype(np.float32) for name, split in data.items()}
    y = {name: split["y"] for name, split in data.items()}

    # Sanity check: Workshop #2's trained head on the extracted vectors.
    reference = w2_head()
    state = torch.load(W2_CHECKPOINT, map_location="cpu", weights_only=False)["state_dict"]
    reference[1].load_state_dict({"weight": state["classifier.weight"], "bias": state["classifier.bias"]})
    sanity = scores(reference, raw["test"], y["test"])
    print(f"Workshop #2 head on extracted test vectors: macro-F1 {sanity['macro_f1']:.4f} (expected 0.7130)")

    if VAE_PATH.exists() and not args.retrain_vae:
        checkpoint = torch.load(VAE_PATH, map_location="cpu", weights_only=False)
        vae = VAE(z_dim=checkpoint["config"]["z_dim"])
        vae.load_state_dict(checkpoint["state_dict"])
        vae.eval()
        print(f"Loaded {VAE_PATH}")
    else:
        print("Training the VAE on all classes…")
        vae = train_vae_all(scaled["train"], scaled["validation"])

    # How faithful are synthetic vectors? Workshop #2's head should still read their label.
    rng = np.random.default_rng(RANDOM_STATE)
    probe, probe_index = synthesise(vae, scaled["train"], 5000, args.noise, rng)
    probe_raw = scaler.inverse_transform(probe).astype(np.float32)
    fidelity = scores(reference, probe_raw, y["train"][probe_index])
    real_fidelity = scores(reference, raw["train"][probe_index], y["train"][probe_index])
    print(f"Label fidelity (W2 head macro-F1): synthetic {fidelity['macro_f1']:.4f} · "
          f"same reviews, real {real_fidelity['macro_f1']:.4f}")

    class_weight = torch.tensor(compute_class_weight(
        "balanced", classes=np.arange(len(LABELS)), y=y["train"]), dtype=torch.float32)
    runs = []
    for fraction in FRACTIONS:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            count = int(round(fraction * len(raw["train"])))
            X, labels = raw["train"], y["train"]
            if count:
                synthetic, index = synthesise(vae, scaled["train"], count, args.noise, rng)
                X = np.concatenate([X, scaler.inverse_transform(synthetic).astype(np.float32)])
                labels = np.concatenate([labels, y["train"][index]])
            model = train_head(X, labels, raw["validation"], y["validation"], class_weight, seed)
            result = {"fraction": fraction, "seed": seed, "train_rows": int(len(X)),
                      "validation": scores(model, raw["validation"], y["validation"]),
                      "test": scores(model, raw["test"], y["test"])}
            runs.append(result)
            print(f"  +{fraction:>4.0%} seed {seed}  val macro-F1 {result['validation']['macro_f1']:.4f}"
                  f"  test macro-F1 {result['test']['macro_f1']:.4f}")

    summary = {}
    for fraction in FRACTIONS:
        values = [r["test"]["macro_f1"] for r in runs if r["fraction"] == fraction]
        summary[f"{fraction:.2f}"] = {"test_macro_f1_mean": float(np.mean(values)),
                                       "test_macro_f1_std": float(np.std(values))}
    print("\nTest macro-F1 (mean ± std over 3 seeds)")
    for key, value in summary.items():
        print(f"  +{float(key):>4.0%}  {value['test_macro_f1_mean']:.4f} ± {value['test_macro_f1_std']:.4f}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "augmentation.json").write_text(json.dumps({
        "noise": args.noise, "w2_head_on_extracted_test": sanity,
        "label_fidelity": {"synthetic": fidelity, "real_same_reviews": real_fidelity},
        "summary": summary, "runs": runs}, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = [100 * f for f in FRACTIONS]
    means = [summary[f"{f:.2f}"]["test_macro_f1_mean"] for f in FRACTIONS]
    ax.plot(xs, means, color="#2a78d6", marker="o", linewidth=2, label="mean of 3 seeds")
    for run in runs:
        ax.scatter(100 * run["fraction"], run["test"]["macro_f1"], color="#2a78d6", alpha=0.35, s=18)
    ax.axhline(means[0], color="#52514e", linestyle="--", linewidth=1)
    ax.annotate("no augmentation", (xs[-1], means[0]), xytext=(0, 4), textcoords="offset points",
                ha="right", fontsize=9, color="#52514e")
    ax.set(xlabel="synthetic vectors added (% of the real training set)", ylabel="test macro-F1",
           title="VAE augmentation of Workshop #2's classifier head", xticks=xs)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "w4_augmentation.png", dpi=200, bbox_inches="tight")
    print(f"Saved {RUNS_DIR / 'augmentation.json'} and {FIGURES_DIR / 'w4_augmentation.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
