"""Latent-space analysis: AE and VAE (unsupervised) against Workshop #2's RoBERTa (supervised).

Every test review is encoded three ways:

* AE   — the bottleneck code z;
* VAE  — the mean µ (not a sample), as the brief asks;
* W2   — the fine-tuned RoBERTa CLS vector itself (standardised), i.e. the frozen
         supervised encoder of Workshop #2.

For each representation it reports the Silhouette Score of the sentiment labels, computed
on the full vectors (not on the 2-D projection), and draws t-SNE and UMAP projections.

    python src/analyze_latent.py
    python src/analyze_latent.py --ae checkpoints/ae_best.pt --vae checkpoints/vae_best.pt
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples

from config import LABELS, PROJECT_ROOT, RANDOM_STATE
from embedding_data import (
    CLASS_COLORS,
    FIGURES_DIR,
    RUNS_DIR,
    encode,
    load_autoencoder,
    load_scaler,
    load_splits,
)


def silhouette(X: np.ndarray, y: np.ndarray) -> dict:
    """Overall and per-class mean silhouette (Euclidean, on the full vectors)."""
    samples = silhouette_samples(X, y)
    report = {"overall": float(samples.mean())}
    for index, name in enumerate(LABELS):
        report[name] = float(samples[y == index].mean())
    return report


def project(X: np.ndarray) -> dict[str, np.ndarray]:
    """2-D t-SNE and UMAP of the same vectors, both seeded."""
    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=RANDOM_STATE)
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=RANDOM_STATE)
    return {"t-SNE": tsne.fit_transform(X), "UMAP": reducer.fit_transform(X)}


def scatter(ax, points: np.ndarray, y: np.ndarray, title: str) -> None:
    # Positives are 80 % of the data: draw them first so the minority classes stay visible.
    for index in (2, 1, 0):
        mask = y == index
        ax.scatter(points[mask, 0], points[mask, 1], s=3, alpha=0.45, linewidths=0,
                   color=CLASS_COLORS[LABELS[index]], label=f"{LABELS[index]} ({mask.sum():,})")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ae", default="checkpoints/ae_best.pt")
    parser.add_argument("--vae", default="checkpoints/vae_best.pt")
    parser.add_argument("--tag", default="", help="suffix for the output files")
    args = parser.parse_args()

    data = load_splits()
    scaler = load_scaler(data["train"]["X"])
    X_test = scaler.transform(data["test"]["X"]).astype(np.float32)
    y_test = data["test"]["y"]

    representations = {
        "AE (unsupervised)": encode(load_autoencoder(PROJECT_ROOT / args.ae), X_test),
        "VAE µ (unsupervised)": encode(load_autoencoder(PROJECT_ROOT / args.vae), X_test),
        "Workshop 2 RoBERTa (supervised)": X_test,
    }

    results = {}
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))
    for column, (name, X) in enumerate(representations.items()):
        score = silhouette(X, y_test)
        results[name] = {"dimensions": int(X.shape[1]), "silhouette": score}
        print(f"{name:<34} dim {X.shape[1]:>4}  silhouette {score['overall']:+.4f}  "
              + "  ".join(f"{k} {score[k]:+.3f}" for k in LABELS))
        for row, (method, points) in enumerate(project(X).items()):
            scatter(axes[row, column], points, y_test,
                    f"{method} · {name}\nsilhouette {score['overall']:+.3f} ({X.shape[1]}-d)")
    axes[0, 0].legend(loc="lower left", markerscale=4, fontsize=9, frameon=False)
    fig.suptitle("Test reviews in each latent space, coloured by sentiment (labels never seen by AE/VAE)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    figure = FIGURES_DIR / f"w4_latent_space{suffix}.png"
    fig.savefig(figure, dpi=200, bbox_inches="tight")
    out = RUNS_DIR / f"latent_analysis{suffix}.json"
    out.write_text(json.dumps({"checkpoints": {"ae": args.ae, "vae": args.vae},
                               "test_reviews": int(len(y_test)), "results": results}, indent=2),
                   encoding="utf-8")
    print(f"Saved {figure} and {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
