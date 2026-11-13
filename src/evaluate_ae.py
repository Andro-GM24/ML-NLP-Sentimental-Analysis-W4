"""Reconstruction and anomaly evaluation on real, frozen Workshop 4 splits.

Run: .venv/bin/python src/evaluate_ae.py
Inspect inputs without ML dependencies: python3 src/evaluate_ae.py --check-only
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"ae": "ae_best.pt", **{
    f"vae_b{beta}": f"vae_b{beta}_best.pt" for beta in (0.5, 1.0, 4.0)}}


def inspect_inputs():
    """Check availability before importing optional, heavyweight dependencies."""
    paths = ["data/gold/embeddings_bert.npy", "data/gold/labels.npy",
             "data/gold/splits.json", "checkpoints/scaler.pkl"]
    paths += [f"checkpoints/{name}" for name in MODELS.values()]
    report = {}
    for name in paths:
        path = ROOT / name
        status = "missing"
        if path.is_file():
            with path.open("rb") as stream:
                prefix = stream.read(160)
            status = "git_lfs_pointer" if prefix.startswith(b"version https://git-lfs.github.com/spec/v1") else "available"
        report[name] = status
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--normal-quantile", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if not 0 < args.normal_quantile < 1 or args.batch_size < 1:
        parser.error("Quantile must be between 0 and 1; batch size must be positive.")
    availability = inspect_inputs()
    out = ROOT / "runs/workshop4/evaluation"
    out.mkdir(parents=True, exist_ok=True)
    ready = all(s == "available" for s in availability.values())
    (out / "input_status.json").write_text(json.dumps({
        "ready": ready, "files": availability,
        "note": "Availability check only; arrays and checkpoints are validated during evaluation."
    }, indent=2))
    for path, status in availability.items():
        print(f"{status:>16}  {path}")
    if args.check_only:
        return 0 if ready else 1
    if not ready:
        print("Evaluation pending: missing inputs or Git LFS pointers. No metrics were generated.")
        return 1

    import joblib
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                                 roc_auc_score, roc_curve)
    from config import LABELS, SPLITS_SHA256
    from embedding_data import load_autoencoder

    torch.set_num_threads(4)
    split_path = ROOT / "data/gold/splits.json"
    if hashlib.sha256(split_path.read_bytes()).hexdigest() != SPLITS_SHA256:
        raise ValueError("Frozen split hash differs from config.py.")
    splits = json.loads(split_path.read_text())
    embeddings = np.load(ROOT / "data/gold/embeddings_bert.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(ROOT / "data/gold/labels.npy", allow_pickle=False)
    if embeddings.shape != (42632, 768) or labels.shape != (42632,):
        raise ValueError("Expected embeddings (42632,768) and labels (42632,).")
    if not np.isin(labels, [0, 1, 2]).all() or not np.isfinite(embeddings).all():
        raise ValueError("Invalid labels or nonfinite embeddings.")
    indices = [np.asarray(splits[s]) for s in ("train", "validation", "test")]
    joined = np.concatenate(indices)
    if not np.issubdtype(joined.dtype, np.integer) or not np.array_equal(np.sort(joined), np.arange(len(labels))):
        raise ValueError("Splits must partition the complete corpus without overlap.")
    scaler = joblib.load(ROOT / "checkpoints/scaler.pkl")
    X = {s: scaler.transform(embeddings[splits[s]]).astype(np.float32)
         for s in ("validation", "test")}
    y = {s: labels[splits[s]] for s in X}
    for s in X:
        if not np.isfinite(X[s]).all() or len(np.unique(y[s] != 2)) != 2:
            raise ValueError(f"Invalid transformed data or missing binary class: {s}")

    def evaluate(model, values):
        reconstructions, mse, correlations, kls = [], [], [], []
        with torch.no_grad():
            for start in range(0, len(values), args.batch_size):
                x = torch.from_numpy(values[start:start + args.batch_size])
                result = model(x)
                reconstruction = result[0]
                mse.append(((reconstruction - x) ** 2).mean(1).numpy())
                a = x - x.mean(1, keepdim=True)
                b = reconstruction - reconstruction.mean(1, keepdim=True)
                denominator = (a.square().sum(1) * b.square().sum(1)).sqrt()
                correlation = torch.where(denominator > 0, (a * b).sum(1) / denominator,
                                          torch.full_like(denominator, float("nan")))
                correlations.append(correlation.numpy())
                if len(result) == 3:
                    mu, log_var = result[1:]
                    kls.append((-0.5 * (1 + log_var - mu.square() - log_var.exp()).mean(1)).numpy())
                reconstructions.append(reconstruction.numpy())
        return {"reconstruction": np.concatenate(reconstructions), "mse": np.concatenate(mse),
                "pearson": np.concatenate(correlations),
                "kl": np.concatenate(kls) if kls else np.zeros(len(values))}

    def summary(result, mask):
        mse = result["mse"][mask]
        corr = result["pearson"][mask]
        finite = np.isfinite(corr)
        return {"count": int(len(mse)), "mse_mean": float(mse.mean()),
                "mse_median": float(np.median(mse)), "mse_p95": float(np.quantile(mse, .95)),
                "pearson_mean": float(corr[finite].mean()) if finite.any() else None,
                "pearson_undefined": int((~finite).sum())}

    results, reconstruction, anomaly = {}, {}, {}
    checkpoint_metadata = {}
    figures = ROOT / "notebooks/figures/evaluation"
    figures.mkdir(parents=True, exist_ok=True)
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_dist, axes_dist = plt.subplots(len(MODELS) * 2 - 1, 1, figsize=(9, 22))
    panel_index = 0
    for name, filename in MODELS.items():
        checkpoint = torch.load(ROOT / "checkpoints" / filename, map_location="cpu", weights_only=False)
        checkpoint_metadata[name] = {"filename": filename, "sha256": hashlib.sha256((ROOT / "checkpoints" / filename).read_bytes()).hexdigest(), "config": checkpoint["config"]}
        model = load_autoencoder(ROOT / "checkpoints" / filename)
        results[name] = {s: evaluate(model, X[s]) for s in X}
        reconstruction[name] = {s: {
            "overall": summary(results[name][s], np.ones(len(y[s]), dtype=bool)),
            "by_class": {label: summary(results[name][s], y[s] == i)
                         for i, label in enumerate(LABELS)}
        } for s in X}
        for kind in (["mse"] if name == "ae" else ["mse", "mse_plus_kl"]):
            score = {s: results[name][s]["mse"] +
                     (results[name][s]["kl"] if kind == "mse_plus_kl" else 0) for s in X}
            if not all(np.isfinite(v).all() for v in score.values()):
                raise ValueError(f"Nonfinite score: {name}/{kind}")
            threshold = float(np.quantile(score["validation"][y["validation"] == 2], args.normal_quantile))
            truth = y["test"] != 2
            prediction = score["test"] > threshold
            precision, recall, f1, _ = precision_recall_fscore_support(
                truth, prediction, average="binary", zero_division=0)
            key = f"{name}/{kind}"
            anomaly[key] = {"threshold": threshold, "test_auc_roc": float(roc_auc_score(truth, score["test"])),
                            "precision": float(precision), "recall": float(recall), "f1": float(f1),
                            "confusion_matrix_normal_anomaly": confusion_matrix(truth, prediction, labels=[False, True]).tolist()}
            fpr, tpr, _ = roc_curve(truth, score["test"])
            ax_roc.plot(fpr, tpr, label=f"{key}: {anomaly[key]['test_auc_roc']:.3f}")
            ax = axes_dist[panel_index]
            panel_index += 1
            for value, label in [(False, "normal: positive"), (True, "anomaly: neutral/negative")]:
                ax.hist(score["test"][truth == value], bins=45, density=True, alpha=.5, label=label)
            ax.axvline(threshold, color="black", linestyle="--", label="validation threshold")
            ax.set(title=key, xlabel="Score", ylabel="Density")
            ax.legend(fontsize=8)
        print(name, reconstruction[name]["test"]["overall"])

    best = min((n for n in MODELS if n != "ae"),
               key=lambda n: reconstruction[n]["validation"]["overall"]["mse_mean"])
    chosen = []
    for i in range(3):
        candidates = np.flatnonzero(y["test"] == i)
        ordered = candidates[np.argsort(results["ae"]["test"]["mse"][candidates])]
        chosen.extend(int(ordered[round(q * (len(ordered) - 1))]) for q in (.1, .5, .9))
    fig, axes = plt.subplots(9, 5, figsize=(17, 13), squeeze=False)
    original = X["test"][chosen]
    ae = results["ae"]["test"]["reconstruction"][chosen]
    vae = results[best]["test"]["reconstruction"][chosen]
    bound = float(max(np.abs(a).max() for a in [original, ae, vae]))
    diff_bound = float(max(np.abs(original - ae).max(), np.abs(original - vae).max()))
    titles = ["Original (scaled)", "AE", best, "|Original - AE|", "|Original - VAE|"]
    for row, index in enumerate(chosen):
        arrays = [original[row], ae[row], vae[row], abs(original[row] - ae[row]), abs(original[row] - vae[row])]
        for col, array in enumerate(arrays):
            ax = axes[row, col]
            ax.imshow(array[None, :], aspect="auto", cmap="coolwarm" if col < 3 else "magma",
                      vmin=-bound if col < 3 else 0, vmax=bound if col < 3 else diff_bound)
            ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col])
            if col == 0:
                ax.set_ylabel(f"row {splits['test'][index]}\n{LABELS[int(y['test'][index])]}", fontsize=8)
    fig.supxlabel("Embedding component (0–767); color encodes value, not text or spatial structure", y=.065)
    fig.subplots_adjust(left=.06, right=.94, bottom=.08, top=.94, hspace=.5, wspace=.12)
    fig.colorbar(axes[0, 0].images[0], cax=fig.add_axes([.08, .025, .47, .01]), orientation="horizontal", label="Standardized value (shared scale)")
    fig.colorbar(axes[0, 3].images[0], cax=fig.add_axes([.62, .025, .3, .01]), orientation="horizontal", label="Absolute error (shared scale)")
    fig.savefig(figures / "reconstruction_panels.png", dpi=160)
    ax_roc.plot([0, 1], [0, 1], "k--")
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", title="Anomaly detection on real test reviews")
    ax_roc.legend(fontsize=8)
    fig_roc.tight_layout()
    fig_roc.savefig(figures / "anomaly_roc.png", dpi=160)
    fig_dist.tight_layout()
    fig_dist.savefig(figures / "anomaly_scores.png", dpi=160)
    fig_box, axes = plt.subplots(1, len(MODELS), figsize=(16, 5), squeeze=False)
    for ax, name in zip(axes[0], MODELS):
        ax.boxplot([results[name]["test"]["mse"][y["test"] == i] for i in range(3)])
        ax.set_xticks([1, 2, 3], LABELS, rotation=30)
        ax.set(title=name, ylabel="MSE")
    fig_box.tight_layout()
    fig_box.savefig(figures / "reconstruction_by_class.png", dpi=160)
    fig_cm, axes_cm = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (key, value) in zip(axes_cm.flat, anomaly.items()):
        matrix = np.asarray(value["confusion_matrix_normal_anomaly"])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), count in np.ndenumerate(matrix):
            ax.text(j, i, str(count), ha="center", va="center", color="white" if count > matrix.max()/2 else "black")
        ax.set(title=key, xlabel="Predicted", ylabel="Actual")
        ax.set_xticks([0, 1], ["Normal", "Anomaly"])
        ax.set_yticks([0, 1], ["Normal", "Anomaly"])
    axes_cm.flat[-1].set_visible(False)
    fig_cm.tight_layout()
    fig_cm.savefig(figures / "anomaly_confusion_matrices.png", dpi=160)
    plt.close("all")
    payload = {"protocol": {"normal": "positive", "anomaly": "neutral or negative",
               "threshold": f"quantile {args.normal_quantile} of normal validation scores; strict >",
               "kl": "mean across latent dimensions; score MSE + KL without beta weighting",
               "reconstruction_space": "standardized 768-dimensional embeddings",
               "best_vae": best, "selection": "lowest overall validation reconstruction MSE",
               "training_provenance": "Expected positive-only checkpoints from train_ae.py; not independently verified",
               "baseline_comparison": "see w1_baselines.json, baseline_comparison.json and all_workshops_auc.json (generated by comparison scripts)",
               "panel_test_row_ids": [int(splits['test'][i]) for i in chosen]},
               "checkpoints": checkpoint_metadata,
               "environment": {"torch": torch.__version__, "numpy": np.__version__, "sklearn": __import__("sklearn").__version__, "matplotlib": matplotlib.__version__, "joblib": joblib.__version__},
               "reconstruction": reconstruction, "anomaly": anomaly}
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    with (out / "test_scores.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row_id", "label", "model", "mse", "pearson", "kl", "mse_plus_kl"])
        for name in MODELS:
            r = results[name]["test"]
            for i, row_id in enumerate(splits["test"]):
                writer.writerow([row_id, LABELS[int(y['test'][i])], name, r['mse'][i],
                                 r['pearson'][i] if np.isfinite(r['pearson'][i]) else "",
                                 r['kl'][i], r['mse'][i] + r['kl'][i]])
    print(f"Saved metrics to {out} and figures to {figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
