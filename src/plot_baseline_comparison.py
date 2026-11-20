"""Plot W1/W2/W4 binary ROC curves using row-aligned saved predictions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--w2-predictions', type=Path, default=ROOT.parent / 'JulianCelisCreator-ML-NLP-Sentimental-Analysis-W2/runs/predictions_test_model_best.csv')
    args = parser.parse_args()
    out = ROOT / 'runs/workshop4/evaluation'
    splits = json.loads((ROOT / 'data/gold/splits.json').read_text())
    ids = splits['test']
    labels = np.load(ROOT / 'data/gold/labels.npy')
    truth = labels[ids] != 2
    names = np.array(['negative', 'neutral', 'positive'])
    curves = {}
    def read(path):
        with path.open() as stream:
            return list(csv.DictReader(stream))
    def align(rows, id_col, label_col, score_col):
        indexed = {int(row[id_col]): row for row in rows}
        if len(indexed) != len(rows) or set(indexed) != set(ids):
            raise ValueError('Prediction row IDs do not match the frozen test.')
        if any(indexed[i][label_col] != names[labels[i]] for i in ids):
            raise ValueError('Prediction labels do not match the frozen test.')
        return np.array([float(indexed[i][score_col]) for i in ids])
    w1 = read(out / 'w1_test_predictions.csv')
    for name in sorted(set(row['model'] for row in w1)):
        curves['W1 ' + name] = align([row for row in w1 if row['model'] == name], 'row_id', 'label', 'anomaly_score')
    curves['W2 RoBERTa'] = 1 - align(read(args.w2_predictions), '', 'true', 'p_positive')
    w4 = read(out / 'test_scores.csv')
    curves['W4 AE MSE'] = align([row for row in w4 if row['model'] == 'ae'], 'row_id', 'label', 'mse')
    curves['W4 VAE beta=0.5 MSE+KL'] = align([row for row in w4 if row['model'] == 'vae_b0.5'], 'row_id', 'label', 'mse_plus_kl')
    fig, ax = plt.subplots(figsize=(9, 7))
    metrics = {}
    for name, score in curves.items():
        if not np.isfinite(score).all():
            raise ValueError('Nonfinite scores.')
        auc = float(roc_auc_score(truth, score))
        metrics[name] = auc
        fpr, tpr, _ = roc_curve(truth, score)
        ax.plot(fpr, tpr, label=f'{name}: {auc:.4f}')
    ax.plot([0,1], [0,1], 'k--', alpha=.4)
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title='W1, W2 and W4: identical binary test task')
    ax.legend(fontsize=8)
    fig.tight_layout()
    figure = ROOT / 'notebooks/figures/evaluation/baseline_comparison_roc.png'
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    (out / 'all_workshops_auc.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
