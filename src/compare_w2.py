"""Compare saved W2 probabilities with W4 on identical test row IDs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=ROOT.parent / 'JulianCelisCreator-ML-NLP-Sentimental-Analysis-W2/runs/predictions_test_model_best.csv')
    args = parser.parse_args()
    with args.predictions.open() as stream:
        rows = list(csv.DictReader(stream))
    splits = json.loads((ROOT / 'data/gold/splits.json').read_text())
    labels = np.load(ROOT / 'data/gold/labels.npy', allow_pickle=False)
    names = ['negative', 'neutral', 'positive']
    indexed = {int(row['']): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(splits['test']):
        raise ValueError('W2 predictions must contain each W4 test row exactly once.')
    if not all(indexed[i]['true'] == names[int(labels[i])] for i in splits['test']):
        raise ValueError('W2 and W4 ground-truth labels differ.')
    probabilities = np.array([[float(indexed[i]['p_' + name]) for name in names] for i in splits['test']])
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)) or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid W2 probabilities.')
    result = {'model': 'W2 RoBERTa (saved test probabilities)',
              'source': str(args.predictions.resolve()),
              'source_sha256': hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
              'test_rows': len(rows), 'alignment': 'row IDs and ground-truth labels verified against W4 test',
              'score': '1 - P(positive)',
              'test_auc_roc': float(roc_auc_score(labels[splits['test']] != 2, 1 - probabilities[:, 2])),
              'w1_status': 'Pending original data and retraining; no saved test probabilities found.'}
    out = ROOT / 'runs/workshop4/evaluation/baseline_comparison.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
