"""Rebuild original W1 baselines and compare binary AUC on the frozen W4 test.

Reads W1 source without modifying that repository. Writes results to W4 only.
"""
import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--w1-source', type=Path, default=ROOT.parent / 'ML-NLP-Sentimental-Analysis/ML_NPL/src')
    args = parser.parse_args()
    # Preserve original relative imports without adding W1's generic 'src' to sys.path.
    source = args.w1_source.resolve()
    spec = importlib.util.spec_from_file_location('w1_original', source / '__init__.py', submodule_search_locations=[str(source)])
    package = importlib.util.module_from_spec(spec)
    sys.modules['w1_original'] = package
    spec.loader.exec_module(package)
    preprocessing = importlib.import_module('w1_original.preprocessing')
    features = importlib.import_module('w1_original.features')
    bronze = ROOT / 'data/bronze/DisneylandReviews.csv'
    digest = hashlib.sha256(bronze.read_bytes()).hexdigest()
    if digest != '3c5eb28455a5156bbe337b3768fdd422a3311c1a91cb33214682bf80419e885c':
        raise ValueError('Original corpus checksum mismatch.')
    clean = preprocessing.clean_dataset(pd.read_csv(bronze, encoding='latin-1'))
    labels = np.load(ROOT / 'data/gold/labels.npy', allow_pickle=False)
    names = ['negative', 'neutral', 'positive']
    if len(clean) != len(labels) or not np.array_equal(clean['sentiment'].astype(str), np.array(names)[labels]):
        raise ValueError('Recovered corpus does not align with W4 labels.')
    splits = json.loads((ROOT / 'data/gold/splits.json').read_text())
    regenerated = preprocessing.split_dataset(clean)
    if any(regenerated[i].index.tolist() != splits[name] for i, name in enumerate(['train', 'validation', 'test'])):
        raise ValueError('Reconstructed W1 splits do not match W4 exactly.')
    # Save recovered silver in W4 for interpretation and reuse.
    silver = clean.copy()
    silver['period'] = silver['period'].dt.to_timestamp()
    (ROOT / 'data/silver').mkdir(parents=True, exist_ok=True)
    silver.to_parquet(ROOT / 'data/silver/reviews.parquet', index=False)
    frames = [clean.loc[splits[name]] for name in ['train', 'validation', 'test']]
    ys = [frame['sentiment'] for frame in frames]
    print('Original checksum, all labels and exact train/validation/test row order verified.', flush=True)
    train, val, test, _, _, _, _, _, weights = features.extract_features(*frames, *ys)
    train, val, test = [x.tocsr() for x in (train, val, test)]
    models = {
        'majority': DummyClassifier(strategy='most_frequent'),
        'logistic_regression': LogisticRegression(max_iter=1000, class_weight=weights, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=400, max_depth=50, criterion='gini', class_weight='balanced_subsample', n_jobs=4, random_state=42),
        'multinomial_nb': MultinomialNB(),
    }
    out = ROOT / 'runs/workshop4/evaluation'
    out.mkdir(parents=True, exist_ok=True)
    report = {'protocol': {'corpus_sha256': digest, 'normal': 'positive', 'anomaly': 'neutral or negative',
              'alignment': 'Full label vector and exact reconstructed split row order verified',
              'model_selection': 'Highest validation sentiment macro-F1, matching W1 ranking criterion',
              'features': 'Original W1 extract_features; learned transforms fitted on train only',
              'random_forest': 'Original hyperparameters; n_jobs=4 limits CPU parallelism',
              'source_hashes': {name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ['preprocessing.py', 'features.py', 'eda.py', 'baselines.py']},
              'versions': {p: __import__('importlib.metadata', fromlist=['version']).version(p) for p in ['numpy','pandas','scipy','scikit-learn']}}, 'models': {}}
    with (out / 'w1_test_predictions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['row_id', 'label', 'model', 'p_positive', 'anomaly_score'])
        for name, model in models.items():
            a, b, c = train, val, test
            if name == 'multinomial_nb':
                nt = train.shape[1] - len(features.NUMERIC_FEATURES)
                text_scaler, numeric_scaler = MaxAbsScaler(), MinMaxScaler()
                a = hstack([text_scaler.fit_transform(train[:, :nt]), numeric_scaler.fit_transform(train[:, nt:].toarray())]).tocsr()
                b, c = [hstack([text_scaler.transform(x[:, :nt]), numeric_scaler.transform(x[:, nt:].toarray())]).tocsr() for x in (val, test)]
            print('Training', name, flush=True)
            with threadpool_limits(limits=4):
                model.fit(a, ys[0])
                val_prediction = model.predict(b)
                probability = model.predict_proba(c)[:, list(model.classes_).index('positive')]
            result = {'validation_sentiment_macro_f1': float(f1_score(ys[1], val_prediction, average='macro')),
                      'test_auc_roc': float(roc_auc_score(ys[2].astype(str) != 'positive', 1-probability))}
            report['models'][name] = result
            for row_id, label, p in zip(splits['test'], ys[2], probability):
                writer.writerow([row_id, label, name, p, 1-p])
            print(name, result, flush=True)
    report['selected_by_validation'] = max(report['models'], key=lambda n: report['models'][n]['validation_sentiment_macro_f1'])
    (out / 'w1_baselines.json').write_text(json.dumps(report, indent=2))
    print('Selected by validation:', report['selected_by_validation'], flush=True)


if __name__ == '__main__':
    main()
