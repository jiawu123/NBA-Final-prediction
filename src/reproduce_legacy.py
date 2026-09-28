"""Reproduce the flawed historical result for audit only; never a forecast."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '999dc8f'


def main():
    source = subprocess.check_output(
        ['git', 'show', f'{COMMIT}:src/NBA_PREDICTION_CODES.py'], cwd=ROOT, text=True)
    # Load historical function definitions without running its import-time tests/training.
    tree = ast.parse(source)
    functions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)], type_ignores=[])
    namespace = dict(globals())
    exec(compile(functions, '<historical source>', 'exec'), namespace)
    data_dir = str(ROOT / 'data')
    advanced, per100 = namespace['load_and_clean_data'](range(2010, 2025), data_dir)
    dataset, teams = namespace['prepare_final_dataset'](
        advanced, per100, 'NBA_Finals_2010_2023.csv', data_dir)
    X, y = namespace['preprocess_data'](dataset)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        model, features = namespace['train_model'](X, y)
    # The legacy training function prints but does not return its fold scores.
    train, test, train_y, _ = train_test_split(X, y, test_size=.2, random_state=42)
    scores = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42),
                             train[features], train_y, cv=5)
    probabilities = model.predict_proba(dataset.loc[dataset.Year == 2024, features])[:, 1]
    result = {'warning': 'Faithful reproduction of an invalid forecasting evaluation.',
              'source_commit': COMMIT, 'rows': len(dataset), 'feature_count': X.shape[1],
              'excluded_years': sorted(set(range(2010, 2025)) - set(dataset.Year)),
              'unknown_2024_encoded_as': dataset.loc[dataset.Year == 2024, 'Is_Winner'].tolist(),
              'selected_features': features.tolist(), 'cv_scores': scores.tolist(),
              'mean_cv_accuracy': float(scores.mean()),
              'probabilities_2024': dict(zip(teams, probabilities.tolist())),
              'train_rows': len(train), 'unused_test_rows': len(test),
              'train_test_overlapping_years': sorted(set(dataset.loc[train.index, 'Year']) & set(dataset.loc[test.index, 'Year'])),
              '2024_training_rows': int(dataset.loc[train.index, 'Year'].eq(2024).sum())}
    target = ROOT / 'reports' / 'results' / 'legacy_reproduction.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(output.getvalue())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
