"""Reproducible audit and retrospective diagnostics of the bundled NBA snapshots.

These CSVs include Finals outcomes in their historical features. This module
intentionally does not expose a live forecasting command.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
FEATURES = ['NRtg', 'ORB%', 'OTOV%']
WARNING = ('Retrospective diagnostics only: historical playoff totals include the '
           'Finals. Temporal validation cannot remove this source-level leakage. '
           'No deployable forecasting accuracy or calibrated 2024 probability is claimed.')


def read_stats(path):
    """Find the real header; discard League Average, never silently skip a season."""
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.reader(handle))
    header = next((i for i, row in enumerate(rows) if row[:2] == ['Rk', 'Team']), None)
    if header is None:
        raise ValueError(f'{Path(path).name}: expected Rk,Team header')
    frame = pd.read_csv(path, skiprows=header)
    frame = frame.loc[:, ~frame.columns.str.startswith('Unnamed')].dropna(axis=1, how='all')
    frame['Team'] = frame['Team'].str.strip().str.rstrip('*')
    frame = frame.loc[frame['Team'].notna() & frame['Team'].ne('League Average')].copy()
    if frame['Team'].duplicated().any():
        raise ValueError(f'{Path(path).name}: duplicate team keys')
    for column in frame.columns.drop('Team'):
        frame[column] = pd.to_numeric(frame[column], errors='raise')
    return frame, header


def load_dataset(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    finals_path = data_dir / 'NBA_Finals_2010_2023.csv'
    finals = pd.read_csv(finals_path)
    if finals['Year'].duplicated().any():
        raise ValueError('Finals labels must have one row per year')
    if finals[['Year', 'East_team', 'West_team']].isna().any().any():
        raise ValueError('Every matchup must have a year and two named finalists')
    if finals['East_team'].eq(finals['West_team']).any():
        raise ValueError('A matchup must contain two different teams')
    invalid = finals.Win_team.notna() & ~(
        finals.Win_team.eq(finals.East_team) | finals.Win_team.eq(finals.West_team))
    if invalid.any():
        raise ValueError('Winner must be one of the two finalists or missing')
    all_stats, coverage, manifest = [], [], []
    for year in sorted(finals.Year):
        tables = []
        info = {'Year': int(year)}
        for kind in ['advanced', 'per100']:
            path = data_dir / f'{kind}_stats_{year}.csv'
            frame, skipped = read_stats(path)
            info[f'{kind}_team_rows'] = len(frame)
            info[f'{kind}_header_row'] = skipped + 1
            manifest.append({'path': f'data/{path.name}',
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            tables.append(frame)
        combined = tables[0].merge(tables[1], on='Team', how='outer',
                                   suffixes=('_adv', '_per100'), validate='one_to_one', indicator=True)
        if not combined['_merge'].eq('both').all():
            raise ValueError(f'{year}: advanced and per100 team keys differ')
        combined = combined.drop(columns='_merge').assign(Year=int(year))
        all_stats.append(combined)
        coverage.append(info)
    stats = pd.concat(all_stats, ignore_index=True)
    long = finals.melt(id_vars=['Year', 'Win_team'], value_vars=['East_team', 'West_team'],
                       var_name='Conference', value_name='Team')
    teams = long.merge(stats, on=['Year', 'Team'], how='left', validate='one_to_one', indicator=True)
    if not teams['_merge'].eq('both').all():
        raise ValueError('Missing statistics for a finalist')
    teams = teams.drop(columns='_merge').sort_values(['Year', 'Conference']).reset_index(drop=True)
    teams['Is_Winner'] = pd.Series(pd.NA, index=teams.index, dtype='Int64')
    known = teams.Win_team.notna()
    teams.loc[known, 'Is_Winner'] = teams.loc[known, 'Team'].eq(teams.loc[known, 'Win_team']).astype(int)
    if not teams.loc[known].groupby('Year').Is_Winner.sum().eq(1).all():
        raise ValueError('Each labeled Finals must have exactly one winner')
    manifest.append({'path': f'data/{finals_path.name}',
                     'sha256': hashlib.sha256(finals_path.read_bytes()).hexdigest()})
    return teams, pd.DataFrame(coverage), manifest


def make_matchups(teams):
    """One independent observation per Finals: East minus West features."""
    rows = []
    for year, group in teams.groupby('Year', sort=True):
        if len(group) != 2 or set(group.Conference) != {'East_team', 'West_team'}:
            raise ValueError(f'{year}: expected exactly one East and one West finalist')
        by_side = group.set_index('Conference')
        east, west = by_side.loc['East_team'], by_side.loc['West_team']
        row = {'Year': int(year), 'East_team': east.Team, 'West_team': west.Team,
               'East_won': east.Is_Winner}
        row.update({feature: east[feature] - west[feature] for feature in FEATURES})
        rows.append(row)
    return pd.DataFrame(rows)


def make_model(name):
    if name == 'Logistic regression':
        estimator = LogisticRegression(C=0.25, fit_intercept=False, random_state=42, max_iter=2000)
    elif name == 'Random forest':
        estimator = RandomForestClassifier(n_estimators=200, max_depth=2,
                                           min_samples_leaf=2, random_state=42, n_jobs=1)
    else:
        raise ValueError(f'Unknown model: {name}')
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), estimator)


def fit_paired_model(train, name):
    X = train[FEATURES].astype(float)
    y = train.East_won.astype(int)
    # Mirroring is performed only AFTER the year split; it does not double sample size.
    return make_model(name).fit(pd.concat([X, -X], ignore_index=True),
                                np.concatenate([y, 1 - y]))


def paired_probability(model, X):
    """Swapping team order must swap the two complementary model probabilities."""
    return (model.predict_proba(X)[:, 1] + 1 - model.predict_proba(-X)[:, 1]) / 2


def walk_forward(matchups, min_train_years=5):
    known = matchups.loc[matchups.East_won.notna()].sort_values('Year')
    if min_train_years < 2 or len(known) <= min_train_years:
        raise ValueError('Need at least two training years and one held-out year')
    rows = []
    for year in known.Year.iloc[min_train_years:]:
        train, test = known.loc[known.Year < year], known.loc[known.Year == year]
        probabilities = {
            '50/50 reference': 0.5,
            # Deterministic selection rule, not a calibrated probability model.
            'Higher net rating': float(np.sign(test.NRtg.iloc[0]) / 2 + 0.5),
        }
        for name in ['Logistic regression', 'Random forest']:
            model = fit_paired_model(train, name)
            probabilities[name] = float(paired_probability(model, test[FEATURES].astype(float))[0])
        for name, probability in probabilities.items():
            target = int(test.East_won.iloc[0])
            # An exact tie has expected accuracy 0.5, rather than favoring East.
            correct = 0.5 if probability == 0.5 else float((probability > 0.5) == target)
            rows.append({'Year': int(year), 'model': name, 'East_won': target,
                         'p_east': probability, 'correct': correct, 'train_series': len(train),
                         'train_last_year': int(train.Year.max())})
    return pd.DataFrame(rows)


def summarize(backtest):
    rows = []
    for name, group in backtest.groupby('model', sort=False):
        probabilistic = name != 'Higher net rating'
        rows.append({'model': name, 'series': len(group), 'correct_or_expected': group.correct.sum(),
                     'accuracy': group.correct.mean(),
                     'brier': brier_score_loss(group.East_won, group.p_east) if probabilistic else None,
                     'log_loss': log_loss(group.East_won, group.p_east, labels=[0, 1]) if probabilistic else None})
    return pd.DataFrame(rows)


def run_analysis(data_dir=DATA_DIR, output_dir=ROOT / 'reports' / 'results'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    teams, coverage, manifest = load_dataset(data_dir)
    matchups = make_matchups(teams)
    backtest = walk_forward(matchups)
    metrics = summarize(backtest)
    known = teams.loc[teams.Is_Winner.notna()]
    historical_pairs = matchups.loc[matchups.East_won.notna()]
    net_rating_correct = ((historical_pairs.NRtg > 0) == historical_pairs.East_won.astype(bool)).sum()
    # A deliberately invalid rule demonstrates the outcome already present in the source.
    leakage_correct = known.W.eq(16).astype(int).eq(known.Is_Winner.astype(int)).sum()
    duplicates = []
    canonical_paths = {entry['path'] for entry in manifest}
    for extra in sorted(Path(data_dir).glob('*.csv')):
        if f'data/{extra.name}' not in canonical_paths:
            digest = hashlib.sha256(extra.read_bytes()).hexdigest()
            duplicates.append({'path': f'data/{extra.name}', 'used': False,
                               'identical_to': [m['path'] for m in manifest if m['sha256'] == digest]})
    audit = {
        'warning': WARNING, 'data_scope': '2010-2024 repository snapshots; labels through 2023',
        'canonical_files': len(manifest), 'team_seasons': int(coverage.advanced_team_rows.sum()),
        'finalist_rows': len(teams), 'labeled_team_rows': len(known),
        'labeled_series': len(historical_pairs), 'unlabeled_years': matchups.loc[matchups.East_won.isna(), 'Year'].tolist(),
        'missing_selected_feature_cells': int(teams[FEATURES].isna().sum().sum()),
        'w_equals_16_correct_team_rows': int(leakage_correct),
        'higher_nrtg_correct_series': int(net_rating_correct),
        'backtest_years': sorted(backtest.Year.unique().tolist()),
        'features': FEATURES, 'extra_files': duplicates, 'source_files': manifest,
    }
    for name, frame in [('coverage', coverage), ('finalists', teams), ('matchups', matchups),
                        ('backtest', backtest), ('metrics', metrics)]:
        frame.to_csv(output_dir / f'{name}.csv', index=False)
    (output_dir / 'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    return audit, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports' / 'results')
    args = parser.parse_args()
    audit, metrics = run_analysis(args.data_dir, args.output_dir)
    print(WARNING)
    print(f"Labeled series: {audit['labeled_series']}; unlabeled years: {audit['unlabeled_years']}")
    print(metrics.to_string(index=False))
    print(f'Results: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
