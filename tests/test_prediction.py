import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location('prediction', Path(__file__).parents[1] / 'src' / 'NBA_PREDICTION_CODES.py')
prediction = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prediction)


@pytest.fixture(scope='module')
def dataset():
    return prediction.load_dataset()


def test_recovers_2014_and_preserves_unlabeled_2024(dataset):
    teams, coverage, manifest = dataset
    assert len(teams) == 30
    assert len(teams.loc[teams.Year == 2014]) == 2
    assert coverage.set_index('Year').loc[2014, 'advanced_header_row'] == 2
    assert teams.loc[teams.Year == 2024, 'Is_Winner'].isna().all()
    known = teams.loc[teams.Is_Winner.notna()]
    assert len(known) == 28
    assert known.groupby('Year').Is_Winner.sum().eq(1).all()
    assert len(manifest) == 31


def test_duplicate_team_keys_fail(tmp_path):
    path = tmp_path / 'duplicates.csv'
    path.write_text('Rk,Team,W\n1,A,16\n2,A,12\n')
    with pytest.raises(ValueError, match='duplicate team'):
        prediction.read_stats(path)


def test_missing_file_fails(tmp_path):
    labels = (prediction.DATA_DIR / 'NBA_Finals_2010_2023.csv').read_text()
    (tmp_path / 'NBA_Finals_2010_2023.csv').write_text(labels)
    with pytest.raises(FileNotFoundError):
        prediction.load_dataset(tmp_path)


def test_invalid_winner_fails(tmp_path):
    (tmp_path / 'NBA_Finals_2010_2023.csv').write_text(
        'Year,East_team,West_team,Win_team\n2023,A,B,C\n')
    with pytest.raises(ValueError, match='Winner must'):
        prediction.load_dataset(tmp_path)


def test_chronology_and_unknown_target_exclusion(dataset):
    pairs = prediction.make_matchups(dataset[0])
    backtest = prediction.walk_forward(pairs)
    assert sorted(backtest.Year.unique()) == list(range(2015, 2024))
    assert (backtest.train_last_year < backtest.Year).all()
    assert backtest.groupby('model').size().eq(9).all()
    changed = pairs.copy()
    changed.loc[changed.Year == 2024, prediction.FEATURES] = 1e9
    pd.testing.assert_frame_equal(backtest, prediction.walk_forward(changed))


@pytest.mark.parametrize('name', ['Logistic regression', 'Random forest'])
def test_swap_symmetry_and_training_only_preprocessing(dataset, name):
    pairs = prediction.make_matchups(dataset[0])
    train = pairs.loc[pairs.Year < 2015]
    model = prediction.fit_paired_model(train, name)
    test = pairs.loc[pairs.Year == 2015, prediction.FEATURES].astype(float)
    np.testing.assert_allclose(prediction.paired_probability(model, test)
                               + prediction.paired_probability(model, -test), 1)
    scaler = model.named_steps['standardscaler']
    assert scaler.n_samples_seen_ == 2 * len(train)
    expected = pd.concat([train[prediction.FEATURES], -train[prediction.FEATURES]])
    np.testing.assert_allclose(scaler.var_, expected.var(ddof=0))


def test_future_features_cannot_change_earlier_folds(dataset):
    pairs = prediction.make_matchups(dataset[0])
    original = prediction.walk_forward(pairs)
    pairs.loc[pairs.Year == 2023, prediction.FEATURES] = -1e6
    changed = prediction.walk_forward(pairs)
    pd.testing.assert_frame_equal(original.loc[original.Year < 2023], changed.loc[changed.Year < 2023])


def test_missing_finalist_is_rejected(dataset):
    teams = dataset[0].iloc[1:].copy()
    with pytest.raises(ValueError, match='exactly one East'):
        prediction.make_matchups(teams)
