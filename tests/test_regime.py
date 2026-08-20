import numpy as np
import pandas as pd
import pytest

import regime
from quant_engine import REGIME_WEIGHTS


@pytest.fixture
def two_regime_close():
    """A trending stretch followed by a choppy one, then trending again."""
    rng = np.random.default_rng(0)
    idx = pd.date_range('2018-01-01', periods=900, freq='B')
    parts = [
        np.cumsum(rng.normal(0.0012, 0.008, 300)),          # trend up
        np.cumsum(rng.normal(0.0, 0.02, 300)),              # choppy
        np.cumsum(rng.normal(0.0012, 0.008, 300)),          # trend up
    ]
    path = np.concatenate([p + (parts[i - 1][-1] if i else 0.0)
                           for i, p in enumerate(parts)])
    return pd.Series(100 * np.exp(path), index=idx)


def test_viterbi_prefers_staying_when_the_penalty_is_high():
    cost = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    free, _ = regime._viterbi(cost, penalty=0.0)
    sticky, _ = regime._viterbi(cost, penalty=100.0)
    assert len(set(free.tolist())) == 2       # switches freely
    assert len(set(sticky.tolist())) == 1     # never switches


def test_viterbi_objective_is_the_penalised_cost():
    cost = np.array([[0.0, 5.0], [5.0, 0.0]])
    labels, obj = regime._viterbi(cost, penalty=1.0)
    # cheapest path is stay-in-0 (0 + 5) or switch (0 + 0 + 1); switch wins
    assert obj == pytest.approx(1.0)
    assert labels.tolist() == [0, 1]


def test_build_features_is_causal(two_regime_close):
    full = regime.build_features(two_regime_close)
    cut = 600
    partial = regime.build_features(two_regime_close.iloc[:cut])
    common = full.index[:cut]
    a = full.loc[common].iloc[-50:]
    b = partial.loc[common].iloc[-50:]
    # values computed from a truncated series must match the full-series
    # values at the same bars
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-8)


def test_features_have_expected_columns(two_regime_close):
    F = regime.build_features(two_regime_close)
    assert list(F.columns) == regime.FEATURES
    assert len(F) == len(two_regime_close)


def test_jump_model_separates_regimes(two_regime_close):
    F = regime.build_features(two_regime_close).dropna()
    Z = regime.standardize(F.to_numpy())
    model = regime.JumpModel(n_states=2, jump_penalty=10.0, seed=0).fit(Z)
    labels = model.named_labels()
    assert set(labels) <= {'trending', 'mean-reverting'}
    # both states actually used
    assert len(set(labels)) == 2


def test_higher_penalty_gives_longer_runs(two_regime_close):
    F = regime.build_features(two_regime_close).dropna()
    Z = regime.standardize(F.to_numpy())
    loose = regime.JumpModel(n_states=2, jump_penalty=1.0, seed=0).fit(Z)
    tight = regime.JumpModel(n_states=2, jump_penalty=200.0, seed=0).fit(Z)
    switches_loose = int((np.diff(loose.labels_) != 0).sum())
    switches_tight = int((np.diff(tight.labels_) != 0).sum())
    assert switches_tight <= switches_loose


def test_labels_are_causal(two_regime_close):
    """The label at bar i must not change when later bars are added."""
    F = regime.build_features(two_regime_close)
    long_labels = regime.label_walk_forward(
        F, n_states=2, jump_penalty=20.0, min_train=300, refit_every=21)
    short_labels = regime.label_walk_forward(
        F.iloc[:700], n_states=2, jump_penalty=20.0, min_train=300,
        refit_every=21)
    overlap = short_labels.dropna().index
    assert len(overlap) > 50
    matched = (long_labels.loc[overlap] == short_labels.loc[overlap]).mean()
    assert matched == 1.0


def test_state_summary_reports_run_lengths():
    labels = pd.Series(['a'] * 10 + ['b'] * 5 + ['a'] * 10)
    out = regime.state_summary(labels)
    assert out['a']['n_runs'] == 2
    assert out['a']['mean_run_length'] == 10.0
    assert out['b']['share'] == pytest.approx(0.2)


def test_state_summary_handles_empty():
    assert regime.state_summary(pd.Series([], dtype=object)) == {}


def test_standardize_only_uses_rows_up_to_upto():
    X = np.vstack([np.zeros((10, 2)), np.full((10, 2), 100.0)])
    Z = regime.standardize(X, upto=9)
    # the first block is constant, so sd falls back to 1 and it maps to 0
    assert np.allclose(Z[:10], 0.0)
    assert Z[10:].max() > 10


def test_estimate_state_weights_shrinks_toward_the_prior():
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(400):
        scores = {k: float(rng.normal()) for k in REGIME_WEIGHTS['mixed']}
        # make momentum the only thing that pays in the trending state
        fwd = 0.02 * scores['momentum'] + rng.normal(0, 0.001)
        samples.append(('trending', scores, fwd))

    table, diag = regime.estimate_state_weights(
        samples, REGIME_WEIGHTS, shrinkage=0.5)
    assert diag['trending']['fitted'] is True
    assert diag['trending']['n_samples'] == 400
    # momentum should rise above its prior but not all the way to 1.0
    assert table['trending']['momentum'] > REGIME_WEIGHTS['trending']['momentum']
    assert table['trending']['momentum'] < 1.0
    assert sum(table['trending'].values()) == pytest.approx(1.0, abs=1e-3)


def test_estimate_state_weights_keeps_prior_without_data():
    table, diag = regime.estimate_state_weights([], REGIME_WEIGHTS)
    for state, weights in REGIME_WEIGHTS.items():
        total = sum(weights.values())
        for sleeve, w in weights.items():
            assert table[state][sleeve] == pytest.approx(w / total, abs=1e-3)
        assert diag[state]['fitted'] is False


def test_weights_always_sum_to_one():
    rng = np.random.default_rng(1)
    samples = [('mixed', {k: float(rng.normal()) for k in REGIME_WEIGHTS['mixed']},
                float(rng.normal(0, 0.01))) for _ in range(200)]
    table, _ = regime.estimate_state_weights(samples, REGIME_WEIGHTS,
                                             shrinkage=1.0)
    for state in table:
        assert sum(table[state].values()) == pytest.approx(1.0, abs=1e-3)
        assert all(w >= 0 for w in table[state].values())
