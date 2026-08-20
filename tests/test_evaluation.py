import numpy as np
import pandas as pd
import pytest

import evaluation as ev


def test_norm_ppf_matches_known_quantiles():
    for p, want in [(0.975, 1.959964), (0.99, 2.326348), (0.001, -3.090232),
                    (0.5, 0.0)]:
        assert ev.norm_ppf(p) == pytest.approx(want, abs=1e-5)


def test_norm_cdf_inverts_ppf():
    for p in (0.01, 0.25, 0.5, 0.8, 0.999):
        assert ev.norm_cdf(ev.norm_ppf(p)) == pytest.approx(p, abs=1e-9)


def test_norm_ppf_rejects_out_of_range():
    with pytest.raises(ValueError):
        ev.norm_ppf(0.0)
    with pytest.raises(ValueError):
        ev.norm_ppf(1.0)


def test_expected_max_t_reproduces_the_usual_hurdles():
    # the numbers behind the Harvey-Liu-Zhu t > 3 convention
    assert ev.expected_max_t(45) == pytest.approx(2.76, abs=0.02)
    assert ev.expected_max_t(200) == pytest.approx(3.26, abs=0.02)
    assert ev.expected_max_t(2000) > ev.expected_max_t(200)


def test_expected_max_sharpe_grows_with_trials():
    a = ev.expected_max_sharpe(10, 1.0)
    b = ev.expected_max_sharpe(1000, 1.0)
    assert 0 < a < b


def test_apply_costs_charges_round_trip_turnover():
    out = ev.apply_costs([0.01, 0.01], [0.5, 0.5], 10)
    # 10bps one way, 50% turnover -> 10 bps charged
    assert out == pytest.approx([0.009, 0.009])
    assert ev.apply_costs([0.01], [0.5], 0) == pytest.approx([0.01])


def test_apply_costs_pads_mismatched_turnover():
    out = ev.apply_costs([0.01, 0.01, 0.01], [0.5, 0.5], 10)
    assert len(out) == 3


def test_deflated_sharpe_falls_as_trials_rise():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, 400)
    one = ev.deflated_sharpe(rets, 1)
    many = ev.deflated_sharpe(rets, 500)
    assert one['sharpe'] == many['sharpe']
    assert many['deflated_sharpe'] < one['deflated_sharpe']
    assert many['sr0'] > one['sr0']


def test_deflated_sharpe_handles_degenerate_input():
    assert ev.deflated_sharpe([0.0, 0.0, 0.0, 0.0], 5) is None
    assert ev.deflated_sharpe([0.01], 5) is None


def test_deflated_t_separates_direction():
    good = ev.deflated_t(4.0, 45)
    bad = ev.deflated_t(-4.0, 45)
    assert good['passes'] and good['clears_positive']
    assert bad['passes'] and not bad['clears_positive']
    assert not ev.deflated_t(1.0, 45)['passes']


def test_pbo_near_half_for_pure_noise():
    rng = np.random.default_rng(3)
    m = rng.normal(0, 0.01, size=(240, 16))
    out = ev.probability_of_backtest_overfitting(m, n_splits=8)
    assert 0.2 <= out['pbo'] <= 0.8
    assert out['n_configs'] == 16


def test_pbo_low_when_one_config_is_genuinely_better():
    rng = np.random.default_rng(4)
    m = rng.normal(0, 0.01, size=(240, 16))
    m[:, 5] += 0.006
    out = ev.probability_of_backtest_overfitting(m, n_splits=8)
    assert out['pbo'] < 0.2
    assert out['passes']


def test_pbo_rejects_bad_shapes():
    assert ev.probability_of_backtest_overfitting(np.zeros((10, 1))) is None
    assert ev.probability_of_backtest_overfitting(np.zeros(10)) is None


def test_regime_labels_track_the_200dma():
    idx = pd.date_range('2020-01-01', periods=400, freq='B')
    up = pd.Series(np.linspace(100, 200, 400), index=idx)
    labels = ev.regime_labels(idx[-5:], up)
    assert set(labels) == {'bull'}

    down = pd.Series(np.linspace(200, 100, 400), index=idx)
    labels = ev.regime_labels(idx[-5:], down)
    assert set(labels) == {'bear'}


def test_split_by_regime_partitions_returns():
    rets = [0.01] * 20 + [-0.01] * 20
    labels = ['bull'] * 20 + ['bear'] * 20

    def stats(r, step):
        return {'mean': float(np.mean(r))}

    out = ev.split_by_regime(rets, labels, 5, stats)
    assert out['bull']['mean'] > 0 > out['bear']['mean']
    assert out['bull']['n_periods'] == 20


def test_null_ic_distribution_centres_near_zero():
    rng = np.random.default_rng(7)
    per_date = []
    for _ in range(40):
        syms = [f'S{i}' for i in range(30)]
        per_date.append({
            'scores': {s: {'composite': rng.normal(), 'trend': rng.normal()}
                       for s in syms},
            'fwd': {s: rng.normal(0, 0.02) for s in syms},
        })
    out = ev.null_ic_distribution(per_date, ['composite', 'trend'],
                                  n_permutations=30, seed=1)
    assert out['n_permutations'] == 30
    # best-of-two sleeves on shuffled data is positive but small
    assert 0 < out['mean'] < 0.1
    assert out['p95'] >= out['mean']


def test_trial_ledger_counts_distinct_configs(tmp_path):
    path = tmp_path / 'trials.jsonl'
    assert ev.count_trials(str(path)) == 1  # nothing logged yet
    ev.log_trial({'a': 1}, {'x': 1}, path=str(path))
    ev.log_trial({'a': 1}, {'x': 2}, path=str(path))  # same config, rerun
    ev.log_trial({'a': 2}, {'x': 3}, path=str(path))
    assert ev.count_trials(str(path)) == 2
    assert ev.count_trials(str(path), distinct=False) == 3


def test_trial_ledger_survives_corrupt_lines(tmp_path):
    path = tmp_path / 'trials.jsonl'
    ev.log_trial({'a': 1}, {}, path=str(path))
    with open(path, 'a') as f:
        f.write('not json\n\n')
    assert ev.count_trials(str(path)) == 1
