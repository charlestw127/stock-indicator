import numpy as np
import pandas as pd
import pytest

import checks
from data_store import DataStore

from conftest import make_ohlcv


@pytest.fixture
def store(tmp_path, rng):
    s = DataStore(path=str(tmp_path / 'test.db'))
    # three names that move together, two that do not
    base = np.cumsum(rng.normal(0.0005, 0.01, 300))
    for sym in ('AAA', 'BBB', 'CCC'):
        noise = np.cumsum(rng.normal(0, 0.001, 300))
        s.store_history(sym, make_ohlcv(100 * np.exp(base + noise)),
                        mark_fresh=True)
    for sym in ('DDD', 'EEE'):
        path = np.cumsum(rng.normal(0.0003, 0.012, 300))
        s.store_history(sym, make_ohlcv(100 * np.exp(path)), mark_fresh=True)
    yield s
    s.close()


def _rec(weights, **extra):
    rec = {'holdings': [{'symbol': s, 'weight': w, 'score': 30.0}
                        for s, w in weights.items()]}
    rec.update(extra)
    return rec


def test_concentration_flags_a_breached_cap():
    rec = _rec({'AAA': 40.0, 'BBB': 30.0, 'CCC': 30.0})
    out = {c['id']: c for c in checks.run_checks(rec)}
    con = out['concentration']
    assert con['status'] == 'fail'
    assert 'AAA' in con['values']['over_cap']
    assert con['values']['max_weight_pct'] == 40.0


def test_concentration_passes_a_spread_book():
    rec = _rec({f'S{i}': 10.0 for i in range(10)})
    out = {c['id']: c for c in checks.run_checks(rec)}
    assert out['concentration']['status'] == 'ok'
    assert out['concentration']['values']['effective_names'] == pytest.approx(10.0)


def test_empty_portfolio_is_handled():
    out = checks.run_checks({'holdings': []})
    assert out[0]['id'] == 'empty'


def test_correlation_clusters_finds_coupled_names(store):
    rec = _rec({'AAA': 20.0, 'BBB': 20.0, 'CCC': 20.0, 'DDD': 20.0,
                'EEE': 20.0})
    out = {c['id']: c for c in checks.run_checks(rec, store=store)}
    cl = out['correlation_clusters']
    assert cl['status'] == 'warn'
    clusters = cl['values']['clusters']
    assert clusters
    assert set(clusters[0]) >= {'AAA', 'BBB', 'CCC'}


def test_correlation_check_survives_an_empty_store(tmp_path):
    s = DataStore(path=str(tmp_path / 'empty.db'))
    out = {c['id']: c for c in checks.run_checks(_rec({'ZZZ': 100.0}), store=s)}
    assert out['correlation_clusters']['values']['clusters'] == []
    s.close()


def test_earnings_proximity_counts_weight_at_risk():
    import datetime as dt
    soon = (dt.date.today() + dt.timedelta(days=5)).strftime('%Y-%m-%d')
    far = (dt.date.today() + dt.timedelta(days=200)).strftime('%Y-%m-%d')
    results = {'symbols': {
        'AAA': {'fundamentals': {'next_earnings': soon}},
        'BBB': {'fundamentals': {'next_earnings': far}},
    }}
    rec = _rec({'AAA': 30.0, 'BBB': 30.0})
    out = {c['id']: c for c in checks.run_checks(rec, results=results)}
    ep = out['earnings_proximity']
    assert ep['values']['weight_pct'] == 30.0
    assert ep['status'] == 'warn'
    assert ep['values']['names'][0]['symbol'] == 'AAA'


def test_earnings_ignores_unparseable_dates():
    results = {'symbols': {'AAA': {'fundamentals': {'next_earnings': 'soon'}}}}
    out = {c['id']: c for c in checks.run_checks(_rec({'AAA': 10.0}),
                                                 results=results)}
    assert out['earnings_proximity']['values']['weight_pct'] == 0.0


def test_turnover_cost_is_reported():
    rec = _rec({'AAA': 50.0, 'BBB': 50.0}, rebalance={
        'base_value': 10_000.0,
        'trades': [{'symbol': 'AAA', 'delta_value': 3000.0},
                   {'symbol': 'BBB', 'delta_value': -2000.0}]})
    out = {c['id']: c for c in checks.run_checks(rec)}
    tc = out['turnover_cost']
    assert tc['values']['turnover'] == pytest.approx(0.5)
    assert tc['values']['estimated_cost'] == pytest.approx(5.0)
    assert tc['status'] == 'warn'


def test_exposure_check_reflects_the_gate():
    rec = _rec({'AAA': 100.0}, exposure=0.6)
    out = {c['id']: c for c in checks.run_checks(
        rec, market={'risk': 'neutral', 'note': 'caution'})}
    assert out['exposure']['status'] == 'warn'
    assert out['exposure']['values']['exposure'] == 0.6

    rec = _rec({'AAA': 100.0}, exposure=1.0)
    out = {c['id']: c for c in checks.run_checks(rec)}
    assert out['exposure']['status'] == 'ok'


def test_score_separation_measures_the_gap():
    results = {'symbols': {}}
    for i in range(20):
        results['symbols'][f'S{i}'] = {'1m': {'score': float(i)}}
    rec = _rec({f'S{i}': 20.0 for i in range(15, 20)})
    out = {c['id']: c for c in checks.run_checks(rec, results=results)}
    sep = out['score_separation']
    assert sep['values']['held_mean'] > sep['values']['rest_mean']
    assert sep['values']['gap_in_sd'] > 0


def test_regime_mix_counts_states():
    results = {'symbols': {
        'AAA': {'1m': {'regime': {'trend': 'trending'}}},
        'BBB': {'1m': {'regime': {'trend': 'trending'}}},
        'CCC': {'1m': {'regime': {'trend': 'mixed'}}},
    }}
    rec = _rec({'AAA': 30.0, 'BBB': 30.0, 'CCC': 30.0})
    out = {c['id']: c for c in checks.run_checks(rec, results=results)}
    assert out['regime_mix']['values']['dominant'] == 'trending'
    assert out['regime_mix']['values']['counts']['trending'] == 2


# -- the critic --------------------------------------------------------

def test_review_without_llm_reports_the_failures():
    rec = _rec({'AAA': 40.0, 'BBB': 60.0})
    ch = checks.run_checks(rec)
    out = checks.review(rec, ch, use_llm=False)
    assert out['blocking'] is True
    assert out['source'] == 'deterministic'
    assert out['n_fail'] >= 1
    assert out['summary']


def test_review_does_not_block_a_clean_book():
    rec = _rec({f'S{i}': 10.0 for i in range(10)})
    ch = checks.run_checks(rec)
    out = checks.review(rec, ch, use_llm=False)
    assert out['blocking'] is False


def test_llm_cannot_change_blocking(monkeypatch):
    """The model may say what it likes; blocking comes from the checks."""
    import llm

    monkeypatch.setattr(llm, 'available', lambda: True)
    monkeypatch.setattr(llm, 'structured', lambda *a, **k: {
        'concerns': [{'check_id': 'concentration', 'severity': 'low',
                      'concern': 'looks fine to me', 'suggestion': 'ship it'}],
        'summary': 'No problems at all.'})
    rec = _rec({'AAA': 40.0, 'BBB': 60.0})   # breaches the cap
    ch = checks.run_checks(rec)
    out = checks.review(rec, ch, use_llm=True)
    assert out['source'] == 'llm'
    assert out['blocking'] is True


def test_llm_concerns_citing_unknown_checks_are_dropped(monkeypatch):
    import llm

    monkeypatch.setattr(llm, 'available', lambda: True)
    monkeypatch.setattr(llm, 'structured', lambda *a, **k: {
        'concerns': [
            {'check_id': 'concentration', 'severity': 'high',
             'concern': 'too concentrated', 'suggestion': 'trim'},
            {'check_id': 'insider_trading_risk', 'severity': 'high',
             'concern': 'invented', 'suggestion': 'invented'}],
        'summary': 's'})
    rec = _rec({'AAA': 40.0, 'BBB': 60.0})
    out = checks.review(rec, checks.run_checks(rec), use_llm=True)
    assert len(out['concerns']) == 1
    assert out['rejected_concerns'] == ['insider_trading_risk']


def test_review_falls_back_when_the_model_fails(monkeypatch):
    import llm

    monkeypatch.setattr(llm, 'available', lambda: True)

    def boom(*a, **k):
        raise llm.LLMUnavailable('down')

    monkeypatch.setattr(llm, 'structured', boom)
    rec = _rec({'AAA': 40.0, 'BBB': 60.0})
    out = checks.review(rec, checks.run_checks(rec), use_llm=True)
    assert out['source'] == 'deterministic'
    assert out['blocking'] is True


def test_log_flags_writes_a_row(tmp_path):
    rec = _rec({'AAA': 40.0, 'BBB': 60.0})
    ch = checks.run_checks(rec)
    out = checks.review(rec, ch, use_llm=False)
    path = tmp_path / 'flags.jsonl'
    row = checks.log_flags(rec, out, path=str(path))
    assert row['symbols'] == ['AAA', 'BBB']
    assert path.exists()
    assert any(f['status'] == 'fail' for f in row['flags'])
