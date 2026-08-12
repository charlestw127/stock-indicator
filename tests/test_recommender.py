import numpy as np

from data_store import DataStore
from recommender import recommend, MAX_NAMES
from conftest import make_ohlcv


def build_store(tmp_path, price_paths):
    store = DataStore(str(tmp_path / 'reco.db'))
    for i, (sym, closes) in enumerate(price_paths.items()):
        store.store_history(sym, make_ohlcv(closes, seed=i), mark_fresh=True)
    return store


def fake_results(scores, extra_periods=None):
    out = {'symbols': {}}
    for sym, score in scores.items():
        entry = {'1m': {'score': score, 'rank': 1,
                        'risk': {'sharpe': 1.0, 'ann_vol': 20.0, 'beta': 1.0},
                        'signals': []}}
        if extra_periods:
            entry.update(extra_periods.get(sym, {}))
        out['symbols'][sym] = entry
    return out


def independent_paths(n, bars=200, seed=3):
    rng = np.random.default_rng(seed)
    return {f'S{k}': 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, bars)))
            for k in range(n)}


def test_respects_max_names(tmp_path):
    paths = independent_paths(30)
    store = build_store(tmp_path, paths)
    results = fake_results({s: 50 - i for i, s in enumerate(paths)})
    rec = recommend(results, store, max_names=10)
    assert len(rec['holdings']) == 10
    # hard cap regardless of the requested size
    rec = recommend(results, store, max_names=99)
    assert len(rec['holdings']) <= MAX_NAMES


def test_never_recommends_negative_scores(tmp_path):
    paths = independent_paths(6)
    store = build_store(tmp_path, paths)
    scores = {s: (10 if i < 2 else -20) for i, s in enumerate(paths)}
    rec = recommend(fake_results(scores), store, max_names=20)
    assert len(rec['holdings']) == 2  # slots stay empty rather than fill with bears


def test_correlation_dedup(tmp_path):
    rng = np.random.default_rng(9)
    base = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, 300)))
    paths = {
        'TWIN_A': base,
        'TWIN_B': base * 1.5,  # perfectly correlated with TWIN_A
        'OTHER': 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, 300))),
    }
    store = build_store(tmp_path, paths)
    results = fake_results({'TWIN_A': 40, 'TWIN_B': 35, 'OTHER': 10})
    rec = recommend(results, store, max_names=20)
    held = {h['symbol'] for h in rec['holdings']}
    assert 'TWIN_A' in held
    assert 'TWIN_B' not in held  # skipped as a near-duplicate
    assert 'OTHER' in held


def test_weights_sum_and_cap(tmp_path):
    paths = independent_paths(15)
    store = build_store(tmp_path, paths)
    results = fake_results({s: 30 - i for i, s in enumerate(paths)})
    rec = recommend(results, store, max_names=15)
    total = sum(h['weight'] for h in rec['holdings'])
    assert 98.0 < total < 102.0
    assert max(h['weight'] for h in rec['holdings']) <= 16.0  # 15% cap + rounding


def test_hysteresis_keeps_incumbents(tmp_path):
    paths = independent_paths(10)
    store = build_store(tmp_path, paths)
    syms = list(paths)
    # S9 is an incumbent now ranked 4th by score. A fresh pick would take
    # the top 3 (S0, S1, S2), but the incumbent survives while it ranks
    # inside max_names * KEEP_FACTOR, displacing the marginal newcomer.
    scores = {s: 30 - i for i, s in enumerate(syms)}   # S0 best ... S9 worst
    scores['S9'] = 27.5                                # between S3 and S2
    rec = recommend(fake_results(scores), store, max_names=3,
                    prev_symbols=['S9'])
    held = {h['symbol'] for h in rec['holdings']}
    assert 'S9' in held
    assert 'S2' not in held
    assert rec['changes']['dropped'] == []


def test_reposition_diff(tmp_path):
    paths = independent_paths(5)
    store = build_store(tmp_path, paths)
    results = fake_results({s: 20 for s in paths})
    positions = [{'symbol': 'S0', 'shares': 10, 'entryPrice': 50},
                 {'symbol': 'ZZZ_NOT_RECOMMENDED', 'shares': 5, 'entryPrice': 10}]
    store.store_history('ZZZ_NOT_RECOMMENDED',
                        make_ohlcv(np.linspace(100, 90, 200), seed=42),
                        mark_fresh=True)
    rec = recommend(results, store, max_names=5, positions=positions)
    vs = rec['vs_current']
    assert 'S0' not in [d['symbol'] for d in vs['not_held']]
    assert 'ZZZ_NOT_RECOMMENDED' in [d['symbol'] for d in vs['held_not_recommended']]
