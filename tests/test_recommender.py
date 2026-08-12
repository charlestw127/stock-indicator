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


def test_rebalance_plan(tmp_path):
    paths = independent_paths(5)
    store = build_store(tmp_path, paths)
    results = fake_results({s: 20 for s in paths})
    store.store_history('OLDPOS',
                        make_ohlcv(np.linspace(100, 90, 200), seed=42),
                        mark_fresh=True)
    positions = [{'symbol': 'S0', 'shares': 10, 'entryPrice': 50},
                 {'symbol': 'OLDPOS', 'shares': 5, 'entryPrice': 10}]
    rec = recommend(results, store, max_names=5, positions=positions)

    rb = rec['rebalance']
    assert rb['from_portfolio'] is True
    assert rb['base_value'] > 0

    trades = {t['symbol']: t for t in rb['trades']}
    # full exit of the non-recommended name uses the exact held share count
    assert trades['OLDPOS']['action'] == 'sell'
    assert trades['OLDPOS']['delta_shares'] == -5
    # recommended names not yet held get funded
    buys = [t for t in rb['trades'] if t['action'] == 'buy']
    assert buys and all(t['delta_value'] > 0 for t in buys)
    # holdings carry dollar and share targets when a base exists
    top = rec['holdings'][0]
    assert top['target_value'] > 0
    assert top['target_shares'] > 0
    assert top['price'] > 0


def test_rebalance_base_override(tmp_path):
    paths = independent_paths(4)
    store = build_store(tmp_path, paths)
    results = fake_results({s: 20 for s in paths})
    rec = recommend(results, store, max_names=4, base_value=10000)

    rb = rec['rebalance']
    assert rb['from_portfolio'] is False
    assert rb['base_value'] == 10000
    # with no current positions everything is a buy summing to about the base
    assert all(t['action'] == 'buy' for t in rb['trades'])
    total_targets = sum(h['target_value'] for h in rec['holdings'])
    assert 9500 < total_targets < 10500
