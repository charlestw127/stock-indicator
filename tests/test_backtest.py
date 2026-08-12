import numpy as np
import pandas as pd

from backtest import run_backtest, event_studies, STRATEGIES
from quant_engine import SymbolAnalyzer
from conftest import make_ohlcv


def build_universe(n_symbols=14, bars=900, seed=5):
    rng = np.random.default_rng(seed)
    analyzers = {}
    closes_all = []
    for k in range(n_symbols):
        drift = rng.normal(0.0004, 0.0006)
        closes = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, bars)))
        closes_all.append(closes)
        analyzers[f'SYM{k}'] = SymbolAnalyzer(make_ohlcv(closes, seed=k))
    spy_closes = np.mean(closes_all, axis=0)
    spy = pd.Series(spy_closes, index=pd.bdate_range(end='2026-08-11', periods=bars))
    return analyzers, spy


def test_run_backtest_structure():
    analyzers, spy = build_universe()
    results = run_backtest(analyzers, spy, period='1m', step=10, years=2, top=0.3)

    for strat in STRATEGIES + ['equal_weight', 'spy']:
        stats = results['strategies'][strat]
        assert stats['ann_vol'] >= 0
        assert -100 <= stats['max_drawdown'] <= 0
        assert 0 <= stats['hit_rate'] <= 100
        curve = results['equity_curves'][strat]
        assert len(curve) == len(results['dates'])
        assert all(np.isfinite(v) and v > 0 for v in curve)

    ic = results['ic']['composite']
    assert ic is not None
    assert -1 <= ic['mean'] <= 1
    assert ic['n'] > 10

    assert results['deciles']
    for d, entry in results['deciles'].items():
        assert entry['n'] > 0


def test_backtest_rejects_tiny_window():
    analyzers, spy = build_universe(bars=400)
    import pytest
    with pytest.raises(ValueError):
        run_backtest(analyzers, spy, step=50, years=0.2)


def test_event_studies_structure():
    analyzers, spy = build_universe(n_symbols=8, bars=800)
    out = event_studies(analyzers, years=2)
    assert set(out) == {'baseline', 'events'}
    assert out['baseline']['5d'] is not None
    for name, entry in out['events'].items():
        assert 'n' in entry
        if entry.get('21d') is not None:
            assert entry['n'] >= 20
