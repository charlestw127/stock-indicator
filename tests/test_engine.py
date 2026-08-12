import numpy as np
import pytest

from quant_engine import (
    SymbolAnalyzer, analyze_symbol, HORIZON_PARAMS, score_to_recommendation,
)
from conftest import make_ohlcv

PERIODS = list(HORIZON_PARAMS)


def test_analyze_symbol_shape(noisy_df):
    res = analyze_symbol(noisy_df, PERIODS)
    assert set(res) == set(PERIODS)
    for period, r in res.items():
        assert -100 <= r['score'] <= 100
        assert set(r['factors']) == {'trend', 'momentum', 'mean_reversion',
                                     'volume_flow', 'quality'}
        for v in r['factors'].values():
            assert -1 <= v <= 1
        assert r['regime']['trend'] in ('trending', 'mean-reverting', 'mixed')
        assert 'risk' in r and 'signals' in r and 'indicators' in r
        assert r['lastPrice'] is not None


def test_no_lookahead(noisy_df):
    """Scoring bar i must not change when future bars are appended."""
    cut = 600
    full = SymbolAnalyzer(noisy_df)
    truncated = SymbolAnalyzer(noisy_df.iloc[:cut])
    for period in ('1w', '1m', '1y'):
        a = full.score_at(period, cut - 1)
        b = truncated.score_at(period, cut - 1)
        assert a['score'] == pytest.approx(b['score'], abs=1e-9), period
        for k in a['factors']:
            assert a['factors'][k] == pytest.approx(b['factors'][k], abs=1e-9)


def test_too_little_history_raises():
    df = make_ohlcv(np.linspace(100, 110, 20))
    with pytest.raises(ValueError):
        SymbolAnalyzer(df)


def test_short_history_degrades_gracefully():
    df = make_ohlcv(np.linspace(100, 120, 45))
    res = analyze_symbol(df, PERIODS)
    for r in res.values():
        assert -100 <= r['score'] <= 100


def test_regime_classification(trending_df, mean_reverting_df):
    trend_res = analyze_symbol(trending_df, ['1m'])['1m']
    mr_res = analyze_symbol(mean_reverting_df, ['1m'])['1m']
    assert trend_res['regime']['hurst'] > mr_res['regime']['hurst']
    assert mr_res['regime']['trend'] in ('mean-reverting', 'mixed')


def test_crash_scores_negative(rng):
    up = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 700)))
    crash = up[-1] * np.exp(np.cumsum(rng.normal(-0.005, 0.03, 300)))
    df = make_ohlcv(np.concatenate([up, crash]))
    res = analyze_symbol(df, ['1m', '1y'])
    assert res['1m']['score'] < 0
    assert res['1y']['score'] < 0


def test_recommendation_thresholds():
    assert score_to_recommendation(50) == 'BUY (Strong)'
    assert score_to_recommendation(20) == 'BUY (Weak)'
    assert score_to_recommendation(0) == 'HOLD'
    assert score_to_recommendation(-20) == 'SELL (Weak)'
    assert score_to_recommendation(-50) == 'SELL (Strong)'


def test_evaluate_matches_score_at(noisy_df):
    an = SymbolAnalyzer(noisy_df)
    full = an.evaluate('1m')
    fast = an.score_at('1m')
    assert full['score'] == pytest.approx(fast['score'], abs=1e-9)
