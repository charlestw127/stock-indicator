import numpy as np
import pandas as pd
import pytest

import quant_indicators as qi
from conftest import make_ohlcv


def test_rsi_bounds(noisy_df):
    from indicators import calculate_rsi
    rsi = calculate_rsi(noisy_df['Close'])
    assert rsi.min() >= 0 and rsi.max() <= 100


def test_atr_positive(noisy_df):
    atr = qi.atr(noisy_df['High'], noisy_df['Low'], noisy_df['Close']).dropna()
    assert (atr > 0).all()


def test_realized_vol_positive(noisy_df):
    vol = qi.realized_volatility(noisy_df['Close']).dropna()
    assert (vol > 0).all()
    # 1.5% daily noise is roughly 24% annualized
    assert 10 < vol.iloc[-1] < 60


def test_hurst_separates_regimes(trending_df, mean_reverting_df):
    h_trend = qi.hurst_exponent(trending_df['Close'])
    h_mr = qi.hurst_exponent(mean_reverting_df['Close'])
    assert h_trend > 0.5
    assert h_mr < 0.48
    assert h_trend > h_mr


def test_half_life_recovers_ou_parameter(mean_reverting_df):
    # phi = 0.9 implies a half-life of about 6.6 days
    hl = qi.half_life_mean_reversion(mean_reverting_df['Close'])
    assert hl is not None
    assert 3 < hl < 15


def test_half_life_none_for_trending(trending_df):
    hl = qi.half_life_mean_reversion(trending_df['Close'])
    assert hl is None or hl > 20


def test_variance_ratio_direction(trending_df, mean_reverting_df):
    vr_trend = qi.variance_ratio(trending_df['Close'])
    vr_mr = qi.variance_ratio(mean_reverting_df['Close'])
    assert vr_trend > 1.0
    assert vr_mr < 1.0


def test_max_drawdown_known_case():
    closes = list(range(10, 110)) + [50.0]  # peak 109, trough 50
    df = make_ohlcv(closes)
    mdd = qi.max_drawdown(df['Close'], window=len(closes))
    assert mdd == pytest.approx((50 / 109 - 1) * 100, abs=0.01)


def test_var_cvar_ordering(noisy_df):
    var, cvar = qi.var_cvar(noisy_df['Close'])
    assert var < 0
    assert cvar <= var


def test_beta_of_scaled_series():
    rng = np.random.default_rng(11)
    bench = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 600)))
    bench = pd.Series(bench, index=pd.bdate_range(end='2026-08-11', periods=600))
    # asset with exactly 2x the benchmark's daily returns
    asset = 100 * np.cumprod(1 + 2 * bench.pct_change().fillna(0))
    asset = pd.Series(asset.values, index=bench.index)
    beta, _ = qi.beta_alpha(asset, bench)
    assert beta == pytest.approx(2.0, abs=0.05)


def test_donchian_breakout_flags():
    closes = [100.0] * 50 + [120.0]  # clean breakout on the last bar
    df = make_ohlcv(closes)
    sig = qi.donchian_breakout(df['High'], df['Low'], df['Close'])
    assert sig.iloc[-1] == 1


def test_mfi_bounds(noisy_df):
    mfi = qi.money_flow_index(noisy_df['High'], noisy_df['Low'],
                              noisy_df['Close'], noisy_df['Volume']).dropna()
    assert mfi.min() >= 0 and mfi.max() <= 100


def test_cmf_bounds(noisy_df):
    cmf = qi.chaikin_money_flow(noisy_df['High'], noisy_df['Low'],
                                noisy_df['Close'], noisy_df['Volume']).dropna()
    assert cmf.min() >= -1.01 and cmf.max() <= 1.01


def test_efficiency_ratio_high_for_straight_line():
    closes = np.linspace(100, 200, 300)
    df = make_ohlcv(closes)
    er = qi.efficiency_ratio(df['Close']).dropna()
    assert er.iloc[-1] > 0.9


def test_dist_52wk_high_at_high():
    closes = np.linspace(100, 200, 300)  # monotonic, always at the high
    df = make_ohlcv(closes)
    dist = qi.dist_from_52wk_high(df['Close'])
    assert dist.iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_sharpe_sign(trending_df):
    assert qi.sharpe_ratio(trending_df['Close']) > 0
