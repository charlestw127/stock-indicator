"""
Quantitative indicator library:

  1. Volatility estimators   - ATR, close-to-close, Parkinson, Garman-Klass,
                               volatility regime percentile
  2. Trend / momentum        - multi-horizon total-return momentum, 12-1
                               momentum, risk-adjusted momentum (t-stat),
                               KAMA, Kaufman efficiency ratio, regression
                               trend (slope + R^2), 52-week-high proximity
  3. Mean-reversion stats    - price z-score, Hurst exponent, Ornstein-
                               Uhlenbeck half-life, variance ratio
  4. Volume / flow           - Money Flow Index, Chaikin Money Flow, OBV
                               (vectorized) + OBV z-score, rolling VWAP
                               deviation, relative volume
  5. Channels / breakouts    - Bollinger, Keltner, TTM squeeze, Donchian
  6. Risk metrics            - Sharpe, Sortino, max drawdown, Calmar,
                               historical VaR / CVaR, beta / alpha vs
                               benchmark, skewness / kurtosis

Series-returning functions align to the input index. Statistics that are only
meaningful on a trailing window (Hurst, half-life, regression trend, risk
metrics) return scalars evaluated on the most recent `window` observations.

Only numpy + pandas are required.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log_returns(close):
    """Daily log returns."""
    return np.log(close / close.shift(1))


def simple_returns(close):
    """Daily simple returns."""
    return close.pct_change()


def _tail(series, window):
    """Last `window` non-NaN observations as a numpy array."""
    return series.dropna().iloc[-window:].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# 1. Volatility estimators
# ---------------------------------------------------------------------------

def true_range(high, low, close):
    """True range: max(H-L, |H-prevC|, |L-prevC|)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(high, low, close, period=14):
    """Average True Range using Wilder's smoothing."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def atr_percent(high, low, close, period=14):
    """ATR as a percent of price - comparable across symbols."""
    return 100.0 * atr(high, low, close, period) / close


def realized_volatility(close, window=21):
    """Annualized close-to-close realized volatility, percent."""
    lr = log_returns(close)
    return lr.rolling(window).std() * np.sqrt(TRADING_DAYS) * 100.0


def parkinson_volatility(high, low, window=21):
    """Parkinson range-based volatility estimator (annualized, percent).

    More efficient than close-to-close because it uses the intraday range.
    """
    hl = np.log(high / low) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    return np.sqrt(factor * hl.rolling(window).mean() * TRADING_DAYS) * 100.0


def garman_klass_volatility(open_, high, low, close, window=21):
    """Garman-Klass OHLC volatility estimator (annualized, percent)."""
    hl = 0.5 * np.log(high / low) ** 2
    co = (2.0 * np.log(2.0) - 1.0) * np.log(close / open_) ** 2
    var = (hl - co).rolling(window).mean()
    var = var.clip(lower=0)
    return np.sqrt(var * TRADING_DAYS) * 100.0


def volatility_percentile(close, vol_window=21, lookback=TRADING_DAYS):
    """Where today's realized vol sits within its own trailing distribution.

    Returns a scalar 0-100. >80 = high-vol regime, <20 = quiet regime.
    """
    vol = realized_volatility(close, vol_window).dropna()
    if len(vol) < vol_window:
        return None
    tail = vol.iloc[-lookback:]
    return float((tail <= vol.iloc[-1]).mean() * 100.0)


# ---------------------------------------------------------------------------
# 2. Trend / momentum
# ---------------------------------------------------------------------------

def total_return_momentum(close, lookback):
    """Total return over `lookback` sessions (classic time-series momentum)."""
    return close / close.shift(lookback) - 1.0


def momentum_12_1(close):
    """12-month momentum skipping the most recent month (the academic
    cross-sectional momentum factor: avoids 1-month reversal)."""
    return close.shift(21) / close.shift(TRADING_DAYS) - 1.0


def risk_adjusted_momentum(close, lookback):
    """Momentum t-statistic: mean daily return / std * sqrt(n).

    Rewards steady trends over volatile ones - the way CTAs size signals.
    """
    lr = log_returns(close)
    mean = lr.rolling(lookback).mean()
    std = lr.rolling(lookback).std()
    return mean / std.replace(0, np.nan) * np.sqrt(lookback)


def efficiency_ratio(close, period=10):
    """Kaufman efficiency ratio: |net change| / sum of |daily changes|.

    1.0 = perfectly efficient trend, 0.0 = pure noise.
    """
    change = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period).sum()
    return (change / volatility.replace(0, np.nan)).clip(0, 1)


def kama(close, er_period=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average - fast in trends, slow in chop."""
    er = efficiency_ratio(close, er_period).fillna(0).to_numpy()
    prices = close.to_numpy(dtype=float)
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    out = np.full_like(prices, np.nan)
    if len(prices) == 0:
        return pd.Series(out, index=close.index)
    out[0] = prices[0]
    for i in range(1, len(prices)):
        out[i] = out[i - 1] + sc[i] * (prices[i] - out[i - 1])
    return pd.Series(out, index=close.index)


def linreg_trend(close, window=63):
    """OLS regression of log price on time over the trailing window.

    Returns (annualized_slope_pct, r_squared).
    slope * r_squared is a standard trend-quality signal.
    """
    y = np.log(_tail(close, window))
    n = len(y)
    if n < max(10, window // 3):
        return None, None
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    annualized = (np.exp(slope * TRADING_DAYS) - 1.0) * 100.0
    return float(annualized), float(max(0.0, min(1.0, r2)))


def dist_from_52wk_high(close, window=TRADING_DAYS):
    """Percent distance from trailing 52-week high (<= 0).

    Proximity to the 52-week high is a documented momentum anomaly
    (George & Hwang 2004).
    """
    rolling_high = close.rolling(window, min_periods=20).max()
    return 100.0 * (close / rolling_high - 1.0)


# ---------------------------------------------------------------------------
# 3. Mean-reversion statistics
# ---------------------------------------------------------------------------

def price_zscore(close, window=20):
    """Z-score of price versus its rolling mean - the basic stat-arb signal."""
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (close - mean) / std.replace(0, np.nan)


def hurst_exponent(close, window=TRADING_DAYS, max_lag=20):
    """Hurst exponent via the variance-of-lagged-differences method.

    H > 0.5 trending / persistent, H < 0.5 mean-reverting, H ~ 0.5 random walk.
    Returns a scalar on the trailing `window` observations.
    """
    prices = np.log(_tail(close, window))
    if len(prices) < max_lag * 3:
        return None
    lags = np.arange(2, max_lag + 1)
    tau = np.array([np.std(prices[lag:] - prices[:-lag]) for lag in lags])
    if np.any(tau <= 0):
        return None
    slope = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    return float(min(max(slope, 0.0), 1.0))


def half_life_mean_reversion(close, window=126):
    """Half-life of mean reversion from an AR(1) / Ornstein-Uhlenbeck fit.

    Regress delta(p) on lagged p; half-life = -ln(2)/b. Short half-life =
    fast reversion (good for mean-reversion trading). Returns days or None
    when the series shows no reversion (b >= 0).
    """
    prices = np.log(_tail(close, window))
    if len(prices) < 30:
        return None
    lagged = prices[:-1]
    delta = np.diff(prices)
    lagged_c = lagged - lagged.mean()
    denom = np.sum(lagged_c ** 2)
    if denom <= 0:
        return None
    b = np.sum(lagged_c * (delta - delta.mean())) / denom
    if b >= 0:
        return None  # no mean reversion detected
    hl = -np.log(2.0) / b
    return float(hl) if 0 < hl < 10 * window else None


def variance_ratio(close, q=5, window=TRADING_DAYS):
    """Lo-MacKinlay variance ratio: Var(q-day returns) / (q * Var(1-day)).

    > 1 = positive autocorrelation (trending), < 1 = mean-reverting.
    """
    lr = log_returns(close).dropna()
    lr = lr.iloc[-window:]
    if len(lr) < q * 10:
        return None
    var1 = lr.var()
    if not var1 or var1 <= 0:
        return None
    varq = lr.rolling(q).sum().dropna().var()
    return float(varq / (q * var1))


# ---------------------------------------------------------------------------
# 4. Volume / flow
# ---------------------------------------------------------------------------

def money_flow_index(high, low, close, volume, period=14):
    """MFI - volume-weighted RSI. >80 overbought, <20 oversold."""
    tp = (high + low + close) / 3.0
    raw_flow = tp * volume
    direction = tp.diff()
    pos_flow = raw_flow.where(direction > 0, 0.0).rolling(period).sum()
    neg_flow = raw_flow.where(direction < 0, 0.0).rolling(period).sum()
    ratio = pos_flow / neg_flow.replace(0, np.nan)
    mfi = 100.0 - 100.0 / (1.0 + ratio)
    return mfi.fillna(50.0)


def chaikin_money_flow(high, low, close, volume, period=20):
    """CMF - accumulation/distribution pressure, roughly -1..1."""
    rng = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / rng
    mfv = (mfm * volume).rolling(period).sum()
    vol_sum = volume.rolling(period).sum().replace(0, np.nan)
    return (mfv / vol_sum).fillna(0.0)


def obv(close, volume):
    """On-balance volume, vectorized (raw cumulative units)."""
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def obv_zscore(close, volume, window=20):
    """Z-score of the OBV change over `window` - flow trend confirmation."""
    line = obv(close, volume)
    change = line.diff(window)
    std = line.diff().rolling(window * 3, min_periods=window).std()
    denom = (std * np.sqrt(window)).replace(0, np.nan)
    return change / denom


def rolling_vwap(high, low, close, volume, window=20):
    """Rolling volume-weighted average price over `window` sessions."""
    tp = (high + low + close) / 3.0
    pv = (tp * volume).rolling(window).sum()
    v = volume.rolling(window).sum().replace(0, np.nan)
    return pv / v


def vwap_deviation(high, low, close, volume, window=20):
    """Percent deviation of close from rolling VWAP - execution-desk anchor."""
    vwap = rolling_vwap(high, low, close, volume, window)
    return 100.0 * (close / vwap - 1.0)


def relative_volume(volume, window=20):
    """Today's volume as a multiple of its trailing average."""
    avg = volume.rolling(window).mean().replace(0, np.nan)
    return volume / avg


# ---------------------------------------------------------------------------
# 5. Channels / breakouts
# ---------------------------------------------------------------------------

def bollinger_bands(close, window=20, num_std=2.0):
    """Returns (middle, upper, lower)."""
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    return mid, mid + num_std * std, mid - num_std * std


def keltner_channels(high, low, close, ema_period=20, atr_period=10, mult=1.5):
    """Returns (middle, upper, lower) using EMA +/- mult * ATR."""
    mid = close.ewm(span=ema_period, adjust=False).mean()
    rng = atr(high, low, close, atr_period)
    return mid, mid + mult * rng, mid - mult * rng


def ttm_squeeze(high, low, close, bb_window=20, bb_std=2.0,
                kc_period=20, kc_mult=1.5):
    """TTM squeeze: Bollinger bands trading inside Keltner channels means
    volatility is compressed and a directional move is loading.

    Returns (squeeze_on: bool Series, fired: bool Series) where `fired`
    marks the bar on which the squeeze released.
    """
    _, bb_up, bb_lo = bollinger_bands(close, bb_window, bb_std)
    _, kc_up, kc_lo = keltner_channels(high, low, close, kc_period, 10, kc_mult)
    squeeze_on = (bb_up < kc_up) & (bb_lo > kc_lo)
    fired = squeeze_on.shift(1).fillna(False) & ~squeeze_on
    return squeeze_on.fillna(False), fired.fillna(False)


def donchian_position(high, low, close, window=20):
    """Position of close within the Donchian channel, -1 (low) .. +1 (high)."""
    upper = high.rolling(window).max()
    lower = low.rolling(window).min()
    mid = (upper + lower) / 2.0
    half = ((upper - lower) / 2.0).replace(0, np.nan)
    return ((close - mid) / half).clip(-1, 1)


def donchian_breakout(high, low, close, window=20):
    """+1 on a close above the prior `window`-day high, -1 below the prior
    low, 0 otherwise. The classic turtle/CTA breakout trigger."""
    prior_high = high.rolling(window).max().shift(1)
    prior_low = low.rolling(window).min().shift(1)
    signal = pd.Series(0, index=close.index)
    signal[close > prior_high] = 1
    signal[close < prior_low] = -1
    return signal


# ---------------------------------------------------------------------------
# 6. Risk metrics (scalars on the trailing window)
# ---------------------------------------------------------------------------

def sharpe_ratio(close, window=TRADING_DAYS, risk_free_annual=0.0):
    """Annualized Sharpe ratio of daily returns over the trailing window."""
    rets = _tail(simple_returns(close), window)
    if len(rets) < 20 or np.std(rets) == 0:
        return None
    rf_daily = risk_free_annual / TRADING_DAYS
    excess = rets - rf_daily
    return float(np.mean(excess) / np.std(excess) * np.sqrt(TRADING_DAYS))


def sortino_ratio(close, window=TRADING_DAYS, risk_free_annual=0.0):
    """Sharpe variant penalizing only downside deviation."""
    rets = _tail(simple_returns(close), window)
    if len(rets) < 20:
        return None
    rf_daily = risk_free_annual / TRADING_DAYS
    excess = rets - rf_daily
    downside = excess[excess < 0]
    if len(downside) == 0:
        return None
    dd = np.sqrt(np.mean(downside ** 2))
    if dd == 0:
        return None
    return float(np.mean(excess) / dd * np.sqrt(TRADING_DAYS))


def max_drawdown(close, window=TRADING_DAYS):
    """Deepest peak-to-trough decline over the window, percent (negative)."""
    prices = _tail(close, window)
    if len(prices) < 5:
        return None
    running_max = np.maximum.accumulate(prices)
    dd = prices / running_max - 1.0
    return float(dd.min() * 100.0)


def calmar_ratio(close, window=TRADING_DAYS):
    """Annualized return / |max drawdown| over the window."""
    prices = _tail(close, window)
    mdd = max_drawdown(close, window)
    if len(prices) < 20 or not mdd:
        return None
    years = len(prices) / TRADING_DAYS
    total = prices[-1] / prices[0]
    if total <= 0 or years <= 0:
        return None
    ann_ret = (total ** (1.0 / years) - 1.0) * 100.0
    return float(ann_ret / abs(mdd))


def var_cvar(close, window=TRADING_DAYS, level=0.05):
    """Historical 1-day VaR and CVaR (expected shortfall) at `level`.

    Returned as negative percents, e.g. (-2.1, -3.4).
    """
    rets = _tail(simple_returns(close), window)
    if len(rets) < 60:
        return None, None
    var = np.quantile(rets, level)
    tail = rets[rets <= var]
    cvar = tail.mean() if len(tail) else var
    return float(var * 100.0), float(cvar * 100.0)


def beta_alpha(close, benchmark_close, window=TRADING_DAYS):
    """OLS beta and annualized alpha (percent) versus a benchmark."""
    if benchmark_close is None:
        return None, None
    joined = pd.concat(
        [simple_returns(close), simple_returns(benchmark_close)],
        axis=1, join="inner",
    ).dropna()
    if len(joined) < 60:
        return None, None
    sub = joined.iloc[-window:].to_numpy(dtype=float)
    asset, bench = sub[:, 0], sub[:, 1]
    var_b = np.var(bench)
    if var_b == 0:
        return None, None
    beta = float(np.cov(asset, bench)[0, 1] / var_b)
    alpha_daily = np.mean(asset) - beta * np.mean(bench)
    return beta, float(alpha_daily * TRADING_DAYS * 100.0)


def return_skew_kurtosis(close, window=TRADING_DAYS):
    """Skewness and excess kurtosis of daily returns - crash-risk flags."""
    rets = _tail(simple_returns(close), window)
    if len(rets) < 60:
        return None, None
    std = np.std(rets)
    if std == 0:
        return None, None
    centered = rets - np.mean(rets)
    skew = float(np.mean(centered ** 3) / std ** 3)
    kurt = float(np.mean(centered ** 4) / std ** 4 - 3.0)
    return skew, kurt
