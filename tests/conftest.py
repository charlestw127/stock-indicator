import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_ohlcv(closes, seed=0, end='2026-08-11'):
    """OHLCV frame around a given close path."""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    r = np.random.default_rng(seed)
    high = closes * (1 + np.abs(r.normal(0, 0.008, n)))
    low = closes * (1 - np.abs(r.normal(0, 0.008, n)))
    open_ = low + (high - low) * r.random(n)
    volume = r.integers(1_000_000, 5_000_000, n).astype(float)
    idx = pd.bdate_range(end=end, periods=n)
    return pd.DataFrame({'Open': open_, 'High': high, 'Low': low,
                         'Close': closes, 'Volume': volume}, index=idx)


@pytest.fixture
def rng():
    return np.random.default_rng(7)


@pytest.fixture
def trending_df(rng):
    # persistent (positively autocorrelated) increments, so it actually trends
    eps = rng.normal(0.0004, 0.01, 1000)
    inc = np.zeros(1000)
    for i in range(1, 1000):
        inc[i] = 0.35 * inc[i - 1] + eps[i]
    return make_ohlcv(100 * np.exp(np.cumsum(inc + 0.0008)), seed=1)


@pytest.fixture
def mean_reverting_df(rng):
    # OU process, phi = 0.9 -> half-life about 6.6 days
    x = np.zeros(1000)
    for i in range(1, 1000):
        x[i] = 0.9 * x[i - 1] + rng.normal(0, 0.02)
    return make_ohlcv(100 * np.exp(x), seed=2)


@pytest.fixture
def noisy_df(rng):
    return make_ohlcv(100 * np.exp(np.cumsum(rng.normal(0, 0.015, 800))), seed=3)
