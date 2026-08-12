"""
Portfolio-level risk analytics: aggregate beta, volatility, VaR,
concentration and correlation structure across the user's positions.
"""

import logging

import numpy as np
import pandas as pd

TRADING_DAYS = 252
RET_WINDOW = 126  # sessions of daily returns used for vol/beta/correlation

logger = logging.getLogger('stock_app.portfolio_risk')


def portfolio_risk(positions, store, benchmark='SPY'):
    """Risk summary for a list of positions.

    positions: [{'symbol', 'shares', 'entryPrice'}, ...]
    Returns None when there is nothing to analyze.
    """
    holdings = []
    for pos in positions or []:
        symbol = pos.get('symbol')
        shares = float(pos.get('shares') or 0)
        if not symbol or shares <= 0:
            continue
        try:
            hist = store.get_history(symbol)
        except Exception as e:
            logger.warning("no history for %s: %s", symbol, e)
            continue
        if hist.empty:
            continue
        close = hist['Close'].astype(float)
        holdings.append((symbol, shares, close))

    if not holdings:
        return None

    values = {sym: shares * float(close.iloc[-1]) for sym, shares, close in holdings}
    total = sum(values.values())
    if total <= 0:
        return None
    weights = {sym: v / total for sym, v in values.items()}

    out = risk_from_weights(weights, store, benchmark)
    if out is None:
        return {'total_value': round(total, 2), 'weights': _round_map(weights)}
    out['total_value'] = round(total, 2)
    return out


def risk_from_weights(weights, store, benchmark='SPY'):
    """Risk summary for a {symbol: weight} book (weights sum to ~1)."""
    closes = {}
    for sym in weights:
        try:
            hist = store.get_history(sym)
        except Exception as e:
            logger.warning("no history for %s: %s", sym, e)
            continue
        if not hist.empty:
            closes[sym] = hist['Close'].astype(float)
    if not closes:
        return None

    rets = pd.DataFrame({
        sym: close.pct_change() for sym, close in closes.items()
    }).dropna(how='all').iloc[-RET_WINDOW:]
    rets = rets.dropna(axis=1, thresh=int(len(rets) * 0.6))
    if rets.empty or len(rets) < 20:
        return None

    w = np.array([weights.get(c, 0.0) for c in rets.columns])
    if w.sum() > 0:
        w = w / w.sum()
    port_rets = (rets.fillna(0.0) * w).sum(axis=1)

    ann_vol = float(port_rets.std() * np.sqrt(TRADING_DAYS) * 100.0)
    var95 = float(np.quantile(port_rets, 0.05) * 100.0)
    tail = port_rets[port_rets <= np.quantile(port_rets, 0.05)]
    cvar95 = float(tail.mean() * 100.0) if len(tail) else var95

    beta = None
    try:
        bench = store.get_history(benchmark)
        if not bench.empty:
            bench_rets = bench['Close'].astype(float).pct_change()
            joined = pd.concat([port_rets, bench_rets], axis=1, join='inner').dropna()
            if len(joined) >= 20:
                a, b = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
                var_b = np.var(b)
                if var_b > 0:
                    beta = float(np.cov(a, b)[0, 1] / var_b)
    except Exception as e:
        logger.warning("benchmark beta failed: %s", e)

    hhi = float(sum(x * x for x in weights.values()))
    top_symbol, top_weight = max(weights.items(), key=lambda kv: kv[1])

    avg_corr = None
    top_pair = None
    if rets.shape[1] >= 2:
        corr = rets.corr()
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        pairs = corr.where(mask).stack().dropna()
        if len(pairs):
            avg_corr = float(pairs.mean())
            (s1, s2), c = max(pairs.items(), key=lambda kv: kv[1])
            top_pair = {'symbols': [s1, s2], 'correlation': round(float(c), 2)}

    return {
        'weights': _round_map(weights),
        'beta': round(beta, 2) if beta is not None else None,
        'ann_vol': round(ann_vol, 1),
        'var_95': round(var95, 2),
        'cvar_95': round(cvar95, 2),
        'hhi': round(hhi, 3),
        'effective_positions': round(1.0 / hhi, 1) if hhi > 0 else None,
        'top_holding': {'symbol': top_symbol, 'weight': round(top_weight * 100, 1)},
        'avg_correlation': round(avg_corr, 2) if avg_corr is not None else None,
        'most_correlated_pair': top_pair,
    }


def _round_map(weights):
    return {k: round(v * 100, 1) for k, v in sorted(
        weights.items(), key=lambda kv: -kv[1])}
