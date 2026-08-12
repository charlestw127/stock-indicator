"""
Recommended portfolio construction from the latest watchlist scan.

Selection uses the 1m composite score - the horizon the walk-forward
backtest validated (top-quintile selection, weekly rebalance). The 1d/1w
scores are attached for entry timing but do not drive selection, so the
recommendation does not churn with every intraday scan. Three more things
keep turnover down:

- hysteresis: an incumbent holding is kept as long as it still ranks in
  the top 1.5x of the target size; a newcomer has to beat that
- correlation dedup: a candidate correlated > 0.92 with a name already
  selected is skipped (stops QQQ and VOO both taking a slot)
- positive-score requirement: slots are left empty rather than filled
  with bearish names

Weighting is inverse-volatility with a mild score tilt, capped at 15% per
name - low-vol names carry more weight, so position risk is roughly
balanced. The same scheme is backtestable via backtest.py --weighting.
"""

import logging

import numpy as np
import pandas as pd

from portfolio_risk import risk_from_weights, RET_WINDOW

logger = logging.getLogger('stock_app.recommender')

MAX_NAMES = 20
CORR_CAP = 0.92        # candidates above this vs a selected name are skipped
KEEP_FACTOR = 1.5      # incumbents survive while ranked within size * this
WEIGHT_CAP = 0.15
SELECT_PERIOD = '1m'   # the backtested selection horizon
TIMING_PERIODS = ('1d', '1w')


def recommend(results, store, max_names=MAX_NAMES, prev_symbols=None,
              positions=None):
    """Build the recommended portfolio.

    results: a scan results dict (results['symbols'][sym][period])
    prev_symbols: previously recommended symbols, for hysteresis
    positions: the user's current positions, for the reposition diff
    Returns a dict for the API, or None if there is nothing to recommend.
    """
    max_names = max(1, min(MAX_NAMES, int(max_names or MAX_NAMES)))
    prev_symbols = set(prev_symbols or [])

    candidates = []
    for symbol, periods in (results.get('symbols') or {}).items():
        if not isinstance(periods, dict):
            continue
        sel = periods.get(SELECT_PERIOD)
        if not isinstance(sel, dict) or sel.get('score') is None:
            continue
        if sel['score'] <= 0:
            continue  # never recommend a bearish name just to fill a slot
        candidates.append((symbol, sel))
    if not candidates:
        return None

    candidates.sort(key=lambda kv: -kv[1]['score'])
    order = {sym: i for i, (sym, _) in enumerate(candidates)}

    returns = _returns_matrix(store, [sym for sym, _ in candidates])

    # incumbents first (in score order), as long as they still rank well
    keep_rank = int(max_names * KEEP_FACTOR)
    selected = []
    for symbol, sel in candidates:
        if len(selected) >= max_names:
            break
        if symbol in prev_symbols and order[symbol] < keep_rank \
                and not _too_correlated(symbol, selected, returns):
            selected.append(symbol)

    for symbol, sel in candidates:
        if len(selected) >= max_names:
            break
        if symbol in selected:
            continue
        if _too_correlated(symbol, selected, returns):
            continue
        selected.append(symbol)

    if not selected:
        return None

    weights = _inverse_vol_weights(selected, results, returns)

    holdings = []
    for symbol in sorted(selected, key=lambda s: -weights.get(s, 0)):
        sel = dict(results['symbols'][symbol].get(SELECT_PERIOD) or {})
        risk = sel.get('risk') or {}
        fund = results['symbols'][symbol].get('fundamentals') or {}
        timing = {}
        for tp in TIMING_PERIODS:
            row = results['symbols'][symbol].get(tp)
            if isinstance(row, dict) and row.get('score') is not None:
                timing[tp] = {'score': round(row['score'], 1),
                              'rank': row.get('rank')}
        signals = (sel.get('signals') or [])[:2]
        holdings.append({
            'symbol': symbol,
            'weight': round(weights.get(symbol, 0) * 100, 1),
            'score': round(sel.get('score', 0), 1),
            'rank': sel.get('rank'),
            'timing': timing,
            'sharpe': risk.get('sharpe'),
            'ann_vol': risk.get('ann_vol'),
            'beta': risk.get('beta'),
            'sector': fund.get('sector'),
            'top_signal': signals[0] if signals else None,
        })

    out = {
        'holdings': holdings,
        'max_names': max_names,
        'selection_horizon': SELECT_PERIOD,
        'risk': risk_from_weights(weights, store),
        'changes': {
            'added': sorted(set(selected) - prev_symbols) if prev_symbols else [],
            'dropped': sorted(prev_symbols - set(selected)) if prev_symbols else [],
        },
    }
    if positions:
        out['vs_current'] = _reposition_diff(selected, weights, positions, store)
    return out


def _returns_matrix(store, symbols):
    closes = {}
    for sym in symbols:
        try:
            hist = store.get_history(sym)
        except Exception:
            continue
        if not hist.empty:
            closes[sym] = hist['Close'].astype(float).pct_change()
    if not closes:
        return pd.DataFrame()
    return pd.DataFrame(closes).dropna(how='all').iloc[-RET_WINDOW:]


def _too_correlated(symbol, selected, returns):
    if returns.empty or symbol not in returns.columns:
        return False
    for held in selected:
        if held not in returns.columns:
            continue
        pair = returns[[symbol, held]].dropna()
        if len(pair) >= 40:
            c = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            if pd.notna(c) and c > CORR_CAP:
                return True
    return False


def _inverse_vol_weights(selected, results, returns):
    """1/vol base weights with a mild score tilt, capped per name."""
    raw = {}
    for symbol in selected:
        vol = None
        if not returns.empty and symbol in returns.columns:
            s = returns[symbol].dropna()
            if len(s) >= 20:
                vol = float(s.std())
        if not vol or vol <= 0:
            vol = 0.02
        sel = results['symbols'][symbol].get(SELECT_PERIOD) or {}
        tilt = 1.0 + max(0.0, float(sel.get('score') or 0)) / 200.0
        raw[symbol] = tilt / vol

    total = sum(raw.values())
    weights = {s: v / total for s, v in raw.items()}
    # cap and redistribute; a few passes converge fine
    for _ in range(5):
        excess = sum(w - WEIGHT_CAP for w in weights.values() if w > WEIGHT_CAP)
        if excess <= 1e-9:
            break
        under = {s: w for s, w in weights.items() if w < WEIGHT_CAP}
        under_total = sum(under.values())
        for s, w in weights.items():
            if w > WEIGHT_CAP:
                weights[s] = WEIGHT_CAP
            elif under_total > 0:
                weights[s] = w + excess * (w / under_total)
    return weights


def _reposition_diff(selected, weights, positions, store):
    """Compare the recommendation with what the user actually holds."""
    current = {}
    for pos in positions or []:
        symbol = pos.get('symbol')
        shares = float(pos.get('shares') or 0)
        if not symbol or shares <= 0:
            continue
        try:
            hist = store.get_history(symbol)
        except Exception:
            continue
        if not hist.empty:
            current[symbol] = shares * float(hist['Close'].iloc[-1])
    total = sum(current.values())
    current_w = {s: v / total for s, v in current.items()} if total > 0 else {}

    add = [{'symbol': s, 'weight': round(weights[s] * 100, 1)}
           for s in selected if s not in current_w]
    review = [{'symbol': s, 'current_weight': round(w * 100, 1)}
              for s, w in sorted(current_w.items(), key=lambda kv: -kv[1])
              if s not in selected]
    return {
        'not_held': sorted(add, key=lambda d: -d['weight']),
        'held_not_recommended': review,
    }
