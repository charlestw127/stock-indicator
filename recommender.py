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
              positions=None, base_value=None, market=None):
    """Build the recommended portfolio.

    results: a scan results dict (results['symbols'][sym][period])
    prev_symbols: previously recommended symbols, for hysteresis
    positions: the user's current positions, for the rebalance plan
    base_value: dollar base to size targets; defaults to the market value
        of the current positions
    market: the market overlay dict; its 'exposure' scales how much of the
        base is invested, with the remainder held as cash
    Returns a dict for the API, or None if there is nothing to recommend.
    """
    max_names = max(1, min(MAX_NAMES, int(max_names or MAX_NAMES)))
    prev_symbols = set(prev_symbols or [])
    exposure = 1.0
    if isinstance(market, dict):
        try:
            exposure = float(market.get('exposure', 1.0))
        except (TypeError, ValueError):
            exposure = 1.0
    exposure = min(1.0, max(0.0, exposure))

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

    prices = _latest_prices(store, selected)
    current_shares, current_values = _current_book(positions, store)
    total_value = sum(current_values.values())
    try:
        override = float(base_value) if base_value else None
    except (TypeError, ValueError):
        override = None
    if override and override > 0:
        base, from_portfolio = override, False
    else:
        base = total_value if total_value > 0 else None
        from_portfolio = True

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
        price = prices.get(symbol)
        weight = weights.get(symbol, 0)
        row = {
            'symbol': symbol,
            'weight': round(weight * 100, 1),
            'price': round(price, 2) if price else None,
            'score': round(sel.get('score', 0), 1),
            'rank': sel.get('rank'),
            'timing': timing,
            'sharpe': risk.get('sharpe'),
            'ann_vol': risk.get('ann_vol'),
            'beta': risk.get('beta'),
            'sector': fund.get('sector'),
            'top_signal': signals[0] if signals else None,
        }
        if base and price:
            row['target_value'] = round(weight * base * exposure, 2)
            row['target_shares'] = round(weight * base * exposure / price, 3)
        holdings.append(row)

    out = {
        'holdings': holdings,
        'max_names': max_names,
        'selection_horizon': SELECT_PERIOD,
        'risk': risk_from_weights(weights, store),
        'changes': {
            'added': sorted(set(selected) - prev_symbols) if prev_symbols else [],
            'dropped': sorted(prev_symbols - set(selected)) if prev_symbols else [],
        },
        'exposure': round(exposure, 3),
        'cash_weight': round((1.0 - exposure) * 100, 1),
    }
    if exposure < 1.0:
        out['exposure_note'] = (
            f"{out['cash_weight']:g}% held in cash: "
            f"{(market or {}).get('note') or 'defensive market regime'}. "
            "Weights below are shares of the invested book. This is drawdown "
            "control, not a forecast - it costs return in a rising tape."
        )
    if base:
        out['rebalance'] = _rebalance_plan(
            selected, weights, base * exposure, current_shares,
            current_values, prices, store, from_portfolio)
        out['rebalance']['exposure'] = round(exposure, 3)
        out['rebalance']['gross_base_value'] = round(base, 2)
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
    # the cap cannot be tighter than an equal split or weight evaporates
    cap = max(WEIGHT_CAP, 1.0 / len(raw))
    # cap and redistribute; a few passes converge fine
    for _ in range(5):
        excess = sum(w - cap for w in weights.values() if w > cap)
        if excess <= 1e-9:
            break
        under = {s: w for s, w in weights.items() if w < cap}
        under_total = sum(under.values())
        for s, w in weights.items():
            if w > cap:
                weights[s] = cap
            elif under_total > 0:
                weights[s] = w + excess * (w / under_total)
    total = sum(weights.values())
    return {s: w / total for s, w in weights.items()}


def _latest_prices(store, symbols):
    prices = {}
    for sym in symbols:
        try:
            hist = store.get_history(sym)
        except Exception:
            continue
        if not hist.empty:
            prices[sym] = float(hist['Close'].iloc[-1])
    return prices


def _current_book(positions, store):
    """Current holdings as ({symbol: shares}, {symbol: market value})."""
    shares_map, values = {}, {}
    for pos in positions or []:
        symbol = pos.get('symbol')
        shares = float(pos.get('shares') or 0)
        if not symbol or shares <= 0:
            continue
        price = _latest_prices(store, [symbol]).get(symbol)
        if price:
            shares_map[symbol] = shares
            values[symbol] = shares * price
    return shares_map, values


def _rebalance_plan(selected, weights, base, current_shares, current_values,
                    prices, store, from_portfolio):
    """Concrete trades to move the current book to the target weights.

    Trades smaller than 1% of the base (or $25) are skipped - matching the
    anti-churn stance of the selection itself. A full exit uses the exact
    held share count instead of a price-derived estimate.
    """
    all_syms = set(selected) | set(current_values)
    prices = dict(prices)
    missing = [s for s in all_syms if s not in prices]
    prices.update(_latest_prices(store, missing))

    min_trade = max(base * 0.01, 25.0)
    trades = []
    buy_total = sell_total = 0.0
    for sym in all_syms:
        cur_v = current_values.get(sym, 0.0)
        tgt_w = weights.get(sym, 0.0) if sym in selected else 0.0
        tgt_v = tgt_w * base
        delta = tgt_v - cur_v
        if abs(delta) < min_trade:
            continue
        price = prices.get(sym)
        if tgt_v == 0 and sym in current_shares:
            delta_shares = -current_shares[sym]
        elif price:
            delta_shares = delta / price
        else:
            delta_shares = None
        if delta > 0:
            buy_total += delta
        else:
            sell_total += -delta
        trades.append({
            'symbol': sym,
            'action': 'buy' if delta > 0 else 'sell',
            'price': round(price, 2) if price else None,
            'current_value': round(cur_v, 2),
            'target_value': round(tgt_v, 2),
            'delta_value': round(delta, 2),
            'delta_shares': round(delta_shares, 3) if delta_shares is not None else None,
            'current_weight': round(cur_v / base * 100, 1),
            'target_weight': round(tgt_w * 100, 1),
        })
    trades.sort(key=lambda t: -abs(t['delta_value']))
    return {
        'base_value': round(base, 2),
        'from_portfolio': from_portfolio,
        'min_trade': round(min_trade, 2),
        'buy_total': round(buy_total, 2),
        'sell_total': round(sell_total, 2),
        'net_cash_needed': round(buy_total - sell_total, 2),
        'trades': trades,
    }
