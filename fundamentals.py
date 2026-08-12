"""
Fundamentals via yfinance, cached in the data store because Ticker.info is
one slow HTTP call per symbol. The background refresher warms a few symbols
per cycle, so the cache fills gradually.

Fundamental metrics are shown alongside technical scores and used for
signals (rich/cheap vs the universe, upcoming earnings), but stay out of
the composite score: only a current snapshot is available, so they can't be
backtested point-in-time, and blending them in would make the backtest
unrepresentative of the live model.
"""

import datetime as dt
import logging

import numpy as np

logger = logging.getLogger('stock_app.fundamentals')

FIELDS = [
    'sector', 'industry', 'marketCap', 'trailingPE', 'forwardPE',
    'priceToBook', 'returnOnEquity', 'profitMargins', 'debtToEquity',
    'dividendYield', 'trailingEps',
]


def fetch_fundamentals(symbol):
    """One live fetch. Returns a dict (possibly sparse for ETFs)."""
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    out = {}
    try:
        info = ticker.info or {}
        for f in FIELDS:
            v = info.get(f)
            if isinstance(v, (int, float)) and not np.isfinite(v):
                v = None
            out[f] = v
    except Exception as e:
        logger.warning("info fetch failed for %s: %s", symbol, e)

    try:
        cal = ticker.calendar
        dates = None
        if isinstance(cal, dict):
            dates = cal.get('Earnings Date')
        if dates:
            first = dates[0]
            if hasattr(first, 'strftime'):
                out['next_earnings'] = first.strftime('%Y-%m-%d')
    except Exception as e:
        logger.debug("calendar fetch failed for %s: %s", symbol, e)

    return out


def get_fundamentals(store, symbol):
    """Cached fundamentals or None. Never triggers a live fetch - the
    background refresher owns that."""
    return store.get_fundamentals(symbol)


def refresh_some(store, symbols, max_fetches=8):
    """Warm the cache for up to `max_fetches` of the stalest symbols."""
    by_age = sorted(symbols, key=lambda s: -(store.fundamentals_age(s) or 1e12))
    fetched = 0
    for symbol in by_age:
        if fetched >= max_fetches:
            break
        age = store.fundamentals_age(symbol)
        if age is not None and age < 7 * 24 * 3600:
            continue
        try:
            store.save_fundamentals(symbol, fetch_fundamentals(symbol))
            fetched += 1
        except Exception as e:
            logger.warning("fundamentals refresh failed for %s: %s", symbol, e)
    return fetched


def universe_medians(store, symbols):
    """Median PE / PB / ROE across whatever is cached, for rich/cheap context."""
    rows = {'trailingPE': [], 'priceToBook': [], 'returnOnEquity': []}
    for symbol in symbols:
        f = store.get_fundamentals(symbol)
        if not f:
            continue
        for k in rows:
            v = f.get(k)
            if isinstance(v, (int, float)) and 0 < v < 1e4:
                rows[k].append(v)
    return {k: (float(np.median(v)) if v else None) for k, v in rows.items()}


def fundamental_summary(fund, medians=None):
    """Compact display dict + signal strings for one symbol."""
    if not fund:
        return None, []
    summary = {
        'sector': fund.get('sector'),
        'pe': _r(fund.get('trailingPE')),
        'forward_pe': _r(fund.get('forwardPE')),
        'pb': _r(fund.get('priceToBook')),
        'roe': _r(fund.get('returnOnEquity'), 100),
        'margin': _r(fund.get('profitMargins'), 100),
        'dividend_yield': _r(fund.get('dividendYield'), 100),
        'next_earnings': fund.get('next_earnings'),
    }
    signals = []

    ne = fund.get('next_earnings')
    if ne:
        try:
            days = (dt.datetime.strptime(ne, '%Y-%m-%d').date() - dt.date.today()).days
            if 0 <= days <= 7:
                signals.append(f"Earnings in {days}d - expect a volatility event")
        except ValueError:
            pass

    pe = fund.get('trailingPE')
    med_pe = (medians or {}).get('trailingPE')
    if pe and med_pe and pe > 0:
        if pe > 2.0 * med_pe:
            signals.append(f"Rich valuation: PE {pe:.0f} vs universe median {med_pe:.0f}")
        elif pe < 0.5 * med_pe:
            signals.append(f"Cheap valuation: PE {pe:.0f} vs universe median {med_pe:.0f}")

    roe = fund.get('returnOnEquity')
    if roe is not None and roe > 0.30:
        signals.append(f"High ROE ({roe * 100:.0f}%)")

    return summary, signals


def _r(v, scale=1):
    if v is None or not isinstance(v, (int, float)) or not np.isfinite(v):
        return None
    return round(v * scale, 2)
