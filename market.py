"""
Market-level regime overlay: SPY vs its 200-day average plus the VIX level
relative to its own history. Used to shade conviction on the whole board,
not to change cross-sectional ranks (a common scaling leaves ordering
unchanged).
"""

import logging

logger = logging.getLogger('stock_app.market')


def market_overlay(store):
    """Returns a dict describing the current market regime, or a minimal
    dict if the data can't be fetched."""
    out = {
        'risk': 'unknown',
        'spy_above_200dma': None,
        'spy_dist_200dma': None,
        'vix': None,
        'vix_percentile': None,
        'score_multiplier': 1.0,
        'note': '',
    }
    try:
        spy = store.get_history('SPY')
        if len(spy) >= 200:
            close = spy['Close'].astype(float)
            sma200 = close.rolling(200).mean().iloc[-1]
            last = close.iloc[-1]
            out['spy_above_200dma'] = bool(last > sma200)
            out['spy_dist_200dma'] = round(float(100.0 * (last / sma200 - 1.0)), 2)
    except Exception as e:
        logger.warning("SPY overlay failed: %s", e)

    try:
        vix = store.get_history('^VIX')
        if len(vix) >= 252:
            vc = vix['Close'].astype(float)
            last = float(vc.iloc[-1])
            pctile = float((vc.iloc[-1260:] <= last).mean() * 100.0)
            out['vix'] = round(last, 2)
            out['vix_percentile'] = round(pctile, 1)
    except Exception as e:
        logger.warning("VIX overlay failed: %s", e)

    above = out['spy_above_200dma']
    vix_hot = out['vix_percentile'] is not None and out['vix_percentile'] >= 80
    if above is None:
        return out

    if above and not vix_hot:
        out['risk'] = 'on'
        out['note'] = 'SPY above 200dma, volatility contained'
    elif not above and vix_hot:
        out['risk'] = 'off'
        out['score_multiplier'] = 0.85
        out['note'] = 'SPY below 200dma with elevated VIX - defensive posture'
    else:
        out['risk'] = 'neutral'
        out['score_multiplier'] = 0.95
        if not above:
            out['note'] = 'SPY below 200dma - trend caution'
        else:
            out['note'] = 'VIX elevated - volatility caution'
    return out
