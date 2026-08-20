"""
Market-level regime overlay: SPY vs its 200-day average plus the VIX level
relative to its own history. Used to shade conviction on the whole board,
not to change cross-sectional ranks (a common scaling leaves ordering
unchanged).

exposure_scalar() turns the same two inputs into a gross-exposure target.
The distinction matters and is the one the literature is clearest about:

- Moreira and Muir (JF 2017) made the case for volatility-managed
  portfolios, but Cederburg, O'Doherty, Wang and Yan (JFE 2020) tested 103
  strategies and found real-time volatility timing generally earns a LOWER
  Sharpe than leaving the position alone. DeMiguel, Martin-Utrera and Uppal
  (JF 2024) found the multifactor version does survive out of sample, so
  the gate belongs at the portfolio level, not on individual factors.
- Daniel and Moskowitz (JFE 2016) show momentum crashes cluster in bear
  markets with high volatility - exactly the SPY-below-200dma-with-hot-VIX
  state - and that scaling exposure there roughly doubles Sharpe.
- Goulding, Harvey and Mazzoleni (JFE 2023) reach the same place from trend
  signals: when slow and fast trend both point down, expected momentum
  return is negative.

So this is drawdown control, not alpha. In a rising sample it costs return;
it is meant to pay in the regime this project's three-year backtest does not
contain. Measure it with backtest.py --gate before trusting it.
"""

import logging

logger = logging.getLogger('stock_app.market')

# Gross exposure by risk state. Cash is the remainder; the model never
# shorts and never levers.
EXPOSURE = {'on': 1.0, 'neutral': 0.6, 'off': 0.3, 'unknown': 1.0}


def classify_risk(spy_above_200dma, vix_percentile, vix_hot_pctile=80):
    """The risk state and its rationale, from the two overlay inputs.

    Kept separate from market_overlay() so the backtester can call it at a
    historical bar with point-in-time values.
    """
    if spy_above_200dma is None:
        return 'unknown', ''
    vix_hot = vix_percentile is not None and vix_percentile >= vix_hot_pctile
    if spy_above_200dma and not vix_hot:
        return 'on', 'SPY above 200dma, volatility contained'
    if not spy_above_200dma and vix_hot:
        return 'off', 'SPY below 200dma with elevated VIX - defensive posture'
    if not spy_above_200dma:
        return 'neutral', 'SPY below 200dma - trend caution'
    return 'neutral', 'VIX elevated - volatility caution'


def exposure_scalar(risk):
    """Target gross exposure for a risk state, in [0, 1]."""
    return EXPOSURE.get(risk, 1.0)


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

    risk, note = classify_risk(out['spy_above_200dma'], out['vix_percentile'])
    out['risk'] = risk
    out['note'] = note
    out['score_multiplier'] = {'on': 1.0, 'neutral': 0.95,
                               'off': 0.85}.get(risk, 1.0)
    out['exposure'] = exposure_scalar(risk)
    return out
