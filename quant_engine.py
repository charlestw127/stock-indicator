"""
Multi-factor scoring engine.

Indicators are grouped into five sleeves (trend, momentum, mean_reversion,
volume_flow, quality), each scored in [-1, 1]. A regime classifier based on
the Hurst exponent, variance ratio, ADX and Kaufman efficiency ratio decides
how much weight each sleeve gets: momentum is weighted up in trending tape,
reversion signals in choppy tape. Scores are haircut when volatility is in
its top decile.

SymbolAnalyzer computes all series once per symbol and can evaluate any
horizon at any point in the history, which is what the backtester uses.
Everything is causal (rolling/ewm/cumsum), so evaluating at bar i only sees
data up to bar i.
"""

import json
import os

import numpy as np
import pandas as pd

import quant_indicators as qi
from indicators import (
    calculate_rsi, calculate_bbp, calculate_macd,
    calculate_stochastic, calculate_adx, calculate_psar,
)

# Lookbacks roughly match the holding period implied by each UI column.
# The optional 'tilt' multiplies the regime weights before renormalizing:
# measured ICs on this universe (results/ic_by_horizon.json) show mean
# reversion predicts at short horizons and momentum at long ones, so the
# tactical columns lean contrarian and the long columns lean momentum.
# Calibrated in-sample - treat the tilts as informed defaults, not truth.
HORIZON_PARAMS = {
    '1d': {'mom_lb': 3,   'fast': 3,  'slow': 10,  'z_win': 5,
           'vol_win': 5,   'rsi': 3,  'linreg_win': 10,  'risk_win': 21,
           'tilt': {'mean_reversion': 1.5, 'momentum': 0.7}},
    '1w': {'mom_lb': 5,   'fast': 5,  'slow': 20,  'z_win': 10,
           'vol_win': 10,  'rsi': 7,  'linreg_win': 21,  'risk_win': 63,
           'tilt': {'mean_reversion': 1.2}},
    '1m': {'mom_lb': 21,  'fast': 10, 'slow': 50,  'z_win': 20,
           'vol_win': 21,  'rsi': 14, 'linreg_win': 63,  'risk_win': 126},
    '6m': {'mom_lb': 126, 'fast': 20, 'slow': 100, 'z_win': 50,
           'vol_win': 63,  'rsi': 14, 'linreg_win': 126, 'risk_win': 252,
           'tilt': {'momentum': 1.3, 'mean_reversion': 0.6}},
    '1y': {'mom_lb': 252, 'fast': 50, 'slow': 200, 'z_win': 100,
           'vol_win': 126, 'rsi': 14, 'linreg_win': 252, 'risk_win': 252,
           'use_12_1': True,
           'tilt': {'momentum': 1.3, 'mean_reversion': 0.6}},
}

# The hand-set prior. Fifteen numbers chosen by judgement, never validated.
# regime_calibrate.py estimates a shrunk version of this table from the data
# and writes it to results/regime_weights.json; load_calibrated_weights()
# picks it up if it is there. Until then this is what runs, and the README's
# caveat about it stands.
REGIME_WEIGHTS = {
    'trending':       {'trend': 0.32, 'momentum': 0.28, 'mean_reversion': 0.08,
                       'volume_flow': 0.17, 'quality': 0.15},
    'mean-reverting': {'trend': 0.10, 'momentum': 0.12, 'mean_reversion': 0.38,
                       'volume_flow': 0.20, 'quality': 0.20},
    'mixed':          {'trend': 0.22, 'momentum': 0.20, 'mean_reversion': 0.20,
                       'volume_flow': 0.18, 'quality': 0.20},
}

# What score_at actually uses. Swapped wholesale rather than mutated so a
# backtest can restore the prior and compare like for like.
ACTIVE_WEIGHTS = {k: dict(v) for k, v in REGIME_WEIGHTS.items()}

CALIBRATED_WEIGHTS_PATH = os.path.join('results', 'regime_weights.json')

MIN_BARS = 30


def set_regime_weights(weights=None):
    """Install a weight table (None restores the hand-set prior)."""
    global ACTIVE_WEIGHTS
    if weights is None:
        ACTIVE_WEIGHTS = {k: dict(v) for k, v in REGIME_WEIGHTS.items()}
    else:
        ACTIVE_WEIGHTS = {k: dict(v) for k, v in weights.items()}
    return ACTIVE_WEIGHTS


def load_calibrated_weights(path=CALIBRATED_WEIGHTS_PATH, quiet=True):
    """Install calibrated per-state weights if the file exists.

    Returns the table in use. Missing or malformed files leave the prior in
    place rather than failing the scan - a dashboard that cannot start
    because a calibration artifact is stale is worse than one running on the
    documented default.
    """
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return ACTIVE_WEIGHTS
    table = payload.get('weights') if isinstance(payload, dict) else None
    if not isinstance(table, dict):
        return ACTIVE_WEIGHTS
    needed = set(REGIME_WEIGHTS['mixed'])
    if not all(isinstance(v, dict) and needed <= set(v) for v in table.values()):
        if not quiet:
            print(f"{path}: unexpected shape, keeping the hand-set prior")
        return ACTIVE_WEIGHTS
    return set_regime_weights(table)


def _val(series, i):
    """series.iloc[i] as a finite float, else None."""
    if series is None or i >= len(series) or i < 0:
        return None
    v = series.iloc[i]
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _clean(v):
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _avg(components):
    vals = [c for c in components if c is not None and np.isfinite(c)]
    if not vals:
        return 0.0
    return float(np.clip(np.mean(vals), -1.0, 1.0))


def classify_trend_regime(hurst, variance_ratio, adx, eff_ratio):
    evidence = 0.0
    if hurst is not None:
        evidence += 2.5 * (hurst - 0.5)
    if adx is not None:
        evidence += (adx - 22.0) / 25.0
    if eff_ratio is not None:
        evidence += (eff_ratio - 0.30)
    if variance_ratio is not None:
        evidence += 0.5 * (variance_ratio - 1.0)

    if evidence > 0.40:
        return 'trending', evidence
    if evidence < -0.35:
        return 'mean-reverting', evidence
    return 'mixed', evidence


def score_to_recommendation(score):
    """Standalone label from the composite; cross-sectional ranking
    overrides this when a universe is analyzed."""
    if score >= 40:
        return 'BUY (Strong)'
    if score >= 15:
        return 'BUY (Weak)'
    if score > -15:
        return 'HOLD'
    if score > -40:
        return 'SELL (Weak)'
    return 'SELL (Strong)'


class SymbolAnalyzer:
    """Per-symbol analysis over a daily OHLCV history."""

    def __init__(self, df, benchmark_close=None):
        df = df.dropna(subset=['Close'])
        if len(df) < MIN_BARS:
            raise ValueError(f"not enough history ({len(df)} bars)")

        self.df = df
        self.index = df.index
        self.n = len(df)
        self.benchmark_close = benchmark_close

        c = df['Close'].astype(float)
        self.close = c
        self.high = df['High'].astype(float) if 'High' in df else c
        self.low = df['Low'].astype(float) if 'Low' in df else c
        self.open = df['Open'].astype(float) if 'Open' in df else c
        self.volume = df['Volume'].astype(float) if 'Volume' in df \
            else pd.Series(1.0, index=c.index)

        h, l, v = self.high, self.low, self.volume
        self.atr14 = qi.atr(h, l, c, 14)
        self.rv21 = qi.realized_volatility(c, 21)
        self.kama10 = qi.kama(c, 10)
        self.mfi14 = qi.money_flow_index(h, l, c, v, 14)
        self.cmf20 = qi.chaikin_money_flow(h, l, c, v, 20)
        self.obv_z = qi.obv_zscore(c, v, 20)
        self.vwap_dev = qi.vwap_deviation(h, l, c, v, 20)
        self.rel_vol = qi.relative_volume(v, 20)
        self.don_pos = qi.donchian_position(h, l, c, 20)
        self.don_break = qi.donchian_breakout(h, l, c, 20)
        self.squeeze_on, self.squeeze_fired = qi.ttm_squeeze(h, l, c)
        self.dist_52 = qi.dist_from_52wk_high(c)
        self.mom_12_1 = qi.momentum_12_1(c)
        self.macd_hist = calculate_macd(c)
        self.bbp20 = calculate_bbp(c, 20)
        self.stoch_k, self.stoch_d = calculate_stochastic(h, l, c)
        self.adx14 = calculate_adx(h, l, c, 14)

        self._horizon_cache = {}

    def _horizon(self, period):
        """Horizon-dependent series, computed once per period."""
        if period in self._horizon_cache:
            return self._horizon_cache[period]
        p = HORIZON_PARAMS.get(period, HORIZON_PARAMS['6m'])
        c = self.close
        h = {
            'params': p,
            'rsi': calculate_rsi(c, p['rsi']),
            'sma_fast': c.rolling(p['fast'], min_periods=p['fast']).mean(),
            'sma_slow': c.rolling(p['slow'], min_periods=max(10, p['slow'] // 2)).mean(),
            'zscore': qi.price_zscore(c, p['z_win']),
            'mom': qi.total_return_momentum(c, p['mom_lb']),
            'mom_tstat': qi.risk_adjusted_momentum(c, p['mom_lb']),
            'er': qi.efficiency_ratio(c, max(10, p['fast'])),
            'horizon_vol': qi.realized_volatility(c, p['vol_win']),
        }
        self._horizon_cache[period] = h
        return h

    def _vol_percentile_at(self, i):
        window = self.rv21.iloc[max(0, i - 251):i + 1].dropna()
        cur = _val(self.rv21, i)
        if cur is None or len(window) < 21:
            return None
        return float((window <= cur).mean() * 100.0)

    def score_at(self, period, i=None):
        """Factors, regime and composite score at bar i (default: latest).
        This is the fast path used by the backtester."""
        if i is None:
            i = self.n - 1
        hz = self._horizon(period)
        p = hz['params']
        close_to_i = self.close.iloc[:i + 1]
        price = _val(self.close, i)

        hurst = qi.hurst_exponent(close_to_i, window=min(i + 1, 504))
        var_ratio = qi.variance_ratio(close_to_i, q=5, window=min(i + 1, 252))
        vol_pctile = self._vol_percentile_at(i)
        eff_ratio = _val(hz['er'], i)
        adx = _val(self.adx14, i)

        trend_regime, evidence = classify_trend_regime(hurst, var_ratio, adx, eff_ratio)
        if vol_pctile is None:
            vol_regime = 'unknown'
        elif vol_pctile >= 80:
            vol_regime = 'high'
        elif vol_pctile <= 20:
            vol_regime = 'low'
        else:
            vol_regime = 'normal'

        # trend sleeve
        sma_fast, sma_slow = _val(hz['sma_fast'], i), _val(hz['sma_slow'], i)
        ma_score = np.tanh(15.0 * (sma_fast / sma_slow - 1.0)) \
            if sma_fast and sma_slow else None
        kama = _val(self.kama10, i)
        kama_score = np.tanh(10.0 * (price / kama - 1.0)) if price and kama else None
        linreg_slope, linreg_r2 = qi.linreg_trend(close_to_i, p['linreg_win'])
        linreg_score = np.tanh(linreg_slope / 40.0) * linreg_r2 \
            if linreg_slope is not None else None
        dpos = _val(self.don_pos, i)
        trend = _avg([ma_score, kama_score, linreg_score,
                      dpos * 0.8 if dpos is not None else None])

        # momentum sleeve
        mom = _val(hz['mom'], i)
        tstat = _val(hz['mom_tstat'], i)
        macd, atr = _val(self.macd_hist, i), _val(self.atr14, i)
        macd_atr = np.tanh(2.0 * macd / atr) if macd is not None and atr else None
        d52 = _val(self.dist_52, i)
        comps = [
            np.tanh(tstat / 2.5) if tstat is not None else None,
            np.tanh(mom * 3.0 * np.sqrt(252.0 / p['mom_lb'])) if mom is not None else None,
            macd_atr,
            np.clip(1.0 + d52 / 25.0, -1.0, 1.0) if d52 is not None else None,
        ]
        if p.get('use_12_1'):
            m121 = _val(self.mom_12_1, i)
            if m121 is not None:
                comps.append(np.tanh(m121 * 2.0))
        momentum = _avg(comps)

        # mean reversion sleeve (contrarian: oversold scores positive)
        z = _val(hz['zscore'], i)
        rsi = _val(hz['rsi'], i)
        bbp = _val(self.bbp20, i)
        sk, sd = _val(self.stoch_k, i), _val(self.stoch_d, i)
        mean_reversion = _avg([
            -np.tanh(z / 2.0) if z is not None else None,
            np.clip(-(rsi - 50.0) / 50.0, -1, 1) if rsi is not None else None,
            np.clip(-(bbp - 0.5) * 2.0, -1, 1) if bbp is not None else None,
            np.clip(-((sk + sd) / 2.0 - 50.0) / 50.0, -1, 1)
            if sk is not None and sd is not None else None,
        ])

        # volume / flow sleeve
        mfi = _val(self.mfi14, i)
        cmf = _val(self.cmf20, i)
        obv_z = _val(self.obv_z, i)
        vdev = _val(self.vwap_dev, i)
        atr_pct = 100.0 * atr / price if atr and price else None
        volume_flow = _avg([
            np.clip((mfi - 50.0) / 50.0, -1, 1) if mfi is not None else None,
            np.tanh(cmf * 4.0) if cmf is not None else None,
            np.tanh(obv_z / 2.0) if obv_z is not None else None,
            np.tanh(1.5 * vdev / (atr_pct or 2.0)) if vdev is not None else None,
        ])

        # quality sleeve
        risk_win = min(p['risk_win'], i + 1)
        sharpe = qi.sharpe_ratio(close_to_i, risk_win)
        mdd = qi.max_drawdown(close_to_i, risk_win)
        quality = _avg([
            np.tanh(sharpe / 1.5) if sharpe is not None else None,
            np.clip(1.0 + mdd / 25.0, -1, 1) if mdd is not None else None,
            (50.0 - vol_pctile) / 100.0 if vol_pctile is not None else None,
        ])

        factors = {
            'trend': round(trend, 3),
            'momentum': round(momentum, 3),
            'mean_reversion': round(mean_reversion, 3),
            'volume_flow': round(volume_flow, 3),
            'quality': round(quality, 3),
        }

        weights = ACTIVE_WEIGHTS[trend_regime]
        tilt = p.get('tilt')
        if tilt:
            weights = {k: w * tilt.get(k, 1.0) for k, w in weights.items()}
            total = sum(weights.values())
            weights = {k: w / total for k, w in weights.items()}
        score = 100.0 * sum(weights[k] * factors[k] for k in weights)
        if vol_pctile is not None and vol_pctile >= 85:
            score *= 0.8
        score = float(np.clip(score, -100.0, 100.0))

        agree = 0.0
        for k, w in weights.items():
            f = factors[k]
            if abs(f) < 0.05:
                agree += 0.5 * w
            elif np.sign(f) == np.sign(score) or score == 0:
                agree += w
        confidence = int(round(agree * 100))

        return {
            'score': score,
            'confidence': confidence,
            'factors': factors,
            'regime': {
                'trend': trend_regime,
                'volatility': vol_regime,
                'evidence': _clean(round(evidence, 2)),
                'hurst': _clean(hurst),
                'variance_ratio': _clean(var_ratio),
                'efficiency_ratio': _clean(eff_ratio),
                'vol_percentile': _clean(vol_pctile),
            },
            # internals reused by evaluate()
            '_aux': {
                'rsi': rsi, 'zscore': z, 'sharpe': sharpe, 'mdd': mdd,
                'macd_atr': macd_atr, 'linreg_slope': linreg_slope,
                'linreg_r2': linreg_r2, 'mom': mom, 'mom_tstat': tstat,
                'atr_pct': atr_pct, 'risk_win': risk_win, 'params': p,
            },
        }

    def evaluate(self, period):
        """Full result for the latest bar: score plus risk metrics, signals
        and the indicator snapshot the UI shows."""
        i = self.n - 1
        base = self.score_at(period, i)
        aux = base.pop('_aux')
        p = aux['params']
        c = self.close
        price = _val(c, i)

        half_life = qi.half_life_mean_reversion(c, window=min(self.n, 126))
        skew, kurt = qi.return_skew_kurtosis(c, window=min(self.n, 252))
        sortino = qi.sortino_ratio(c, aux['risk_win'])
        calmar = qi.calmar_ratio(c, aux['risk_win'])
        var95, cvar95 = qi.var_cvar(c, min(aux['risk_win'], 252))
        beta, alpha = qi.beta_alpha(c, self.benchmark_close, min(aux['risk_win'], 252))
        horizon_vol = _val(self._horizon(period)['horizon_vol'], i)

        base['regime']['half_life_days'] = _clean(half_life)
        base['risk'] = {
            'sharpe': _clean(aux['sharpe']),
            'sortino': _clean(sortino),
            'calmar': _clean(calmar),
            'max_drawdown': _clean(aux['mdd']),
            'var_95': _clean(var95),
            'cvar_95': _clean(cvar95),
            'beta': _clean(beta),
            'alpha': _clean(alpha),
            'ann_vol': _clean(horizon_vol),
            'skew': _clean(skew),
            'kurtosis': _clean(kurt),
        }
        base['signals'] = self._signals(period, base, aux, half_life, beta)

        # PSAR is loop-based, so only run it on the reported tail
        psar = calculate_psar(self.high.iloc[-252:], self.low.iloc[-252:], c.iloc[-252:])
        obv_z = _val(self.obv_z, i)
        sma_fast = _val(self._horizon(period)['sma_fast'], i)
        sma_slow = _val(self._horizon(period)['sma_slow'], i)
        ma_score = np.tanh(15.0 * (sma_fast / sma_slow - 1.0)) \
            if sma_fast and sma_slow else None

        base['indicators'] = {
            'rsi': _clean(aux['rsi']),
            'bbp': _clean(_val(self.bbp20, i)),
            'macd': _clean(_val(self.macd_hist, i)),
            'stoch_k': _clean(_val(self.stoch_k, i)),
            'stoch_d': _clean(_val(self.stoch_d, i)),
            'adx': _clean(_val(self.adx14, i)),
            'obv': _clean(50.0 + 25.0 * np.tanh(obv_z / 2.0) if obv_z is not None else None),
            'ma_crossover': _clean(ma_score),
            'psar': _clean(_val(psar, len(psar) - 1)),
            'zscore': _clean(aux['zscore']),
            'atr_pct': _clean(aux['atr_pct']),
            'realized_vol': _clean(_val(self.rv21, i)),
            'momentum': _clean(aux['mom']),
            'momentum_tstat': _clean(aux['mom_tstat']),
            'momentum_12_1': _clean(_val(self.mom_12_1, i)),
            'kama': _clean(_val(self.kama10, i)),
            'linreg_slope': _clean(aux['linreg_slope']),
            'linreg_r2': _clean(aux['linreg_r2']),
            'dist_52wk_high': _clean(_val(self.dist_52, i)),
            'mfi': _clean(_val(self.mfi14, i)),
            'cmf': _clean(_val(self.cmf20, i)),
            'obv_zscore': _clean(obv_z),
            'vwap_dev': _clean(_val(self.vwap_dev, i)),
            'rel_volume': _clean(_val(self.rel_vol, i)),
            'donchian_pos': _clean(_val(self.don_pos, i)),
            'donchian_breakout': _clean(_val(self.don_break, i)),
            'squeeze_on': bool(self.squeeze_on.iloc[i]),
        }

        base['lastDate'] = self.index[i].strftime('%Y-%m-%d')
        base['lastPrice'] = _clean(price)
        base['recommendation'] = score_to_recommendation(base['score'])
        return base

    def _signals(self, period, base, aux, half_life, beta):
        """Readable event/state flags, most actionable first."""
        i = self.n - 1
        out = []
        p = aux['params']
        regime = base['regime']

        if bool(self.squeeze_fired.iloc[max(0, i - 2):i + 1].any()):
            direction = 'long' if (aux['macd_atr'] or 0) >= 0 else 'short'
            out.append(f"TTM squeeze fired {direction} - volatility expansion underway")
        elif bool(self.squeeze_on.iloc[i]):
            out.append("TTM squeeze on - volatility compressed, breakout loading")

        dbreak = _val(self.don_break, i)
        if dbreak == 1:
            out.append("20d Donchian breakout up (CTA trend entry)")
        elif dbreak == -1:
            out.append("20d Donchian breakdown (CTA trend exit/short)")

        z = aux['zscore']
        if z is not None:
            if z <= -2:
                out.append(f"Stretched {abs(z):.1f} sigma below mean - reversion setup")
            elif z >= 2:
                out.append(f"Stretched {z:.1f} sigma above mean - extended")

        rsi = aux['rsi']
        if rsi is not None:
            if rsi <= 30:
                out.append(f"RSI({p['rsi']}) oversold at {rsi:.0f}")
            elif rsi >= 70:
                out.append(f"RSI({p['rsi']}) overbought at {rsi:.0f}")

        d52 = _val(self.dist_52, i)
        if d52 is not None:
            if d52 >= -1:
                out.append("At 52-week high (momentum anomaly: strength persists)")
            elif d52 >= -5:
                out.append(f"Within {abs(d52):.1f}% of 52-week high")
            elif d52 <= -40:
                out.append(f"{abs(d52):.0f}% below 52-week high - deep drawdown name")

        tail = self.macd_hist.dropna().iloc[-3:]
        if len(tail) == 3:
            if tail.iloc[-1] > 0 and (tail < 0).any():
                out.append("MACD histogram flipped positive")
            elif tail.iloc[-1] < 0 and (tail > 0).any():
                out.append("MACD histogram flipped negative")

        if regime['trend'] == 'mean-reverting' and half_life is not None and half_life < 25:
            out.append(f"Fast mean reversion (half-life {half_life:.0f}d) - fade extremes")
        elif regime['trend'] == 'trending' and regime['hurst'] is not None \
                and regime['hurst'] > 0.58:
            out.append(f"Persistent trend regime (Hurst {regime['hurst']:.2f}) - ride momentum")

        vp = regime['vol_percentile']
        if regime['volatility'] == 'high' and vp is not None:
            out.append(f"High-vol regime ({vp:.0f}th pctile) - reduce position size")
        elif regime['volatility'] == 'low' and vp is not None:
            out.append(f"Low-vol regime ({vp:.0f}th pctile) - cheap optionality")

        if beta is not None and beta >= 1.6:
            out.append(f"High beta ({beta:.1f}) - amplifies market moves")

        return out[:5]


def analyze_symbol(df, periods, benchmark_close=None):
    """Analyze one symbol at every requested horizon. Returns {period: result}."""
    analyzer = SymbolAnalyzer(df, benchmark_close)
    return {period: analyzer.evaluate(period) for period in periods}
