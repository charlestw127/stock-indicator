"""
Walk-forward backtest of the scoring engine.

At each rebalance date, every symbol is scored using only data up to that
date (SymbolAnalyzer is causal), symbols are ranked cross-sectionally, and
the top fraction is held equal-weight until the next rebalance. The same
machinery runs each factor sleeve as a standalone strategy so the composite
can be compared against its parts, plus equal-weight and SPY buy-and-hold
benchmarks.

Also computes:
- information coefficients (Spearman rank correlation between scores and
  forward returns) per sleeve
- decile analysis of the composite score
- event studies for discrete signals (Donchian breakouts, squeeze fires,
  z-score stretches, MACD flips)

Usage:
    python backtest.py [--period 1m] [--step 5] [--years 3] [--top 0.2]

Results land in results/backtest_summary.json and results/*.png.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from data_store import DataStore
from quant_engine import SymbolAnalyzer

TRADING_DAYS = 252
MIN_HISTORY = 300  # bars a symbol needs before it enters the universe

STRATEGIES = ['composite', 'trend', 'momentum', 'mean_reversion',
              'volume_flow', 'quality']
EVENT_HORIZONS = [5, 10, 21]


def load_universe(store, symbols, min_bars=MIN_HISTORY):
    """Build SymbolAnalyzers from cached history. Symbols without enough
    bars are skipped and reported."""
    analyzers = {}
    skipped = []
    for symbol in symbols:
        try:
            df = store.get_history(symbol)
        except Exception:
            df = None
        if df is None or len(df) < min_bars:
            skipped.append(symbol)
            continue
        try:
            analyzers[symbol] = SymbolAnalyzer(df)
        except ValueError:
            skipped.append(symbol)
    return analyzers, skipped


def run_backtest(analyzers, spy_close, period='1m', step=5, years=3, top=0.2,
                 weighting='equal'):
    """Core walk-forward loop. Returns the full results dict."""
    calendar = spy_close.index
    end = len(calendar) - step
    start = max(MIN_HISTORY, len(calendar) - int(years * TRADING_DAYS))
    rebalance_positions = list(range(start, end, step))
    if len(rebalance_positions) < 10:
        raise ValueError("not enough history for the requested window")

    dates = [calendar[p] for p in rebalance_positions]
    # position of each date in each symbol's own index (pad = last bar <= date)
    sym_pos = {}
    for symbol, an in analyzers.items():
        sym_pos[symbol] = an.index.get_indexer(dates, method='pad')

    per_date = []  # {date, scores: {sym: {composite, sleeves...}}, fwd: {sym: ret}}
    for k, pos in enumerate(rebalance_positions):
        date = calendar[pos]
        next_date = calendar[pos + step]
        scores, fwd = {}, {}
        for symbol, an in analyzers.items():
            i = sym_pos[symbol][k]
            if i < MIN_HISTORY - 1:
                continue
            # skip stale/delisted series
            if (date - an.index[i]).days > 10:
                continue
            j = an.index.get_indexer([next_date], method='pad')[0]
            if j <= i:
                continue
            res = an.score_at(period, i)
            entry = dict(res['factors'])
            entry['composite'] = res['score']
            vol = an.rv21.iloc[i]
            entry['_vol'] = float(vol) if np.isfinite(vol) else None
            scores[symbol] = entry
            c0, c1 = float(an.close.iloc[i]), float(an.close.iloc[j])
            if c0 > 0:
                fwd[symbol] = c1 / c0 - 1.0
        per_date.append({'date': date, 'scores': scores, 'fwd': fwd})

    spy_rets = []
    for k, pos in enumerate(rebalance_positions):
        c0 = float(spy_close.iloc[pos])
        c1 = float(spy_close.iloc[pos + step])
        spy_rets.append(c1 / c0 - 1.0 if c0 > 0 else 0.0)

    results = {
        'config': {'period': period, 'step': step, 'years': years, 'top': top,
                   'weighting': weighting,
                   'n_symbols': len(analyzers), 'n_rebalances': len(per_date),
                   'start': str(dates[0].date()), 'end': str(dates[-1].date())},
        'strategies': {},
        'ic': {},
        'deciles': None,
        'equity_curves': {},
        'dates': [str(d.date()) for d in dates],
    }

    for strat in STRATEGIES:
        rets, turnover, prev_held = [], [], None
        for snap in per_date:
            usable = [s for s in snap['scores'] if s in snap['fwd']]
            if len(usable) < 5:
                rets.append(0.0)
                continue
            ranked = sorted(usable, key=lambda s: snap['scores'][s][strat],
                            reverse=True)
            held = set(ranked[:max(1, int(len(ranked) * top))])
            w = _holding_weights(held, snap, weighting)
            rets.append(float(sum(snap['fwd'][s] * w[s] for s in held)))
            if prev_held:
                turnover.append(len(held - prev_held) / max(1, len(held)))
            prev_held = held
        results['strategies'][strat] = _strategy_stats(rets, step)
        results['strategies'][strat]['avg_turnover'] = \
            round(float(np.mean(turnover)), 3) if turnover else None
        results['equity_curves'][strat] = _equity(rets)

    # benchmarks
    ew_rets = [float(np.mean(list(s['fwd'].values()))) if s['fwd'] else 0.0
               for s in per_date]
    results['strategies']['equal_weight'] = _strategy_stats(ew_rets, step)
    results['strategies']['spy'] = _strategy_stats(spy_rets, step)
    results['equity_curves']['equal_weight'] = _equity(ew_rets)
    results['equity_curves']['spy'] = _equity(spy_rets)

    # information coefficients
    for strat in STRATEGIES:
        ics = []
        for snap in per_date:
            usable = [s for s in snap['scores'] if s in snap['fwd']]
            if len(usable) < 8:
                continue
            x = pd.Series([snap['scores'][s][strat] for s in usable]).rank()
            y = pd.Series([snap['fwd'][s] for s in usable]).rank()
            if x.std() == 0 or y.std() == 0:
                continue
            ics.append(float(np.corrcoef(x, y)[0, 1]))
        results['ic'][strat] = _ic_stats(ics)

    results['deciles'] = _decile_table(per_date, step)
    return results


def _holding_weights(held, snap, weighting):
    """Weights for the held basket: equal, or inverse-volatility capped at
    15% per name (the recommender's scheme)."""
    if weighting != 'inv_vol' or len(held) <= 1:
        return {s: 1.0 / len(held) for s in held}
    raw = {}
    for s in held:
        vol = snap['scores'][s].get('_vol')
        raw[s] = 1.0 / max(vol if vol else 20.0, 1e-6)
    total = sum(raw.values())
    weights = {s: v / total for s, v in raw.items()}
    cap = max(0.15, 1.0 / len(raw))
    for _ in range(5):
        excess = sum(w - cap for w in weights.values() if w > cap)
        if excess <= 1e-9:
            break
        under_total = sum(w for w in weights.values() if w < cap)
        for s, w in weights.items():
            if w > cap:
                weights[s] = cap
            elif under_total > 0:
                weights[s] = w + excess * (w / under_total)
    total = sum(weights.values())
    return {s: w / total for s, w in weights.items()}


def _strategy_stats(rets, step):
    rets = np.asarray(rets, dtype=float)
    eq = np.cumprod(1.0 + rets)
    periods_per_year = TRADING_DAYS / step
    years = len(rets) / periods_per_year
    cagr = (eq[-1] ** (1.0 / years) - 1.0) * 100.0 if years > 0 and eq[-1] > 0 else None
    vol = float(np.std(rets) * np.sqrt(periods_per_year) * 100.0)
    sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(periods_per_year)) \
        if np.std(rets) > 0 else None
    running_max = np.maximum.accumulate(eq)
    max_dd = float((eq / running_max - 1.0).min() * 100.0)
    return {
        'cagr': round(cagr, 2) if cagr is not None else None,
        'ann_vol': round(vol, 2),
        'sharpe': round(sharpe, 2) if sharpe is not None else None,
        'max_drawdown': round(max_dd, 2),
        'hit_rate': round(float((rets > 0).mean()) * 100.0, 1),
        'total_return': round((eq[-1] - 1.0) * 100.0, 1),
    }


def _ic_stats(ics):
    if not ics:
        return None
    ics = np.asarray(ics)
    mean, std = float(ics.mean()), float(ics.std())
    return {
        'mean': round(mean, 4),
        'std': round(std, 4),
        'ir': round(mean / std, 3) if std > 0 else None,
        't_stat': round(mean / std * np.sqrt(len(ics)), 2) if std > 0 else None,
        'pct_positive': round(float((ics > 0).mean()) * 100.0, 1),
        'n': len(ics),
    }


def _decile_table(per_date, step):
    buckets = {d: [] for d in range(1, 11)}
    for snap in per_date:
        usable = [s for s in snap['scores'] if s in snap['fwd']]
        if len(usable) < 10:
            continue
        ranked = sorted(usable, key=lambda s: snap['scores'][s]['composite'])
        n = len(ranked)
        for idx, s in enumerate(ranked):
            decile = min(10, int(idx / n * 10) + 1)  # 10 = best scores
            buckets[decile].append(snap['fwd'][s])
    periods_per_year = TRADING_DAYS / step
    out = {}
    for d, rets in buckets.items():
        if rets:
            ann = (np.mean(rets)) * periods_per_year * 100.0
            out[d] = {'ann_return': round(float(ann), 2), 'n': len(rets)}
    return out


def _equity(rets):
    return [round(float(x), 4) for x in np.cumprod(1.0 + np.asarray(rets))]


# -- event studies ------------------------------------------------------

def event_studies(analyzers, years=3):
    """Mean forward return after each discrete signal vs the unconditional
    baseline over the same window."""
    events = {
        'donchian_breakout_up': [], 'donchian_breakout_down': [],
        'squeeze_fired': [], 'zscore_below_-2': [], 'zscore_above_2': [],
        'macd_flip_positive': [],
    }
    baseline = {h: [] for h in EVENT_HORIZONS}
    window = int(years * TRADING_DAYS)

    for symbol, an in analyzers.items():
        n = an.n
        start = max(MIN_HISTORY, n - window)
        close = an.close.to_numpy()
        z = an._horizon('1m')['zscore']
        macd = an.macd_hist

        def fwd(i, h):
            if i + h >= n or close[i] <= 0:
                return None
            return close[i + h] / close[i] - 1.0

        for i in range(start, n - max(EVENT_HORIZONS)):
            for h in EVENT_HORIZONS:
                r = fwd(i, h)
                if r is not None:
                    baseline[h].append(r)

            row = {}
            db = an.don_break.iloc[i]
            if db == 1:
                row['donchian_breakout_up'] = True
            elif db == -1:
                row['donchian_breakout_down'] = True
            if bool(an.squeeze_fired.iloc[i]):
                row['squeeze_fired'] = True
            zi = z.iloc[i]
            if pd.notna(zi):
                if zi <= -2:
                    row['zscore_below_-2'] = True
                elif zi >= 2:
                    row['zscore_above_2'] = True
            if i > 0 and pd.notna(macd.iloc[i]) and pd.notna(macd.iloc[i - 1]):
                if macd.iloc[i] > 0 and macd.iloc[i - 1] <= 0:
                    row['macd_flip_positive'] = True

            for name in row:
                rets = {h: fwd(i, h) for h in EVENT_HORIZONS}
                if all(v is not None for v in rets.values()):
                    events[name].append(rets)

    out = {'baseline': {}, 'events': {}}
    for h in EVENT_HORIZONS:
        out['baseline'][f'{h}d'] = round(float(np.mean(baseline[h])) * 100.0, 3) \
            if baseline[h] else None
    for name, rows in events.items():
        if len(rows) < 20:
            out['events'][name] = {'n': len(rows)}
            continue
        entry = {'n': len(rows)}
        for h in EVENT_HORIZONS:
            vals = [r[h] for r in rows]
            entry[f'{h}d'] = round(float(np.mean(vals)) * 100.0, 3)
        out['events'][name] = entry
    return out


# -- charts -------------------------------------------------------------

def make_charts(results, events, outdir='results'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    dates = pd.to_datetime(results['dates'])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    order = ['composite', 'momentum', 'trend', 'mean_reversion',
             'volume_flow', 'quality', 'equal_weight', 'spy']
    for name in order:
        curve = results['equity_curves'].get(name)
        if not curve:
            continue
        lw = 2.2 if name in ('composite', 'spy') else 1.1
        ax.plot(dates[:len(curve)], curve, label=name, linewidth=lw)
    ax.set_title(f"Equity curves, top-{int(results['config']['top'] * 100)}% "
                 f"long-only, {results['config']['step']}d rebalance")
    ax.set_ylabel('Growth of $1')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'equity_curves.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [s for s in STRATEGIES if results['ic'].get(s)]
    means = [results['ic'][s]['mean'] for s in names]
    tstats = [results['ic'][s]['t_stat'] for s in names]
    colors = ['#2a9d8f' if m >= 0 else '#e63946' for m in means]
    bars = ax.bar(names, means, color=colors)
    for bar, t in zip(bars, tstats):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"t={t}", ha='center',
                va='bottom' if bar.get_height() >= 0 else 'top', fontsize=8)
    ax.set_title(f"Mean IC by sleeve ({results['config']['step']}d forward returns)")
    ax.set_ylabel('Spearman IC')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'factor_ic.png'), dpi=150)
    plt.close(fig)

    if results['deciles']:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ds = sorted(results['deciles'])
        vals = [results['deciles'][d]['ann_return'] for d in ds]
        ax.bar([str(d) for d in ds], vals,
               color=['#e63946' if v < 0 else '#2a9d8f' for v in vals])
        ax.set_title('Annualized forward return by composite-score decile '
                     '(1 = lowest scores, 10 = highest)')
        ax.set_ylabel('Ann. return %')
        ax.grid(alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, 'deciles.png'), dpi=150)
        plt.close(fig)

    ev = {k: v for k, v in events['events'].items() if v.get('21d') is not None}
    if ev:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        names = list(ev)
        x = np.arange(len(names))
        w = 0.25
        for off, h in zip((-w, 0, w), EVENT_HORIZONS):
            ax.bar(x + off, [ev[n][f'{h}d'] for n in names], width=w, label=f'+{h}d')
        for off, h in zip((-w, 0, w), EVENT_HORIZONS):
            base = events['baseline'][f'{h}d']
            if base is not None:
                ax.axhline(base, linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
        ax.set_title('Mean forward return after signal (dashed = unconditional baseline)')
        ax.set_ylabel('Mean fwd return %')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, 'event_study.png'), dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--period', default='1m',
                        help='scoring horizon (default 1m)')
    parser.add_argument('--step', type=int, default=5,
                        help='rebalance every N sessions (default 5)')
    parser.add_argument('--years', type=float, default=3)
    parser.add_argument('--top', type=float, default=0.2,
                        help='fraction of universe held (default 0.2)')
    parser.add_argument('--symbols', default=None,
                        help='comma-separated override of the universe')
    parser.add_argument('--weighting', default='equal',
                        choices=['equal', 'inv_vol'],
                        help='basket weighting for held names (default equal)')
    parser.add_argument('--out', default='results/backtest_summary.json',
                        help='output JSON path')
    parser.add_argument('--no-charts', action='store_true')
    args = parser.parse_args()

    store = DataStore()
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    else:
        with open('config.json') as f:
            symbols = [s.strip() for s in
                       json.load(f)['watchlist']['symbols'].split(',') if s.strip()]

    print(f"loading universe ({len(symbols)} symbols)...")
    t0 = time.time()
    analyzers, skipped = load_universe(store, symbols)
    print(f"  {len(analyzers)} usable, {len(skipped)} skipped "
          f"({time.time() - t0:.0f}s)")
    if skipped:
        print(f"  skipped: {', '.join(skipped)}")

    spy = store.get_history('SPY')
    if spy.empty:
        raise SystemExit("SPY history unavailable; cannot build the calendar")

    print(f"running walk-forward backtest ({args.years}y, "
          f"{args.step}d rebalance, {args.period} horizon)...")
    t0 = time.time()
    results = run_backtest(analyzers, spy['Close'].astype(float),
                           period=args.period, step=args.step,
                           years=args.years, top=args.top,
                           weighting=args.weighting)
    print(f"  done in {time.time() - t0:.0f}s")

    print("running event studies...")
    events = event_studies(analyzers, years=args.years)

    os.makedirs('results', exist_ok=True)
    summary = {'backtest': results, 'event_studies': events}
    curves = summary['backtest'].pop('equity_curves')
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    summary['backtest']['equity_curves'] = curves

    if not args.no_charts:
        try:
            make_charts(results, events)
            print("charts written to results/")
        except ImportError:
            print("matplotlib not installed; skipping charts")

    print("\nstrategy summary:")
    header = f"{'strategy':<16} {'CAGR%':>7} {'Sharpe':>7} {'MaxDD%':>8} {'Hit%':>6}"
    print(header)
    for name, s in results['strategies'].items():
        print(f"{name:<16} {s['cagr'] if s['cagr'] is not None else '-':>7} "
              f"{s['sharpe'] if s['sharpe'] is not None else '-':>7} "
              f"{s['max_drawdown']:>8} {s['hit_rate']:>6}")

    print("\nIC summary (composite):", results['ic'].get('composite'))
    print(args.out, "written")


if __name__ == '__main__':
    main()
