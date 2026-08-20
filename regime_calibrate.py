"""
Estimate per-regime sleeve weights from the data, and check whether they
beat the hand-set table out of sample.

The README is candid that REGIME_WEIGHTS is fifteen numbers picked by
judgement and that "whether this earns its keep is examined in the results".
This script is that examination, run as a proper experiment rather than a
fit:

1. Split the history into a discovery window and a later validation window.
   Nothing from validation touches the estimate.
2. On discovery, label every bar with a regime and collect
   (state, sleeve scores, forward return) at each rebalance.
3. Ridge-regress forward return on sleeve scores within each state, clip
   negatives, normalise, and shrink toward the hand table
   (regime.estimate_state_weights).
4. Re-score the validation window under both tables and compare mean rank
   IC and the resulting top-quintile Sharpe.
5. Write results/regime_weights.json only if the calibrated table actually
   wins on validation. A losing run reports the loss and changes nothing.

Two labellers are available. `classifier` is the existing
quant_engine.classify_trend_regime, which is what the live dashboard uses,
so its weights apply directly. `jump` is the statistical jump model in
regime.py, following Shu, Yu and Mulvey (arXiv 2402.05272); it is slower and
is there to answer whether the hand thresholds are the weak link.

Usage:
    python regime_calibrate.py                          # classifier labels
    python regime_calibrate.py --labeler jump --states 3
    python regime_calibrate.py --shrinkage 0.3 --apply

Without --apply nothing is written; the run only reports what it found.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import evaluation as ev
import quant_engine as qe
import regime
from backtest import (MIN_HISTORY, STRATEGIES, TRADING_DAYS, _equity,
                      _holding_weights, _ic_stats, _strategy_stats,
                      load_universe)
from data_store import DataStore

SLEEVES = STRATEGIES[1:]  # everything except 'composite'
OUT_PATH = os.path.join('results', 'regime_weights.json')

# How much validation Sharpe a calibrated table may give up while still
# counting as a win. Set below 1.0 only because Sharpe on a few dozen
# held-out rebalances is itself noisy; it is not licence to trade real
# portfolio quality for a marginally better ranking.
SHARPE_TOLERANCE = 0.95


def collect_samples(analyzers, calendar, period, step, start, end,
                    labels_by_symbol=None):
    """Walk the window and record what each sleeve said and what happened.

    Returns (samples, per_date) where samples feed the weight estimate and
    per_date has the shape run_backtest uses, so the same IC and portfolio
    code can score a candidate table.
    """
    positions = list(range(start, end - step, step))
    dates = [calendar[p] for p in positions]
    sym_pos = {s: an.index.get_indexer(dates, method='pad')
               for s, an in analyzers.items()}

    samples, per_date = [], []
    for k, pos in enumerate(positions):
        date, next_date = calendar[pos], calendar[pos + step]
        scores, fwd, states = {}, {}, {}
        for symbol, an in analyzers.items():
            i = sym_pos[symbol][k]
            if i < MIN_HISTORY - 1 or (date - an.index[i]).days > 10:
                continue
            j = an.index.get_indexer([next_date], method='pad')[0]
            if j <= i:
                continue
            res = an.score_at(period, i)
            entry = dict(res['factors'])
            entry['composite'] = res['score']
            scores[symbol] = entry
            c0, c1 = float(an.close.iloc[i]), float(an.close.iloc[j])
            if c0 <= 0:
                continue
            fwd[symbol] = c1 / c0 - 1.0

            if labels_by_symbol is None:
                state = res['regime']['trend']
            else:
                series = labels_by_symbol.get(symbol)
                state = None
                if series is not None and i < len(series):
                    v = series.iloc[i]
                    state = v if isinstance(v, str) else None
            states[symbol] = state
            if state is not None:
                samples.append((state, res['factors'], fwd[symbol]))
        per_date.append({'date': date, 'scores': scores, 'fwd': fwd,
                         'states': states})
    return samples, per_date


def score_table(per_date, table, states_default='mixed', top=0.2):
    """Mean rank IC and top-quintile stats for a candidate weight table.

    The composite is recomputed from the stored sleeve scores, so this
    costs nothing beyond a dot product per name and never re-reads prices.
    """
    ics, rets = [], []
    for snap in per_date:
        usable = [s for s in snap['scores'] if s in snap['fwd']]
        if len(usable) < 8:
            continue
        composites = {}
        for s in usable:
            state = snap['states'].get(s) or states_default
            w = table.get(state) or table.get(states_default)
            composites[s] = sum(w[k] * snap['scores'][s][k] for k in w)
        x = pd.Series([composites[s] for s in usable]).rank()
        y = pd.Series([snap['fwd'][s] for s in usable]).rank()
        if x.std() > 0 and y.std() > 0:
            ics.append(float(np.corrcoef(x, y)[0, 1]))
        ranked = sorted(usable, key=lambda s: -composites[s])
        held = set(ranked[:max(1, int(len(ranked) * top))])
        w = _holding_weights(held, snap, 'equal')
        rets.append(float(sum(snap['fwd'][s] * w[s] for s in held)))
    return ics, rets


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--period', default='1m')
    parser.add_argument('--step', type=int, default=21,
                        help='rebalance spacing in sessions (default 21; the '
                             'monthly horizon is where the IC lives)')
    parser.add_argument('--labeler', default='classifier',
                        choices=['classifier', 'jump'])
    parser.add_argument('--states', type=int, default=3)
    parser.add_argument('--jump-penalty', type=float, default=20.0)
    parser.add_argument('--shrinkage', type=float, default=0.5,
                        help='weight on the fitted table vs the prior '
                             '(default 0.5)')
    parser.add_argument('--validation-years', type=float, default=2.0)
    parser.add_argument('--symbols', default=None)
    parser.add_argument('--apply', action='store_true',
                        help='write results/regime_weights.json if the '
                             'calibrated table wins on validation')
    parser.add_argument('--out', default=OUT_PATH)
    args = parser.parse_args()

    store = DataStore()
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    else:
        with open('config.json') as f:
            symbols = [s.strip() for s in
                       json.load(f)['watchlist']['symbols'].split(',') if s.strip()]

    print(f"loading universe ({len(symbols)} symbols)...")
    analyzers, skipped = load_universe(store, symbols)
    print(f"  {len(analyzers)} usable, {len(skipped)} skipped")
    spy = store.get_history('SPY')
    if spy.empty:
        raise SystemExit("SPY history unavailable; cannot build the calendar")
    calendar = spy['Close'].astype(float).index

    split = len(calendar) - int(args.validation_years * TRADING_DAYS)
    if split <= MIN_HISTORY + 60:
        raise SystemExit("not enough history to hold out a validation window")

    labels_by_symbol = None
    if args.labeler == 'jump':
        print(f"labelling with a {args.states}-state jump model "
              f"(penalty {args.jump_penalty})...")
        t0 = time.time()
        labels_by_symbol = {}
        for n, (symbol, an) in enumerate(analyzers.items(), 1):
            feats = regime.build_features(an.close, an.high, an.low)
            labels_by_symbol[symbol] = regime.label_walk_forward(
                feats, n_states=args.states, jump_penalty=args.jump_penalty,
                min_train=MIN_HISTORY, refit_every=21)
            if n % 10 == 0:
                print(f"  {n}/{len(analyzers)} ({time.time() - t0:.0f}s)")
        pooled = pd.concat([s.dropna() for s in labels_by_symbol.values()])
        print("  state persistence:",
              json.dumps(regime.state_summary(pooled), indent=None))

    print("collecting discovery samples...")
    t0 = time.time()
    disc_samples, disc_dates = collect_samples(
        analyzers, calendar, args.period, args.step,
        MIN_HISTORY, split, labels_by_symbol)
    print(f"  {len(disc_samples)} samples over {len(disc_dates)} rebalances "
          f"({time.time() - t0:.0f}s)")

    print("collecting validation samples...")
    _, val_dates = collect_samples(
        analyzers, calendar, args.period, args.step,
        split, len(calendar), labels_by_symbol)
    print(f"  {len(val_dates)} rebalances")

    prior = qe.REGIME_WEIGHTS
    if args.labeler == 'jump' and args.states == 2:
        prior = {k: v for k, v in prior.items() if k != 'mixed'}

    table, diag = regime.estimate_state_weights(
        disc_samples, prior, shrinkage=args.shrinkage)

    print("\ncalibrated weights (shrunk toward the prior):")
    header = f"{'state':<16}" + "".join(f"{s:>16}" for s in SLEEVES)
    print(header)
    for state in table:
        row = f"{state:<16}"
        for s in SLEEVES:
            row += f"{table[state][s]:>8.3f}{'':>8}"
        print(row)
        d = diag[state]
        print(f"{'  prior':<16}" + "".join(
            f"{prior[state][s]:>8.3f}{'':>8}" for s in SLEEVES)
            + f"   n={d['n_samples']}"
            + ("" if d['fitted'] else "  (too few samples, prior kept)"))

    results = {}
    for name, tab in (('prior', prior), ('calibrated', table)):
        for split_name, dates in (('discovery', disc_dates),
                                  ('validation', val_dates)):
            ics, rets = score_table(dates, tab)
            results[(name, split_name)] = {
                'ic': _ic_stats(ics),
                'stats': _strategy_stats(rets, args.step) if rets else None,
            }

    print(f"\nout-of-sample comparison ({args.validation_years:g}y held out, "
          f"{args.labeler} labels):")
    print(f"{'table':<12} {'split':<12} {'mean IC':>9} {'IC t':>7} "
          f"{'CAGR%':>8} {'Sharpe':>8} {'MaxDD%':>8}")
    for key in [('prior', 'discovery'), ('calibrated', 'discovery'),
                ('prior', 'validation'), ('calibrated', 'validation')]:
        r = results[key]
        ic, st = r['ic'] or {}, r['stats'] or {}
        print(f"{key[0]:<12} {key[1]:<12} "
              f"{ic.get('mean', '-'):>9} {ic.get('t_stat', '-'):>7} "
              f"{st.get('cagr', '-'):>8} {st.get('sharpe', '-'):>8} "
              f"{st.get('max_drawdown', '-'):>8}")

    prior_val = (results[('prior', 'validation')]['ic'] or {}).get('mean')
    cal_val = (results[('calibrated', 'validation')]['ic'] or {}).get('mean')
    n_trials = ev.count_trials()
    ev.log_trial({'job': 'regime_calibrate', 'labeler': args.labeler,
                  'states': args.states, 'shrinkage': args.shrinkage,
                  'period': args.period, 'step': args.step},
                 {'prior_val_ic': prior_val, 'calibrated_val_ic': cal_val})

    # Winning means better ranking AND no worse portfolio. Mean IC alone is
    # too loose a test: a table can order names slightly better and still
    # build a worse book, which is what the first 15-year run did (IC
    # +0.0174 vs +0.0090, Sharpe 1.57 vs 1.73). Since the README's own
    # conclusion is that this system's value is portfolio construction
    # rather than ranking, a table that trades Sharpe for IC is not an
    # improvement to the thing that actually works.
    prior_sr = ((results[('prior', 'validation')]['stats'] or {})
                .get('sharpe'))
    cal_sr = ((results[('calibrated', 'validation')]['stats'] or {})
              .get('sharpe'))
    ic_better = (prior_val is not None and cal_val is not None
                 and cal_val > prior_val)
    sharpe_ok = (prior_sr is None or cal_sr is None
                 or cal_sr >= prior_sr * SHARPE_TOLERANCE)
    won = bool(ic_better and sharpe_ok)

    print()
    if won:
        print(f"  calibrated table beat the prior out of sample "
              f"({cal_val:+.4f} vs {prior_val:+.4f} mean IC, "
              f"Sharpe {cal_sr} vs {prior_sr})")
    elif ic_better and not sharpe_ok:
        print(f"  calibrated table ranked better ({cal_val:+.4f} vs "
              f"{prior_val:+.4f} mean IC) but built a worse portfolio "
              f"(Sharpe {cal_sr} vs {prior_sr}) - not an improvement")
    else:
        print(f"  calibrated table did NOT beat the prior out of sample "
              f"({cal_val} vs {prior_val} mean IC)")
    val_ic = results[('calibrated', 'validation')]['ic'] or {}
    if val_ic.get('t_stat') is not None:
        h = ev.deflated_t(val_ic['t_stat'], n_trials)
        if h['clears_positive']:
            verdict = 'clears'
        elif h['passes']:
            verdict = 'significant in the wrong direction'
        else:
            verdict = 'does not clear'
        print(f"  validation IC t {h['t_stat']} vs best-of-"
              f"{h['n_trials']} hurdle {h['hurdle']} -> {verdict}")
    print("  with 80 names the per-date IC standard error is about 0.11, so "
          "a difference this size is suggestive at best")

    payload = {
        'weights': table,
        'labeler': args.labeler,
        'shrinkage': args.shrinkage,
        'period': args.period,
        'step': args.step,
        'discovery_end': str(calendar[split].date()),
        'validation_end': str(calendar[-1].date()),
        'n_samples': len(disc_samples),
        'diagnostics': diag,
        'comparison': {f'{k[0]}_{k[1]}': v for k, v in results.items()},
        'beat_prior_out_of_sample': bool(won),
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    report = args.out.replace('.json', '_report.json')
    with open(report, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n{report} written")

    if args.apply and won:
        with open(args.out, 'w') as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"{args.out} written; quant_engine will pick it up on next start")
    elif args.apply:
        print("not applied: the calibrated table has to win on validation "
              "first")
    else:
        print("not applied (pass --apply to install a winning table)")


if __name__ == '__main__':
    main()
