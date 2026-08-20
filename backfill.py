"""
Pull the full history for every watchlist symbol into the price cache.

The cache was built by the dashboard, which only ever needed five years,
and the backtest inherited that limit. Five years is not enough for this
universe. With 80 names the standard error of a per-date Spearman IC is
about 1/sqrt(79) = 0.11, so statistical power comes from the number of
rebalances rather than the size of the cross-section: reaching a t of 3 on
a true IC of 0.02 needs roughly (3 * 0.12 / 0.02)^2 = 320 non-overlapping
periods. At a weekly rebalance that is six years of data before the
warm-up, and the factor search wants a discovery window on top of that.

Downloading more costs nothing but time, which makes it the cheapest
improvement available to every measurement in this project.

Note that data_store.HISTORY_PERIOD is also a ceiling, not just a floor:
the weekly full refresh replaces each symbol's rows wholesale so adjusted
prices stay consistent, so a shorter setting would silently truncate
whatever this script fetches.

Usage:
    python backfill.py                  # every watchlist symbol
    python backfill.py --period max     # everything Yahoo will give
    python backfill.py --symbols SPY,QQQ --force
"""

import argparse
import json
import time

import pandas as pd

from data_store import DataStore, HISTORY_PERIOD, _yf_download

EXTRA = ['SPY', '^VIX']   # needed by the market overlay and the calendar


def load_symbols(path='config.json'):
    with open(path) as f:
        raw = json.load(f)['watchlist']['symbols']
    return [s.strip().upper() for s in raw.split(',') if s.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--period', default=HISTORY_PERIOD,
                        help=f'yfinance period (default {HISTORY_PERIOD})')
    parser.add_argument('--symbols', default=None,
                        help='comma-separated override of the watchlist')
    parser.add_argument('--force', action='store_true',
                        help='re-download even where the cache already '
                             'reaches back far enough')
    parser.add_argument('--sleep', type=float, default=0.4,
                        help='pause between symbols, to stay polite')
    args = parser.parse_args()

    symbols = ([s.strip().upper() for s in args.symbols.split(',') if s.strip()]
               if args.symbols else load_symbols())
    for extra in EXTRA:
        if extra not in symbols:
            symbols.append(extra)

    store = DataStore()
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=365 * 6)
    grown = skipped = failed = 0
    t0 = time.time()

    print(f"backfilling {len(symbols)} symbols at period={args.period}")
    for n, symbol in enumerate(symbols, 1):
        before = store.cached_history(symbol)
        have_from = before.index[0] if not before.empty else None
        if not args.force and have_from is not None and have_from < cutoff:
            print(f"  [{n:>3}/{len(symbols)}] {symbol:<8} already back to "
                  f"{have_from.date()}, skipped")
            skipped += 1
            continue
        try:
            df = _yf_download(symbol, period=args.period)
        except Exception as e:
            print(f"  [{n:>3}/{len(symbols)}] {symbol:<8} download failed: {e}")
            failed += 1
            continue
        if df is None or df.empty:
            print(f"  [{n:>3}/{len(symbols)}] {symbol:<8} no data returned")
            failed += 1
            continue

        # replace rather than merge: adjusted prices are only consistent
        # within a single download
        store.store_history(symbol, df, replace=True, mark_fresh=True)
        after = store.cached_history(symbol)
        print(f"  [{n:>3}/{len(symbols)}] {symbol:<8} {len(before):>5} -> "
              f"{len(after):>5} bars, from {after.index[0].date()}")
        grown += 1
        time.sleep(args.sleep)

    print(f"\n{grown} refreshed, {skipped} already deep enough, {failed} failed "
          f"({time.time() - t0:.0f}s)")

    spy = store.cached_history('SPY')
    if not spy.empty:
        years = (spy.index[-1] - spy.index[0]).days / 365.25
        print(f"calendar now spans {years:.1f} years "
              f"({spy.index[0].date()} to {spy.index[-1].date()}, "
              f"{len(spy)} bars)")
        print("re-run: python backtest.py --null-audit 200")
    store.close()


if __name__ == '__main__':
    main()
