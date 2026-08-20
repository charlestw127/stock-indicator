"""
SQLite-backed storage: price cache, analysis run history, fundamentals cache.

Prices are cached locally and updated incrementally, so a scan of the full
watchlist only hits Yahoo for bars we don't already have. A full re-download
happens weekly per symbol to pick up dividend/split adjustments.
"""

import json
import sqlite3
import threading
import time
import datetime as dt
import logging

import pandas as pd

logger = logging.getLogger('stock_app.data_store')

DB_PATH = 'market_data.db'

# how long a cached symbol stays fresh before we look for new bars
# How much history a full refresh pulls. The weekly refresh replaces a
# symbol's rows wholesale so adjusted prices stay consistent, which means
# this value is also a ceiling: shortening it silently truncates the cache
# on the next refresh. An 80-name cross-section needs every year it can
# get - the per-date IC standard error is about 1/sqrt(79) = 0.11, so
# statistical power comes from the number of rebalances, not the universe.
HISTORY_PERIOD = '15y'

FETCH_TTL_SECONDS = 15 * 60
# full re-download interval, to keep adjusted prices consistent
FULL_REFRESH_SECONDS = 7 * 24 * 3600

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (symbol, date)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    universe TEXT NOT NULL,
    results TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS score_history (
    run_id INTEGER NOT NULL,
    ts REAL NOT NULL,
    symbol TEXT NOT NULL,
    period TEXT NOT NULL,
    score REAL,
    rank INTEGER,
    recommendation TEXT
);
CREATE INDEX IF NOT EXISTS idx_score_history_symbol
    ON score_history (symbol, period, ts);
CREATE TABLE IF NOT EXISTS fundamentals (
    symbol TEXT PRIMARY KEY,
    data TEXT,
    fetched_at REAL
);
"""


class DataStore:
    def __init__(self, path=DB_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self):
        with self._lock:
            self._conn.close()

    # -- meta -----------------------------------------------------------

    def _meta_get(self, key):
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def _meta_set(self, key, value):
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)))

    def get_meta_json(self, key):
        with self._lock:
            raw = self._meta_get(key)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return None

    def set_meta_json(self, key, value):
        with self._lock:
            self._meta_set(key, json.dumps(value))
            self._conn.commit()

    # -- prices ---------------------------------------------------------

    def cached_history(self, symbol):
        """Cached OHLCV for a symbol as a DataFrame (may be empty)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT date, open, high, low, close, volume FROM prices "
                "WHERE symbol=? ORDER BY date", (symbol,)).fetchall()
        if not rows:
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        df = pd.DataFrame(rows, columns=['date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df.index = pd.to_datetime(df.pop('date'))
        return df

    def store_history(self, symbol, df, replace=False, mark_fresh=False):
        """Upsert OHLCV rows. `replace` drops existing rows first (used for
        the weekly full refresh so adjusted prices stay consistent).
        `mark_fresh` stamps the fetch metadata so get_history serves this
        data without trying the network - used by tests and imports."""
        if df is None or df.empty:
            return
        records = []
        for ts, row in df.iterrows():
            close = row.get('Close')
            if close is None or pd.isna(close):
                continue
            records.append((
                symbol, ts.strftime('%Y-%m-%d'),
                _num(row.get('Open')), _num(row.get('High')),
                _num(row.get('Low')), _num(close), _num(row.get('Volume')),
            ))
        with self._lock:
            if replace:
                self._conn.execute("DELETE FROM prices WHERE symbol=?", (symbol,))
            self._conn.executemany(
                "INSERT INTO prices (symbol, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(symbol, date) DO UPDATE SET "
                "open=excluded.open, high=excluded.high, low=excluded.low, "
                "close=excluded.close, volume=excluded.volume",
                records)
            if mark_fresh:
                now = time.time()
                self._meta_set(f'fetch:{symbol}', now)
                self._meta_set(f'full:{symbol}', now)
            self._conn.commit()

    def get_history(self, symbol, downloader=None):
        """Cached history for a symbol, refreshed from Yahoo when stale.

        `downloader(symbol, start=None)` can be injected for tests; the
        default uses yfinance. Falls back to whatever is cached if the
        network fails.
        """
        now = time.time()
        with self._lock:
            last_fetch = self._meta_get(f'fetch:{symbol}')
            last_full = self._meta_get(f'full:{symbol}')

        cached = self.cached_history(symbol)
        fresh = last_fetch and (now - float(last_fetch)) < FETCH_TTL_SECONDS
        if not cached.empty and fresh:
            return cached

        if downloader is None:
            downloader = _yf_download

        needs_full = cached.empty or not last_full or \
            (now - float(last_full)) > FULL_REFRESH_SECONDS
        try:
            if needs_full:
                df = downloader(symbol)
                self.store_history(symbol, df, replace=True)
                with self._lock:
                    self._meta_set(f'full:{symbol}', now)
                    self._meta_set(f'fetch:{symbol}', now)
                    self._conn.commit()
            else:
                # re-pull the last few bars; today's bar changes intraday
                start = cached.index[-1] - pd.Timedelta(days=7)
                df = downloader(symbol, start=start.strftime('%Y-%m-%d'))
                self.store_history(symbol, df)
                with self._lock:
                    self._meta_set(f'fetch:{symbol}', now)
                    self._conn.commit()
        except Exception as e:
            logger.warning("fetch failed for %s, using cache: %s", symbol, e)
            return cached

        return self.cached_history(symbol)

    # -- analysis runs --------------------------------------------------

    def save_run(self, universe, results):
        """Persist a completed scan and index its scores for history."""
        ts = time.time()
        payload = json.dumps(results)
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO runs (ts, universe, results) VALUES (?, ?, ?)",
                (ts, ','.join(sorted(universe)), payload))
            run_id = cur.lastrowid
            rows = []
            for symbol, periods in results.get('symbols', {}).items():
                if not isinstance(periods, dict):
                    continue
                for period, res in periods.items():
                    if not isinstance(res, dict) or res.get('score') is None:
                        continue
                    rows.append((run_id, ts, symbol, period, res['score'],
                                 res.get('rank'), res.get('recommendation')))
            self._conn.executemany(
                "INSERT INTO score_history (run_id, ts, symbol, period, score, rank, recommendation) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
            # keep the runs table from growing without bound
            self._conn.execute(
                "DELETE FROM runs WHERE id NOT IN (SELECT id FROM runs ORDER BY id DESC LIMIT 50)")
            self._conn.commit()
        return run_id

    def latest_run(self, max_age_seconds=None):
        """Most recent stored run as (id, ts, universe list, results dict)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id, ts, universe, results FROM runs ORDER BY id DESC LIMIT 1").fetchone()
        if not row:
            return None
        run_id, ts, universe, payload = row
        if max_age_seconds is not None and (time.time() - ts) > max_age_seconds:
            return None
        return run_id, ts, universe.split(','), json.loads(payload)

    def previous_ranks(self, before_run_id=None):
        """symbol -> {period: rank} from the newest run (or the newest one
        before `before_run_id`)."""
        with self._lock:
            if before_run_id is None:
                row = self._conn.execute(
                    "SELECT id FROM runs ORDER BY id DESC LIMIT 1").fetchone()
            else:
                row = self._conn.execute(
                    "SELECT id FROM runs WHERE id < ? ORDER BY id DESC LIMIT 1",
                    (before_run_id,)).fetchone()
            if not row:
                return {}
            rows = self._conn.execute(
                "SELECT symbol, period, rank FROM score_history WHERE run_id=?",
                (row[0],)).fetchall()
        out = {}
        for symbol, period, rank in rows:
            if rank is not None:
                out.setdefault(symbol, {})[period] = rank
        return out

    def score_history(self, symbol, period='1m', limit=200):
        """Score/rank time series for one symbol, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, score, rank, recommendation FROM score_history "
                "WHERE symbol=? AND period=? ORDER BY ts DESC LIMIT ?",
                (symbol, period, limit)).fetchall()
        return [
            {'ts': dt.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M'),
             'score': score, 'rank': rank, 'recommendation': rec}
            for ts, score, rank, rec in reversed(rows)
        ]

    # -- fundamentals ---------------------------------------------------

    def get_fundamentals(self, symbol, max_age_seconds=7 * 24 * 3600):
        with self._lock:
            row = self._conn.execute(
                "SELECT data, fetched_at FROM fundamentals WHERE symbol=?",
                (symbol,)).fetchone()
        if not row or row[0] is None:
            return None
        if (time.time() - row[1]) > max_age_seconds:
            return None
        return json.loads(row[0])

    def save_fundamentals(self, symbol, data):
        with self._lock:
            self._conn.execute(
                "INSERT INTO fundamentals (symbol, data, fetched_at) VALUES (?, ?, ?) "
                "ON CONFLICT(symbol) DO UPDATE SET data=excluded.data, fetched_at=excluded.fetched_at",
                (symbol, json.dumps(data), time.time()))
            self._conn.commit()

    def fundamentals_age(self, symbol):
        with self._lock:
            row = self._conn.execute(
                "SELECT fetched_at FROM fundamentals WHERE symbol=?", (symbol,)).fetchone()
        return (time.time() - row[0]) if row else None


def _num(v):
    if v is None or pd.isna(v):
        return None
    return float(v)


def _yf_download(symbol, start=None, period=HISTORY_PERIOD):
    import yfinance as yf
    if start:
        df = yf.download(symbol, start=start, auto_adjust=True, progress=False)
    else:
        df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df
