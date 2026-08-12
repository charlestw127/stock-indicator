import numpy as np
import pandas as pd

from data_store import DataStore
from conftest import make_ohlcv


def make_store(tmp_path):
    return DataStore(str(tmp_path / 'test.db'))


def test_history_roundtrip(tmp_path):
    store = make_store(tmp_path)
    df = make_ohlcv(np.linspace(100, 120, 50))
    store.store_history('TEST', df)
    out = store.cached_history('TEST')
    assert len(out) == 50
    assert out['Close'].iloc[-1] == df['Close'].iloc[-1]


def test_get_history_uses_cache_within_ttl(tmp_path):
    store = make_store(tmp_path)
    calls = []

    def downloader(symbol, start=None):
        calls.append((symbol, start))
        return make_ohlcv(np.linspace(100, 120, 400))

    first = store.get_history('TEST', downloader=downloader)
    assert len(first) == 400
    assert len(calls) == 1

    second = store.get_history('TEST', downloader=downloader)
    assert len(second) == 400
    assert len(calls) == 1  # served from cache, no second download


def test_get_history_survives_download_failure(tmp_path):
    store = make_store(tmp_path)
    store.store_history('TEST', make_ohlcv(np.linspace(100, 110, 100)))

    def broken(symbol, start=None):
        raise ConnectionError('offline')

    out = store.get_history('TEST', downloader=broken)
    assert len(out) == 100


def test_run_storage_and_movers(tmp_path):
    store = make_store(tmp_path)
    run1 = {'symbols': {'AAPL': {'1m': {'score': 10.0, 'rank': 7,
                                        'recommendation': 'WEAK SELL'}}}}
    run2 = {'symbols': {'AAPL': {'1m': {'score': 40.0, 'rank': 2,
                                        'recommendation': 'BUY'}}}}
    store.save_run(['AAPL'], run1)
    prev = store.previous_ranks()
    assert prev['AAPL']['1m'] == 7

    store.save_run(['AAPL'], run2)
    latest = store.latest_run()
    assert latest is not None
    _, _, universe, results = latest
    assert universe == ['AAPL']
    assert results['symbols']['AAPL']['1m']['rank'] == 2

    history = store.score_history('AAPL', '1m')
    assert [h['rank'] for h in history] == [7, 2]


def test_latest_run_age_limit(tmp_path):
    store = make_store(tmp_path)
    store.save_run(['AAPL'], {'symbols': {}})
    assert store.latest_run(max_age_seconds=60) is not None
    assert store.latest_run(max_age_seconds=0) is None


def test_fundamentals_cache(tmp_path):
    store = make_store(tmp_path)
    assert store.get_fundamentals('AAPL') is None
    store.save_fundamentals('AAPL', {'trailingPE': 30.5, 'sector': 'Technology'})
    out = store.get_fundamentals('AAPL')
    assert out['sector'] == 'Technology'
    assert store.get_fundamentals('AAPL', max_age_seconds=0) is None
    assert store.fundamentals_age('AAPL') < 5
