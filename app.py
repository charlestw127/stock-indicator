"""
Flask app for the stock scanner.

Prices come from a local SQLite cache (data_store.py) that is refreshed
incrementally, so a full watchlist scan does not re-download years of
history. A background thread re-runs the watchlist scan on the configured
interval and stores each run, which gives the UI instant responses, a score
history per symbol, and rank movers between runs.

Scoring is done by the multi-factor engine in quant_engine.py. Market-level
context (SPY trend, VIX) and per-symbol fundamentals are layered on top but
kept out of the composite score - see fundamentals.py for why.
"""

from flask import Flask, request, jsonify, render_template_string, send_from_directory
import datetime as dt
import json
import logging
import os
import threading
import time
import traceback

import numpy as np
import pandas as pd

from indicators import calculate_rsi, calculate_bbp, calculate_macd
from strategies import (
    get_manual_recommendation, get_score_from_indicators,
    calculate_percentile_rank, rank_to_recommendation,
)
from quant_engine import analyze_symbol, load_calibrated_weights
from data_store import DataStore
from market import market_overlay
from portfolio_risk import portfolio_risk
from recommender import recommend
from checks import run_checks, review, log_flags
from brief import generate
import fundamentals as fnd
from utils import time_period_to_start_date
from templates.base import BASE_TEMPLATE

CONFIG_FILE = 'config.json'
ALL_PERIODS = ['1d', '1w', '1m', '6m', '1y']

DEFAULT_CONFIG = {
    "settings": {
        "refreshInterval": 1800,
        "hideNonBuys": True,
        "hideRanksAbove": 7,
        "recommendationSize": 20
    },
    "watchlist": {
        "symbols": "AAPL,MSFT,GOOGL,AMZN,NVDA,AMD,INTC,TSM,CRM,ADBE,JPM,BAC,GS,V,MA,BLK,JNJ,PFE,MRNA,UNH,CVS,COST,WMT,TGT,MCD,SBUX,NKE,CAT,DE,BA,GE,XOM,CVX,NEE,TSLA,F,GM,T,VZ,NFLX,DIS,ETSY,SHOP,BABA,COIN,ABNB,HOOD,PLTR,U,SNAP,PINS,BRK-B,BRK-A,SPY,QQQ,DIA,IWM,VTI,XLF,XLK,XLE,XLV,XLI,XLP,XLY,EFA,EEM,FXI,EWJ,TLT,HYG,AGG,GLD,SLV,USO,VXX,SH,ARKK,ICLN,SOXX,HACK,SMH"
    },
    "portfolio": {
        "symbols": "",
        "positions": []
    }
}

if not os.path.exists('logs'):
    os.makedirs('logs')
logging.basicConfig(
    filename='logs/stock_app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('stock_app')

app = Flask(__name__)
store = DataStore()
# Install calibrated per-regime sleeve weights if regime_calibrate.py has
# produced a set that beat the hand-set prior out of sample. Missing file =
# the documented default stays in force.
_active_weights = load_calibrated_weights()
# reentrant: _serve_watchlist holds it across its cache check + scan so two
# simultaneous page loads don't both run a full scan
scan_lock = threading.RLock()


def load_config():
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return DEFAULT_CONFIG
    except Exception as e:
        logger.error("error loading configuration: %s", e)
        return DEFAULT_CONFIG


def save_config(config):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return config
    except Exception as e:
        logger.error("error saving configuration: %s", e)
        return config


def watchlist_symbols(config):
    return [s.strip().upper() for s in
            config['watchlist']['symbols'].split(',') if s.strip()]


def extract_scalar_value(value):
    try:
        if hasattr(value, 'iloc'):
            return value.iloc[0]
        if hasattr(value, 'item'):
            return value.item()
        return value
    except Exception:
        return None


def clean_nan_for_json(value):
    if isinstance(value, (float, np.float64, np.float32)) and \
            (np.isnan(value) or np.isinf(value)):
        return None
    return value


def legacy_analyze(data, periods):
    """Fallback when the engine can't run (usually too little history):
    basic RSI/BBP/MACD on each period's tail window."""
    out = {}
    close = data['Close']
    end_date = data.index[-1]
    for time_period in periods:
        try:
            start = time_period_to_start_date(
                time_period,
                end_date.to_pydatetime() if hasattr(end_date, 'to_pydatetime') else end_date)
            window = close[close.index >= pd.Timestamp(start)]
            if len(window) < 2:
                out[time_period] = {'error': 'Not enough data for this period'}
                continue
            rsi = extract_scalar_value(calculate_rsi(window).iloc[-1])
            bbp = extract_scalar_value(calculate_bbp(window).iloc[-1])
            macd = extract_scalar_value(calculate_macd(window).iloc[-1])
            out[time_period] = {
                'lastDate': end_date.strftime('%Y-%m-%d'),
                'lastPrice': clean_nan_for_json(extract_scalar_value(close.iloc[-1])),
                'recommendation': get_manual_recommendation(rsi, bbp, macd),
                'score': get_score_from_indicators({'rsi': rsi, 'bbp': bbp, 'macd': macd}),
                'indicators': {
                    'rsi': clean_nan_for_json(rsi),
                    'bbp': clean_nan_for_json(bbp),
                    'macd': clean_nan_for_json(macd),
                },
            }
        except Exception as e:
            out[time_period] = {'error': f'Legacy analysis failed: {e}'}
    return out


def run_scan(symbols, periods):
    """Score every symbol at every horizon and rank them cross-sectionally.
    Returns the results dict used by both the API and the background thread."""
    with scan_lock:
        return _run_scan_locked(symbols, periods)


def _run_scan_locked(symbols, periods):
    benchmark_close = None
    try:
        spy = store.get_history('SPY')
        if not spy.empty:
            benchmark_close = spy['Close'].astype(float)
    except Exception as e:
        logger.warning("benchmark unavailable: %s", e)

    medians = fnd.universe_medians(store, symbols)
    results = {'symbols': {}, 'scores': {}}

    for symbol in symbols:
        try:
            data = store.get_history(symbol)
            if data.empty or 'Close' not in data.columns:
                results['symbols'][symbol] = {
                    p: {'error': 'No data available'} for p in periods}
                continue

            try:
                per_period = analyze_symbol(data, periods, benchmark_close)
            except Exception as e:
                logger.warning("engine failed for %s, using legacy path: %s", symbol, e)
                per_period = legacy_analyze(data, periods)

            results['symbols'][symbol] = per_period
            for period, res in per_period.items():
                if isinstance(res, dict) and res.get('score') is not None:
                    results['scores'].setdefault(period, {})[symbol] = res['score']

            fund = fnd.get_fundamentals(store, symbol)
            summary, fund_signals = fnd.fundamental_summary(fund, medians)
            if summary:
                summary['signals'] = fund_signals
                results['symbols'][symbol]['fundamentals'] = summary

            month = per_period.get('1m') or {}
            if month.get('score') is not None:
                print(f"  {symbol}: 1m score {month['score']:.1f} "
                      f"({(month.get('regime') or {}).get('trend', '?')})")
        except Exception as e:
            logger.error("error processing %s: %s", symbol, e)
            logger.error(traceback.format_exc())
            results['symbols'][symbol] = {'error': str(e)}

    _apply_ranks(results)
    _apply_cross_section(results, periods)
    return results


def _apply_ranks(results):
    for period, scores_dict in results.get('scores', {}).items():
        if not scores_dict:
            continue
        syms = list(scores_dict.keys())
        try:
            ranks = calculate_percentile_rank([scores_dict[s] for s in syms])
        except Exception as e:
            logger.error("ranking failed for %s: %s", period, e)
            continue
        for symbol, rank in zip(syms, ranks):
            res = results['symbols'].get(symbol, {}).get(period)
            if isinstance(res, dict):
                res['rank'] = int(rank)
                res['recommendation'] = rank_to_recommendation(rank)


def _apply_cross_section(results, periods):
    """Universe and sector percentiles for each score."""
    sectors = {}
    for symbol in results['symbols']:
        fund = results['symbols'][symbol].get('fundamentals') or {}
        if fund.get('sector'):
            sectors[symbol] = fund['sector']

    for period in periods:
        scores = results.get('scores', {}).get(period, {})
        if len(scores) < 2:
            continue
        values = sorted(scores.values())
        n = len(values)
        by_sector = {}
        for symbol, sector in sectors.items():
            if symbol in scores:
                by_sector.setdefault(sector, []).append(scores[symbol])
        for symbol, score in scores.items():
            res = results['symbols'].get(symbol, {}).get(period)
            if not isinstance(res, dict):
                continue
            pctile = sum(1 for v in values if v <= score) / n * 100.0
            cs = {'universe_pctile': round(pctile, 1)}
            sector = sectors.get(symbol)
            peers = by_sector.get(sector, [])
            if sector and len(peers) >= 3:
                cs['sector'] = sector
                cs['sector_pctile'] = round(
                    sum(1 for v in peers if v <= score) / len(peers) * 100.0, 1)
            res['cross_section'] = cs


def _compute_movers(results):
    """Rank jumps of 3+ versus the most recent stored run."""
    prev = store.previous_ranks()
    movers = []
    for symbol, periods in results['symbols'].items():
        if not isinstance(periods, dict):
            continue
        for period, res in periods.items():
            if not isinstance(res, dict) or 'rank' not in res:
                continue
            old = prev.get(symbol, {}).get(period)
            if old is not None and abs(res['rank'] - old) >= 3:
                movers.append({'symbol': symbol, 'period': period,
                               'from': old, 'to': res['rank']})
    movers.sort(key=lambda m: -abs(m['to'] - m['from']))
    return movers[:10]


def _filter_results(results, config):
    hide_non_buys = config['settings'].get('hideNonBuys', False)
    hide_ranks_above = config['settings'].get('hideRanksAbove', 0)
    if not (hide_non_buys or hide_ranks_above > 0):
        return results

    filtered = {}
    for symbol, periods in results['symbols'].items():
        if not isinstance(periods, dict):
            continue
        for period, res in periods.items():
            if isinstance(res, dict) and 'rank' in res:
                is_buy = 'BUY' in res.get('recommendation', '')
                if (not hide_non_buys or is_buy) and \
                        (hide_ranks_above == 0 or res['rank'] < hide_ranks_above):
                    filtered[symbol] = periods
                    break
    if filtered:
        results = dict(results)
        results['symbols'] = filtered
    return results


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _json_response(payload, status=200):
    return app.response_class(
        response=json.dumps(payload, cls=NpEncoder),
        status=status, mimetype='application/json')


# -- background refresher -----------------------------------------------

def _background_refresher():
    time.sleep(10)  # let the server come up first
    while True:
        interval = 1800
        try:
            config = load_config()
            interval = int(config['settings'].get('refreshInterval', 1800))
            symbols = watchlist_symbols(config)
            print(f"background scan: {len(symbols)} symbols")
            results = run_scan(symbols, ALL_PERIODS)
            results['movers'] = _compute_movers(results)
            run_id = store.save_run(symbols, results)
            fetched = fnd.refresh_some(store, symbols, max_fetches=8)
            logger.info("background scan complete (run %s, %s fundamentals refreshed)",
                        run_id, fetched)
        except Exception:
            logger.exception("background scan failed")
            interval = 600
        time.sleep(max(300, interval))


def start_background_refresher():
    thread = threading.Thread(target=_background_refresher,
                              daemon=True, name='refresher')
    thread.start()


# -- routes -------------------------------------------------------------

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


@app.route('/')
def index():
    logger.info("main page accessed")
    for directory in ('static/css', 'static/js'):
        os.makedirs(directory, exist_ok=True)
    # root-level asset sources get copied into static/ when present
    for js_file in ('config.js', 'portfolio.js', 'main.js', 'utils.js'):
        try:
            if os.path.exists(js_file):
                with open(js_file) as src, \
                        open(os.path.join('static/js', js_file), 'w') as dest:
                    dest.write(src.read())
        except Exception as e:
            logger.warning("could not copy %s: %s", js_file, e)
    return render_template_string(BASE_TEMPLATE)


@app.route('/api/config/get', methods=['GET'])
def get_config():
    return jsonify(load_config())


@app.route('/api/config/save', methods=['POST'])
def save_config_api():
    try:
        return jsonify(save_config(request.json))
    except Exception as e:
        logger.error("error saving configuration: %s", e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_stocks():
    try:
        data = request.json
        symbols = [s.strip().upper() for s in data.get('symbols', []) if s.strip()]
        periods = data.get('periodsToAnalyze', ALL_PERIODS)
        is_portfolio = data.get('isPortfolio', False)
        if not symbols:
            return jsonify({'error': 'No symbols provided'}), 400

        config = load_config()
        print(f"analyze request: {len(symbols)} symbols, portfolio={is_portfolio}")
        logger.info("analyzing %s symbols, periods %s", len(symbols), periods)

        if is_portfolio:
            results = run_scan(symbols, periods)
            results['portfolioRisk'] = portfolio_risk(
                config.get('portfolio', {}).get('positions'), store)
        else:
            results = _serve_watchlist(symbols, periods, config)
            results = _filter_results(results, config)

        results['market'] = market_overlay(store)
        results.pop('scores', None)
        return _json_response(results)
    except Exception as e:
        logger.error("error in analyze_stocks: %s", e)
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


def _serve_watchlist(symbols, periods, config):
    """Serve from the last stored run when it is fresh and covers the
    request; otherwise scan now and store the result. The lock spans the
    cache check so concurrent page loads don't both trigger a full scan."""
    with scan_lock:
        max_age = int(config['settings'].get('refreshInterval', 1800))
        latest = store.latest_run(max_age_seconds=max_age)
        if latest:
            run_id, ts, universe, stored = latest
            if set(symbols) <= set(universe) and \
                    all(_has_period(stored, universe, p) for p in periods):
                results = {
                    'symbols': {s: stored['symbols'][s] for s in symbols
                                if s in stored.get('symbols', {})},
                    'movers': stored.get('movers', []),
                    'cached': True,
                    'asOf': dt.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M'),
                }
                print(f"  served from run {run_id} ({results['asOf']})")
                return results

        results = run_scan(symbols, periods)
        results['movers'] = _compute_movers(results)
        store.save_run(symbols, results)
        return results


def _has_period(stored, universe, period):
    for symbol in universe:
        periods = stored.get('symbols', {}).get(symbol)
        if isinstance(periods, dict) and period in periods:
            return True
    return False


@app.route('/api/recommendation', methods=['GET'])
def recommendation():
    """Suggested portfolio (max 20 names) built from the latest scan."""
    try:
        config = load_config()
        symbols = watchlist_symbols(config)
        latest = store.latest_run()
        if latest:
            _, ts, _, results = latest
            as_of = dt.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')
        else:
            results = run_scan(symbols, ALL_PERIODS)
            store.save_run(symbols, results)
            as_of = dt.datetime.now().strftime('%Y-%m-%d %H:%M')

        prev = store.get_meta_json('last_recommendation') or {}
        market = market_overlay(store)
        if request.args.get('gate', 'on') == 'off':
            market = dict(market, exposure=1.0)
        rec = recommend(
            results, store,
            max_names=config['settings'].get('recommendationSize', 20),
            prev_symbols=prev.get('symbols'),
            positions=config.get('portfolio', {}).get('positions'),
            base_value=request.args.get('base', type=float),
            market=market)
        if rec is None:
            return jsonify({'holdings': [], 'asOf': as_of,
                            'note': 'No names with a positive score to recommend right now.'})

        store.set_meta_json('last_recommendation', {
            'symbols': [h['symbol'] for h in rec['holdings']],
            'ts': time.time(),
        })
        rec['asOf'] = as_of
        return _json_response(rec)
    except Exception as e:
        logger.error("error in recommendation: %s", e)
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/history/<symbol>', methods=['GET'])
def symbol_history(symbol):
    period = request.args.get('period', '1m')
    return jsonify({
        'symbol': symbol.upper(),
        'period': period,
        'history': store.score_history(symbol.upper(), period),
    })


@app.route('/api/market', methods=['GET'])
def market_state():
    return jsonify(market_overlay(store))


@app.route('/api/review', methods=['GET'])
def review_recommendation():
    """Deterministic risk checks on the current recommendation, ranked and
    explained by an LLM if one is reachable.

    The checks are computed in Python; the model only ranks and narrates
    them and cannot change whether the review blocks. See checks.py for why.
    """
    try:
        config = load_config()
        latest = store.latest_run()
        if not latest:
            return jsonify({'error': 'No scan yet. Load the dashboard first.'}), 404
        _, _, _, results = latest
        market = market_overlay(store)
        prev = store.get_meta_json('last_recommendation') or {}
        rec = recommend(
            results, store,
            max_names=config['settings'].get('recommendationSize', 20),
            prev_symbols=prev.get('symbols'),
            positions=config.get('portfolio', {}).get('positions'),
            market=market)
        if rec is None:
            return jsonify({'checks': [], 'concerns': [], 'blocking': False,
                            'summary': 'Nothing recommended right now.'})
        use_llm = request.args.get('llm', 'on') != 'off'
        checks = run_checks(rec, results=results, store=store, market=market)
        out = review(rec, checks, use_llm=use_llm)
        if request.args.get('log', 'on') != 'off':
            log_flags(rec, out)
        return _json_response(out)
    except Exception as e:
        logger.error("error in review: %s", e)
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/brief', methods=['GET'])
def morning_brief():
    """Plain-language narration of the latest scan.

    Every sentence is checked against the facts it cites before it is
    returned; `rejected` lists what was thrown away and why. The brief adds
    no information - it restates numbers the dashboard already shows.
    """
    try:
        config = load_config()
        latest = store.latest_run()
        if not latest:
            return jsonify({'error': 'No scan yet. Load the dashboard first.'}), 404
        _, ts, _, results = latest
        market = market_overlay(store)
        prev = store.get_meta_json('last_recommendation') or {}
        rec = recommend(
            results, store,
            max_names=config['settings'].get('recommendationSize', 20),
            prev_symbols=prev.get('symbols'),
            positions=config.get('portfolio', {}).get('positions'),
            market=market)
        backtest = None
        path = os.path.join('results', 'backtest_summary.json')
        if os.path.exists(path):
            try:
                with open(path) as f:
                    backtest = json.load(f)
            except (OSError, ValueError):
                backtest = None
        payload = generate(
            results=results, rec=rec, market=market, backtest=backtest,
            movers=results.get('movers'),
            use_llm=request.args.get('llm', 'on') != 'off')
        payload['asOf'] = dt.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')
        return _json_response(payload)
    except Exception as e:
        logger.error("error in brief: %s", e)
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/backtest', methods=['GET'])
def backtest_results():
    path = os.path.join('results', 'backtest_summary.json')
    if not os.path.exists(path):
        return jsonify({'error': 'No backtest results. Run: python backtest.py'}), 404
    with open(path) as f:
        return jsonify(json.load(f))


if __name__ == '__main__':
    print("stock scanner starting on http://localhost:5000")
    print("background refresher scans the watchlist on the configured interval")
    if os.path.exists(os.path.join('results', 'regime_weights.json')):
        print("using calibrated regime weights from results/regime_weights.json")
    logger.info("starting stock scanner")
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not app.debug:
        start_background_refresher()
    app.run(debug=True, port=5000)
