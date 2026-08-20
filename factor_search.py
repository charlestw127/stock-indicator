"""
LLM-proposed factor expressions, judged by the existing backtest machinery.

This is the one use of a language model in the trading literature that
survives contact with an honest evaluation. The multi-agent traders
(TradingAgents, FinMem, FinAgent, FinCon) report Sharpe ratios of 2-8 on a
handful of tickers over a few months, and every replication that controlled
for the model's training cutoff found the edge gone: FinSABER over twenty
years, Profit Mirage before and after the cutoff, StockBench on post-cutoff
data, The Alpha Illusion once costs are charged. What does hold up in
structure - though not in reported magnitude - is AlphaAgent (KDD 2025) and
RD-Agent(Q) (NeurIPS 2025): let the model write candidate factor formulas,
and let a deterministic backtester decide.

The design follows from the failure modes rather than from the successes:

- The model is BLIND. It sees the operator list, the sleeve descriptions and
  a table of previously scored expressions with their statistics. It never
  sees a price, a date, a ticker or a return. It cannot recall 2021 because
  it is never told which year it is looking at.
- Splits are FROZEN before the first proposal, and the holdout takes an
  explicit flag to open. Bailey et al. (AMS 2014) showed that with five
  years of data, more than ~45 configurations is enough to manufacture an
  in-sample Sharpe of 1 with an out-of-sample expectation of zero.
- Every expression scored is COUNTED, survivor or not, in
  results/trials.jsonl. The deflated t-hurdle uses that count. Forgetting to
  log a failure is how a t of 2.5 becomes meaningless.
- Survivors must be ORIGINAL (low correlation to the five existing sleeves)
  and STABLE (second-half IC at least half the first-half, same sign).
  AlphaAgent's regularizers exist because unconstrained search finds
  redundant, decaying factors; AlphaEval (KDD 2026) independently scored
  LLM-mined alphas lowest of any method on perturbation robustness.

Expected yield, stated in advance so it cannot be rationalised later: zero
to two factors, with a discovery t around 2.5-3 that holds sign on the
sealed holdout. Anything resembling the 0.05-0.15 ICs in the papers means
the split leaked, not that the search worked. With 80 names the per-date IC
standard error is about 1/sqrt(79) = 0.11.

Usage:
    python factor_search.py --propose 20            # LLM proposals
    python factor_search.py --propose 20 --random   # no API key needed
    python factor_search.py --review                # re-score saved survivors
    python factor_search.py --open-holdout          # spend the sealed window
"""

import argparse
import ast
import json
import math
import os
import time

import numpy as np
import pandas as pd

import evaluation as ev
import llm
from backtest import MIN_HISTORY, STRATEGIES, load_universe
from data_store import DataStore

SLEEVES = STRATEGIES[1:]
CANDIDATES_PATH = os.path.join('results', 'factor_candidates.jsonl')
SURVIVORS_PATH = os.path.join('results', 'factor_survivors.json')

# Frozen before any proposal is made. Change these and the gate is void.
#
# Re-set once, when backfill.py took the cache from five years to fifteen.
# The previous values (2024-08-30 / 2025-08-18) were chosen to fit the
# shallower cache and gave discovery under two years, about 22 monthly
# rebalances. Any expression scored under those splits was measured on a
# different dataset and its statistics do not carry over; results/
# factor_candidates.jsonl keeps them for the record, not as evidence.
#
# Re-freezing because the underlying data changed is legitimate. Re-freezing
# after seeing which side of a boundary a favoured expression falls on is
# not, and is the single easiest way to void everything below.
DISCOVERY_END = '2021-07-29'       # ~8.7y of discovery after the warm-up
VALIDATION_END = '2024-05-10'      # ~2.8y; holdout is everything after this

MAX_DEPTH = 4
WINDOWS = [3, 5, 10, 20, 21, 42, 63, 126, 252]

# Acceptance thresholds, also frozen in advance.
GATE = {
    'discovery_t': 2.5,
    'validation_t': 1.5,
    'decay_ratio': 0.5,       # second-half IC / first-half IC
    'max_sleeve_corr': 0.6,
    'min_positive_dates': 0.55,
}


# -- the expression language -------------------------------------------

def _safe_div(a, b):
    out = a / b.replace(0.0, np.nan) if isinstance(b, pd.Series) else a / b
    return out.replace([np.inf, -np.inf], np.nan) if isinstance(out, pd.Series) else out


OPERATORS = {
    'ts_mean': (lambda x, w: x.rolling(int(w), min_periods=int(w)).mean(),
                'rolling mean over w bars'),
    'ts_std': (lambda x, w: x.rolling(int(w), min_periods=int(w)).std(),
               'rolling standard deviation'),
    'ts_min': (lambda x, w: x.rolling(int(w), min_periods=int(w)).min(),
               'rolling minimum'),
    'ts_max': (lambda x, w: x.rolling(int(w), min_periods=int(w)).max(),
               'rolling maximum'),
    'ts_sum': (lambda x, w: x.rolling(int(w), min_periods=int(w)).sum(),
               'rolling sum'),
    'ts_delta': (lambda x, w: x - x.shift(int(w)),
                 'change over w bars: x - x[w bars ago]'),
    'ts_rank': (lambda x, w: x.rolling(int(w), min_periods=int(w))
                .apply(lambda a: (a <= a[-1]).mean(), raw=True),
                'where the latest value sits within its own last w bars, 0 to 1'),
    'pct_change': (lambda x, w: x.pct_change(int(w))
                   .replace([np.inf, -np.inf], np.nan),
                   'fractional change over w bars'),
    'zscore': (lambda x, w: _safe_div(
        x - x.rolling(int(w), min_periods=int(w)).mean(),
        x.rolling(int(w), min_periods=int(w)).std()),
        'standardised against the last w bars'),
    'tanh': (lambda x: np.tanh(x), 'squash to (-1, 1)'),
    'sign': (lambda x: np.sign(x), '-1, 0 or 1'),
    'absv': (lambda x: abs(x), 'absolute value'),
    'logv': (lambda x: np.log(x.clip(lower=1e-9)) if isinstance(x, pd.Series)
             else math.log(max(x, 1e-9)), 'natural log, floored at 1e-9'),
    'clip': (lambda x, lo, hi: x.clip(lo, hi), 'clamp between lo and hi'),
}

# Every operand is already causal in quant_indicators, so an expression
# built from them cannot see the future by construction.
OPERANDS = {
    'close': 'closing price',
    'high': 'session high',
    'low': 'session low',
    'volume': 'share volume',
    'ret1': 'one-bar fractional return',
    'atr14': 'average true range, 14 bars (Wilder)',
    'rv21': 'annualised realised volatility, 21 bars',
    'kama10': 'Kaufman adaptive moving average',
    'mfi14': 'money flow index, 0-100',
    'cmf20': 'Chaikin money flow, roughly -1 to 1',
    'obv_z': 'on-balance-volume z-score',
    'vwap_dev': 'percent deviation from 20-bar VWAP',
    'rel_vol': 'volume over its own 20-bar average',
    'don_pos': 'position in the 20-bar Donchian channel, -1 to 1',
    'dist_52': 'percent below the 52-week high (0 at the high)',
    'mom_12_1': '12-month return skipping the last month',
    'macd_hist': 'MACD histogram',
    'bbp20': 'Bollinger %B, 0-1',
    'stoch_k': 'stochastic %K, 0-100',
    'stoch_d': 'stochastic %D, 0-100',
    'adx14': 'ADX, trend strength, 0-100',
}


class ExpressionError(ValueError):
    """The expression is outside the language."""


def validate(expr, max_depth=MAX_DEPTH):
    """Parse and whitelist-check. Returns the compiled AST.

    A rejected expression costs nothing; an accepted one that reaches into
    the interpreter costs everything, so this is a whitelist and not a
    blacklist. Attribute access, subscripting, comprehensions, lambdas,
    imports and names outside OPERANDS/OPERATORS are all refused.
    """
    try:
        tree = ast.parse(expr.strip(), mode='eval')
    except SyntaxError as e:
        raise ExpressionError(f"will not parse: {e}") from e

    def walk(node, depth):
        if depth > max_depth:
            raise ExpressionError(f"deeper than {max_depth} levels")
        if isinstance(node, ast.Expression):
            return walk(node.body, depth)
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                raise ExpressionError("only + - * / are allowed")
            walk(node.left, depth + 1)
            walk(node.right, depth + 1)
            return
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, ast.USub):
                raise ExpressionError("only unary minus is allowed")
            walk(node.operand, depth + 1)
            return
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in OPERATORS:
                raise ExpressionError("unknown function")
            if node.keywords:
                raise ExpressionError("keyword arguments are not allowed")
            for arg in node.args:
                walk(arg, depth + 1)
            return
        if isinstance(node, ast.Name):
            if node.id not in OPERANDS:
                raise ExpressionError(f"unknown operand '{node.id}'")
            return
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise ExpressionError("only numeric constants are allowed")
            return
        raise ExpressionError(f"{type(node).__name__} is not allowed here")

    walk(tree, 0)
    return tree


def canonical(expr):
    """Normalised form, so the same idea written twice is caught once."""
    return ast.dump(validate(expr))


def operand_series(an):
    """The operand namespace for one symbol."""
    return {
        'close': an.close, 'high': an.high, 'low': an.low,
        'volume': an.volume, 'ret1': an.close.pct_change(),
        'atr14': an.atr14, 'rv21': an.rv21, 'kama10': an.kama10,
        'mfi14': an.mfi14, 'cmf20': an.cmf20, 'obv_z': an.obv_z,
        'vwap_dev': an.vwap_dev, 'rel_vol': an.rel_vol,
        'don_pos': an.don_pos, 'dist_52': an.dist_52,
        'mom_12_1': an.mom_12_1, 'macd_hist': an.macd_hist,
        'bbp20': an.bbp20, 'stoch_k': an.stoch_k, 'stoch_d': an.stoch_d,
        'adx14': an.adx14,
    }


def evaluate(expr, an):
    """Evaluate a validated expression into a series for one symbol."""
    tree = validate(expr)
    env = {name: fn for name, (fn, _) in OPERATORS.items()}
    env.update(operand_series(an))
    code = compile(tree, '<factor>', 'eval')
    try:
        out = eval(code, {'__builtins__': {}}, env)  # noqa: S307 - AST whitelisted
    except Exception as e:
        raise ExpressionError(f"failed to evaluate: {e}") from e
    if not isinstance(out, pd.Series):
        raise ExpressionError("expression must produce a series, not a scalar")
    return out.replace([np.inf, -np.inf], np.nan)


def describe_language():
    """The DSL as the model sees it. No prices, no dates, no tickers."""
    lines = ["OPERANDS (each is a causal daily series for one instrument):"]
    for name, doc in OPERANDS.items():
        lines.append(f"  {name}: {doc}")
    lines.append("")
    lines.append("OPERATORS:")
    for name, (fn, doc) in OPERATORS.items():
        arity = fn.__code__.co_argcount
        args = {1: '(x)', 2: '(x, w)', 3: '(x, lo, hi)'}.get(arity, '(...)')
        lines.append(f"  {name}{args}: {doc}")
    lines.append("")
    lines.append(f"  arithmetic: + - * / and unary minus")
    lines.append(f"  window arguments w must come from {WINDOWS}")
    lines.append(f"  maximum nesting depth {MAX_DEPTH}")
    return "\n".join(lines)


# -- the evaluation panel ----------------------------------------------

def build_panel(analyzers, calendar, period, step, start_pos, end_pos):
    """Scores, forward returns and bar indices at every rebalance.

    Built once and reused for every candidate, so scoring an expression
    costs one pass over each symbol's history rather than a re-run of the
    backtest.
    """
    positions = list(range(start_pos, end_pos - step, step))
    if len(positions) < 10:
        raise ValueError("not enough rebalances in this window")
    dates = [calendar[p] for p in positions]
    symbols = sorted(analyzers)

    n_d, n_s = len(dates), len(symbols)
    bar = np.full((n_d, n_s), -1, dtype=int)
    fwd = np.full((n_d, n_s), np.nan)
    sleeves = {s: np.full((n_d, n_s), np.nan) for s in SLEEVES}

    for si, symbol in enumerate(symbols):
        an = analyzers[symbol]
        idx = an.index.get_indexer(dates, method='pad')
        for k, i in enumerate(idx):
            if i < MIN_HISTORY - 1:
                continue
            if (dates[k] - an.index[i]).days > 10:
                continue
            j = an.index.get_indexer([calendar[positions[k] + step]], method='pad')[0]
            if j <= i:
                continue
            c0, c1 = float(an.close.iloc[i]), float(an.close.iloc[j])
            if c0 <= 0:
                continue
            bar[k, si] = i
            fwd[k, si] = c1 / c0 - 1.0
            res = an.score_at(period, i)
            for s in SLEEVES:
                sleeves[s][k, si] = res['factors'][s]

    return {'dates': dates, 'symbols': symbols, 'bar': bar, 'fwd': fwd,
            'sleeves': sleeves, 'positions': positions, 'step': step}


def _rank(a):
    order = np.argsort(np.argsort(a))
    return order.astype(float)


def _ic_series(values, fwd, min_names=8):
    """Per-date Spearman IC between a value matrix and forward returns."""
    ics = []
    for k in range(values.shape[0]):
        v, f = values[k], fwd[k]
        mask = np.isfinite(v) & np.isfinite(f)
        if mask.sum() < min_names:
            continue
        x, y = _rank(v[mask]), _rank(f[mask])
        if x.std() == 0 or y.std() == 0:
            continue
        ics.append(float(np.corrcoef(x, y)[0, 1]))
    return np.asarray(ics)


def _summarize(ics):
    if len(ics) < 5:
        return None
    mean, sd = float(ics.mean()), float(ics.std(ddof=1))
    half = len(ics) // 2
    first = float(ics[:half].mean())
    second = float(ics[half:].mean())
    decay = second / first if abs(first) > 1e-9 else 0.0
    return {
        'mean_ic': round(mean, 5),
        't_stat': round(mean / sd * math.sqrt(len(ics)), 3) if sd > 0 else None,
        'pct_positive': round(float((ics > 0).mean()), 3),
        'first_half_ic': round(first, 5),
        'second_half_ic': round(second, 5),
        'decay_ratio': round(decay, 3),
        'n_dates': len(ics),
    }


def expression_values(expr, analyzers, panel):
    """Matrix of expression values aligned to the panel."""
    out = np.full(panel['fwd'].shape, np.nan)
    for si, symbol in enumerate(panel['symbols']):
        try:
            series = evaluate(expr, analyzers[symbol]).to_numpy(dtype=float)
        except ExpressionError:
            continue
        bars = panel['bar'][:, si]
        ok = bars >= 0
        idx = bars[ok]
        idx = np.where(idx < len(series), idx, len(series) - 1)
        out[ok, si] = series[idx]
    return out


def score_expression(expr, analyzers, panel):
    """Full report card for one candidate on one window."""
    values = expression_values(expr, analyzers, panel)
    coverage = float(np.isfinite(values).mean())
    stats = _summarize(_ic_series(values, panel['fwd']))
    if stats is None:
        return {'error': 'too few usable dates', 'coverage': round(coverage, 3)}
    corr = {}
    for s in SLEEVES:
        pairs = []
        for k in range(values.shape[0]):
            v, w = values[k], panel['sleeves'][s][k]
            mask = np.isfinite(v) & np.isfinite(w)
            if mask.sum() >= 8 and v[mask].std() > 0 and w[mask].std() > 0:
                pairs.append(float(np.corrcoef(_rank(v[mask]),
                                               _rank(w[mask]))[0, 1]))
        corr[s] = round(float(np.mean(pairs)), 3) if pairs else None
    stats['coverage'] = round(coverage, 3)
    stats['sleeve_corr'] = corr
    finite = [abs(c) for c in corr.values() if c is not None]
    stats['max_sleeve_corr'] = round(max(finite), 3) if finite else None
    return stats


def passes_discovery(stats):
    """The frozen acceptance rule, applied to the discovery window."""
    if not stats or stats.get('error'):
        return False, 'not scoreable'
    t = stats.get('t_stat')
    if t is None or abs(t) < GATE['discovery_t']:
        return False, f"|t| {abs(t) if t else 0:.2f} < {GATE['discovery_t']}"
    if stats['coverage'] < 0.5:
        return False, f"covers only {stats['coverage']:.0%} of the panel"
    decay = stats['decay_ratio']
    if decay < GATE['decay_ratio']:
        return False, f"decays: second half is {decay:.2f} of the first"
    pos = stats['pct_positive'] if t > 0 else 1.0 - stats['pct_positive']
    if pos < GATE['min_positive_dates']:
        return False, f"only {pos:.0%} of dates agree in sign"
    msc = stats.get('max_sleeve_corr')
    if msc is not None and msc > GATE['max_sleeve_corr']:
        return False, f"correlated {msc:.2f} with an existing sleeve"
    return True, 'passes discovery'


# -- proposals ---------------------------------------------------------

PROPOSAL_SCHEMA = {
    'type': 'object',
    'properties': {
        'proposals': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'thesis': {'type': 'string'},
                    'expression': {'type': 'string'},
                },
                'required': ['thesis', 'expression'],
                'additionalProperties': False,
            },
        },
    },
    'required': ['proposals'],
    'additionalProperties': False,
}

SYSTEM = """You propose candidate cross-sectional equity factors as formulas \
in a restricted language. A deterministic walk-forward backtester scores \
every proposal; you never see its data.

Rules you must follow:
- Use only the listed operands and operators. Window arguments must come \
from the allowed list. Stay within the nesting depth.
- Each proposal needs a one-sentence economic thesis that the formula \
actually implements. A thesis that does not match its formula is rejected.
- Aim for LOW correlation with the five existing sleeves described below. A \
proposal that restates an existing sleeve is worthless even if it scores \
well.
- Prefer ratios and standardised quantities over raw levels: the factor is \
ranked across instruments of very different prices and volatilities, so \
anything scale-dependent will rank by market cap instead of by signal.
- Vary the ideas. Do not submit ten variations of one formula with \
different windows.

You will not be told which instruments, which dates, or what happened. Do \
not speculate about any of that; propose from economic reasoning only."""


def propose_llm(history, k, effort='high'):
    """Ask the model for k new expressions."""
    sleeve_doc = """The five existing sleeves, which new factors should NOT
duplicate:
  trend: moving-average structure, price vs KAMA, regression trend graded by
    fit quality, Donchian channel position
  momentum: risk-adjusted momentum t-stat, total-return momentum, MACD
    normalised by ATR, proximity to the 52-week high, 12-1 momentum
  mean_reversion: price z-score, RSI, Bollinger %B, stochastic, scored
    contrarian so oversold is positive
  volume_flow: money flow index, Chaikin money flow, OBV trend, VWAP
    deviation
  quality: rolling Sharpe, drawdown depth, volatility regime"""

    if history:
        rows = ["Previously scored proposals (discovery window):",
                "expression | mean IC | t | decay | max sleeve corr | verdict"]
        for h in history[-40:]:
            s = h.get('discovery') or {}
            rows.append(
                f"  {h['expression']} | {s.get('mean_ic')} | {s.get('t_stat')} "
                f"| {s.get('decay_ratio')} | {s.get('max_sleeve_corr')} "
                f"| {h.get('verdict')}")
        history_doc = "\n".join(rows)
    else:
        history_doc = "No proposals have been scored yet."

    prompt = (f"{describe_language()}\n\n{sleeve_doc}\n\n{history_doc}\n\n"
              f"Propose {k} new expressions. Learn from what scored badly: "
              f"if a family of ideas keeps failing, change family rather "
              f"than tuning its windows.")
    payload = llm.structured(SYSTEM, prompt, PROPOSAL_SCHEMA, effort=effort)
    return payload.get('proposals', [])


def propose_random(k, seed=None, max_tries=400):
    """Sample the grammar directly.

    Present so the machinery can be exercised and tested without an API key,
    and so that any claim the LLM proposals are better has something to be
    better than. Comparing the two hit rates is the only way to know whether
    the model is contributing anything.
    """
    rng = np.random.default_rng(seed)
    unary = [n for n, (f, _) in OPERATORS.items() if f.__code__.co_argcount == 1]
    windowed = [n for n, (f, _) in OPERATORS.items()
                if f.__code__.co_argcount == 2]
    operands = list(OPERANDS)

    def atom():
        return str(rng.choice(operands))

    def term(depth=0):
        r = rng.random()
        if depth >= 2 or r < 0.25:
            return atom()
        if r < 0.7:
            return (f"{rng.choice(windowed)}({term(depth + 1)}, "
                    f"{rng.choice(WINDOWS)})")
        return f"{rng.choice(unary)}({term(depth + 1)})"

    out, seen = [], set()
    for _ in range(max_tries):
        if len(out) >= k:
            break
        op = str(rng.choice(['-', '/', '*', '+']))
        expr = f"{term()} {op} {term()}"
        try:
            key = canonical(expr)
        except ExpressionError:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append({'thesis': 'randomly sampled from the grammar',
                    'expression': expr})
    return out


# -- persistence -------------------------------------------------------

def load_history(path=CANDIDATES_PATH):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def append_candidate(row, path=CANDIDATES_PATH):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(row, default=str) + '\n')


# -- driver ------------------------------------------------------------

def _split_positions(calendar, discovery_end, validation_end):
    d_end = calendar.get_indexer([pd.Timestamp(discovery_end)], method='pad')[0]
    v_end = calendar.get_indexer([pd.Timestamp(validation_end)], method='pad')[0]
    if d_end <= MIN_HISTORY or v_end <= d_end:
        raise SystemExit(
            "not enough cached history for the frozen splits; the discovery "
            "window needs at least 300 bars before " + str(discovery_end))
    return d_end, v_end


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--propose', type=int, default=10,
                        help='how many new expressions to score')
    parser.add_argument('--random', action='store_true',
                        help='sample the grammar instead of calling the LLM')
    parser.add_argument('--rounds', type=int, default=1,
                        help='proposal rounds; later rounds see earlier scores')
    parser.add_argument('--period', default='1m')
    parser.add_argument('--step', type=int, default=21)
    parser.add_argument('--symbols', default=None)
    parser.add_argument('--effort', default='high',
                        choices=['low', 'medium', 'high', 'xhigh', 'max'])
    parser.add_argument('--discovery-end', default=DISCOVERY_END)
    parser.add_argument('--validation-end', default=VALIDATION_END)
    parser.add_argument('--open-holdout', action='store_true',
                        help='score current survivors on the sealed window; '
                             'do this once, and treat the answer as final')
    parser.add_argument('--review', action='store_true',
                        help='re-score saved survivors without proposing')
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

    d_end, v_end = _split_positions(calendar, args.discovery_end,
                                    args.validation_end)
    print(f"splits: discovery to {calendar[d_end].date()}, "
          f"validation to {calendar[v_end].date()}, "
          f"holdout {calendar[min(v_end + 1, len(calendar) - 1)].date()} "
          f"onward ({'OPEN' if args.open_holdout else 'sealed'})")

    print("building panels...")
    t0 = time.time()
    discovery = build_panel(analyzers, calendar, args.period, args.step,
                            MIN_HISTORY, d_end)
    validation = build_panel(analyzers, calendar, args.period, args.step,
                             d_end, v_end)
    print(f"  discovery {len(discovery['dates'])} rebalances, "
          f"validation {len(validation['dates'])} ({time.time() - t0:.0f}s)")

    history = load_history()
    survivors = []

    if not args.review:
        for rnd in range(args.rounds):
            # vary the seed with what has already been tried, so a rerun
            # explores instead of re-proposing the same dozen expressions
            seed = len(history) * 1000 + rnd
            if args.random:
                proposals = propose_random(args.propose, seed=seed)
                source = 'random'
            else:
                try:
                    proposals = propose_llm(history, args.propose, args.effort)
                    source = 'llm'
                except llm.LLMUnavailable as e:
                    print(f"  LLM unavailable ({e}); falling back to random "
                          f"sampling")
                    proposals = propose_random(args.propose, seed=seed)
                    source = 'random'

            print(f"\nround {rnd + 1}: scoring {len(proposals)} proposals "
                  f"({source})")
            seen = {h.get('canonical') for h in history}
            for p in proposals:
                expr = (p.get('expression') or '').strip()
                try:
                    key = canonical(expr)
                except ExpressionError as e:
                    print(f"  rejected  {expr[:60]:<60} {e}")
                    append_candidate({'expression': expr, 'source': source,
                                      'verdict': f'invalid: {e}'})
                    history.append({'expression': expr, 'verdict': 'invalid'})
                    continue
                if key in seen:
                    print(f"  duplicate {expr[:60]}")
                    continue
                seen.add(key)

                stats = score_expression(expr, analyzers, discovery)
                ok, why = passes_discovery(stats)
                # every scored expression counts toward the hurdle
                ev.log_trial({'job': 'factor_search', 'expression': expr,
                              'period': args.period, 'step': args.step},
                             {'discovery_t': (stats or {}).get('t_stat')})
                row = {'expression': expr, 'thesis': p.get('thesis'),
                       'canonical': key, 'source': source,
                       'discovery': stats, 'verdict': why}
                if ok:
                    val = score_expression(expr, analyzers, validation)
                    row['validation'] = val
                    vt = (val or {}).get('t_stat')
                    dt = stats.get('t_stat')
                    same_sign = (vt is not None and dt is not None
                                 and np.sign(vt) == np.sign(dt))
                    if vt is not None and abs(vt) >= GATE['validation_t'] \
                            and same_sign:
                        row['verdict'] = 'SURVIVOR'
                        survivors.append(row)
                    else:
                        row['verdict'] = (
                            f"failed validation (t {vt}, discovery t {dt})")
                mark = 'PASS' if row['verdict'] == 'SURVIVOR' else '    '
                t = (stats or {}).get('t_stat')
                print(f"  {mark} {expr[:56]:<56} t={t if t is not None else '-':>6} "
                      f"{row['verdict']}")
                append_candidate(row)
                history.append(row)

    if args.review or not survivors:
        survivors = [h for h in load_history() if h.get('verdict') == 'SURVIVOR']

    n_trials = ev.count_trials()
    hurdle = ev.expected_max_t(n_trials)
    print(f"\n{len(survivors)} survivor(s); {n_trials} expressions scored in "
          f"total, so the best-of-N t hurdle is {hurdle:.2f}")
    for row in survivors:
        d = row.get('discovery') or {}
        v = row.get('validation') or {}
        clears = (d.get('t_stat') is not None
                  and abs(d['t_stat']) > hurdle)
        print(f"  {row['expression']}")
        print(f"    thesis      {row.get('thesis')}")
        print(f"    discovery   IC {d.get('mean_ic')} t {d.get('t_stat')} "
              f"decay {d.get('decay_ratio')} corr {d.get('max_sleeve_corr')}")
        print(f"    validation  IC {v.get('mean_ic')} t {v.get('t_stat')}")
        print(f"    vs hurdle   {'clears' if clears else 'DOES NOT CLEAR'} "
              f"the multiple-testing bar")

    if args.open_holdout and survivors:
        print("\nopening the sealed holdout - this is a one-time spend")
        holdout = build_panel(analyzers, calendar, args.period, args.step,
                              v_end, len(calendar))
        for row in survivors:
            row['holdout'] = score_expression(row['expression'], analyzers,
                                              holdout)
            h = row['holdout']
            print(f"  {row['expression']}")
            print(f"    holdout     IC {h.get('mean_ic')} t {h.get('t_stat')}")

    os.makedirs('results', exist_ok=True)
    with open(SURVIVORS_PATH, 'w') as f:
        json.dump({'survivors': survivors, 'n_trials': n_trials,
                   'hurdle_t': round(hurdle, 3), 'gate': GATE,
                   'discovery_end': args.discovery_end,
                   'validation_end': args.validation_end}, f, indent=2,
                  default=str)
    print(f"\n{SURVIVORS_PATH} written")
    if not survivors:
        print("no survivors is the expected outcome, not a failure - see the "
              "yield note at the top of this file")


if __name__ == '__main__':
    main()
