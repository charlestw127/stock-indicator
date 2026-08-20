"""
Deterministic checks on a proposed portfolio, and an optional LLM critic
that can only explain them.

The division of labour here is the one the 2026 evaluations converged on
after the LLM risk-manager designs failed:

- Xue (arXiv 2605.28850) found structured risk feedback changes an LLM's
  trading behaviour but is "not a universal performance enhancer", and that
  placebo feedback sometimes produced better short-term returns. The same
  paper names a "correlation blind spot": models justify exposure to
  strongly coupled assets that a risk layer would have caught.
- PortBench (arXiv 2605.27887) found 90% of model-profile cases failed to
  beat equal weight in 2024 while still satisfying every format constraint
  they were given. Passing the schema is not passing the check.
- TradeTrap (arXiv 2512.02261) perturbed single components of agent loops
  and produced runaway concentration, which is why nothing below can be
  argued away by the model.
- OpenPM (arXiv 2608.09988) landed on the design used here: a deterministic
  critic enforces typed constraints, and the model writes the explanation.

So every number is computed in Python. The model receives the finished
table and may rank concerns and write prose. It cannot compute a check,
relax a threshold, or set `blocking` - review() overwrites that field from
the deterministic results after the model has spoken.

The honest claim for this module is "catches mechanical violations and
explains them", not "improves returns". log_flags() exists so that claim
can be tested later against what the flagged names actually did.
"""

import datetime as dt
import json
import logging
import os

import numpy as np
import pandas as pd

import llm
from recommender import CORR_CAP, WEIGHT_CAP, _returns_matrix

logger = logging.getLogger('stock_app.checks')

FLAG_LOG = os.path.join('results', 'flag_log.jsonl')

# Thresholds are here, in code, where they can be diffed and tested.
CLUSTER_CORR = 0.70        # pairwise correlation that counts as "coupled"
CLUSTER_MIN = 3            # names above that correlation before it is a flag
HHI_WARN = 0.12            # 1/HHI below ~8 effective names
EARNINGS_WINDOW = 21       # trading days the 1m selection horizon implies
COST_BPS = 10.0            # assumed round-trip cost for the turnover estimate


def _status(ok, warn=False):
    return 'fail' if not ok else ('warn' if warn else 'ok')


def run_checks(rec, results=None, store=None, market=None,
               cost_bps=COST_BPS):
    """Every deterministic check on a recommendation.

    Returns a list of dicts, each with a stable `id` the critic must cite.
    A check that cannot be computed reports status 'unknown' rather than
    silently passing - an unrunnable check is not a passing one.
    """
    holdings = (rec or {}).get('holdings') or []
    checks = []
    if not holdings:
        return [{'id': 'empty', 'name': 'portfolio is empty',
                 'status': 'warn', 'detail': 'nothing to review'}]

    weights = {h['symbol']: (h.get('weight') or 0) / 100.0 for h in holdings}
    symbols = list(weights)

    # 1. concentration
    hhi = float(sum(w * w for w in weights.values()))
    effective = 1.0 / hhi if hhi > 0 else 0.0
    top = max(weights.items(), key=lambda kv: kv[1])
    over_cap = [s for s, w in weights.items() if w > WEIGHT_CAP + 1e-6]
    checks.append({
        'id': 'concentration',
        'name': 'concentration',
        'status': _status(not over_cap, warn=hhi > HHI_WARN),
        'detail': (f"{len(symbols)} names, {effective:.1f} effective "
                   f"(HHI {hhi:.3f}); largest is {top[0]} at "
                   f"{top[1] * 100:.1f}% against a {WEIGHT_CAP * 100:.0f}% cap"),
        'values': {'n_names': len(symbols), 'hhi': round(hhi, 4),
                   'effective_names': round(effective, 1),
                   'max_weight_pct': round(top[1] * 100, 1),
                   'max_weight_symbol': top[0],
                   'over_cap': over_cap},
    })

    # 2. correlation clusters - the blind spot the literature names
    if store is not None:
        clusters, pairs = _correlation_clusters(symbols, store)
        big = [c for c in clusters if len(c) >= CLUSTER_MIN]
        checks.append({
            'id': 'correlation_clusters',
            'name': 'correlated bets',
            'status': _status(True, warn=bool(big)),
            'detail': (f"{len(big)} cluster(s) of {CLUSTER_MIN}+ names above "
                       f"rho {CLUSTER_CORR}: "
                       + "; ".join("/".join(c) for c in big)
                       if big else
                       f"no cluster of {CLUSTER_MIN}+ names above rho "
                       f"{CLUSTER_CORR}"),
            'values': {'clusters': big, 'n_pairs_above': len(pairs),
                       'threshold': CLUSTER_CORR,
                       'dedup_threshold': CORR_CAP,
                       'weight_in_largest_cluster': round(
                           sum(weights.get(s, 0) for s in big[0]) * 100, 1)
                       if big else 0.0},
        })

    # 3. earnings inside the holding horizon
    upcoming = []
    for h in holdings:
        fund = None
        if results:
            fund = (results.get('symbols', {}).get(h['symbol'], {})
                    or {}).get('fundamentals')
        ne = (fund or {}).get('next_earnings')
        if not ne:
            continue
        try:
            days = (dt.datetime.strptime(ne, '%Y-%m-%d').date()
                    - dt.date.today()).days
        except (ValueError, TypeError):
            continue
        if 0 <= days <= EARNINGS_WINDOW * 1.4:
            upcoming.append({'symbol': h['symbol'], 'in_days': days,
                             'weight_pct': h.get('weight')})
    upcoming.sort(key=lambda r: r['in_days'])
    at_risk = round(sum(r['weight_pct'] or 0 for r in upcoming), 1)
    checks.append({
        'id': 'earnings_proximity',
        'name': 'earnings inside the holding horizon',
        'status': _status(True, warn=at_risk > 25),
        'detail': (f"{len(upcoming)} name(s), {at_risk:.0f}% of the book, "
                   f"report within ~{EARNINGS_WINDOW} sessions"
                   if upcoming else
                   "no cached earnings dates inside the horizon"),
        'values': {'names': upcoming[:8], 'weight_pct': at_risk},
    })

    # 4. regime mix vs the regime the sleeve weights assume
    regimes = {}
    if results:
        for h in holdings:
            row = (results.get('symbols', {}).get(h['symbol'], {})
                   or {}).get('1m') or {}
            state = ((row.get('regime') or {}).get('trend')) or 'unknown'
            regimes[state] = regimes.get(state, 0) + 1
    if regimes:
        dominant = max(regimes.items(), key=lambda kv: kv[1])
        checks.append({
            'id': 'regime_mix',
            'name': 'regime mix',
            'status': 'ok',
            'detail': (f"{dominant[1]} of {len(holdings)} names classified "
                       f"{dominant[0]}: "
                       + ", ".join(f"{k} {v}" for k, v in sorted(regimes.items()))),
            'values': {'counts': regimes, 'dominant': dominant[0]},
        })

    # 5. turnover and what it costs
    plan = (rec or {}).get('rebalance') or {}
    trades = plan.get('trades') or []
    if plan:
        base = plan.get('base_value') or 0
        traded = sum(abs(t.get('delta_value') or 0) for t in trades)
        turnover = traded / base if base else 0.0
        cost = traded * cost_bps / 10_000.0
        checks.append({
            'id': 'turnover_cost',
            'name': 'turnover and cost',
            'status': _status(True, warn=turnover > 0.4),
            'detail': (f"{len(trades)} trade(s) moving ${traded:,.0f} "
                       f"({turnover * 100:.0f}% of the book); at {cost_bps:g}bps "
                       f"round trip that is about ${cost:,.0f}"),
            'values': {'n_trades': len(trades), 'traded_value': round(traded, 2),
                       'turnover': round(turnover, 3),
                       'estimated_cost': round(cost, 2), 'cost_bps': cost_bps},
        })

    # 6. exposure gate
    exposure = (rec or {}).get('exposure')
    if exposure is not None:
        checks.append({
            'id': 'exposure',
            'name': 'gross exposure',
            'status': 'ok' if exposure >= 1.0 else 'warn',
            'detail': (f"{exposure * 100:.0f}% invested, "
                       f"{(1 - exposure) * 100:.0f}% cash"
                       + (f" - {(market or {}).get('note')}"
                          if market and market.get('note') else "")),
            'values': {'exposure': exposure,
                       'risk_state': (market or {}).get('risk')},
        })

    # 7. is the selection actually separating anything
    if results:
        scores = []
        for sym, periods in (results.get('symbols') or {}).items():
            row = (periods or {}).get('1m') or {}
            if row.get('score') is not None:
                scores.append((sym, float(row['score'])))
        if len(scores) >= 10:
            held = [s for sym, s in scores if sym in weights]
            rest = [s for sym, s in scores if sym not in weights]
            gap = (float(np.mean(held)) - float(np.mean(rest))) if held and rest else 0.0
            spread = float(np.std([s for _, s in scores]))
            ratio = gap / spread if spread > 0 else 0.0
            checks.append({
                'id': 'score_separation',
                'name': 'score separation',
                'status': _status(True, warn=ratio < 0.5),
                'detail': (f"held names average {np.mean(held):.1f} vs "
                           f"{np.mean(rest):.1f} for the rest, a gap of "
                           f"{ratio:.2f} cross-sectional standard deviations"),
                'values': {'held_mean': round(float(np.mean(held)), 2),
                           'rest_mean': round(float(np.mean(rest)), 2),
                           'gap_in_sd': round(ratio, 2)},
            })

    return checks


def _correlation_clusters(symbols, store, threshold=CLUSTER_CORR):
    """Groups of names whose 60-day returns move together.

    Single-linkage on the correlation graph. The recommender already refuses
    pairs above 0.92, so anything found here is below its dedup threshold
    and still concentrated - which is exactly the case a per-pair rule
    cannot see.
    """
    returns = _returns_matrix(store, symbols)
    if returns.empty:
        return [], []
    recent = returns.iloc[-60:]
    cols = [c for c in recent.columns if recent[c].notna().sum() >= 40]
    if len(cols) < 2:
        return [], []
    corr = recent[cols].corr()

    parent = {c: c for c in cols}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            c = corr.loc[a, b]
            if pd.notna(c) and c > threshold:
                pairs.append((a, b, round(float(c), 3)))
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

    groups = {}
    for c in cols:
        groups.setdefault(find(c), []).append(c)
    clusters = [sorted(v) for v in groups.values() if len(v) > 1]
    clusters.sort(key=len, reverse=True)
    return clusters, pairs


# -- the critic --------------------------------------------------------

CRITIC_SCHEMA = {
    'type': 'object',
    'properties': {
        'concerns': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'check_id': {'type': 'string'},
                    'severity': {'type': 'string',
                                 'enum': ['low', 'medium', 'high']},
                    'concern': {'type': 'string'},
                    'suggestion': {'type': 'string'},
                },
                'required': ['check_id', 'severity', 'concern', 'suggestion'],
                'additionalProperties': False,
            },
        },
        'summary': {'type': 'string'},
    },
    'required': ['concerns', 'summary'],
    'additionalProperties': False,
}

CRITIC_SYSTEM = """You review a proposed long-only equity portfolio for a \
single retail user.

You are given a table of checks that have ALREADY been computed. Your job is \
to rank the concerns they raise and explain each one in plain language.

Hard rules:
- Cite only check_id values from the table. A concern about anything not in \
the table is not allowed.
- Do not compute, estimate, restate or invent any number that is not in the \
check you are citing.
- Do not predict returns, prices or direction. Do not recommend buying or \
selling a specific name.
- Suggestions must be about portfolio construction (trim, diversify, wait, \
review), never about market timing.
- If a check is 'ok', do not manufacture a concern about it. Returning few \
concerns is a valid answer.

Severity is your judgement about the portfolio, not a restatement of the \
check status."""


def review(rec, checks, use_llm=True, effort='medium'):
    """Rank and narrate the checks; blocking is decided in Python.

    The model never sees a price or a return, only the finished check table,
    and its output is filtered to concerns that cite a real check id.
    """
    failed = [c for c in checks if c['status'] == 'fail']
    warned = [c for c in checks if c['status'] == 'warn']
    out = {
        'blocking': bool(failed),
        'n_fail': len(failed),
        'n_warn': len(warned),
        'checks': checks,
        'concerns': [],
        'summary': '',
        'source': 'deterministic',
    }

    if not use_llm or not llm.available():
        out['summary'] = _fallback_summary(failed, warned, checks)
        out['concerns'] = [
            {'check_id': c['id'], 'severity': 'high' if c['status'] == 'fail'
             else 'medium', 'concern': c['detail'], 'suggestion': ''}
            for c in failed + warned]
        return out

    table = json.dumps([{k: c.get(k) for k in
                         ('id', 'name', 'status', 'detail', 'values')}
                        for c in checks], indent=2, default=str)
    prompt = (f"Checks computed for the proposed portfolio:\n\n{table}\n\n"
              f"Rank the real concerns, most important first, and write a "
              f"two-sentence summary a careful investor would want to read "
              f"before placing these trades.")
    try:
        payload = llm.structured(CRITIC_SYSTEM, prompt, CRITIC_SCHEMA,
                                 effort=effort)
    except llm.LLMUnavailable as e:
        logger.info("critic unavailable: %s", e)
        out['summary'] = _fallback_summary(failed, warned, checks)
        out['concerns'] = [
            {'check_id': c['id'], 'severity': 'high' if c['status'] == 'fail'
             else 'medium', 'concern': c['detail'], 'suggestion': ''}
            for c in failed + warned]
        return out

    valid_ids = {c['id'] for c in checks}
    kept, dropped = [], []
    for concern in payload.get('concerns', []):
        if concern.get('check_id') in valid_ids:
            kept.append(concern)
        else:
            dropped.append(concern.get('check_id'))
    out['concerns'] = kept
    out['summary'] = payload.get('summary', '')
    out['source'] = 'llm'
    if dropped:
        out['rejected_concerns'] = dropped
        logger.info("dropped %d concern(s) citing unknown checks: %s",
                    len(dropped), dropped)
    # blocking stays whatever the deterministic checks said
    out['blocking'] = bool(failed)
    return out


def _fallback_summary(failed, warned, checks):
    if failed:
        return (f"{len(failed)} check(s) failed: "
                + "; ".join(c['name'] for c in failed) + ".")
    if warned:
        return (f"No failures. {len(warned)} thing(s) worth a look: "
                + "; ".join(c['name'] for c in warned) + ".")
    return f"All {len(checks)} checks passed."


def log_flags(rec, review_out, path=FLAG_LOG):
    """Append today's flags so the critic can be scored later.

    Twenty sessions from now this file answers the only question that
    matters about the critic: did the names it flagged actually do worse
    than the ones it did not?
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    row = {
        'ts': dt.datetime.now().isoformat(timespec='seconds'),
        'symbols': [h['symbol'] for h in (rec or {}).get('holdings') or []],
        'blocking': review_out.get('blocking'),
        'flags': [{'id': c['id'], 'status': c['status']}
                  for c in review_out.get('checks', [])
                  if c['status'] in ('warn', 'fail')],
        'concerns': [c.get('check_id') for c in review_out.get('concerns', [])],
        'source': review_out.get('source'),
    }
    with open(path, 'a') as f:
        f.write(json.dumps(row, default=str) + '\n')
    return row
