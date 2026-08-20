"""
A grounded morning brief: plain-language narration of numbers this system
already computed, with every sentence checked before it is shown.

Why it is built this way rather than "send the scan to a model and ask what
it thinks":

- Geng et al. (arXiv 2602.18895) tested LLMs explaining a credit model in
  two roles. Given the model's feature ranking they reproduced it reliably;
  asked to work it out themselves they showed "limited alignment". So the
  model here is handed finished numbers and never asked to infer any.
- Zandi et al. (arXiv 2608.17715) ran three narration pipelines over three
  models and found pipeline design mattered more than model choice, and that
  models were inconsistent about the DIRECTION of a factor's influence. That
  specific failure - saying momentum helped when it hurt - is what
  _check_polarity exists to catch.
- FinGround (ACL 2026) cut unsupported claims by 68% by decomposing output
  into atomic claims and verifying each against the source. verify() is that
  idea without the retrieval: every sentence must cite fact ids, and every
  numeral in it must be a number that is actually in the facts.
- Fons et al. (arXiv 2507.00718) tag each sentence of a generated report as
  data-derived, reasoning, or outside knowledge. That tagging is in the
  schema, and 'external' sentences are dropped rather than shown.

What this is not: evidence of anything. No paper shows a daily narrative
improves a retail investor's decisions. The value is legibility, and since
the README's own analysis concludes the edge is in portfolio construction
rather than stock picking, the brief is told to talk about the portfolio and
the regime rather than to pitch names.
"""

import datetime as dt
import hashlib
import json
import logging
import re

import llm

logger = logging.getLogger('stock_app.brief')

MAX_SENTENCES = 12
NUMBER_TOLERANCE = 0.02   # relative slack when matching a quoted number

POSITIVE_WORDS = {'help', 'helps', 'helped', 'helping', 'support', 'supports',
                  'supported', 'supportive', 'positive', 'strong', 'strength',
                  'favourable', 'favorable', 'tailwind', 'boost', 'boosts',
                  'lifted', 'lifts', 'constructive', 'improving', 'improved'}
NEGATIVE_WORDS = {'hurt', 'hurts', 'hurting', 'drag', 'drags', 'dragged',
                  'weigh', 'weighs', 'weighed', 'negative', 'weak', 'weakness',
                  'headwind', 'detract', 'detracts', 'detracted', 'poor',
                  'deteriorating', 'worsening', 'pressured'}

BRIEF_SCHEMA = {
    'type': 'object',
    'properties': {
        'sentences': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'text': {'type': 'string'},
                    'cited_ids': {'type': 'array', 'items': {'type': 'string'}},
                    'claim_type': {'type': 'string',
                                   'enum': ['data', 'reasoning', 'external']},
                },
                'required': ['text', 'cited_ids', 'claim_type'],
                'additionalProperties': False,
            },
        },
    },
    'required': ['sentences'],
    'additionalProperties': False,
}

SYSTEM = """You write a short morning brief for one person about their own \
stock screening dashboard. You are a narrator of numbers that have already \
been computed. You are not an analyst and you have no information beyond \
what you are given.

Hard rules, all enforced after you answer:
- Every sentence must cite the ids of the facts it uses. A sentence citing \
nothing is dropped.
- Every number you write must appear in the facts you cite. Do not compute \
new numbers, do not average, do not convert, do not round differently.
- claim_type is 'data' if the sentence only restates facts, 'reasoning' if \
it connects facts the dashboard already relates, and 'external' if it uses \
anything else. External sentences are dropped, so do not write any.
- Never predict a price, a return or a direction. Never say a name will or \
should go up or down. Never give advice to buy or sell.
- When you describe a factor's contribution, its direction must match the \
sign of the number. A negative value is a drag, never support.

Style: plain, specific, unhurried. Lead with the portfolio and the market \
regime, not with individual names. Mention what CHANGED since the last scan \
before anything that stayed the same. If the honest summary is "nothing much \
moved", write that in one sentence and stop. At most 8 sentences."""


# -- facts -------------------------------------------------------------

def build_facts(results=None, rec=None, market=None, backtest=None,
                movers=None):
    """Everything the model is allowed to know, each item with an id.

    Deliberately excludes prices, price history and anything the model could
    use to recognise a date. It gets scores, signs, ranks, weights and
    counts, which is what a narration needs.
    """
    facts = {}

    def add(fid, label, value, extra=None):
        row = {'id': fid, 'label': label, 'value': value}
        if extra:
            row.update(extra)
        facts[fid] = row

    if market:
        add('market.risk', 'market risk state', market.get('risk'))
        add('market.note', 'market regime note', market.get('note'))
        if market.get('spy_dist_200dma') is not None:
            add('market.spy_200dma', 'SPY distance from its 200-day average '
                'in percent', market.get('spy_dist_200dma'))
        if market.get('vix_percentile') is not None:
            add('market.vix_pctile', 'VIX percentile against five years',
                market.get('vix_percentile'))
        if market.get('exposure') is not None:
            add('market.exposure', 'target gross exposure, 1.0 being fully '
                'invested', market.get('exposure'))

    if rec:
        holdings = rec.get('holdings') or []
        add('portfolio.n_names', 'number of recommended names', len(holdings))
        add('portfolio.horizon', 'selection horizon',
            rec.get('selection_horizon'))
        if rec.get('cash_weight') is not None:
            add('portfolio.cash_pct', 'percent of the book held in cash',
                rec.get('cash_weight'))
        changes = rec.get('changes') or {}
        add('portfolio.added', 'names added since the last recommendation',
            changes.get('added') or [])
        add('portfolio.dropped', 'names dropped since the last recommendation',
            changes.get('dropped') or [])
        risk = rec.get('risk') or {}
        for key, label in (('beta', 'portfolio beta vs SPY'),
                           ('ann_vol', 'portfolio annualised volatility '
                                       'in percent')):
            if risk.get(key) is not None:
                add(f'portfolio.{key}', label, risk[key])
        for h in holdings[:8]:
            sym = h['symbol']
            add(f'holding.{sym}.weight', f'{sym} target weight in percent',
                h.get('weight'))
            add(f'holding.{sym}.score', f'{sym} composite score at the '
                f'selection horizon', h.get('score'))
        plan = rec.get('rebalance') or {}
        if plan:
            add('rebalance.n_trades', 'number of trades in the plan',
                len(plan.get('trades') or []))
            add('rebalance.net_cash', 'net cash the plan needs in dollars',
                plan.get('net_cash_needed'))

    if movers:
        add('movers.up', 'symbols whose rank improved most since the last '
            'scan', [m.get('symbol') for m in (movers.get('up') or [])][:5])
        add('movers.down', 'symbols whose rank fell most since the last scan',
            [m.get('symbol') for m in (movers.get('down') or [])][:5])

    if results:
        syms = results.get('symbols') or {}
        sleeve_totals, n = {}, 0
        for sym, periods in syms.items():
            row = (periods or {}).get('1m') or {}
            factors = row.get('factors') or {}
            if not factors:
                continue
            n += 1
            for k, v in factors.items():
                if isinstance(v, (int, float)):
                    sleeve_totals[k] = sleeve_totals.get(k, 0.0) + float(v)
        if n:
            for k, total in sleeve_totals.items():
                add(f'sleeve.{k}', f'average {k} sleeve score across the '
                    f'watchlist, from -1 to +1', round(total / n, 3),
                    {'sleeve': k})
            add('universe.n_scored', 'number of symbols scored', n)

    if backtest:
        strategies = (backtest.get('backtest') or {}).get('strategies') or {}
        comp = strategies.get('composite') or {}
        for key, label in (('cagr', 'backtested composite CAGR in percent'),
                           ('sharpe', 'backtested composite Sharpe'),
                           ('max_drawdown', 'backtested composite worst '
                                            'drawdown in percent')):
            if comp.get(key) is not None:
                add(f'backtest.{key}', label, comp[key])
        diag = (backtest.get('backtest') or {}).get('diagnostics') or {}
        pbo = (diag.get('pbo') or {}).get('pbo')
        if pbo is not None:
            add('backtest.pbo', 'probability of backtest overfitting, where '
                'above 0.5 means the comparison is uninformative', pbo)

    return facts


# -- verification ------------------------------------------------------

def _collect_numbers(value, into):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        into.add(round(float(value), 6))
    elif isinstance(value, dict):
        for v in value.values():
            _collect_numbers(v, into)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _collect_numbers(v, into)


def allowed_numbers(facts, cited_ids=None):
    """Numbers a sentence is permitted to contain."""
    out = set()
    items = facts.values() if cited_ids is None else \
        [facts[i] for i in cited_ids if i in facts]
    for row in items:
        _collect_numbers(row.get('value'), out)
    # counts of listed things are verifiable, so allow them
    for row in items:
        v = row.get('value')
        if isinstance(v, (list, tuple)):
            out.add(float(len(v)))
    return out


# A number only counts as a claim when it stands alone. Digits glued to
# letters are part of a term - "200dma", "12-1 momentum", "S&P500" - and
# rejecting a sentence for naming an indicator would make the verifier
# useless while catching nothing.
NUMBER_RE = re.compile(
    r'(?<![A-Za-z0-9.])-?\d+(?:,\d{3})*(?:\.\d+)?(?![A-Za-z0-9])')


def _numbers_in(text):
    out = []
    for match in NUMBER_RE.findall(text):
        try:
            out.append(float(match.replace(',', '')))
        except ValueError:
            continue
    return out


def _number_ok(value, allowed):
    for a in allowed:
        if a == value:
            return True
        scale = max(abs(a), abs(value), 1e-9)
        if abs(a - value) / scale <= NUMBER_TOLERANCE:
            return True
    return False


def _check_polarity(text, facts, cited_ids):
    """Direction words must agree with the sign of the cited sleeve.

    This is the specific error Zandi et al. found models make most often
    when narrating a numeric model, so it fails closed.
    """
    words = set(re.findall(r"[a-zA-Z']+", text.lower()))
    said_positive = bool(words & POSITIVE_WORDS)
    said_negative = bool(words & NEGATIVE_WORDS)
    if not (said_positive or said_negative):
        return True, None
    for fid in cited_ids:
        row = facts.get(fid)
        if not row or not isinstance(row.get('value'), (int, float)):
            continue
        if not fid.startswith('sleeve.'):
            continue
        v = float(row['value'])
        if abs(v) < 1e-6:
            continue
        if v > 0 and said_negative and not said_positive:
            return False, f"{fid} is {v:+.3f} but the sentence reads negative"
        if v < 0 and said_positive and not said_negative:
            return False, f"{fid} is {v:+.3f} but the sentence reads positive"
    return True, None


def verify(sentences, facts, max_sentences=MAX_SENTENCES):
    """Keep only sentences that survive every check.

    Returns (kept, rejected). Rejections carry a reason so the confabulation
    rate can be tracked rather than guessed at.
    """
    kept, rejected = [], []
    for s in sentences[:max_sentences * 2]:
        text = (s.get('text') or '').strip()
        cited = [c for c in (s.get('cited_ids') or []) if isinstance(c, str)]
        claim = s.get('claim_type')

        if not text:
            continue
        if claim == 'external':
            rejected.append({'text': text, 'reason': 'claims outside knowledge'})
            continue
        unknown = [c for c in cited if c not in facts]
        if unknown:
            rejected.append({'text': text,
                             'reason': f"cites unknown fact(s) {unknown}"})
            continue
        if not cited:
            rejected.append({'text': text, 'reason': 'cites no facts'})
            continue

        allowed = allowed_numbers(facts, cited)
        bad = [n for n in _numbers_in(text) if not _number_ok(n, allowed)]
        if bad:
            rejected.append({'text': text,
                             'reason': f"number(s) {bad} are not in the "
                                       f"cited facts"})
            continue

        ok, why = _check_polarity(text, facts, cited)
        if not ok:
            rejected.append({'text': text, 'reason': f"direction mismatch: {why}"})
            continue

        kept.append({'text': text, 'cited_ids': cited, 'claim_type': claim})
        if len(kept) >= max_sentences:
            break
    return kept, rejected


# -- generation --------------------------------------------------------

def _facts_prompt(facts):
    lines = []
    for fid, row in facts.items():
        lines.append(f"{fid} | {row['label']} | {json.dumps(row['value'], default=str)}")
    return "\n".join(lines)


def fallback_brief(facts):
    """A deterministic brief, used when no model is reachable.

    Deliberately plain. It is also the control the LLM version should be
    compared against: if the generated brief is not clearly more useful than
    this, the API call is not worth making.
    """
    out = []
    risk = facts.get('market.risk', {}).get('value')
    note = facts.get('market.note', {}).get('value')
    if risk:
        line = f"Market risk state is {risk}"
        cited = ['market.risk']
        if note:
            line += f" ({note})"
            cited.append('market.note')
        out.append({'text': line + ".", 'cited_ids': cited,
                    'claim_type': 'data'})
    n = facts.get('portfolio.n_names', {}).get('value')
    cash = facts.get('portfolio.cash_pct', {}).get('value')
    if n is not None:
        line = f"The recommendation holds {n} names"
        if cash:
            line += f" with {cash}% in cash"
        out.append({'text': line + ".",
                    'cited_ids': ['portfolio.n_names'] +
                                 (['portfolio.cash_pct'] if cash else []),
                    'claim_type': 'data'})
    added = facts.get('portfolio.added', {}).get('value') or []
    dropped = facts.get('portfolio.dropped', {}).get('value') or []
    if added or dropped:
        out.append({'text': f"Added {', '.join(added) or 'nothing'}; "
                            f"dropped {', '.join(dropped) or 'nothing'}.",
                    'cited_ids': ['portfolio.added', 'portfolio.dropped'],
                    'claim_type': 'data'})
    else:
        out.append({'text': "No changes to the recommended names since the "
                            "last scan.",
                    'cited_ids': ['portfolio.added', 'portfolio.dropped'],
                    'claim_type': 'data'})
    sleeves = {k: v for k, v in facts.items() if k.startswith('sleeve.')}
    if sleeves:
        best = max(sleeves.values(), key=lambda r: r['value'])
        worst = min(sleeves.values(), key=lambda r: r['value'])
        out.append({
            'text': f"Across the watchlist the {best['sleeve']} sleeve reads "
                    f"{best['value']:+.3f} and {worst['sleeve']} reads "
                    f"{worst['value']:+.3f}.",
            'cited_ids': [best['id'], worst['id']], 'claim_type': 'data'})
    return out


def generate(results=None, rec=None, market=None, backtest=None, movers=None,
             use_llm=True, effort='medium'):
    """Build the facts, narrate them, verify, and report what was dropped."""
    facts = build_facts(results, rec, market, backtest, movers)
    payload = {
        'as_of': dt.datetime.now().isoformat(timespec='seconds'),
        'facts_hash': hashlib.sha256(
            json.dumps({k: v['value'] for k, v in facts.items()},
                       sort_keys=True, default=str).encode()).hexdigest()[:12],
        'n_facts': len(facts),
        'source': 'deterministic',
    }

    sentences = None
    if use_llm and llm.available():
        prompt = (f"Facts, one per line as `id | label | value`:\n\n"
                  f"{_facts_prompt(facts)}\n\n"
                  f"Write the brief.")
        try:
            raw = llm.structured(SYSTEM, prompt, BRIEF_SCHEMA, effort=effort)
            sentences = raw.get('sentences') or []
            payload['source'] = 'llm'
        except llm.LLMUnavailable as e:
            logger.info("brief falling back to template: %s", e)

    if sentences is None:
        sentences = fallback_brief(facts)

    kept, rejected = verify(sentences, facts)
    if not kept:
        kept, _ = verify(fallback_brief(facts), facts)
        payload['source'] += '+fallback'

    payload['sentences'] = kept
    payload['text'] = " ".join(s['text'] for s in kept)
    payload['rejected'] = rejected
    payload['confabulation_rate'] = (
        round(len(rejected) / (len(kept) + len(rejected)), 3)
        if (kept or rejected) else 0.0)
    return payload
