"""
Backtest evaluation statistics: costs, multiple-testing corrections and
overfitting diagnostics.

The walk-forward harness in backtest.py answers "what did this configuration
return". That is not the same question as "is this configuration real". The
methodology literature is blunt about the gap:

- Bailey, Borwein, Lopez de Prado and Zhu (AMS 2014) show that with five
  years of data, trying more than about 45 configurations almost guarantees
  an in-sample Sharpe of 1 whose out-of-sample expectation is zero. Every
  tilt, weight table and sleeve definition tried counts as a trial.
- Harvey, Liu and Zhu (RFS 2016) put the hurdle for a new factor at t > 3
  rather than t > 2, for the same reason.
- Bailey and Lopez de Prado (2014) deflate an observed Sharpe by the number
  of trials, the sample length and the return distribution's skew and
  kurtosis.
- Bailey et al. (2015) measure the probability that the selection procedure
  itself is uninformative (PBO, via combinatorially symmetric cross
  validation).
- Nikolopoulos (2026) argues the only honest check on a search procedure is
  to run the whole thing on data with no signal in it and see what it finds.

Everything here is pure numpy so the project keeps its four dependencies.
Normal cdf/ppf are implemented locally rather than pulling in scipy.

References are listed in docs/research-llm-agents.md.
"""

import hashlib
import json
import math
import os
from itertools import combinations

import numpy as np

EULER_MASCHERONI = 0.5772156649015329
TRADING_DAYS = 252
TRIAL_LOG = os.path.join('results', 'trials.jsonl')


# -- normal distribution helpers ---------------------------------------

def norm_cdf(x):
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def norm_ppf(p):
    """Standard normal inverse CDF (Acklam rational approximation with one
    Halley refinement, good to roughly 1e-12)."""
    p = float(p)
    if not 0.0 < p < 1.0:
        raise ValueError("norm_ppf requires 0 < p < 1")

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    e = norm_cdf(x) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


# -- transaction costs -------------------------------------------------

def apply_costs(rets, turnovers, cost_bps):
    """Charge cost_bps of one-way turnover against each period's return.

    Turnover is the fraction of the book replaced at a rebalance, so a
    round trip of that fraction costs 2 * cost_bps. The harness measures
    turnover as names swapped over names held, a fair proxy for sum |dw|
    in a roughly equal-weight basket.

    The README already flags that "even a few bps of cost per trade would
    visibly dent the composite's edge". This turns that into a number.
    """
    rets = np.asarray(rets, dtype=float)
    if not cost_bps:
        return rets
    t = np.asarray(turnovers, dtype=float)
    if t.shape != rets.shape:
        padded = np.zeros_like(rets)
        if len(t):
            take = t[-len(rets):] if len(t) > len(rets) else t
            padded[len(rets) - len(take):] = take
        t = padded
    return rets - 2.0 * t * (float(cost_bps) / 10_000.0)


# -- Sharpe deflation --------------------------------------------------

def sharpe_stats(rets):
    """Per-period Sharpe plus the moments the deflation needs."""
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3:
        return None
    sd = float(r.std(ddof=1))
    if sd <= 0:
        return None
    mean = float(r.mean())
    z = (r - mean) / sd
    return {
        'sharpe': mean / sd,
        'skew': float((z ** 3).mean()),
        'kurtosis': float((z ** 4).mean()),  # non-excess
        'n': n,
    }


def expected_max_sharpe(n_trials, sr_std=1.0):
    """Expected maximum per-period Sharpe across n_trials configurations
    whose true Sharpe is zero (Bailey and Lopez de Prado 2014).

    sr_std is the cross-sectional spread of the trials' Sharpe ratios; pass
    the observed spread when a grid was run, otherwise the answer comes back
    in units of that spread.
    """
    n = max(2, int(n_trials))
    return sr_std * ((1.0 - EULER_MASCHERONI) * norm_ppf(1.0 - 1.0 / n)
                     + EULER_MASCHERONI * norm_ppf(1.0 - 1.0 / (n * math.e)))


def expected_max_t(n_trials):
    """Hurdle for a t-statistic that is the best of n_trials draws from a
    standard normal: sqrt(2 ln N). N = 45 gives 2.8 and N = 200 gives 3.3,
    which is where the Harvey-Liu-Zhu t > 3 convention comes from."""
    n = max(2, int(n_trials))
    return math.sqrt(2.0 * math.log(n))


def deflated_sharpe(rets, n_trials, sr_std=None, benchmark_sr=None):
    """Probability the observed Sharpe beats what the best of n_trials
    lucky draws would have produced.

    A deflated Sharpe below 0.95 means the result is not distinguishable
    from noise once the search is accounted for.
    """
    stats = sharpe_stats(rets)
    if stats is None:
        return None
    sr, n = stats['sharpe'], stats['n']
    skew, kurt = stats['skew'], stats['kurtosis']

    if benchmark_sr is not None:
        sr0 = float(benchmark_sr)
    else:
        spread = sr_std if sr_std is not None else abs(sr)
        sr0 = expected_max_sharpe(n_trials, spread if spread > 0 else 1e-9)

    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if denom <= 0 or n < 2:
        return None
    dsr = norm_cdf((sr - sr0) * math.sqrt(n - 1) / math.sqrt(denom))
    return {
        'sharpe': round(sr, 4),
        'sharpe_annual': round(sr * math.sqrt(TRADING_DAYS), 3),
        'sr0': round(sr0, 4),
        'deflated_sharpe': round(float(dsr), 4),
        'n_trials': int(n_trials),
        'n_obs': n,
        'skew': round(skew, 3),
        'kurtosis': round(kurt, 3),
        'passes': bool(dsr >= 0.95),
    }


def deflated_t(t_stat, n_trials):
    """Compare an IC t-statistic against the best-of-N null hurdle.

    `passes` is the two-sided test and answers "is this distinguishable
    from noise". `clears_positive` is the one that matters for a long-only
    score: an IC that is significantly NEGATIVE is a strong result and a
    bad one, so the two are reported separately.
    """
    hurdle = expected_max_t(n_trials)
    t = float(t_stat)
    return {
        't_stat': round(t, 3),
        'hurdle': round(hurdle, 3),
        'n_trials': int(n_trials),
        'passes': bool(abs(t) > hurdle),
        'clears_positive': bool(t > hurdle),
    }


# -- probability of backtest overfitting -------------------------------

def probability_of_backtest_overfitting(matrix, n_splits=10):
    """PBO via combinatorially symmetric cross validation.

    matrix: (T periods, N configurations) of per-period returns.

    Split the rows into n_splits blocks and, for every way of choosing half
    the blocks as in-sample, take the configuration with the best in-sample
    Sharpe and look up where it ranks out of sample. If the in-sample winner
    is routinely a below-median performer out of sample, the selection
    procedure carries no information and PBO approaches 1.
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2:
        return None
    t, n_cfg = m.shape
    s = int(n_splits)
    if s % 2:
        s -= 1
    s = max(4, min(s, t // 2))
    if t < s * 2:
        return None

    blocks = np.array_split(np.arange(t), s)
    logits = []
    for combo in combinations(range(s), s // 2):
        rest = [i for i in range(s) if i not in combo]
        is_sr = _sharpe_cols(m[np.concatenate([blocks[i] for i in combo])])
        oos_sr = _sharpe_cols(m[np.concatenate([blocks[i] for i in rest])])
        if is_sr is None or oos_sr is None:
            continue
        best = int(np.nanargmax(is_sr))
        rank = np.argsort(np.argsort(oos_sr))  # 0 = worst
        omega = (rank[best] + 1.0) / (n_cfg + 1.0)
        omega = min(max(omega, 1e-6), 1.0 - 1e-6)
        logits.append(math.log(omega / (1.0 - omega)))

    if not logits:
        return None
    logits = np.asarray(logits)
    return {
        'pbo': round(float((logits <= 0).mean()), 4),
        'n_combinations': len(logits),
        'n_configs': n_cfg,
        'n_splits': s,
        'median_logit': round(float(np.median(logits)), 3),
        'passes': bool((logits <= 0).mean() < 0.5),
    }


def _sharpe_cols(block):
    """Per-column Sharpe of a (rows, configs) block."""
    if len(block) < 2:
        return None
    sd = block.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        sr = block.mean(axis=0) / sd
    if np.all(np.isnan(sr)):
        return None
    return np.nan_to_num(sr, nan=-1e9)


# -- regime split ------------------------------------------------------

def regime_labels(dates, spy_close, window=200):
    """Label each date bull or bear by SPY against its 200-day average.

    FinSABER's central finding is that timing strategies look fine in a
    rising tape and fall apart in a falling one; averaging the two hides it.
    The same caution applies here, since the headline sample is almost
    entirely a bull market.
    """
    sma = spy_close.rolling(window, min_periods=max(20, window // 2)).mean()
    out = []
    for d in dates:
        try:
            pos = spy_close.index.get_indexer([d], method='pad')[0]
        except Exception:
            pos = -1
        if pos < 0 or not np.isfinite(sma.iloc[pos]):
            out.append('unknown')
        else:
            out.append('bull' if float(spy_close.iloc[pos]) >= float(sma.iloc[pos])
                       else 'bear')
    return out


def split_by_regime(rets, labels, step, stats_fn):
    """Run stats_fn over the returns belonging to each regime."""
    rets = np.asarray(rets, dtype=float)
    labels = np.asarray(labels[:len(rets)])
    out = {}
    for regime in ('bull', 'bear'):
        mask = labels == regime
        if mask.sum() >= 8:
            row = stats_fn(rets[mask], step)
            row['n_periods'] = int(mask.sum())
            out[regime] = row
    return out


# -- null audit --------------------------------------------------------

def null_ic_distribution(per_date, strategies, n_permutations=200, seed=0):
    """Distribution of the best mean IC achievable on data with no signal.

    Forward returns are shuffled within each rebalance date, which destroys
    the cross-sectional relationship while preserving each date's return
    distribution. The 95th percentile of the best-sleeve IC across
    permutations is the bar a real sleeve has to clear, and it prices in the
    fact that the best of several sleeves was picked after looking.
    """
    rng = np.random.default_rng(seed)
    strategies = list(strategies)
    snaps = []
    for snap in per_date:
        usable = [s for s in snap['scores'] if s in snap['fwd']]
        if len(usable) < 8:
            continue
        fwd = np.array([snap['fwd'][s] for s in usable], dtype=float)
        xs = {}
        for strat in strategies:
            xs[strat] = _rank(np.array(
                [snap['scores'][s][strat] for s in usable], dtype=float))
        snaps.append((fwd, xs))
    if not snaps:
        return None

    best_ics = []
    for _ in range(int(n_permutations)):
        sums = {s: [] for s in strategies}
        for fwd, xs in snaps:
            y = _rank(rng.permutation(fwd))
            if y.std() == 0:
                continue
            for strat in strategies:
                x = xs[strat]
                if x.std() == 0:
                    continue
                sums[strat].append(float(np.corrcoef(x, y)[0, 1]))
        means = [float(np.mean(v)) for v in sums.values() if v]
        if means:
            best_ics.append(max(means))

    if not best_ics:
        return None
    arr = np.asarray(best_ics)
    return {
        'n_permutations': len(arr),
        'mean': round(float(arr.mean()), 5),
        'p95': round(float(np.percentile(arr, 95)), 5),
        'p99': round(float(np.percentile(arr, 99)), 5),
        'max': round(float(arr.max()), 5),
    }


def _rank(values):
    return np.argsort(np.argsort(values)).astype(float)


# -- trial ledger ------------------------------------------------------

def log_trial(config, metrics, path=TRIAL_LOG):
    """Append one evaluated configuration to the trial ledger.

    The deflation above is only honest if N counts every configuration ever
    scored, not just the ones that were kept. Nothing here can verify the
    ledger is complete - the discipline is the point.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    payload = json.dumps(config, sort_keys=True, default=str)
    row = {
        'hash': hashlib.sha256(payload.encode()).hexdigest()[:12],
        'config': config,
        'metrics': metrics,
    }
    with open(path, 'a') as f:
        f.write(json.dumps(row, default=str) + '\n')
    return row['hash']


def count_trials(path=TRIAL_LOG, distinct=True):
    """How many configurations have been scored so far (minimum 1)."""
    if not os.path.exists(path):
        return 1
    seen, total = set(), 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                seen.add(json.loads(line).get('hash'))
            except json.JSONDecodeError:
                continue
    return max(1, len(seen) if distinct else total)
