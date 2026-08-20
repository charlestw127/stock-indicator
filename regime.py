"""
Statistical jump model for per-symbol regime detection.

quant_engine.classify_trend_regime pools four statistics into a scalar and
thresholds it at two hand-picked cut points; REGIME_WEIGHTS then maps the
resulting label onto fifteen hand-picked sleeve weights. That is a
reasonable prior and it is completely unvalidated - the README says so.

The jump model is the smallest principled replacement with real evidence
behind it. Shu, Yu and Mulvey (Journal of Derivatives and Investment
Management 2024, arXiv 2402.05272) fit k-means to return/risk features with
a penalty on every state switch, and pick that penalty by cross-validation
on strategy performance rather than on how tidy the labels look. On US, DE
and JP indices from 1990 to 2023, with costs and a one-day implementation
delay, it beat both a hidden Markov model and buy-and-hold on volatility and
drawdown. Their follow-up (Annals of Operations Research 2024, arXiv
2406.09578) adds a classifier that forecasts the next state, which is the
direct analogue of what happens here.

Why a jump model rather than an HMM: the penalty formulation is a hard
constraint on how often the state may change, so it does not produce the
three-day "regimes" a Gaussian HMM happily fits to noise. state_summary()
reports mean run length precisely so that failure stays visible.

Everything here is causal by construction. label_walk_forward() refits on an
expanding window and only ever keeps the label of the last bar in that
window, so the label at bar i never depends on a bar after i. There is a
test for this (test_regime.test_labels_are_causal).
"""

import numpy as np
import pandas as pd

import quant_indicators as qi

# Order matters: states are named by how persistent their centroid looks, so
# the vocabulary stays compatible with REGIME_WEIGHTS and the dashboard.
STATE_NAMES = {
    2: ['trending', 'mean-reverting'],
    3: ['trending', 'mixed', 'mean-reverting'],
}

FEATURES = ['ret_20', 'ret_60', 'vol_21', 'downside_21',
            'hurst', 'variance_ratio', 'efficiency_ratio']

# Features whose high end means "this trend is persistent". Used only to
# name the states, never to fit them.
PERSISTENCE_FEATURES = ['hurst', 'variance_ratio', 'efficiency_ratio']


# -- feature construction ----------------------------------------------

def build_features(close, high=None, low=None, hurst_window=252,
                   vr_window=126):
    """Causal regime features for one symbol, as a (T, d) DataFrame.

    Deliberately reuses the statistics the existing classifier already
    computes, so the jump model is a different way of reading the same
    evidence rather than a different set of inputs.
    """
    close = pd.Series(close).astype(float)
    rets = close.pct_change()

    out = pd.DataFrame(index=close.index)
    out['ret_20'] = close.pct_change(20)
    out['ret_60'] = close.pct_change(60)
    out['vol_21'] = rets.rolling(21, min_periods=10).std() * np.sqrt(252)
    downside = rets.where(rets < 0, 0.0)
    out['downside_21'] = downside.rolling(21, min_periods=10).std() * np.sqrt(252)
    out['hurst'] = _rolling_stat(close, hurst_window,
                                 lambda s: qi.hurst_exponent(s, window=len(s)))
    out['variance_ratio'] = _rolling_stat(
        close, vr_window, lambda s: qi.variance_ratio(s, q=5, window=len(s)))
    out['efficiency_ratio'] = qi.efficiency_ratio(close, 20)
    return out[FEATURES]


def _rolling_stat(series, window, fn, stride=5):
    """Apply a whole-window statistic on a stride and forward-fill.

    Hurst and the variance ratio are expensive and change slowly, so
    recomputing them every fifth bar and holding the value in between costs
    nothing in signal and a lot less in time. Forward-filling only ever
    carries a past value forward, so it stays causal.
    """
    values = pd.Series(np.nan, index=series.index, dtype=float)
    for i in range(window, len(series), stride):
        try:
            v = fn(series.iloc[max(0, i - window + 1):i + 1])
        except Exception:
            v = None
        if v is not None and np.isfinite(v):
            values.iloc[i] = float(v)
    return values.ffill()


def standardize(X, upto=None):
    """Z-score features using only rows up to `upto` (default: all)."""
    X = np.asarray(X, dtype=float)
    ref = X if upto is None else X[:upto + 1]
    mu = np.nanmean(ref, axis=0)
    sd = np.nanstd(ref, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
    Z = (X - mu) / sd
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


# -- the model ---------------------------------------------------------

class JumpModel:
    """k-means with a penalty on every state change.

    Fitting minimises

        sum_t ||x_t - mu_{s_t}||^2  +  jump_penalty * #{t : s_t != s_{t-1}}

    by coordinate descent: the state path is solved exactly by dynamic
    programming given the centroids, then the centroids are recomputed as
    within-state means. Both steps decrease the objective, so it converges.
    """

    def __init__(self, n_states=3, jump_penalty=20.0, max_iter=30,
                 n_init=5, seed=0):
        self.n_states = int(n_states)
        self.jump_penalty = float(jump_penalty)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.seed = int(seed)
        self.centroids_ = None
        self.labels_ = None
        self.objective_ = None
        self.names_ = None

    def fit(self, Z):
        Z = np.asarray(Z, dtype=float)
        if len(Z) < self.n_states * 5:
            raise ValueError("not enough rows to fit a jump model")
        rng = np.random.default_rng(self.seed)

        best = None
        for _ in range(self.n_init):
            centroids = _kmeans_plus_plus(Z, self.n_states, rng)
            labels = None
            for _ in range(self.max_iter):
                cost = _sq_dist(Z, centroids)
                new_labels, objective = _viterbi(cost, self.jump_penalty)
                if labels is not None and np.array_equal(new_labels, labels):
                    break
                labels = new_labels
                for k in range(self.n_states):
                    mask = labels == k
                    if mask.any():
                        centroids[k] = Z[mask].mean(axis=0)
            if best is None or objective < best[2]:
                best = (labels, centroids, objective)

        self.labels_, self.centroids_, self.objective_ = best
        self._name_states()
        return self

    def _name_states(self):
        """Attach the trending / mixed / mean-reverting vocabulary.

        States are ordered by how persistent their centroid looks on the
        Hurst / variance-ratio / efficiency-ratio block. This uses only the
        fitted centroids, so it introduces no new information - it just
        stops the state index being arbitrary between refits.
        """
        idx = [FEATURES.index(f) for f in PERSISTENCE_FEATURES]
        score = self.centroids_[:, idx].mean(axis=1)
        order = np.argsort(-score)  # most persistent first
        names = STATE_NAMES.get(self.n_states)
        if names is None:
            names = [f'state_{i}' for i in range(self.n_states)]
        self.names_ = {int(state): names[rank]
                       for rank, state in enumerate(order)}

    def named_labels(self):
        return np.array([self.names_[int(k)] for k in self.labels_])

    def label_last(self, Z):
        """State of the final row, given the fitted centroids.

        Used by the online path: fit on everything up to bar i, then read
        off bar i. The full DP is rerun so the last label is consistent
        with the penalty rather than a nearest-centroid shortcut.
        """
        cost = _sq_dist(np.asarray(Z, dtype=float), self.centroids_)
        labels, _ = _viterbi(cost, self.jump_penalty)
        return self.names_[int(labels[-1])]


def _kmeans_plus_plus(Z, k, rng):
    n = len(Z)
    centroids = np.empty((k, Z.shape[1]))
    centroids[0] = Z[rng.integers(n)]
    closest = ((Z - centroids[0]) ** 2).sum(axis=1)
    for i in range(1, k):
        total = closest.sum()
        if not np.isfinite(total) or total <= 0:
            centroids[i] = Z[rng.integers(n)]
        else:
            centroids[i] = Z[rng.choice(n, p=closest / total)]
        closest = np.minimum(closest, ((Z - centroids[i]) ** 2).sum(axis=1))
    return centroids


def _sq_dist(Z, centroids):
    """(T, K) squared euclidean distance from every row to every centroid."""
    return ((Z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)


def _viterbi(cost, penalty):
    """Exact minimiser of the penalised assignment problem."""
    T, K = cost.shape
    switch = penalty * (1.0 - np.eye(K))
    V = np.empty((T, K))
    back = np.zeros((T, K), dtype=int)
    V[0] = cost[0]
    for t in range(1, T):
        M = V[t - 1][:, None] + switch      # M[j, k] = come from j, land on k
        back[t] = M.argmin(axis=0)
        V[t] = cost[t] + M.min(axis=0)

    labels = np.zeros(T, dtype=int)
    labels[-1] = int(V[-1].argmin())
    for t in range(T - 1, 0, -1):
        labels[t - 1] = back[t, labels[t]]
    return labels, float(V[-1].min())


# -- causal labelling --------------------------------------------------

def label_walk_forward(features, n_states=3, jump_penalty=20.0,
                       min_train=252, refit_every=21, seed=0):
    """Regime label at every bar, using only that bar's past.

    The model is refit every `refit_every` bars on the expanding window and
    the label of the window's final bar is kept. Between refits the stored
    centroids are reused, which is what a live system would do anyway -
    refitting on every bar is both slower and no more honest.

    Returns a pandas Series of names, NaN before `min_train`.
    """
    X = np.asarray(features, dtype=float)
    index = features.index if hasattr(features, 'index') else pd.RangeIndex(len(X))
    out = pd.Series(index=index, dtype=object)
    if len(X) <= min_train:
        return out

    model = None
    for i in range(min_train, len(X)):
        if model is None or (i - min_train) % refit_every == 0:
            Z = standardize(X[:i + 1], upto=i)
            try:
                model = JumpModel(n_states=n_states, jump_penalty=jump_penalty,
                                  seed=seed).fit(Z)
            except ValueError:
                continue
            out.iloc[i] = model.names_[int(model.labels_[-1])]
        else:
            Z = standardize(X[:i + 1], upto=i)
            out.iloc[i] = model.label_last(Z)
    return out


def state_summary(labels):
    """Share of time and mean run length per state.

    A state that lasts three days is a filter artifact, not a regime. This
    is the number that makes that obvious, and it is the reason the jump
    penalty exists.
    """
    s = pd.Series(labels).dropna()
    if s.empty:
        return {}
    runs = (s != s.shift()).cumsum()
    lengths = s.groupby(runs).agg(['first', 'size'])
    out = {}
    for name, grp in lengths.groupby('first'):
        out[str(name)] = {
            'share': round(float((s == name).mean()), 3),
            'n_runs': int(len(grp)),
            'mean_run_length': round(float(grp['size'].mean()), 1),
            'median_run_length': float(grp['size'].median()),
        }
    return out


# -- per-state sleeve weights ------------------------------------------

def estimate_state_weights(samples, prior, shrinkage=0.5, ridge=1.0,
                           min_samples=60):
    """Sleeve weights per regime, estimated then shrunk toward the prior.

    samples: list of (state_name, {sleeve: score}, forward_return).
    prior:   the existing hand-set table, {state: {sleeve: weight}}.

    Within each state, ridge-regress the forward return on the sleeve
    scores, clip negative coefficients to zero (this is a long-only score
    where a negative sleeve weight would mean deliberately buying what the
    sleeve dislikes), normalise, then blend with the prior.

    Shrinkage is not optional decoration. With 80 names and a few hundred
    rebalances per state there is nowhere near enough data to trust a raw
    fit, and the point of the exercise is to let the data move fifteen
    hand-set numbers a little, not to hand them over to it.
    """
    sleeves = sorted(next(iter(prior.values())).keys())
    grouped = {}
    for state, scores, fwd in samples:
        if state is None or not np.isfinite(fwd):
            continue
        row = [float(scores.get(s) or 0.0) for s in sleeves]
        grouped.setdefault(state, ([], []))
        grouped[state][0].append(row)
        grouped[state][1].append(float(fwd))

    out, diagnostics = {}, {}
    for state, base in prior.items():
        fitted = None
        X_list, y_list = grouped.get(state, ([], []))
        n = len(y_list)
        if n >= min_samples:
            X = np.asarray(X_list, dtype=float)
            y = np.asarray(y_list, dtype=float)
            X = X - X.mean(axis=0)
            y = y - y.mean()
            A = X.T @ X + ridge * np.eye(X.shape[1])
            try:
                coef = np.linalg.solve(A, X.T @ y)
            except np.linalg.LinAlgError:
                coef = None
            if coef is not None:
                pos = np.clip(coef, 0.0, None)
                if pos.sum() > 0:
                    fitted = pos / pos.sum()

        prior_vec = np.array([base[s] for s in sleeves], dtype=float)
        prior_vec = prior_vec / prior_vec.sum()
        if fitted is None:
            blended = prior_vec
        else:
            blended = shrinkage * fitted + (1.0 - shrinkage) * prior_vec
            blended = blended / blended.sum()
        out[state] = {s: round(float(w), 4) for s, w in zip(sleeves, blended)}
        diagnostics[state] = {
            'n_samples': n,
            'fitted': fitted is not None,
            'raw': {s: round(float(w), 4) for s, w in zip(sleeves, fitted)}
            if fitted is not None else None,
        }
    return out, diagnostics
