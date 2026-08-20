import numpy as np
import pandas as pd
import pytest

import factor_search as fs
from quant_engine import SymbolAnalyzer


@pytest.fixture
def analyzer(noisy_df):
    return SymbolAnalyzer(noisy_df)


# -- the language is a whitelist ---------------------------------------

@pytest.mark.parametrize('expr', [
    'close',
    'zscore(close, 20) - zscore(volume, 20)',
    'ts_delta(mfi14, 5) / atr14',
    'tanh(don_pos * 2)',
    'clip(obv_z, -3, 3)',
    '-ts_mean(ret1, 10)',
    'ts_rank(rel_vol, 63) * sign(macd_hist)',
])
def test_accepts_valid_expressions(expr):
    assert fs.validate(expr) is not None


@pytest.mark.parametrize('expr, fragment', [
    ('__import__("os").system("ls")', 'unknown function'),
    ('close.rolling(5).mean()', 'unknown function'),
    ('open', 'unknown operand'),
    ('ts_mean(close, 20)[0]', 'Subscript'),
    ('lambda x: x', 'Lambda'),
    ('[close for _ in range(3)]', 'not allowed'),
    ('close ** 2', 'only + - * /'),
    ('close if close else low', 'not allowed'),
    ('ts_mean(close, w=20)', 'keyword'),
    ('eval("1")', 'unknown function'),
    ('close + "a"', 'numeric constants'),
])
def test_rejects_everything_else(expr, fragment):
    with pytest.raises(fs.ExpressionError) as exc:
        fs.validate(expr)
    assert fragment in str(exc.value)


def test_rejects_expressions_deeper_than_the_limit():
    deep = 'close'
    for _ in range(fs.MAX_DEPTH + 2):
        deep = f'ts_mean({deep}, 3)'
    with pytest.raises(fs.ExpressionError, match='deeper than'):
        fs.validate(deep)


def test_canonical_collapses_equivalent_spellings():
    assert fs.canonical('close-low') == fs.canonical('close - low')
    assert fs.canonical('ts_mean(close,20)') == fs.canonical('ts_mean( close , 20 )')
    assert fs.canonical('close - low') != fs.canonical('low - close')


# -- evaluation --------------------------------------------------------

def test_evaluate_returns_a_series(analyzer):
    out = fs.evaluate('zscore(close, 20)', analyzer)
    assert isinstance(out, pd.Series)
    assert len(out) == analyzer.n


def test_evaluate_rejects_scalar_results(analyzer):
    with pytest.raises(fs.ExpressionError, match='series'):
        fs.evaluate('logv(3.0)', analyzer)


def test_evaluate_is_causal(analyzer, noisy_df):
    """A value at bar i must not move when later bars are appended."""
    short = SymbolAnalyzer(noisy_df.iloc[:200])
    expr = 'zscore(close, 20) - ts_mean(ret1, 10)'
    full_vals = fs.evaluate(expr, analyzer).iloc[:200]
    short_vals = fs.evaluate(expr, short)
    pd.testing.assert_series_equal(full_vals.iloc[-50:], short_vals.iloc[-50:],
                                   check_exact=False, atol=1e-10)


def test_evaluate_scrubs_infinities(analyzer):
    out = fs.evaluate('close / ts_delta(close, 1)', analyzer)
    assert not np.isinf(out.dropna()).any()


def test_operands_all_resolve(analyzer):
    env = fs.operand_series(analyzer)
    assert set(env) == set(fs.OPERANDS)
    for name, series in env.items():
        assert isinstance(series, pd.Series), name


def test_describe_language_lists_everything():
    doc = fs.describe_language()
    for name in fs.OPERANDS:
        assert name in doc
    for name in fs.OPERATORS:
        assert name in doc


# -- scoring and the gate ----------------------------------------------

def _panel(n_dates=40, n_syms=30, seed=0, signal=0.0):
    rng = np.random.default_rng(seed)
    fwd = rng.normal(0, 0.02, size=(n_dates, n_syms))
    sleeves = {s: rng.normal(0, 0.5, size=(n_dates, n_syms))
               for s in fs.SLEEVES}
    return {'fwd': fwd, 'sleeves': sleeves,
            'symbols': [f'S{i}' for i in range(n_syms)],
            'bar': np.zeros((n_dates, n_syms), dtype=int),
            'dates': list(range(n_dates)), 'positions': [], 'step': 5}


def test_ic_series_detects_a_planted_signal():
    panel = _panel()
    # values perfectly ranked with forward returns
    ics = fs._ic_series(panel['fwd'] * 3.0, panel['fwd'])
    assert np.allclose(ics, 1.0)


def test_ic_series_is_near_zero_on_noise():
    panel = _panel(seed=2)
    rng = np.random.default_rng(99)
    ics = fs._ic_series(rng.normal(size=panel['fwd'].shape), panel['fwd'])
    assert abs(float(np.mean(ics))) < 0.15


def test_summarize_reports_decay():
    ics = np.concatenate([np.full(20, 0.10), np.full(20, 0.02)])
    out = fs._summarize(ics)
    assert out['first_half_ic'] == pytest.approx(0.10)
    assert out['second_half_ic'] == pytest.approx(0.02)
    assert out['decay_ratio'] == pytest.approx(0.2)


def test_summarize_needs_enough_dates():
    assert fs._summarize(np.array([0.1, 0.2])) is None


def test_gate_rejects_weak_t():
    stats = {'t_stat': 1.0, 'coverage': 0.9, 'decay_ratio': 1.0,
             'pct_positive': 0.6, 'max_sleeve_corr': 0.1}
    ok, why = fs.passes_discovery(stats)
    assert not ok and '< 2.5' in why


def test_gate_rejects_decaying_factor():
    stats = {'t_stat': 3.0, 'coverage': 0.9, 'decay_ratio': 0.1,
             'pct_positive': 0.7, 'max_sleeve_corr': 0.1}
    ok, why = fs.passes_discovery(stats)
    assert not ok and 'decays' in why


def test_gate_rejects_redundant_factor():
    stats = {'t_stat': 3.0, 'coverage': 0.9, 'decay_ratio': 1.0,
             'pct_positive': 0.7, 'max_sleeve_corr': 0.95}
    ok, why = fs.passes_discovery(stats)
    assert not ok and 'correlated' in why


def test_gate_rejects_thin_coverage():
    stats = {'t_stat': 3.0, 'coverage': 0.2, 'decay_ratio': 1.0,
             'pct_positive': 0.7, 'max_sleeve_corr': 0.1}
    ok, why = fs.passes_discovery(stats)
    assert not ok and 'covers' in why


def test_gate_accepts_a_clean_factor():
    stats = {'t_stat': 3.0, 'coverage': 0.9, 'decay_ratio': 1.0,
             'pct_positive': 0.7, 'max_sleeve_corr': 0.2}
    ok, why = fs.passes_discovery(stats)
    assert ok


def test_gate_handles_negative_t_with_consistent_sign():
    stats = {'t_stat': -3.0, 'coverage': 0.9, 'decay_ratio': 1.0,
             'pct_positive': 0.3, 'max_sleeve_corr': 0.2}
    ok, _ = fs.passes_discovery(stats)
    assert ok  # a reliably inverted factor is still a factor


def test_gate_rejects_unscoreable():
    assert fs.passes_discovery(None)[0] is False
    assert fs.passes_discovery({'error': 'x'})[0] is False


# -- proposals ---------------------------------------------------------

def test_random_proposals_are_all_valid():
    props = fs.propose_random(25, seed=5)
    assert len(props) == 25
    for p in props:
        assert fs.validate(p['expression']) is not None
        assert p['thesis']


def test_random_proposals_are_distinct():
    props = fs.propose_random(25, seed=5)
    keys = {fs.canonical(p['expression']) for p in props}
    assert len(keys) == len(props)


def test_random_proposals_vary_with_seed():
    a = {p['expression'] for p in fs.propose_random(10, seed=1)}
    b = {p['expression'] for p in fs.propose_random(10, seed=2)}
    assert a != b
