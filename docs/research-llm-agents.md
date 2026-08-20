# LLM and AI trading research applied to Stock Indicator

A survey of the 2023 to August 2026 literature on LLM trading agents, LLM factor
mining, text signals, reinforcement learning, time-series foundation models, regime
models and backtest methodology, read against this codebase. Roughly 120 papers were
checked across five strands; the ones that matter are indexed at the end with links.

The question asked of every paper was the one the README asks of every sleeve: is
there out-of-sample evidence, and would it survive in an 80-name daily-bar universe
with no lookahead?

**Implementation status.** Items 1, 2, 3, 4, 6 and 7 of the build order below are
implemented and tested, and the price cache has been deepened from five years to
fifteen; see [What was built](#what-was-built) for the modules, the commands and the
results. The short version: the largest single improvement was the download, not any
of the modelling. Over ten years the composite beats the equal-weight watchlist by 47
points, its information coefficient clears what shuffled data produces, and the
probability of backtest overfitting fell from 0.67 to 0.48 - but its IC t-stat of
1.44 still does not clear the multiple-testing hurdle, costs remove 40% of the CAGR
at 20 bps, and its Sharpe halves in bear regimes. An earlier pass on the five-year
cache concluded the selection edge had vanished entirely; that conclusion was an
artifact of the window, which is itself the most useful thing this exercise
demonstrated.

## The short version

1. **Do not put an LLM in the scoring or trading path.** The multi-agent trader
   papers (TradingAgents, FinCon, FinMem, FinAgent, HedgeAgents) report Sharpe 2-8
   on 3-8 stocks over 3-9 months using models whose training data covers the test
   window. Every independent re-evaluation since - FinSABER over 20 years, Profit
   Mirage before vs after the model's cutoff, StockBench on post-cutoff data, the
   ICAIF 2026 TradingAgents replication, The Alpha Illusion with costs - finds the
   edge goes to zero or below buy-and-hold. A causal OHLCV factor model with regime
   gating is the control group these systems keep losing to. That is what this repo
   already is.
2. **The one LLM technique with a defensible design is factor proposal.** AlphaAgent
   (KDD 2025), RD-Agent(Q) (NeurIPS 2025), QuantaAlpha and AlphaMemo (2026) use the
   model to write candidate factor formulas and a deterministic backtester to judge
   them. The reported ICs are not trustworthy (most backbones' cutoffs sit inside the
   test windows), but the architecture is sound if the model is blind to prices,
   dates and tickers and sees only operator names and IC feedback. It fits directly
   on top of `quant_indicators.py` and `run_backtest`.
3. **The strongest evidence-backed upgrade needs no LLM at all**: replace the hand
   3x5 regime table with a statistical jump model plus per-state sleeve weights
   shrunk toward the current table (Shu, Yu and Mulvey 2024), and use the market
   overlay to scale gross exposure rather than just display it (Daniel and Moskowitz
   2016; Goulding, Harvey and Mazzoleni 2023). FinSABER's closing recommendation is
   literally "trend detection and regime-aware risk controls over scaling framework
   complexity."
4. **Text signals are real but small, short-horizon, small-cap and decaying.**
   Lopez-Lira and Tang's own numbers fall from Sharpe 6.5 (2021Q4) to 1.2 (2024) and
   go negative at 20 bp of cost; Chen, Kelly and Xiu find news predictability
   "dissipates quickly for large stocks." For this universe, expect zero standalone
   IC. The right move is to start collecting timestamped headlines now, score them
   with a cutoff-safe model, and keep the sleeve at weight zero until it has six
   months of forward IC.
5. **Reinforcement learning and deep sequence models are not worth building here.**
   The most rigorous RL study (Kashif and Slepaczuk 2026: 16 walk-forward folds,
   three markets, survivorship-free, with costs) finds no significant excess return
   over buy-and-hold. Zero-shot Chronos/TimesFM have negative R-squared on daily
   returns across 94 countries (Rahimikia et al. 2025); gradient-boosted trees win.
   Sixty thousand stock-days is far below what any of these need.
6. **Where LLMs do earn a place is as a narrator and a critic**, both downstream of
   deterministic checks: a morning brief that can only cite numbers present in the
   scan JSON, and a rebalance reviewer whose `blocking` flag can only be true when a
   Python check failed. The 2026 explainability papers (Geng et al.; Zandi et al.)
   show LLMs reproduce a supplied ranking reliably but get the direction of effects
   wrong when left to infer them, so the pipeline design matters more than the model.
7. **The backtester was missing the things the methodology literature says matter
   most**: transaction costs, a deflated t-stat that counts every configuration
   tried, probability of backtest overfitting, a bull/bear split, and a permutation
   audit. They were cheap to add and they are the prerequisites for any of the
   experiments above to mean anything. They are now in `evaluation.py`, and they
   promptly overturned one headline claim and then, once the data was deep enough,
   partly restored it.
8. **Before any of it, get more data.** At N=80 the per-date IC standard error is
   about 0.11, so statistical power comes from the number of rebalances rather than
   the size of the cross-section. Going from five years of daily bars to fifteen
   changed more conclusions than every modelling change attempted here combined, and
   cost one download. This should have been the first item, not an afterthought.

## Where the field stands

### LLM trading agents do not survive honest evaluation

The agent frameworks share a pattern: an LLM (usually GPT-4 class) reads prices, news
and sometimes charts, debates with itself through analyst/researcher/trader/risk
roles, and emits buy/sell decisions. TradingAgents (arXiv 2412.20138) reports Sharpe
5.6-8.2 on AAPL, GOOGL and AMZN over January-March 2024 with o1-preview, which was
trained on that period. FinMem (2311.13743), FinAgent (KDD 2024), FinCon (NeurIPS
2024) and HedgeAgents (WWW 2025) have the same shape: 5-8 tickers, under a year, test
window inside the model's training data.

What happened when people checked:

| study | what it did | result |
| --- | --- | --- |
| FinSABER (KDD 2026, 2505.07078) | FinMem and FinAgent over 2004-2024, 100+ symbols incl. delisted, vs B&H, SMA, XGBoost, RL | Replicates the original windows, then: random-5 universe Sharpe 0.25 / 0.09 vs B&H 0.31; bear Sharpe -0.38 vs -0.28; no alpha (p > 0.34) |
| Profit Mirage (2510.07920) | Five frameworks with GPT-4o, in-cutoff 2021 vs post-cutoff 2024 with matched market returns | Sharpe falls 51-62%, total return 50-72%; GPT-4o recalls historical prices at 85% accuracy |
| StockBench (2510.02209) | 14 frontier models, 20 DJIA names, Mar-Jun 2025 (post-cutoff) | "Most LLM agents fail to outperform" B&H; 0 of 14 beat it in the Jan-Apr 2025 drawdown; reasoning models underperform instruct models |
| ICAIF 2026 replication | TradingAgents on GOOG, May-Jul 2025 | 15.8-18.1% vs 19.1% buy-and-hold; "do not currently justify their complexity" |
| The Alpha Illusion (2605.16895) | Five frameworks with commission, spread, impact and token cost | 35 of 40 system x friction cells unmodeled in the originals; TradingAgents Sharpe 0.43 to 0.22, below B&H |
| KTD-Fin (2605.28359) | Ten agents on CSI 300 with ticker/date masking and Barra attribution | Returns "largely explained by passive market and style exposure" |
| Agentic Trading audit (2605.19337) | 77 studies | Of 19 with closed-loop evaluation, 2 report time-consistent splits, 1 models costs, 1 handles survivorship; none fully reproducible |

LiveTradeBench (2511.03628) ran 21 models live for 50 days and found a Spearman
correlation of 0.05 between LMArena rank and trading return. Agent Market Arena found
the agent architecture explains more variance than the backbone. TradeTrap
(2512.02261) showed small perturbations to any component of an agent loop produce
runaway concentration. The conclusion the field itself has reached is the one in The
Alpha Illusion: use LLMs as "auditable information interfaces upstream of independent
calibration, risk, and execution modules."

### Lookahead bias is the central methodological problem

Any LLM trained after the backtest period has seen the answers. This is now measured
rather than argued:

- Glasserman and Lin (2309.17322): replacing company names with nonsense tokens
  *raised* in-sample GPT sentiment returns (25 to 31 bp/day), because the model's
  knowledge of the firm was a distraction as much as a leak. Anonymization is a
  validated, free remedy.
- Sarkar and Vafa (SSRN 4754678): prompting "pretend it is 2019" does not remove
  leakage; only models trained before the period do.
- Levy (JAR 2026): perturbing the last digit of financial statements drops GPT-4's
  earnings-direction accuracy from 60% to chance. The Kim-Muhn-Nikolaev "Financial
  Statement Analysis with LLMs" paper that made that claim was withdrawn in February
  2025.
- Didisheim, Fraschini and Somoza (Econ Letters 2025): memorization is strong for
  index levels and macro series, minimal for daily single-stock data. Good for a
  daily stock signal, bad for any regime or macro prompt.
- ChronoBERT/ChronoGPT (2502.21206) and DatedGPT (2603.11838) release year-sliced
  models with annual cutoffs; they show the lookahead premium for next-day news
  signals is modest (about 26 bp per standard deviation) but non-zero.
- MemGuard-Alpha (2603.26797): on S&P 100 names, contaminated signals earn 2 bp/day
  in-sample vs 14.5 bp clean, i.e. memorization inflates in-sample and hurts live.
- Kong et al. (2602.14233) audit 164 finance-LLM papers: no single bias (lookahead,
  survivorship, narrative, objective, cost) is discussed in more than 28% of them.

Practical rule for this repo: any LLM output used in a backtest must come from a
model whose cutoff predates the window, or from a forward test that starts the day
collection starts. The `test_no_lookahead` discipline has to extend to the model.

### The transferable LLM idea: factor proposal engines

The alpha-mining line uses the LLM to write formulas and a backtester to score them.

| system | venue | mechanism | reported (S&P 500 where given) | caveat |
| --- | --- | --- | --- | --- |
| AlphaGen (2306.12964) | KDD 2023 | RL over reverse-Polish formula tokens, reward = marginal IC of the combined pool | CSI300 IC 0.073 | pre-LLM baseline; code released |
| AlphaForge (2406.18394) | AAAI 2025 | generator plus a dynamic combiner that re-scores the factor zoo on trailing IC each date | nine-month real account +21.7% excess on CSI500 | the only dynamic-weighting idea with real-money evidence |
| AlphaAgent (2502.16789) | KDD 2025 | idea agent, factor agent with AST originality and complexity penalties, eval agent | IC 0.0056, ICIR 0.055, 8.7% excess, IR 1.05 | factor-level IC indistinguishable from zero; backbones cover the test window |
| RD-Agent(Q) (2505.15155) | NeurIPS 2025 | hypothesis, Co-STEER code generation, Qlib backtest, bandit scheduler | CSI300 IC 0.053 vs Alpha158 0.034; under $10 per session | scores IC 0.0019 on S&P 500 in AlphaAgent's replication |
| QuantaAlpha (2602.07085) | 2026 | evolutionary search over whole mining trajectories | CSI300 IC 0.15 | double anything else in the literature; no leakage control |
| AlphaMemo (2606.20625) | 2026 | memory of which AST edits worked | S&P 500 IC 0.041, RankICIR 0.20, Sharpe 1.07 | authors: offline discovery, not live use |
| AlphaEval (2508.13174) | KDD 2026 | backtest-free scoring of candidate factors | LLM-mined alphas score highest on predictive power and lowest on perturbation robustness (AlphaAgent 0.42 vs AlphaGen 1.00) | independent evaluator |

Every positive result is on Chinese A-shares with 300-1000+ names or the full S&P
500; US factor ICs are 0.003-0.04. CogAlpha (ACL 2026) is the only paper that
includes an 80-ish name universe (the 89-stock HSI) and does not break out its
results. The honest reading: LLMs are a competent proposal engine, and the
performance claims are not yet trustworthy. The architecture is worth borrowing; the
numbers are not.

### Alpha-mining detail: what each system actually does

Worth keeping the mechanisms straight, because the useful parts are separable from
the reported numbers.

- **AlphaGen** (KDD 2023) is the pre-LLM baseline: a PPO policy emits reverse-Polish
  formula tokens over about 20 operators, and the reward is the *marginal* IC gain of
  a linearly combined pool rather than any single alpha's IC, so the pool is optimised
  for synergy. Test IC 0.0725 on CSI300 (2020-2021) against 0.018 for genetic
  programming and 0.026 for LightGBM. **AlphaQCM** (ICML 2025) rebuilds the same
  search with distributional RL and states plainly that the gain is largest "on large
  datasets comprising numerous stocks" — an explicit warning for an 80-name universe.
- **AlphaForge** (AAAI 2025) is the one with real-money evidence: a nine-month live
  CSI500 account returned 21.7% excess. Its transferable idea is the *dynamic
  combiner* — at each date, re-score the whole factor zoo on trailing n-day IC/ICIR,
  keep what clears a threshold, take the top N and refit. That is the cleanest
  published form of "replace fixed weights with rolling-IC weights", and it is a
  statistical procedure, not an LLM one. The lookback n is itself a fitted parameter.
- **AlphaAgent** (KDD 2025) contributes three regularizers aimed squarely at alpha
  decay: an AST-subtree originality penalty against Alpha101 and the existing pool, an
  LLM-judged check that the stated hypothesis matches the formula it produced, and a
  complexity penalty on tree depth and parameter count. Its hit ratio for top-5%
  alphas is 0.29 with the regularizers and 0.16 without — which is the part worth
  copying, regardless of what one thinks of the headline IC.
- **RD-Agent(Q)** (NeurIPS 2025) is the most engineering-complete open codebase: a
  research stage writes hypotheses, Co-STEER writes and debugs the factor code, Qlib
  runs the real backtest, and a multi-armed bandit allocates between factor-side and
  model-side work. 44 loops, 24 valid, 8 promoted, under $10 of tokens per session.
  The authors note it "relies solely on the LLM's internal financial knowledge" and
  has no online regime adaptation.
- **AlphaEval** (KDD 2026) is the independent evaluator, and the most useful single
  result in this literature: scoring candidate factors *without* a backtest on five
  axes (predictive power, rank-entropy stability, perturbation fidelity, LLM logic
  score, diversity entropy), it finds LLM-mined alphas rank highest on predictive
  power and **lowest on robustness** — AlphaAgent scores 0.415 on perturbation
  fidelity against AlphaGen's 0.997. It also finds alphas with perturbation fidelity
  above 0.9 have significantly lower drawdowns. Usable as a cheap pre-filter before
  spending a real backtest trial.
- **MadEvolve** (2026) is the honest counterexample on process: it burns 743-1,059
  candidates per run and assesses its own p-hacking by comparing in-sample to
  out-of-sample degradation against the multiple-testing discount, rather than
  ignoring the question. **Beyond Prompting** (2026) gates on IC t >= 3 plus an FF5
  redundancy screen with a library frozen in Dec 2020 and a blind 2021-2024 window,
  but never discloses how many candidates were tested — so its t > 3 cannot actually
  be checked against effective multiplicity. Both illustrate that the trial count is
  the load-bearing number.

**Label construction (Trading-R1, arXiv 2509.11420).** Independent of anything LLM,
its labelling scheme is worth borrowing: forward returns normalised by trailing
volatility, measured over multiple horizons, then bucketed by asymmetric quantiles
into discrete classes. It makes targets comparable across names of very different
volatility, which is exactly the problem an 80-name mixed stock/ETF cross-section has.

### Text signals: real, small, short, small-cap, decaying

| paper | finding |
| --- | --- |
| Lopez-Lira and Tang (JFE 2026, 2304.07619) | GPT-4 on headlines, 4,123 firms, Oct 2021-May 2024 (post-cutoff). Tradable drift Sharpe 6.5 (2021Q4), 3.7 (2022), 2.3 (2023), 1.2 (Jan-May 2024); unprofitable at 20 bp round-trip; concentrated in small caps and negative news |
| Chen, Kelly and Xiu (SSRN 4416687) | frozen-LLM embeddings of articles beat bag-of-words across 16 markets; persists days for small stocks, "dissipates quickly for large stocks" |
| Yilki (2606.29290) | FinBERT embeddings of 10-K MD&A for 255 S&P 500 firms, 2011-2025: long-short Sharpe 0.86, FF5 alpha 7.3%/yr - a realistic large-cap magnitude |
| Yang, "When Valid Signals Fail" (2604.10996) | LLM features with held-out IC > 0.15 became noise during a macro shock; a price-only agent won |
| FinCall-Surprise (ACL 2026) | earnings-surprise prediction from call text: "high accuracy is an illusion caused by class imbalance" |

Sources a hobbyist can actually get, and whether their timestamps support a
no-lookahead backtest:

| source | point-in-time? | practical limit |
| --- | --- | --- |
| yfinance `Ticker.news` | `content.pubDate` in ISO, but **no history** (10-50 latest items) | already a dependency, zero cost, forward-collect only |
| Finnhub company-news | unix `datetime`, **1 year back** on the free tier, 60 calls/min | best free backfill; some items are republished, so treat the stamp as publication, not first availability |
| Alpha Vantage `NEWS_SENTIMENT` | `time_published` to the minute, `time_from`/`time_to`, history from ~Mar 2022 | 25 requests/day free, so a 3-year backfill is slow but possible; ships its own sentiment score |
| SEC EDGAR full-text + XBRL | acceptance timestamps to the second; filings after 17:30 ET count as next day | the gold standard, free, 10 req/s; 8-K items 2.02 and 8.01 are the event feed |
| GDELT DOC 2.0 | 15-minute stamps, but the DOC API is a rolling 3-month window | no tickers, organisation-name matching only, noisy for large caps ("Apple") |
| earnings transcripts | call date only, not release time | Roic.ai free 2 years, API Ninjas free non-commercial; fine for T+1 use |
| Polygon news | — | no usable free news tier |
| Reddit | forward only; Pushshift is gone | 100 QPM, non-commercial, needs approval |
| X / Twitter | — | pay-per-read, no free tier |

Scoring should use a small local model — FinBERT (110M, runs on CPU) or the
year-sliced ChronoGPT-Instruct checkpoints from `manelalab` — with tickers and
company names replaced by a placeholder. Glasserman and Lin's result makes the
placeholder free: anonymising *raised* returns because the model's firm knowledge was
a distraction as much as a leak. Score items whose `pubDate` precedes the close to
day T and everything else to T+1, and log the model name, prompt hash and raw text so
a score can be reproduced later. FinMTEB (EMNLP 2025) is worth knowing before picking
an embedding model: FinBERT beats general BERT by 15.6% on financial tasks, general
MTEB rank barely correlates with financial performance, and bag-of-words still beats
dense embeddings on financial semantic similarity.

### Reinforcement learning and deep time-series models

- Kashif and Slepaczuk (2605.17307): SAC on Nasdaq-100, Nikkei and Euro Stoxx,
  2003-2026, 16 walk-forward folds, survivorship-free, 2 bp costs: "no strategy
  achieves statistically significant excess returns relative to buy and hold under
  HAC-robust inference." 14-23 GPU-hours per fold.
- FinRL-DeepSeek (2502.07393): information ratios near zero for every agent;
  128 GB RAM. FLAG-Trader: a 135M-parameter LLM as PPO policy tested over seven
  months of a bull market.
- Rahimikia, Ni and Wang (2511.18578): daily excess returns, 94 countries, about two
  billion observations. Zero-shot Chronos-large R-squared -1.37%, TimesFM -2.80%;
  fine-tuning "fails to close the gap"; CatBoost/LightGBM are the best models.
  Tan et al. (NeurIPS 2024) show removing the LLM from LLM-for-time-series models
  does not hurt.
- Kronos (AAAI 2026, 2508.02739) is the one financial-native foundation model with
  open weights (4M-102M parameters, runs on CPU). Zero-shot RankIC about 0.025 on
  its own benchmark; no US large-cap portfolio evidence.
- Qlib and FinTSB benchmarks: on hand-engineered tabular features LightGBM/XGBoost
  (IC 0.045-0.050 on CSI300) beat Transformers (0.026) and LSTMs (0.032). MASTER's
  edge over XGBoost is about 25% of IC on 300-800 names with 12 years of training.
- Nagel (NBER 2025) shows the "virtue of complexity" forecasts reduce to a
  volatility-timed momentum strategy; Fallahgoul (2506.03780) bounds the data a
  high-dimensional model needs at 25-30 years.

With about 750 daily bars and 80 names, the only defensible ML is a shallow
gradient-boosted ranker or ridge regression over the existing indicator features, run
as a challenger to the hand-weighted composite.

### Regime detection: the best-evidenced upgrade

- Shu, Yu and Mulvey (2402.05272, code at github.com/Yizhan-Oliver-Shu/jump-models):
  statistical jump model (k-means on return/risk features plus a per-switch penalty,
  with the penalty chosen by cross-validation on strategy performance, not label
  fit). US/DE/JP indices 1990-2023 with costs and a one-day delay: lower volatility
  and max drawdown and higher Sharpe than HMM and buy-and-hold.
- Shu, Yu and Mulvey (Annals of OR 2024, 2406.09578): per-asset jump-model labels, a
  gradient-boosted classifier forecasting the next state, then allocation. This is
  the direct analogue of replacing `REGIME_WEIGHTS`.
- Daniel and Moskowitz (JFE 2016): momentum crashes follow bear markets with high
  volatility; dynamic scaling roughly doubles Sharpe. Goulding, Harvey and Mazzoleni
  (JFE 2023): slow and fast trend signals both negative implies negative expected
  momentum return. Suominen and Hjalmarsson (2026): time-series momentum fails at
  valuation extremes.
- Cederburg et al. (JFE 2020): single-strategy volatility timing generally earns
  *lower* real-time Sharpe than unmanaged; DeMiguel, Martin-Utrera and Uppal (JF
  2024): the multifactor version survives out of sample net of costs. So gate at the
  portfolio and sleeve level, not per factor.
- LLM regime labeling (Yi et al. 2605.30363) detects FOMC policy breaks with F1 0.82
  but shows no allocation gain, and macro text is where memorization is worst.

### LLMs as narrators and critics

- Geng et al. (2602.18895): LLMs "reliably reproduce reference rankings under
  controlled prompts but show limited alignment when generating explanations
  autonomously." Feed the model the ranking; do not ask it to infer one.
- Zandi et al. (2608.17715): across three pipelines and three models narrating a
  credit model, pipeline design mattered more than model choice, and models were
  inconsistent on the *direction* of a factor's influence.
- FinGround (ACL 2026): decomposing output into atomic claims and verifying each
  against the source cut hallucinations 68%.
- Xue (2605.28850): structured risk feedback changes LLM trading behavior but is
  "not a universal performance enhancer"; placebo feedback sometimes did better;
  LLMs have a "correlation blind spot" for coupled positions. PortBench
  (2605.27887): 90% of model-profile cases fail to beat equal weight in 2024 while
  satisfying format constraints. OpenPM (2608.09988) lands on the design the
  evidence supports: a deterministic critic enforces typed constraints, the LLM
  explains.
- Reward hacking (Thaman 2605.02964; METR 2025): agents given write access to their
  evaluator rewrote it in up to 30% of tasks. Any research loop must keep the
  backtester read-only and out of the agent's process.

### What the backtester is missing

- Bailey and Lopez de Prado, Deflated Sharpe Ratio (SSRN 2460551) and Probability of
  Backtest Overfitting via CSCV (SSRN 2326253); Harvey, Liu and Zhu (RFS 2016):
  t > 3 for a new factor. "Pseudo-Mathematics and Financial Charlatanism" (AMS
  2014): with five years of data, trying more than about 45 configurations almost
  guarantees an in-sample Sharpe of 1 with an out-of-sample expectation of zero.
- Arian, Norouzi and Seco (KBS 2024): plain walk-forward has the weakest
  false-discovery control of the tested schemes; combinatorial purged CV is best.
- Nikolopoulos (2604.15531): run the entire pipeline, including the specification
  search, on zero-predictability synthetic data; if it still finds significance, the
  workflow is the problem. The horizon tilts in `HORIZON_PARAMS` were chosen
  in-sample and are the obvious candidate for this audit.
- McLean and Pontiff (JF 2016): published predictors lose 26% out of sample and 58%
  post-publication, fastest for high-return, low-liquidity predictors.
- Azevedo, Hoegner and Velikov: costs and decay cut ML strategy returns about 57%.

## What to build, in order

| # | change | evidence | effort | LLM | module | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Backtester hardening: costs, deflated t, PBO, bull/bear split, 10y history, forward log | strong, methodological | low | no | `backtest.py`, `evaluation.py` | **done** |
| 2 | Jump-model regime with per-state weights shrunk to the hand table | strong (Shu-Yu-Mulvey, with costs) | medium | no | `regime.py`, `regime_calibrate.py` | **done — calibration lost out of sample, prior kept** |
| 3 | Overlay-driven gross exposure scaling | strong for drawdown, not alpha | low | no | `market.py`, `recommender.py` | **done** |
| 4 | `factor_search.py`: LLM proposes formulas, backtester judges, gate counts trials | moderate architecture, weak numbers | medium | yes, offline | `factor_search.py` | **done — no survivors yet** |
| 5 | Ridge / LightGBM ranker challenger over existing features | strong that trees beat deep nets; weak that either beats hand weights at N=80 | medium | no | new `challenger.py` | not built |
| 6 | Morning brief: grounded narration of scan JSON with claim verification | moderate (translator role works) | medium | yes | `brief.py`, `/api/brief` | **done** |
| 7 | Rebalance critic: Python checks, LLM explains, cannot relax a limit | moderate | medium | yes | `checks.py`, `/api/review` | **done** |
| 8 | News sleeve at weight zero with a forward-IC tracker | real but small; zero expected here | medium | small local model | `data_store.py`, new job | not built |
| 9 | Kronos-small / Chronos-bolt zero-shot as a sixth-sleeve candidate | weak | low (one afternoon) | no | `factor_search` scoring | not built |
| - | Multi-agent LLM trader, RL policies, memory/reflection modules, LLM regime labels, LLM reading filings into the score, MASTER/Stockformer-class nets | negative or absent | high | - | - | skip |

### 1. Backtester hardening

All of this sits in `backtest.py` and uses the `per_date` structure `run_backtest`
already builds.

- **Transaction costs.** Per rebalance, `cost = sum(|dw_i|) * c` with `c` swept
  over 5, 10 and 20 bp; report net CAGR/Sharpe next to gross in `_strategy_stats`.
  The README's "a few bps would visibly dent the edge" becomes a number.
- **Longer history.** `--years` defaults to 3. yfinance provides 15+ years for this
  universe, and the 80-name cross-section needs every observation it can get (see
  the statistics section below). Use 10 as the default and keep 3 as the
  "current regime" view. This also gives the bull/bear split something to split.
- **Bull/bear split.** Define regime dates by SPY vs its 200dma (already computed in
  `market.py`) and report every strategy's Sharpe and max drawdown per regime.
  FinSABER's table is the template.
- **Deflated t-stat.** Keep a counter of every configuration evaluated (tilts,
  regime weights, sleeve weights, search candidates) in a file the CLI appends to.
  Report `_ic_stats` alongside the expected maximum t under the null for that many
  trials, `sqrt(2 ln N)` as a first approximation (N = 45 gives 2.8; N = 200 gives
  3.3), and the full deflated Sharpe:

  ```
  SR0 = sqrt(Var[SR_n]) * ((1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N e)))   # g = 0.5772
  DSR = Phi((SR - SR0) * sqrt(T - 1) / sqrt(1 - skew * SR + (kurt - 1) / 4 * SR^2))
  ```

- **Probability of backtest overfitting.** Over the grid of configurations, split
  the per-date return matrix into S blocks, and for each of the C(S, S/2)
  train/test assignments pick the in-sample winner and record its out-of-sample
  rank; `PBO = share of assignments where the winner ranks below median`. Above 0.5
  means the selection process carries no information.
- **Null-pipeline audit.** Shuffle forward returns within each date and rerun the
  horizon-tilt calibration 200 times; the 95th percentile of the null's best IC is
  the hurdle the real tilts must clear.
- **Forward log.** Append each scan's recommendation (symbols, weights, JSON hash,
  timestamp) to a table before the close. After 250 trading days this is the only
  number that is immune to every bias above.

### 2. Regime model

`classify_trend_regime` pools four statistics into a scalar and thresholds it;
`REGIME_WEIGHTS` maps the label to 15 hand-set weights. The replacement:

1. Per symbol, fit a two- or three-state jump model on features already computed
   (realized vol, downside deviation, Hurst, variance ratio, efficiency ratio, 20
   and 60 day returns). Choose the jump penalty by walk-forward on the *next* fold's
   strategy Sharpe, never on label smoothness. Report mean state run length so a
   three-day "regime" is visible as a filter artifact.
2. Optionally, a shallow gradient-boosted classifier predicting next week's state
   from lagged features (Shu, Yu and Mulvey's second stage).
3. Per state, estimate sleeve weights by ridge regression of forward return on the
   five sleeve scores within that state, shrunk toward the current hand table. The
   15 hand numbers become 15 estimated numbers with a documented prior. It will
   most likely reproduce the README's finding (reversion at 5d, momentum at 63d) and
   answer whether "mixed" is a real state or a hedge.

The existing `test_regime_classification` and `test_no_lookahead` tests carry over;
state labels must only use bars up to `i`.

### 3. Exposure gating

`market_overlay` computes a `score_multiplier` of 0.85 in the risk-off state and
`app.py` attaches it to results for display only. The literature supports using it
for drawdown control: scale gross exposure in `recommend` (for example 1.0 / 0.6 /
0.3 across risk on / neutral / off) and down-weight the momentum sleeve when SPY is
below its 200dma with VIX above its 80th percentile. Backtest it with costs, because
the gate trades. Expect it to cost CAGR in the 2023-2026 sample and to earn its keep
in the regime the sample does not contain; the 10-year backtest is what makes that
visible.

### 4. Factor search with an LLM proposal engine

A new `factor_search.py`, about 300 lines:

- **DSL.** Operands are the causal functions in `quant_indicators.py` with integer
  windows from a fixed grid; operators are `rank`, `ts_mean`, `ts_std`, `ts_delta`,
  `sub`, `div`, `mul`, `neg`, `clip`, `zscore`; maximum tree depth 4. Parse with
  `ast.parse`, reject anything outside the whitelist, deduplicate canonical ASTs.
- **Prompt loop.** The model receives the DSL, the sleeve docstrings from
  `quant_engine.py`, and a table of previously scored expressions (expression, mean
  IC, t, first-half vs second-half IC, max correlation to the five sleeves). It
  returns K new `{thesis, expression}` pairs. It never sees prices, dates or tickers,
  which closes the Profit Mirage channel. Keep the thesis so a second cheap call can
  reject expressions whose logic does not match it (AlphaAgent's alignment check).
- **Scoring.** `score_expression(analyzers, expr, period, step, years)` evaluates the
  expression at the same rebalance positions `run_backtest` uses, computes per-date
  Spearman IC against the same `fwd` dict, and returns `_ic_stats` plus the half
  split and the sleeve correlations. Because every indicator is already rolling and
  causal, no-lookahead is inherited; the only new risk is a window longer than
  `MIN_HISTORY`.
- **Splits, frozen before the first proposal.** Discovery 2010-2018, validation
  2019-2021, sealed holdout 2022-2024, and the 2025-2026 window opened exactly once.
- **Gate.** On discovery: t >= 2.5, second-half IC at least half the first-half IC
  with the same sign, max |rho| to any sleeve <= 0.6, at least 55% positive IC
  dates. Survivors are scored once on validation and kept if t >= 1.5 with the same
  sign. Report the deflated t using N = every expression tried. Cap accepted
  changes at two per quarter so the forward log can attribute drift.
- **Hardening.** The agent writes only to a candidates file; the backtester, data
  and gate run in a separate process with an integrity hash and return scalars.
- **Outcome.** Expect most runs to keep zero to two factors. Anything resembling the
  0.05-0.15 ICs in the papers means the split leaked. Cost: 300 proposals at a few
  thousand tokens each is single-digit dollars; the bottleneck is `run_backtest`.

### 5. Challenger ranker

Target: forward 21d or 63d return, cross-sectionally demeaned (the horizons where
momentum IC is significant). Model: ridge on the five sleeve scores first; then
LightGBM with at most 200 trees, depth 3-4, `min_child_samples >= 200` over the
sleeve inputs, regime statistics and overlay. Expanding walk-forward, monthly refit,
purge the last horizon of training rows, embargo five days. Compare rank-IC and its
t-stat against the hand composite on identical dates. With monthly IC noise of about
0.11 at N = 80 you need 36+ test months to detect IC 0.05; the likely answer is "not
distinguishable from the hand composite," which still bounds how much weight tuning
is worth. If ridge ties LightGBM, stop there.

### 6. Morning brief

A `/api/brief` endpoint in `app.py` that calls Claude once per scan:

- Input is pre-digested JSON with an ID on every number: composite and sleeve
  contributions with sign, regime label, rank deltas, recommendation diff, overlay.
  No prices, no history, no tickers the model could recognize beyond what the user
  already sees.
- Structured output: a list of `{sentence, cited_ids, claim_type}` where
  `claim_type` is one of `data`, `reasoning`, `external`. Drop `external`
  sentences. Extract every numeral from each sentence and reject the sentence if the
  number is not in the JSON (FinGround without the retrieval).
- Direction check: polarity words in a sentence must agree with the sign of the
  sleeve it cites; fail closed. This is the error Zandi et al. found most common.
- No forecast field exists in the schema; the system prompt states the model
  narrates a deterministic model. Log the input and output hashes.
- Test it by injecting sign flips and deleted fields and measuring whether the brief
  changes; track the confabulation rate over time.

No paper shows a narrative improves a retail user's decisions. The value is
legibility, and since the README's own analysis says the edge is portfolio
construction, the brief should talk about the portfolio and the regime, not the
picks.

### 7. Rebalance critic

The checks run in Python inside `recommend`: HHI and maximum weight, correlation
clusters among the selected names (flag clusters of more than three names with
60-day rho > 0.7 - the "correlation blind spot"), earnings dates inside the holding
horizon from the fundamentals cache, regime label vs the regime the weights were
estimated on, turnover and estimated dollar cost. The model receives the check table
and writes a ranked list of concerns citing check IDs, with a single `blocking`
boolean that may only be true when a deterministic check failed. It can never relax
a limit (TradeTrap). Measure it by logging flags and, 20 trading days later,
comparing flagged vs unflagged names. The honest claim is "catches mechanical
violations and explains them," not "improves returns."

### 8. News sleeve, weight zero

- Nightly job: pull Finnhub and yfinance news for the watchlist; deduplicate on
  normalized title and URL; store `news_items(symbol, published_at_utc, source,
  title, url, first_seen_at)` in `data_store.py`.
- Scorer: replace tickers and names with a placeholder, run FinBERT or a ChronoGPT
  checkpoint locally, write `news_scores(symbol, asof_date, model, n_items,
  mean_score, neg_share, max_abs, extreme_event_flag, computed_at)`. Never
  overwrite; every row is point-in-time.
- Features: decayed mean sentiment (half-life 1-3 days), item-count shock vs the
  60-day median, a negative-news flag. The flag is the one most likely to interact
  with the 2-sigma oversold event study - Lopez-Lira and Tang find negative news
  drifts longest - so the first test is whether oversold events with a negative-news
  flag behave differently from those without.
- Sixth sleeve on the dashboard, weight zero in the composite, with a rolling 63-day
  rank IC and hit rate at 1d and 1w. Promotion rule written into the repo before
  data exists: at least six months of scores, mean 1d rank IC above 0.02 with
  t > 2, positive in four of six months. Treat IC above 0.05 in the first six months
  as a bug.
- One labelled secondary test on the Finnhub 12-month backfill, scored with a model
  whose cutoff predates it, reported separately.

### 9. Foundation model, one afternoon

Kronos-small (25M) and chronos-bolt run on CPU. Per symbol, forecast 10 days ahead,
rank cross-sectionally, compute rank IC on the same dates as the composite through
`score_expression`. Expect 0.00-0.02. Do not fine-tune; 60k samples is exactly
where Rahimikia et al. find fine-tuning fails. Gate it like any other candidate.

## What was built

Items 1, 2, 3, 4, 6 and 7 are implemented. Items 5, 8 and 9 are not; the reasons are
at the end of this section. The test suite went from 40 tests to 146.

| module | what it is | status |
| --- | --- | --- |
| [evaluation.py](../evaluation.py) | deflated Sharpe, PBO/CSCV, cost model, regime split, null audit, trial ledger | new |
| [regime.py](../regime.py) | statistical jump model, causal walk-forward labelling, shrunk per-state weights | new |
| [regime_calibrate.py](../regime_calibrate.py) | offline experiment: estimate weights on discovery, test on validation, install only if they win | new |
| [factor_search.py](../factor_search.py) | whitelisted expression DSL, LLM proposal loop, frozen splits, acceptance gate | new |
| [llm.py](../llm.py) | Anthropic wrapper; every feature degrades to a deterministic half without it | new |
| [checks.py](../checks.py) | seven deterministic portfolio checks plus an LLM critic that cannot change them | new |
| [brief.py](../brief.py) | grounded narration with per-sentence verification | new |
| [backfill.py](../backfill.py) | pulls fifteen years of daily bars for the whole watchlist | new |
| [backtest.py](../backtest.py) | costs, bull/bear split, diagnostics, 10-year default, trial logging | changed |
| [data_store.py](../data_store.py) | `HISTORY_PERIOD`, so a full refresh stops truncating to five years | changed |
| [market.py](../market.py) | `classify_risk` / `exposure_scalar` split out for point-in-time use | changed |
| [quant_engine.py](../quant_engine.py) | `ACTIVE_WEIGHTS` + `load_calibrated_weights()` | changed |
| [recommender.py](../recommender.py) | exposure gating, cash weight, gated rebalance plan | changed |
| [app.py](../app.py) | `/api/brief`, `/api/review`, market passed to the recommender | changed |

### What the honesty checks did to the headline result

The first pass ran on the five years of daily bars the cache then held, and concluded
the composite's selection edge had evaporated. That conclusion was wrong, and wrong in
an instructive way: the window was the artifact, not the edge. `backfill.py` took the
cache to fifteen years and `data_store.HISTORY_PERIOD` now sets that as a floor and a
ceiling both, since the weekly full refresh replaces rows wholesale and a shorter
setting would silently truncate.

Over 83 symbols from 2016-08-11 to 2026-08-07, 503 rebalances, 1m scores, 5-day
rebalance, top 20%:

```
strategy           CAGR%  Sharpe   MaxDD%   Hit%  bull SR  bear SR
composite           20.0     1.2   -25.31   61.4     1.34     0.66
trend               24.3    1.29   -29.76   62.6     1.48     0.52
momentum            21.3    1.24   -29.18   62.6     1.38     0.68
mean_reversion     20.76    0.96   -37.87   59.6     0.97     1.22
quality             15.3    1.11   -22.75   64.2     1.41     0.35
equal_weight       19.05    1.12   -29.51   63.4     1.16     1.28
spy                15.38    0.96   -29.07   63.6     0.98     1.13
```

Five findings, in descending order of how much they should change behaviour:

1. **The selection edge is real over ten years.** $10,000 became $61,680 in the
   composite against $56,980 equal-weighting the watchlist and $41,680 in SPY: 47
   points over the watchlist, 200 over SPY, with the second shallowest drawdown of
   anything tested (-25.31%). Over the shallower cache these two tied exactly.
2. **The bear market is the weak point, and the short window hid it.** With 2018,
   2020 and 2022 in sample, the composite runs Sharpe 1.34 in bull regimes and
   **0.66** in bear ones (11.5% CAGR over 81 bear rebalances against 21.7% over 422
   bull). Mean reversion is the only sleeve that prefers bear regimes (1.22 vs 0.97).
   This is the strongest argument for the exposure gate, and it is exactly the
   FinSABER finding - strategies that look fine in a rising tape and fall apart in a
   falling one - showing up in a small live system.
3. **Costs take roughly 40% of the CAGR.** 20.0% gross, 17.83% at 5 bps, 15.71% at 10,
   11.56% at 20.
4. **The ranking now clears the shuffle bar but not the multiple-testing bar.** The
   permutation audit puts the best-of-six-sleeves IC on signal-free data at 0.0052
   mean and 0.0112 at the 95th percentile over 200 permutations; the composite's
   measured 0.016 is above it. But its t-stat is 1.44 against a best-of-55 hurdle of
   2.83. Note what moved: the same 0.016 IC that sat *inside* the noise on the
   shallower cache is now outside it, because 503 rebalances tightened the null. The
   signal did not improve; the measurement did.
5. **PBO fell from 0.67 to 0.48**, just below the line where the procedure that picks
   a best sleeve would carry no information at all.

The lesson for this project is the one the report argued for on statistical grounds
before any of it was run: at N=80 the binding constraint is the number of
rebalances, and a download was worth more than any modelling change attempted here.

### The regime experiment: better ranking, worse portfolio

On the shallow cache the calibration failed outright - it looked spectacular on
discovery (IC 0.047 vs 0.007) and came out worse on validation (-0.088 vs -0.056),
textbook in-sample fitting, and the gate refused it.

With fifteen years and three years held out it is a subtler failure:

| table | split | mean IC | IC t | CAGR% | Sharpe | MaxDD% |
| --- | --- | --- | --- | --- | --- | --- |
| prior | discovery | 0.0073 | 0.36 | 17.07 | 1.23 | -21.27 |
| calibrated | discovery | 0.0148 | 0.71 | 17.31 | 1.20 | -27.25 |
| prior | validation | 0.0090 | 0.25 | 30.55 | **1.73** | **-12.45** |
| calibrated | validation | **0.0174** | 0.47 | 25.78 | 1.57 | -15.48 |

The calibrated table ranks names *better* out of sample - nearly double the IC - and
builds a *worse* portfolio: lower CAGR, lower Sharpe, deeper drawdown. That exposed a
flaw in the original gate, which tested mean IC alone and would have installed it.

The gate now requires better ranking **and** no material loss of validation Sharpe
(`SHARPE_TOLERANCE = 0.95`). Given that this system's measured value is portfolio
construction rather than ranking, a table that trades Sharpe for IC is not an
improvement to the part that works. With the tightened rule `--apply` is a no-op and
the hand-set prior stays in force - the second time it has survived a serious attempt
to replace it.

The jump model itself behaves: on AAPL it produces states with 14-24 day mean run
lengths rather than the three-day artifacts a Gaussian HMM fits to noise, and
`test_labels_are_causal` asserts that the label at bar *i* is unchanged when later
bars are appended.

### The factor search runs, and the validation split earns its keep

The frozen splits were re-set once when the cache deepened - discovery to 2021-07-29
(~8.7 years after the warm-up), validation to 2024-05-10, holdout after. Re-freezing
because the dataset changed is legitimate; re-freezing after seeing which side of a
boundary a favoured expression falls on is not, and the expressions scored under the
old splits are archived as `factor_candidates_5y_cache.jsonl` rather than carried
over, because their statistics were measured on different data.

On 104 discovery rebalances, 20 randomly sampled expressions produced this:

- four cleared the discovery gate at |t| between 2.71 and 3.57
- **all four then failed validation**, at |t| between 0.53 and 0.60
- two more were caught by the decay filter, holding under half their first-half IC
- zero survivors

That is the two-stage gate doing precisely what it exists for. Noise expressions
*will* reach |t| > 3 on a hundred rebalances - that is what the best-of-N hurdle, now
2.83 after 55 scored expressions, is measuring - and the only thing that reliably
kills them is a split they have never seen. Anyone reading a single-split t-statistic
of 3.5 as a discovery should look at this table first.

The DSL is a whitelist over the AST, not a blacklist. Imports, attribute access,
subscripts, lambdas, comprehensions, exponentiation, keyword arguments and any
unlisted name are rejected before evaluation, with a test for each. Expression
evaluation inherits no-lookahead because every operand in `quant_indicators.py` is
already rolling; `test_evaluate_is_causal` pins it. Without credentials the search
samples its own grammar, which doubles as the control: if LLM proposals are not
measurably better than random ones, the call is not paying for itself.

### The brief and the critic

`/api/brief` builds a fact table where every number carries an id, asks the model for
sentences that cite ids, then verifies each one before returning it. A sentence is
dropped if it claims outside knowledge, cites nothing, cites an unknown id, contains a
number not present in the facts it cited, or uses direction language that contradicts
the sign of the sleeve it cites. That last check is aimed at the specific failure
Zandi et al. found most often. `confabulation_rate` is returned alongside the text, so
the failure rate is measured rather than assumed.

The number check needed one refinement that only showed up in testing: digits glued to
letters are part of a term, not a claim. "SPY is below its 200dma" must not be
rejected because 200 is not in the facts, so the regex requires a number to stand
alone.

`/api/review` computes seven checks in Python — concentration and HHI, correlation
clusters, earnings inside the horizon, regime mix, turnover and its dollar cost,
exposure, and score separation — and lets the model rank and explain them. It cannot
compute a check, relax a threshold, or set `blocking`; `review()` overwrites that
field from the deterministic results after the model has spoken, and
`test_llm_cannot_change_blocking` asserts it. Concerns citing an id that does not
exist are dropped.

On the live watchlist the correlation check immediately earned its place, flagging
EFA/SPY/XLI as a three-name cluster above rho 0.7 — all below the recommender's 0.92
pairwise dedup threshold, and therefore invisible to it. That is precisely the
"correlation blind spot" Xue names.

### Exposure gating

`market.classify_risk` and `market.exposure_scalar` were split out so the backtester
can evaluate the gate at a historical bar with point-in-time values, using a pad
indexer on both SPY and the VIX so no bar after the decision is visible. Exposure is
1.0 / 0.6 / 0.3 across risk on / neutral / off. The recommender scales dollar targets
and the rebalance plan by it and reports `cash_weight` with a note saying plainly that
this is drawdown control and will cost return in a rising tape. `backtest.py --gate`
measures it.

### What was not built, and why

- **Item 5, the LightGBM challenger.** LightGBM and scikit-learn are not installed,
  and the project's four-dependency discipline is worth more than a challenger whose
  expected result is "not distinguishable from the hand composite". The ridge baseline
  the report recommends first is a dozen lines of numpy when it is wanted, and the
  honest prerequisite is more history, not more model.
- **Item 8, the news sleeve.** It needs an API key, a nightly job and six months of
  forward collection before its sleeve can be scored. Nothing about it can be
  validated today, and the expected standalone IC for 80 large caps is zero. The
  design is specified above; starting the collection is the first step whenever it is
  wanted.
- **Item 9, Kronos/Chronos zero-shot.** A ~100MB model download for an experiment
  whose literature-implied answer is IC 0.00-0.02. `factor_search.score_expression`
  is the right place to hang it when curiosity wins.

### Running it

```bash
python backfill.py                                  # fifteen years of daily bars
python backtest.py --years 10 --null-audit 200      # honest headline numbers
python backtest.py --cost-bps 10 --gate             # net of costs, regime-gated
python regime_calibrate.py --labeler jump --apply   # calibrate; installs only if it wins
python factor_search.py --propose 20                # LLM proposals, gated
python factor_search.py --propose 20 --random       # the control, no API key
python -m pytest tests                              # 146 tests
```

`results/trials.jsonl` is shared across all of these on purpose. Every configuration
scored against this dataset counts toward the multiple-testing hurdle, whether it came
from a backtest sweep, a regime calibration or a factor proposal. Deleting it to make
a result look better is the one thing that would invalidate everything above.

## The 80-name problem

With N = 80 names per date, the null standard error of a per-date Spearman IC is
about 1 / sqrt(79) = 0.11, versus 0.045 for a 500-name universe. That is why the
weekly composite IC of 0.016 has t = 0.86 over 151 rebalances, and why the quarterly
momentum IC of 0.12 rests on 11 observations. To reach the Harvey-Liu-Zhu t > 3 on a
true IC of 0.02 with a per-date standard deviation of 0.12 takes about
(3 x 0.12 / 0.02)^2 = 320 non-overlapping periods - six-plus years of weekly data.
Three things follow:

1. Run the backtest on the full available history by default, not three years.
2. Score ETFs and single stocks as separate cross-sections; ETFs have lower
   idiosyncratic variance and cluster in the middle deciles, which dilutes every IC
   measured on the mixed list.
3. Add a pooled panel test (date fixed effects, clustered standard errors) and a
   per-asset time-series IC beside the cross-sectional IC, because at this N every
   observation is needed.

None of the published LLM-mining recipes were run on a universe this small, and the
one paper that used an 89-name index did not report it separately. Treat every
technique above as an experiment whose most likely honest outcome is "no detectable
improvement," and design the gates so that outcome is what gets reported.

## Paper index

### LLM trading agents and benchmarks
- TradingAgents, Xiao et al. 2024/25 - https://arxiv.org/abs/2412.20138
- FinMem, Yu et al. 2023 - https://arxiv.org/abs/2311.13743
- FinAgent, Zhang et al., KDD 2024 - https://arxiv.org/abs/2402.18485
- FinCon, Yu et al., NeurIPS 2024 - https://arxiv.org/abs/2407.06567
- HedgeAgents, Li et al., WWW 2025 - https://arxiv.org/abs/2502.13165
- FinRobot, AI4Finance 2024 - https://arxiv.org/abs/2405.14767
- QuantAgent, Wang et al. 2024 - https://arxiv.org/abs/2402.03755
- MarketSenseAI 2.0, Fatouros et al. 2025 - https://arxiv.org/abs/2502.00415; Signal or Noise 2026 - https://arxiv.org/abs/2604.17327
- Trading-R1, Xiao et al. 2025 - https://arxiv.org/abs/2509.11420
- FLAG-Trader, Xiong et al., ACL Findings 2025 - https://arxiv.org/abs/2502.11433
- FinRL-DeepSeek, Benhenda 2025 - https://arxiv.org/abs/2502.07393
- FinSABER, Li, Kim, Cucuringu, Ma, KDD 2026 - https://arxiv.org/abs/2505.07078
- Profit Mirage, Li et al. 2025 - https://arxiv.org/abs/2510.07920
- StockBench, Chen et al. 2025/26 - https://arxiv.org/abs/2510.02209
- LiveTradeBench, Yu, Li, You 2025 - https://arxiv.org/abs/2511.03628
- Agent Market Arena, Qian et al. 2025 - https://arxiv.org/abs/2510.11695
- AI-Trader, Fan et al. 2025 - https://arxiv.org/abs/2512.10971
- InvestorBench, Li et al. 2024 - https://arxiv.org/abs/2412.18174
- KTD-Fin, Zhu et al. 2026 - https://arxiv.org/abs/2605.28359
- The Alpha Illusion, Ye et al. 2026 - https://arxiv.org/abs/2605.16895
- TradingAgents reproducibility, ICAIF 2026 - https://dl.acm.org/doi/abs/10.1145/3800973.3801029
- TradeTrap, Yan et al. 2025 - https://arxiv.org/abs/2512.02261
- CLQT, Qu and Chen 2026 - https://arxiv.org/abs/2606.29771
- Agentic Trading audit, Xia et al. 2026 - https://arxiv.org/abs/2605.19337
- Toward Reliable Evaluation of Financial MAS, Nguyen and Pham 2026 - https://arxiv.org/abs/2603.27539

### Lookahead bias and contamination
- Glasserman and Lin 2023 - https://arxiv.org/abs/2309.17322
- Sarkar and Vafa 2024 - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4754678
- Levy, Caution Ahead, JAR 2026 - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5082861
- Didisheim, Fraschini and Somoza, Econ Letters 2025 - https://www.sciencedirect.com/science/article/pii/S0165176525004392
- Gao, Jiang and Yan, Lookahead Propensity 2025 - https://arxiv.org/abs/2512.23847
- DatedGPT, Yan et al. 2026 - https://arxiv.org/abs/2603.11838
- ChronoBERT/ChronoGPT, He, Lv, Manela, Wu 2025 - https://arxiv.org/abs/2502.21206
- MemGuard-Alpha, Roy and Roy 2026 - https://arxiv.org/abs/2603.26797
- FinCAD, Li, Wang, Ma 2026 - https://arxiv.org/abs/2605.24564
- Look-Ahead-Bench, Benhenda 2026 - https://arxiv.org/abs/2601.13770
- Kong et al., Explicit Bias Consideration 2026 - https://arxiv.org/abs/2602.14233
- Fonseca, Temporal Non-Interference 2026 - https://arxiv.org/abs/2607.04958

### LLM alpha mining
- AlphaGen, Yu et al., KDD 2023 - https://arxiv.org/abs/2306.12964
- AlphaQCM, Zhu and Zhu, ICML 2025 - https://proceedings.mlr.press/v267/zhu25ag.html
- AlphaForge, Shi et al., AAAI 2025 - https://arxiv.org/abs/2406.18394
- Alpha-GPT, Wang et al. 2023 - https://arxiv.org/abs/2308.00016; Alpha-GPT 2.0 - https://arxiv.org/abs/2402.09746
- AlphaAgent, Tang et al., KDD 2025 - https://arxiv.org/abs/2502.16789
- RD-Agent(Q), Li et al., NeurIPS 2025 - https://arxiv.org/abs/2505.15155
- LLM-guided MCTS alpha search, Shi, Duan, Li, AAAI 2026 - https://arxiv.org/abs/2505.11122
- CogAlpha, Liu et al., ACL 2026 - https://arxiv.org/abs/2511.18850
- HARLA, Yu et al., FCS 2026 - https://link.springer.com/article/10.1007/s11704-025-41061-5
- QuantaAlpha, Han et al. 2026 - https://arxiv.org/abs/2602.07085
- AlphaMemo, Yu et al. 2026 - https://arxiv.org/abs/2606.20625
- AlphaEval, Ding et al., KDD 2026 - https://arxiv.org/abs/2508.13174
- AlphaForgeBench, Zhang et al. 2026 - https://arxiv.org/abs/2602.18481
- Alpha-R1, Jiang et al. 2025 - https://arxiv.org/abs/2512.23515
- MadEvolve, Kvasiuk et al. 2026 - https://arxiv.org/abs/2605.23007
- Beyond Prompting, Huang and Fan 2026 - https://arxiv.org/abs/2603.14288
- Reward Hacking Benchmark, Thaman 2026 - https://arxiv.org/abs/2605.02964; METR 2025 - https://metr.org/blog/2025-06-05-recent-reward-hacking/

### Text and news signals
- Lopez-Lira and Tang, JFE 2026 - https://arxiv.org/abs/2304.07619
- Chen, Kelly and Xiu - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4416687
- Kirtac and Germano, FRL 2024 - https://arxiv.org/abs/2412.19245
- Guo and Hauptmann 2024 - https://arxiv.org/abs/2407.18103
- Kim, Muhn, Nikolaev (withdrawn) - https://arxiv.org/abs/2407.17866
- From Text to Alpha, Choi et al. 2025 - https://arxiv.org/abs/2510.03195
- Yilki, supply-chain propagation 2026 - https://arxiv.org/abs/2606.29290
- FinCall-Surprise, ACL 2026 - https://arxiv.org/abs/2510.03965
- Yang, When Valid Signals Fail 2026 - https://arxiv.org/abs/2604.10996
- FinMTEB, Tang and Yang, EMNLP 2025 - https://arxiv.org/abs/2502.10990
- FinBERT, Araci 2019 - https://arxiv.org/abs/1908.10063
- Hansen and Kazinnik, Fedspeak - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4399406

### RL, foundation models, cross-sectional ML
- Kashif and Slepaczuk 2026 - https://arxiv.org/abs/2605.17307
- GIFT, Wu et al. 2026 - https://arxiv.org/abs/2606.08450
- MetaTrader bilevel RL, Yuan et al. 2025 - https://arxiv.org/abs/2505.12759
- Tan et al., NeurIPS 2024 - https://arxiv.org/abs/2406.16964
- Rahimikia, Ni, Wang 2025 - https://arxiv.org/abs/2511.18578
- Noguer i Alonso and Franklin 2026 - https://arxiv.org/abs/2606.27100
- Kronos, Shi et al., AAAI 2026 - https://arxiv.org/abs/2508.02739
- Chronos-2 - https://arxiv.org/abs/2510.15821
- MASTER, Li et al., AAAI 2024 - https://arxiv.org/abs/2312.15235
- FinTSB, Hu et al. 2026 - https://arxiv.org/abs/2502.18834
- Qlib benchmarks - https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md
- Gu, Kelly, Xiu, RFS 2020 - https://academic.oup.com/rfs/article/33/5/2223/5758276
- Avramov, Cheng, Metzker, Mgmt Sci 2023 - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3450322
- Azevedo, Hoegner, Velikov - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4702406
- Nagel, Seemingly Virtuous Complexity, NBER 2025 - https://www.nber.org/papers/w34104
- Fallahgoul 2025 - https://arxiv.org/abs/2506.03780
- Capponi et al., Nonstationarity-Complexity 2025/26 - https://arxiv.org/abs/2512.23596

### Regime and risk
- Shu, Yu, Mulvey, jump models 2024 - https://arxiv.org/abs/2402.05272 (code: https://github.com/Yizhan-Oliver-Shu/jump-models)
- Shu, Yu, Mulvey, asset-specific regime forecasts 2024 - https://arxiv.org/abs/2406.09578
- Shu and Mulvey, dynamic factor allocation 2024 - https://arxiv.org/abs/2410.14841
- Daniel and Moskowitz, JFE 2016 - https://www.sciencedirect.com/science/article/pii/S0304405X16301490
- Goulding, Harvey, Mazzoleni, JFE 2023 - https://www.sciencedirect.com/science/article/abs/pii/S0304405X23001034
- Suominen and Hjalmarsson, FM 2026 - https://onlinelibrary.wiley.com/doi/full/10.1111/fima.70055
- Moreira and Muir, JF 2017 - https://onlinelibrary.wiley.com/doi/10.1111/jofi.12513
- Cederburg et al., JFE 2020 - https://www.sciencedirect.com/science/article/abs/pii/S0304405X2030132X
- DeMiguel, Martin-Utrera, Uppal, JF 2024 - https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13395
- Yi et al., LLM regime shift detection 2026 - https://arxiv.org/abs/2605.30363

### Explanation, critique, evaluation methodology
- Geng et al., LLM post-hoc explainers 2026 - https://arxiv.org/abs/2602.18895
- Zandi et al., communicating credit risk 2026 - https://arxiv.org/abs/2608.17715
- Matton et al., Walk the Talk, ICLR 2025 - https://arxiv.org/abs/2504.14150
- FinGround, ACL 2026 - https://arxiv.org/abs/2604.23588
- AI Analyst, Fons et al. 2025 - https://arxiv.org/abs/2507.00718
- Xue, risk-feedback alignment 2026 - https://arxiv.org/abs/2605.28850
- PortBench, Zhao, Chen, Su 2026 - https://arxiv.org/abs/2605.27887
- OpenPM, Cai et al. 2026 - https://arxiv.org/abs/2608.09988
- Bailey and Lopez de Prado, Deflated Sharpe - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- Bailey et al., Probability of Backtest Overfitting - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- Bailey et al., Pseudo-Mathematics, AMS 2014 - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659
- Harvey, Liu, Zhu, RFS 2016 - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2249314
- Harvey and Liu, JF 2020 - https://arxiv.org/abs/2006.04269
- Arian, Norouzi, Seco, KBS 2024 - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4778909
- Nikolopoulos, Spurious Predictability 2026 - https://arxiv.org/abs/2604.15531
- McLean and Pontiff, JF 2016 - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2156623
- Jensen, Kelly, Pedersen, JF 2023 - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3774514
