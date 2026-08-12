/**
 * Plain-language explanations for every metric on the dashboard, plus small
 * helpers to render "?" tooltips. Keep these readable for someone who has
 * never traded: say what the number means and how to react to it.
 * Texts must not contain double quotes (they land in HTML attributes).
 */

const METRIC_HELP = {
    score: 'Composite score from -100 (bearish) to +100 (bullish). A weighted blend of the five factor sleeves: trend, momentum, mean reversion, volume flow and quality. The weights adapt to the stock’s regime.',
    confidence: 'How much the five factor sleeves agree with each other. 80% means most evidence points the same way; near 50% means the sleeves disagree (e.g. strong trend but overbought), so trust the score less.',
    rank: 'Decile of the composite score within your watchlist: 1 = top 10% of names scanned, 10 = bottom 10%. BUY/SELL labels come from this rank, so they are relative to the rest of the list, not absolute predictions.',
    regime: 'How this stock has been trading. Trending: moves tend to continue, so trend and momentum signals get more weight. Mean-reverting: moves tend to snap back, so oversold/overbought signals get more weight. Based on the Hurst exponent, variance ratio, ADX and efficiency ratio.',
    profile: 'Fundamentals for context (not part of the score): sector, PE = price versus one year of earnings (how expensive it is), ROE = profit per dollar of shareholder equity (profitability quality), and the next earnings date - a scheduled volatility event.',
    risk: 'Risk statistics for holding this name. They describe consequences, not direction, so they inform position size rather than the buy/sell score.',
    sharpe: 'Sharpe ratio: return earned per unit of risk taken. Below 0 = losing, around 1 = good, above 2 = excellent.',
    maxdd: 'Maximum drawdown: the worst peak-to-trough fall over the window. -18% means at the worst moment a holder was down 18% from the high. The how-much-pain number.',
    beta: 'Beta: sensitivity to the S&P 500. 1.0 moves with the market, 2.0 swings twice as hard both ways, 0.5 half as hard, negative tends to move opposite the market.',
    annVol: 'Annualized volatility: the typical size of a one-standard-deviation year. 16% vol means a plus-or-minus 16% year is ordinary.',
    varCvar: 'VaR 95%: on 19 of 20 days your daily loss should be smaller than this - a threshold, not a worst case. CVaR: the average loss on the worst 5% of days, i.e. what a genuinely bad day actually costs. CVaR is always at least as bad as VaR; a wide gap means fat tails.',
    effectivePositions: 'How many truly independent positions this portfolio behaves like (1 divided by the concentration index). Four equal holdings = 4.0; four holdings dominated by one = closer to 2. Low numbers mean concentration risk.',
    topHolding: 'Largest position as a share of portfolio value. A quick concentration check.',
    avgCorrelation: 'Average pairwise correlation: how much your holdings move together, from -1 to +1. Near +0.8 your positions are effectively one bet wearing several tickers; near 0 they genuinely diversify each other.',
    correlatedPair: 'The two holdings that move together the most. A high value flags redundancy - overlapping funds (e.g. QQQ and VOO) or same-sector names.',
    marketRisk: 'Market-wide weather, not specific to any stock: SPY versus its 200-day average (is the broad market in an uptrend) plus the VIX level. Risk-off means be defensive regardless of individual scores. This banner does not change any ranks.',
    vix: 'VIX: the market’s expected volatility over the next 30 days, implied by S&P 500 option prices - the fear gauge. The percentile shows where today sits versus the last 5 years.',
    movers: 'Names whose rank jumped 3 or more places since the previous scan - where the model changed its mind fastest.',
};

// Matched by substring against each signal string.
const SIGNAL_HELP = [
    ['squeeze fired', 'The squeeze released: after a stretch of unusually compressed volatility, price is starting to move. Direction is taken from momentum at the moment of release.'],
    ['squeeze on', 'Bollinger bands are trading inside the Keltner channels: volatility is unusually compressed. Like a coiled spring, this often precedes a sharp move - direction unknown until it fires.'],
    ['Donchian breakout', 'Price closed above its highest high of the last 20 days - the classic trend-following entry. Caveat from our own backtest: in this large-cap universe, buying 20-day breakouts has returned less than average.'],
    ['Donchian breakdown', 'Price closed below its lowest low of the last 20 days. Trend-followers treat this as an exit or short trigger; in our backtest these were often washed-out lows that bounced.'],
    ['sigma below mean', 'Price is stretched unusually far below its recent average (measured in standard deviations). Statistically the strongest signal we tested: these stretches tended to snap back over the following days.'],
    ['sigma above mean', 'Price is stretched unusually far above its recent average. Extended names tended to pause or pull back over the following week in our tests.'],
    ['oversold', 'RSI is low: recent losses have dominated recent gains. Contrarian bullish - beaten-down names tend to bounce, but check the trend regime before catching knives.'],
    ['overbought', 'RSI is high: recent gains have dominated. The move may need to cool off; contrarian bearish in choppy regimes, less meaningful in strong trends.'],
    ['52-week high', 'Proximity to the 52-week high. Counterintuitively bullish: names at or near their highs have historically kept performing (the momentum anomaly), because holders have no losses to sell into.'],
    ['MACD histogram', 'The gap between MACD and its signal line just changed sign - short-term momentum flipped direction.'],
    ['half-life', 'How fast this stock historically snaps back to its own average. A short half-life (days) means stretches away from the mean have been quick round trips - good conditions for fading extremes.'],
    ['Persistent trend', 'The Hurst exponent is well above 0.5: moves in this name have tended to continue rather than reverse. Favor riding momentum over fading it.'],
    ['High-vol regime', 'Volatility is in the top of its own 1-year range. Whatever your conviction, swings will be larger - standard practice is to reduce position size.'],
    ['Low-vol regime', 'Volatility is unusually quiet for this name. Calm often precedes storms; options are relatively cheap.'],
    ['High beta', 'This name amplifies market moves. It will likely fall harder than the index in a selloff.'],
    ['Earnings in', 'A scheduled earnings report is imminent. Earnings gaps can overwhelm any technical setup - expect a volatility event.'],
    ['Rich valuation', 'Priced expensively versus the rest of the watchlist on earnings. Not part of the score, but stretched valuations add risk if sentiment turns.'],
    ['Cheap valuation', 'Priced cheaply versus the rest of the watchlist on earnings. Can be an opportunity or a warning - the market may know something.'],
    ['High ROE', 'The business earns strong profits on its equity - a quality marker independent of the chart.'],
];

function signalHelp(text) {
    const hit = SIGNAL_HELP.find(([key]) => text.includes(key));
    return hit ? hit[1] : '';
}

function helpIcon(key) {
    const text = METRIC_HELP[key];
    if (!text) return '';
    return `<span class="help-tip" data-bs-toggle="tooltip" data-bs-placement="top" title="${text}">?</span>`;
}

/**
 * Activate Bootstrap tooltips on freshly rendered content.
 */
function initTooltips(root) {
    if (typeof bootstrap === 'undefined' || !bootstrap.Tooltip) return;
    (root || document).querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => {
        if (!bootstrap.Tooltip.getInstance(el)) {
            new bootstrap.Tooltip(el, { container: 'body', delay: { show: 150, hide: 0 } });
        }
    });
}
