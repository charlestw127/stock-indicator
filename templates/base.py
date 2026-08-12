"""
Base HTML template for the stock trading application with portfolio management.
"""

BASE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Trading Recommendation System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    <div class="header">
        <h1>Stock Trading Recommendation System</h1>
    </div>
    
    <div class="container">
        <!-- Configuration Row -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <span>Configuration</span>
                        <button class="btn btn-sm btn-outline-secondary" id="edit-config-button">Edit</button>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h5>Settings</h5>
                                <div class="mb-2"><strong>Auto-refresh:</strong> <span id="refresh-interval-display">30 minutes</span></div>
                                <div class="mb-2"><strong>Hide non-buys:</strong> <span id="hide-non-buys-display">Yes</span></div>
                                <div class="mb-2"><strong>Hide ranks above:</strong> <span id="hide-ranks-display">7</span></div>
                                <div class="d-flex">
                                    <div class="refresh-status me-3">Auto-refreshing every 30 minutes</div>
                                    <div id="next-refresh" class="refresh-status"></div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="d-grid gap-2">
                                    <button type="button" class="btn btn-primary" id="manual-analyze-button">Run Analysis Now</button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Portfolio Row -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <span>My Portfolio</span>
                        <div>
                            <button class="btn btn-sm btn-outline-primary me-2" id="edit-portfolio-button">Edit Portfolio</button>
                            <button class="btn btn-sm btn-outline-success" id="analyze-portfolio-button">Analyze Portfolio</button>
                        </div>
                    </div>
                    <div class="card-body" id="portfolio-area">
                        <p class="text-center text-muted">
                            Portfolio analysis will appear here.
                        </p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Recommended Portfolio Row -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <span>Recommended Portfolio <span id="rec-asof" class="text-muted small fw-normal"></span></span>
                        <button class="btn btn-sm btn-outline-primary" id="refresh-recommendation-button">Refresh</button>
                    </div>
                    <div class="card-body" id="recommendation-area">
                        <p class="text-center text-muted">
                            The recommended portfolio will appear after the first scan.
                        </p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Dashboard Guide Row -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <span>How to Read This Dashboard</span>
                        <button class="btn btn-sm btn-outline-secondary" type="button"
                                data-bs-toggle="collapse" data-bs-target="#dashboard-guide">
                            Show / Hide
                        </button>
                    </div>
                    <div class="card-body collapse" id="dashboard-guide">
                        <div class="row">
                            <div class="col-md-6">
                                <h6>The score and the rank</h6>
                                <p class="small">
                                    Every stock gets a <strong>composite score</strong> from -100 (bearish) to +100
                                    (bullish), blended from five factor sleeves. Hover any cell in the table to see
                                    the breakdown:
                                </p>
                                <ul class="small mb-3">
                                    <li><strong>Trend</strong> - is the price structure pointing up? Moving-average
                                        stack, adaptive average, regression slope, channel position.</li>
                                    <li><strong>Momentum</strong> - how strong and <em>steady</em> recent gains are,
                                        plus closeness to the 52-week high (names near highs tend to keep working).</li>
                                    <li><strong>Mean reversion</strong> - the contrarian sleeve. Scores
                                        <em>positive when the stock is beaten down</em> (stretched below average,
                                        RSI low), because stretches tend to snap back.</li>
                                    <li><strong>Volume flow</strong> - is real money confirming the move? Money Flow
                                        Index, Chaikin Money Flow, OBV, price vs VWAP.</li>
                                    <li><strong>Quality</strong> - risk-adjusted health: Sharpe ratio, drawdown
                                        depth, volatility regime.</li>
                                </ul>
                                <p class="small">
                                    The sleeve weights adapt to each stock's <strong>regime</strong>: trending names
                                    lean on trend and momentum, choppy mean-reverting names lean on the contrarian
                                    sleeve. High volatility applies a 20% conviction haircut.
                                </p>
                                <p class="small mb-0">
                                    The <strong>rank (1-10)</strong> is the score's decile <em>within your
                                    watchlist</em>: rank 1 = top 10% of names scanned. BUY/SELL labels come from the
                                    rank, so they mean "looks best among these names", not a guaranteed prediction.
                                    In a falling market, rank 1 can just mean least bad.
                                </p>
                            </div>
                            <div class="col-md-6">
                                <h6>What drives the signal vs what is context</h6>
                                <p class="small">
                                    About twenty indicators feed the score through the five sleeves. Four regime
                                    statistics (Hurst exponent, variance ratio, ADX, efficiency ratio) choose the
                                    sleeve weights. Everything else on screen is deliberately <em>context</em>, not
                                    part of the score:
                                </p>
                                <ul class="small mb-3">
                                    <li><strong>Risk stats</strong> (Sharpe, max drawdown, beta, VaR/CVaR) describe
                                        the consequences of holding - use them for position sizing.</li>
                                    <li><strong>Fundamentals</strong> (PE, ROE, earnings dates) are shown but kept
                                        out of the score because they cannot be backtested point-in-time here.</li>
                                    <li><strong>The market banner</strong> (SPY vs 200-day average, VIX) sets overall
                                        aggressiveness but never changes individual ranks.</li>
                                    <li><strong>Signals</strong> are discrete events (squeeze fired, breakout,
                                        oversold stretch). Hover any signal for what it means and how it tested.</li>
                                </ul>
                                <h6>Reading the risk numbers</h6>
                                <ul class="small mb-3">
                                    <li><strong>Sharpe</strong>: return per unit of risk. 1 is good, 2+ is excellent.</li>
                                    <li><strong>MaxDD</strong>: worst peak-to-trough fall - the how-much-pain number.</li>
                                    <li><strong>Beta</strong>: market sensitivity. 2 = double the market's swings.</li>
                                    <li><strong>VaR 95%</strong>: on 19 of 20 days, losses stay smaller than this.</li>
                                    <li><strong>CVaR</strong>: the average loss on the worst 1-in-20 days.</li>
                                    <li><strong>Pairwise correlation</strong>: how much holdings move together -
                                        near +1 means several tickers making one bet.</li>
                                </ul>
                                <p class="small text-muted mb-0">
                                    Research tool, not investment advice. See the README for full backtests of every
                                    sleeve and signal, including the ones that did not work.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Main Analysis Row -->
        <div class="row">
            <!-- Analysis Progress -->
            <div class="col-12 mb-3">
                <div class="card" id="progress-card" style="display: none;">
                    <div class="card-header">Analysis Progress</div>
                    <div class="card-body" id="progress-area">
                        <div class="progress mb-3" style="height: 25px;">
                            <div class="progress-bar progress-bar-striped progress-bar-animated" 
                                role="progressbar" style="width: 10%;" 
                                aria-valuenow="10" aria-valuemin="0" aria-valuemax="100">
                                Starting analysis...
                            </div>
                        </div>
                        <div id="progress-details" class="text-muted small">
                            <p>Initializing analysis...</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Market Analysis Results -->
            <div class="col-12">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <span>Market Analysis</span>
                        <div class="input-group" style="max-width: 500px;">
                            <input type="text" class="form-control form-control-sm" id="symbols" placeholder="Enter tickers (comma separated)">
                            <button class="btn btn-sm btn-outline-secondary" id="update-watchlist-button">Update Watchlist</button>
                        </div>
                    </div>
                    <div class="card-body" id="results-area">
                        <p class="text-center text-muted">
                            Analysis will begin automatically. Results will appear here.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Configuration Modal -->
    <div class="modal fade" id="configModal" tabindex="-1" aria-labelledby="configModalLabel" aria-hidden="true">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="configModalLabel">Edit Configuration</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <form id="config-form">
                        <div class="mb-3">
                            <label for="refreshInterval" class="form-label">Auto-refresh Interval (minutes):</label>
                            <input type="number" class="form-control" id="refreshInterval" min="1" max="60" value="30">
                        </div>
                        <div class="mb-3 form-check">
                            <input type="checkbox" class="form-check-input" id="hideNonBuys">
                            <label class="form-check-label" for="hideNonBuys">Hide non-buy recommendations</label>
                        </div>
                        <div class="mb-3">
                            <label for="hideRanksAbove" class="form-label">Hide ranks above:</label>
                            <select class="form-select" id="hideRanksAbove">
                                <option value="0">Show all ranks</option>
                                <option value="5">Hide ranks 5-10 (HOLD and SELL)</option>
                                <option value="7">Hide ranks 7-10 (SELL only)</option>
                                <option value="3">Hide ranks 3-10 (Show only strongest BUYs)</option>
                            </select>
                        </div>
                        <div class="mb-3">
                            <label for="watchlistSymbols" class="form-label">Watchlist Symbols (comma separated):</label>
                            <textarea class="form-control" id="watchlistSymbols" rows="3"></textarea>
                        </div>
                    </form>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                    <button type="button" class="btn btn-primary" id="save-config-button">Save Changes</button>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Portfolio Modal -->
    <div class="modal fade" id="portfolioModal" tabindex="-1" aria-labelledby="portfolioModalLabel" aria-hidden="true">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="portfolioModalLabel">Edit Portfolio</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <div class="mb-3">
                        <label for="portfolioSymbols" class="form-label">Portfolio Symbols (comma separated):</label>
                        <input type="text" class="form-control" id="portfolioSymbols">
                    </div>
                    <div id="portfolio-positions-container">
                        <!-- Positions will be added dynamically -->
                        <div class="text-center text-muted">
                            Enter symbols above to manage positions
                        </div>
                    </div>
                    <button type="button" class="btn btn-sm btn-outline-primary mt-3" id="add-position-button">+ Add Position</button>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                    <button type="button" class="btn btn-primary" id="save-portfolio-button">Save Portfolio</button>
                </div>
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p>Stock Trading Recommendation System &copy; 2025</p>
    </div>
    
    <!-- JavaScript files -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script src="/static/js/glossary.js"></script>
    <script src="/static/js/recommend.js"></script>
    <script src="/static/js/main.js"></script>
    <script src="/static/js/config.js"></script>
    <script src="/static/js/portfolio.js"></script>
    <script src="/static/js/utils.js"></script>
</body>
</html>
"""