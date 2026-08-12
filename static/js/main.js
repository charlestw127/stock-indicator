/**
 * Main JavaScript functionality for stock trading app with auto-refresh.
 * Renders the regime-aware multi-factor quant analysis: composite scores,
 * factor sleeves, regime labels, risk metrics and trade signals.
 */

// Global variables
let analysisResults = {};
let refreshInterval = 1800; // 30 minutes in seconds
let countdown = refreshInterval;
let intervalId = null;
let periodsToAnalyze = ['1w', '1m', '6m', '1y', '5y'];
let currentSort = {
    column: 'symbol', // Default sort by symbol
    direction: 'asc'  // Default ascending
};

// Rank -> cell color styles (1 = strongest buy, 10 = strongest sell)
const RANK_STYLES = {
    1: 'background-color: #006400 !important; color: white !important;',
    2: 'background-color: #28a745 !important; color: white !important;',
    3: 'background-color: #5cb85c !important; color: white !important;',
    4: 'background-color: #8fca8f !important; color: black !important;',
    5: 'background-color: #ffc107 !important; color: black !important;',
    6: 'background-color: #ffc107 !important; color: black !important;',
    7: 'background-color: #ffa6a6 !important; color: black !important;',
    8: 'background-color: #f86b7a !important; color: white !important;',
    9: 'background-color: #dc3545 !important; color: white !important;',
    10: 'background-color: #a50000 !important; color: white !important;'
};

/** Format a factor score like +0.42 / -0.17 */
function fmtFactor(v) {
    if (v === null || v === undefined || isNaN(v)) return '–';
    return (v >= 0 ? '+' : '') + Number(v).toFixed(2);
}

/** Format a number with fixed decimals, dash when missing */
function fmtNum(v, decimals = 1) {
    if (v === null || v === undefined || isNaN(v)) return '–';
    return Number(v).toFixed(decimals);
}

/**
 * Tooltip for a period cell: composite score, confidence and the five
 * factor sleeves computed by the quant engine.
 */
function buildPeriodTooltip(periodData) {
    const lines = [periodData.recommendation];
    if (periodData.score !== null && periodData.score !== undefined) {
        let line = `Composite score: ${fmtNum(periodData.score, 1)}`;
        if (periodData.confidence !== undefined && periodData.confidence !== null) {
            line += ` | Confidence: ${periodData.confidence}%`;
        }
        lines.push(line);
    }
    const f = periodData.factors;
    if (f) {
        lines.push(`Trend ${fmtFactor(f.trend)} | Momentum ${fmtFactor(f.momentum)} | MeanRev ${fmtFactor(f.mean_reversion)}`);
        lines.push(`Flow ${fmtFactor(f.volume_flow)} | Quality ${fmtFactor(f.quality)}`);
    }
    const r = periodData.regime;
    if (r) {
        lines.push(`Regime: ${r.trend}, ${r.volatility} vol` +
            (r.hurst !== null && r.hurst !== undefined ? ` (Hurst ${fmtNum(r.hurst, 2)})` : ''));
    }
    const cs = periodData.cross_section;
    if (cs) {
        let line = `Universe percentile: ${fmtNum(cs.universe_pctile, 0)}`;
        if (cs.sector_pctile !== undefined && cs.sector_pctile !== null) {
            line += ` | ${cs.sector}: ${fmtNum(cs.sector_pctile, 0)}`;
        }
        lines.push(line);
    }
    return lines.join('\n');
}

/**
 * Rank-based action recommendation (same thresholds as before, driven by
 * cross-sectional percentile ranks).
 */
function actionFromRanks(shortTermRank, mediumTermRank) {
    if (shortTermRank >= 9 && mediumTermRank >= 8) {
        return {
            actionClass: 'text-danger fw-bold', actionText: 'SELL NOW',
            timeHorizon: 'Immediate',
            reasonText: 'Multiple indicators suggest bearish conditions',
            strategyText: 'Consider liquidating position to prevent further losses'
        };
    }
    if (shortTermRank >= 7) {
        return {
            actionClass: 'text-danger', actionText: 'CONSIDER SELLING',
            timeHorizon: '1-4 weeks',
            reasonText: 'Indicators suggest potential reversal ahead',
            strategyText: 'Consider reducing position or setting stop losses'
        };
    }
    if (shortTermRank <= 2 && mediumTermRank <= 3) {
        return {
            actionClass: 'text-success fw-bold', actionText: 'BUY NOW',
            timeHorizon: 'Now to 2 weeks',
            reasonText: 'Multiple indicators suggest bullish conditions',
            strategyText: 'Consider dollar-cost averaging to build position'
        };
    }
    if (shortTermRank <= 4) {
        return {
            actionClass: 'text-success', actionText: 'CONSIDER BUYING',
            timeHorizon: '2-6 weeks',
            reasonText: 'Technical patterns indicate potential upside',
            strategyText: 'Consider partial position entry with staged buying'
        };
    }
    return {
        actionClass: 'text-warning', actionText: 'HOLD',
        timeHorizon: '3-6 months',
        reasonText: 'Mixed signals suggest sideways price action',
        strategyText: 'Wait for clearer directional signals'
    };
}

/**
 * Build one <tr> of the market analysis table (used by both the initial
 * render and re-sorting).
 */
function buildRowHtml(row) {
    let html = `<tr><td class="fw-bold">${row.symbol}</td>`;

    // Cells for each time period
    periodsToAnalyze.forEach(period => {
        const periodData = row.periods[period];

        if (!periodData || periodData.isError) {
            html += `<td class="text-danger" title="${periodData ? periodData.errorMsg : 'No data'}">Error</td>`;
            return;
        }
        if (periodData.recommendation === 'N/A') {
            html += `<td class="text-muted">N/A</td>`;
            return;
        }

        const recClass = `rank-${periodData.rank || 5}`;
        let styles = RANK_STYLES[periodData.rank] || '';
        styles += ' font-weight: bold; text-align: center; padding: 5px; border-radius: 4px;';

        const displayText = periodData.recommendation.split(' ')[0]; // BUY/SELL/HOLD
        const rankDisplay = `${displayText} (${periodData.rank})`;
        const tooltip = buildPeriodTooltip(periodData);

        html += `<td class="${recClass}" style="${styles}" title="${tooltip}">${rankDisplay}</td>`;
    });

    // Short/medium-term ranks drive the action call
    const shortTermRank = Math.min(
        row.periods['1w']?.rank || 5,
        row.periods['1m']?.rank || 5
    );
    const mediumTermRank = Math.min(
        row.periods['6m']?.rank || 5,
        row.periods['1y']?.rank || 5
    );
    const action = actionFromRanks(shortTermRank, mediumTermRank);

    let actionStyleClass = 'action-hold';
    if (action.actionText.includes('SELL')) {
        actionStyleClass = 'action-sell';
    } else if (action.actionText.includes('BUY')) {
        actionStyleClass = 'action-buy';
    }

    // Quant context: prefer the 1m horizon, fall back to 1w
    const q = (row.periods['1m'] && row.periods['1m'].regime) ? row.periods['1m'] :
              (row.periods['1w'] && row.periods['1w'].regime) ? row.periods['1w'] : null;

    let regimeHtml = '';
    let riskHtml = '';
    let signalsHtml = '';
    let fundHtml = '';

    const fund = row.fundamentals;
    if (fund && (fund.sector || fund.pe !== null)) {
        const bits = [];
        if (fund.sector) bits.push(fund.sector);
        if (fund.pe !== null && fund.pe !== undefined) bits.push(`PE ${fmtNum(fund.pe, 1)}`);
        if (fund.roe !== null && fund.roe !== undefined) bits.push(`ROE ${fmtNum(fund.roe, 0)}%`);
        if (fund.next_earnings) bits.push(`Earnings ${fund.next_earnings}`);
        if (bits.length > 0) {
            fundHtml = `
                <div class="detail-row">
                    <span class="label">Profile:</span>
                    <span class="value">${bits.join(' · ')}</span>
                </div>`;
        }
    }

    if (q) {
        const r = q.regime;
        const regimeClass = r.trend === 'trending' ? 'trending' :
                            r.trend === 'mean-reverting' ? 'mean-reverting' : '';
        const volBadge = r.volatility === 'high' ?
            `<span class="quant-regime high-vol">high vol</span>` :
            (r.volatility && r.volatility !== 'unknown' ?
                `<span class="quant-regime">${r.volatility} vol</span>` : '');
        regimeHtml = `
            <div class="detail-row">
                <span class="label">Regime:</span>
                <span class="value">
                    <span class="quant-regime ${regimeClass}">${r.trend}</span>
                    ${volBadge}
                </span>
            </div>`;

        if (q.risk) {
            const k = q.risk;
            riskHtml = `
                <div class="detail-row">
                    <span class="label">Risk:</span>
                    <span class="value">Sharpe ${fmtNum(k.sharpe, 2)} · MaxDD ${fmtNum(k.max_drawdown, 0)}% · β ${fmtNum(k.beta, 2)}</span>
                </div>`;
        }

        const signals = (q.signals || []).concat((fund && fund.signals) || []).slice(0, 4);
        if (signals.length > 0) {
            signalsHtml = `<div class="quant-signals">` +
                signals.map(s => `<div class="quant-signal">${s}</div>`).join('') +
                `</div>`;
        }
    }

    html += `
        <td class="${action.actionClass}">
            <div class="action-recommendation ${actionStyleClass}">
                ${action.actionText}
            </div>
            <div class="recommendation-details">
                ${regimeHtml}
                ${fundHtml}
                <div class="detail-row">
                    <span class="label">Horizon:</span>
                    <span class="value">${action.timeHorizon}</span>
                </div>
                <div class="detail-row">
                    <span class="label">Reason:</span>
                    <span class="value">${action.reasonText}</span>
                </div>
                <div class="detail-row">
                    <span class="label">Strategy:</span>
                    <span class="value">${action.strategyText}</span>
                </div>
                ${riskHtml}
                ${signalsHtml}
            </div>
        </td>`;

    html += `</tr>`;
    return html;
}

// Wait for document to load
document.addEventListener("DOMContentLoaded", function() {
    console.log("DOM fully loaded");

    // Get DOM elements
    const symbolsInput = document.getElementById('symbols');
    const resultsArea = document.getElementById('results-area');
    const nextRefreshElement = document.getElementById('next-refresh');
    const manualAnalyzeButton = document.getElementById('manual-analyze-button');
    const progressCard = document.getElementById('progress-card');
    const progressBar = document.querySelector('.progress-bar');
    const progressDetails = document.getElementById('progress-details');

    // Add event listener for manual analyze button
    if (manualAnalyzeButton) {
        manualAnalyzeButton.addEventListener('click', function() {
            console.log("Manual analyze button clicked");
            // Reset the auto-refresh countdown when manually triggered
            countdown = refreshInterval;
            updateRefreshStatus();
            handleAnalyzeStocks();
        });
    }

    // Ensure symbols input has a default value if empty
    if (symbolsInput && (!symbolsInput.value || symbolsInput.value.trim() === '')) {
        symbolsInput.value = "AAPL,MSFT,GOOGL";
        console.log("Set default symbols:", symbolsInput.value);
    }

    // Add a slight delay to ensure everything is loaded properly
    setTimeout(() => {
        // Start first analysis
        handleAnalyzeStocks();

        // Set up auto-refresh
        startAutoRefresh();
    }, 1000);

    /**
     * Start auto-refresh timer
     */
    function startAutoRefresh() {
        countdown = refreshInterval;
        updateRefreshStatus();

        // Clear any existing interval
        if (intervalId) {
            clearInterval(intervalId);
        }

        // Set up the countdown and refresh
        intervalId = setInterval(function() {
            countdown--;
            updateRefreshStatus();

            if (countdown <= 0) {
                handleAnalyzeStocks();
                countdown = refreshInterval;
            }
        }, 1000);
    }

    /**
     * Update refresh interval
     * @param {number} seconds - New refresh interval in seconds
     */
    function updateRefreshInterval(seconds) {
        refreshInterval = seconds;
        countdown = refreshInterval;
        console.log(`Refresh interval updated to ${refreshInterval} seconds`);
        startAutoRefresh(); // Restart timer with new interval
    }

    /**
     * Update the refresh status display
     */
    function updateRefreshStatus() {
        if (nextRefreshElement) {
            const minutes = Math.floor(countdown / 60);
            const seconds = countdown % 60;
            nextRefreshElement.textContent = `Next refresh in ${minutes}m ${seconds}s`;
        }
    }

    /**
     * Analyze stocks handler - auto-called by timer or manual button
     */
    async function handleAnalyzeStocks() {
        if (!symbolsInput || !resultsArea || !progressCard) {
            console.error("Required elements not found for handleAnalyzeStocks");
            return;
        }

        console.log("Starting stock analysis...");

        // Get symbols
        const symbols = symbolsInput.value.split(',').map(s => s.trim().toUpperCase()).filter(s => s !== '');
        console.log("Symbols to analyze:", symbols);

        if (symbols.length === 0) {
            console.error("No symbols to analyze");
            resultsArea.innerHTML = '<div class="alert alert-warning">Please enter at least one symbol</div>';
            return;
        }

        // Show the progress card
        progressCard.style.display = 'block';

        // Update progress area with initial info
        progressBar.style.width = '10%';
        progressBar.setAttribute('aria-valuenow', 10);
        progressBar.textContent = 'Starting analysis...';
        progressBar.classList.add('progress-bar-animated');
        progressBar.classList.remove('bg-danger');
        progressBar.classList.add('bg-primary');

        progressDetails.innerHTML = `<p>Starting quant analysis for: ${symbols.join(', ')}</p>`;

        // If this is the first analysis, show a placeholder message in results area
        if (!resultsArea.innerHTML || resultsArea.innerHTML.includes('Enter tickers')) {
            resultsArea.innerHTML = '<p class="text-center text-muted">Analysis in progress. Results will appear here when complete.</p>';
        }

        // If manual button exists, disable it during analysis
        const manualButton = document.getElementById('manual-analyze-button');
        if (manualButton) {
            manualButton.disabled = true;
            manualButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing...';
        }

        // Update progress periodically
        let progress = 10;
        const progressInterval = setInterval(() => {
            if (progress < 90) {
                progress += 5;
                progressBar.style.width = `${progress}%`;
                progressBar.setAttribute('aria-valuenow', progress);

                if (progress < 30) {
                    progressBar.textContent = 'Fetching price history...';
                } else if (progress < 60) {
                    progressBar.textContent = 'Computing factors & regimes...';
                } else {
                    progressBar.textContent = 'Scoring & ranking universe...';
                }

                // Add random progress details
                if (progress % 15 === 0) {
                    const randomSymbol = symbols[Math.floor(Math.random() * symbols.length)];
                    const newDetail = document.createElement('p');
                    newDetail.textContent = `Processing ${randomSymbol} across all horizons...`;
                    progressDetails.prepend(newDetail);

                    // Keep only the last 5 messages
                    if (progressDetails.childElementCount > 5) {
                        progressDetails.removeChild(progressDetails.lastChild);
                    }
                }
            }
        }, 800);

        try {
            console.log(`Analyzing: ${symbols.join(', ')} | All time periods`);

            // API request for all time periods
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    symbols,
                    periodsToAnalyze: ['1w', '1m', '6m', '1y', '5y']
                })
            });

            // Stop the progress animation
            clearInterval(progressInterval);

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Server error: ${response.statusText}. ${errorText}`);
            }

            // Set progress to 100%
            progressBar.style.width = '100%';
            progressBar.setAttribute('aria-valuenow', 100);
            progressBar.textContent = 'Complete!';
            progressBar.classList.remove('progress-bar-animated');

            // Process results
            analysisResults = await response.json();
            console.log('Received analysis results');

            // Add a slight delay before displaying results for better UX
            setTimeout(() => {
                displayResults(analysisResults);

                // Hide progress card after a delay
                setTimeout(() => {
                    progressCard.style.display = 'none';
                }, 1500);

                // Re-enable the manual button
                if (manualButton) {
                    manualButton.disabled = false;
                    manualButton.textContent = 'Run Analysis Now';
                }
            }, 500);

        } catch (error) {
            // Stop the progress animation
            clearInterval(progressInterval);

            console.error('Error:', error);

            // Update progress area to show error
            progressBar.style.width = '100%';
            progressBar.setAttribute('aria-valuenow', 100);
            progressBar.textContent = 'Error!';
            progressBar.classList.remove('progress-bar-animated');
            progressBar.classList.remove('bg-primary');
            progressBar.classList.add('bg-danger');

            // Add error message to progress details
            const errorMsg = document.createElement('p');
            errorMsg.className = 'text-danger';
            errorMsg.textContent = `Error: ${error.message || 'Analysis failed'}`;
            progressDetails.prepend(errorMsg);

            // If we don't have any results yet, update the results area
            if (!analysisResults.symbols || Object.keys(analysisResults.symbols).length === 0) {
                resultsArea.innerHTML = `<div class="alert alert-danger">${error.message || 'Analysis failed'}</div>`;
            }

            // Re-enable the manual button
            if (manualButton) {
                manualButton.disabled = false;
                manualButton.textContent = 'Try Again';
            }

            // Don't hide the progress card on error, let user see what happened
        }
    }

    /**
     * Display results in the UI with sorting capability
     */
    function displayResults(data) {
        if (!resultsArea) {
            console.error("Results area not found!");
            return;
        }

        if (!data.symbols || Object.keys(data.symbols).length === 0) {
            resultsArea.innerHTML = '<div class="alert alert-warning">No symbols to analyze.</div>';
            console.log('No symbols found');
            return;
        }

        const symbols = Object.keys(data.symbols);
        console.log(`Displaying results for ${symbols.length} symbols`);

        // Store processed data for sorting
        let rowsData = [];

        // Process data for each symbol
        symbols.forEach(symbol => {
            const rowData = {
                symbol: symbol,
                fundamentals: data.symbols[symbol].fundamentals || null,
                periods: {}
            };

            // Process each time period
            periodsToAnalyze.forEach(period => {
                const periodData = data.symbols[symbol][period];

                if (!periodData || periodData.error) {
                    rowData.periods[period] = {
                        recommendation: 'N/A',
                        sortValue: 0,
                        isError: !!periodData?.error,
                        errorMsg: periodData?.error || 'No data'
                    };
                    return;
                }

                // Get recommendation and assign a sort value
                const recommendation = periodData.recommendation || 'N/A';
                const rank = periodData.rank || 5;  // Default to middle rank if not available

                rowData.periods[period] = {
                    recommendation: recommendation,
                    sortValue: 11 - rank,  // Invert rank for sorting (1=highest, 10=lowest)
                    rank: rank,
                    isError: false,
                    // Quant engine payload
                    score: periodData.score,
                    confidence: periodData.confidence,
                    factors: periodData.factors,
                    regime: periodData.regime,
                    risk: periodData.risk,
                    signals: periodData.signals,
                    cross_section: periodData.cross_section
                };
            });

            rowsData.push(rowData);
        });

        // Sort rows based on current sort settings
        sortRows(rowsData, currentSort.column, currentSort.direction);

        // Market regime banner, rank movers, cache note
        let preamble = '';
        const mkt = data.market;
        if (mkt && mkt.risk && mkt.risk !== 'unknown') {
            const cls = mkt.risk === 'on' ? 'alert-success' :
                        mkt.risk === 'off' ? 'alert-danger' : 'alert-warning';
            const parts = [`Market: risk-${mkt.risk}`];
            if (mkt.spy_dist_200dma !== null && mkt.spy_dist_200dma !== undefined) {
                parts.push(`SPY ${mkt.spy_dist_200dma >= 0 ? '+' : ''}${mkt.spy_dist_200dma}% vs 200dma`);
            }
            if (mkt.vix !== null && mkt.vix !== undefined) {
                parts.push(`VIX ${mkt.vix}`);
            }
            const note = mkt.note ? ` — ${mkt.note}` : '';
            preamble += `<div class="alert ${cls} py-2 mb-2">${parts.join(' | ')}${note}</div>`;
        }
        if (data.movers && data.movers.length > 0) {
            const items = data.movers.slice(0, 6)
                .map(m => `${m.symbol} ${m.period}: ${m.from} → ${m.to}`).join(', ');
            preamble += `<div class="text-muted small mb-2">Rank movers since last scan: ${items}</div>`;
        }
        if (data.cached && data.asOf) {
            preamble += `<div class="text-muted small mb-2">Served from background scan at ${data.asOf}</div>`;
        }

        // Build the table HTML
        let html = preamble + `
            <div class="table-responsive">
                <table class="table table-bordered table-hover recommendation-table" id="market-analysis-table">
                    <thead class="table-light">
                        <tr>
                            <th class="sortable" data-column="symbol">
                                Symbol
                                <span class="sort-icon" id="sort-symbol">↕</span>
                            </th>`;

        // Add column headers for each time period
        periodsToAnalyze.forEach(period => {
            html += `
                <th class="sortable" data-column="${period}">
                    ${period}
                    <span class="sort-icon" id="sort-${period}">↕</span>
                </th>`;
        });

        // Add action column header
        html += `
            <th class="sortable" data-column="action">
                Action
                <span class="sort-icon" id="sort-action">↕</span>
            </th>
        </tr></thead><tbody>`;

        // Add rows for each symbol
        rowsData.forEach(row => {
            html += buildRowHtml(row);
        });

        html += `</tbody></table></div>`;

        // Add timestamp
        const timestamp = new Date().toLocaleString();
        html += `<div class="text-end text-muted small">Last updated: ${timestamp} · Hover a cell for factor breakdown</div>`;

        // Set the HTML content
        resultsArea.innerHTML = html;

        // Update the sort icon for current sort
        updateSortIcon(currentSort.column, currentSort.direction);

        // Add event listeners to sortable headers
        document.querySelectorAll('#market-analysis-table .sortable').forEach(header => {
            header.addEventListener('click', function() {
                const column = this.getAttribute('data-column');

                // Toggle direction if same column, otherwise default to ascending
                let direction = 'asc';
                if (column === currentSort.column) {
                    direction = currentSort.direction === 'asc' ? 'desc' : 'asc';
                }

                // Update current sort state
                currentSort.column = column;
                currentSort.direction = direction;

                // Sort and redisplay
                sortRows(rowsData, column, direction);
                updateTableBody(rowsData);
                updateSortIcon(column, direction);
            });
        });
    }

    /**
     * Sort rows based on specified column and direction
     */
    function sortRows(rows, column, direction) {
        rows.sort((a, b) => {
            let valueA, valueB;

            if (column === 'symbol') {
                // Sort by symbol name
                valueA = a.symbol;
                valueB = b.symbol;
                return direction === 'asc' ?
                    valueA.localeCompare(valueB) :
                    valueB.localeCompare(valueA);
            } else if (column === 'action') {
                // Sort by recommendation strength (based on 1w period)
                valueA = a.periods['1w']?.sortValue || 0;
                valueB = b.periods['1w']?.sortValue || 0;

                // For ascending, strongest BUY at top
                // For descending, strongest SELL at top
                return direction === 'asc' ?
                    valueB - valueA :
                    valueA - valueB;
            } else {
                // Sort by recommendation strength for a time period
                valueA = a.periods[column]?.sortValue || 0;
                valueB = b.periods[column]?.sortValue || 0;

                // For ascending, strongest BUY at top
                // For descending, strongest SELL at top
                return direction === 'asc' ?
                    valueB - valueA :  // Note: Inverted for more intuitive sorting
                    valueA - valueB;
            }
        });
    }

    /**
     * Update the table body with sorted data
     */
    function updateTableBody(rowsData) {
        const tbody = document.querySelector('#market-analysis-table tbody');
        if (!tbody) return;

        tbody.innerHTML = rowsData.map(buildRowHtml).join('');
    }

    /**
     * Update the sort icon for the active column
     */
    function updateSortIcon(column, direction) {
        // Reset all icons
        document.querySelectorAll('#market-analysis-table .sort-icon').forEach(icon => {
            icon.textContent = '↕';
            icon.classList.remove('active');
        });

        // Update the active icon
        const activeIcon = document.getElementById(`sort-${column}`);
        if (activeIcon) {
            activeIcon.textContent = direction === 'asc' ? '↑' : '↓';
            activeIcon.classList.add('active');
        }
    }

    // Expose functions to global scope
    window.handleAnalyzeStocks = handleAnalyzeStocks;
    window.updateRefreshInterval = updateRefreshInterval;
    window.startAutoRefresh = startAutoRefresh;
    window.updateRefreshStatus = updateRefreshStatus;
});
