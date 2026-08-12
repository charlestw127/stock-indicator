/**
 * Portfolio management for stock trading app
 */

// Portfolio data
let portfolioData = {
    symbols: "",
    positions: []
};

// Wait for document to load
document.addEventListener("DOMContentLoaded", function() {
    // Get portfolio elements
    const portfolioArea = document.getElementById('portfolio-area');
    const editPortfolioButton = document.getElementById('edit-portfolio-button');
    const analyzePortfolioButton = document.getElementById('analyze-portfolio-button');
    const savePortfolioButton = document.getElementById('save-portfolio-button');
    const portfolioSymbolsInput = document.getElementById('portfolioSymbols');
    const portfolioPositionsContainer = document.getElementById('portfolio-positions-container');
    const addPositionButton = document.getElementById('add-position-button');
    
    // Initialize
    if (window.appConfig && window.appConfig.portfolio) {
        updatePortfolioData(window.appConfig.portfolio);
    }
    
    // Set up event listeners
    if (editPortfolioButton) {
        editPortfolioButton.addEventListener('click', function() {
            openPortfolioModal();
        });
    }
    
    if (analyzePortfolioButton) {
        analyzePortfolioButton.addEventListener('click', function() {
            analyzePortfolio();
        });
    }
    
    if (savePortfolioButton) {
        savePortfolioButton.addEventListener('click', function() {
            savePortfolio();
        });
    }
    
    if (addPositionButton) {
        addPositionButton.addEventListener('click', function() {
            addEmptyPosition();
        });
    }
    
    if (portfolioSymbolsInput) {
        portfolioSymbolsInput.addEventListener('change', function() {
            updatePositionsFromSymbols();
        });
    }
    
    /**
     * Update portfolio data
     * @param {Object} data - Portfolio data
     */
    function updatePortfolioData(data) {
        portfolioData = data;
        
        // Request portfolio analysis
        setTimeout(() => {
            analyzePortfolio();
        }, 1000);
    }
    
    /**
     * Analyze portfolio
     */
    function analyzePortfolio() {
        if (!portfolioData.symbols || portfolioData.symbols.trim() === '') {
            if (portfolioArea) {
                portfolioArea.innerHTML = `
                    <div class="alert alert-info">
                        Your portfolio is empty. Click "Edit Portfolio" to add stocks.
                    </div>
                `;
            }
            return;
        }
        
        // Show loading indicator
        if (portfolioArea) {
            portfolioArea.innerHTML = `
                <div class="text-center">
                    <div class="spinner-border text-primary"></div>
                    <p class="mt-2">Analyzing portfolio...</p>
                </div>
            `;
        }
        
        // Request analysis for portfolio symbols
        fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                symbols: portfolioData.symbols.split(',').map(s => s.trim()),
                periodsToAnalyze: ['1w', '1m', '6m', '1y', '5y'],
                isPortfolio: true
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Portfolio analysis data received:', 
                Object.keys(data.symbols).map(sym => ({
                    symbol: sym, 
                    lastPrices: Object.keys(data.symbols[sym])
                        .filter(k => ['1w', '1m', '6m', '1y', '5y'].includes(k))
                        .map(p => ({ 
                            period: p, 
                            price: data.symbols[sym][p]?.lastPrice 
                        }))
                }))
            );
            displayPortfolioAnalysis(data);
        })
        .catch(error => {
            console.error('Error analyzing portfolio:', error);
            if (portfolioArea) {
                portfolioArea.innerHTML = `
                    <div class="alert alert-danger">
                        Failed to analyze portfolio: ${error.message}
                    </div>
                `;
            }
        });
    }
    
    /**
     * Generate detailed recommendation based on indicators
     * @param {Object} symbolData - Analysis data for a symbol
     * @param {Number} shortTermRank - Short term rank value
     * @param {Number} mediumTermRank - Medium term rank value
     * @returns {Object} - Action recommendation with text, class, and details
     */
    function generateDetailedRecommendation(symbolData, shortTermRank, mediumTermRank) {
        // Default values
        let actionClass = 'text-warning';
        let actionText = 'HOLD';
        let timeHorizon = '3-6 months';
        let reasonText = '';
        let strategyText = '';
        
        // Get indicators from the most recent period (1w)
        const weekData = symbolData['1w'] || {};
        const monthData = symbolData['1m'] || {};
        const indicators = weekData.indicators || {};
        
        // Extract key indicators
        const rsi = indicators.rsi || 50;
        const bbp = indicators.bbp || 0.5;
        const macd = indicators.macd || 0;
        const adx = indicators.adx || 25;
        
        // Determine trend strength from ADX
        const trendStrength = adx < 20 ? 'weak' : adx < 40 ? 'moderate' : 'strong';
        
        // Generate detailed recommendation based on ranks and indicators
        if (shortTermRank >= 9 && mediumTermRank >= 8) {
            // Strong sell signal
            actionClass = 'text-danger fw-bold';
            actionText = 'SELL NOW';
            timeHorizon = 'Immediate';
            
            if (rsi > 70) {
                reasonText = 'RSI indicates overbought conditions';
            } else if (bbp > 0.85) {
                reasonText = 'Price is near the upper Bollinger Band';
            } else if (macd < 0) {
                reasonText = 'MACD indicates downward momentum';
            } else {
                reasonText = 'Multiple indicators suggest bearish conditions';
            }
            
            strategyText = 'Consider liquidating position to prevent further losses';
        } 
        else if (shortTermRank >= 7 && mediumTermRank >= 6) {
            // Moderate sell signal
            actionClass = 'text-danger';
            actionText = 'CONSIDER SELLING';
            timeHorizon = '1-4 weeks';
            
            if (rsi > 60 && rsi < 70) {
                reasonText = 'RSI approaching overbought territory';
            } else if (bbp > 0.7) {
                reasonText = 'Price moving toward upper Bollinger Band';
            } else if (macd < 0 && trendStrength === 'moderate') {
                reasonText = 'MACD indicates building downward momentum';
            } else {
                reasonText = 'Indicators suggest potential reversal ahead';
            }
            
            strategyText = 'Consider reducing position size or setting tighter stop losses';
        }
        else if (shortTermRank <= 2 && mediumTermRank <= 3) {
            // Strong buy signal
            actionClass = 'text-success fw-bold';
            actionText = 'BUY MORE';
            timeHorizon = 'Now to 2 weeks';
            
            if (rsi < 30) {
                reasonText = 'RSI indicates oversold conditions';
            } else if (bbp < 0.15) {
                reasonText = 'Price is near the lower Bollinger Band';
            } else if (macd > 0 && trendStrength === 'strong') {
                reasonText = 'MACD shows strong upward momentum';
            } else {
                reasonText = 'Multiple indicators suggest bullish conditions';
            }
            
            strategyText = 'Consider dollar-cost averaging to increase position';
        }
        else if (shortTermRank <= 4 && mediumTermRank <= 5) {
            // Moderate buy signal
            actionClass = 'text-success';
            actionText = 'CONSIDER BUYING';
            timeHorizon = '2-6 weeks';
            
            if (rsi > 30 && rsi < 40) {
                reasonText = 'RSI recovering from oversold conditions';
            } else if (bbp > 0.3 && bbp < 0.5) {
                reasonText = 'Price crossing middle Bollinger Band from below';
            } else if (macd > 0) {
                reasonText = 'MACD showing positive momentum';
            } else {
                reasonText = 'Technical patterns indicate potential upside';
            }
            
            strategyText = 'Consider partial position entry with staged buying';
        }
        else if (shortTermRank >= 6 && mediumTermRank <= 4) {
            // Short-term bearish, medium-term bullish
            actionClass = 'text-primary';
            actionText = 'HOLD & MONITOR';
            timeHorizon = '1-3 months';
            reasonText = 'Short-term weakness but medium-term potential';
            strategyText = 'Watch for short-term bottom before adding to position';
        }
        else if (shortTermRank <= 4 && mediumTermRank >= 6) {
            // Short-term bullish, medium-term bearish
            actionClass = 'text-info';
            actionText = 'SHORT-TERM OPPORTUNITY';
            timeHorizon = '1-4 weeks';
            reasonText = 'Short-term strength may be temporary';
            strategyText = 'Consider tactical trading with tight stop-loss';
        }
        else {
            // Neutral signals
            actionClass = 'text-warning';
            actionText = 'HOLD';
            timeHorizon = '3-6 months';
            
            if (Math.abs(macd) < 0.001) {
                reasonText = 'MACD near zero line suggests consolidation';
                strategyText = 'Monitor for breakout signals in either direction';
            } else if (rsi > 40 && rsi < 60) {
                reasonText = 'RSI in neutral zone suggests balanced market';
                strategyText = 'Maintain position and reassess in 2-4 weeks';
            } else if (bbp > 0.4 && bbp < 0.6) {
                reasonText = 'Price near middle Bollinger Band suggests equilibrium';
                strategyText = 'Hold and wait for clearer directional signals';
            } else {
                reasonText = 'Mixed signals suggest sideways price action';
                strategyText = 'Consider writing covered calls to generate income while holding';
            }
        }
        
        return {
            actionClass,
            actionText,
            timeHorizon,
            reasonText,
            strategyText
        };
    }
    
    /**
     * Helper function to find the last price for a symbol across all timeframes
     * Prioritizes the most recent data and handles null values better
     */
    function findLastPrice(data, symbol) {
        if (!data.symbols[symbol]) return null;
        
        // Priority order to check for prices (1w is likely most recent)
        const timeframes = ['1w', '1m', '6m', '1y', '5y'];
        
        // First check if 1w has price
        for (const period of timeframes) {
            const periodData = data.symbols[symbol][period];
            if (periodData && 
                periodData.lastPrice !== undefined && 
                periodData.lastPrice !== null && 
                !isNaN(periodData.lastPrice)) {
                return periodData.lastPrice;
            }
        }
        
        // Fallback - try to extract from any available period
        for (const period in data.symbols[symbol]) {
            const periodData = data.symbols[symbol][period];
            if (periodData && 
                periodData.lastPrice !== undefined && 
                periodData.lastPrice !== null && 
                !isNaN(periodData.lastPrice)) {
                return periodData.lastPrice;
            }
        }
        
        return null;
    }
    
    /**
     * Display portfolio analysis
     * @param {Object} data - Analysis data
     */
    function displayPortfolioAnalysis(data) {
        if (!portfolioArea) return;
        
        if (!data.symbols || Object.keys(data.symbols).length === 0) {
            portfolioArea.innerHTML = `
                <div class="alert alert-warning">
                    No analysis data available for your portfolio symbols.
                </div>
            `;
            return;
        }
        
        // Prepare table HTML
        let html = `
            <div class="table-responsive">
                <table class="table table-bordered table-hover portfolio-table" id="portfolio-table">
                    <thead class="table-light">
                        <tr>
                            <th>Symbol</th>
                            <th>Position</th>
                            <th>1w</th>
                            <th>1m</th>
                            <th>6m</th>
                            <th>1y</th>
                            <th>5y</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        // Get portfolio symbols
        const portfolioSymbols = portfolioData.symbols.split(',').map(s => s.trim());
        
        // Add a row for each portfolio symbol
        portfolioSymbols.forEach(symbol => {
            if (!data.symbols[symbol]) return;
            
            // Find position info
            const position = portfolioData.positions.find(p => p.symbol === symbol) || { shares: 0, entryPrice: 0 };
            
            // Get last price using improved consistent method
            const lastPrice = findLastPrice(data, symbol);
            
            // Skip if we couldn't get a price
            if (lastPrice === null) {
                console.log(`Skipping ${symbol} - no valid price found`);
                return;
            }
            
            // Calculate position value
            const positionValue = position.shares * lastPrice;
            const avgPrice = position.entryPrice || 0;
            const costBasis = position.shares * avgPrice;
            const pnl = positionValue - costBasis;
            const pnlPercent = costBasis > 0 ? (pnl / costBasis) * 100 : 0;
            
            // Determine overall action based on short/medium-term indicators
            const shortTermRank = Math.min(
                data.symbols[symbol]['1w']?.rank || 5,
                data.symbols[symbol]['1m']?.rank || 5
            );
            
            const mediumTermRank = Math.min(
                data.symbols[symbol]['6m']?.rank || 5,
                data.symbols[symbol]['1y']?.rank || 5
            );
            
            // Generate detailed recommendation
            const recommendation = generateDetailedRecommendation(
                data.symbols[symbol], 
                shortTermRank, 
                mediumTermRank
            );
            
            // Determine action class for styling
            let actionStyleClass = 'action-hold';
            if (recommendation.actionText.includes('SELL')) {
                actionStyleClass = 'action-sell';
            } else if (recommendation.actionText.includes('BUY')) {
                actionStyleClass = 'action-buy';
            } else if (recommendation.actionText.includes('OPPORTUNITY')) {
                actionStyleClass = 'action-opportunity';
            } else if (recommendation.actionText.includes('MONITOR')) {
                actionStyleClass = 'action-monitor';
            }
            
            html += `<tr>
                <td class="fw-bold">${symbol}</td>
                <td>
                    <div class="position-info">
                        <div class="position-shares">${position.shares} shares @ $${avgPrice.toFixed(2)}</div>
                        <div class="profit-loss ${pnl >= 0 ? 'positive' : 'negative'}">
                            <span class="amount ${pnl >= 0 ? 'text-success' : 'text-danger'}">${pnl >= 0 ? '+' : ''}$${Math.abs(pnl).toFixed(2)}</span>
                            <span class="percent">(${pnlPercent.toFixed(2)}%)</span>
                        </div>
                        <div class="current-price">Current: $${lastPrice.toFixed(2)}</div>
                    </div>
                </td>
            `;
            
            // Add cells for time periods
            ['1w', '1m', '6m', '1y', '5y'].forEach(period => {
                const periodData = data.symbols[symbol][period];
                
                if (!periodData || periodData.error) {
                    html += `<td class="text-muted">N/A</td>`;
                    return;
                }
                
                // Rank-based styling (using the correct approach for td elements)
                const rank = periodData.rank || 5;
                const recClass = `rank-${rank}`;
                let styles = '';
                
                // Colors based on rank (same as in main.js)
                if (rank === 1) {
                    styles = 'background-color: #006400 !important; color: white !important;';
                } else if (rank === 2) {
                    styles = 'background-color: #28a745 !important; color: white !important;';
                } else if (rank === 3) {
                    styles = 'background-color: #5cb85c !important; color: white !important;';
                } else if (rank === 4) {
                    styles = 'background-color: #8fca8f !important; color: black !important;';
                } else if (rank === 5 || rank === 6) {
                    styles = 'background-color: #ffc107 !important; color: black !important;';
                } else if (rank === 7) {
                    styles = 'background-color: #ffa6a6 !important; color: black !important;';
                } else if (rank === 8) {
                    styles = 'background-color: #f86b7a !important; color: white !important;';
                } else if (rank === 9) {
                    styles = 'background-color: #dc3545 !important; color: white !important;';
                } else if (rank === 10) {
                    styles = 'background-color: #a50000 !important; color: white !important;';
                }
                
                // Common styles
                styles += ' font-weight: bold; text-align: center; padding: 5px; border-radius: 4px;';
                
                const displayText = periodData.recommendation.split(' ')[0]; // Just show BUY/SELL/HOLD
                const rankDisplay = `${displayText} (${rank})`;
                
                // Apply class and style directly to the td element
                html += `<td class="${recClass}" style="${styles}" title="${periodData.recommendation}">${rankDisplay}</td>`;
            });
            
            // Add action cell with detailed recommendation using improved layout
            html += `
                <td class="${recommendation.actionClass}">
                    <div class="action-recommendation ${actionStyleClass}">
                        ${recommendation.actionText}
                    </div>
                    <div class="recommendation-details">
                        <div class="detail-row">
                            <span class="label">Horizon:</span>
                            <span class="value">${recommendation.timeHorizon}</span>
                        </div>
                        <div class="detail-row">
                            <span class="label">Reason:</span>
                            <span class="value">${recommendation.reasonText}</span>
                        </div>
                        <div class="detail-row">
                            <span class="label">Strategy:</span>
                            <span class="value">${recommendation.strategyText}</span>
                        </div>
                    </div>
                </td>
            </tr>`;
        });
        
        html += `</tbody></table></div>`;
        
        // Add portfolio metrics summary (optional enhancement)
        let totalValue = 0;
        let totalCost = 0;
        let totalProfitLoss = 0;
        
        portfolioData.positions.forEach(position => {
            const symbol = position.symbol;
            const lastPrice = findLastPrice(data, symbol);
            if (lastPrice) {
                const positionValue = position.shares * lastPrice;
                const costBasis = position.shares * position.entryPrice;
                totalValue += positionValue;
                totalCost += costBasis;
                totalProfitLoss += (positionValue - costBasis);
            }
        });
        
        // Only add portfolio metrics if we have valid calculations
        if (totalValue > 0) {
            const totalProfitLossPercent = (totalProfitLoss / totalCost) * 100;
            const isProfitable = totalProfitLoss >= 0;
            
            html += `
            <div class="portfolio-metrics">
                <h5>Portfolio Summary</h5>
                <div class="metric-row">
                    <div class="metric-label">Total Cost Basis</div>
                    <div class="metric-value">$${totalCost.toFixed(2)}</div>
                </div>
                <div class="metric-row">
                    <div class="metric-label">Total Market Value</div>
                    <div class="metric-value">$${totalValue.toFixed(2)}</div>
                </div>
                <div class="metric-row">
                    <div class="metric-label">Total P&L</div>
                    <div class="metric-value ${isProfitable ? 'positive' : 'negative'}">
                        ${isProfitable ? '+' : ''}$${Math.abs(totalProfitLoss).toFixed(2)} 
                        (${isProfitable ? '+' : ''}${totalProfitLossPercent.toFixed(2)}%)
                    </div>
                </div>
            </div>`;
        }
        
        // Portfolio-level risk metrics from the server
        const risk = data.portfolioRisk;
        if (risk && risk.ann_vol !== undefined) {
            const fmt = (v, suffix = '') => (v === null || v === undefined) ? 'N/A' : `${v}${suffix}`;
            let pairText = 'N/A';
            if (risk.most_correlated_pair) {
                pairText = `${risk.most_correlated_pair.symbols.join(' / ')} (${risk.most_correlated_pair.correlation})`;
            }
            html += `
            <div class="portfolio-metrics">
                <h5>Portfolio Risk</h5>
                <div class="metric-row">
                    <div class="metric-label">Beta vs SPY</div>
                    <div class="metric-value">${fmt(risk.beta)}</div>
                </div>
                <div class="metric-row">
                    <div class="metric-label">Annualized Volatility</div>
                    <div class="metric-value">${fmt(risk.ann_vol, '%')}</div>
                </div>
                <div class="metric-row">
                    <div class="metric-label">1-day VaR / CVaR (95%)</div>
                    <div class="metric-value">${fmt(risk.var_95, '%')} / ${fmt(risk.cvar_95, '%')}</div>
                </div>
                <div class="metric-row">
                    <div class="metric-label">Effective Positions (1/HHI)</div>
                    <div class="metric-value">${fmt(risk.effective_positions)}</div>
                </div>
                <div class="metric-row">
                    <div class="metric-label">Top Holding</div>
                    <div class="metric-value">${risk.top_holding ? `${risk.top_holding.symbol} (${risk.top_holding.weight}%)` : 'N/A'}</div>
                </div>
                <div class="metric-row">
                    <div class="metric-label">Avg Pairwise Correlation</div>
                    <div class="metric-value">${fmt(risk.avg_correlation)}</div>
                </div>
                <div class="metric-row">
                    <div class="metric-label">Most Correlated Pair</div>
                    <div class="metric-value">${pairText}</div>
                </div>
            </div>`;
        }

        // Add timestamp
        const timestamp = new Date().toLocaleString();
        html += `<div class="timestamp">Last updated: ${timestamp}</div>`;

        // Set HTML
        portfolioArea.innerHTML = html;
    }
    
    /**
     * Open portfolio modal
     */
    function openPortfolioModal() {
        // Populate portfolio symbols
        if (portfolioSymbolsInput) {
            portfolioSymbolsInput.value = portfolioData.symbols;
        }
        
        // Populate positions
        updatePositionsUI();
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('portfolioModal'));
        modal.show();
    }
    
    /**
     * Save portfolio
     */
    function savePortfolio() {
        // Gather portfolio data from form
        const symbols = portfolioSymbolsInput.value.trim();
        const positions = [];
        
        // Collect position data from form
        document.querySelectorAll('.position-row').forEach(row => {
            const symbol = row.querySelector('.position-symbol').value.trim().toUpperCase();
            const shares = parseFloat(row.querySelector('.position-shares').value) || 0;
            const avgPrice = parseFloat(row.querySelector('.position-price').value) || 0;
            
            if (symbol && shares > 0) {
                positions.push({
                    symbol,
                    shares,
                    entryPrice: avgPrice // Using entryPrice field for average price
                });
            }
        });
        
        // Update portfolio data
        portfolioData = {
            symbols,
            positions
        };
        
        // Update app config
        if (window.appConfig) {
            window.appConfig.portfolio = portfolioData;
            
            // Save to server
            fetch('/api/config/save', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(window.appConfig)
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Server error: ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('Portfolio saved:', data);
                
                // Close modal
                const modal = bootstrap.Modal.getInstance(document.getElementById('portfolioModal'));
                if (modal) {
                    modal.hide();
                }
                
                // Analyze portfolio with new data
                analyzePortfolio();
            })
            .catch(error => {
                console.error('Error saving portfolio:', error);
                alert('Failed to save portfolio. Please try again.');
            });
        }
    }
    
    /**
     * Update positions UI in modal
     */
    function updatePositionsUI() {
        if (!portfolioPositionsContainer) return;
        
        let html = '';
        
        if (portfolioData.positions && portfolioData.positions.length > 0) {
            portfolioData.positions.forEach((position, index) => {
                html += createPositionRowHTML(position, index);
            });
        } else {
            html = `
                <div class="text-center text-muted">
                    No positions added yet. Enter symbols and click "Add Position".
                </div>
            `;
        }
        
        portfolioPositionsContainer.innerHTML = html;
        
        // Add event listeners for delete buttons
        document.querySelectorAll('.delete-position-button').forEach(button => {
            button.addEventListener('click', function() {
                const index = parseInt(this.dataset.index);
                deletePosition(index);
            });
        });
    }
    
    /**
     * Create HTML for a position row
     * @param {Object} position - Position data
     * @param {number} index - Position index
     * @returns {string} HTML string
     */
    function createPositionRowHTML(position, index) {
        return `
            <div class="row position-row align-items-end mb-3">
                <div class="col-md-4">
                    <label class="form-label">Symbol</label>
                    <input type="text" class="form-control position-symbol" value="${position.symbol || ''}" required>
                </div>
                <div class="col-md-3">
                    <label class="form-label">Shares</label>
                    <input type="number" class="form-control position-shares" value="${position.shares || ''}" min="0" step="any" required>
                </div>
                <div class="col-md-4">
                    <label class="form-label">Average Price</label>
                    <div class="input-group">
                        <span class="input-group-text">$</span>
                        <input type="number" class="form-control position-price" value="${position.entryPrice || ''}" min="0" step="any">
                    </div>
                </div>
                <div class="col-md-1">
                    <button type="button" class="btn btn-sm btn-outline-danger delete-position-button" data-index="${index}">
                        ×
                    </button>
                </div>
            </div>
        `;
    }
    
    /**
     * Add empty position to UI
     */
    function addEmptyPosition() {
        if (!portfolioPositionsContainer) return;
        
        // Remove placeholder text if present
        if (portfolioPositionsContainer.querySelector('.text-muted')) {
            portfolioPositionsContainer.innerHTML = '';
        }
        
        // Create new position row
        const position = { symbol: '', shares: 0, entryPrice: 0 };
        const index = document.querySelectorAll('.position-row').length;
        
        // Add row to container
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = createPositionRowHTML(position, index);
        
        while (tempDiv.firstChild) {
            portfolioPositionsContainer.appendChild(tempDiv.firstChild);
        }
        
        // Add event listener for delete button
        const deleteButton = portfolioPositionsContainer.querySelector(`.delete-position-button[data-index="${index}"]`);
        if (deleteButton) {
            deleteButton.addEventListener('click', function() {
                deletePosition(index);
            });
        }
    }
    
    /**
     * Delete position at index
     * @param {number} index - Position index
     */
    function deletePosition(index) {
        const rows = document.querySelectorAll('.position-row');
        if (index >= 0 && index < rows.length) {
            rows[index].remove();
            
            // Update indices for remaining rows
            document.querySelectorAll('.delete-position-button').forEach((button, i) => {
                button.dataset.index = i;
            });
            
            // Add placeholder if no positions left
            if (document.querySelectorAll('.position-row').length === 0) {
                portfolioPositionsContainer.innerHTML = `
                    <div class="text-center text-muted">
                        No positions added yet. Enter symbols and click "Add Position".
                    </div>
                `;
            }
        }
    }
    
    /**
     * Update positions based on entered symbols
     */
    function updatePositionsFromSymbols() {
        if (!portfolioSymbolsInput || !portfolioPositionsContainer) return;
        
        const symbols = portfolioSymbolsInput.value.trim().split(',').map(s => s.trim().toUpperCase()).filter(s => s !== '');
        
        // Keep existing positions for symbols that still exist
        const existingPositions = [];
        const existingSymbols = [];
        
        document.querySelectorAll('.position-row').forEach(row => {
            const symbolInput = row.querySelector('.position-symbol');
            const sharesInput = row.querySelector('.position-shares');
            const priceInput = row.querySelector('.position-price');
            
            if (symbolInput && sharesInput && priceInput) {
                const symbol = symbolInput.value.trim().toUpperCase();
                const shares = parseFloat(sharesInput.value) || 0;
                const price = parseFloat(priceInput.value) || 0;
                
                if (symbol && shares > 0 && symbols.includes(symbol)) {
                    existingPositions.push({
                        symbol,
                        shares,
                        entryPrice: price
                    });
                    existingSymbols.push(symbol);
                }
            }
        });
        
        // Add empty positions for new symbols
        const newSymbols = symbols.filter(s => !existingSymbols.includes(s));
        const newPositions = newSymbols.map(symbol => ({
            symbol,
            shares: 0,
            entryPrice: 0
        }));
        
        // Update positions in UI
        const allPositions = [...existingPositions, ...newPositions];
        
        if (allPositions.length > 0) {
            let html = '';
            allPositions.forEach((position, index) => {
                html += createPositionRowHTML(position, index);
            });
            portfolioPositionsContainer.innerHTML = html;
            
            // Add event listeners for delete buttons
            document.querySelectorAll('.delete-position-button').forEach(button => {
                button.addEventListener('click', function() {
                    const index = parseInt(this.dataset.index);
                    deletePosition(index);
                });
            });
        } else {
            portfolioPositionsContainer.innerHTML = `
                <div class="text-center text-muted">
                    No positions added yet. Enter symbols and click "Add Position".
                </div>
            `;
        }
    }
    
    // Expose functions to global scope
    window.updatePortfolioData = updatePortfolioData;
    window.analyzePortfolio = analyzePortfolio;
});