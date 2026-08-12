/**
 * Configuration management for stock trading app
 */

// Global configuration object
let appConfig = {
    settings: {
        refreshInterval: 1800, // 30 minutes in seconds
        hideNonBuys: true,
        hideRanksAbove: 7
    },
    watchlist: {
        symbols: "AAPL,MSFT,GOOGL,AMZN,NVDA,AMD,INTC,TSM,CRM,ADBE,JPM,BAC,GS,V,MA,BLK,JNJ,PFE,MRNA,UNH,CVS,COST,WMT,TGT,MCD,SBUX,NKE,CAT,DE,BA,GE,XOM,CVX,NEE,TSLA,F,GM,T,VZ,NFLX,DIS,ETSY,SHOP,BABA,COIN,ABNB,HOOD,PLTR,U,SNAP,PINS,BRK-B,BRK-A,SPY,QQQ,DIA,IWM,VTI,XLF,XLK,XLE,XLV,XLI,XLP,XLY,EFA,EEM,FXI,EWJ,TLT,HYG,AGG,GLD,SLV,USO,VXX,SH,ARKK,ICLN,SOXX,HACK,SMH"
    },
    portfolio: {
        symbols: "",
        positions: []
    }
};

// Wait for document to load
document.addEventListener("DOMContentLoaded", function() {
    // Get configuration elements
    const editConfigButton = document.getElementById('edit-config-button');
    const saveConfigButton = document.getElementById('save-config-button');
    const updateWatchlistButton = document.getElementById('update-watchlist-button');
    const symbolsInput = document.getElementById('symbols');
    
    // Configuration displays
    const refreshIntervalDisplay = document.getElementById('refresh-interval-display');
    const hideNonBuysDisplay = document.getElementById('hide-non-buys-display');
    const hideRanksDisplay = document.getElementById('hide-ranks-display');
    
    // Configuration form fields
    const refreshIntervalInput = document.getElementById('refreshInterval');
    const hideNonBuysInput = document.getElementById('hideNonBuys');
    const hideRanksAboveInput = document.getElementById('hideRanksAbove');
    const watchlistSymbolsInput = document.getElementById('watchlistSymbols');
    
    // Initialize
    loadConfig();
    
    // Set up event listeners
    if (editConfigButton) {
        editConfigButton.addEventListener('click', function() {
            openConfigModal();
        });
    }
    
    if (saveConfigButton) {
        saveConfigButton.addEventListener('click', function() {
            saveConfig();
        });
    }
    
    if (updateWatchlistButton) {
        updateWatchlistButton.addEventListener('click', function() {
            updateWatchlist();
        });
    }
    
    /**
     * Load configuration from server
     */
    function loadConfig() {
        console.log("Loading configuration...");
        
        // Update the symbols input right away with default values
        if (symbolsInput) {
            symbolsInput.value = appConfig.watchlist.symbols;
            console.log("Initialized symbols input with:", appConfig.watchlist.symbols);
        }
        
        fetch('/api/config/get')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Server error: ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('Loaded configuration:', data);
                appConfig = data;
                
                // Update UI with loaded config
                updateConfigDisplay();
                
                // Update the symbols input
                if (symbolsInput) {
                    symbolsInput.value = appConfig.watchlist.symbols;
                    console.log("Updated symbols input with:", appConfig.watchlist.symbols);
                }
                
                // Update refresh interval
                if (typeof window.updateRefreshInterval === 'function') {
                    window.updateRefreshInterval(appConfig.settings.refreshInterval);
                }
                
                // Notify portfolio.js
                if (typeof window.updatePortfolioData === 'function') {
                    window.updatePortfolioData(appConfig.portfolio);
                }
                
                // Make sure analysis runs
                if (typeof window.handleAnalyzeStocks === 'function') {
                    console.log("Triggering analysis after config load");
                    setTimeout(() => window.handleAnalyzeStocks(), 500);
                }
            })
            .catch(error => {
                console.error('Error loading configuration:', error);
                // If server config fails, use default config
                updateConfigDisplay();
                
                // Set default symbols
                if (symbolsInput) {
                    symbolsInput.value = appConfig.watchlist.symbols;
                    console.log("Using default symbols after load error:", appConfig.watchlist.symbols);
                }
                
                // Make sure analysis runs even after error
                if (typeof window.handleAnalyzeStocks === 'function') {
                    console.log("Triggering analysis after config load error");
                    setTimeout(() => window.handleAnalyzeStocks(), 500);
                }
            });
    }
    
    /**
     * Save configuration to server
     */
    function saveConfig() {
        // Gather configuration from form
        const newConfig = {
            settings: {
                refreshInterval: parseInt(refreshIntervalInput.value) * 60, // Convert minutes to seconds
                hideNonBuys: hideNonBuysInput.checked,
                hideRanksAbove: parseInt(hideRanksAboveInput.value)
            },
            watchlist: {
                symbols: watchlistSymbolsInput.value.trim()
            },
            portfolio: appConfig.portfolio // Keep the portfolio data unchanged
        };
        
        // Send to server
        fetch('/api/config/save', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(newConfig)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Configuration saved:', data);
            appConfig = data;
            
            // Update UI
            updateConfigDisplay();
            
            // Update the symbols input
            if (symbolsInput) {
                symbolsInput.value = appConfig.watchlist.symbols;
            }
            
            // Update refresh interval
            if (typeof window.updateRefreshInterval === 'function') {
                window.updateRefreshInterval(appConfig.settings.refreshInterval);
            }
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('configModal'));
            if (modal) {
                modal.hide();
            }
            
            // Re-run analysis with new settings
            if (typeof window.handleAnalyzeStocks === 'function') {
                window.handleAnalyzeStocks();
            }
        })
        .catch(error => {
            console.error('Error saving configuration:', error);
            alert('Failed to save configuration. Please try again.');
        });
    }
    
    /**
     * Update watchlist from input field
     */
    function updateWatchlist() {
        if (!symbolsInput) return;
        
        const symbols = symbolsInput.value.trim();
        appConfig.watchlist.symbols = symbols;
        
        // Save updated watchlist
        fetch('/api/config/save', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(appConfig)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Watchlist updated:', data);
            
            // Re-run analysis with new watchlist
            if (typeof window.handleAnalyzeStocks === 'function') {
                window.handleAnalyzeStocks();
            }
        })
        .catch(error => {
            console.error('Error updating watchlist:', error);
            alert('Failed to update watchlist. Please try again.');
        });
    }
    
    /**
     * Open configuration modal and populate fields
     */
    function openConfigModal() {
        // Populate form fields
        refreshIntervalInput.value = Math.floor(appConfig.settings.refreshInterval / 60); // Convert seconds to minutes
        hideNonBuysInput.checked = appConfig.settings.hideNonBuys;
        hideRanksAboveInput.value = appConfig.settings.hideRanksAbove;
        watchlistSymbolsInput.value = appConfig.watchlist.symbols;
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('configModal'));
        modal.show();
    }
    
    /**
     * Update configuration display in UI
     */
    function updateConfigDisplay() {
        if (refreshIntervalDisplay) {
            const minutes = Math.floor(appConfig.settings.refreshInterval / 60);
            refreshIntervalDisplay.textContent = `${minutes} minute${minutes !== 1 ? 's' : ''}`;
        }
        
        if (hideNonBuysDisplay) {
            hideNonBuysDisplay.textContent = appConfig.settings.hideNonBuys ? 'Yes' : 'No';
        }
        
        if (hideRanksDisplay) {
            if (appConfig.settings.hideRanksAbove === 0) {
                hideRanksDisplay.textContent = 'Show all';
            } else {
                hideRanksDisplay.textContent = appConfig.settings.hideRanksAbove;
            }
        }
    }
    
    // Expose functions to global scope
    window.appConfig = appConfig;
    window.loadConfig = loadConfig;
    window.saveConfig = saveConfig;
});