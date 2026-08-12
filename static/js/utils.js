/**
 * Utility functions for stock trading app
 */

/**
 * Time periods mapping to days
 */
const periods = {
    '1w': 7, 
    '1m': 30, 
    '6m': 180,
    '1y': 365, 
    '5y': 365 * 5
};

/**
 * Convert time period to days
 * @param {string} period - Time period code
 * @returns {number} - Number of days
 */
function getTimePeriodDays(period) {
    return periods[period] || 180; // Default to 6 months
}

/**
 * Get color for rank (1-10)
 * @param {number} rank - Rank from 1 to 10
 * @returns {Object} - Object with backgroundColor and textColor
 */
function getRankColors(rank) {
    const colors = {
        1: { bg: '#006400', text: 'white' },   // Dark Green
        2: { bg: '#28a745', text: 'white' },   // Green
        3: { bg: '#5cb85c', text: 'white' },   // Light Green
        4: { bg: '#8fca8f', text: 'black' },   // Very Light Green
        5: { bg: '#ffc107', text: 'black' },   // Yellow
        6: { bg: '#ffc107', text: 'black' },   // Yellow
        7: { bg: '#ffa6a6', text: 'black' },   // Very Light Red
        8: { bg: '#f86b7a', text: 'white' },   // Light Red
        9: { bg: '#dc3545', text: 'white' },   // Red
        10: { bg: '#a50000', text: 'white' }   // Dark Red
    };
    
    return colors[rank] || colors[5]; // Default to rank 5 (HOLD) if invalid
}

/**
 * Get rank icon prefix
 * @param {number} rank - Rank from 1 to 10
 * @returns {string} - Icon string
 */
function getRankIcon(rank) {
    const icons = {
        1: '↑↑',   // Strong Buy
        2: '↑',    // Buy
        3: '↗',    // Buy (weaker)
        4: '↗',    // Weak Buy
        5: '→',    // Hold
        6: '→',    // Hold
        7: '↘',    // Weak Sell
        8: '↓',    // Sell
        9: '↓',    // Sell
        10: '↓↓'   // Strong Sell
    };
    
    return icons[rank] || '→'; // Default to HOLD if invalid
}

/**
 * Format price with appropriate precision
 * @param {number} price - Price value
 * @returns {string} - Formatted price string
 */
function formatPrice(price) {
    if (price === null || price === undefined || isNaN(price)) {
        return 'N/A';
    }
    
    // Format based on magnitude
    if (price < 0.1) {
        return price.toFixed(6); // More precision for very small values
    } else if (price < 1) {
        return price.toFixed(4);
    } else if (price < 10) {
        return price.toFixed(3);
    } else if (price < 1000) {
        return price.toFixed(2);
    } else {
        return price.toFixed(0); // No decimals for large values
    }
}

/**
 * Format percentage with appropriate sign and precision
 * @param {number} percent - Percentage value
 * @returns {string} - Formatted percentage string
 */
function formatPercent(percent) {
    if (percent === null || percent === undefined || isNaN(percent)) {
        return 'N/A';
    }
    
    const sign = percent >= 0 ? '+' : '';
    return `${sign}${percent.toFixed(2)}%`;
}

/**
 * Format recommendation text (used for tooltips)
 * @param {Object} data - Period data containing indicators
 * @returns {string} - Formatted recommendation with indicator values
 */
function formatRecommendationDetails(data) {
    if (!data || !data.indicators) {
        return 'No data available';
    }
    
    // Get indicator values
    const rsi = data.indicators.rsi;
    const bbp = data.indicators.bbp;
    const macd = data.indicators.macd;
    
    // Format details
    return `${data.recommendation}
RSI: ${rsi !== null ? rsi.toFixed(2) : 'N/A'} 
BBP: ${bbp !== null ? bbp.toFixed(2) : 'N/A'}
MACD: ${macd !== null ? macd.toFixed(6) : 'N/A'}`;
}