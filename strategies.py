"""
Enhanced strategy functions for stock trading recommendations.
"""

import pandas as pd
import numpy as np

def extract_scalar(value):
    """Extract a scalar value from various data types."""
    try:
        if isinstance(value, (pd.Series, pd.DataFrame)):
            if len(value) > 0:
                return extract_scalar(value.iloc[0])
            return 0
        elif hasattr(value, 'item'):
            return value.item()
        elif isinstance(value, (list, np.ndarray)) and len(value) > 0:
            return extract_scalar(value[0])
        return float(value) if value is not None else 0
    except Exception as e:
        # In case of any error, return a neutral value
        return 0

def get_score_from_indicators(indicators_dict):
    """
    Calculate a composite score from multiple indicators.
    Higher score indicates stronger buy signal; lower score indicates stronger sell signal.
    Normalized to approximately -100 to 100 range.
    
    Args:
        indicators_dict: Dictionary containing all indicator values
        
    Returns:
        score: Composite score
    """
    try:
        # Extract scalar values for all indicators
        rsi = extract_scalar(indicators_dict.get('rsi', 50))
        bbp = extract_scalar(indicators_dict.get('bbp', 0.5))
        macd = extract_scalar(indicators_dict.get('macd', 0))
        stoch_k = extract_scalar(indicators_dict.get('stoch_k', 50))
        stoch_d = extract_scalar(indicators_dict.get('stoch_d', 50))
        adx = extract_scalar(indicators_dict.get('adx', 25))
        obv = extract_scalar(indicators_dict.get('obv', 50))
        ma_crossover = extract_scalar(indicators_dict.get('ma_crossover', 0))
        psar = extract_scalar(indicators_dict.get('psar', 0))
        
        # Calculate weighted scores for each indicator
        
        # RSI (0-100): below 30 is oversold (bullish), above 70 is overbought (bearish)
        # Normalize to -1 to 1 range: 0 = 0, 100 = -1, 0 = 1
        rsi_score = -1 * (rsi - 50) / 50  # Range: -1 to 1
        
        # BBP (0-1): 0 = lower band (bullish), 1 = upper band (bearish)
        # Normalize to -1 to 1 range: 0.5 = 0, 0 = 1, 1 = -1
        bbp_score = -1 * (bbp - 0.5) * 2  # Range: -1 to 1
        
        # MACD: Positive values are bullish, negative values are bearish
        # Normalize to -1 to 1 range using sigmoid function
        macd_score = np.tanh(macd * 5)  # Range: -1 to 1
        
        # Stochastic (0-100): below 20 is oversold (bullish), above 80 is overbought (bearish)
        # Use average of K and D lines
        stoch_avg = (stoch_k + stoch_d) / 2
        stoch_score = -1 * (stoch_avg - 50) / 50  # Range: -1 to 1
        
        # ADX (0-100): higher values indicate stronger trend
        # This is a trend strength indicator, not direction
        # Apply as a multiplier to other signals
        adx_factor = 0.5 + (adx / 100)  # Range: 0.5 to 1.5
        
        # OBV (0-100): trend confirmation
        # Normalized to -1 to 1: 50 = 0, 100 = 1, 0 = -1
        obv_score = (obv - 50) / 50  # Range: -1 to 1
        
        # MA Crossover: directly used as -1 to 1 signal
        
        # PSAR: directly used as -1 to 1 signal
        
        # Calculate weighted composite score
        # Weights are based on reliability and predictive power of each indicator
        score = (
            rsi_score * 20 +        # RSI: 20% weight
            bbp_score * 15 +        # BBP: 15% weight
            macd_score * 20 +       # MACD: 20% weight
            stoch_score * 10 +      # Stochastic: 10% weight
            obv_score * 5 +         # OBV: 5% weight
            ma_crossover * 15 +     # MA Crossover: 15% weight
            psar * 15               # PSAR: 15% weight
        )
        
        # Apply ADX factor as a multiplier to emphasize strong trends
        score = score * adx_factor
        
        # Ensure score is within a reasonable range (-100 to 100)
        score = np.clip(score * 100, -100, 100)
        
        return score
    except Exception as e:
        # If there's any error, use a simpler calculation based on RSI, BBP, and MACD
        try:
            rsi = extract_scalar(indicators_dict.get('rsi', 50))
            bbp = extract_scalar(indicators_dict.get('bbp', 0.5))
            macd = extract_scalar(indicators_dict.get('macd', 0))
            
            # Simple scoring
            rsi_score = -1 * (rsi - 50) / 50
            bbp_score = -1 * (bbp - 0.5) * 2
            macd_score = np.tanh(macd * 5)
            
            # Equal weights
            simple_score = (rsi_score + bbp_score + macd_score) / 3 * 100
            return np.clip(simple_score, -100, 100)
        except:
            # If even that fails, return neutral
            return 0

def calculate_percentile_rank(scores):
    """
    Calculate percentile ranks for a list of scores.
    
    Args:
        scores: List or Series of scores
        
    Returns:
        ranks: List of ranks from 1 to 10 (1 = top 10%, 10 = bottom 10%)
    """
    if len(scores) <= 1:
        return [5]  # Default to middle rank for single stock
    
    try:
        # Convert to pandas Series for easier percentile calculation
        scores_series = pd.Series(scores)
        
        # Calculate percentile rank (0-100) for each score
        percentiles = scores_series.rank(pct=True) * 100
        
        # Map percentiles to ranks 1-10
        # Higher score = better rank (1 is best)
        ranks = (100 - percentiles) // 10 + 1
        
        # Ensure ranks are integers 1-10
        ranks = ranks.astype(int)
        ranks = ranks.apply(lambda x: min(max(x, 1), 10))
        
        return ranks.tolist()
    except Exception as e:
        # In case of error, return middle ranks for all
        return [5] * len(scores)

def rank_to_recommendation(rank):
    """
    Convert rank (1-10) to recommendation text.
    
    Args:
        rank: Integer rank from 1 to 10
        
    Returns:
        recommendation: Text recommendation
    """
    if rank == 1:
        return "STRONG BUY"
    elif rank in (2, 3):
        return "BUY"
    elif rank == 4:
        return "WEAK BUY"
    elif rank in (5, 6):
        return "HOLD"
    elif rank == 7:
        return "WEAK SELL"
    elif rank in (8, 9):
        return "SELL"
    elif rank == 10:
        return "STRONG SELL"
    else:
        return "HOLD"

def get_manual_recommendation(rsi, bbp, macd):
    """
    Legacy function for backward compatibility.
    Get trading recommendation based on manual strategy.
    """
    signal = 0  # Possible values: -1 (sell), 0 (hold), 1 (buy)
    
    # Extract scalar values
    rsi_val = extract_scalar(rsi)
    bbp_val = extract_scalar(bbp)
    macd_val = extract_scalar(macd)
    
    # Rule 1: RSI conditions
    if rsi_val < 30:
        signal += 1  # Oversold, bullish signal
    elif rsi_val > 70:
        signal -= 1  # Overbought, bearish signal
    
    # Rule 2: Bollinger Band conditions
    if bbp_val < 0.2:
        signal += 1  # Price near lower band, bullish signal
    elif bbp_val > 0.8:
        signal -= 1  # Price near upper band, bearish signal
    
    # Rule 3: MACD conditions
    if macd_val > 0:
        signal += 1  # MACD above signal line, bullish signal
    else:
        signal -= 1  # MACD below signal line, bearish signal
    
    # Normalize signal to -1, 0, 1
    if signal >= 2:
        return "BUY (Strong)"
    elif signal == 1:
        return "BUY (Weak)"
    elif signal == 0:
        return "HOLD"
    elif signal == -1:
        return "SELL (Weak)"
    else:  # signal <= -2
        return "SELL (Strong)"