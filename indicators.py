"""
Extended technical indicator functions for stock analysis.
"""

import pandas as pd
import numpy as np

def calculate_rsi(prices, period=14):
    """Calculate RSI for a price series."""
    # Calculate daily price changes
    delta = prices.diff()
    
    # Create copies for gain (positive changes) and loss (negative changes)
    gain = delta.copy()
    loss = delta.copy()
    
    # Set all negative changes to 0 in gain series and all positive to 0 in loss series
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    
    # Convert losses to positive values
    loss = -loss
    
    # Calculate average gain and loss over the period
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    # Calculate RS (Relative Strength) = avg_gain / avg_loss
    # Handle division by zero
    avg_loss_nonzero = avg_loss.replace(0, np.nan)
    rs = avg_gain / avg_loss_nonzero
    rs = rs.fillna(0)
    
    # Calculate RSI = 100 - (100 / (1 + RS))
    rsi_values = 100 - (100 / (1 + rs))
    rsi_values = rsi_values.fillna(50)  # Fill NaN values with neutral RSI
    
    return rsi_values

def calculate_bbp(prices, window=20):
    """Calculate Bollinger Band Percentage for a price series."""
    # Calculate rolling mean (SMA)
    rolling_mean = prices.rolling(window=window).mean()
    
    # Calculate rolling standard deviation
    rolling_std = prices.rolling(window=window).std()
    
    # Calculate upper and lower bands
    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)
    
    # Calculate Bollinger Band Percentage (BBP)
    # Handle division by zero and NaN values
    band_range = upper_band - lower_band
    band_range_nonzero = band_range.replace(0, np.nan)
    
    bbp = (prices - lower_band) / band_range_nonzero
    bbp = bbp.fillna(0.5)  # Fill NaN values with middle value
    
    return bbp

def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD histogram for a price series."""
    # Calculate fast and slow EMAs
    fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate MACD histogram (the actionable signal)
    macd_histogram = macd_line - signal_line
    
    return macd_histogram

def calculate_stochastic(high_prices, low_prices, close_prices, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator.
    
    Args:
        high_prices: Series of high prices
        low_prices: Series of low prices  
        close_prices: Series of closing prices
        k_period: K line period
        d_period: D line period
    
    Returns:
        k_line, d_line: Tuple of Series containing K and D values
    """
    try:
        # Make sure inputs are pandas Series
        if not isinstance(high_prices, pd.Series):
            high_prices = pd.Series(high_prices)
        if not isinstance(low_prices, pd.Series):
            low_prices = pd.Series(low_prices)
        if not isinstance(close_prices, pd.Series):
            close_prices = pd.Series(close_prices)
            
        # Calculate lowest low and highest high over k_period
        lowest_low = low_prices.rolling(window=k_period, min_periods=1).min()
        highest_high = high_prices.rolling(window=k_period, min_periods=1).max()
        
        # Calculate %K
        # %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
        range_diff = highest_high - lowest_low
        range_diff_nonzero = range_diff.replace(0, np.nan)
        k_line = 100 * ((close_prices - lowest_low) / range_diff_nonzero)
        k_line = k_line.fillna(50)  # Fill NaN values with middle value
        
        # Calculate %D (moving average of %K)
        d_line = k_line.rolling(window=d_period, min_periods=1).mean()
        d_line = d_line.fillna(50)  # Fill NaN values with middle value
        
        return k_line, d_line
    
    except Exception as e:
        # Return default values in case of error
        default_values = pd.Series(50, index=close_prices.index)
        return default_values, default_values

def calculate_adx(high_prices, low_prices, close_prices, period=14):
    """Calculate Average Directional Index (ADX).
    
    Args:
        high_prices: Series of high prices
        low_prices: Series of low prices
        close_prices: Series of closing prices
        period: Period for calculation
    
    Returns:
        adx_values: Series with ADX values
    """
    try:
        # Make sure inputs are pandas Series
        if not isinstance(high_prices, pd.Series):
            high_prices = pd.Series(high_prices)
        if not isinstance(low_prices, pd.Series):
            low_prices = pd.Series(low_prices)
        if not isinstance(close_prices, pd.Series):
            close_prices = pd.Series(close_prices)
            
        # Step 1: Calculate +DM, -DM
        high_diff = high_prices.diff()
        low_diff = low_prices.diff()
        
        # +DM: If current high - previous high > previous low - current low, then current high - previous high, else 0
        # -DM: If previous low - current low > current high - previous high, then previous low - current low, else 0
        plus_dm = pd.Series(0, index=high_diff.index)
        minus_dm = pd.Series(0, index=low_diff.index)
        
        # Calculate +DM and -DM
        plus_dm[((high_diff > 0) & (high_diff > low_diff.abs()))] = high_diff
        minus_dm[((low_diff < 0) & (low_diff.abs() > high_diff))] = -low_diff
        
        # Step 2: Calculate True Range (TR)
        tr1 = high_prices - low_prices  # Current high - current low
        tr2 = (high_prices - close_prices.shift(1)).abs()  # Current high - previous close
        tr3 = (low_prices - close_prices.shift(1)).abs()  # Current low - previous close
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)  # Take the maximum of the three
        
        # Step 3: Smooth the TR, +DM, and -DM using Wilder's smoothing
        # First value is a simple average
        smoothed_tr = tr.rolling(window=period, min_periods=1).mean()
        smoothed_plus_dm = plus_dm.rolling(window=period, min_periods=1).mean()
        smoothed_minus_dm = minus_dm.rolling(window=period, min_periods=1).mean()
        
        # Step 4: Calculate +DI and -DI
        # +DI = 100 * Smoothed +DM / Smoothed TR
        # -DI = 100 * Smoothed -DM / Smoothed TR
        smoothed_tr_nonzero = smoothed_tr.replace(0, np.nan)
        plus_di = 100 * (smoothed_plus_dm / smoothed_tr_nonzero)
        minus_di = 100 * (smoothed_minus_dm / smoothed_tr_nonzero)
        
        # Step 5: Calculate Directional Index (DX)
        # DX = 100 * |+DI - -DI| / (|+DI| + |-DI|)
        di_sum = plus_di.abs() + minus_di.abs()
        di_sum_nonzero = di_sum.replace(0, np.nan)
        dx = 100 * ((plus_di - minus_di).abs() / di_sum_nonzero)
        
        # Step 6: Calculate ADX (smoothed DX)
        adx = dx.rolling(window=period, min_periods=1).mean()
        adx = adx.fillna(25)  # Fill NaN values with modest trend strength
        
        return adx
    
    except Exception as e:
        # Return default values in case of error
        return pd.Series(25, index=close_prices.index)

def calculate_obv(close_prices, volumes):
    """Calculate On-Balance Volume (OBV).
    
    Args:
        close_prices: Series of closing prices
        volumes: Series of volume data
        
    Returns:
        obv_values: Series with OBV values
    """
    try:
        # Make sure inputs are pandas Series
        if not isinstance(close_prices, pd.Series):
            close_prices = pd.Series(close_prices)
        if not isinstance(volumes, pd.Series):
            volumes = pd.Series(volumes, index=close_prices.index)
            
        # Calculate price changes
        price_change = close_prices.diff()
        
        # Initialize OBV with first volume value
        obv = pd.Series(0, index=close_prices.index)
        
        # Calculate OBV values
        for i in range(1, len(close_prices)):
            if price_change.iloc[i] > 0:  # Price up, add volume
                obv.iloc[i] = obv.iloc[i-1] + volumes.iloc[i]
            elif price_change.iloc[i] < 0:  # Price down, subtract volume
                obv.iloc[i] = obv.iloc[i-1] - volumes.iloc[i]
            else:  # Price unchanged, OBV unchanged
                obv.iloc[i] = obv.iloc[i-1]
        
        # Normalize OBV to a 0-100 scale for easier comparison with other indicators
        # using a rolling window to determine local min/max
        window = 20
        obv_min = obv.rolling(window=window, min_periods=1).min()
        obv_max = obv.rolling(window=window, min_periods=1).max()
        
        # Avoid division by zero
        range_diff = obv_max - obv_min
        range_diff_nonzero = range_diff.replace(0, np.nan)
        
        normalized_obv = 100 * (obv - obv_min) / range_diff_nonzero
        normalized_obv = normalized_obv.fillna(50)  # Fill NaN values with middle value
        
        return normalized_obv
    
    except Exception as e:
        # Return default values in case of error
        return pd.Series(50, index=close_prices.index)

def calculate_ma_crossover(prices, fast_period=10, slow_period=50):
    """Calculate Moving Average Crossover signal.
    
    Args:
        prices: Series of price data
        fast_period: Period for fast moving average
        slow_period: Period for slow moving average
        
    Returns:
        crossover_signal: Series with crossover signal values (-1 to 1)
    """
    try:
        # Make sure input is a pandas Series
        if not isinstance(prices, pd.Series):
            prices = pd.Series(prices)
            
        # Calculate fast and slow moving averages
        fast_ma = prices.rolling(window=fast_period, min_periods=1).mean()
        slow_ma = prices.rolling(window=slow_period, min_periods=1).mean()
        
        # Calculate the difference between fast and slow MAs
        ma_diff = fast_ma - slow_ma
        
        # Normalize the difference to get a signal between -1 and 1
        # Calculate max absolute difference in the lookback period
        window = slow_period
        abs_diff = ma_diff.abs()
        max_diff = abs_diff.rolling(window=window, min_periods=1).max()
        
        # Avoid division by zero
        max_diff_nonzero = max_diff.replace(0, np.nan)
        
        # Normalize to -1 to 1 range
        normalized_diff = ma_diff / max_diff_nonzero
        normalized_diff = normalized_diff.fillna(0)  # Fill NaN values with neutral signal
        
        return normalized_diff
    
    except Exception as e:
        # Return default values in case of error
        return pd.Series(0, index=prices.index)

def calculate_psar(high_prices, low_prices, close_prices, af_start=0.02, af_max=0.2, af_step=0.02):
    """Calculate Parabolic SAR (Stop and Reverse).
    
    Args:
        high_prices: Series of high prices
        low_prices: Series of low prices
        close_prices: Series of closing prices
        af_start: Starting acceleration factor
        af_max: Maximum acceleration factor
        af_step: Acceleration factor step
        
    Returns:
        psar_signal: Series with PSAR signal values (-1 to 1)
    """
    try:
        # Make sure inputs are pandas Series
        if not isinstance(high_prices, pd.Series):
            high_prices = pd.Series(high_prices)
        if not isinstance(low_prices, pd.Series):
            low_prices = pd.Series(low_prices)
        if not isinstance(close_prices, pd.Series):
            close_prices = pd.Series(close_prices)
            
        # Check if we have enough data points
        if len(close_prices) < 3:
            return pd.Series(0, index=close_prices.index)
            
        # Initialize PSAR, direction, extreme point, and acceleration factor
        psar = pd.Series(index=close_prices.index)
        direction = pd.Series(index=close_prices.index)  # 1 for uptrend, -1 for downtrend
        ep = pd.Series(index=close_prices.index)  # Extreme point
        af = pd.Series(index=close_prices.index)  # Acceleration factor
        
        # Set initial values
        # Assume initial trend is up (can be enhanced with trend detection)
        direction.iloc[0] = 1
        psar.iloc[0] = low_prices.iloc[0]
        ep.iloc[0] = high_prices.iloc[0]
        af.iloc[0] = af_start
        
        # Calculate PSAR values
        for i in range(1, len(close_prices)):
            # Previous PSAR
            prev_psar = psar.iloc[i-1]
            
            # Calculate current PSAR
            if direction.iloc[i-1] == 1:  # Uptrend
                # PSAR = Previous PSAR + Previous AF * (Previous EP - Previous PSAR)
                psar.iloc[i] = prev_psar + af.iloc[i-1] * (ep.iloc[i-1] - prev_psar)
                
                # Make sure PSAR is not above the previous two lows
                if i >= 2:
                    psar.iloc[i] = min(psar.iloc[i], low_prices.iloc[i-1], low_prices.iloc[i-2])
                    
                # Check if trend reverses
                if psar.iloc[i] > low_prices.iloc[i]:
                    direction.iloc[i] = -1  # Reverse to downtrend
                    psar.iloc[i] = ep.iloc[i-1]  # Set PSAR to previous extreme point
                    ep.iloc[i] = low_prices.iloc[i]  # Set EP to current low
                    af.iloc[i] = af_start  # Reset AF
                else:
                    direction.iloc[i] = 1  # Continue uptrend
                    # Update extreme point and acceleration factor
                    if high_prices.iloc[i] > ep.iloc[i-1]:
                        ep.iloc[i] = high_prices.iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + af_step, af_max)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
            else:  # Downtrend
                # PSAR = Previous PSAR - Previous AF * (Previous PSAR - Previous EP)
                psar.iloc[i] = prev_psar - af.iloc[i-1] * (prev_psar - ep.iloc[i-1])
                
                # Make sure PSAR is not below the previous two highs
                if i >= 2:
                    psar.iloc[i] = max(psar.iloc[i], high_prices.iloc[i-1], high_prices.iloc[i-2])
                    
                # Check if trend reverses
                if psar.iloc[i] < high_prices.iloc[i]:
                    direction.iloc[i] = 1  # Reverse to uptrend
                    psar.iloc[i] = ep.iloc[i-1]  # Set PSAR to previous extreme point
                    ep.iloc[i] = high_prices.iloc[i]  # Set EP to current high
                    af.iloc[i] = af_start  # Reset AF
                else:
                    direction.iloc[i] = -1  # Continue downtrend
                    # Update extreme point and acceleration factor
                    if low_prices.iloc[i] < ep.iloc[i-1]:
                        ep.iloc[i] = low_prices.iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + af_step, af_max)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
        
        # Create a signal based on the relationship between price and PSAR
        psar_signal = pd.Series(index=close_prices.index)
        
        # Calculate the distance between price and PSAR as a ratio of price
        # Use try/except to handle division by zero or other errors
        try:
            distance = (close_prices - psar) / close_prices
            
            # Normalize to a -1 to 1 scale using a sigmoid-like function
            # Positive values indicate price above PSAR (bullish)
            # Negative values indicate price below PSAR (bearish)
            psar_signal = distance.apply(lambda x: 2 / (1 + np.exp(-10 * x)) - 1)
        except:
            # Default to direction as a fallback
            psar_signal = direction
        
        return psar_signal
    
    except Exception as e:
        # Return default values in case of error
        return pd.Series(0, index=close_prices.index)