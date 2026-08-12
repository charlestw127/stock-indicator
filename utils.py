"""
Utility functions for stock trading application.
"""

import datetime as dt

def time_period_to_start_date(time_period, end_date):
    """Convert time period to a start date."""
    periods = {
        '1d': dt.timedelta(days=5),  # a few bars so short indicators have data
        '1w': dt.timedelta(weeks=1),
        '1m': dt.timedelta(days=30),
        '6m': dt.timedelta(days=180),
        '1y': dt.timedelta(days=365)
    }
    
    delta = periods.get(time_period, dt.timedelta(days=180))  # Default to 6 months
    return end_date - delta

def format_percentage(value, include_sign=True):
    """Format a percentage value."""
    if value is None:
        return "N/A"
    
    sign = "+" if include_sign and value > 0 else ""
    return f"{sign}{value:.2f}%"

def format_price(value, default="N/A"):
    """Format a price value."""
    if value is None:
        return default
    
    # Format based on magnitude
    if value < 0.1:
        return f"${value:.6f}"  # More precision for very small values
    elif value < 1:
        return f"${value:.4f}"
    elif value < 10:
        return f"${value:.3f}"
    elif value < 1000:
        return f"${value:.2f}"
    else:
        return f"${int(value):,}"  # No decimals for large values

def parse_date(date_str):
    """Parse a date string to a datetime object."""
    try:
        return dt.datetime.strptime(date_str, "%Y-%m-%d")
    except:
        try:
            return dt.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        except:
            return None