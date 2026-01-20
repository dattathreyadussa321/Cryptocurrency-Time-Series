"""
Data processing module for cryptocurrency time series data.
"""

import pandas as pd
from config import DEFAULT_RESAMPLE_PERIOD


def resample(df, time_period=DEFAULT_RESAMPLE_PERIOD):
    """
    Resample minute-by-minute data to specified time period (daily, weekly, etc.).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with timestamp index and OHLCV data
    time_period : str
        Resampling period (e.g., '1d' for daily, '1w' for weekly)
        Default: '1d' (daily)
    
    Returns:
    --------
    pd.DataFrame
        Resampled DataFrame with Open, Close, and Volume columns
    """
    # Get the opening price (first price of the period)
    df1 = df[['Open']].resample(time_period).first()
    
    # Get the closing price (last price of the period)
    df2 = df[['Close']].resample(time_period).last()
    
    # Get the total volume (sum of all volumes in the period)
    df3 = df[['Volume']].resample(time_period).sum()
    
    # Combine the dataframes
    resampled_df = pd.concat([df1, df2, df3], axis=1)
    
    return resampled_df


def clean_data(df, drop_na=True):
    """
    Clean the dataframe by handling missing values and duplicates.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to clean
    drop_na : bool
        Whether to drop rows with NaN values (default: True)
    
    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame
    """
    # Remove duplicate indices
    df = df[~df.index.duplicated(keep='first')]
    
    # Sort by index
    df.sort_index(inplace=True)
    
    # Handle missing values
    if drop_na:
        df = df.dropna()
    else:
        # Forward fill missing values
        df = df.fillna(method='ffill')
    
    return df


def filter_by_date(df, start_date=None, end_date=None):
    """
    Filter dataframe by date range.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with timestamp index
    start_date : str or datetime
        Start date (inclusive)
    end_date : str or datetime
        End date (inclusive)
    
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame
    """
    if start_date is not None:
        df = df[df.index >= start_date]
    
    if end_date is not None:
        df = df[df.index <= end_date]
    
    return df


def get_data_summary(df):
    """
    Get summary statistics for the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to summarize
    
    Returns:
    --------
    dict
        Dictionary with summary statistics
    """
    summary = {
        'total_records': len(df),
        'date_range': {
            'start': df.index.min(),
            'end': df.index.max(),
        },
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
    }
    
    # Add price statistics if Close column exists
    if 'Close' in df.columns:
        summary['price_stats'] = {
            'min': float(df['Close'].min()),
            'max': float(df['Close'].max()),
            'mean': float(df['Close'].mean()),
            'std': float(df['Close'].std()),
        }
    
    return summary


if __name__ == "__main__":
    # Test the module
    from data_loader import get_data, load_all_years
    
    print("Testing data_processor module...")
    
    # Load data
    print("\nLoading Bitcoin data...")
    btc = load_all_years('bitcoin')
    
    # Resample to daily
    print("\nResampling to daily data...")
    btc_daily = resample(btc, '1d')
    print(btc_daily.head())
    print(f"Shape: {btc_daily.shape}")
    
    # Clean data
    print("\nCleaning data...")
    btc_clean = clean_data(btc_daily)
    print(f"Shape after cleaning: {btc_clean.shape}")
    
    # Get summary
    print("\nData summary:")
    summary = get_data_summary(btc_clean)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Filter by date
    print("\nFiltering data (2020-2021)...")
    btc_filtered = filter_by_date(btc_clean, '2020-01-01', '2021-12-31')
    print(f"Filtered shape: {btc_filtered.shape}")
    print(f"Date range: {btc_filtered.index.min()} to {btc_filtered.index.max()}")
