"""
Feature engineering module for creating technical indicators.
"""

import numpy as np
import pandas as pd


def add_cols(df):
    """
    Add technical indicators and features to the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with Open, Close, and Volume columns
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional technical indicator columns
    """
    df = df.copy()
    
    # Logging the opening and closing prices
    # Helps normalize prices and make them more normally distributed
    df['log_open'] = np.log(df['Open'])
    df['log_close'] = np.log(df['Close'])
    
    # Log returns (shifted one day)
    # Makes returns additive rather than multiplicative
    df['return'] = np.log(df['Close'] / df['Close'].shift(1)).shift(1)
    
    # Previous close prices (last 7 days)
    # Knowing recent closing prices helps predict next day's price
    for i in range(7):
        days_ago = f'close_{i+1}_prior'
        df[days_ago] = df['log_close'].shift(i + 1)
    
    # Simple Moving Averages (SMA)
    # Smooths the data and helps determine price direction
    df['sma_7'] = df['Close'].rolling(7).mean().shift(1)
    df['sma_30'] = df['Close'].rolling(30).mean().shift(1)
    df['sma_50'] = df['Close'].rolling(50).mean().shift(1)
    df['sma_200'] = df['Close'].rolling(200).mean().shift(1)
    
    # Distance from SMA
    # Helps identify possible reversals in price direction
    df['dist_sma_7'] = (df['Close'] - df['sma_7']).shift(1)
    df['dist_sma_30'] = (df['Close'] - df['sma_30']).shift(1)
    df['dist_sma_50'] = (df['Close'] - df['sma_50']).shift(1)
    df['dist_sma_200'] = (df['Close'] - df['sma_200']).shift(1)
    
    # Momentum indicators
    # Shows how strong an asset is moving in a particular direction
    df['momentum_7'] = df['return'].rolling(7).mean().shift(1)
    df['momentum_30'] = df['return'].rolling(30).mean().shift(1)
    df['momentum_50'] = df['return'].rolling(50).mean().shift(1)
    df['momentum_200'] = df['return'].rolling(200).mean().shift(1)
    
    # Volatility indicators
    # Shows the degree to which prices move (higher volatility = higher risk)
    df['volatility_7'] = df['return'].rolling(7).std().shift(1)
    df['volatility_30'] = df['return'].rolling(30).std().shift(1)
    df['volatility_50'] = df['return'].rolling(50).std().shift(1)
    df['volatility_200'] = df['return'].rolling(200).std().shift(1)
    
    # Volume indicators
    # Volume can confirm momentum or alert for possible reversal
    df['volume_7'] = df['Volume'].rolling(7).mean().shift(1)
    df['volume_14'] = df['Volume'].rolling(14).mean().shift(1)
    df['volume_30'] = df['Volume'].rolling(30).mean().shift(1)
    df['volume_50'] = df['Volume'].rolling(50).mean().shift(1)
    
    # Shift volume to prevent data leakage
    df['Volume'] = df['Volume'].shift(1)
    
    # Drop rows with NaN values (created by rolling windows and shifts)
    df.dropna(inplace=True)
    
    return df


def get_feature_columns(include_target=False):
    """
    Get list of feature column names.
    
    Parameters:
    -----------
    include_target : bool
        Whether to include the target column (log_close)
    
    Returns:
    --------
    list
        List of feature column names
    """
    features = [
        'log_open', 'return',
        'close_1_prior', 'close_2_prior', 'close_3_prior', 'close_4_prior',
        'close_5_prior', 'close_6_prior', 'close_7_prior',
        'sma_7', 'sma_30', 'sma_50', 'sma_200',
        'dist_sma_7', 'dist_sma_30', 'dist_sma_50', 'dist_sma_200',
        'momentum_7', 'momentum_30', 'momentum_50', 'momentum_200',
        'volatility_7', 'volatility_30', 'volatility_50', 'volatility_200',
        'volume_7', 'volume_14', 'volume_30', 'volume_50',
        'Open', 'Volume'
    ]
    
    if include_target:
        features.append('log_close')
    
    return features


def calculate_feature_importance(df, target='log_close'):
    """
    Calculate correlation of features with target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features
    target : str
        Target column name
    
    Returns:
    --------
    pd.Series
        Feature correlations sorted by absolute value
    """
    correlations = df.corr()[target].drop(target)
    return correlations.abs().sort_values(ascending=False)


if __name__ == "__main__":
    # Test the module
    from data_loader import load_all_years
    from data_processor import resample
    
    print("Testing feature_engineering module...")
    
    # Load and resample data
    print("\nLoading and resampling Bitcoin data...")
    btc = load_all_years('bitcoin')
    btc_daily = resample(btc, '1d')
    
    # Add features
    print("\nAdding technical indicators...")
    btc_features = add_cols(btc_daily)
    print(f"Shape before features: {btc_daily.shape}")
    print(f"Shape after features: {btc_features.shape}")
    print(f"\nColumn names:\n{btc_features.columns.tolist()}")
    
    # Show sample data
    print("\nSample data with features:")
    print(btc_features.head())
    
    # Calculate feature importance
    print("\nTop 10 features by correlation with log_close:")
    importance = calculate_feature_importance(btc_features)
    print(importance.head(10))
