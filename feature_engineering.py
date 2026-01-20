import numpy as np
import pandas as pd


def add_cols(df):
    df = df.copy()
    
    df['log_open'] = np.log(df['Open'])
    df['log_close'] = np.log(df['Close'])
    df['return'] = np.log(df['Close'] / df['Close'].shift(1)).shift(1)
    
    for i in range(7):
        days_ago = f'close_{i+1}_prior'
        df[days_ago] = df['log_close'].shift(i + 1)
    
    df['sma_7'] = df['Close'].rolling(7).mean().shift(1)
    df['sma_30'] = df['Close'].rolling(30).mean().shift(1)
    df['sma_50'] = df['Close'].rolling(50).mean().shift(1)
    df['sma_200'] = df['Close'].rolling(200).mean().shift(1)
    
    df['dist_sma_7'] = (df['Close'] - df['sma_7']).shift(1)
    df['dist_sma_30'] = (df['Close'] - df['sma_30']).shift(1)
    df['dist_sma_50'] = (df['Close'] - df['sma_50']).shift(1)
    df['dist_sma_200'] = (df['Close'] - df['sma_200']).shift(1)
    
    df['momentum_7'] = df['return'].rolling(7).mean().shift(1)
    df['momentum_30'] = df['return'].rolling(30).mean().shift(1)
    df['momentum_50'] = df['return'].rolling(50).mean().shift(1)
    df['momentum_200'] = df['return'].rolling(200).mean().shift(1)
    
    df['volatility_7'] = df['return'].rolling(7).std().shift(1)
    df['volatility_30'] = df['return'].rolling(30).std().shift(1)
    df['volatility_50'] = df['return'].rolling(50).std().shift(1)
    df['volatility_200'] = df['return'].rolling(200).std().shift(1)
    
    df['volume_7'] = df['Volume'].rolling(7).mean().shift(1)
    df['volume_14'] = df['Volume'].rolling(14).mean().shift(1)
    df['volume_30'] = df['Volume'].rolling(30).mean().shift(1)
    df['volume_50'] = df['Volume'].rolling(50).mean().shift(1)
    
    df['Volume'] = df['Volume'].shift(1)
    
    df.dropna(inplace=True)
    
    return df


def get_feature_columns(include_target=False):
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
    correlations = df.corr()[target].drop(target)
    return correlations.abs().sort_values(ascending=False)


if __name__ == "__main__":
    from data_loader import load_all_years
    from data_processor import resample
    
    print("Testing feature_engineering module...")
    
    print("\nLoading and resampling Bitcoin data...")
    btc = load_all_years('bitcoin')
    btc_daily = resample(btc, '1d')
    
    print("\nAdding technical indicators...")
    btc_features = add_cols(btc_daily)
    print(f"Shape before features: {btc_daily.shape}")
    print(f"Shape after features: {btc_features.shape}")
    print(f"\nColumn names:\n{btc_features.columns.tolist()}")
    
    print("\nSample data with features:")
    print(btc_features.head())
    
    print("\nTop 10 features by correlation with log_close:")
    importance = calculate_feature_importance(btc_features)
    print(importance.head(10))
