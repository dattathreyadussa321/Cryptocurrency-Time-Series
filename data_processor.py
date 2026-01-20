import pandas as pd
from config import DEFAULT_RESAMPLE_PERIOD


def resample(df, time_period=DEFAULT_RESAMPLE_PERIOD):
    df1 = df[['Open']].resample(time_period).first()
    df2 = df[['Close']].resample(time_period).last()
    df3 = df[['Volume']].resample(time_period).sum()
    resampled_df = pd.concat([df1, df2, df3], axis=1)
    return resampled_df


def clean_data(df, drop_na=True):
    df = df[~df.index.duplicated(keep='first')]
    df.sort_index(inplace=True)
    
    if drop_na:
        df = df.dropna()
    else:
        df = df.fillna(method='ffill')
    
    return df


def filter_by_date(df, start_date=None, end_date=None):
    if start_date is not None:
        df = df[df.index >= start_date]
    
    if end_date is not None:
        df = df[df.index <= end_date]
    
    return df


def get_data_summary(df):
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
    
    if 'Close' in df.columns:
        summary['price_stats'] = {
            'min': float(df['Close'].min()),
            'max': float(df['Close'].max()),
            'mean': float(df['Close'].mean()),
            'std': float(df['Close'].std()),
        }
    
    return summary


if __name__ == "__main__":
    from data_loader import get_data, load_all_years
    
    print("Testing data_processor module...")
    
    print("\nLoading Bitcoin data...")
    btc = load_all_years('bitcoin')
    
    print("\nResampling to daily data...")
    btc_daily = resample(btc, '1d')
    print(btc_daily.head())
    print(f"Shape: {btc_daily.shape}")
    
    print("\nCleaning data...")
    btc_clean = clean_data(btc_daily)
    print(f"Shape after cleaning: {btc_clean.shape}")
    
    print("\nData summary:")
    summary = get_data_summary(btc_clean)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\nFiltering data (2020-2021)...")
    btc_filtered = filter_by_date(btc_clean, '2020-01-01', '2021-12-31')
    print(f"Filtered shape: {btc_filtered.shape}")
    print(f"Date range: {btc_filtered.index.min()} to {btc_filtered.index.max()}")
