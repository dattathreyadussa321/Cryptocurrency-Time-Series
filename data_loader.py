"""
Data loading module for cryptocurrency time series data.
"""

import pandas as pd
import os
from config import DATA_DIR, DATA_COLUMNS


def get_data(coin='bitcoin', year=2017):
    """
    Load cryptocurrency data for a specific coin and year.
    
    Parameters:
    -----------
    coin : str
        Cryptocurrency name (default: 'bitcoin')
    year : int
        Year of data to load (default: 2017)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with timestamp index and cryptocurrency data
    """
    filepath = os.path.join(DATA_DIR, f'{coin}_{year}.csv')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df = df[DATA_COLUMNS]
    
    return df


def load_all_years(coin='bitcoin', years=None):
    """
    Load and concatenate cryptocurrency data for multiple years.
    
    Parameters:
    -----------
    coin : str
        Cryptocurrency name (default: 'bitcoin')
    years : list
        List of years to load (default: [2017, 2018, 2019, 2020, 2021])
    
    Returns:
    --------
    pd.DataFrame
        Concatenated DataFrame with all years of data
    """
    if years is None:
        years = [2017, 2018, 2019, 2020, 2021]
    
    dataframes = []
    for year in years:
        try:
            df = get_data(coin, year)
            dataframes.append(df)
            print(f"Loaded {coin} data for {year}: {len(df)} records")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue
    
    if not dataframes:
        raise ValueError(f"No data files found for {coin}")
    
    combined_df = pd.concat(dataframes, axis=0)
    print(f"\nTotal records loaded: {len(combined_df)}")
    
    return combined_df


def get_available_data():
    """
    Get a list of available cryptocurrency data files.
    
    Returns:
    --------
    dict
        Dictionary with coin names as keys and lists of available years as values
    """
    available_data = {}
    
    if not os.path.exists(DATA_DIR):
        return available_data
    
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.csv'):
            # Parse filename: coin_year.csv
            parts = filename.replace('.csv', '').split('_')
            if len(parts) == 2:
                coin, year = parts
                try:
                    year = int(year)
                    if coin not in available_data:
                        available_data[coin] = []
                    available_data[coin].append(year)
                except ValueError:
                    continue
    
    # Sort years for each coin
    for coin in available_data:
        available_data[coin].sort()
    
    return available_data


if __name__ == "__main__":
    # Test the module
    print("Testing data_loader module...")
    print("\nAvailable data:")
    print(get_available_data())
    
    print("\nLoading Bitcoin 2017 data...")
    btc_2017 = get_data('bitcoin', 2017)
    print(btc_2017.head())
    print(f"Shape: {btc_2017.shape}")
    
    print("\nLoading all Bitcoin years...")
    btc_all = load_all_years('bitcoin')
    print(btc_all.head())
    print(f"Shape: {btc_all.shape}")
    print(f"Date range: {btc_all.index.min()} to {btc_all.index.max()}")
