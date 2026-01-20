"""
Time series models module for cryptocurrency price prediction.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')


def white_noise_model(df):
    """
    Fit a white noise model (ARIMA with all parameters = 0).
    Tests for complete randomness in the data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with log_close column
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with model results and RMSE scores
    """
    y = df['log_close']
    train_size = [len(y), 500]  # Reduced from [len(y), 500, 250, 100, 50] for faster execution
    
    aic_scores = []
    
    for size in train_size:
        model = ARIMA(endog=y.tail(size), order=(0, 0, 0))
        fitmodel = model.fit()
        rmse = np.sqrt(fitmodel.mse)
        aic_scores.append(pd.DataFrame(['white noise', size, (0, 0, 0), fitmodel.aic, rmse]).T)
    
    wn_df = pd.concat(aic_scores, axis=0)
    wn_df = wn_df.set_axis(['model type', 'train_size', 'order', 'AIC', 'RMSE'], axis=1)
    return_df = wn_df.sort_values(['train_size', 'RMSE'], ascending=[False, True]).reset_index(drop=True)
    
    return return_df


def random_walk_model(df, for_backtest=False):
    """
    Fit a random walk model (ARIMA with d=1, p=q=0).
    Checks for randomness in differences but not the series itself.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with log_close column
    for_backtest : bool
        If True, return fitted model; if False, return results DataFrame
    
    Returns:
    --------
    pd.DataFrame or ARIMAResults
        Results DataFrame or fitted model
    """
    y = df['log_close']
    train_size = [len(y), 500]  # Reduced for faster execution
    
    aic_scores = []
    
    for size in train_size:
        model = ARIMA(endog=y.tail(size), order=(0, 1, 0))
        fitmodel = model.fit()
        rmse = np.sqrt(fitmodel.mse)
        aic_scores.append(pd.DataFrame(['random walk', size, (0, 1, 0), fitmodel.aic, rmse]).T)
    
    rw_df = pd.concat(aic_scores, axis=0)
    rw_df = rw_df.set_axis(['model type', 'train_size', 'order', 'AIC', 'RMSE'], axis=1)
    return_df = rw_df.sort_values(['train_size', 'RMSE'], ascending=[False, True]).reset_index(drop=True)
    
    if for_backtest:
        best_model = ARIMA(endog=y.tail(return_df['train_size'][0]), order=(0, 1, 0))
        best_fit_model = best_model.fit()
        return best_fit_model
    else:
        return return_df


def random_walk_drift_model(df):
    """
    Fit a random walk with drift model (ARIMA with d=1, p=q=0, trend='c').
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with log_close column
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with model results and RMSE scores
    """
    y = df['log_close']
    train_size = [len(y), 500]  # Reduced for faster execution
    
    aic_scores = []
    
    for size in train_size:
        model = ARIMA(endog=y.tail(size), order=(0, 1, 0))
        fitmodel = model.fit()
        rmse = np.sqrt(fitmodel.mse)
        aic_scores.append(pd.DataFrame(['random walk drift', size, (0, 1, 0), fitmodel.aic, rmse]).T)
    
    rwd_df = pd.concat(aic_scores, axis=0)
    rwd_df = rwd_df.set_axis(['model type', 'train_size', 'order', 'AIC', 'RMSE'], axis=1)
    return_df = rwd_df.sort_values(['train_size', 'RMSE'], ascending=[False, True]).reset_index(drop=True)
    
    return return_df


def ARMA_model(df):
    """
    Fit ARMA models with different p and q values.
    Autoregressive Moving Average without differencing.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with log_close column
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with model results and RMSE scores
    """
    y = df['log_close']
    aic_scores = []
    train_size = [len(y)]  # Only use full dataset for faster execution
    
    for p in range(3):  # Reduced from 6 to 3
        for q in range(3):  # Reduced from 6 to 3
            for size in train_size:
                try:
                    model = ARIMA(endog=y.tail(size), order=(p, 0, q))
                    fitmodel = model.fit()
                    rmse = np.sqrt(fitmodel.mse)
                    aic_scores.append(pd.DataFrame(['ARMA', size, (p, 0, q), fitmodel.aic, rmse]).T)
                except:
                    continue
    
    arma_df = pd.concat(aic_scores, axis=0)
    arma_df = arma_df.set_axis(['model type', 'train_size', 'order', 'AIC', 'RMSE'], axis=1)
    return_df = arma_df.sort_values(['train_size', 'RMSE'], ascending=[False, True]).reset_index(drop=True)
    
    return return_df


def ARIMA_model(df):
    """
    Fit ARIMA models with different p, d, and q values.
    Autoregressive Integrated Moving Average with differencing.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with log_close column
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with model results and RMSE scores
    """
    y = df['log_close']
    aic_scores = []
    train_size = [len(y)]  # Only use full dataset for faster execution
    
    for p in range(3):
        for d in range(1, 4):
            for q in range(3):
                for size in train_size:
                    try:
                        model = ARIMA(endog=y.tail(size), order=(p, d, q))
                        fitmodel = model.fit()
                        rmse = np.sqrt(fitmodel.mse)
                        aic_scores.append(pd.DataFrame(['ARIMA', size, (p, d, q), fitmodel.aic, rmse]).T)
                    except:
                        continue
    
    arima_df = pd.concat(aic_scores, axis=0)
    arima_df = arima_df.set_axis(['model type', 'train_size', 'order', 'AIC', 'RMSE'], axis=1)
    return_df = arima_df.sort_values(['train_size', 'RMSE'], ascending=[False, True]).reset_index(drop=True)
    
    return return_df


def ARIMAX_model(df, test_size=3):
    """
    Fit ARIMAX models with exogenous variables.
    Extended ARIMA with additional predictor variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and log_close column
    test_size : int
        Not used, kept for compatibility
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with model results and RMSE scores
    """
    df_copy = df.drop(['Close', 'log_close_diff'], axis=1, errors='ignore')
    
    x = df_copy.drop('log_close', axis=1)
    y = df_copy['log_close']
    
    aic_scores = []
    train_size = [len(y)]  # Only use full dataset for faster execution
    
    for p in range(3):
        for d in range(1, 4):
            for q in range(3):
                for size in train_size:
                    try:
                        model = ARIMA(endog=y.tail(size), exog=x.tail(size), order=(p, d, q))
                        fitmodel = model.fit()
                        rmse = np.sqrt(fitmodel.mse)
                        aic_scores.append(pd.DataFrame(['ARIMAX', size, (p, d, q), fitmodel.aic, rmse]).T)
                    except:
                        continue
    
    arimax_df = pd.concat(aic_scores, axis=0)
    arimax_df = arimax_df.set_axis(['model type', 'train_size', 'order', 'AIC', 'RMSE'], axis=1)
    return_df = arimax_df.sort_values(['train_size', 'RMSE'], ascending=[False, True]).reset_index(drop=True)
    
    return return_df


def compare_models(df):
    """
    Run all models and compare their performance.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and log_close column
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with all model results sorted by RMSE
    """
    print("Running Random Walk model...")
    rw = random_walk_model(df)
    
    print("Running Random Walk with Drift model...")
    rwd = random_walk_drift_model(df)
    
    print("Running ARIMA models...")
    arima = ARIMA_model(df)
    
    print("Running ARIMAX models...")
    arimax = ARIMAX_model(df)
    
    # Combine all models
    all_models = pd.concat([rw, rwd, arima, arimax], axis=0)
    all_models = all_models.sort_values(['train_size', 'RMSE'], ascending=[False, True])
    
    return all_models


def fit_best_model(df, order=(2, 1, 2)):
    """
    Fit the best performing model.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with log_close column
    order : tuple
        ARIMA order (p, d, q)
    
    Returns:
    --------
    ARIMAResults
        Fitted ARIMA model
    """
    y = df['log_close']
    model = ARIMA(endog=y, order=order)
    fitted_model = model.fit()
    
    return fitted_model


if __name__ == "__main__":
    # Test the module
    from data_loader import load_all_years
    from data_processor import resample, filter_by_date
    from feature_engineering import add_cols
    
    print("Testing time_series_models module...")
    
    # Load and prepare data
    print("\nLoading data...")
    btc = load_all_years('bitcoin')
    btc_daily = resample(btc, '1d')
    btc_features = add_cols(btc_daily)
    btc_train = filter_by_date(btc_features, end_date='2021-10-01')
    
    print(f"Training data shape: {btc_train.shape}")
    
    # Test ARIMA model
    print("\nTesting ARIMA model (this may take a minute)...")
    arima_results = ARIMA_model(btc_train)
    print("\nTop 5 ARIMA models:")
    print(arima_results.head())
    
    # Fit best model
    print("\nFitting best model...")
    best_model = fit_best_model(btc_train, order=(2, 1, 2))
    print(best_model.summary())
