"""
Forecasting module for cryptocurrency price prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')


def predictions(df, start_date='2021-10-01', days_to_predict=30, order=(2, 1, 2)):
    """
    Generate price predictions for multiple days ahead.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with log_close column
    start_date : str
        Starting date for predictions (format: 'YYYY-MM-DD')
    days_to_predict : int
        Number of days to generate predictions for
    order : tuple
        ARIMA order (p, d, q)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with actual prices and predictions
    """
    date = datetime.strptime(start_date, '%Y-%m-%d')
    d = pd.date_range(start=date, end=date + timedelta(days_to_predict))
    
    data = df['log_close']
    predictions_list = []
    
    print(f"Generating predictions for {len(d)} days...")
    
    for idx, pred_date in enumerate(d):
        # Use data up to the current date
        y = data[data.index < pred_date]
        
        if len(y) < 10:  # Need minimum data points
            continue
        
        try:
            # Fit model and generate forecast
            model = ARIMA(endog=y, order=order)
            fitmodel = model.fit()
            y_pred = fitmodel.forecast(3)  # Forecast 3 days ahead
            
            # Convert from log space back to price
            pred_today = np.exp(y_pred.values[0]) if len(y_pred) > 0 else np.nan
            pred_tomorrow = np.exp(y_pred.values[1]) if len(y_pred) > 1 else np.nan
            pred_2_days = np.exp(y_pred.values[2]) if len(y_pred) > 2 else np.nan
            
            predictions_list.append([pred_date, pred_today, pred_tomorrow, pred_2_days])
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(d)} days...")
        except Exception as e:
            print(f"  Warning: Could not generate prediction for {pred_date}: {e}")
            predictions_list.append([pred_date, np.nan, np.nan, np.nan])
    
    # Create predictions dataframe
    preds = pd.DataFrame(
        predictions_list,
        columns=['timestamp', 'pred_today', 'pred_tomorrow', 'pred_2_days']
    ).set_index('timestamp')
    
    # Merge with original dataframe
    df2 = df.merge(preds, left_index=True, right_index=True, how='left')
    
    print(f"Predictions generated successfully!")
    
    return df2


def predict_next_days(df, days_ahead=7, order=(2, 1, 2)):
    """
    Predict prices for the next N days from the last date in the data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with log_close column
    days_ahead : int
        Number of days to predict into the future
    order : tuple
        ARIMA order (p, d, q)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with predicted prices and dates
    """
    y = df['log_close']
    
    # Fit model on all available data
    model = ARIMA(endog=y, order=order)
    fitted_model = model.fit()
    
    # Generate forecast
    forecast = fitted_model.forecast(steps=days_ahead)
    
    # Create date range for predictions
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead)
    
    # Convert from log space to prices
    predicted_prices = np.exp(forecast.values)
    
    # Create results dataframe
    results = pd.DataFrame({
        'date': future_dates,
        'predicted_price': predicted_prices,
        'log_predicted_price': forecast.values
    })
    results.set_index('date', inplace=True)
    
    return results


def calculate_prediction_metrics(df, actual_col='Close', pred_col='pred_today'):
    """
    Calculate metrics for prediction accuracy.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with actual and predicted prices
    actual_col : str
        Column name for actual prices
    pred_col : str
        Column name for predicted prices
    
    Returns:
    --------
    dict
        Dictionary with RMSE, MAE, and MAPE metrics
    """
    # Filter out NaN values
    valid_data = df[[actual_col, pred_col]].dropna()
    
    if len(valid_data) == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'count': 0}
    
    actual = valid_data[actual_col]
    predicted = valid_data[pred_col]
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'count': len(valid_data)
    }


def get_trading_signals(df, pred_col='pred_tomorrow'):
    """
    Generate buy/sell/hold signals based on predictions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with Close and predicted prices
    pred_col : str
        Column name for predicted prices
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with trading signals
    """
    df = df.copy()
    
    # Calculate expected price change
    df['expected_change'] = df[pred_col] - df['Close']
    df['expected_change_pct'] = (df['expected_change'] / df['Close']) * 100
    
    # Generate signals
    # Buy if expected increase > 1%, Sell if expected decrease > 1%, otherwise Hold
    df['signal'] = 'HOLD'
    df.loc[df['expected_change_pct'] > 1, 'signal'] = 'BUY'
    df.loc[df['expected_change_pct'] < -1, 'signal'] = 'SELL'
    
    return df


if __name__ == "__main__":
    # Test the module
    from data_loader import load_all_years
    from data_processor import resample
    from feature_engineering import add_cols
    
    print("Testing forecasting module...")
    
    # Load and prepare data
    print("\nLoading data...")
    btc = load_all_years('bitcoin')
    btc_daily = resample(btc, '1d')
    btc_features = add_cols(btc_daily)
    
    print(f"Data shape: {btc_features.shape}")
    
    # Generate predictions for October 2021
    print("\nGenerating predictions for October 2021 (first 10 days as test)...")
    preds = predictions(btc_features, start_date='2021-10-01', days_to_predict=10)
    
    print("\nSample predictions:")
    print(preds[['Close', 'pred_today', 'pred_tomorrow', 'pred_2_days']].tail(15))
    
    # Calculate metrics
    print("\nCalculating prediction metrics...")
    metrics = calculate_prediction_metrics(preds, 'Close', 'pred_today')
    print(f"RMSE: ${metrics['rmse']:,.2f}")
    print(f"MAE: ${metrics['mae']:,.2f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"Valid predictions: {metrics['count']}")
    
    # Predict next 7 days
    print("\nPredicting next 7 days from last date...")
    future_preds = predict_next_days(btc_features, days_ahead=7)
    print(future_preds)
