"""
Visualization module for cryptocurrency data and predictions.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Flask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import CHART_CONFIG, IMAGES_DIR
import os


def plot_historical_prices(df, save_path=None, show_sma=True):
    """
    Plot historical Bitcoin prices with optional SMA overlays.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with Close and SMA columns
    save_path : str
        Path to save the figure (optional)
    show_sma : bool
        Whether to show SMA lines
    
    Returns:
    --------
    str
        Path where the figure was saved
    """
    fig, ax = plt.subplots(figsize=CHART_CONFIG['figsize'])
    
    # Plot closing price
    ax.plot(df.index, df['Close'], color=CHART_CONFIG['colors']['primary'], linewidth=2, label='Close Price')
    
    # Plot SMAs if requested and available
    if show_sma:
        if 'sma_30' in df.columns:
            ax.plot(df.index, df['sma_30'], color=CHART_CONFIG['colors']['secondary'], 
                   linewidth=1.5, alpha=0.8, label='30 Day Moving Average')
        if 'sma_200' in df.columns:
            ax.plot(df.index, df['sma_200'], color=CHART_CONFIG['colors']['tertiary'], 
                   linewidth=1.5, alpha=0.8, label='200 Day Moving Average')
    
    ax.set_ylabel('Price of Bitcoin', size=CHART_CONFIG['fontsize']['title'])
    ax.set_xlabel('Date', size=CHART_CONFIG['fontsize']['label'])
    ax.yaxis.set_major_formatter('${x:1,.0f}')
    ax.legend(loc=2, fontsize=CHART_CONFIG['fontsize']['legend'], edgecolor='1')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = os.path.join(IMAGES_DIR, 'historical_prices.png')
    
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_predictions(df, save_path=None):
    """
    Plot predicted prices vs actual prices.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with Close and prediction columns
    save_path : str
        Path to save the figure (optional)
    
    Returns:
    --------
    str
        Path where the figure was saved
    """
    fig, ax = plt.subplots(figsize=CHART_CONFIG['figsize'])
    
    # Plot actual close price
    ax.plot(df.index, df['Close'], color=CHART_CONFIG['colors']['primary'], 
           linewidth=2, label='Actual Close Price')
    
    # Plot predictions if available
    if 'pred_today' in df.columns:
        ax.plot(df.index, df['pred_today'], color=CHART_CONFIG['colors']['accent'], 
               linewidth=1.5, alpha=0.8, label='Predicted Today')
    
    if 'pred_tomorrow' in df.columns:
        # Shift predictions to align with the day they're predicting
        ax.plot(df.index, df['pred_tomorrow'].shift(1), color=CHART_CONFIG['colors']['tertiary'], 
               linewidth=1.5, alpha=0.8, label='Predicted Tomorrow')
    
    if 'pred_2_days' in df.columns:
        ax.plot(df.index, df['pred_2_days'].shift(2), color=CHART_CONFIG['colors']['quaternary'], 
               linewidth=1.5, alpha=0.8, label='Predicted Two Days Out')
    
    ax.set_ylabel('Price of Bitcoin', size=CHART_CONFIG['fontsize']['title'])
    ax.set_xlabel('Date', size=CHART_CONFIG['fontsize']['label'])
    ax.yaxis.set_major_formatter('${x:1,.0f}')
    ax.legend(loc=2, fontsize=CHART_CONFIG['fontsize']['legend'], edgecolor='1')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = os.path.join(IMAGES_DIR, 'predictions_chart.png')
    
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_predictions_overlap(df, save_path=None):
    """
    Plot all predictions overlapped to show conservative nature.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with Close and prediction columns
    save_path : str
        Path to save the figure (optional)
    
    Returns:
    --------
    str
        Path where the figure was saved
    """
    fig, ax = plt.subplots(figsize=CHART_CONFIG['figsize'])
    
    # Plot actual close price
    ax.plot(df.index, df['Close'], color=CHART_CONFIG['colors']['primary'], 
           linewidth=2, label='Actual Close Price')
    
    # Plot all predictions shifted back to overlap
    if 'pred_today' in df.columns:
        ax.plot(df.index, df['pred_today'].shift(-1), color=CHART_CONFIG['colors']['accent'], 
               linewidth=1.5, alpha=0.7, label='Predicted Today')
    
    if 'pred_tomorrow' in df.columns:
        ax.plot(df.index, df['pred_tomorrow'].shift(-1), color=CHART_CONFIG['colors']['tertiary'], 
               linewidth=1.5, alpha=0.7, label='Predicted Tomorrow')
    
    if 'pred_2_days' in df.columns:
        ax.plot(df.index, df['pred_2_days'].shift(-1), color=CHART_CONFIG['colors']['quaternary'], 
               linewidth=1.5, alpha=0.7, label='Predicted Two Days Out')
    
    ax.set_ylabel('Price of Bitcoin', size=CHART_CONFIG['fontsize']['title'])
    ax.set_xlabel('Date', size=CHART_CONFIG['fontsize']['label'])
    ax.yaxis.set_major_formatter('${x:1,.0f}')
    ax.legend(loc=2, fontsize=CHART_CONFIG['fontsize']['legend'], edgecolor='1')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = os.path.join(IMAGES_DIR, 'overlap_chart.png')
    
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_returns_distribution(df, save_path=None):
    """
    Plot distribution of returns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with return column
    save_path : str
        Path to save the figure (optional)
    
    Returns:
    --------
    str
        Path where the figure was saved
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'return' in df.columns:
        df['return'].hist(bins=50, color=CHART_CONFIG['colors']['primary'], 
                         alpha=0.7, edgecolor='black', ax=ax)
        ax.set_xlabel('Daily Return', size=CHART_CONFIG['fontsize']['label'])
        ax.set_ylabel('Frequency', size=CHART_CONFIG['fontsize']['label'])
        ax.set_title('Distribution of Daily Returns', size=CHART_CONFIG['fontsize']['title'])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = os.path.join(IMAGES_DIR, 'returns_distribution.png')
    
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_model_comparison(models_df, save_path=None):
    """
    Create a bar chart comparing model RMSE values.
    
    Parameters:
    -----------
    models_df : pd.DataFrame
        DataFrame with model comparison results
    save_path : str
        Path to save the figure (optional)
    
    Returns:
    --------
    str
        Path where the figure was saved
    """
    # Get top 10 models
    top_models = models_df.head(10).copy()
    top_models['RMSE'] = pd.to_numeric(top_models['RMSE'])
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create labels
    labels = [f"{row['model type']}\n{row['order']}" for _, row in top_models.iterrows()]
    
    # Create bar chart
    bars = ax.barh(range(len(top_models)), top_models['RMSE'], 
                   color=CHART_CONFIG['colors']['primary'], alpha=0.7)
    
    ax.set_yticks(range(len(top_models)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('RMSE', size=CHART_CONFIG['fontsize']['label'])
    ax.set_title('Model Comparison (Top 10)', size=CHART_CONFIG['fontsize']['title'])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_models['RMSE'])):
        ax.text(value, i, f' {value:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = os.path.join(IMAGES_DIR, 'model_comparison.png')
    
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return save_path


def create_all_visualizations(df, predictions_df=None, models_df=None):
    """
    Create all visualizations and return their paths.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Main dataframe with historical data
    predictions_df : pd.DataFrame
        DataFrame with predictions (optional)
    models_df : pd.DataFrame
        DataFrame with model comparison results (optional)
    
    Returns:
    --------
    dict
        Dictionary with visualization names and file paths
    """
    vis_paths = {}
    
    print("Creating historical price chart...")
    vis_paths['historical'] = plot_historical_prices(df)
    
    print("Creating returns distribution chart...")
    vis_paths['returns'] = plot_returns_distribution(df)
    
    if predictions_df is not None and 'pred_today' in predictions_df.columns:
        print("Creating predictions chart...")
        vis_paths['predictions'] = plot_predictions(predictions_df)
        
        print("Creating overlap chart...")
        vis_paths['overlap'] = plot_predictions_overlap(predictions_df)
    
    if models_df is not None:
        print("Creating model comparison chart...")
        vis_paths['models'] = plot_model_comparison(models_df)
    
    return vis_paths


if __name__ == "__main__":
    # Test the module
    from data_loader import load_all_years
    from data_processor import resample
    from feature_engineering import add_cols
    
    print("Testing visualization module...")
    
    # Load data
    print("\nLoading data...")
    btc = load_all_years('bitcoin')
    btc_daily = resample(btc, '1d')
    btc_features = add_cols(btc_daily)
    
    # Create visualizations
    print("\nCreating visualizations...")
    vis_paths = create_all_visualizations(btc_features)
    
    print("\nGenerated visualizations:")
    for name, path in vis_paths.items():
        print(f"  {name}: {path}")
