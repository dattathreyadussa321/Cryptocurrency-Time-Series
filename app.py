"""
Flask web application for cryptocurrency time series analysis and forecasting.
"""

from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import os
import json
from datetime import datetime

# Import our modules
from data_loader import load_all_years, get_available_data
from data_processor import resample, filter_by_date, get_data_summary
from feature_engineering import add_cols
from time_series_models import ARIMA_model, compare_models, fit_best_model
from forecasting import predictions, predict_next_days, calculate_prediction_metrics, get_trading_signals
from visualization import create_all_visualizations, plot_historical_prices, plot_predictions
from config import FLASK_CONFIG

app = Flask(__name__)
CORS(app)

# Global data cache
data_cache = {}


def load_data(coin='bitcoin', force_reload=False):
    """Load and cache cryptocurrency data."""
    cache_key = f"{coin}_data"
    
    if cache_key not in data_cache or force_reload:
        print(f"Loading {coin} data...")
        raw_data = load_all_years(coin)
        daily_data = resample(raw_data, '1d')
        featured_data = add_cols(daily_data)
        
        data_cache[cache_key] = {
            'raw': raw_data,
            'daily': daily_data,
            'featured': featured_data
        }
        print(f"{coin} data loaded and cached!")
    
    return data_cache[cache_key]


@app.route('/')
def index():
    """Render main dashboard."""
    return render_template('index.html')


@app.route('/api/available-data')
def available_data():
    """Get available cryptocurrency data."""
    try:
        available = get_available_data()
        return jsonify({
            'success': True,
            'data': available
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/data/<coin>')
def get_data(coin='bitcoin'):
    """Get cryptocurrency data with summary."""
    try:
        data = load_data(coin)
        summary = get_data_summary(data['featured'])
        
        # Convert timestamps to strings for JSON serialization
        if 'date_range' in summary:
            summary['date_range']['start'] = summary['date_range']['start'].strftime('%Y-%m-%d')
            summary['date_range']['end'] = summary['date_range']['end'].strftime('%Y-%m-%d')
        
        # Convert price_stats to native Python types
        if 'price_stats' in summary:
            summary['price_stats'] = {
                'min': float(summary['price_stats']['min']),
                'max': float(summary['price_stats']['max']),
                'mean': float(summary['price_stats']['mean']),
                'std': float(summary['price_stats']['std'])
            }
        
        # Convert missing_values to integers
        if 'missing_values' in summary:
            summary['missing_values'] = {k: int(v) for k, v in summary['missing_values'].items()}
        
        # Convert data_types to strings
        if 'data_types' in summary:
            summary['data_types'] = {k: str(v) for k, v in summary['data_types'].items()}
        
        # Get recent data for display
        recent_data = data['featured'].tail(30).reset_index()
        recent_data['timestamp'] = recent_data['timestamp'].astype(str)
        
        return jsonify({
            'success': True,
            'summary': summary,
            'recent_data': recent_data.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/historical-chart/<coin>')
def historical_chart(coin='bitcoin'):
    """Generate and return historical price chart."""
    try:
        data = load_data(coin)
        chart_path = plot_historical_prices(data['featured'], 
                                            save_path=f'static/charts/historical_{coin}.png')
        
        return jsonify({
            'success': True,
            'chart_url': f'/static/charts/historical_{coin}.png'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/run-models/<coin>', methods=['POST'])
def run_models(coin='bitcoin'):
    """Run model comparison."""
    try:
        data = load_data(coin)
        
        # Filter training data
        train_end = request.json.get('train_end', '2021-10-01')
        train_data = filter_by_date(data['featured'], end_date=train_end)
        
        print(f"Running model comparison on {len(train_data)} days of data...")
        
        # Run models (this takes time)
        models_results = compare_models(train_data)
        
        # Convert to JSON-serializable format
        models_json = models_results.head(20).copy()
        models_json['RMSE'] = models_json['RMSE'].astype(float)
        models_json['AIC'] = models_json['AIC'].astype(float)
        models_json['order'] = models_json['order'].astype(str)
        
        return jsonify({
            'success': True,
            'models': models_json.to_dict(orient='records'),
            'best_model': {
                'type': models_json.iloc[0]['model type'],
                'order': str(models_json.iloc[0]['order']),
                'rmse': float(models_json.iloc[0]['RMSE']),
                'aic': float(models_json.iloc[0]['AIC'])
            }
        })
    except Exception as e:
        print(f"Error running models: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict/<coin>', methods=['POST'])
def predict(coin='bitcoin'):
    """Generate predictions."""
    try:
        data = load_data(coin)
        
        # Get parameters from request
        start_date = request.json.get('start_date', '2021-10-01')
        days = request.json.get('days', 30)
        order = tuple(request.json.get('order', [2, 1, 2]))
        
        print(f"Generating predictions from {start_date} for {days} days...")
        
        # Generate predictions
        preds = predictions(data['featured'], start_date=start_date, 
                          days_to_predict=days, order=order)
        
        # Calculate metrics
        metrics = calculate_prediction_metrics(preds, 'Close', 'pred_today')
        
        # Generate predictions chart
        chart_path = plot_predictions(preds, save_path=f'static/charts/predictions_{coin}.png')
        
        # Get prediction data for display
        pred_data = preds[['Close', 'pred_today', 'pred_tomorrow', 'pred_2_days']].tail(days + 5)
        pred_data = pred_data.reset_index()
        pred_data['timestamp'] = pred_data['timestamp'].astype(str)
        # Convert to dict first, then replace NaN with None
        pred_dict = pred_data.to_dict(orient='records')
        # Replace NaN values with None in the dict
        import math
        for record in pred_dict:
            for key, value in record.items():
                if isinstance(value, float) and math.isnan(value):
                    record[key] = None
        
        # Generate trading signals
        signals_df = get_trading_signals(preds, 'pred_tomorrow')
        recent_signals = signals_df[['Close', 'pred_tomorrow', 'expected_change_pct', 'signal']].tail(10)
        recent_signals = recent_signals.reset_index()
        recent_signals['timestamp'] = recent_signals['timestamp'].astype(str)
        # Convert to dict first, then replace NaN with None
        signals_dict = recent_signals.to_dict(orient='records')
        for record in signals_dict:
            for key, value in record.items():
                if isinstance(value, float) and math.isnan(value):
                    record[key] = None
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'chart_url': f'/static/charts/predictions_{coin}.png',
            'predictions': pred_dict,
            'signals': signals_dict
        })
    except Exception as e:
        print(f"Error generating predictions: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/future-predict/<coin>', methods=['POST'])
def future_predict(coin='bitcoin'):
    """Predict future prices (beyond available data)."""
    try:
        data = load_data(coin)
        
        # Get parameters
        days_ahead = request.json.get('days_ahead', 7)
        order = tuple(request.json.get('order', [2, 1, 2]))
        
        print(f"Predicting next {days_ahead} days...")
        
        # Generate future predictions
        future_preds = predict_next_days(data['featured'], days_ahead=days_ahead, order=order)
        
        # Convert to JSON
        future_preds_json = future_preds.reset_index()
        future_preds_json['date'] = future_preds_json['date'].astype(str)
        
        return jsonify({
            'success': True,
            'predictions': future_preds_json.to_dict(orient='records')
        })
    except Exception as e:
        print(f"Error predicting future: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/static/charts/<filename>')
def serve_chart(filename):
    """Serve generated chart images."""
    return send_file(f'static/charts/{filename}', mimetype='image/png')


if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs('static/charts', exist_ok=True)
    
    print("Starting Flask application...")
    print("Loading initial data...")
    
    # Pre-load Bitcoin data
    try:
        load_data('bitcoin')
        print("Bitcoin data preloaded successfully!")
    except Exception as e:
        print(f"Warning: Could not preload Bitcoin data: {e}")
    
    print(f"\nServer starting on http://{FLASK_CONFIG['HOST']}:{FLASK_CONFIG['PORT']}")
    print("Open your browser and navigate to the URL above")
    
    app.run(
        host=FLASK_CONFIG['HOST'],
        port=FLASK_CONFIG['PORT'],
        debug=FLASK_CONFIG['DEBUG']
    )
