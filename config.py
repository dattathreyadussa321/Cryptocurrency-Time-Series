"""
Configuration settings for the Cryptocurrency Time Series project.
"""

import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data directory
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Images directory
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images')

# Available cryptocurrencies
AVAILABLE_COINS = ['bitcoin', 'ether']

# Available years
AVAILABLE_YEARS = [2017, 2018, 2019, 2020, 2021]

# Data columns to keep
DATA_COLUMNS = ['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'Target']

# Default resampling period
DEFAULT_RESAMPLE_PERIOD = '1d'

# Model parameters
MODEL_PARAMS = {
    'white_noise': {'order': (0, 0, 0)},
    'random_walk': {'order': (0, 1, 0)},
    'random_walk_drift': {'order': (0, 1, 0), 'trend': 'c'},
    'best_arima': {'order': (2, 1, 2)},
}

# Train/test split date
TRAIN_TEST_SPLIT_DATE = '2021-10-01'

# Flask configuration
FLASK_CONFIG = {
    'DEBUG': True,
    'HOST': '0.0.0.0',
    'PORT': 5000,
}

# Chart styling
CHART_CONFIG = {
    'figsize': (20, 10),
    'colors': {
        'primary': 'navy',
        'secondary': 'cyan',
        'tertiary': 'magenta',
        'quaternary': 'cornflowerblue',
        'accent': 'indigo',
    },
    'fontsize': {
        'title': 20,
        'label': 16,
        'legend': 12,
    }
}
