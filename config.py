import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images')

AVAILABLE_COINS = ['bitcoin', 'ether']
AVAILABLE_YEARS = [2017, 2018, 2019, 2020, 2021]
DATA_COLUMNS = ['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'Target']
DEFAULT_RESAMPLE_PERIOD = '1d'

MODEL_PARAMS = {
    'white_noise': {'order': (0, 0, 0)},
    'random_walk': {'order': (0, 1, 0)},
    'random_walk_drift': {'order': (0, 1, 0), 'trend': 'c'},
    'best_arima': {'order': (2, 1, 2)},
}

TRAIN_TEST_SPLIT_DATE = '2021-10-01'

FLASK_CONFIG = {
    'DEBUG': True,
    'HOST': '0.0.0.0',
    'PORT': 5000,
}

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
