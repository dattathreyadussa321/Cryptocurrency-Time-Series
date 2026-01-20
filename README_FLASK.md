# Cryptocurrency Time Series Analysis - Flask Application

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A professional web application for cryptocurrency time series analysis and price forecasting using advanced ARIMA models.

## ✨ Features

- **📊 Interactive Dashboard** - Modern, elegant dark-themed UI with glassmorphism effects
- **📈 Historical Analysis** - Visualize Bitcoin price trends with multiple moving averages
- **🔬 Model Comparison** - Compare 6 different time series models (ARIMA, Random Walk, ARIMAX, etc.)
- **🔮 Price Forecasting** - Generate predictions for future Bitcoin prices
- **📉 Technical Indicators** - 33 advanced technical indicators including SMA, momentum, volatility, and volume metrics
- **🎯 Trading Signals** - Automated BUY/SELL/HOLD recommendations based on predictions

## 🚀 Quick Start

### Installation

1. Navigate to the project directory:
```powershell
cd c:\TFI\Projects\cryptocurrency_time_series
```

2. Install required dependencies:
```powershell
pip install -r requirements.txt
```

### Running the Application

Start the Flask server:
```powershell
python app.py
```

The application will be available at **http://127.0.0.1:5000**

## 📁 Project Structure

```
cryptocurrency_time_series/
├── app.py                      # Flask application server
├── config.py                   # Configuration settings
├── data_loader.py              # Data loading module
├── data_processor.py           # Data preprocessing
├── feature_engineering.py      # Technical indicator creation
├── time_series_models.py       # Time series models implementation
├── forecasting.py              # Price prediction module
├── visualization.py            # Chart generation
├── requirements.txt            # Python dependencies
├── templates/
│   └── index.html             # Main dashboard UI
├── static/
│   ├── css/style.css          # Premium CSS styling
│   ├── js/main.js             # JavaScript functionality
│   └── charts/                # Generated charts
└── data/                       # Bitcoin & Ether CSV data (2017-2021)
```

## 🎨 UI Screenshots

The application features a stunning dark-themed interface with:
- Glassmorphism card effects
- Vibrant gradient colors
- Smooth animations
- Responsive design

## 📊 Available Models

1. **White Noise** - ARIMA(0,0,0)
2. **Random Walk** - ARIMA(0,1,0)
3. **Random Walk with Drift** - ARIMA(0,1,0) with trend
4. **ARMA** - Autoregressive Moving Average
5. **ARIMA** - Full ARIMA with differencing
6. **ARIMAX** - Extended ARIMA with exogenous variables

## 🔧 Technical Details

### Data Processing
- Processes 2.2+ million minute-level records
- Resamples to daily data (1,589 records)
- Creates 33 technical indicators
- Handles data from 2017-2021

### Time Series Analysis
- Best Model: ARIMA(2,1,2)
- RMSE-based model selection
- Support for multiple training window sizes
- Out-of-sample prediction validation

### API Endpoints

- `GET /` - Main dashboard
- `GET /api/data/<coin>` - Get cryptocurrency data summary
- `GET /api/historical-chart/<coin>` - Generate historical chart
- `POST /api/run-models/<coin>` - Run model comparison
- `POST /api/predict/<coin>` - Generate predictions
- `POST /api/future-predict/<coin>` - Forecast future prices

## 📈 Usage Examples

### Load Historical Data
1. Navigate to the dashboard
2. Click "Load Chart" to view Bitcoin historical prices with moving averages

### Run Model Comparison
1. Go to the "Models" section
2. Click "Run Models" (takes 1-2 minutes)
3. View comparison of all models with RMSE scores

### Generate Predictions
1. Navigate to "Predictions" section
2. Select start date and number of days
3. Click "Generate Predictions"
4. View predicted prices and trading signals

### Future Forecast
1. Scroll to "Future Price Forecast"
2. Select number of days ahead
3. Click "Forecast Future"
4. View predictions beyond available data

## 📦 Dependencies

- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- statsmodels >= 0.13.0
- scikit-learn >= 0.24.0
- Flask >= 2.0.0
- Flask-CORS >= 3.0.0
- scipy >= 1.7.0

## 🎯 Model Performance

The best performing model (ARIMA 2,1,2) achieves:
- Training on full dataset (1,183 days)
- Conservative predictions that closely mirror previous day's price
- Suitable for risk-averse trading strategies

## 📄 License

This project is part of a time series analysis study. Original authors: TJ Bray, Aalok Joshi, Paul Lindquist.

## 🤝 Contributing

This is an academic project. For questions or suggestions, please refer to the original authors.

## 📚 Documentation

For more information, see:
- [Original Notebook](Final_Main_Notebook.ipynb)
- [Project Presentation](Project_Presentation.pdf)
- Project README with detailed methodology

## 🔗 Data Source

Dataset from [Kaggle - Cryptocurrency Extra Data Bitcoin](https://www.kaggle.com/yamqwe/cryptocurrency-extra-data-bitcoin) by Yam Peleg for the G-Research Crypto Forecasting competition.

---

**Built with ❤️ using Flask, Python, and advanced time series analysis**
