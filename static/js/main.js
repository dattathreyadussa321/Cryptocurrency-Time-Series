// JavaScript for Cryptocurrency Time Series Dashboard

const API_BASE = '';

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', function () {
    loadDataSummary();
});

// Show/hide loading overlay
function showLoading(text = 'Processing...') {
    document.getElementById('loading-overlay').style.display = 'flex';
    document.getElementById('loading-text').textContent = text;
}

function hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}

// Load data summary
async function loadDataSummary() {
    try {
        const response = await fetch(`${API_BASE}/api/data/bitcoin`);
        const data = await response.json();

        if (data.success) {
            const summary = data.summary;

            // Update stats
            document.getElementById('total-records').textContent = summary.total_records.toLocaleString();

            if (summary.price_stats) {
                // Get the last data point for current price
                if (data.recent_data && data.recent_data.length > 0) {
                    const lastData = data.recent_data[data.recent_data.length - 1];
                    if (lastData.Close) {
                        document.getElementById('current-price').textContent =
                            '$' + parseFloat(lastData.Close).toLocaleString(undefined, { maximumFractionDigits: 0 });
                    }
                }

                document.getElementById('max-price').textContent =
                    '$' + summary.price_stats.max.toLocaleString(undefined, { maximumFractionDigits: 0 });
            }

            console.log('Data summary loaded successfully:', summary);
        } else {
            console.error('Error loading data summary:', data.error);
        }
    } catch (error) {
        console.error('Error loading data summary:', error);
    }
}

// Load historical chart
async function loadHistoricalChart() {
    const container = document.getElementById('historical-chart');
    container.innerHTML = '<div class="loading-state"><div class="loading-spinner"></div><p>Loading historical chart...</p></div>';

    try {
        const response = await fetch(`${API_BASE}/api/historical-chart/bitcoin`);
        const data = await response.json();

        if (data.success) {
            container.innerHTML = `<img src="${data.chart_url}?t=${Date.now()}" alt="Historical Price Chart">`;
        } else {
            container.innerHTML = `<div class="loading-state"><p style="color: #f5576c;">Error: ${data.error}</p></div>`;
        }
    } catch (error) {
        console.error('Error loading chart:', error);
        container.innerHTML = `<div class="loading-state"><p style="color: #f5576c;">Error loading chart</p></div>`;
    }
}

// Run models comparison
async function runModels() {
    showLoading('Running time series models... This may take 1-2 minutes');
    const container = document.getElementById('models-container');
    container.innerHTML = '<div class="loading-state"><div class="loading-spinner"></div><p>Running models, please wait...</p></div>';

    try {
        const response = await fetch(`${API_BASE}/api/run-models/bitcoin`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                train_end: '2021-10-01'
            })
        });

        const data = await response.json();

        if (data.success) {
            // Display best model
            const bestModel = data.best_model;
            document.getElementById('model-rmse').textContent = bestModel.rmse.toFixed(4);

            // Create table
            let html = `
                <div style="margin-bottom: 20px; padding: 20px; background: var(--dark-tertiary); border-radius: 12px;">
                    <h4 style="margin-bottom: 10px;">🏆 Best Model</h4>
                    <p><strong>Type:</strong> ${bestModel.type}</p>
                    <p><strong>Order:</strong> ${bestModel.order}</p>
                    <p><strong>RMSE:</strong> ${bestModel.rmse.toFixed(4)}</p>
                    <p><strong>AIC:</strong> ${bestModel.aic.toFixed(2)}</p>
                </div>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Model Type</th>
                                <th>Order</th>
                                <th>Train Size</th>
                                <th>RMSE</th>
                                <th>AIC</th>
                            </tr>
                        </thead>
                        <tbody>
            `;

            data.models.slice(0, 15).forEach((model, index) => {
                html += `
                    <tr>
                        <td>${index + 1}</td>
                        <td>${model['model type']}</td>
                        <td>${model.order}</td>
                        <td>${model.train_size}</td>
                        <td>${parseFloat(model.RMSE).toFixed(4)}</td>
                        <td>${parseFloat(model.AIC).toFixed(2)}</td>
                    </tr>
                `;
            });

            html += '</tbody></table></div>';
            container.innerHTML = html;
        } else {
            container.innerHTML = `<div class="loading-state"><p style="color: #f5576c;">Error: ${data.error}</p></div>`;
        }
    } catch (error) {
        console.error('Error running models:', error);
        container.innerHTML = `<div class="loading-state"><p style="color: #f5576c;">Error running models</p></div>`;
    } finally {
        hideLoading();
    }
}

// Generate predictions
async function generatePredictions() {
    const startDate = document.getElementById('start-date').value;
    const days = parseInt(document.getElementById('pred-days').value);

    if (!startDate || days < 1) {
        alert('Please enter valid parameters');
        return;
    }

    showLoading('Generating predictions... This may take a minute');
    const container = document.getElementById('predictions-container');
    container.innerHTML = '<div class="loading-state"><div class="loading-spinner"></div><p>Generating predictions...</p></div>';

    try {
        const response = await fetch(`${API_BASE}/api/predict/bitcoin`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                start_date: startDate,
                days: days,
                order: [2, 1, 2]
            })
        });

        const data = await response.json();

        if (data.success) {
            // Show metrics
            const metricsCard = document.getElementById('metrics-card');
            metricsCard.style.display = 'block';
            document.getElementById('metric-rmse').textContent =
                '$' + data.metrics.rmse.toLocaleString(undefined, { maximumFractionDigits: 2 });
            document.getElementById('metric-mae').textContent =
                '$' + data.metrics.mae.toLocaleString(undefined, { maximumFractionDigits: 2 });
            document.getElementById('metric-mape').textContent =
                data.metrics.mape.toFixed(2) + '%';

            // Show chart
            let html = `<img src="${data.chart_url}?t=${Date.now()}" alt="Predictions Chart">`;

            // Show recent predictions table
            if (data.predictions && data.predictions.length > 0) {
                html += `
                    <div class="table-container" style="margin-top: 30px;">
                        <h4 style="margin-bottom: 15px;">Recent Predictions</h4>
                        <table>
                            <thead>
                                <tr>
                                    <th>Date</th>
                                    <th>Actual Close</th>
                                    <th>Predicted Today</th>
                                    <th>Predicted Tomorrow</th>
                                    <th>Predicted 2 Days</th>
                                </tr>
                            </thead>
                            <tbody>
                `;

                data.predictions.slice(-10).forEach(pred => {
                    html += `
                        <tr>
                            <td>${pred.timestamp}</td>
                            <td>${pred.Close ? '$' + pred.Close.toLocaleString(undefined, { maximumFractionDigits: 2 }) : 'N/A'}</td>
                            <td>${pred.pred_today ? '$' + pred.pred_today.toLocaleString(undefined, { maximumFractionDigits: 2 }) : 'N/A'}</td>
                            <td>${pred.pred_tomorrow ? '$' + pred.pred_tomorrow.toLocaleString(undefined, { maximumFractionDigits: 2 }) : 'N/A'}</td>
                            <td>${pred.pred_2_days ? '$' + pred.pred_2_days.toLocaleString(undefined, { maximumFractionDigits: 2 }) : 'N/A'}</td>
                        </tr>
                    `;
                });

                html += '</tbody></table></div>';
            }

            // Show trading signals
            if (data.signals && data.signals.length > 0) {
                html += `
                    <div class="table-container" style="margin-top: 30px;">
                        <h4 style="margin-bottom: 15px;">📊 Trading Signals</h4>
                        <table>
                            <thead>
                                <tr>
                                    <th>Date</th>
                                    <th>Current Price</th>
                                    <th>Predicted Tomorrow</th>
                                    <th>Expected Change</th>
                                    <th>Signal</th>
                                </tr>
                            </thead>
                            <tbody>
                `;

                data.signals.forEach(signal => {
                    const signalClass = signal.signal === 'BUY' ? 'style="color: #43e97b;"' :
                        signal.signal === 'SELL' ? 'style="color: #f5576c;"' : '';
                    html += `
                        <tr>
                            <td>${signal.timestamp}</td>
                            <td>$${signal.Close ? signal.Close.toLocaleString(undefined, { maximumFractionDigits: 2 }) : 'N/A'}</td>
                            <td>$${signal.pred_tomorrow ? signal.pred_tomorrow.toLocaleString(undefined, { maximumFractionDigits: 2 }) : 'N/A'}</td>
                            <td>${signal.expected_change_pct ? signal.expected_change_pct.toFixed(2) + '%' : 'N/A'}</td>
                            <td ${signalClass}><strong>${signal.signal}</strong></td>
                        </tr>
                    `;
                });

                html += '</tbody></table></div>';
            }

            container.innerHTML = html;
        } else {
            container.innerHTML = `<div class="loading-state"><p style="color: #f5576c;">Error: ${data.error}</p></div>`;
        }
    } catch (error) {
        console.error('Error generating predictions:', error);
        container.innerHTML = `<div class="loading-state"><p style="color: #f5576c;">Error generating predictions</p></div>`;
    } finally {
        hideLoading();
    }
}

// Predict future prices
async function predictFuture() {
    const daysAhead = parseInt(document.getElementById('future-days').value);

    if (daysAhead < 1) {
        alert('Please enter valid number of days');
        return;
    }

    showLoading('Forecasting future prices...');
    const container = document.getElementById('future-container');
    container.innerHTML = '<div class="loading-state"><div class="loading-spinner"></div><p>Forecasting...</p></div>';

    try {
        const response = await fetch(`${API_BASE}/api/future-predict/bitcoin`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                days_ahead: daysAhead,
                order: [2, 1, 2]
            })
        });

        const data = await response.json();

        if (data.success) {
            let html = `
                <h4 style="margin-bottom: 15px;">🚀 Future Price Forecast (Next ${daysAhead} Days)</h4>
                <table>
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Predicted Price</th>
                            <th>Log Price</th>
                        </tr>
                    </thead>
                    <tbody>
            `;

            data.predictions.forEach(pred => {
                html += `
                    <tr>
                        <td>${pred.date}</td>
                        <td>$${pred.predicted_price.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                        <td>${pred.log_predicted_price.toFixed(4)}</td>
                    </tr>
                `;
            });

            html += '</tbody></table>';
            container.innerHTML = html;
        } else {
            container.innerHTML = `<div class="loading-state"><p style="color: #f5576c;">Error: ${data.error}</p></div>`;
        }
    } catch (error) {
        console.error('Error predicting future:', error);
        container.innerHTML = `<div class="loading-state"><p style="color: #f5576c;">Error predicting future</p></div>`;
    } finally {
        hideLoading();
    }
}

// Smooth scrolling for navigation
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', function (e) {
        e.preventDefault();
        const targetId = this.getAttribute('href');
        const targetElement = document.querySelector(targetId);

        if (targetElement) {
            targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });

            // Update active nav link
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            this.classList.add('active');
        }
    });
});
