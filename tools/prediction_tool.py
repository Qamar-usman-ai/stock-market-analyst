"""
Tool 4: ARIMA/SARIMA Future Price Prediction Tool
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import warnings
warnings.filterwarnings('ignore')

from langchain_core.tools import tool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


@tool
def predict_stock_price(
    ticker: str,
    forecast_days: int = 30,
    train_split: float = 0.7,
    val_split: float = 0.15,
    use_sarima: bool = True,
) -> str:
    """
    Predict future stock prices using ARIMA/SARIMA models.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        forecast_days: Number of days to forecast into the future (default: 30)
        train_split: Fraction of data for training (default: 0.7)
        val_split: Fraction of data for validation (default: 0.15)
        use_sarima: Use SARIMA if True, else ARIMA (default: True)
    
    Returns:
        JSON string with predictions, metrics, and chart paths
    """
    try:
        csv_path = f"outputs/data/{ticker.upper()}_historical.csv"
        if not os.path.exists(csv_path):
            return json.dumps({"error": f"Data not found for {ticker}. Run scrape_stock_data first.", "status": "failed"})
        
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        prices = df['Close'].values
        dates = df['Date'].values
        n = len(prices)
        
        # ─── Train / Validation / Test Split ────────────────────────────────────
        train_end = int(n * train_split)
        val_end = int(n * (train_split + val_split))
        
        train_data = prices[:train_end]
        val_data = prices[train_end:val_end]
        test_data = prices[val_end:]
        
        train_dates = dates[:train_end]
        val_dates = dates[train_end:val_end]
        test_dates = dates[val_end:]
        
        # ─── Stationarity Check ──────────────────────────────────────────────────
        adf_result = adfuller(train_data)
        is_stationary = adf_result[1] < 0.05
        d_order = 0 if is_stationary else 1
        
        # ─── Auto-parameter selection (simplified grid search) ───────────────────
        best_aic = np.inf
        best_params = (1, d_order, 1)
        
        for p in range(0, 4):
            for q in range(0, 4):
                try:
                    if use_sarima:
                        model_try = SARIMAX(
                            train_data, order=(p, d_order, q),
                            seasonal_order=(1, 1, 1, 5),
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                    else:
                        from statsmodels.tsa.arima.model import ARIMA
                        model_try = ARIMA(train_data, order=(p, d_order, q))
                    
                    result_try = model_try.fit(disp=False)
                    if result_try.aic < best_aic:
                        best_aic = result_try.aic
                        best_params = (p, d_order, q)
                except Exception:
                    continue
        
        # ─── Fit Final Model ─────────────────────────────────────────────────────
        p, d, q = best_params
        
        if use_sarima:
            model = SARIMAX(
                train_data, order=(p, d, q),
                seasonal_order=(1, 1, 1, 5),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
        else:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(train_data, order=(p, d, q))
        
        fitted_model = model.fit(disp=False)
        
        # ─── Validation Predictions ──────────────────────────────────────────────
        val_pred = []
        history = list(train_data)
        
        for i in range(len(val_data)):
            if use_sarima:
                temp_model = SARIMAX(
                    history, order=(p, d, q),
                    seasonal_order=(1, 1, 1, 5),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                from statsmodels.tsa.arima.model import ARIMA
                temp_model = ARIMA(history, order=(p, d, q))
            
            temp_fit = temp_model.fit(disp=False)
            yhat = temp_fit.forecast(steps=1)[0]
            val_pred.append(yhat)
            history.append(val_data[i])
        
        val_pred = np.array(val_pred)
        
        # ─── Test Predictions ────────────────────────────────────────────────────
        test_pred = []
        history = list(prices[:val_end])
        
        for i in range(len(test_data)):
            if use_sarima:
                temp_model = SARIMAX(
                    history, order=(p, d, q),
                    seasonal_order=(1, 1, 1, 5),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                from statsmodels.tsa.arima.model import ARIMA
                temp_model = ARIMA(history, order=(p, d, q))
            
            temp_fit = temp_model.fit(disp=False)
            yhat = temp_fit.forecast(steps=1)[0]
            test_pred.append(yhat)
            history.append(test_data[i])
        
        test_pred = np.array(test_pred)
        
        # ─── Metrics ─────────────────────────────────────────────────────────────
        def compute_metrics(actual, predicted, name):
            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            mse = mean_squared_error(actual, predicted)
            r2 = r2_score(actual, predicted)
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            return {
                "set": name,
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "MSE": round(mse, 4),
                "R2_Score": round(r2, 4),
                "MAPE_pct": round(mape, 2)
            }
        
        val_metrics = compute_metrics(val_data, val_pred, "Validation")
        test_metrics = compute_metrics(test_data, test_pred, "Test")
        
        # ─── Future Forecast ─────────────────────────────────────────────────────
        final_model = SARIMAX(
            prices, order=(p, d, q),
            seasonal_order=(1, 1, 1, 5),
            enforce_stationarity=False,
            enforce_invertibility=False
        ) if use_sarima else __import__('statsmodels.tsa.arima.model', fromlist=['ARIMA']).ARIMA(prices, order=(p, d, q))
        
        final_fit = final_model.fit(disp=False)
        forecast_result = final_fit.get_forecast(steps=forecast_days)
        forecast_mean = forecast_result.predicted_mean
        forecast_conf = forecast_result.conf_int()
        
        last_date = pd.to_datetime(dates[-1])
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_days, freq='B'
        )[:forecast_days]
        
        # ─── Visualizations ──────────────────────────────────────────────────────
        os.makedirs("outputs/charts", exist_ok=True)
        
        # Chart 1: Full prediction chart
        fig1 = go.Figure()
        
        fig1.add_trace(go.Scatter(
            x=pd.to_datetime(train_dates), y=train_data,
            name='Training Data', line=dict(color='#4CAF50', width=1.5)
        ))
        fig1.add_trace(go.Scatter(
            x=pd.to_datetime(val_dates), y=val_data,
            name='Validation Data', line=dict(color='#2196F3', width=1.5)
        ))
        fig1.add_trace(go.Scatter(
            x=pd.to_datetime(val_dates), y=val_pred,
            name='Val Predictions', line=dict(color='#FF9800', width=1.5, dash='dot')
        ))
        fig1.add_trace(go.Scatter(
            x=pd.to_datetime(test_dates), y=test_data,
            name='Test Data', line=dict(color='#9C27B0', width=1.5)
        ))
        fig1.add_trace(go.Scatter(
            x=pd.to_datetime(test_dates), y=test_pred,
            name='Test Predictions', line=dict(color='#E91E63', width=1.5, dash='dot')
        ))
        fig1.add_trace(go.Scatter(
            x=future_dates, y=forecast_mean,
            name='Future Forecast', line=dict(color='#FF5722', width=2.5),
            mode='lines+markers'
        ))
        fig1.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(forecast_conf.iloc[:, 1]) + list(forecast_conf.iloc[:, 0][::-1]),
            fill='toself', fillcolor='rgba(255,87,34,0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval'
        ))
        
        fig1.update_layout(
            template='plotly_dark',
            title=dict(text=f"{ticker.upper()} - {'SARIMA' if use_sarima else 'ARIMA'} Price Prediction", font=dict(size=20)),
            xaxis_title="Date", yaxis_title="Price (USD)",
            height=600, showlegend=True
        )
        
        path1 = f"outputs/charts/{ticker}_prediction.html"
        fig1.write_html(path1)
        
        # Chart 2: Metrics comparison
        fig2 = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Actual vs Predicted (Validation)", "Actual vs Predicted (Test)"]
        )
        
        fig2.add_trace(go.Scatter(
            x=val_data, y=val_pred, mode='markers',
            name='Val: Actual vs Pred', marker=dict(color='#2196F3', size=5, opacity=0.7)
        ), row=1, col=1)
        val_line = np.linspace(val_data.min(), val_data.max(), 100)
        fig2.add_trace(go.Scatter(
            x=val_line, y=val_line, mode='lines',
            name='Perfect Fit', line=dict(color='red', dash='dash')
        ), row=1, col=1)
        
        fig2.add_trace(go.Scatter(
            x=test_data, y=test_pred, mode='markers',
            name='Test: Actual vs Pred', marker=dict(color='#9C27B0', size=5, opacity=0.7)
        ), row=1, col=2)
        test_line = np.linspace(test_data.min(), test_data.max(), 100)
        fig2.add_trace(go.Scatter(
            x=test_line, y=test_line, mode='lines',
            name='Perfect Fit', line=dict(color='red', dash='dash')
        ), row=1, col=2)
        
        fig2.update_layout(
            template='plotly_dark',
            title=dict(text=f"{ticker.upper()} - Model Accuracy: Actual vs Predicted", font=dict(size=18)),
            height=500
        )
        
        path2 = f"outputs/charts/{ticker}_model_accuracy.html"
        fig2.write_html(path2)
        
        # ─── Forecast Summary ─────────────────────────────────────────────────────
        current_price = float(prices[-1])
        forecast_end_price = float(forecast_mean.iloc[-1])
        price_change = ((forecast_end_price - current_price) / current_price) * 100
        
        forecast_data = []
        for i in range(len(future_dates)):
            forecast_data.append({
                "date": str(future_dates[i])[:10],
                "predicted_price": round(float(forecast_mean.iloc[i]), 2),
                "lower_bound": round(float(forecast_conf.iloc[i, 0]), 2),
                "upper_bound": round(float(forecast_conf.iloc[i, 1]), 2),
            })
        
        result = {
            "ticker": ticker.upper(),
            "model": "SARIMA" if use_sarima else "ARIMA",
            "model_params": {
                "order": list(best_params),
                "seasonal_order": [1, 1, 1, 5] if use_sarima else None,
                "aic": round(best_aic, 2),
                "d_order": d_order,
                "is_stationary": is_stationary,
                "adf_pvalue": round(float(adf_result[1]), 4),
            },
            "data_split": {
                "train_samples": int(train_end),
                "val_samples": int(val_end - train_end),
                "test_samples": int(n - val_end),
                "train_pct": round(train_split * 100),
                "val_pct": round(val_split * 100),
                "test_pct": round((1 - train_split - val_split) * 100),
            },
            "metrics": {
                "validation": val_metrics,
                "test": test_metrics,
            },
            "forecast_summary": {
                "current_price": round(current_price, 2),
                "forecast_days": forecast_days,
                "forecast_end_price": round(forecast_end_price, 2),
                "price_change_pct": round(price_change, 2),
                "direction": "UP 📈" if price_change > 0 else "DOWN 📉",
                "min_forecast": round(float(forecast_mean.min()), 2),
                "max_forecast": round(float(forecast_mean.max()), 2),
            },
            "forecast_data": forecast_data[:10],
            "chart_paths": {
                "prediction": path1,
                "model_accuracy": path2
            },
            "status": "success"
        }
        
        return json.dumps(result, default=str)
    
    except Exception as e:
        import traceback
        return json.dumps({"error": str(e), "traceback": traceback.format_exc(), "status": "failed"})
