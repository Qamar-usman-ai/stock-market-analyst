"""
Tool 3: Stock Data Visualization Tool
Generates comprehensive charts and analysis plots
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from langchain_core.tools import tool


@tool
def generate_stock_visualizations(ticker: str) -> str:
    """
    Generate comprehensive stock analysis visualizations including price charts,
    volume analysis, technical indicators, and correlation plots.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
    
    Returns:
        JSON string with paths to generated chart files
    """
    try:
        csv_path = f"outputs/data/{ticker.upper()}_historical.csv"
        if not os.path.exists(csv_path):
            return json.dumps({"error": f"Data file not found for {ticker}. Run scrape_stock_data first.", "status": "failed"})
        
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        os.makedirs("outputs/charts", exist_ok=True)
        chart_paths = {}
        
        # ─── Chart 1: Candlestick with Volume ───────────────────────────────────
        fig1 = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=[
                f"{ticker.upper()} - Price History with Moving Averages",
                "Volume",
                "MACD"
            ]
        )
        
        fig1.add_trace(go.Candlestick(
            x=df['Date'], open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name='Price', increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ), row=1, col=1)
        
        for ma, color in [('MA_20', '#FF6B6B'), ('MA_50', '#4ECDC4'), ('MA_200', '#45B7D1')]:
            if ma in df.columns:
                fig1.add_trace(go.Scatter(
                    x=df['Date'], y=df[ma], name=ma,
                    line=dict(color=color, width=1.5)
                ), row=1, col=1)
        
        colors = ['#26a69a' if c >= o else '#ef5350'
                  for c, o in zip(df['Close'], df['Open'])]
        fig1.add_trace(go.Bar(
            x=df['Date'], y=df['Volume'],
            name='Volume', marker_color=colors, opacity=0.7
        ), row=2, col=1)
        
        if 'MACD' in df.columns:
            fig1.add_trace(go.Scatter(
                x=df['Date'], y=df['MACD'],
                name='MACD', line=dict(color='#2196F3', width=1.5)
            ), row=3, col=1)
            fig1.add_trace(go.Scatter(
                x=df['Date'], y=df['Signal_Line'],
                name='Signal', line=dict(color='#FF9800', width=1.5)
            ), row=3, col=1)
            
            macd_hist = df['MACD'] - df['Signal_Line']
            fig1.add_trace(go.Bar(
                x=df['Date'], y=macd_hist, name='MACD Hist',
                marker_color=['#26a69a' if v >= 0 else '#ef5350' for v in macd_hist],
                opacity=0.6
            ), row=3, col=1)
        
        fig1.update_layout(
            template='plotly_dark', height=800,
            title=dict(text=f"{ticker.upper()} Comprehensive Price Analysis", font=dict(size=20)),
            xaxis_rangeslider_visible=False, showlegend=True
        )
        
        path1 = f"outputs/charts/{ticker}_price_analysis.html"
        fig1.write_html(path1)
        chart_paths["price_analysis"] = path1
        
        # ─── Chart 2: RSI & Bollinger Bands ─────────────────────────────────────
        fig2 = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=["Bollinger Bands", "RSI (14)"]
        )
        
        fig2.add_trace(go.Scatter(
            x=df['Date'], y=df['Close'],
            name='Close', line=dict(color='#E0E0E0', width=2)
        ), row=1, col=1)
        
        if all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            fig2.add_trace(go.Scatter(
                x=df['Date'], y=df['BB_Upper'],
                name='BB Upper', line=dict(color='#FF6B6B', width=1, dash='dash')
            ), row=1, col=1)
            fig2.add_trace(go.Scatter(
                x=df['Date'], y=df['BB_Lower'],
                name='BB Lower', line=dict(color='#4ECDC4', width=1, dash='dash'),
                fill='tonexty', fillcolor='rgba(100,100,255,0.1)'
            ), row=1, col=1)
            fig2.add_trace(go.Scatter(
                x=df['Date'], y=df['BB_Middle'],
                name='BB Middle', line=dict(color='#FFC107', width=1, dash='dot')
            ), row=1, col=1)
        
        if 'RSI' in df.columns:
            fig2.add_trace(go.Scatter(
                x=df['Date'], y=df['RSI'],
                name='RSI', line=dict(color='#9C27B0', width=2)
            ), row=2, col=1)
            fig2.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Overbought (70)")
            fig2.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Oversold (30)")
        
        fig2.update_layout(
            template='plotly_dark', height=700,
            title=dict(text=f"{ticker.upper()} - Bollinger Bands & RSI", font=dict(size=20))
        )
        
        path2 = f"outputs/charts/{ticker}_technical_indicators.html"
        fig2.write_html(path2)
        chart_paths["technical_indicators"] = path2
        
        # ─── Chart 3: Returns Distribution ──────────────────────────────────────
        df['Daily_Return'] = df['Close'].pct_change() * 100
        df['Cumulative_Return'] = (1 + df['Close'].pct_change()).cumprod() - 1
        df['Rolling_Volatility_30'] = df['Daily_Return'].rolling(30).std()
        
        fig3 = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Cumulative Returns (%)",
                "Daily Returns Distribution",
                "Rolling 30-Day Volatility",
                "Monthly Returns Heatmap"
            ]
        )
        
        fig3.add_trace(go.Scatter(
            x=df['Date'], y=df['Cumulative_Return'] * 100,
            name='Cumulative Return', line=dict(color='#4CAF50', width=2),
            fill='tozeroy', fillcolor='rgba(76,175,80,0.1)'
        ), row=1, col=1)
        
        fig3.add_trace(go.Histogram(
            x=df['Daily_Return'].dropna(),
            name='Daily Returns', nbinsx=50,
            marker_color='#2196F3', opacity=0.8
        ), row=1, col=2)
        
        fig3.add_trace(go.Scatter(
            x=df['Date'], y=df['Rolling_Volatility_30'],
            name='30d Volatility', line=dict(color='#FF9800', width=2)
        ), row=2, col=1)
        
        # Monthly returns
        df['Month'] = df['Date'].dt.to_period('M')
        monthly_ret = df.groupby('Month')['Daily_Return'].sum().reset_index()
        monthly_ret['Month_str'] = monthly_ret['Month'].astype(str)
        
        fig3.add_trace(go.Bar(
            x=monthly_ret['Month_str'],
            y=monthly_ret['Daily_Return'],
            name='Monthly Returns',
            marker_color=['#4CAF50' if v >= 0 else '#F44336' for v in monthly_ret['Daily_Return']]
        ), row=2, col=2)
        
        fig3.update_layout(
            template='plotly_dark', height=700,
            title=dict(text=f"{ticker.upper()} - Returns & Volatility Analysis", font=dict(size=20)),
            showlegend=False
        )
        
        path3 = f"outputs/charts/{ticker}_returns_analysis.html"
        fig3.write_html(path3)
        chart_paths["returns_analysis"] = path3
        
        # ─── Chart 4: Volume Analysis ────────────────────────────────────────────
        df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
        df['Price_Volume'] = df['Close'] * df['Volume']
        
        fig4 = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=["Volume vs Price", "On-Balance Volume (OBV)"]
        )
        
        fig4.add_trace(go.Bar(
            x=df['Date'], y=df['Volume'],
            name='Volume',
            marker_color=['#26a69a' if c >= o else '#ef5350'
                         for c, o in zip(df['Close'], df['Open'])],
            opacity=0.7
        ), row=1, col=1)
        
        fig4.add_trace(go.Scatter(
            x=df['Date'], y=df['Volume_MA_20'],
            name='Vol MA20', line=dict(color='#FF9800', width=2)
        ), row=1, col=1)
        
        # OBV
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        fig4.add_trace(go.Scatter(
            x=df['Date'], y=obv,
            name='OBV', line=dict(color='#E91E63', width=2)
        ), row=2, col=1)
        
        fig4.update_layout(
            template='plotly_dark', height=600,
            title=dict(text=f"{ticker.upper()} - Volume Analysis", font=dict(size=20))
        )
        
        path4 = f"outputs/charts/{ticker}_volume_analysis.html"
        fig4.write_html(path4)
        chart_paths["volume_analysis"] = path4
        
        result = {
            "ticker": ticker.upper(),
            "charts_generated": len(chart_paths),
            "chart_paths": chart_paths,
            "stats": {
                "total_days": len(df),
                "avg_daily_return": round(float(df['Daily_Return'].mean()), 4),
                "daily_volatility": round(float(df['Daily_Return'].std()), 4),
                "annualized_volatility": round(float(df['Daily_Return'].std() * np.sqrt(252)), 4),
                "max_drawdown": round(float((df['Close'] / df['Close'].cummax() - 1).min() * 100), 2),
                "sharpe_ratio": round(
                    float(df['Daily_Return'].mean() / df['Daily_Return'].std() * np.sqrt(252)), 3
                ) if df['Daily_Return'].std() != 0 else 0,
            },
            "status": "success"
        }
        
        return json.dumps(result, default=str)
    
    except Exception as e:
        import traceback
        return json.dumps({"error": str(e), "traceback": traceback.format_exc(), "status": "failed"})
