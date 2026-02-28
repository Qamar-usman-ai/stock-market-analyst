"""
Tool 1: Stock Data Scraper using yfinance
"""
import yfinance as yf
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Optional
from langchain_core.tools import tool


@tool
def scrape_stock_data(
    ticker: str,
    period_years: int = 2,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """
    Scrape historical stock data using yfinance.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
        period_years: Number of years of historical data to fetch (default: 2)
        start_date: Optional start date in YYYY-MM-DD format
        end_date: Optional end date in YYYY-MM-DD format
    
    Returns:
        JSON string with stock data summary and file path
    """
    try:
        stock = yf.Ticker(ticker.upper())
        
        # Set date range
        if start_date and end_date:
            hist = stock.history(start=start_date, end=end_date)
        else:
            end = datetime.now()
            start = end - timedelta(days=period_years * 365)
            hist = stock.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        
        if hist.empty:
            return json.dumps({"error": f"No data found for ticker {ticker}"})
        
        # Get company info
        info = stock.info
        company_name = info.get("longName", ticker)
        sector = info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")
        market_cap = info.get("marketCap", 0)
        pe_ratio = info.get("trailingPE", None)
        
        # Calculate basic statistics
        hist.reset_index(inplace=True)
        hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)
        
        current_price = float(hist['Close'].iloc[-1])
        start_price = float(hist['Close'].iloc[0])
        price_change_pct = ((current_price - start_price) / start_price) * 100
        
        avg_volume = float(hist['Volume'].mean())
        avg_price = float(hist['Close'].mean())
        max_price = float(hist['Close'].max())
        min_price = float(hist['Close'].min())
        
        # Moving averages
        hist['MA_20'] = hist['Close'].rolling(window=20).mean()
        hist['MA_50'] = hist['Close'].rolling(window=50).mean()
        hist['MA_200'] = hist['Close'].rolling(window=200).mean()
        
        # RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        hist['BB_Middle'] = hist['Close'].rolling(window=20).mean()
        hist['BB_Upper'] = hist['BB_Middle'] + 2 * hist['Close'].rolling(window=20).std()
        hist['BB_Lower'] = hist['BB_Middle'] - 2 * hist['Close'].rolling(window=20).std()
        
        # MACD
        exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
        hist['MACD'] = exp1 - exp2
        hist['Signal_Line'] = hist['MACD'].ewm(span=9, adjust=False).mean()
        
        # Save data
        import os
        os.makedirs("outputs/data", exist_ok=True)
        csv_path = f"outputs/data/{ticker}_historical.csv"
        hist.to_csv(csv_path, index=False)
        
        summary = {
            "ticker": ticker.upper(),
            "company_name": company_name,
            "sector": sector,
            "industry": industry,
            "market_cap": market_cap,
            "pe_ratio": pe_ratio,
            "data_points": len(hist),
            "date_range": {
                "start": str(hist['Date'].iloc[0]),
                "end": str(hist['Date'].iloc[-1])
            },
            "price_stats": {
                "current_price": round(current_price, 2),
                "start_price": round(start_price, 2),
                "price_change_pct": round(price_change_pct, 2),
                "avg_price": round(avg_price, 2),
                "max_price": round(max_price, 2),
                "min_price": round(min_price, 2),
            },
            "latest_indicators": {
                "rsi": round(float(hist['RSI'].iloc[-1]), 2) if not pd.isna(hist['RSI'].iloc[-1]) else None,
                "macd": round(float(hist['MACD'].iloc[-1]), 4) if not pd.isna(hist['MACD'].iloc[-1]) else None,
                "ma_20": round(float(hist['MA_20'].iloc[-1]), 2) if not pd.isna(hist['MA_20'].iloc[-1]) else None,
                "ma_50": round(float(hist['MA_50'].iloc[-1]), 2) if not pd.isna(hist['MA_50'].iloc[-1]) else None,
            },
            "avg_volume": int(avg_volume),
            "csv_path": csv_path,
            "status": "success"
        }
        
        return json.dumps(summary, default=str)
    
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker, "status": "failed"})
