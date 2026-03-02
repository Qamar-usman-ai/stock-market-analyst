import yfinance as yf
import pandas as pd
import json
import os
import requests
from datetime import datetime, timedelta
from typing import Optional
from langchain_core.tools import tool

# ─── BROWSER EMULATION (ESSENTIAL FOR AZURE) ──────────────────────────────────
# This session mimics a real user to prevent Yahoo Finance from blocking Azure IPs
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Origin': 'https://finance.yahoo.com',
    'Referer': 'https://finance.yahoo.com/'
})

@tool
def scrape_stock_data(
    ticker: str,
    period_years: int = 2,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """
    Scrape historical stock data using yfinance with Azure-safe headers.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'V')
        period_years: Years of data to fetch (default: 2)
        start_date: Optional YYYY-MM-DD
        end_date: Optional YYYY-MM-DD
    """
    try:
        ticker_upper = ticker.upper()
        stock = yf.Ticker(ticker_upper, session=session)
        
        # 1. Fetch data using 'period' for higher reliability on cloud servers
        if start_date and end_date:
            hist = stock.history(start=start_date, end=end_date)
        else:
            hist = stock.history(period=f"{period_years}y")
        
        if hist.empty:
            return json.dumps({
                "error": f"Yahoo Finance returned no data for {ticker_upper}. This usually means the IP is blocked or the ticker is invalid.",
                "status": "failed"
            })

        # 2. FIX: FLATTEN MULTI-INDEX COLUMNS (Crucial for 2025/2026 yfinance)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
            
        # 3. CLEAN DATA
        hist.reset_index(inplace=True)
        # Remove timezones for compatibility with CSV and Excel
        if hist['Date'].dt.tz is not None:
            hist['Date'] = hist['Date'].dt.tz_localize(None)

        # 4. CALCULATE TECHNICAL INDICATORS
        # Moving Averages
        hist['MA_20'] = hist['Close'].rolling(window=20).mean()
        hist['MA_50'] = hist['Close'].rolling(window=50).mean()
        hist['MA_200'] = hist['Close'].rolling(window=200).mean()
        
        # RSI (Relative Strength Index)
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
        hist['MACD'] = exp1 - exp2
        hist['Signal_Line'] = hist['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        hist['BB_Middle'] = hist['Close'].rolling(window=20).mean()
        hist['BB_Upper'] = hist['BB_Middle'] + 2 * hist['Close'].rolling(window=20).std()
        hist['BB_Lower'] = hist['BB_Middle'] - 2 * hist['Close'].rolling(window=20).std()

        # 5. GET COMPANY INFO SAFELY
        info = stock.info
        current_price = float(hist['Close'].iloc[-1])
        start_price = float(hist['Close'].iloc[0])
        
        # 6. SAVE TO CSV (ENSURE DIRECTORY EXISTS)
        os.makedirs("outputs/data", exist_ok=True)
        csv_path = f"outputs/data/{ticker_upper}_historical.csv"
        hist.to_csv(csv_path, index=False)
        
        # 7. BUILD SUMMARY
        summary = {
            "ticker": ticker_upper,
            "company_name": info.get("longName", ticker_upper),
            "sector": info.get("sector", "N/A"),
            "price_stats": {
                "current_price": round(current_price, 2),
                "price_change_pct": round(((current_price - start_price) / start_price) * 100, 2),
                "avg_price": round(float(hist['Close'].mean()), 2)
            },
            "latest_indicators": {
                "rsi": round(float(hist['RSI'].iloc[-1]), 2) if not pd.isna(hist['RSI'].iloc[-1]) else None,
                "macd": round(float(hist['MACD'].iloc[-1]), 4) if not pd.isna(hist['MACD'].iloc[-1]) else None,
                "ma_50": round(float(hist['MA_50'].iloc[-1]), 2) if not pd.isna(hist['MA_50'].iloc[-1]) else None,
            },
            "csv_path": csv_path,
            "status": "success"
        }
        
        return json.dumps(summary)
    
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker, "status": "failed"})
