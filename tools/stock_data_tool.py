import yfinance as yf
import pandas as pd
import json
import os
import requests
from langchain_core.tools import tool

# Mimic a real browser to prevent Azure from being blocked by Yahoo Finance
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
})

@tool
def scrape_stock_data(ticker: str, period_years: int = 2) -> str:
    """
    Fetches historical stock data and saves it to a UTF-8 encoded CSV.
    """
    try:
        symbol = ticker.upper().strip()
        
        # 1. Fetch data using the 'download' method (more reliable on Cloud IPs)
        df = yf.download(symbol, period=f"{period_years}y", session=session, progress=False)
        
        if df.empty:
            return json.dumps({
                "error": f"No data found for {symbol}. Azure IP may be blocked.",
                "status": "failed"
            }, ensure_ascii=False)

        # 2. Fix for 2026 yfinance MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 3. Clean and normalize data
        df.reset_index(inplace=True)
        # Ensure 'Date' is simple datetime without timezone for CSV compatibility
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

        # 4. Save to CSV with explicit UTF-8 encoding
        os.makedirs("outputs/data", exist_ok=True)
        csv_path = f"outputs/data/{symbol}_historical.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')

        # 5. Return JSON summary (ensure_ascii=False is the crucial fix for the codec error)
        summary = {
            "ticker": symbol,
            "current_price": round(float(df['Close'].iloc[-1]), 2),
            "csv_path": csv_path,
            "status": "success",
            "message": "Data saved successfully" # No emojis here to stay safe!
        }
        
        return json.dumps(summary, ensure_ascii=False)
    
    except Exception as e:
        return json.dumps({"error": str(e), "status": "failed"}, ensure_ascii=False)
