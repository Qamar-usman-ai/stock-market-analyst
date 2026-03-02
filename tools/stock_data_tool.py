import yfinance as yf
import pandas as pd
import json
import os
import requests
from langchain_core.tools import tool

# Enhanced Session to bypass Azure-specific blocks
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9',
})

@tool
def scrape_stock_data(ticker: str, period_years: int = 2) -> str:
    """
    Force-scrapes stock data using a download-first approach.
    """
    try:
        symbol = ticker.upper().strip()
        
        # 1. Use yf.download instead of Ticker.history (often works better on Cloud)
        # We fetch 2 years of data directly
        df = yf.download(symbol, period=f"{period_years}y", session=session, progress=False)
        
        if df.empty:
            return json.dumps({"error": f"Cloud Blocked: Yahoo refused data for {symbol}", "status": "failed"}, ensure_ascii=False)

        # 2. Fix the Multi-Index (yfinance 2025/2026 fix)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 3. Standardize column names and types
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

        # 4. Technical Indicators
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(lambda x: x>0, 0).rolling(14).mean() / 
                                      -df['Close'].diff().where(lambda x: x<0, 0).rolling(14).mean())))

        # 5. FORCE ABSOLUTE PATHS (Azure Fix)
        # This ensures the Scraper and Predictor are talking to the EXACT same folder
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "outputs", "data")
        os.makedirs(data_dir, exist_ok=True)
        
        csv_path = os.path.join(data_dir, f"{symbol}_historical.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')

        return json.dumps({
            "ticker": symbol,
            "csv_path": csv_path, # Returning the absolute path
            "current_price": round(float(df['Close'].iloc[-1]), 2),
            "status": "success",
            "message": f"Successfully saved {len(df)} rows to {csv_path} ✅"
        }, ensure_ascii=False)
    
    except Exception as e:
        return json.dumps({"error": str(e), "status": "failed"}, ensure_ascii=False)
