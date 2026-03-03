"""
Tool 1: Stock Data Scraper - CLOUD FIXED
========================================
PROBLEM: yfinance is blocked by Yahoo Finance on ALL cloud IPs (Azure, AWS, GCP).
No User-Agent trick fixes this. The IP itself gets rejected.

SOLUTION: Replace yfinance with real APIs that work on cloud:
  Primary:  Finnhub  (free, 60 req/min, no IP blocking)
  Fallback: Alpha Vantage (free, 25 req/day)

SETUP: Add these to your environment variables / Azure App Settings:
  FINNHUB_API_KEY=your_key_here       <- get free at https://finnhub.io
  ALPHAVANTAGE_API_KEY=your_key_here  <- get free at https://alphavantage.co
"""

import os
import json
import math
import logging
import requests
import pandas as pd
import numpy as np
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
AV_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
FINNHUB_BASE = "https://finnhub.io/api/v1"
AV_BASE = "https://www.alphavantage.co/query"


# ─── HTTP helper ────────────────────────────────────────────────────────────────
def _get(url: str, params: dict) -> dict:
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


# ─── Finnhub data fetch ──────────────────────────────────────────────────────────
def _finnhub_candles(symbol: str, period_years: int) -> pd.DataFrame:
    """Fetch OHLCV from Finnhub /stock/candle (works on all cloud IPs)."""
    import time as _time
    now = int(_time.time())
    from_ts = now - (period_years * 365 * 24 * 3600)

    data = _get(f"{FINNHUB_BASE}/stock/candle", {
        "symbol": symbol,
        "resolution": "D",
        "from": from_ts,
        "to": now,
        "token": FINNHUB_KEY,
    })

    if data.get("s") != "ok":
        raise ValueError(f"Finnhub returned status: {data.get('s')} — {data}")

    df = pd.DataFrame({
        "Date":   pd.to_datetime(data["t"], unit="s").tz_localize(None),
        "Open":   data["o"],
        "High":   data["h"],
        "Low":    data["l"],
        "Close":  data["c"],
        "Volume": data["v"],
    })
    return df.sort_values("Date").reset_index(drop=True)


def _finnhub_quote(symbol: str) -> dict:
    """Fetch live quote from Finnhub."""
    return _get(f"{FINNHUB_BASE}/quote", {"symbol": symbol, "token": FINNHUB_KEY})


def _finnhub_profile(symbol: str) -> dict:
    """Fetch company profile from Finnhub."""
    return _get(f"{FINNHUB_BASE}/stock/profile2", {"symbol": symbol, "token": FINNHUB_KEY})


# ─── Alpha Vantage fallback ──────────────────────────────────────────────────────
def _av_candles(symbol: str) -> pd.DataFrame:
    """Fallback: fetch daily OHLCV from Alpha Vantage."""
    data = _get(AV_BASE, {
        "function":   "TIME_SERIES_DAILY_ADJUSTED",
        "symbol":     symbol,
        "outputsize": "full",
        "apikey":     AV_KEY,
    })

    ts = data.get("Time Series (Daily)")
    if not ts:
        raise ValueError(f"Alpha Vantage error: {data.get('Information') or data.get('Note') or data}")

    rows = []
    for date_str, vals in ts.items():
        rows.append({
            "Date":   pd.to_datetime(date_str),
            "Open":   float(vals["1. open"]),
            "High":   float(vals["2. high"]),
            "Low":    float(vals["3. low"]),
            "Close":  float(vals["5. adjusted close"]),
            "Volume": float(vals["6. volume"]),
        })

    df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    return df


# ─── Technical Indicators ────────────────────────────────────────────────────────
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).round(2)


def _macd(series: pd.Series):
    e12 = series.ewm(span=12, adjust=False).mean()
    e26 = series.ewm(span=26, adjust=False).mean()
    m = e12 - e26
    s = m.ewm(span=9, adjust=False).mean()
    return m.round(4), s.round(4), (m - s).round(4)


def _bollinger(series: pd.Series, period: int = 20):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (sma + 2*std).round(4), sma.round(4), (sma - 2*std).round(4)


def _safe(val):
    """Convert numpy/nan to plain Python for JSON serialization."""
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    if hasattr(val, "item"):
        return val.item()
    return val


# ─── Main Tool ───────────────────────────────────────────────────────────────────
@tool
def scrape_stock_data(ticker: str, period_years: int = 2) -> str:
    """
    Fetches historical OHLCV data and technical indicators for a stock.
    Uses Finnhub (primary) and Alpha Vantage (fallback). Works on Azure/cloud.

    Args:
        ticker: Stock symbol e.g. 'AAPL', 'TSLA', 'MSFT'
        period_years: How many years of history to fetch (default 2)
    """
    symbol = ticker.upper().strip()
    df = None
    source = ""

    # ── 1. Fetch OHLCV ─────────────────────────────────────────────────────────
    if FINNHUB_KEY:
        try:
            df = _finnhub_candles(symbol, period_years)
            source = "finnhub"
            logger.info(f"Finnhub: got {len(df)} rows for {symbol}")
        except Exception as e:
            logger.warning(f"Finnhub failed ({e}), trying Alpha Vantage...")

    if df is None or df.empty:
        if AV_KEY:
            try:
                df = _av_candles(symbol)
                # Trim to requested period
                cutoff = pd.Timestamp.now() - pd.DateOffset(years=period_years)
                df = df[df["Date"] >= cutoff].reset_index(drop=True)
                source = "alphavantage"
                logger.info(f"Alpha Vantage: got {len(df)} rows for {symbol}")
            except Exception as e:
                logger.error(f"Alpha Vantage also failed: {e}")
                return json.dumps({"error": str(e), "status": "failed", "symbol": symbol}, ensure_ascii=False)
        else:
            return json.dumps({
                "error": "No API keys configured. Set FINNHUB_API_KEY and/or ALPHAVANTAGE_API_KEY in environment variables.",
                "status": "failed",
                "symbol": symbol
            }, ensure_ascii=False)

    if df is None or df.empty:
        return json.dumps({"error": f"No data found for {symbol}", "status": "failed"}, ensure_ascii=False)

    # ── 2. Technical Indicators ────────────────────────────────────────────────
    close = df["Close"]
    df["RSI"]         = _rsi(close)
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = _macd(close)
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"]   = _bollinger(close)
    df["MA_20"]  = close.rolling(20).mean().round(4)
    df["MA_50"]  = close.rolling(50).mean().round(4)
    df["MA_200"] = close.rolling(200).mean().round(4)

    # ── 3. Fetch Company Info (Finnhub only, non-fatal if fails) ───────────────
    company_info = {}
    if FINNHUB_KEY and source == "finnhub":
        try:
            profile = _finnhub_profile(symbol)
            quote   = _finnhub_quote(symbol)
            company_info = {
                "longName":      profile.get("name"),
                "sector":        profile.get("finnhubIndustry"),
                "marketCap":     profile.get("marketCapitalization"),
                "currentPrice":  quote.get("c"),
                "change":        quote.get("d"),
                "changePct":     quote.get("dp"),
                "52w_high":      quote.get("h"),
                "52w_low":       quote.get("l"),
                "open":          quote.get("o"),
                "prevClose":     quote.get("pc"),
            }
        except Exception as e:
            logger.warning(f"Could not fetch company info: {e}")

    # ── 4. Save CSV ────────────────────────────────────────────────────────────
    os.makedirs("outputs/data", exist_ok=True)
    csv_path = f"outputs/data/{symbol}_historical.csv"
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # ── 5. Build summary ───────────────────────────────────────────────────────
    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest
    pct_chg = round((float(latest["Close"]) - float(prev["Close"])) / float(prev["Close"]) * 100, 2)

    summary = {
        "ticker":          symbol,
        "source":          source,
        "current_price":   _safe(latest["Close"]),
        "price_change_pct": pct_chg,
        "latest_rsi":      _safe(latest["RSI"]),
        "latest_macd":     _safe(latest["MACD"]),
        "ma_20":           _safe(latest["MA_20"]),
        "ma_50":           _safe(latest["MA_50"]),
        "ma_200":          _safe(latest["MA_200"]),
        "52w_high":        _safe(float(close.max())),
        "52w_low":         _safe(float(close.min())),
        "data_points":     len(df),
        "csv_path":        csv_path,
        "company_info":    company_info,
        "status":          "success",
        "ohlcv": {
            "dates":  df["Date"].dt.strftime("%Y-%m-%d").tolist(),
            "open":   [_safe(v) for v in df["Open"]],
            "high":   [_safe(v) for v in df["High"]],
            "low":    [_safe(v) for v in df["Low"]],
            "close":  [_safe(v) for v in df["Close"]],
            "volume": [_safe(v) for v in df["Volume"]],
        },
        "indicators": {
            "rsi":         [_safe(v) for v in df["RSI"]],
            "macd":        [_safe(v) for v in df["MACD"]],
            "macd_signal": [_safe(v) for v in df["MACD_Signal"]],
            "macd_hist":   [_safe(v) for v in df["MACD_Hist"]],
            "bb_upper":    [_safe(v) for v in df["BB_Upper"]],
            "bb_mid":      [_safe(v) for v in df["BB_Mid"]],
            "bb_lower":    [_safe(v) for v in df["BB_Lower"]],
            "ma_20":       [_safe(v) for v in df["MA_20"]],
            "ma_50":       [_safe(v) for v in df["MA_50"]],
            "ma_200":      [_safe(v) for v in df["MA_200"]],
        },
    }

    return json.dumps(summary, ensure_ascii=False)
