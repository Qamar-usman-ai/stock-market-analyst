"""
Tool 1: Stock Data - FINAL FIX
================================
Root cause: os.environ does not reliably propagate across async contexts
on Azure Container Apps.

Fix: Accept finnhub_api_key as a direct tool parameter instead of
reading from environment. The key is passed straight from main.py
through the agent to the tool call.
"""

import os
import json
import math
import time
import logging
import requests
import pandas as pd
import numpy as np
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

FINNHUB_BASE = "https://finnhub.io/api/v1"
AV_BASE      = "https://www.alphavantage.co/query"


def _get(url: str, params: dict) -> dict:
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _finnhub_candles(symbol: str, period_years: int, api_key: str) -> pd.DataFrame:
    now     = int(time.time())
    from_ts = now - (period_years * 365 * 24 * 3600)
    data = _get(f"{FINNHUB_BASE}/stock/candle", {
        "symbol": symbol, "resolution": "D",
        "from": from_ts, "to": now, "token": api_key,
    })
    if data.get("s") != "ok":
        raise ValueError(f"Finnhub status: {data.get('s')} — {data}")
    df = pd.DataFrame({
        "Date":   pd.to_datetime(data["t"], unit="s").tz_localize(None),
        "Open":   data["o"], "High": data["h"],
        "Low":    data["l"], "Close": data["c"], "Volume": data["v"],
    })
    return df.sort_values("Date").reset_index(drop=True)


def _finnhub_profile(symbol: str, api_key: str) -> dict:
    try:
        return _get(f"{FINNHUB_BASE}/stock/profile2", {"symbol": symbol, "token": api_key})
    except Exception as e:
        logger.warning(f"Profile failed: {e}")
        return {}


def _finnhub_quote(symbol: str, api_key: str) -> dict:
    try:
        return _get(f"{FINNHUB_BASE}/quote", {"symbol": symbol, "token": api_key})
    except Exception as e:
        logger.warning(f"Quote failed: {e}")
        return {}


def _av_candles(symbol: str, api_key: str) -> pd.DataFrame:
    data = _get(AV_BASE, {
        "function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": symbol,
        "outputsize": "full", "apikey": api_key,
    })
    ts = data.get("Time Series (Daily)")
    if not ts:
        raise ValueError(f"Alpha Vantage error: {data}")
    rows = [{
        "Date": pd.to_datetime(d),
        "Open": float(v["1. open"]), "High": float(v["2. high"]),
        "Low":  float(v["3. low"]),  "Close": float(v["5. adjusted close"]),
        "Volume": float(v["6. volume"]),
    } for d, v in ts.items()]
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def _rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return (100 - 100 / (1 + g / l.replace(0, np.nan))).round(2)


def _macd(s: pd.Series):
    e12 = s.ewm(span=12, adjust=False).mean()
    e26 = s.ewm(span=26, adjust=False).mean()
    m   = e12 - e26
    sig = m.ewm(span=9, adjust=False).mean()
    return m.round(4), sig.round(4), (m - sig).round(4)


def _bollinger(s: pd.Series, p: int = 20):
    sma = s.rolling(p).mean()
    std = s.rolling(p).std()
    return (sma + 2*std).round(4), sma.round(4), (sma - 2*std).round(4)


def _safe(v):
    if v is None: return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
    return v.item() if hasattr(v, "item") else v


@tool
def scrape_stock_data(ticker: str, finnhub_api_key: str, period_years: int = 2) -> str:
    """
    Fetches historical OHLCV data and technical indicators for a stock.
    Uses Finnhub API (primary) — works on Azure/cloud.

    Args:
        ticker: Stock symbol e.g. 'AAPL', 'TSLA', 'MSFT'
        finnhub_api_key: Finnhub API key passed directly from the request
        period_years: Years of history to fetch (default 2)
    """
    symbol      = ticker.upper().strip()
    # Try passed key first, then fall back to env var as last resort
    finnhub_key = finnhub_api_key.strip() if finnhub_api_key else os.environ.get("FINNHUB_API_KEY", "").strip()
    av_key      = os.environ.get("ALPHAVANTAGE_API_KEY", "").strip()

    logger.info(f"scrape_stock_data: {symbol} | finnhub_key present: {bool(finnhub_key)}")

    if not finnhub_key and not av_key:
        return json.dumps({
            "error": "No API key available. finnhub_api_key parameter is empty.",
            "status": "failed", "ticker": symbol
        }, ensure_ascii=False)

    df     = None
    source = ""

    if finnhub_key:
        try:
            df     = _finnhub_candles(symbol, period_years, finnhub_key)
            source = "finnhub"
            logger.info(f"Finnhub OK: {len(df)} rows for {symbol}")
        except Exception as e:
            logger.warning(f"Finnhub failed: {e}")

    if (df is None or df.empty) and av_key:
        try:
            df     = _av_candles(symbol, av_key)
            cutoff = pd.Timestamp.now() - pd.DateOffset(years=period_years)
            df     = df[df["Date"] >= cutoff].reset_index(drop=True)
            source = "alphavantage"
        except Exception as e:
            logger.error(f"Alpha Vantage also failed: {e}")

    if df is None or df.empty:
        return json.dumps({
            "error": f"No data for {symbol}. Finnhub key may be invalid. Test it at: https://finnhub.io/api/v1/quote?symbol=AAPL&token={finnhub_key[:8]}...",
            "status": "failed", "ticker": symbol
        }, ensure_ascii=False)

    close = df["Close"]
    df["RSI"]         = _rsi(close)
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = _macd(close)
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"]   = _bollinger(close)
    df["MA_20"]  = close.rolling(20).mean().round(4)
    df["MA_50"]  = close.rolling(50).mean().round(4)
    df["MA_200"] = close.rolling(200).mean().round(4)

    company_info = {}
    if finnhub_key:
        profile      = _finnhub_profile(symbol, finnhub_key)
        quote        = _finnhub_quote(symbol, finnhub_key)
        company_info = {
            "longName":     profile.get("name"),
            "sector":       profile.get("finnhubIndustry"),
            "marketCap":    profile.get("marketCapitalization"),
            "currentPrice": quote.get("c"),
            "change":       quote.get("d"),
            "changePct":    quote.get("dp"),
            "prevClose":    quote.get("pc"),
        }

    os.makedirs("outputs/data", exist_ok=True)
    csv_path = f"outputs/data/{symbol}_historical.csv"
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest
    pct    = round((float(latest["Close"]) - float(prev["Close"])) / float(prev["Close"]) * 100, 2)

    return json.dumps({
        "ticker": symbol, "source": source, "status": "success",
        "current_price": _safe(latest["Close"]), "price_change_pct": pct,
        "latest_rsi":    _safe(latest["RSI"]),   "latest_macd": _safe(latest["MACD"]),
        "ma_20": _safe(latest["MA_20"]), "ma_50": _safe(latest["MA_50"]),
        "ma_200": _safe(latest["MA_200"]),
        "52w_high": _safe(float(close.max())), "52w_low": _safe(float(close.min())),
        "data_points": len(df), "csv_path": csv_path,
        "company_info": company_info,
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
    }, ensure_ascii=False)
