"""
Tool 2: News + Sentiment - CLOUD FIXED
=======================================
PROBLEM: yfinance .news scrapes Yahoo Finance which blocks cloud IPs.

SOLUTION: Use Finnhub /company-news API (free, works on Azure/cloud).
Fallback to Google News RSS (no API key needed).

SETUP: Set FINNHUB_API_KEY in your environment variables.
"""

import os
import json
import logging
import requests
import feedparser
from datetime import datetime, timedelta
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
FINNHUB_BASE = "https://finnhub.io/api/v1"

POSITIVE_WORDS = {
    "beat", "beats", "surge", "soars", "rally", "record", "profit", "growth",
    "buyback", "dividend", "upgrade", "outperform", "bullish", "strong", "gain",
    "rises", "expansion", "acquisition", "breakthrough", "innovation", "exceed",
}
NEGATIVE_WORDS = {
    "miss", "plunge", "crash", "loss", "decline", "downgrade", "underperform",
    "bearish", "weak", "lawsuit", "fraud", "recall", "layoffs", "bankruptcy",
    "investigation", "fine", "penalty", "warning", "disappointing", "cut",
}


def _score_sentiment(text: str) -> dict:
    words = set(text.lower().split())
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    total = pos + neg or 1
    score = round((pos - neg) / total, 3)
    label = "positive" if score > 0.1 else ("negative" if score < -0.1 else "neutral")
    return {"score": score, "label": label}


def _finnhub_news(symbol: str) -> list:
    """Fetch company news from Finnhub (works on Azure)."""
    today = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    resp = requests.get(f"{FINNHUB_BASE}/company-news", params={
        "symbol": symbol,
        "from":   from_date,
        "to":     today,
        "token":  FINNHUB_KEY,
    }, timeout=15)
    resp.raise_for_status()
    items = resp.json() or []

    return [{
        "title":     i.get("headline", ""),
        "summary":   i.get("summary", ""),
        "link":      i.get("url", ""),
        "published": datetime.fromtimestamp(i.get("datetime", 0)).strftime("%Y-%m-%d %H:%M"),
        "source":    i.get("source", "Finnhub"),
    } for i in items[:20]]


def _google_rss_news(symbol: str) -> list:
    """Fallback: Google News RSS — no API key, works on cloud."""
    url = f"https://news.google.com/rss/search?q={symbol}+stock+market&hl=en-US&gl=US&ceid=US:en"
    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (compatible; RSS reader)"
        })
        feed = feedparser.parse(resp.content)
        return [{
            "title":     e.get("title", ""),
            "summary":   e.get("summary", ""),
            "link":      e.get("link", ""),
            "published": e.get("published", ""),
            "source":    "Google News",
        } for e in feed.entries[:15]]
    except Exception as e:
        logger.warning(f"Google RSS failed: {e}")
        return []


@tool
def scrape_stock_news(ticker: str) -> str:
    """
    Fetches recent news and performs sentiment analysis for a stock.
    Uses Finnhub news API (primary) and Google News RSS (fallback).
    Works on Azure/cloud — no Yahoo Finance scraping.

    Args:
        ticker: Stock symbol e.g. 'AAPL', 'TSLA'
    """
    symbol = ticker.upper().strip()
    articles = []
    source = ""

    if FINNHUB_KEY:
        try:
            articles = _finnhub_news(symbol)
            source = "finnhub"
        except Exception as e:
            logger.warning(f"Finnhub news failed: {e}")

    if not articles:
        articles = _google_rss_news(symbol)
        source = "google_rss"

    if not articles:
        return json.dumps({
            "symbol": symbol,
            "news_count": 0,
            "articles": [],
            "sentiment_summary": {"overall": "neutral", "avg_score": 0},
            "warning": "No news found. Set FINNHUB_API_KEY for reliable news.",
            "status": "empty"
        }, ensure_ascii=False)

    # Score sentiment
    enriched = []
    for art in articles:
        text = f"{art['title']} {art['summary']}"
        enriched.append({**art, "sentiment": _score_sentiment(text)})

    scores = [a["sentiment"]["score"] for a in enriched]
    avg_score = round(sum(scores) / len(scores), 3)
    pos = sum(1 for s in scores if s > 0.1)
    neg = sum(1 for s in scores if s < -0.1)

    overall = "positive" if avg_score > 0.1 else ("negative" if avg_score < -0.1 else "neutral")

    return json.dumps({
        "symbol":    symbol,
        "source":    source,
        "news_count": len(enriched),
        "articles":  enriched,
        "sentiment_summary": {
            "overall":           overall,
            "avg_score":         avg_score,
            "positive_articles": pos,
            "negative_articles": neg,
            "neutral_articles":  len(scores) - pos - neg,
        },
        "status": "success"
    }, ensure_ascii=False)
