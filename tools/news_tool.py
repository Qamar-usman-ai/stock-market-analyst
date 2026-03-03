"""
Tool 2: News + Sentiment - FIXED
Critical fix: read FINNHUB_API_KEY inside function, not at module level.
"""
import os, json, logging, requests, feedparser
from datetime import datetime, timedelta
from langchain_core.tools import tool

logger = logging.getLogger(__name__)
FINNHUB_BASE = "https://finnhub.io/api/v1"

POSITIVE_WORDS = {"beat","beats","surge","soars","rally","record","profit","growth","buyback","dividend","upgrade","outperform","bullish","strong","gain","rises","expansion","acquisition","breakthrough","innovation","exceed","exceeds","positive","revenue","earnings","partnership","launch"}
NEGATIVE_WORDS = {"miss","misses","plunge","crash","loss","losses","decline","downgrade","underperform","bearish","weak","lawsuit","fraud","recall","layoffs","bankruptcy","investigation","fine","penalty","warning","disappointing","cut","cuts","drop","fell","falling","risk","concern"}

def _score(text):
    words = set(text.lower().split())
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    total = pos + neg or 1
    score = round((pos - neg) / total, 3)
    label = "positive" if score > 0.1 else ("negative" if score < -0.1 else "neutral")
    return {"score": score, "label": label, "positive_hits": pos, "negative_hits": neg}

def _finnhub_news(symbol, api_key):
    today = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    try:
        resp = requests.get(f"{FINNHUB_BASE}/company-news",
            params={"symbol": symbol, "from": from_date, "to": today, "token": api_key},
            timeout=15)
        resp.raise_for_status()
        items = resp.json() or []
        logger.info(f"Finnhub news: {len(items)} for {symbol}")
        return [{"title": i.get("headline",""), "summary": i.get("summary",""),
                 "link": i.get("url",""),
                 "published": datetime.fromtimestamp(i.get("datetime",0)).strftime("%Y-%m-%d %H:%M"),
                 "source": i.get("source","Finnhub")} for i in items[:20]]
    except Exception as e:
        logger.warning(f"Finnhub news failed: {e}")
        return []

def _google_rss(symbol):
    try:
        url = f"https://news.google.com/rss/search?q={symbol}+stock+market&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        feed = feedparser.parse(resp.content)
        return [{"title": e.get("title",""), "summary": e.get("summary",""),
                 "link": e.get("link",""), "published": e.get("published",""),
                 "source": "Google News"} for e in feed.entries[:15]]
    except Exception as e:
        logger.warning(f"Google RSS failed: {e}")
        return []

@tool
def scrape_stock_news(ticker: str) -> str:
    """
    Fetches recent news and sentiment analysis for a stock ticker.
    Uses Finnhub (primary) and Google News RSS (fallback). Works on Azure.

    Args:
        ticker: Stock symbol e.g. 'AAPL', 'TSLA'
    """
    symbol = ticker.upper().strip()

    # CRITICAL FIX: read key inside function, not at module level
    finnhub_key = os.environ.get("FINNHUB_API_KEY", "").strip()
    logger.info(f"scrape_stock_news: {symbol} | key present: {bool(finnhub_key)}")

    articles = []
    source = ""

    if finnhub_key:
        articles = _finnhub_news(symbol, finnhub_key)
        if articles:
            source = "finnhub"

    if not articles:
        logger.info(f"Falling back to Google RSS for {symbol}")
        articles = _google_rss(symbol)
        source = "google_rss"

    if not articles:
        return json.dumps({"symbol": symbol, "news_count": 0, "articles": [],
            "sentiment_summary": {"overall": "neutral", "avg_score": 0},
            "status": "empty"}, ensure_ascii=False)

    enriched = [{**a, "sentiment": _score(f"{a['title']} {a['summary']}")} for a in articles]
    scores = [a["sentiment"]["score"] for a in enriched]
    avg = round(sum(scores) / len(scores), 3)
    pos = sum(1 for s in scores if s > 0.1)
    neg = sum(1 for s in scores if s < -0.1)
    overall = "positive" if avg > 0.1 else ("negative" if avg < -0.1 else "neutral")

    os.makedirs("outputs/data", exist_ok=True)
    with open(f"outputs/data/{symbol}_news.json", "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    return json.dumps({
        "symbol": symbol, "source": source, "status": "success",
        "news_count": len(enriched), "articles": enriched,
        "sentiment_summary": {"overall": overall, "avg_score": avg,
            "positive_articles": pos, "negative_articles": neg,
            "neutral_articles": len(scores) - pos - neg},
    }, ensure_ascii=False)
