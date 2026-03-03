import os, json, logging, requests, feedparser
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
FINNHUB_BASE = "https://finnhub.io/api/v1"

POSITIVE = {"beat","beats","surge","soars","rally","record","profit","growth","buyback",
            "dividend","upgrade","outperform","bullish","strong","gain","rises","expansion",
            "acquisition","breakthrough","innovation","exceed","positive","revenue","earnings"}
NEGATIVE = {"miss","misses","plunge","crash","loss","losses","decline","downgrade",
            "underperform","bearish","weak","lawsuit","fraud","recall","layoffs","bankruptcy",
            "investigation","fine","penalty","warning","disappointing","cut","cuts","drop"}

def _score(text):
    words = set(text.lower().split())
    pos = len(words & POSITIVE)
    neg = len(words & NEGATIVE)
    total = pos + neg or 1
    score = round((pos - neg) / total, 3)
    label = "positive" if score > 0.1 else ("negative" if score < -0.1 else "neutral")
    return {"score": score, "label": label}

def _finnhub_news(symbol, api_key):
    today = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    try:
        r = requests.get(f"{FINNHUB_BASE}/company-news",
            params={"symbol": symbol, "from": from_date, "to": today, "token": api_key},
            timeout=15)
        r.raise_for_status()
        items = r.json() or []
        print(f"[NEWS] Finnhub: {len(items)} for {symbol}")
        return [{"title": i.get("headline",""), "summary": i.get("summary",""),
                 "link": i.get("url",""),
                 "published": datetime.fromtimestamp(i.get("datetime",0)).strftime("%Y-%m-%d %H:%M"),
                 "source": i.get("source","Finnhub")} for i in items[:20]]
    except Exception as e:
        print(f"[NEWS] Finnhub failed: {e}")
        return []

def _google_rss(symbol):
    try:
        url = f"https://news.google.com/rss/search?q={symbol}+stock+market&hl=en-US&gl=US&ceid=US:en"
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        feed = feedparser.parse(r.content)
        print(f"[NEWS] Google RSS: {len(feed.entries)} for {symbol}")
        return [{"title": e.get("title",""), "summary": e.get("summary",""),
                 "link": e.get("link",""), "published": e.get("published",""),
                 "source": "Google News"} for e in feed.entries[:15]]
    except Exception as e:
        print(f"[NEWS] Google RSS failed: {e}")
        return []

def _fetch_stock_news(ticker: str, finnhub_api_key: str) -> dict:
    symbol = ticker.upper().strip()
    key_info = "YES("+finnhub_api_key[:6]+")" if finnhub_api_key else "EMPTY"
    print(f"[NEWS] {symbol} | key={key_info}")
    articles, source = [], ""
    if finnhub_api_key:
        articles = _finnhub_news(symbol, finnhub_api_key)
        if articles:
            source = "finnhub"
    if not articles:
        articles = _google_rss(symbol)
        source = "google_rss"
    if not articles:
        return {"symbol": symbol, "news_count": 0, "articles": [],
                "sentiment_summary": {"overall": "neutral", "avg_score": 0}, "status": "empty"}
    enriched = [{**a, "sentiment": _score(f"{a['title']} {a['summary']}")} for a in articles]
    scores = [a["sentiment"]["score"] for a in enriched]
    avg = round(sum(scores) / len(scores), 3)
    pos = sum(1 for s in scores if s > 0.1)
    neg = sum(1 for s in scores if s < -0.1)
    overall = "positive" if avg > 0.1 else ("negative" if avg < -0.1 else "neutral")
    os.makedirs("outputs/data", exist_ok=True)
    with open(f"outputs/data/{symbol}_news.json", "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    return {"symbol": symbol, "source": source, "status": "success",
            "news_count": len(enriched), "articles": enriched,
            "sentiment_summary": {"overall": overall, "avg_score": avg,
                "positive_articles": pos, "negative_articles": neg,
                "neutral_articles": len(scores) - pos - neg}}
