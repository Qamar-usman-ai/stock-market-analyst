"""
Tool 2: Latest Stock Market News Scraper
"""
import requests
import json
import yfinance as yf
from datetime import datetime, timedelta
from langchain_core.tools import tool
from bs4 import BeautifulSoup


@tool
def scrape_stock_news(ticker: str, max_articles: int = 15) -> str:
    """
    Scrape latest news articles related to a stock ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        max_articles: Maximum number of articles to retrieve (default: 15)
    
    Returns:
        JSON string with news articles and sentiment analysis
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        company_name = info.get("longName", ticker)
        
        news_list = []
        
        # Method 1: yfinance built-in news
        yf_news = stock.news
        if yf_news:
            for article in yf_news[:max_articles]:
                pub_time = article.get("providerPublishTime", 0)
                if pub_time:
                    pub_date = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d %H:%M")
                else:
                    pub_date = "Unknown"
                
                news_list.append({
                    "title": article.get("title", ""),
                    "publisher": article.get("publisher", ""),
                    "link": article.get("link", ""),
                    "published_date": pub_date,
                    "source": "yfinance"
                })
        
        # Method 2: Yahoo Finance RSS feed
        try:
            rss_url = f"https://finance.yahoo.com/rss/headline?s={ticker.upper()}"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(rss_url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "xml")
                items = soup.find_all("item")
                for item in items[:5]:
                    title = item.find("title")
                    link = item.find("link")
                    pub_date = item.find("pubDate")
                    
                    if title:
                        news_list.append({
                            "title": title.text if title else "",
                            "publisher": "Yahoo Finance RSS",
                            "link": link.text if link else "",
                            "published_date": pub_date.text if pub_date else "Unknown",
                            "source": "yahoo_rss"
                        })
        except Exception:
            pass
        
        # Deduplicate by title
        seen_titles = set()
        unique_news = []
        for article in news_list:
            title_lower = article["title"].lower()
            if title_lower not in seen_titles and article["title"]:
                seen_titles.add(title_lower)
                unique_news.append(article)
        
        news_list = unique_news[:max_articles]
        
        # Simple keyword-based sentiment analysis
        positive_words = [
            "surge", "gain", "rise", "growth", "profit", "beat", "exceed",
            "strong", "bullish", "upgrade", "buy", "positive", "record", 
            "high", "rally", "soar", "jump", "boost", "improve", "success",
            "revenue", "earnings beat", "outperform"
        ]
        negative_words = [
            "fall", "drop", "decline", "loss", "miss", "weak", "bearish",
            "downgrade", "sell", "negative", "low", "crash", "plunge", "sink",
            "cut", "layoff", "lawsuit", "investigation", "concern", "risk",
            "debt", "bankruptcy", "disappointing", "underperform"
        ]
        
        sentiment_scores = []
        for article in news_list:
            title_lower = article["title"].lower()
            pos_count = sum(1 for w in positive_words if w in title_lower)
            neg_count = sum(1 for w in negative_words if w in title_lower)
            
            if pos_count > neg_count:
                sentiment = "positive"
                score = pos_count - neg_count
            elif neg_count > pos_count:
                sentiment = "negative"
                score = -(neg_count - pos_count)
            else:
                sentiment = "neutral"
                score = 0
            
            article["sentiment"] = sentiment
            article["sentiment_score"] = score
            sentiment_scores.append(score)
        
        # Overall market sentiment
        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            if avg_sentiment > 0.3:
                overall_sentiment = "Bullish"
            elif avg_sentiment < -0.3:
                overall_sentiment = "Bearish"
            else:
                overall_sentiment = "Neutral"
        else:
            overall_sentiment = "Neutral"
            avg_sentiment = 0
        
        positive_count = sum(1 for a in news_list if a["sentiment"] == "positive")
        negative_count = sum(1 for a in news_list if a["sentiment"] == "negative")
        neutral_count = sum(1 for a in news_list if a["sentiment"] == "neutral")
        
        result = {
            "ticker": ticker.upper(),
            "company_name": company_name,
            "articles_found": len(news_list),
            "overall_sentiment": overall_sentiment,
            "sentiment_breakdown": {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count,
                "avg_score": round(avg_sentiment, 3)
            },
            "news_articles": news_list,
            "status": "success"
        }
        
        # Save news to JSON
        import os
        os.makedirs("outputs/data", exist_ok=True)
        news_path = f"outputs/data/{ticker}_news.json"
        with open(news_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        
        result["news_file"] = news_path
        return json.dumps(result, default=str)
    
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker, "status": "failed"})
