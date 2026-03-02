# tools/__init__.py
from .stock_data_tool import scrape_stock_data
from .news_tool import scrape_stock_news
from .visualization_tool import generate_stock_visualizations
from .prediction_tool import predict_stock_price

# Create the ALL_TOOLS list that your agent is trying to import
ALL_TOOLS = [
    scrape_stock_data,
    scrape_stock_news,
    generate_stock_visualizations,
    predict_stock_price
]

# Also export individual tools
__all__ = [
    'scrape_stock_data',
    'scrape_stock_news',
    'generate_stock_visualizations',
    'predict_stock_price',
    'ALL_TOOLS'
]
