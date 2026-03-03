# 📈 Stock Market AI Analyst

An AI-powered stock market analysis agent built with **LangGraph**, **FastAPI**, **ARIMA/SARIMA** predictions, and a modern dark-themed frontend. Uses the **Groq API** (LLaMA 3.3 70B) for LLM reasoning and **Finnhub API** for reliable cloud-compatible stock data.

> ✅ **Cloud Ready** — Works on Azure, AWS, GCP. No Yahoo Finance scraping.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (HTML/JS)                        │
│     User enters Groq key + Finnhub key, selects stock         │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP POST /api/analyze
┌─────────────────────▼───────────────────────────────────────┐
│                  FastAPI Backend (main.py)                    │
│            Background job queue + status polling              │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              LangGraph Agent (agents/stock_agent.py)          │
│              Orchestrates 4 tools in sequence                  │
└──────┬──────────────┬──────────────┬──────────────┬─────────┘
       │              │              │              │
  Tool 1          Tool 2         Tool 3          Tool 4
  Finnhub        Finnhub        Plotly          SARIMA
  Stock Data     News+Sentiment  Charts         Prediction
```

---

## 📁 Project Structure

```
stock-market-analyst/
│
├── main.py                        # FastAPI app + API routes
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container image
├── docker-compose.yml             # Local multi-container setup
├── azure-deploy.yaml              # Azure deployment config
├── README.md                      # This file
│
├── agents/
│   ├── __init__.py
│   └── stock_agent.py             # LangGraph StateGraph agent
│
├── tools/
│   ├── __init__.py
│   ├── stock_data_tool.py         # Tool 1: Finnhub stock data + indicators
│   ├── news_tool.py               # Tool 2: Finnhub news + sentiment
│   ├── visualization_tool.py      # Tool 3: Plotly charts (6 charts)
│   └── prediction_tool.py         # Tool 4: ARIMA/SARIMA forecast
│
├── static/
│   └── index.html                 # Frontend SPA (dark theme)
│
└── outputs/                       # Generated files (gitignored)
    ├── data/                      # CSV + JSON data files
    ├── charts/                    # Interactive HTML charts
    └── *.md                       # Investment reports
```

---

## 🛠️ The 4 AI Tools

| # | Tool | File | Description |
|---|------|------|-------------|
| 1 | `scrape_stock_data` | `tools/stock_data_tool.py` | Fetches OHLCV from **Finnhub API**, computes RSI, MACD, Bollinger Bands, Moving Averages |
| 2 | `scrape_stock_news` | `tools/news_tool.py` | Gets latest news from **Finnhub company-news API** + Google News RSS fallback, keyword sentiment analysis |
| 3 | `generate_stock_visualizations` | `tools/visualization_tool.py` | 6 interactive Plotly charts: candlestick, technical indicators, returns, volume, prediction, model accuracy |
| 4 | `predict_stock_price` | `tools/prediction_tool.py` | Auto-selects ARIMA/SARIMA params via AIC grid search, train/val/test split, full metrics |

---

## 🔑 API Keys Required

This app needs **2 free API keys** — both entered directly in the frontend UI. No `.env` files or Azure App Settings needed.

### 1. Groq API Key (LLM)
- Go to **[console.groq.com](https://console.groq.com)**
- Sign up free → Create API Key
- Used for: LLaMA 3.3 70B reasoning and investment report generation

### 2. Finnhub API Key (Stock Data)
- Go to **[finnhub.io/register](https://finnhub.io/register)**
- Sign up free → copy API key from dashboard
- Free tier: **60 requests/minute**
- Used for: OHLCV price history, company news, company profile
- ✅ **Works on Azure/cloud** — proper REST API, no IP blocking

> ⚠️ **Why not yfinance?** yfinance scrapes Yahoo Finance which **blocks all cloud datacenter IPs** (Azure, AWS, GCP). No User-Agent trick fixes this — the IP itself is rejected. Finnhub is the correct solution.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Groq API Key (free)
- Finnhub API Key (free)

### Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/Qamar-usman-ai/stock-market-analyst.git
cd stock-market-analyst

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
uvicorn main:app --reload --port 8000

# 5. Open browser at http://localhost:8000
# Enter your Groq + Finnhub keys in the UI and start analyzing
```

### Docker Setup

```bash
# Build and run
docker-compose up --build

# Or with Docker directly
docker build -t stock-analyst .
docker run -p 8000:8000 stock-analyst
```

---

## 📦 Requirements

```txt
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
langgraph>=0.1.0
langchain>=0.2.0
langchain-groq>=0.1.0
finnhub-python>=2.4.20
feedparser>=6.0.11
pandas>=2.2.0
numpy>=1.26.0
statsmodels>=0.14.0
pmdarima>=2.0.4
plotly>=5.22.0
python-dotenv>=1.0.0
aiofiles>=23.2.1
```

---

## ☁️ Deploy to Azure

### Option 1: Azure Container Apps (Recommended)

```bash
# 1. Login
az login

# 2. Create resource group
az group create --name stock-analyst-rg --location eastus

# 3. Create Azure Container Registry
az acr create --resource-group stock-analyst-rg \
  --name mystockanalystacr --sku Basic

# 4. Build and push image
az acr build --registry mystockanalystacr \
  --image stock-market-analyst:latest .

# 5. Create Container App environment
az containerapp env create \
  --name stock-analyst-env \
  --resource-group stock-analyst-rg \
  --location eastus

# 6. Deploy
az containerapp create \
  --name stock-market-analyst \
  --resource-group stock-analyst-rg \
  --environment stock-analyst-env \
  --image mystockanalystacr.azurecr.io/stock-market-analyst:latest \
  --target-port 8000 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 3
```

### Option 2: Azure App Service

```bash
az webapp create \
  --resource-group stock-analyst-rg \
  --plan myAppServicePlan \
  --name stock-market-analyst \
  --multicontainer-config-type compose \
  --multicontainer-config-file docker-compose.yml
```

### CI/CD with GitHub Actions

1. Create service principal:

```bash
az ad sp create-for-rbac --name "stock-analyst-sp" \
  --role contributor \
  --scopes /subscriptions/<SUBSCRIPTION_ID>/resourceGroups/stock-analyst-rg \
  --sdk-auth
```

2. Add output as GitHub Secret: `AZURE_CREDENTIALS`
3. Push to `main` → auto-deploys via `.github/workflows/`

---

## 📊 ARIMA/SARIMA Model Details

| Setting | Detail |
|---------|--------|
| Stationarity test | Augmented Dickey-Fuller (ADF) |
| Parameter selection | Grid search p=0..3, q=0..3, minimize AIC |
| SARIMA seasonal order | (1,1,1,5) — weekly seasonality |
| Data split | Configurable Train / Validation / Test via UI |
| Metrics | MAE, RMSE, MSE, R², MAPE |
| Forecast horizon | 7–90 days (UI slider) |

---

## 📈 Features

- **6 Interactive Charts** — Candlestick, Bollinger Bands, RSI, MACD, Volume, SARIMA Forecast
- **AI Investment Report** — Full markdown report with BUY / HOLD / SELL recommendation
- **News Sentiment Analysis** — Keyword scoring across latest Finnhub headlines
- **Technical Indicators** — RSI, MACD, Bollinger Bands, MA20/50/200
- **Dark Theme UI** — GitHub-style dark interface, fully responsive
- **Real-time Progress Bar** — Non-blocking background jobs with live status polling
- **CSV + JSON Export** — All data saved as downloadable files

---

## 🔌 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Frontend UI |
| `/api/analyze` | POST | Start analysis job |
| `/api/status/{job_id}` | GET | Poll job status and progress |
| `/api/report/{ticker}` | GET | Get markdown investment report |
| `/api/charts/{ticker}` | GET | Get chart file paths |
| `/api/tickers` | GET | List of popular stocks |
| `/health` | GET | Health check |

### POST `/api/analyze` — Request Body

```json
{
  "ticker": "AAPL",
  "groq_api_key": "gsk_xxxxxxxxxxxx",
  "finnhub_api_key": "xxxxxxxxxxxx",
  "period_years": 2,
  "forecast_days": 30,
  "train_split": 0.70,
  "val_split": 0.15,
  "use_sarima": true
}
```

---

## ❓ Troubleshooting

**Stock data not loading on Azure**
Caused by `yfinance` which scrapes Yahoo Finance — Yahoo blocks all cloud IPs. This project uses Finnhub API which works everywhere.

**"No data found" error**
- Verify your Finnhub API key is correct in the UI
- Confirm the ticker is valid (e.g. `AAPL`, not `Apple`)

**Report generation fails**
- Verify your Groq API key at [console.groq.com](https://console.groq.com)
- Groq free tier has rate limits — wait 1 minute and retry

**Charts not showing**
- Charts generate after data fetch — wait for 100% progress bar
- Ensure `outputs/charts/` directory is writable in your container

---

## 📝 License

MIT License — free to use and modify.

---

## 👤 Author

**Qamar Usman**
- GitHub: [@Qamar-usman-ai](https://github.com/Qamar-usman-ai)
