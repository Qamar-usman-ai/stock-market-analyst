# 📈 Stock Market AI Analyst

An AI-powered stock market analysis agent built with **LangGraph**, **FastAPI**, **ARIMA/SARIMA** predictions, and a modern dark-themed frontend. Uses the **Groq API** (LLaMA model) for LLM reasoning.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (HTML/JS)                        │
│         User selects stock, enters Groq API key               │
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
  YFinance       News            Viz            SARIMA
  Scraper       Scraper        Charts         Prediction
```

## 📁 Project Structure

```
stock-agent/
├── main.py                    # FastAPI app + API routes
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container image
├── docker-compose.yml         # Local multi-container setup
├── azure-deploy.yaml          # Azure deployment config
├── .gitignore
├── README.md
│
├── agents/
│   ├── __init__.py
│   └── stock_agent.py         # LangGraph StateGraph agent
│
├── tools/
│   ├── __init__.py
│   ├── stock_data_tool.py     # Tool 1: YFinance data scraper
│   ├── news_tool.py           # Tool 2: News + sentiment
│   ├── visualization_tool.py  # Tool 3: Plotly charts (6 charts)
│   └── prediction_tool.py     # Tool 4: ARIMA/SARIMA forecast
│
├── static/
│   └── index.html             # Complete frontend SPA
│
└── outputs/                   # Generated files (gitignored)
    ├── data/                  # CSV + JSON data
    ├── charts/                # HTML chart files
    └── *.md                   # Investment reports
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [Groq API Key](https://console.groq.com) (free tier available)

### Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/stock-market-analyst.git
cd stock-market-analyst

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
uvicorn main:app --reload --port 8000

# 5. Open browser
open http://localhost:8000
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

## 🛠️ The 4 Tools

| Tool | File | Description |
|------|------|-------------|
| **scrape_stock_data** | `tools/stock_data_tool.py` | Fetches OHLCV data, calculates RSI, MACD, Bollinger Bands, Moving Averages |
| **scrape_stock_news** | `tools/news_tool.py` | Gets latest news via yfinance + Yahoo RSS, keyword sentiment analysis |
| **generate_stock_visualizations** | `tools/visualization_tool.py` | 4 interactive Plotly charts (candlestick, technical, returns, volume) |
| **predict_stock_price** | `tools/prediction_tool.py` | Auto-selects ARIMA/SARIMA params via AIC grid search, train/val/test split, metrics |

---

## 📊 ARIMA/SARIMA Model Details

- **Stationarity**: Augmented Dickey-Fuller test determines differencing order `d`
- **Parameter selection**: Grid search over p=0..3, q=0..3, minimize AIC
- **SARIMA seasonal order**: (1,1,1,5) — weekly seasonality
- **Data split**: Train / Validation / Test (configurable via UI)
- **Metrics reported**: MAE, RMSE, MSE, R², MAPE

---

## ☁️ Deploy to Azure

### Option 1: Azure Container Apps (Recommended)

```bash
# 1. Login to Azure
az login

# 2. Create resource group
az group create --name stock-analyst-rg --location eastus

# 3. Create Azure Container Registry
az acr create --resource-group stock-analyst-rg \
  --name mystockanalystacr --sku Basic

# 4. Build & push image
az acr build --registry mystockanalystacr \
  --image stock-market-analyst:latest .

# 5. Create Container App environment
az containerapp env create \
  --name stock-analyst-env \
  --resource-group stock-analyst-rg \
  --location eastus

# 6. Deploy Container App
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

1. Set up Azure credentials as GitHub secret:
```bash
az ad sp create-for-rbac --name "stock-analyst-sp" \
  --role contributor \
  --scopes /subscriptions/<SUBSCRIPTION_ID>/resourceGroups/stock-analyst-rg \
  --sdk-auth
```

2. Add to GitHub Secrets: `AZURE_CREDENTIALS`
3. Push to `main` branch — auto-deploys!

---

## 📤 GitHub Upload

```bash
# Initialize git
git init
git add .
git commit -m "Initial commit: Stock Market AI Analyst"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/stock-market-analyst.git
git branch -M main
git push -u origin main
```

---

## 🔑 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Frontend UI |
| `/api/analyze` | POST | Start analysis job |
| `/api/status/{job_id}` | GET | Get job status + progress |
| `/api/jobs` | GET | List all jobs |
| `/api/report/{ticker}` | GET | Get markdown report |
| `/api/charts/{ticker}` | GET | Get chart file paths |
| `/api/tickers` | GET | Popular stock list |
| `/health` | GET | Health check |

---

## 📝 License

MIT License — free to use and modify.
