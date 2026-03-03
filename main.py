"""
Main FastAPI Application
Stock Market Analysis AI Agent
"""
import asyncio
import json
import os
import uuid
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Stock Market AI Analyst",
    description="AI-powered stock market analysis using LangGraph + ARIMA/SARIMA",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("outputs/data", exist_ok=True)
os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("static", exist_ok=True)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/static", StaticFiles(directory="static"), name="static")

jobs: dict = {}


# ─── Request Models ───────────────────────────────────────────────────────────
class AnalysisRequest(BaseModel):
    ticker: str
    groq_api_key: str
    finnhub_api_key: str          # ← NEW: comes from frontend input
    period_years: int = 2
    forecast_days: int = 30
    train_split: float = 0.70
    val_split: float = 0.15
    use_sarima: bool = True


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    message: str
    result: Optional[dict] = None


# ─── Background Analysis Task ─────────────────────────────────────────────────
async def run_analysis_job(job_id: str, request: AnalysisRequest):
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["progress"] = 5
        jobs[job_id]["message"] = "Initializing AI agent..."

        from agents.stock_agent import run_stock_analysis

        groq_key    = request.groq_api_key.strip()
        finnhub_key = request.finnhub_api_key.strip()   # ← NEW

        # Inject both keys into the environment so all tools can read them
        os.environ["GROQ_API_KEY"]              = groq_key
        os.environ["FINNHUB_API_KEY"]           = finnhub_key   # ← NEW
        os.environ["ANALYSIS_PERIOD_YEARS"]     = str(request.period_years)
        os.environ["ANALYSIS_FORECAST_DAYS"]    = str(request.forecast_days)
        os.environ["ANALYSIS_TRAIN_SPLIT"]      = str(request.train_split)
        os.environ["ANALYSIS_VAL_SPLIT"]        = str(request.val_split)

        jobs[job_id]["progress"] = 10
        jobs[job_id]["message"] = f"Fetching historical data for {request.ticker}..."

        jobs[job_id]["progress"] = 20
        jobs[job_id]["message"] = "Running LangGraph agent pipeline..."

        final_state = await run_stock_analysis(
            ticker=request.ticker,
            groq_api_key=groq_key,
            period_years=request.period_years,
            forecast_days=request.forecast_days,
            train_split=request.train_split,
            val_split=request.val_split,
        )

        jobs[job_id]["progress"] = 90
        jobs[job_id]["message"] = "Compiling final report..."

        ticker = request.ticker.upper()

        charts = {
            "price_analysis":        f"/outputs/charts/{ticker}_price_analysis.html",
            "technical_indicators":  f"/outputs/charts/{ticker}_technical_indicators.html",
            "returns_analysis":      f"/outputs/charts/{ticker}_returns_analysis.html",
            "volume_analysis":       f"/outputs/charts/{ticker}_volume_analysis.html",
            "prediction":            f"/outputs/charts/{ticker}_prediction.html",
            "model_accuracy":        f"/outputs/charts/{ticker}_model_accuracy.html",
        }
        existing_charts = {k: v for k, v in charts.items() if os.path.exists(v.lstrip("/"))}

        jobs[job_id]["status"]   = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"]  = "Analysis complete!"
        jobs[job_id]["result"]   = {
            "ticker":       ticker,
            "final_report": final_state.get("final_report", ""),
            "report_file":  f"/outputs/{ticker}_investment_report.md",
            "charts":       existing_charts,
            "completed_at": datetime.now().isoformat(),
        }

    except Exception as e:
        jobs[job_id]["status"]   = "failed"
        jobs[job_id]["progress"] = 0
        jobs[job_id]["message"]  = f"Error: {str(e)}"
        jobs[job_id]["result"]   = {"error": str(e)}


# ─── API Routes ───────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = "static/index.html"
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Frontend not found. Place index.html in /static/</h1>")


@app.post("/api/analyze", response_model=dict)
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    if not request.ticker:
        raise HTTPException(status_code=400, detail="Ticker symbol is required")
    if not request.groq_api_key:
        raise HTTPException(status_code=400, detail="Groq API key is required")
    if not request.finnhub_api_key:                                          # ← NEW
        raise HTTPException(status_code=400, detail="Finnhub API key is required")  # ← NEW

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id":     job_id,
        "status":     "queued",
        "progress":   0,
        "message":    "Job queued...",
        "ticker":     request.ticker.upper(),
        "created_at": datetime.now().isoformat(),
        "result":     None,
    }

    background_tasks.add_task(run_analysis_job, job_id, request)
    return {"job_id": job_id, "status": "queued", "message": "Analysis started!"}


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/api/report/{ticker}")
async def get_report(ticker: str):
    report_path = f"outputs/{ticker.upper()}_investment_report.md"
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found")
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return {"ticker": ticker.upper(), "report": f.read()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encoding error: {str(e)}")


@app.get("/api/charts/{ticker}")
async def get_charts(ticker: str):
    chart_dir    = "outputs/charts"
    ticker_upper = ticker.upper()
    charts = {}
    for ct in ["price_analysis", "technical_indicators", "returns_analysis",
               "volume_analysis", "prediction", "model_accuracy"]:
        path = f"{chart_dir}/{ticker_upper}_{ct}.html"
        if os.path.exists(path):
            charts[ct] = f"/outputs/charts/{ticker_upper}_{ct}.html"
    return {"ticker": ticker_upper, "charts": charts}


@app.get("/api/tickers")
async def get_popular_tickers():
    return {
        "tickers": [
            {"symbol": "AAPL",  "name": "Apple Inc."},
            {"symbol": "MSFT",  "name": "Microsoft Corp."},
            {"symbol": "NVDA",  "name": "NVIDIA Corp."},
            {"symbol": "TSLA",  "name": "Tesla Inc."},
            {"symbol": "GOOGL", "name": "Alphabet Inc."},
        ]
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
