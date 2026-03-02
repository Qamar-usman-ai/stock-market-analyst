# agents/stock_agent.py
"""
LangGraph Stock Analysis Agent
Uses 4 tools: data scraping, news, visualization, prediction
"""
import json
import os
import traceback
from typing import TypedDict, Annotated, Sequence, Dict, Any, List
from datetime import datetime
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator

# Import tools directly (more reliable than relying on __init__.py)
try:
    from tools.stock_data_tool import scrape_stock_data
    from tools.news_tool import scrape_stock_news
    from tools.visualization_tool import generate_stock_visualizations
    from tools.prediction_tool import predict_stock_price
    TOOLS_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"❌ Error importing tools: {e}")
    TOOLS_IMPORT_SUCCESS = False
    # Create placeholder functions for when imports fail
    def scrape_stock_data(ticker, period_years=2, start_date=None, end_date=None):
        return json.dumps({"error": "Tool import failed", "status": "failed"})
    def scrape_stock_news(ticker, max_articles=15):
        return json.dumps({"error": "Tool import failed", "status": "failed"})
    def generate_stock_visualizations(ticker):
        return json.dumps({"error": "Tool import failed", "status": "failed"})
    def predict_stock_price(ticker, forecast_days=30, train_split=0.7, val_split=0.15, use_sarima=True):
        return json.dumps({"error": "Tool import failed", "status": "failed"})

# Create ALL_TOOLS list
ALL_TOOLS = [
    scrape_stock_data,
    scrape_stock_news,
    generate_stock_visualizations,
    predict_stock_price
]

# ─── Ensure directories exist ────────────────────────────────────────────────
def ensure_directories():
    """Create necessary directories if they don't exist"""
    dirs = ["outputs", "outputs/data", "outputs/charts"]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Ensured directory: {dir_path}")

ensure_directories()

# ─── State Definition ────────────────────────────────────────────────────────
class AgentState(TypedDict):
    """State schema for the stock analysis agent"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    ticker: str
    groq_api_key: str
    analysis_config: dict
    final_report: str
    tool_results: Dict[str, Any]  # Store tool results
    errors: List[str]  # Track any errors
    execution_step: int  # Track which step we're on

# ─── Agent Node ──────────────────────────────────────────────────────────────
def create_agent_node(groq_api_key: str):
    """Create the main agent node that decides which tools to call"""
    
    def agent_node(state: AgentState):
        ticker = state['ticker'].upper()
        step = state.get('execution_step', 0)
        
        print(f"\n{'='*60}")
        print(f"🚀 AGENT STEP {step + 1} for {ticker}")
        print(f"{'='*60}")
        
        # Check which tools have been used
        used_tools = set()
        for msg in state.get('messages', []):
            if isinstance(msg, ToolMessage):
                used_tools.add(msg.name)
        
        print(f"📊 Tools used so far: {used_tools}")
        
        # Determine next tool to call based on what's been used
        tool_sequence = [
            ("scrape_stock_data", f"Get historical data for {ticker}"),
            ("scrape_stock_news", f"Get latest news for {ticker}"),
            ("generate_stock_visualizations", f"Create charts for {ticker}"),
            ("predict_stock_price", f"Run price prediction for {ticker}")
        ]
        
        # Find the next tool not yet used
        next_tool = None
        next_instruction = None
        for tool_name, instruction in tool_sequence:
            if tool_name not in used_tools:
                next_tool = tool_name
                next_instruction = instruction
                break
        
        # Create system prompt based on what needs to be done
        if next_tool:
            system_prompt = f"""You are an expert stock market analyst AI agent analyzing {ticker}.

CURRENT PROGRESS: You have used tools: {', '.join(used_tools) if used_tools else 'None'}

NEXT TASK: {next_instruction}

You have access to these tools:
1. scrape_stock_data - Fetches historical price data, technical indicators
2. scrape_stock_news - Gets latest news and market sentiment analysis  
3. generate_stock_visualizations - Creates comprehensive interactive charts
4. predict_stock_price - Runs ARIMA/SARIMA model for future price predictions

IMPORTANT: 
- Call ONLY the next tool in sequence: {next_tool}
- Use the ticker symbol exactly as: {ticker}
- After getting results, you'll be called again for the next tool
"""
        else:
            # All tools are done, proceed to report generation
            system_prompt = f"""All tools have been executed for {ticker}. 
Proceed to generate the final investment report by responding with "GENERATE_REPORT"."""
        
        try:
            llm = ChatGroq(
                api_key=groq_api_key,
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=1024,
            )
            
            llm_with_tools = llm.bind_tools(ALL_TOOLS)
            
            messages = [HumanMessage(content=system_prompt)] + list(state.get("messages", []))
            
            if next_tool:
                # We still have tools to call
                response = llm_with_tools.invoke(messages)
                print(f"🤖 Agent response: {response.content[:100]}...")
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    print(f"📞 Calling tool: {response.tool_calls[0]['name']}")
            else:
                # All tools done, move to report
                response = AIMessage(content="GENERATE_REPORT")
            
            return {
                "messages": [response],
                "execution_step": step + 1
            }
            
        except Exception as e:
            error_msg = f"Agent error: {str(e)}\n{traceback.format_exc()}"
            print(f"❌ {error_msg}")
            return {
                "messages": [AIMessage(content=f"Error: {str(e)}")],
                "errors": state.get('errors', []) + [error_msg],
                "execution_step": step + 1
            }
    
    return agent_node

# ─── Router ───────────────────────────────────────────────────────────────────
def should_continue(state: AgentState) -> str:
    """Determine the next step in the graph"""
    
    # Check for errors
    if state.get('errors') and len(state.get('errors', [])) > 3:
        print("⚠️ Too many errors, forcing report generation")
        return "report"
    
    # Get the last message
    messages = state.get('messages', [])
    if not messages:
        return "agent"
    
    last_message = messages[-1]
    
    # Check if we should generate report
    if isinstance(last_message, AIMessage):
        if last_message.content == "GENERATE_REPORT":
            print("📝 Moving to report generation")
            return "report"
    
    # Check for tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"🔧 Routing to tools node")
        return "tools"
    
    # Check if we've used all tools
    used_tools = set()
    for msg in messages:
        if isinstance(msg, ToolMessage):
            used_tools.add(msg.name)
    
    print(f"✅ Tools used: {used_tools}")
    
    if len(used_tools) >= 4:
        print("📊 All tools used, moving to report")
        return "report"
    
    # Default: continue with agent
    return "agent"

# ─── Report Generator Node ───────────────────────────────────────────────────
def create_report_node(groq_api_key: str):
    """Create the final report generation node"""
    
    def report_node(state: AgentState):
        ticker = state['ticker'].upper()
        print(f"\n{'='*60}")
        print(f"📝 GENERATING FINAL REPORT for {ticker}")
        print(f"{'='*60}")
        
        # Collect all tool results
        tool_results = {}
        errors = state.get('errors', [])
        
        # Parse tool messages
        for msg in state.get('messages', []):
            if isinstance(msg, ToolMessage):
                try:
                    data = json.loads(msg.content)
                    tool_results[msg.name] = data
                    print(f"✅ Parsed result from {msg.name}")
                except json.JSONDecodeError:
                    # If not JSON, store as raw text
                    tool_results[msg.name] = {"raw": msg.content[:500]}
                    print(f"⚠️ {msg.name} returned non-JSON response")
                except Exception as e:
                    errors.append(f"Error parsing {msg.name}: {str(e)}")
        
        # Check for actual data files
        data_file = f"outputs/data/{ticker}_historical.csv"
        news_file = f"outputs/data/{ticker}_news.json"
        charts_exist = os.path.exists(f"outputs/charts/{ticker}_price_analysis.html")
        
        file_status = {
            "historical_data": os.path.exists(data_file),
            "news_data": os.path.exists(news_file),
            "charts_generated": charts_exist,
            "data_dir_exists": os.path.exists("outputs/data"),
            "charts_dir_exists": os.path.exists("outputs/charts")
        }
        
        # Try to read actual data if it exists
        data_preview = {}
        if os.path.exists(data_file):
            try:
                import pandas as pd
                df = pd.read_csv(data_file)
                data_preview = {
                    "rows": len(df),
                    "columns": list(df.columns),
                    "date_range": [str(df['Date'].iloc[0]), str(df['Date'].iloc[-1])] if 'Date' in df.columns else None,
                    "current_price": float(df['Close'].iloc[-1]) if 'Close' in df.columns else None
                }
                print(f"✅ Loaded data file: {len(df)} rows")
            except Exception as e:
                errors.append(f"Error reading data file: {str(e)}")
        
        # Read news if exists
        news_preview = {}
        if os.path.exists(news_file):
            try:
                with open(news_file, 'r') as f:
                    news_data = json.load(f)
                news_preview = {
                    "articles": len(news_data.get('news_articles', [])),
                    "sentiment": news_data.get('overall_sentiment', 'Unknown')
                }
                print(f"✅ Loaded news file")
            except Exception as e:
                errors.append(f"Error reading news file: {str(e)}")
        
        # Create report using LLM
        try:
            llm = ChatGroq(
                api_key=groq_api_key,
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=4096,
            )
            
            # Build report prompt
            report_prompt = f"""You are a senior Wall Street analyst. Based on the actual data below, generate a professional investment report for {ticker}.

## 📊 ACTUAL DATA AVAILABLE:
- Historical data exists: {file_status['historical_data']}
- News data exists: {file_status['news_data']}
- Charts generated: {file_status['charts_generated']}

## 📈 DATA PREVIEW:
{json.dumps(data_preview, indent=2) if data_preview else "No historical data available"}

## 📰 NEWS PREVIEW:
{json.dumps(news_preview, indent=2) if news_preview else "No news data available"}

## 🔧 TOOL RESULTS:
{json.dumps(tool_results, indent=2, default=str)[:2000]}

## ⚠️ ERRORS (if any):
{chr(10).join(errors) if errors else "No errors recorded"}

Generate a detailed investment report with these sections:

# 📊 STOCK ANALYSIS REPORT: {ticker}

## Executive Summary
[Brief overview with clear BUY/HOLD/SELL recommendation based on ACTUAL data]

## 📈 Price & Technical Analysis
[Analyze price data, trends, moving averages, RSI, MACD from the available data]

## 📰 Market Sentiment & News Analysis  
[Analyze news sentiment and key events from the actual news data]

## 🔮 Price Prediction & Forecast
[Discuss any predictions available, model accuracy metrics if present]

## ⚠️ Risk Assessment
[Key risks based on actual data: volatility, market position, etc.]

## 💡 Investment Recommendation
[Final verdict with specific reasoning based on the data]

## 📋 Key Metrics Summary
[Table of most important metrics from the actual data]

IMPORTANT: 
- Use ONLY actual data that exists. Do not make up numbers.
- If data is missing for a section, explain that honestly.
- Be specific with numbers when available.
"""
            
            print("🤖 Generating report with LLM...")
            report_response = llm.invoke([HumanMessage(content=report_prompt)])
            final_report = report_response.content
            
            # Save report
            report_path = f"outputs/{ticker}_investment_report.md"
            with open(report_path, "w") as f:
                f.write(final_report)
            
            print(f"✅ Report saved to {report_path}")
            
            # Also save a summary JSON
            summary = {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "data_available": file_status,
                "tools_executed": list(tool_results.keys()),
                "errors": errors,
                "report_path": report_path
            }
            
            summary_path = f"outputs/{ticker}_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            
        except Exception as e:
            error_msg = f"Report generation error: {str(e)}\n{traceback.format_exc()}"
            print(f"❌ {error_msg}")
            final_report = f"Error generating report: {str(e)}"
        
        return {
            "final_report": final_report,
            "tool_results": tool_results,
            "messages": [AIMessage(content=final_report)]
        }
    
    return report_node

# ─── Graph Builder ────────────────────────────────────────────────────────────
def build_stock_agent(groq_api_key: str):
    """Build the LangGraph agent"""
    
    # Verify Groq API key
    if not groq_api_key or groq_api_key == "your-groq-api-key-here":
        print("⚠️ WARNING: Groq API key not set or invalid")
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Create nodes
    tool_node = ToolNode(ALL_TOOLS)
    
    workflow.add_node("agent", create_agent_node(groq_api_key))
    workflow.add_node("tools", tool_node)
    workflow.add_node("report", create_report_node(groq_api_key))
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "report": "report",
            "agent": "agent"
        }
    )
    
    workflow.add_edge("tools", "agent")
    workflow.add_edge("report", END)
    
    print("✅ Stock agent graph built successfully")
    return workflow.compile()

# ─── Main Run Function ────────────────────────────────────────────────────────
async def run_stock_analysis(
    ticker: str,
    groq_api_key: str,
    period_years: int = 2,
    forecast_days: int = 30,
    train_split: float = 0.70,
    val_split: float = 0.15,
) -> Dict[str, Any]:
    """
    Run complete stock analysis pipeline.
    
    Args:
        ticker: Stock symbol to analyze
        groq_api_key: Groq API key for LLM
        period_years: Years of historical data
        forecast_days: Days to forecast
        train_split: Training data ratio
        val_split: Validation data ratio
    
    Returns:
        Final state dictionary with report and results
    """
    print(f"\n{'='*60}")
    print(f"📈 STARTING STOCK ANALYSIS FOR {ticker.upper()}")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  - Period: {period_years} years")
    print(f"  - Forecast: {forecast_days} days")
    print(f"  - Train/Val split: {train_split}/{val_split}")
    
    # Validate inputs
    if not groq_api_key or groq_api_key == "your-groq-api-key-here":
        print("❌ ERROR: Valid Groq API key is required")
        return {
            "error": "Groq API key not set",
            "final_report": "Error: Groq API key not configured properly."
        }
    
    # Build and run agent
    try:
        app = build_stock_agent(groq_api_key)
        
        initial_state = {
            "messages": [
                HumanMessage(content=f"Analyze {ticker.upper()} stock completely using all available tools.")
            ],
            "ticker": ticker.upper(),
            "groq_api_key": groq_api_key,
            "analysis_config": {
                "period_years": period_years,
                "forecast_days": forecast_days,
                "train_split": train_split,
                "val_split": val_split,
            },
            "final_report": "",
            "tool_results": {},
            "errors": [],
            "execution_step": 0
        }
        
        print("🚀 Running agent...")
        final_state = await app.ainvoke(initial_state, {"recursion_limit": 25})
        
        # Check if we got a report
        if final_state.get("final_report"):
            print(f"\n✅ Analysis complete! Report saved to outputs/{ticker.upper()}_investment_report.md")
        else:
            print("\n⚠️ Analysis completed but no report was generated")
        
        return final_state
        
    except Exception as e:
        error_msg = f"Fatal error in run_stock_analysis: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        
        # Save error to file
        with open(f"outputs/{ticker.upper()}_error.log", "w") as f:
            f.write(error_msg)
        
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "final_report": f"Error during analysis: {str(e)}"
        }

# ─── Utility Functions ───────────────────────────────────────────────────────
def check_analysis_status(ticker: str) -> Dict[str, Any]:
    """Check the status of a previous analysis"""
    ticker = ticker.upper()
    status = {
        "ticker": ticker,
        "report_exists": os.path.exists(f"outputs/{ticker}_investment_report.md"),
        "data_exists": os.path.exists(f"outputs/data/{ticker}_historical.csv"),
        "news_exists": os.path.exists(f"outputs/data/{ticker}_news.json"),
        "charts_exist": os.path.exists(f"outputs/charts/{ticker}_price_analysis.html"),
        "summary_exists": os.path.exists(f"outputs/{ticker}_summary.json")
    }
    
    if status["summary_exists"]:
        try:
            with open(f"outputs/{ticker}_summary.json", "r") as f:
                status["summary"] = json.load(f)
        except:
            pass
    
    return status

def list_analyzed_stocks() -> List[str]:
    """List all stocks that have been analyzed"""
    stocks = []
    if os.path.exists("outputs"):
        for file in os.listdir("outputs"):
            if file.endswith("_investment_report.md"):
                stocks.append(file.replace("_investment_report.md", ""))
    return stocks

# Export the main function
__all__ = ['run_stock_analysis', 'check_analysis_status', 'list_analyzed_stocks']
