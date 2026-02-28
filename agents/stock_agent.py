"""
LangGraph Stock Analysis Agent
Uses 4 tools: data scraping, news, visualization, prediction
"""
import json
import os
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator

from tools import ALL_TOOLS


# ─── State Definition ────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    ticker: str
    groq_api_key: str
    analysis_config: dict
    final_report: str


# ─── Agent Node ──────────────────────────────────────────────────────────────
def create_agent_node(groq_api_key: str):
    def agent_node(state: AgentState):
        llm = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=4096,
        )
        
        llm_with_tools = llm.bind_tools(ALL_TOOLS)
        
        system_prompt = f"""You are an expert stock market analyst AI agent. 
You have access to 4 powerful tools:

1. **scrape_stock_data** - Fetches historical price data, technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
2. **scrape_stock_news** - Gets latest news and market sentiment analysis  
3. **generate_stock_visualizations** - Creates comprehensive interactive charts (candlestick, volume, returns, technical indicators)
4. **predict_stock_price** - Runs ARIMA/SARIMA model for future price predictions with train/val/test split metrics

Analysis Configuration:
- Ticker: {state['ticker']}
- Training Period: {state['analysis_config'].get('period_years', 2)} years
- Forecast Days: {state['analysis_config'].get('forecast_days', 30)}
- Train Split: {state['analysis_config'].get('train_split', 0.70)}
- Val Split: {state['analysis_config'].get('val_split', 0.15)}

YOUR TASK:
Execute ALL 4 tools in this exact order:
1. First call scrape_stock_data to get historical data
2. Then call scrape_stock_news to get latest news
3. Then call generate_stock_visualizations to create charts
4. Finally call predict_stock_price with the SARIMA model

After ALL tools complete, synthesize all results into a comprehensive investment report."""
        
        messages = [HumanMessage(content=system_prompt)] + list(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    return agent_node


# ─── Router ───────────────────────────────────────────────────────────────────
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "report"


# ─── Report Generator Node ───────────────────────────────────────────────────
def create_report_node(groq_api_key: str):
    def report_node(state: AgentState):
        # Collect all tool results
        tool_results = {}
        for msg in state["messages"]:
            if isinstance(msg, ToolMessage):
                try:
                    data = json.loads(msg.content)
                    tool_results[msg.name] = data
                except Exception:
                    tool_results[msg.name] = {"raw": msg.content}
        
        stock_data = tool_results.get("scrape_stock_data", {})
        news_data = tool_results.get("scrape_stock_news", {})
        viz_data = tool_results.get("generate_stock_visualizations", {})
        pred_data = tool_results.get("predict_stock_price", {})
        
        llm = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=4096,
        )
        
        report_prompt = f"""You are a senior Wall Street analyst. Based on the complete analysis data below, 
generate a professional investment report with a clear BUY / HOLD / SELL recommendation.

## STOCK DATA ANALYSIS:
{json.dumps(stock_data, indent=2, default=str)[:2000]}

## NEWS SENTIMENT:
{json.dumps(news_data, indent=2, default=str)[:1500]}

## TECHNICAL ANALYSIS:
{json.dumps(viz_data.get('stats', {}), indent=2, default=str)}

## ARIMA/SARIMA PREDICTION RESULTS:
{json.dumps(pred_data, indent=2, default=str)[:2000]}

Generate a detailed investment report with these sections:

# 📊 STOCK ANALYSIS REPORT: {state['ticker'].upper()}

## Executive Summary
[Brief overview with clear BUY/HOLD/SELL recommendation and confidence level]

## 📈 Price & Technical Analysis
[Analyze current price, trend, moving averages, RSI, MACD, Bollinger Bands]

## 📰 Market Sentiment & News Analysis  
[Analyze news sentiment, key events affecting the stock]

## 🔮 Price Prediction & Forecast
[Discuss model accuracy (MAE, RMSE, R2, MAPE), forecast direction and target price]

## ⚠️ Risk Assessment
[Key risks: volatility, market cap, sector risks, model limitations]

## 💡 Investment Recommendation
[Final verdict: BUY / HOLD / SELL with clear reasoning, suggested entry/exit points, time horizon]

## 📋 Key Metrics Summary
[Table of most important metrics]

Be specific with numbers and give actionable advice."""
        
        report_response = llm.invoke([HumanMessage(content=report_prompt)])
        final_report = report_response.content
        
        # Save report
        os.makedirs("outputs", exist_ok=True)
        report_path = f"outputs/{state['ticker'].upper()}_investment_report.md"
        with open(report_path, "w") as f:
            f.write(final_report)
        
        return {
            "final_report": final_report,
            "messages": [AIMessage(content=final_report)]
        }
    
    return report_node


# ─── Graph Builder ────────────────────────────────────────────────────────────
def build_stock_agent(groq_api_key: str):
    workflow = StateGraph(AgentState)
    
    tool_node = ToolNode(ALL_TOOLS)
    
    workflow.add_node("agent", create_agent_node(groq_api_key))
    workflow.add_node("tools", tool_node)
    workflow.add_node("report", create_report_node(groq_api_key))
    
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "report": "report"}
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# ─── Main Run Function ────────────────────────────────────────────────────────
async def run_stock_analysis(
    ticker: str,
    groq_api_key: str,
    period_years: int = 2,
    forecast_days: int = 30,
    train_split: float = 0.70,
    val_split: float = 0.15,
):
    """
    Run complete stock analysis pipeline.
    Returns final state with report and all outputs.
    """
    app = build_stock_agent(groq_api_key)
    
    initial_state = {
        "messages": [
            HumanMessage(content=f"Analyze {ticker} stock completely using all 4 tools.")
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
    }
    
    final_state = await app.ainvoke(initial_state, {"recursion_limit": 25})
    return final_state
