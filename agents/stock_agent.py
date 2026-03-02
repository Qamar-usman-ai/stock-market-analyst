# agents/stock_agent.py
"""
LangGraph Stock Analysis Agent
Uses 4 tools: data scraping, news, visualization, prediction
Corrected for UTF-8 and Azure Compatibility.
"""
import json
import os
import traceback
import operator
from typing import TypedDict, Annotated, Sequence, Dict, Any, List, Optional
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# ─── Tool Imports ─────────────────────────────────────────────────────────────
try:
    from tools.stock_data_tool import scrape_stock_data
    from tools.news_tool import scrape_stock_news
    from tools.visualization_tool import generate_stock_visualizations
    from tools.prediction_tool import predict_stock_price
    TOOLS_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"❌ Error importing tools: {e}")
    TOOLS_IMPORT_SUCCESS = False
    # Fallback placeholders
    def scrape_stock_data(*args, **kwargs): return json.dumps({"error": "Import failed"}, ensure_ascii=False)
    def scrape_stock_news(*args, **kwargs): return json.dumps({"error": "Import failed"}, ensure_ascii=False)
    def generate_stock_visualizations(*args, **kwargs): return json.dumps({"error": "Import failed"}, ensure_ascii=False)
    def predict_stock_price(*args, **kwargs): return json.dumps({"error": "Import failed"}, ensure_ascii=False)

ALL_TOOLS = [scrape_stock_data, scrape_stock_news, generate_stock_visualizations, predict_stock_price]

# ─── Ensure directories exist ────────────────────────────────────────────────
def ensure_directories():
    dirs = ["outputs", "outputs/data", "outputs/charts"]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

ensure_directories()

# ─── State Definition ────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    ticker: str
    groq_api_key: str
    analysis_config: dict
    final_report: str
    tool_results: Dict[str, Any]
    errors: List[str]
    execution_step: int

# ─── Agent Node ──────────────────────────────────────────────────────────────
def create_agent_node(groq_api_key: str):
    def agent_node(state: AgentState):
        ticker = state['ticker'].upper()
        step = state.get('execution_step', 0)
        
        used_tools = set()
        for msg in state.get('messages', []):
            if isinstance(msg, ToolMessage):
                used_tools.add(msg.name)
        
        tool_sequence = [
            ("scrape_stock_data", f"Get historical data for {ticker}"),
            ("scrape_stock_news", f"Get latest news for {ticker}"),
            ("generate_stock_visualizations", f"Create charts for {ticker}"),
            ("predict_stock_price", f"Run price prediction for {ticker}")
        ]
        
        next_tool = next((name for name, _ in tool_sequence if name not in used_tools), None)
        next_instruction = next((inst for name, inst in tool_sequence if name not in used_tools), None)
        
        if next_tool:
            system_prompt = f"Analyze {ticker}. Next tool: {next_tool}. Tasks remaining: {4 - len(used_tools)}."
        else:
            system_prompt = "All tools executed. Respond with 'GENERATE_REPORT'."
        
        try:
            llm = ChatGroq(
                api_key=groq_api_key,
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=1024,
            ).bind_tools(ALL_TOOLS)
            
            messages = [HumanMessage(content=system_prompt)] + list(state.get("messages", []))
            
            if next_tool:
                response = llm.invoke(messages)
            else:
                response = AIMessage(content="GENERATE_REPORT")
            
            return {"messages": [response], "execution_step": step + 1}
            
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"Error: {str(e)}")],
                "errors": state.get('errors', []) + [str(e)],
                "execution_step": step + 1
            }
    return agent_node

# ─── Router ───────────────────────────────────────────────────────────────────
def should_continue(state: AgentState) -> str:
    if len(state.get('errors', [])) > 3: return "report"
    messages = state.get('messages', [])
    if not messages: return "agent"
    
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.content == "GENERATE_REPORT":
        return "report"
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
        
    return "agent"

# ─── Report Generator Node ───────────────────────────────────────────────────
def create_report_node(groq_api_key: str):
    def report_node(state: AgentState):
        ticker = state['ticker'].upper()
        tool_results = {}
        
        for msg in state.get('messages', []):
            if isinstance(msg, ToolMessage):
                try:
                    tool_results[msg.name] = json.loads(msg.content)
                except:
                    tool_results[msg.name] = {"raw": msg.content[:500]}
        
        # Build prompt using available tool results
        try:
            llm = ChatGroq(
                api_key=groq_api_key,
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=4096,
            )
            
            report_prompt = f"Generate a detailed investment report for {ticker} using these tool results: {json.dumps(tool_results, ensure_ascii=False)[:3000]}"
            report_response = llm.invoke([HumanMessage(content=report_prompt)])
            final_report = report_response.content
            
            # CRITICAL FIX: Save with UTF-8 encoding
            report_path = f"outputs/{ticker}_investment_report.md"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(final_report)
                
            # Save summary as UTF-8
            with open(f"outputs/{ticker}_summary.json", "w", encoding="utf-8") as f:
                json.dump({"ticker": ticker, "status": "complete", "errors": state.get('errors')}, f, ensure_ascii=False)
                
        except Exception as e:
            final_report = f"Error generating report: {str(e)}"
        
        return {"final_report": final_report, "messages": [AIMessage(content=final_report)]}
    return report_node

# ─── Graph Builder ────────────────────────────────────────────────────────────
def build_stock_agent(groq_api_key: str):
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", create_agent_node(groq_api_key))
    workflow.add_node("tools", ToolNode(ALL_TOOLS))
    workflow.add_node("report", create_report_node(groq_api_key))
    
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "report": "report", "agent": "agent"})
    workflow.add_edge("tools", "agent")
    workflow.add_edge("report", END)
    
    return workflow.compile()

# ─── Main Run Function ────────────────────────────────────────────────────────
async def run_stock_analysis(ticker: str, groq_api_key: str, **kwargs) -> Dict[str, Any]:
    try:
        app = build_stock_agent(groq_api_key)
        initial_state = {
            "messages": [HumanMessage(content=f"Analyze {ticker.upper()}")],
            "ticker": ticker.upper(),
            "groq_api_key": groq_api_key,
            "analysis_config": kwargs,
            "final_report": "",
            "tool_results": {},
            "errors": [],
            "execution_step": 0
        }
        return await app.ainvoke(initial_state, {"recursion_limit": 25})
    except Exception as e:
        # UTF-8 Safe Error Logging
        err_msg = traceback.format_exc()
        with open(f"outputs/{ticker.upper()}_error.log", "w", encoding="utf-8") as f:
            f.write(err_msg)
        return {"error": str(e), "final_report": f"Fatal error: {str(e)}"}

def check_analysis_status(ticker: str) -> Dict[str, Any]:
    ticker = ticker.upper()
    return {
        "ticker": ticker,
        "report_exists": os.path.exists(f"outputs/{ticker}_investment_report.md"),
        "summary_exists": os.path.exists(f"outputs/{ticker}_summary.json")
    }

def list_analyzed_stocks() -> List[str]:
    return [f.replace("_investment_report.md", "") for f in os.listdir("outputs") if f.endswith("_investment_report.md")]

__all__ = ['run_stock_analysis', 'check_analysis_status', 'list_analyzed_stocks']
