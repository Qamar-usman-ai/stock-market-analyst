"""
LangGraph Stock Analysis Agent - FINAL FIX
==========================================
Root cause of "no API key" error:
  os.environ changes in main.py do NOT reliably reach tool functions
  in async background tasks on Azure Container Apps.

Fix:
  Pass finnhub_api_key directly from the agent state into each tool call.
  No environment variable dependency for the API key.
"""
import json
import os
import traceback
import operator
from typing import TypedDict, Annotated, Sequence, Dict, Any, List
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

try:
    from tools.stock_data_tool import scrape_stock_data
    from tools.news_tool import scrape_stock_news
    from tools.visualization_tool import generate_stock_visualizations
    from tools.prediction_tool import predict_stock_price
    TOOLS_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Error importing tools: {e}")
    TOOLS_IMPORT_SUCCESS = False
    def scrape_stock_data(*a, **k): return json.dumps({"error": "Import failed"})
    def scrape_stock_news(*a, **k): return json.dumps({"error": "Import failed"})
    def generate_stock_visualizations(*a, **k): return json.dumps({"error": "Import failed"})
    def predict_stock_price(*a, **k): return json.dumps({"error": "Import failed"})

ALL_TOOLS = [scrape_stock_data, scrape_stock_news, generate_stock_visualizations, predict_stock_price]

os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/data", exist_ok=True)
os.makedirs("outputs/charts", exist_ok=True)


class AgentState(TypedDict):
    messages:        Annotated[Sequence[BaseMessage], operator.add]
    ticker:          str
    groq_api_key:    str
    finnhub_api_key: str          # ← KEY ADDITION: stored in state
    analysis_config: dict
    final_report:    str
    tool_results:    Dict[str, Any]
    errors:          List[str]
    execution_step:  int


def create_agent_node(groq_api_key: str, finnhub_api_key: str):
    def agent_node(state: AgentState):
        ticker = state["ticker"].upper()
        step   = state.get("execution_step", 0)

        # Which tools have already been called?
        used_tools = set()
        for msg in state.get("messages", []):
            if isinstance(msg, ToolMessage):
                used_tools.add(msg.name)

        # Ordered sequence of tool calls
        # ── CRITICAL: pass finnhub_api_key directly in each tool call ──────────
        fkey = state.get("finnhub_api_key", finnhub_api_key)   # read from state

        tool_sequence = [
            ("scrape_stock_data",
             f"Call scrape_stock_data with ticker='{ticker}' and finnhub_api_key='{fkey}' and period_years=2"),

            ("scrape_stock_news",
             f"Call scrape_stock_news with ticker='{ticker}' and finnhub_api_key='{fkey}'"),

            ("generate_stock_visualizations",
             f"Call generate_stock_visualizations with ticker='{ticker}'"),

            ("predict_stock_price",
             f"Call predict_stock_price with ticker='{ticker}'"),
        ]

        next_item = next(((name, inst) for name, inst in tool_sequence if name not in used_tools), None)

        if next_item:
            tool_name, instruction = next_item
            system_prompt = (
                f"You are a stock analyst. {instruction}. "
                f"You MUST call the tool {tool_name} right now with the exact parameters specified. "
                f"Do not write any text. Just call the tool."
            )
        else:
            system_prompt = "All 4 tools have been called. Respond with exactly: GENERATE_REPORT"

        try:
            llm = ChatGroq(
                api_key=groq_api_key,
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=1024,
            ).bind_tools(ALL_TOOLS)

            messages = [HumanMessage(content=system_prompt)] + list(state.get("messages", []))

            if next_item:
                response = llm.invoke(messages)
            else:
                response = AIMessage(content="GENERATE_REPORT")

            return {"messages": [response], "execution_step": step + 1}

        except Exception as e:
            return {
                "messages":       [AIMessage(content=f"Error: {str(e)}")],
                "errors":         state.get("errors", []) + [str(e)],
                "execution_step": step + 1,
            }

    return agent_node


def should_continue(state: AgentState) -> str:
    if len(state.get("errors", [])) > 3:
        return "report"

    messages = state.get("messages", [])
    if not messages:
        return "agent"

    last = messages[-1]

    if isinstance(last, AIMessage) and last.content == "GENERATE_REPORT":
        return "report"

    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    # Safety: if too many steps, force report
    if state.get("execution_step", 0) > 20:
        return "report"

    return "agent"


def create_report_node(groq_api_key: str):
    def report_node(state: AgentState):
        ticker      = state["ticker"].upper()
        tool_results = {}

        for msg in state.get("messages", []):
            if isinstance(msg, ToolMessage):
                try:
                    tool_results[msg.name] = json.loads(msg.content)
                except Exception:
                    tool_results[msg.name] = {"raw": str(msg.content)[:500]}

        try:
            llm = ChatGroq(
                api_key=groq_api_key,
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=4096,
            )

            # Check if stock data was actually retrieved
            stock_data = tool_results.get("scrape_stock_data", {})
            has_data   = stock_data.get("status") == "success"

            report_prompt = f"""
You are a professional stock market analyst. Generate a comprehensive investment report for {ticker}.

Here is the data from our analysis tools:
{json.dumps(tool_results, ensure_ascii=False)[:4000]}

{"The stock data was successfully retrieved. Use the price, RSI, MACD, moving averages, and other indicators in your analysis." if has_data else "Note: Stock price data could not be retrieved. Base your analysis only on available news data."}

Your report must include:
1. Executive Summary with current price and key metrics (if available)
2. Technical Analysis (RSI, MACD, Bollinger Bands, Moving Averages) - only if data available
3. News Sentiment Analysis
4. Price Prediction Summary (if available)
5. Risk Assessment
6. Clear BUY / HOLD / SELL recommendation with reasoning
7. Price targets (if data available)

Write in professional markdown format.
"""
            response      = llm.invoke([HumanMessage(content=report_prompt)])
            final_report  = response.content

            os.makedirs("outputs", exist_ok=True)
            report_path = f"outputs/{ticker}_investment_report.md"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(final_report)

            summary_path = f"outputs/{ticker}_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump({
                    "ticker":      ticker,
                    "status":      "complete",
                    "has_data":    has_data,
                    "tools_used":  list(tool_results.keys()),
                    "errors":      state.get("errors", []),
                    "generated":   datetime.now().isoformat(),
                }, f, ensure_ascii=False)

        except Exception as e:
            final_report = f"Error generating report: {str(e)}"

        return {
            "final_report": final_report,
            "messages":     [AIMessage(content=final_report)],
        }

    return report_node


def build_stock_agent(groq_api_key: str, finnhub_api_key: str):
    workflow = StateGraph(AgentState)
    workflow.add_node("agent",  create_agent_node(groq_api_key, finnhub_api_key))
    workflow.add_node("tools",  ToolNode(ALL_TOOLS))
    workflow.add_node("report", create_report_node(groq_api_key))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent", should_continue,
        {"tools": "tools", "report": "report", "agent": "agent"}
    )
    workflow.add_edge("tools",  "agent")
    workflow.add_edge("report", END)

    return workflow.compile()


async def run_stock_analysis(
    ticker: str,
    groq_api_key: str,
    finnhub_api_key: str = "",   # ← receives key directly
    **kwargs
) -> Dict[str, Any]:
    try:
        # Also set env var as backup — belt AND suspenders
        if finnhub_api_key:
            os.environ["FINNHUB_API_KEY"] = finnhub_api_key

        app = build_stock_agent(groq_api_key, finnhub_api_key)

        initial_state = {
            "messages":        [HumanMessage(content=f"Analyze {ticker.upper()} stock completely.")],
            "ticker":          ticker.upper(),
            "groq_api_key":    groq_api_key,
            "finnhub_api_key": finnhub_api_key,   # ← stored in state
            "analysis_config": kwargs,
            "final_report":    "",
            "tool_results":    {},
            "errors":          [],
            "execution_step":  0,
        }

        return await app.ainvoke(initial_state, {"recursion_limit": 30})

    except Exception as e:
        tb = traceback.format_exc()
        try:
            with open(f"outputs/{ticker.upper()}_error.log", "w", encoding="utf-8") as f:
                f.write(f"Error: {str(e)}\n\n{tb}")
        except Exception:
            pass
        return {"error": str(e), "final_report": f"Fatal error: {str(e)}"}


def check_analysis_status(ticker: str) -> Dict[str, Any]:
    ticker = ticker.upper()
    return {
        "ticker":         ticker,
        "report_exists":  os.path.exists(f"outputs/{ticker}_investment_report.md"),
        "summary_exists": os.path.exists(f"outputs/{ticker}_summary.json"),
    }


__all__ = ["run_stock_analysis", "check_analysis_status"]
