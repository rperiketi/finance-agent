"""LangGraph orchestration wiring the four agents together with shared state.

ingest -> categorize -> analyze -> predict -> END
(ingest routes straight to END if validation fails, so downstream agents
never run against empty/invalid data).
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from langgraph_finance.agents.analysis_agent import analysis_node
from langgraph_finance.agents.categorization_agent import categorization_node
from langgraph_finance.agents.ingestion_agent import ingestion_node
from langgraph_finance.agents.prediction_agent import prediction_node
from langgraph_finance.state import FinanceState, new_state


def _route_after_ingestion(state: FinanceState) -> str:
    report = state.get("ingestion_report") or {}
    return "categorize" if report.get("success") else "end"


def build_graph():
    graph = StateGraph(FinanceState)
    graph.add_node("ingest", ingestion_node)
    graph.add_node("categorize", categorization_node)
    graph.add_node("analyze", analysis_node)
    graph.add_node("predict", prediction_node)

    graph.add_edge(START, "ingest")
    graph.add_conditional_edges(
        "ingest", _route_after_ingestion, {"categorize": "categorize", "end": END}
    )
    graph.add_edge("categorize", "analyze")
    graph.add_edge("analyze", "predict")
    graph.add_edge("predict", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_pipeline(
    csv_text: str | None = None,
    *,
    file_bytes: bytes | None = None,
    filename: str | None = None,
    json_records: list | None = None,
    source_url: str | None = None,
) -> FinanceState:
    """Runs the full ingestion -> categorization -> analysis -> prediction
    pipeline. Accepts CSV text, raw file bytes (CSV/Excel/JSON), inline JSON
    records, or a URL to fetch any of the above from."""
    graph = get_graph()
    return graph.invoke(
        new_state(
            csv_text,
            file_bytes=file_bytes,
            filename=filename,
            json_records=json_records,
            source_url=source_url,
        )
    )
