from langgraph_finance.agents.analysis_agent import analysis_node
from langgraph_finance.agents.categorization_agent import categorization_node
from langgraph_finance.agents.ingestion_agent import ingestion_node
from langgraph_finance.agents.prediction_agent import prediction_node
from langgraph_finance.graph import run_pipeline
from langgraph_finance.state import new_state


def test_full_pipeline_end_to_end(sample_csv_text, patched_llm_for_pipeline, patched_default_store):
    state = run_pipeline(sample_csv_text)

    assert state["ingestion_report"]["success"] is True
    assert len(state["transactions"]) == 15

    categories = {t["category"] for t in state["categorized_transactions"]}
    assert categories == {"Groceries", "Dining", "Transportation", "Utilities", "Subscriptions"}

    assert state["analysis"]["total_expenses"] == 852.0
    assert state["analysis"]["summary_text"]

    assert state["prediction"]["next_month_label"] == "2026-04"
    assert state["prediction"]["trend"] in {"increasing", "stable", "decreasing"}
    assert state["errors"] == []


def test_pipeline_short_circuits_on_ingestion_failure(patched_llm_for_pipeline, patched_default_store):
    state = run_pipeline("not,a,valid\nheader,row,here\n")

    assert state["ingestion_report"]["success"] is False
    assert "categorized_transactions" not in state
    assert "analysis" not in state
    assert "prediction" not in state
    assert state["errors"]


def test_agents_compose_without_the_graph(sample_csv_text, patched_llm_for_pipeline, patched_default_store):
    """Each node is independently callable and composable — a sanity check that
    the LangGraph wiring isn't hiding implicit shared state between agents."""
    state = new_state(sample_csv_text)
    state = ingestion_node(state)
    state = categorization_node(state)
    state = analysis_node(state)
    state = prediction_node(state)

    assert state["prediction"]["next_month_total"] > 0
    assert len(state["categorized_transactions"]) == 15
