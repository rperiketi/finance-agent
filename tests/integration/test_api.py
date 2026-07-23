import io

from langgraph_finance.api import create_app


def test_index_endpoint_serves_dashboard_page():
    client = create_app().test_client()

    resp = client.get("/")

    assert resp.status_code == 200
    assert resp.content_type.startswith("text/html")
    assert b"Finance Multi-Agent Analyzer" in resp.data


def test_health_endpoint():
    client = create_app().test_client()

    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_analyze_endpoint_with_json_body(sample_csv_text, patched_llm_for_pipeline, patched_default_store):
    client = create_app().test_client()

    resp = client.post("/analyze", json={"csv_text": sample_csv_text})

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ingestion_report"]["success"] is True
    assert len(body["categorized_transactions"]) == 15
    assert body["analysis"]["total_expenses"] == 852.0
    assert body["prediction"]["next_month_label"] == "2026-04"


def test_analyze_endpoint_with_file_upload(sample_csv_text, patched_llm_for_pipeline, patched_default_store):
    client = create_app().test_client()

    data = {"file": (io.BytesIO(sample_csv_text.encode("utf-8")), "transactions.csv")}
    resp = client.post("/analyze", data=data, content_type="multipart/form-data")

    assert resp.status_code == 200
    assert resp.get_json()["ingestion_report"]["success"] is True


def test_analyze_endpoint_with_xlsx_upload(patched_llm_for_pipeline, patched_default_store):
    import pandas as pd

    client = create_app().test_client()
    buf = io.BytesIO()
    pd.DataFrame([
        {"date": "2026-01-05", "description": "WHOLE FOODS MARKET #1234", "amount": -100.0},
        {"date": "2026-01-10", "description": "STARBUCKS STORE 4021", "amount": -50.0},
        {"date": "2026-01-15", "description": "SHELL OIL 57443210", "amount": -40.0},
        {"date": "2026-01-20", "description": "PG&E ELECTRIC BILL", "amount": -60.0},
        {"date": "2026-01-25", "description": "NETFLIX.COM", "amount": -15.0},
    ]).to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)

    data = {"file": (buf, "transactions.xlsx")}
    resp = client.post("/analyze", data=data, content_type="multipart/form-data")

    assert resp.status_code == 200
    assert resp.get_json()["ingestion_report"]["success"] is True


def test_analyze_endpoint_with_json_data(patched_llm_for_pipeline, patched_default_store):
    client = create_app().test_client()

    resp = client.post("/analyze", json={"data": [
        {"date": "2026-01-05", "description": "WHOLE FOODS MARKET #1234", "amount": -100.0},
        {"date": "2026-01-10", "description": "STARBUCKS STORE 4021", "amount": -50.0},
    ]})

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ingestion_report"]["success"] is True
    assert len(body["categorized_transactions"]) == 2


def test_analyze_endpoint_with_url(sample_csv_text, patched_llm_for_pipeline, patched_default_store, monkeypatch):
    monkeypatch.setattr(
        "langgraph_finance.agents.ingestion_agent._fetch_url",
        lambda url: (sample_csv_text.encode("utf-8"), "remote.csv", None),
    )
    client = create_app().test_client()

    resp = client.post("/analyze", json={"url": "https://example.com/transactions.csv"})

    assert resp.status_code == 200
    assert resp.get_json()["ingestion_report"]["success"] is True


def test_analyze_endpoint_requires_csv_payload():
    client = create_app().test_client()

    resp = client.post("/analyze", json={})

    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_analyze_endpoint_returns_422_on_invalid_csv(patched_llm_for_pipeline, patched_default_store):
    client = create_app().test_client()

    resp = client.post("/analyze", json={"csv_text": "not,a,valid\nheader,row,here\n"})

    assert resp.status_code == 422
    assert resp.get_json()["ingestion_report"]["success"] is False
