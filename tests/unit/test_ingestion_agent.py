import io
import json

import pandas as pd

from langgraph_finance.agents.ingestion_agent import ingest, ingest_from_source, ingestion_node
from langgraph_finance.state import new_state


def test_valid_csv_with_no_sign_info_defaults_to_all_expenses(sample_csv_text):
    """sample_csv_text has plain positive amounts and no type column — the
    dataset is assumed to be an expenditures-only export."""
    transactions, report = ingest(sample_csv_text)

    assert report["success"] is True
    assert report["rows_received"] == 15
    assert report["rows_valid"] == 15
    assert report["rows_dropped"] == 0
    assert len(transactions) == 15
    assert transactions[0] == {
        "date": "2026-01-05",
        "description": "WHOLE FOODS MARKET #1234",
        "amount": -100.0,
    }
    assert any("assuming this is an expenditures-only export" in w for w in report["warnings"])


def test_signed_amounts_are_trusted_as_is():
    csv_text = (
        "date,description,amount\n"
        "2026-01-05,PAYROLL DEPOSIT,2000.00\n"
        "2026-01-06,WHOLE FOODS MARKET,-85.50\n"
    )
    transactions, report = ingest(csv_text)

    assert report["success"] is True
    assert transactions[0]["amount"] == 2000.0
    assert transactions[1]["amount"] == -85.50
    assert not any("expenditures-only" in w for w in report["warnings"])


def test_unsigned_amount_with_type_column_infers_sign():
    csv_text = (
        "date,description,amount,movement\n"
        "2026-01-05,SALARY,2000.00,credit\n"
        "2026-01-06,WOOLWORTHS,85.50,debit\n"
    )
    transactions, report = ingest(csv_text)

    assert report["success"] is True
    assert transactions[0]["amount"] == 2000.0
    assert transactions[1]["amount"] == -85.50
    assert report["column_mapping_used"]["type"] == "movement"


def test_separate_debit_credit_columns_are_combined():
    csv_text = (
        "date,description,debit,credit\n"
        "2026-01-05,SALARY,,2000.00\n"
        "2026-01-06,WOOLWORTHS,85.50,\n"
    )
    transactions, report = ingest(csv_text)

    assert report["success"] is True
    assert transactions[0]["amount"] == 2000.0
    assert transactions[1]["amount"] == -85.50
    assert report["column_mapping_used"]["debit"] == "debit"
    assert report["column_mapping_used"]["credit"] == "credit"


def test_recognizes_common_bank_export_aliases():
    csv_text = (
        "transaction_date,narrative,txn_amount\n"
        "2026-01-05,COFFEE SHOP,-10.00\n"
        "2026-01-06,GROCERY STORE,-55.00\n"
    )
    transactions, report = ingest(csv_text)

    assert report["success"] is True
    assert report["column_mapping_used"]["date"] == "transaction_date"
    assert report["column_mapping_used"]["description"] == "narrative"
    assert report["column_mapping_used"]["amount"] == "txn_amount"
    assert transactions[0]["description"] == "COFFEE SHOP"


def test_recognizes_substring_column_names_like_real_bank_exports():
    # Mirrors a real ANZ-style export: no column is literally named
    # "description", but "txn_description" clearly is one, and "movement"
    # carries debit/credit direction for an unsigned amount column.
    csv_text = (
        "status,account,currency,txn_description,merchant_id,date,amount,"
        "transaction_id,customer_id,movement\n"
        "posted,123,AUD,WOOLWORTHS METRO,M1,2026-01-05,42.50,T1,C1,debit\n"
        "posted,123,AUD,BP FUEL,M2,2026-01-06,60.00,T2,C1,debit\n"
    )
    transactions, report = ingest(csv_text)

    assert report["success"] is True
    assert report["column_mapping_used"]["description"] == "txn_description"
    assert [t["description"] for t in transactions] == ["WOOLWORTHS METRO", "BP FUEL"]
    assert all(t["amount"] < 0 for t in transactions)


def test_missing_required_column_errors():
    csv_text = "foo,bar\n1,2\n"
    transactions, report = ingest(csv_text)

    assert report["success"] is False
    assert transactions == []
    assert any("Missing required column" in e for e in report["errors"])


def test_empty_csv_errors():
    transactions, report = ingest("")

    assert report["success"] is False
    assert "No CSV content provided." in report["errors"]


def test_invalid_rows_are_dropped_with_warning():
    csv_text = (
        "date,description,amount\n"
        "2026-01-05,GOOD ROW,-10.00\n"
        "not-a-date,BAD DATE,-10.00\n"
        "2026-01-06,BAD AMOUNT,not-a-number\n"
        "2026-01-07,,-10.00\n"
    )
    transactions, report = ingest(csv_text)

    assert report["success"] is True
    assert report["rows_received"] == 4
    assert report["rows_valid"] == 1
    assert report["rows_dropped"] == 3
    assert any("Dropped 3 of 4" in w for w in report["warnings"])


def test_ingestion_node_wraps_state(sample_csv_text):
    state = new_state(sample_csv_text)
    new = ingestion_node(state)

    assert new["ingestion_report"]["success"] is True
    assert len(new["transactions"]) == 15
    assert new["errors"] == []


def _xlsx_bytes(rows: list[dict]) -> bytes:
    buf = io.BytesIO()
    pd.DataFrame(rows).to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def test_ingests_xlsx_file_bytes():
    data = _xlsx_bytes([
        {"date": "2026-01-05", "description": "WHOLE FOODS MARKET", "amount": -100.0},
        {"date": "2026-01-06", "description": "ACME PAYROLL", "amount": 3000.0},
    ])

    transactions, report = ingest_from_source(file_bytes=data, filename="statement.xlsx")

    assert report["success"] is True
    assert len(transactions) == 2
    assert transactions[0]["amount"] == -100.0
    assert transactions[1]["amount"] == 3000.0


def test_ingests_xlsx_by_content_sniffing_without_extension():
    data = _xlsx_bytes([
        {"date": "2026-01-05", "description": "WHOLE FOODS MARKET", "amount": -100.0},
        {"date": "2026-01-06", "description": "STARBUCKS", "amount": -12.0},
    ])

    transactions, report = ingest_from_source(file_bytes=data, filename=None)

    assert report["success"] is True
    assert len(transactions) == 2


def test_ingests_json_list_of_records():
    records = [
        {"date": "2026-01-05", "description": "WHOLE FOODS MARKET", "amount": -100.0},
        {"date": "2026-01-06", "description": "ACME PAYROLL", "amount": 3000.0},
    ]

    transactions, report = ingest_from_source(json_records=records)

    assert report["success"] is True
    assert len(transactions) == 2
    assert transactions[0]["description"] == "WHOLE FOODS MARKET"


def test_ingests_json_wrapped_under_common_key():
    payload = {
        "transactions": [
            {"date": "2026-01-05", "description": "WHOLE FOODS MARKET", "amount": -100.0},
            {"date": "2026-01-06", "description": "STARBUCKS", "amount": -12.0},
        ]
    }

    transactions, report = ingest_from_source(json_records=payload)

    assert report["success"] is True
    assert len(transactions) == 2


def test_ingests_json_bytes_sniffed_from_file_upload():
    payload = json.dumps([
        {"date": "2026-01-05", "description": "WHOLE FOODS MARKET", "amount": -100.0},
        {"date": "2026-01-06", "description": "STARBUCKS", "amount": -12.0},
    ]).encode("utf-8")

    transactions, report = ingest_from_source(file_bytes=payload, filename="export.json")

    assert report["success"] is True
    assert len(transactions) == 2


def test_ingests_from_url(monkeypatch):
    csv_bytes = b"date,description,amount\n2026-01-05,WHOLE FOODS MARKET,-100.00\n2026-01-06,STARBUCKS,-12.00\n"
    monkeypatch.setattr(
        "langgraph_finance.agents.ingestion_agent._fetch_url",
        lambda url: (csv_bytes, "remote.csv", None),
    )

    transactions, report = ingest_from_source(source_url="https://example.com/transactions.csv")

    assert report["success"] is True
    assert len(transactions) == 2


def test_url_fetch_failure_is_reported(monkeypatch):
    monkeypatch.setattr(
        "langgraph_finance.agents.ingestion_agent._fetch_url",
        lambda url: (None, None, "Failed to fetch URL: timed out"),
    )

    transactions, report = ingest_from_source(source_url="https://example.com/nope.csv")

    assert report["success"] is False
    assert transactions == []
    assert any("Failed to fetch URL" in e for e in report["errors"])


def test_url_must_be_http_or_https():
    from langgraph_finance.agents.ingestion_agent import _fetch_url

    data, filename, err = _fetch_url("ftp://example.com/file.csv")

    assert data is None
    assert err is not None
    assert "http" in err.lower()


def test_no_source_provided_returns_clear_error():
    transactions, report = ingest_from_source()

    assert report["success"] is False
    assert transactions == []
    assert any("No data provided" in e for e in report["errors"])


def test_malformed_json_data_reports_error():
    transactions, report = ingest_from_source(json_records="not a list or dict")

    assert report["success"] is False
    assert transactions == []
    assert any("Failed to parse JSON data" in e for e in report["errors"])
