"""Flask REST API wrapping the LangGraph pipeline.

POST /analyze accepts a file upload (CSV/Excel/JSON), JSON
{"csv_text": ...}, {"url": ...}, {"data": [...]}, or a raw text/csv body,
and returns ingestion/categorization/analysis/prediction results as JSON.
"""

from __future__ import annotations

import os

from flask import Flask, Request, jsonify, render_template, request

from langgraph_finance.graph import run_pipeline


def _extract_source(req: Request) -> dict:
    """Returns kwargs for `run_pipeline`, or `{}` if no usable input found."""
    if "file" in req.files and req.files["file"].filename:
        f = req.files["file"]
        return {"file_bytes": f.read(), "filename": f.filename}

    if req.is_json:
        body = req.get_json(silent=True) or {}
        if body.get("url"):
            return {"source_url": body["url"]}
        if isinstance(body.get("data"), (list, dict)):
            return {"json_records": body["data"]}
        if body.get("csv_text"):
            return {"csv_text": body["csv_text"]}
        return {}

    if req.data:
        return {"csv_text": req.data.decode("utf-8", errors="replace")}

    return {}


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.post("/analyze")
    def analyze():
        source = _extract_source(request)
        if not source:
            return jsonify(
                error=(
                    "Provide a file upload ('file': .csv/.xlsx/.xls/.json), JSON "
                    "{'csv_text': ...}, {'url': ...}, {'data': [...]}, or a raw text/csv body."
                )
            ), 400

        state = run_pipeline(**source)

        response = {
            "ingestion_report": state.get("ingestion_report", {}),
            "categorization_notes": state.get("categorization_notes", []),
            "categorized_transactions": state.get("categorized_transactions", []),
            "analysis": state.get("analysis", {}),
            "prediction": state.get("prediction", {}),
            "errors": state.get("errors", []),
        }
        status_code = 200 if response["ingestion_report"].get("success") else 422
        return jsonify(response), status_code

    return app


app = create_app()

if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_RUN_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug)
