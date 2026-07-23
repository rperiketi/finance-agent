# 💰 Finance AI Agent

A modular, end-to-end personal finance analysis tool that turns raw bank transaction CSVs into categorized expenses, spending forecasts, and actionable insights.

---

## Features

| Phase | Feature | Tech |
|-------|---------|------|
| 1 | CSV ingestion + data cleaning | Pandas |
| 2 | Expense categorization | TF-IDF + Logistic Regression |
| 3 | Spending forecast | Facebook Prophet / Linear Regression |
| 4 | Financial insights | Statistical rules + optional LLM |
| 5 | Agent orchestration | Python classes |
| 6 | Interactive dashboard | Streamlit + Plotly |

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

Open your browser to `http://localhost:8501`

---

## Dataset

Download the **XYZ Bank ATM Transactions** dataset from Kaggle:
https://www.kaggle.com/datasets/nitsbat/data-of-atm-transaction-of-xyz-bank

Place the CSV in the `data/` folder, then upload it in the app.

The app also includes built-in **sample data** so you can try it immediately without a CSV.

---

## Optional: LLM-Enhanced Insights

Set your OpenAI API key to unlock GPT-powered financial recommendations:

```bash
export OPENAI_API_KEY="sk-..."
```

Then check **"Enhance with LLM"** in the sidebar.

---

## Project Structure

```
finance-agent/
├── app.py                          # Streamlit UI (Phase 6)
├── requirements.txt
│
├── ingestion/
│   └── loader.py                   # CSV parsing + normalisation (Phase 1)
│
├── processing/
│   └── cleaner.py                  # Merchant cleaning + feature engineering (Phase 1)
│
├── models/
│   ├── categorizer.py              # TF-IDF + Logistic Regression (Phase 2)
│   └── forecaster.py              # Prophet / LinearRegression (Phase 3)
│
├── agents/
│   ├── ingestion_agent.py          # Load + validate pipeline (Phase 5)
│   ├── analysis_agent.py           # Categorize + forecast (Phase 5)
│   └── recommendation_agent.py    # Insights + LLM (Phase 4 + 5)
│
├── utils/
│   └── insights.py                 # Statistical rules engine (Phase 4)
│
└── data/                           # Drop your CSV here
```

---

## Architecture

```
CSV Upload
    │
    ▼
IngestionAgent  →  load_transactions()  →  clean_transactions()
    │
    ▼
AnalysisAgent
    ├── ExpenseCategorizer  (TF-IDF + LogReg)  →  category labels
    └── SpendingForecaster  (Prophet)           →  30-day forecast
    │
    ▼
RecommendationAgent
    ├── Rule-based insights  (Pandas statistics)
    └── LLM reasoning        (GPT-3.5, optional)
    │
    ▼
Streamlit Dashboard
    ├── KPI Metrics
    ├── Monthly Income vs Expenses
    ├── Category Breakdown (pie + bar)
    ├── Forecast Chart (with confidence bands)
    ├── Insights Panel
    └── Raw Data + CSV Export
```
