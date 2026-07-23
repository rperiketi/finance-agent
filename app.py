"""
Finance AI Agent - Streamlit Application
Phase 6: Full UI combining all agents and models
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Load .env file if present (optional dependency)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

import io
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from agents.ingestion_agent import IngestionAgent
from agents.analysis_agent import AnalysisAgent
from agents.recommendation_agent import RecommendationAgent
from models.categorizer import CATEGORIES

CLIP_MODES = {
    "Quantile (clip 1st–99th percentile)": "quantile",
    "None (keep raw amounts)": "none",
    "IQR (Tukey fences on spends/income)": "iqr",
}

# -- Page config ---------------------------------------------------------------
st.set_page_config(
    page_title="Finance AI Agent",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Custom CSS ----------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { color: #666; font-size: 1rem; margin-bottom: 1.5rem; }
    .insight-card {
        background: #f8f9ff; border-left: 4px solid #667eea;
        padding: 0.7rem 1rem; border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem; font-size: 0.93rem;
    }
    .warning-card {
        background: #fff8f0; border-left: 4px solid #f6a623;
        padding: 0.7rem 1rem; border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
    }
    .good-card {
        background: #f0fff4; border-left: 4px solid #48bb78;
        padding: 0.7rem 1rem; border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# -- Session state -------------------------------------------------------------
for key in ("clean_df", "analysis", "recommendations", "ingestion_report", "manual_train_pairs"):
    if key not in st.session_state:
        st.session_state[key] = None


# -- Sidebar -------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Configuration")
    st.divider()

    uploaded = st.file_uploader(
        "Upload transactions",
        type=["csv", "xlsx"],
        help="CSV or Excel export. Column names can vary; overrides below.",
    )
    peek_cols: list[str] = []
    if uploaded is not None:
        fb = uploaded.getvalue()
        try:
            if uploaded.name.lower().endswith(".xlsx"):
                _peek = pd.read_excel(io.BytesIO(fb), engine="openpyxl", nrows=0)
            else:
                _peek = pd.read_csv(io.BytesIO(fb), nrows=0)
            peek_cols = [str(c).strip() for c in _peek.columns.tolist()]
        except Exception:
            peek_cols = []

    use_sample = st.checkbox(
        "Use sample data (demo)",
        value=uploaded is None,
        help="If off, upload a file to analyse.",
    )

    st.checkbox(
        "Prefer day-first dates (DD/MM)",
        key="sb_dayfirst",
        help="If unchecked, ingestion tries US and EU order and picks best parse rate.",
    )
    st.checkbox(
        "European decimals (comma as decimal separator)",
        key="sb_euro_nums",
        help='For numbers like 1.234,56.',
    )

    clip_label = st.selectbox(
        "Amount outlier handling",
        list(CLIP_MODES.keys()),
        index=0,
    )
    amount_clip_mode = CLIP_MODES[clip_label]

    user_mapping = {}
    if peek_cols:
        opts = ["Auto"] + peek_cols
        with st.expander("Column mapping (optional)", expanded=False):
            st.caption("Choose **Auto** to use inferred columns.")
            mode = st.radio(
                "Amount layout",
                ("Single signed amount column", "Separate debit / credit columns"),
                index=0,
                horizontal=False,
                key="sb_amt_layout",
            )
            user_mapping["amount_mode"] = "split" if "Separate" in mode else "single"

            user_mapping["date"] = st.selectbox("Date column", opts, key="sb_map_date")
            user_mapping["description"] = st.selectbox("Description column", opts, key="sb_map_desc")
            user_mapping["category"] = st.selectbox(
                "Pre-labelled category column (optional)",
                ["Auto"] + peek_cols,
                key="sb_map_cat",
                help="When set, overrides ML for labelled rows.",
            )
            user_mapping["type"] = st.selectbox("Debit/credit column (optional)", opts, key="sb_map_typ")

            if user_mapping["amount_mode"] == "single":
                user_mapping["amount"] = st.selectbox("Amount column", opts, key="sb_map_amt")
            else:
                user_mapping["debit_column"] = st.selectbox("Debit/outflow column", opts, key="sb_map_dr")
                user_mapping["credit_column"] = st.selectbox("Credit/inflow column", opts, key="sb_map_cr")

    st.divider()
    forecast_days = st.slider("Forecast horizon (days)", 7, 90, 30, step=7)
    use_llm = st.checkbox(
        "Enhance with LLM (requires OPENAI_API_KEY)",
        value=False,
        help="Set OPENAI_API_KEY environment variable to enable GPT-powered insights.",
    )

    st.divider()
    run_btn   = st.button("Run Analysis", type="primary", use_container_width=True)
    reset_btn = st.button("Reset", use_container_width=True)

    if reset_btn:
        for key in ("clean_df", "analysis", "recommendations", "ingestion_report", "manual_train_pairs"):
            st.session_state[key] = None
        st.rerun()

    st.divider()
    st.markdown("**Dataset**")
    st.markdown(
        "[XYZ Bank ATM Transactions](https://www.kaggle.com/datasets/nitsbat/data-of-atm-transaction-of-xyz-bank)"
    )
    st.caption("Download CSV from Kaggle and upload above.")


# -- Header --------------------------------------------------------------------
st.markdown('<div class="main-header">Finance AI Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload bank transactions to get categorization, forecasts, and actionable insights.</div>',
    unsafe_allow_html=True,
)


# -- Run pipeline --------------------------------------------------------------
if run_btn:
    with st.spinner("Running analysis pipeline..."):
        if uploaded is not None:
            source = io.BytesIO(uploaded.getvalue())
            upload_name = uploaded.name
        elif use_sample:
            source = None
            upload_name = None
        else:
            st.warning("Upload a CSV/XLSX or enable sample data.")
            st.stop()

        dayfirst = True if st.session_state.get("sb_dayfirst") else None
        euro = bool(st.session_state.get("sb_euro_nums"))

        cm = None
        if peek_cols and user_mapping:
            cm = dict(user_mapping)

        ing_agent = IngestionAgent()
        clean_df, report = ing_agent.run(
            source,
            upload_name=upload_name,
            column_mapping=cm,
            dayfirst=dayfirst,
            european_decimal=euro,
            amount_clip_mode=amount_clip_mode,
        )
        st.session_state["ingestion_report"] = report

        if not report.success or clean_df.empty:
            err_msg = "; ".join(report.errors) if report.errors else "Empty result after cleaning."
            st.error(f"Ingestion failed: {err_msg}")
            st.stop()

        st.session_state["clean_df"] = clean_df

        # Agent 2: Analysis
        analysis_agent = AnalysisAgent()
        analysis = analysis_agent.run(clean_df, forecast_days=forecast_days)
        st.session_state["analysis"] = analysis

        # Agent 3: Recommendations
        rec_agent = RecommendationAgent(use_llm=use_llm)
        forecast_total = analysis.next_month_estimate.get("forecast_total_30d", 0)
        recs = rec_agent.run(
            analysis.categorized_df,
            forecast_total=forecast_total,
            category_summary=analysis.category_summary,
        )
        st.session_state["recommendations"] = recs


# -- Dashboard -----------------------------------------------------------------
if st.session_state["clean_df"] is not None and not st.session_state["clean_df"].empty:
    df       = st.session_state["clean_df"]
    analysis = st.session_state["analysis"]
    recs     = st.session_state["recommendations"]
    report   = st.session_state["ingestion_report"]

    # Ingestion warnings
    if report and report.warnings:
        for w in report.warnings:
            st.info(w)

    # -- KPI row ---------------------------------------------------------------
    cat_df   = analysis.categorized_df if (analysis and not analysis.categorized_df.empty) else df
    income   = cat_df[cat_df["amount"] > 0]["amount"].sum()
    expenses = cat_df[cat_df["amount"] < 0]["amount"].abs().sum()
    net      = income - expenses
    est      = analysis.next_month_estimate if analysis else {}

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Income",   f"${income:,.0f}")
    c2.metric("Total Expenses", f"${expenses:,.0f}")
    c3.metric("Net Savings",    f"${net:,.0f}",
              delta=f"{net/income*100:.1f}% savings rate" if income else None)
    c4.metric("Transactions",   f"{len(cat_df):,}")
    if est:
        c5.metric("Forecast (30d)", f"${est.get('forecast_total_30d', 0):,.0f}",
                  delta=f"{est.get('change_pct', 0):+.1f}% vs last 30d",
                  delta_color="inverse")

    st.divider()

    # -- Tabs ------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Categories", "Forecast", "Insights", "Raw Data",
    ])

    # ---- TAB 1: Overview -----------------------------------------------------
    with tab1:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Monthly Income vs Expenses")
            monthly = (
                cat_df.copy()
                .assign(
                    month_label=lambda x: x["date"].dt.strftime("%b %Y"),
                    month_sort=lambda x: x["date"].dt.to_period("M"),
                )
                .groupby(["month_sort", "month_label"])
                .apply(lambda g: pd.Series({
                    "Income":   g.loc[g["amount"] > 0, "amount"].sum(),
                    "Expenses": g.loc[g["amount"] < 0, "amount"].abs().sum(),
                }), include_groups=False)
                .reset_index()
                .sort_values("month_sort")
            )
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(name="Income",   x=monthly["month_label"],
                                     y=monthly["Income"],   marker_color="#48bb78"))
            fig_bar.add_trace(go.Bar(name="Expenses", x=monthly["month_label"],
                                     y=monthly["Expenses"], marker_color="#fc8181"))
            fig_bar.update_layout(barmode="group", height=350,
                                  xaxis_tickangle=-30, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_b:
            st.subheader("Net Savings Trend")
            monthly["Net"] = monthly["Income"] - monthly["Expenses"]
            fig_line = px.line(monthly, x="month_label", y="Net",
                               markers=True, color_discrete_sequence=["#667eea"])
            fig_line.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_line.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0),
                                   xaxis_title="", yaxis_title="Net ($)")
            st.plotly_chart(fig_line, use_container_width=True)

        # Spending heatmap
        if "day_of_week" in cat_df.columns and "month" in cat_df.columns:
            st.subheader("Spending Heatmap (Day of Week vs Month)")
            exp_only = cat_df[cat_df["amount"] < 0].copy()
            exp_only["amount"]   = exp_only["amount"].abs()
            exp_only["dow_name"] = exp_only["date"].dt.day_name()
            heatmap_data = (
                exp_only.groupby(["dow_name", "month"])["amount"]
                .sum().reset_index()
                .pivot(index="dow_name", columns="month", values="amount")
                .fillna(0)
            )
            dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            heatmap_data = heatmap_data.reindex(
                [d for d in dow_order if d in heatmap_data.index]
            )
            fig_heat = px.imshow(
                heatmap_data, color_continuous_scale="Blues",
                labels={"x": "Month", "y": "Day", "color": "Spend ($)"},
                aspect="auto",
            )
            fig_heat.update_layout(height=280, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_heat, use_container_width=True)

    # ---- TAB 2: Categories ---------------------------------------------------
    with tab2:
        if analysis and not analysis.category_summary.empty:
            col_a, col_b = st.columns([1.2, 1])

            with col_a:
                st.subheader("Spending by Category")
                fig_pie = px.pie(
                    analysis.category_summary,
                    names="category", values="total",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    hole=0.4,
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                fig_pie.update_layout(height=400, showlegend=False,
                                      margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_b:
                st.subheader("Category Breakdown")
                st.dataframe(
                    analysis.category_summary.style.format({
                        "total":       "${:,.0f}",
                        "avg_per_txn": "${:,.0f}",
                        "pct_of_total": "{:.1f}%",
                    }),
                    use_container_width=True,
                    height=380,
                )

            st.subheader("Category Spending Over Time")
            cat_time = (
                analysis.categorized_df[analysis.categorized_df["amount"] < 0]
                .copy()
                .assign(amount=lambda x: x["amount"].abs(),
                        period=lambda x: x["date"].dt.strftime("%b %Y"),
                        month_sort=lambda x: x["date"].dt.to_period("M"))
                .groupby(["month_sort", "period", "category"])["amount"]
                .sum().reset_index()
                .sort_values("month_sort")
            )
            if not cat_time.empty:
                fig_area = px.bar(
                    cat_time, x="period", y="amount", color="category",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    barmode="stack",
                )
                fig_area.update_layout(height=380, xaxis_tickangle=-30,
                                       margin=dict(l=0, r=0, t=20, b=0),
                                       xaxis_title="", yaxis_title="Spend ($)")
                st.plotly_chart(fig_area, use_container_width=True)

            if analysis and not analysis.category_forecast.empty:
                st.subheader("Next Month Category Forecast")
                fig_cf = px.bar(
                    analysis.category_forecast,
                    x="forecast", y="category", orientation="h",
                    color="forecast", color_continuous_scale="Blues",
                    text="forecast",
                )
                fig_cf.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
                fig_cf.update_layout(height=350, showlegend=False,
                                     margin=dict(l=0, r=0, t=20, b=0),
                                     xaxis_title="Forecast ($)", yaxis_title="",
                                     coloraxis_showscale=False)
                st.plotly_chart(fig_cf, use_container_width=True)
        else:
            st.info("Run the analysis to see category breakdown.")

        undf = getattr(analysis, "uncertain_df", None) if analysis else None
        if undf is not None and not undf.empty:
            st.subheader("Uncertain classifications (weak ML scores)")
            st.caption(
                "Rule-based matches never appear here — only low-confidence classifier rows."
            )
            st.dataframe(
                undf.style.format({"ml_confidence": "{:.3f}"}),
                use_container_width=True,
                hide_index=True,
            )
            with st.expander("Retrain classifier on these descriptions"):
                u_show = undf.head(15).reset_index(drop=True)
                for ri in range(len(u_show)):
                    row = u_show.iloc[ri]
                    desc = row["description"]
                    short = desc[:120] + ("…" if len(desc) > 120 else "")
                    st.markdown(f"**#{ri + 1}** `{short}`")
                    guess = row.get("ml_guess", "Other")
                    idx_default = (
                        CATEGORIES.index(guess) if guess in CATEGORIES else len(CATEGORIES) - 1
                    )
                    st.selectbox(
                        "Manual category",
                        CATEGORIES,
                        index=idx_default,
                        key=f"mtrain_pick_{ri}",
                        label_visibility="collapsed",
                    )
                if st.button("Retrain classifier and save model"):
                    pairs = []
                    for ri in range(len(u_show)):
                        txt = str(u_show.iloc[ri]["description"])
                        lbl = str(st.session_state.get(f"mtrain_pick_{ri}", "Other"))
                        pairs.append((txt, lbl))
                    AnalysisAgent().retrain_categorizer(pairs)
                    st.success(
                        "Model updated — run **Run Analysis** again to refresh categories."
                    )

    # ---- TAB 3: Forecast -----------------------------------------------------
    with tab3:
        if analysis and not analysis.forecast_df.empty:
            est = analysis.next_month_estimate or {}
            st.subheader(f"Spending Forecast - Next {forecast_days} Days")
            st.caption(f"Model: {est.get('model', 'N/A')}")

            diagnostics = {
                str(k)[5:]: v for k, v in est.items() if str(k).startswith("diag_")
            }
            if diagnostics:
                with st.expander("Forecast diagnostics (holdout quality checks)"):
                    st.json(diagnostics)

            m1, m2, m3 = st.columns(3)
            m1.metric("Forecast Total (30d)", f"${est.get('forecast_total_30d', 0):,.0f}")
            m2.metric("Daily Average",        f"${est.get('forecast_daily_avg', 0):,.2f}")
            m3.metric("vs Last 30 Days",      f"{est.get('change_pct', 0):+.1f}%",
                      delta_color="inverse")

            fdf = analysis.forecast_df.copy()
            fdf["ds"] = pd.to_datetime(fdf["ds"])
            historical_end = df["date"].max()
            hist   = fdf[fdf["ds"] <= historical_end]
            future = fdf[fdf["ds"] > historical_end]

            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=hist["ds"], y=hist["yhat"],
                mode="lines", name="Historical (fitted)",
                line=dict(color="#667eea", width=1.5),
            ))
            fig_fc.add_trace(go.Scatter(
                x=future["ds"], y=future["yhat"],
                mode="lines", name="Forecast",
                line=dict(color="#f6a623", width=2.5, dash="dot"),
            ))
            if "yhat_upper" in future.columns and not future.empty:
                fig_fc.add_trace(go.Scatter(
                    x=pd.concat([future["ds"], future["ds"][::-1]]),
                    y=pd.concat([future["yhat_upper"], future["yhat_lower"][::-1]]),
                    fill="toself", fillcolor="rgba(246,166,35,0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Confidence Band",
                ))
            # add_vline requires a numeric x for datetime axes (milliseconds since epoch)
            historical_end_ms = int(pd.Timestamp(historical_end).timestamp() * 1000)
            fig_fc.add_vline(x=historical_end_ms, line_dash="dash",
                             line_color="gray", annotation_text="Today")
            fig_fc.update_layout(
                height=440,
                xaxis_title="Date", yaxis_title="Daily Spend ($)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_fc, use_container_width=True)
        else:
            st.info("Run the analysis to see the forecast.")

    # ---- TAB 4: Insights -----------------------------------------------------
    with tab4:
        if recs:
            if recs.used_llm:
                st.success("Enhanced with LLM reasoning")
            else:
                st.info("Rule-based insights active. Add OPENAI_API_KEY for LLM enhancement.")

            st.subheader("Financial Insights & Recommendations")
            for i, insight in enumerate(recs.all_insights, 1):
                if any(w in insight.lower() for w in ["great", "well done", "strong", "less"]):
                    card_class = "good-card"
                    prefix = "[+]"
                elif any(w in insight.lower() for w in ["increase", "more", "deficit", "low", "critically"]):
                    card_class = "warning-card"
                    prefix = "[!]"
                else:
                    card_class = "insight-card"
                    prefix = "[i]"
                st.markdown(
                    f'<div class="{card_class}"><b>{prefix} #{i}</b> &mdash; {insight}</div>',
                    unsafe_allow_html=True,
                )

            st.divider()
            with st.expander("Full Spending Summary"):
                st.text(recs.summary_text)
        else:
            st.info("Run the analysis to generate insights.")

    # ---- TAB 5: Raw Data -----------------------------------------------------
    with tab5:
        st.subheader("Cleaned Transactions")
        display_df = (
            analysis.categorized_df
            if (analysis and not analysis.categorized_df.empty)
            else df
        )
        cols_to_show = [
            c for c in [
                "date", "merchant", "description", "source_category",
                "amount", "type", "category", "is_transfer",
            ]
            if c in display_df.columns
        ]
        st.dataframe(
            display_df[cols_to_show].style.format({"amount": "${:,.2f}"}),
            use_container_width=True,
            height=500,
        )
        csv_out = display_df[cols_to_show].to_csv(index=False)
        st.download_button(
            "Download CSV",
            data=csv_out,
            file_name="finance_ai_categorized.csv",
            mime="text/csv",
        )
        if report:
            st.subheader("Ingestion Report")
            st.text(str(report))

else:
    # Landing state
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Upload")
        st.markdown("Upload your bank CSV or use the built-in sample data to explore.")
    with col2:
        st.markdown("### Analyze")
        st.markdown("ML categorization and Prophet forecasting runs automatically.")
    with col3:
        st.markdown("### Act")
        st.markdown("Get measurable insights to reduce spending and save more.")
    st.markdown("---")
    st.markdown("Click **Run Analysis** in the sidebar to start.")
