"""
Phase 3 – Forecasting

Predicts future spending using Prophet when available with holdout diagnostics,
otherwise linear regression. Uses rolling smoothing when spend is sparse (many zero days).
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

from sklearn.linear_model import LinearRegression

from models.evaluation import mape as mape_metric, rmse as rmse_metric


def _daily_expenses(df: pd.DataFrame) -> pd.DataFrame:
    expenses = df[df["amount"] < 0].copy()
    expenses["amount"] = expenses["amount"].abs()
    daily = (
        expenses.groupby("date")["amount"]
        .sum()
        .reset_index()
        .rename(columns={"date": "ds", "amount": "y"})
    )
    daily["ds"] = pd.to_datetime(daily["ds"])
    full_range = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
    daily = daily.set_index("ds").reindex(full_range, fill_value=0).reset_index()
    daily.columns = ["ds", "y"]
    return daily


def _apply_sparse_smoothing_if_needed(daily: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta = {"sparse_smoothing": False, "zero_ratio": 0.0}
    if daily.empty:
        return daily, meta
    zr = float((daily["y"] == 0).mean())
    meta["zero_ratio"] = zr
    if zr > 0.55:
        out = daily.copy()
        out["y"] = out["y"].rolling(window=7, center=True, min_periods=1).mean()
        meta["sparse_smoothing"] = True
        return out, meta
    return daily, meta


def _prophet_kwargs_for_series(daily: pd.DataFrame) -> dict[str, Any]:
    if daily.empty or len(daily) < 2:
        return {}
    span = (daily["ds"].iloc[-1] - daily["ds"].iloc[0]).days + 1
    mean_y = float(daily["y"].mean())
    zero_ratio = float((daily["y"] == 0).mean())

    yearly = span >= 730
    additive = zero_ratio > 0.45 or mean_y < 25
    seas_mode = "additive" if additive else "multiplicative"
    grids = [0.05, 0.15, 0.4]
    return {
        "yearly_seasonality": yearly,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "seasonality_mode": seas_mode,
        "changepoint_prior_scale": grids[1],
        "_cps_candidates": grids,
    }


def _prophet_holdout_fit(
    train: pd.DataFrame,
    horizons: int,
    changepoint_prior_scale: float,
    seasonality_mode: str,
    yearly_seasonality: bool,
) -> np.ndarray | None:
    """Return in-sample predictions for last `horizons` points (simulated extrapolation one-step style)."""
    if not PROPHET_AVAILABLE or len(train) <= horizons + 14:
        return None
    m = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
    )
    fit_df = train.iloc[: -horizons].rename(columns={"y": "y"})
    m.fit(fit_df[["ds", "y"]])
    future = train[["ds"]]
    preds = m.predict(future[["ds"]])
    pr = preds["yhat"].values[-horizons:]
    actual = train["y"].values[-horizons:]
    return np.array(actual), np.array(pr)


class SpendingForecaster:
    """Forecast daily spending for the next N days."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        raw_daily = _daily_expenses(df)
        self.daily_smooth, smooth_meta = _apply_sparse_smoothing_if_needed(raw_daily)
        self._daily_raw = raw_daily
        self._smooth_meta = smooth_meta
        self.daily_fit = self.daily_smooth  # DataFrame prophet fits on
        self._prophet_model: Optional[object] = None
        self._lr_model: Optional[LinearRegression] = None
        self._forecast_df: Optional[pd.DataFrame] = None
        self.model_used: str = ""
        self.backtest_metrics: dict[str, Any] = {}

    def _pick_changepoint_scale(self, daily_fit: pd.DataFrame, base_kwargs: dict) -> float:
        cands = list(base_kwargs.get("_cps_candidates", [0.05, 0.15, 0.4]))
        h = min(14, max(5, len(daily_fit) // 8))
        if h < 5 or not PROPHET_AVAILABLE or len(daily_fit) < h + 20:
            return float(base_kwargs.get("changepoint_prior_scale", 0.15))

        default_cp = 0.15
        best_cp, best_mape = default_cp, float("inf")
        for cp in cands:
            out = _prophet_holdout_fit(
                daily_fit,
                horizons=h,
                changepoint_prior_scale=cp,
                seasonality_mode=base_kwargs["seasonality_mode"],
                yearly_seasonality=base_kwargs["yearly_seasonality"],
            )
            if out is None:
                continue
            act, pred = out
            sc = mape_metric(act, pred)
            if sc < best_mape:
                best_mape, best_cp = sc, cp
        if best_mape < float("inf"):
            self.backtest_metrics["holdout_mape_pct"] = round(best_mape * 100, 2)
            self.backtest_metrics["holdout_days"] = h
            self.backtest_metrics["changepoint_prior_scale"] = best_cp
        return best_cp

    def _run_backtest_lr(self):
        if len(self.daily_fit) <= 20:
            return
        train_n = len(self.daily_fit) - 14
        if train_n < 10:
            return
        X = np.arange(train_n).reshape(-1, 1)
        y = self.daily_fit["y"].values[:train_n]
        lr = LinearRegression().fit(X, y)
        Xh = np.arange(train_n, len(self.daily_fit)).reshape(-1, 1)
        preds = lr.predict(Xh).clip(min=0)
        act = self.daily_fit["y"].values[train_n:]
        if len(act):
            self.backtest_metrics["holdout_rmse"] = round(rmse_metric(act, preds), 4)
            self.backtest_metrics["holdout_mape_pct"] = round(mape_metric(act, preds) * 100, 2)

    # ── Fit ────────────────────────────────────────────────────────────────────
    def _fit_prophet_with_tuning(self):
        base = dict(_prophet_kwargs_for_series(self.daily_fit))
        if not base:
            self._lr_model = None
            raise ValueError("insufficient_series")
        cands = list(base.pop("_cps_candidates", [0.05, 0.15, 0.4]))
        base["_cps_candidates"] = cands
        cp = self._pick_changepoint_scale(self.daily_fit, base)
        self._prophet_model = Prophet(
            yearly_seasonality=base["yearly_seasonality"],
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=cp,
            seasonality_mode=base["seasonality_mode"],
        )
        self._prophet_model.fit(self.daily_fit[["ds", "y"]])
        self.model_used = "Prophet"
        if self._smooth_meta.get("sparse_smoothing"):
            self.model_used = "Prophet (7-day smoothed)"
        zm = round(self._smooth_meta.get("zero_ratio", 0) * 100, 1)
        self.backtest_metrics["sparse_zero_ratio_pct"] = zm

    def _fit_linear(self):
        X = np.arange(len(self.daily_fit)).reshape(-1, 1)
        y = self.daily_fit["y"].values
        self._lr_model = LinearRegression().fit(X, y)
        self.model_used = "Linear Regression"

    # ── Forecast ───────────────────────────────────────────────────────────────
    def forecast(self, days: int = 30) -> pd.DataFrame:
        self.backtest_metrics = {}
        zm = round(self._smooth_meta.get("zero_ratio", 0) * 100, 1)
        self.backtest_metrics["sparse_zero_ratio_pct"] = zm

        if self.daily_fit.empty:
            self._forecast_df = pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])
            return self._forecast_df

        if PROPHET_AVAILABLE and len(self.daily_fit) >= 14:
            try:
                self._fit_prophet_with_tuning()
                future = self._prophet_model.make_future_dataframe(periods=days, freq="D")
                raw = self._prophet_model.predict(future)
                raw["yhat"] = raw["yhat"].clip(lower=0)
                raw["yhat_lower"] = raw["yhat_lower"].clip(lower=0)
                raw["yhat_upper"] = raw["yhat_upper"].clip(lower=0)
                self._forecast_df = raw[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
            except Exception:
                self._forecast_df = None
                self._lr_model = None
                self._fit_linear_with_forecast(days)
        else:
            self._fit_linear_with_forecast(days)

        return self._forecast_df if self._forecast_df is not None else pd.DataFrame()

    def _fit_linear_with_forecast(self, days: int) -> None:
        self._fit_linear()
        n = len(self.daily_fit)
        future_idx = np.arange(n, n + days).reshape(-1, 1)
        future_dates = pd.date_range(
            self.daily_fit["ds"].iloc[-1] + pd.Timedelta(days=1), periods=days
        )
        preds = self._lr_model.predict(future_idx).clip(min=0)  # type: ignore
        hist_y = self._daily_raw["y"] if len(self._daily_raw) == n else self.daily_fit["y"]
        historical = pd.DataFrame({
            "ds": self.daily_fit["ds"],
            "yhat": hist_y.values,
            "yhat_lower": hist_y.values * 0.85,
            "yhat_upper": hist_y.values * 1.15,
        })
        future_df = pd.DataFrame({
            "ds": future_dates,
            "yhat": preds,
            "yhat_lower": (preds * 0.85),
            "yhat_upper": (preds * 1.15),
        })
        self._forecast_df = pd.concat([historical, future_df], ignore_index=True)
        self._run_backtest_lr()

    # ── Summary ─────────────────────────────────────────────────────────────────
    def next_month_estimate(self) -> dict:
        if self._forecast_df is None:
            self.forecast(days=30)
        future = (
            self._forecast_df.tail(30)
            if self._forecast_df is not None and not self._forecast_df.empty
            else None
        )
        if future is None or future.empty:
            return {}
        total = future["yhat"].sum()
        daily_avg = future["yhat"].mean()
        last_30 = self._daily_raw.tail(30)["y"].sum()
        change_pct = ((total - last_30) / last_30 * 100) if last_30 > 0 else 0
        out = {
            "model": self.model_used,
            "forecast_total_30d": round(total, 2),
            "forecast_daily_avg": round(daily_avg, 2),
            "last_30d_actual": round(last_30, 2),
            "change_pct": round(change_pct, 1),
        }
        out.update({f"diag_{k}": v for k, v in self.backtest_metrics.items()})
        return out

    def category_forecast(self, df_categorized: pd.DataFrame, days: int = 30) -> pd.DataFrame:
        results = []
        expenses = df_categorized[df_categorized["amount"] < 0].copy()
        expenses["amount"] = expenses["amount"].abs()

        for cat, grp in expenses.groupby("category"):
            monthly = (
                grp.groupby(["year", "month"])["amount"]
                .sum()
                .reset_index()
                .sort_values(["year", "month"])
            )
            amt_vals = monthly["amount"].tolist()
            if len(monthly) == 0:
                continue

            if len(monthly) < 3:
                naive = amt_vals[-1] if amt_vals else 0.0
                trail = monthly["amount"].tail(2).mean() if len(monthly) >= 2 else naive
                results.append({"category": cat, "forecast": round(max(float(trail), 0), 2)})
                continue

            prev = amt_vals[-1]
            seasonal = amt_vals[-2] if len(amt_vals) >= 2 else prev
            if len(monthly) < 6:
                next_val = max(0, (prev + seasonal) / 2)
                results.append({"category": cat, "forecast": round(next_val, 2)})
                continue

            X = np.arange(len(monthly)).reshape(-1, 1)
            y = monthly["amount"].values
            lr = LinearRegression().fit(X, y)
            raw_next = float(lr.predict([[len(monthly)]])[0])
            trail_mean = float(monthly["amount"].iloc[-min(6, len(monthly)) :].mean())
            next_val = max(0, min(raw_next, trail_mean * 2.5, y[-1] * 4 if len(y) else raw_next))

            results.append({"category": cat, "forecast": round(next_val, 2)})

        return pd.DataFrame(results).sort_values("forecast", ascending=False).reset_index(drop=True)
