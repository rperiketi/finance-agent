"""Holdout metrics for daily spend forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((actual - pred) ** 2)))


def mape(actual: np.ndarray, pred: np.ndarray, epsilon: float = 1e-9) -> float:
    denom = np.maximum(np.abs(actual), epsilon)
    return float(np.nanmean(np.abs((actual - pred) / denom)))


def daily_expense_holdout_metrics(daily_df: pd.DataFrame, preds: pd.Series | np.ndarray, holdout_days: int) -> dict:
    """
    daily_df: rows with ds, y sorted by ds
    preds: predictions aligned index with daily_df on last holdout_days
    """
    if len(daily_df) <= holdout_days + 7:
        return {}
    hist = daily_df.iloc[:-holdout_days]
    act = hist["y"].values
    actual_idx = hist.index.union(daily_df.iloc[-holdout_days:].index)
    # preds should be aligned to full dataframe yhat column
    if hasattr(preds, "iloc"):
        pr = preds.reindex(hist.index).values
    else:
        pr = preds[: len(hist)]
        if len(pr) != len(act):
            return {}
    return {
        "holdout_rmse": round(rmse(act, np.asarray(pr)), 4),
        "holdout_mape_pct": round(mape(act, np.asarray(pr)) * 100, 2),
        "train_points": len(hist),
        "holdout_days": holdout_days,
    }
