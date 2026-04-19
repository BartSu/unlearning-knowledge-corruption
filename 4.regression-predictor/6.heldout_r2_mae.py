"""Held-out (LOO) R² / MAE / RMSE on the audit predictor (n=10).

LOO at n=10 is the closest approximation to a held-out report without
re-running unlearn. Each fold: Ridge(alpha=1.0) trained on 9 forget sets,
predicts the geo_L{1,2,3} ratio of the 1 held-out forget set.

We reuse the predictions already written by 4.audit_experiments.py
(part2_audit_predictions.csv) and add MAE + a mean-predictor baseline so
the reader can judge whether the audit beats "just predict the mean".
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PRED = ROOT / "audit" / "part2_audit_predictions.csv"
SUMMARY = ROOT / "audit" / "audit_summary.json"
LAYERS = ("L1_forget", "L2_locality", "L3_spillover")


def metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    err = y - yhat
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mean_abs_y": float(np.abs(y).mean()),
        "mae_over_mean_y": float(np.abs(err).mean() / max(np.abs(y).mean(), 1e-12)),
    }


def baseline_loo_mean(y: np.ndarray) -> np.ndarray:
    """Held-out mean predictor: for each i, predict mean of all other y."""
    n = len(y)
    total = y.sum()
    return (total - y) / (n - 1)


def main():
    df = pd.read_csv(PRED)
    out = {}
    print(f"{'layer':<14s} {'R²':>7s} {'MAE':>8s} {'RMSE':>8s} "
          f"{'baseline R²':>12s} {'baseline MAE':>13s}")
    for layer in LAYERS:
        y = df[f"true_geo_{layer}"].values.astype(float)
        yhat = df[f"pred_geo_{layer}"].values.astype(float)
        m = metrics(y, yhat)
        b = metrics(y, baseline_loo_mean(y))
        out[layer] = {"audit": m, "baseline_loo_mean": b}
        print(f"{layer:<14s} {m['r2']:+.3f} {m['mae']:8.4f} {m['rmse']:8.4f} "
              f"{b['r2']:+12.3f} {b['mae']:13.4f}")

    with open(SUMMARY) as f:
        summary = json.load(f)
    summary["heldout_r2_mae"] = {
        "n": int(len(df)),
        "protocol": "LOO over 10 forget-set clusters; Ridge(alpha=1.0) on 12-d forget-set geometry",
        "note": "Baseline predicts the mean of the 9 training targets (also LOO).",
        "layers": out,
    }
    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote heldout_r2_mae to {SUMMARY}")


if __name__ == "__main__":
    main()
