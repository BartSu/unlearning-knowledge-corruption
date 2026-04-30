"""Alternative predictors on the same 12-dim forget-set geometry.

Reproduces the LOO protocol of 4.audit_experiments.py and runs Ridge /
RandomForest / GradientBoosting / Lasso side by side. Paper headline
(Ridge in 4.audit_experiments) is unchanged; this script only augments
audit_summary.json with `alt_predictors` for ablation.

Reads
-----
- ../../4.feature-engineering/forget_set_geometry.csv   (X, 12 dims)
- audit/part1_corruption_profile.csv                    (y, 3 layers)

Writes
------
- audit/audit_summary.json["alt_predictors"][model][layer] = {
      r2, mae, spearman_rho, rho_ci_low_95, rho_ci_high_95,
      topk_pct=10: {hits, recall, lift_over_random}
  }
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import LeaveOneOut

ROOT = Path(__file__).resolve().parents[2]
FORGET_GEOMETRY_CSV = ROOT / "4.feature-engineering" / "forget_set_geometry.csv"
PROFILE_CSV = Path(__file__).resolve().parent / "audit" / "part1_corruption_profile.csv"
SUMMARY = Path(__file__).resolve().parent / "audit" / "audit_summary.json"

LAYERS = ("L1_forget", "L2_locality", "L3_spillover")
N_BOOT = 10000
SEED = 0
TOPK_PCT = 10


def loo_predict(model_factory, Xs: np.ndarray, y: np.ndarray) -> np.ndarray:
    yhat = np.empty_like(y)
    loo = LeaveOneOut()
    for tr, te in loo.split(Xs):
        m = model_factory().fit(Xs[tr], y[tr])
        yhat[te] = m.predict(Xs[te])
    return yhat


def bootstrap_rho_ci(y: np.ndarray, yhat: np.ndarray, n_boot: int,
                     rng: np.random.Generator) -> tuple[float, float]:
    n = len(y)
    rhos = np.empty(n_boot)
    k = 0
    tries = 0
    while k < n_boot and tries < n_boot * 20:
        tries += 1
        idx = rng.integers(0, n, size=n)
        yt, yp = y[idx], yhat[idx]
        if np.unique(yt).size < 2 or np.unique(yp).size < 2:
            continue
        r, _ = spearmanr(yt, yp)
        if np.isnan(r):
            continue
        rhos[k] = r
        k += 1
    rhos = rhos[:k]
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def topk_recall(y: np.ndarray, yhat: np.ndarray, k: int) -> dict:
    n = len(y)
    true_top = set(np.argsort(-y)[:k].tolist())
    pred_top = set(np.argsort(-yhat)[:k].tolist())
    hits = len(true_top & pred_top)
    return {
        "k": int(k),
        "hits": int(hits),
        "recall": hits / k,
        "random_recall": k / n,
        "lift_over_random": (hits / k) / (k / n) if k > 0 else float("nan"),
    }


def main():
    feat = pd.read_csv(FORGET_GEOMETRY_CSV)
    prof = pd.read_csv(PROFILE_CSV)
    df = feat.merge(prof, on="forget_cluster", how="inner").sort_values("forget_cluster")
    feature_cols = [c for c in feat.columns if c != "forget_cluster"]
    X = df[feature_cols].values.astype(float)

    X_mean, X_std = X.mean(0), X.std(0) + 1e-12
    Xs = (X - X_mean) / X_std

    n = len(df)
    k = max(1, int(round(n * TOPK_PCT / 100)))
    rng = np.random.default_rng(SEED)

    models = {
        "ridge_alpha1":            lambda: Ridge(alpha=1.0),
        "lasso_alpha0p01":         lambda: Lasso(alpha=0.01, max_iter=20000),
        "random_forest_n200":      lambda: RandomForestRegressor(
                                        n_estimators=200, max_depth=None,
                                        min_samples_leaf=2, random_state=0, n_jobs=-1),
        "gradient_boosting_n200":  lambda: GradientBoostingRegressor(
                                        n_estimators=200, max_depth=3,
                                        learning_rate=0.05, random_state=0),
    }

    print("=" * 78)
    print(f"  Alt predictors on 12-dim forget-set geometry  (n={n}, LOO, k_top={k})")
    print("=" * 78)
    print(f"  {'model':<26s} {'layer':<14s}  R²       MAE      ρ(95% CI)               top-{TOPK_PCT}% recall  lift")

    out = {
        "n": n,
        "n_boot": N_BOOT,
        "seed": SEED,
        "topk_pct": TOPK_PCT,
        "k": k,
        "feature_cols": feature_cols,
        "models": {},
    }

    for model_name, factory in models.items():
        out["models"][model_name] = {}
        for layer in LAYERS:
            target = f"geo_{layer}"
            y = df[target].values.astype(float)
            yhat = loo_predict(factory, Xs, y)

            r2 = 1.0 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            mae = float(np.abs(y - yhat).mean())
            rho, _ = spearmanr(y, yhat)
            ci_lo, ci_hi = bootstrap_rho_ci(y, yhat, N_BOOT, rng)
            tk = topk_recall(y, yhat, k)

            row = {
                "r2": float(r2),
                "mae": mae,
                "spearman_rho": float(rho),
                "rho_ci_low_95": ci_lo,
                "rho_ci_high_95": ci_hi,
                "topk": tk,
            }
            out["models"][model_name][layer] = row
            print(f"  {model_name:<26s} {layer:<14s}  "
                  f"{r2:+.3f}   {mae:.4f}   "
                  f"{rho:+.3f} [{ci_lo:+.2f}, {ci_hi:+.2f}]   "
                  f"{tk['hits']}/{tk['k']} ({tk['recall']:.2f})    "
                  f"{tk['lift_over_random']:.2f}x")
        print()

    if SUMMARY.exists():
        prev = json.loads(SUMMARY.read_text())
    else:
        prev = {}
    prev["alt_predictors"] = out
    SUMMARY.write_text(json.dumps(prev, indent=2))
    print(f"Wrote alt_predictors to {SUMMARY}")


if __name__ == "__main__":
    main()
