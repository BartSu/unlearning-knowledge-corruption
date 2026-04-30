"""Independent NPO audit (mirrors 4.audit_experiments.py protocol on NPO labels).

Same RF protocol, same LOO over n=100 forget sets, same 12-d forget-set
geometry input. Only the unlearner (and therefore ground-truth labels) changes
from GradAscent → NPO. Output is the NPO equivalent of audit_summary.json's
audit_predictor / bootstrap_rho_ci / ranking_metrics blocks.

Reads
-----
- ../../4.feature-engineering/forget_set_geometry.csv   (X, 12-d, trainer-agnostic)
- audit/npo100_headline.json["per_forget_profile"]      (y, NPO 100 forget sets)

Writes
------
- audit/npo_audit_summary.json {layer_headline, audit_predictor, bootstrap_rho_ci, ranking_metrics}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut

ROOT = Path(__file__).resolve().parents[2]
FORGET_GEOMETRY_CSV = ROOT / "4.feature-engineering" / "forget_set_geometry.csv"
NPO_HEAD_JSON = Path(__file__).resolve().parent / "audit" / "npo100_headline.json"
OUT_SUMMARY = Path(__file__).resolve().parent / "audit" / "npo_audit_summary.json"

LAYERS = ("L1_forget", "L2_locality", "L3_spillover")
TOPK_PCT = (5, 10, 20)
N_BOOT = 10000
SEED = 0


def loo_rf(Xs: np.ndarray, y: np.ndarray) -> np.ndarray:
    yhat = np.empty_like(y)
    loo = LeaveOneOut()
    for tr, te in loo.split(Xs):
        m = RandomForestRegressor(
            n_estimators=200, max_depth=None, min_samples_leaf=2,
            random_state=0, n_jobs=-1,
        ).fit(Xs[tr], y[tr])
        yhat[te] = m.predict(Xs[te])
    return yhat


def bootstrap_rho_ci(y: np.ndarray, yhat: np.ndarray, n_boot: int,
                     rng: np.random.Generator) -> tuple[float, float, float]:
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
    return float(rhos.mean()), float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def topk_block(y: np.ndarray, yhat: np.ndarray, k: int, n_boot: int,
               rng: np.random.Generator) -> dict:
    n = len(y)
    true_top = set(np.argsort(-y)[:k].tolist())
    pred_top = set(np.argsort(-yhat)[:k].tolist())
    hits = len(true_top & pred_top)
    rec = np.empty(n_boot)
    j = 0
    tries = 0
    while j < n_boot and tries < n_boot * 20:
        tries += 1
        idx = rng.integers(0, n, size=n)
        yt = y[idx]; yp = yhat[idx]
        if np.unique(yt).size < 2 or np.unique(yp).size < 2:
            continue
        tt = set(np.argsort(-yt)[:k].tolist())
        pt = set(np.argsort(-yp)[:k].tolist())
        rec[j] = len(tt & pt) / k
        j += 1
    rec = rec[:j]
    return {
        "k": int(k),
        "hits": int(hits),
        "recall_point": hits / k,
        "recall_ci_low_95": float(np.percentile(rec, 2.5)) if j else float("nan"),
        "recall_ci_high_95": float(np.percentile(rec, 97.5)) if j else float("nan"),
        "random_recall": k / n,
        "lift_over_random": (hits / k) / (k / n) if k > 0 else float("nan"),
    }


def main():
    npo = json.loads(NPO_HEAD_JSON.read_text())
    prof = pd.DataFrame(npo["per_forget_profile"])
    feat = pd.read_csv(FORGET_GEOMETRY_CSV)
    df = feat.merge(prof, on="forget_cluster", how="inner").sort_values("forget_cluster")
    if len(df) != len(prof):
        miss = set(prof["forget_cluster"]) - set(df["forget_cluster"])
        raise ValueError(f"Missing geometry features for: {miss}")

    feature_cols = [c for c in feat.columns if c != "forget_cluster"]
    X = df[feature_cols].values.astype(float)
    Xm, Xstd = X.mean(0), X.std(0) + 1e-12
    Xs = (X - Xm) / Xstd

    n = len(df)
    rng = np.random.default_rng(SEED)
    print("=" * 78)
    print(f"  Independent NPO audit  (n={n}, RF n=200/min_leaf=2/seed=0, LOO)")
    print("=" * 78)
    print(f"  {'layer':<14s}  R²       MAE      ρ(Spearman)   ρ 95% CI            top-5%/10%/20% recall")

    out = {
        "layer_headline": npo["headline"],
        "n": n,
        "audit_predictor": {},
        "bootstrap_rho_ci": {"n": n, "n_boot": N_BOOT, "seed": SEED, "layers": {}},
        "ranking_metrics": {"n": n, "topk_pct": list(TOPK_PCT),
                            "k_values": {f"{p}pct": max(1, int(round(n * p / 100))) for p in TOPK_PCT},
                            "layers": {}},
    }

    for layer in LAYERS:
        target = f"geo_{layer}"
        y = df[target].values.astype(float)
        yhat = loo_rf(Xs, y)

        ss_res = float(((y - yhat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        mae = float(np.abs(y - yhat).mean())
        rho, _ = spearmanr(y, yhat)
        tau, _ = kendalltau(y, yhat)

        rho_mean, ci_lo, ci_hi = bootstrap_rho_ci(y, yhat, N_BOOT, rng)

        out["audit_predictor"][f"geo_{layer}"] = {
            "r2": r2, "mae": mae, "rmse": float(np.sqrt(((y - yhat) ** 2).mean())),
            "spearman_rho": float(rho), "kendall_tau": float(tau),
        }
        out["bootstrap_rho_ci"]["layers"][layer] = {
            "rho_point": float(rho), "rho_mean": rho_mean,
            "rho_ci_low_95": ci_lo, "rho_ci_high_95": ci_hi,
        }

        topk_summary = []
        layer_topk = {}
        for p in TOPK_PCT:
            k = max(1, int(round(n * p / 100)))
            tk = topk_block(y, yhat, k, N_BOOT, rng)
            layer_topk[f"{p}pct"] = tk
            topk_summary.append(f"{tk['hits']}/{tk['k']}({tk['lift_over_random']:.1f}x)")
        out["ranking_metrics"]["layers"][layer] = {
            "spearman_rho": float(rho), "kendall_tau": float(tau),
            "topk": layer_topk,
        }

        print(f"  {layer:<14s}  {r2:+.3f}   {mae:.4f}   "
              f"{rho:+.3f}        [{ci_lo:+.2f}, {ci_hi:+.2f}]   "
              f"{' / '.join(topk_summary)}")

    OUT_SUMMARY.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
