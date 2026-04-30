"""Ranking-side audit metrics on LOO predictions (n=100).

Motivation
----------
Reporting only continuous R² / Spearman ρ obscures Act II's actual sales pitch:
"cheap pre-screen → only run real unlearn on top-k risky forget sets". The right
question is "of the truly worst k% forget sets, how many does the audit catch?"
— i.e. top-k recall, NDCG, Kendall tau, pairwise concordance.

Reads
-----
- audit/part2_audit_predictions.csv   (written by 4.audit_experiments.py)

Writes
------
- Augments audit/audit_summary.json with top-level field `ranking_metrics`.

Run after 4.audit_experiments.py. Idempotent; preserves other top-level fields
(layer_headline, audit_predictor, coverage_vs_spillover, bootstrap_rho_ci,
heldout_r2_mae).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parent
PRED = ROOT / "audit" / "part2_audit_predictions.csv"
SUMMARY = ROOT / "audit" / "audit_summary.json"

LAYERS = ("L1_forget", "L2_locality", "L3_spillover")
TOPK_PCT = (5, 10, 20)
N_BOOT = 10000
SEED = 0


def dcg(rel: np.ndarray) -> float:
    rel = np.asarray(rel, dtype=float)
    if len(rel) == 0:
        return 0.0
    return float(((2.0 ** rel - 1.0) / np.log2(np.arange(2, len(rel) + 2))).sum())


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """NDCG@k with relevance = min-max normalized y_true (graded, not binary).

    Catches the worst storms with high weight; degenerate (all-equal y_true)
    returns 1.0.
    """
    lo, hi = float(y_true.min()), float(y_true.max())
    if hi - lo < 1e-12:
        return 1.0
    rel = (y_true - lo) / (hi - lo)
    order_pred = np.argsort(-y_pred)[:k]
    order_ideal = np.argsort(-y_true)[:k]
    idcg = dcg(rel[order_ideal])
    if idcg <= 0:
        return 0.0
    return dcg(rel[order_pred]) / idcg


def pairwise_concordance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of distinct ordered pairs (i,j) with sign(Δy_true) == sign(Δy_pred)."""
    n = len(y_true)
    i, j = np.triu_indices(n, k=1)
    s_true = np.sign(y_true[i] - y_true[j])
    s_pred = np.sign(y_pred[i] - y_pred[j])
    valid = s_true != 0
    if valid.sum() == 0:
        return float("nan")
    return float((s_true[valid] == s_pred[valid]).mean())


def topk_recall_with_ci(y_true: np.ndarray, y_pred: np.ndarray, k: int,
                        n_boot: int, rng: np.random.Generator) -> dict:
    """Top-k recall = |true_top_k ∩ pred_top_k| / k, with bootstrap CI over
    resampled forget sets (preserves k as a fixed integer)."""
    n = len(y_true)
    true_top = set(np.argsort(-y_true)[:k].tolist())
    pred_top = set(np.argsort(-y_pred)[:k].tolist())
    point_hits = len(true_top & pred_top)

    rec_boot = np.empty(n_boot)
    j = 0
    tries = 0
    while j < n_boot and tries < n_boot * 20:
        tries += 1
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        if np.unique(yt).size < 2 or np.unique(yp).size < 2:
            continue
        tt = set(np.argsort(-yt)[:k].tolist())
        pt = set(np.argsort(-yp)[:k].tolist())
        rec_boot[j] = len(tt & pt) / k
        j += 1
    rec_boot = rec_boot[:j]
    return {
        "k": int(k),
        "hits": int(point_hits),
        "recall_point": point_hits / k,
        "recall_mean_boot": float(rec_boot.mean()) if j else float("nan"),
        "recall_ci_low_95": float(np.percentile(rec_boot, 2.5)) if j else float("nan"),
        "recall_ci_high_95": float(np.percentile(rec_boot, 97.5)) if j else float("nan"),
        "random_recall": k / n,
        "lift_over_random": (point_hits / k) / (k / n) if k > 0 else float("nan"),
        "n_boot_effective": int(j),
    }


def main():
    df = pd.read_csv(PRED)
    n = len(df)
    rng = np.random.default_rng(SEED)

    print("=" * 78)
    print(f"  Ranking metrics on LOO predictions (n_forget_sets = {n})")
    print("=" * 78)

    out = {
        "n": n,
        "n_boot": N_BOOT,
        "seed": SEED,
        "topk_pct": list(TOPK_PCT),
        "k_values": {f"{p}pct": max(1, int(round(n * p / 100))) for p in TOPK_PCT},
        "layers": {},
    }

    for layer in LAYERS:
        y_true = df[f"true_geo_{layer}"].values.astype(float)
        y_pred = df[f"pred_geo_{layer}"].values.astype(float)

        rho, _ = spearmanr(y_true, y_pred)
        tau, _ = kendalltau(y_true, y_pred)
        pcf = pairwise_concordance(y_true, y_pred)

        layer_out = {
            "spearman_rho": float(rho),
            "kendall_tau": float(tau),
            "pairwise_concordance": pcf,
            "topk": {},
        }
        for p in TOPK_PCT:
            k = max(1, int(round(n * p / 100)))
            tk = topk_recall_with_ci(y_true, y_pred, k, N_BOOT, rng)
            tk["ndcg"] = ndcg_at_k(y_true, y_pred, k)
            layer_out["topk"][f"{p}pct"] = tk

        out["layers"][layer] = layer_out

        print(f"\n  {layer}")
        print(f"    Spearman ρ = {rho:+.3f}   Kendall τ = {tau:+.3f}   "
              f"pairwise_concord = {pcf:.3f}")
        for p in TOPK_PCT:
            tk = layer_out["topk"][f"{p}pct"]
            print(f"    top-{p:>2d}% (k={tk['k']:>2d}): "
                  f"hits={tk['hits']:>2d}/{tk['k']:<2d}  "
                  f"recall={tk['recall_point']:.2f} "
                  f"[CI {tk['recall_ci_low_95']:.2f}, {tk['recall_ci_high_95']:.2f}]  "
                  f"random={tk['random_recall']:.2f}  "
                  f"lift={tk['lift_over_random']:.2f}x  "
                  f"NDCG={tk['ndcg']:.3f}")

    if SUMMARY.exists():
        prev = json.loads(SUMMARY.read_text())
    else:
        prev = {}
    prev["ranking_metrics"] = out
    SUMMARY.write_text(json.dumps(prev, indent=2))
    print(f"\nWrote ranking_metrics to {SUMMARY}")


if __name__ == "__main__":
    main()
