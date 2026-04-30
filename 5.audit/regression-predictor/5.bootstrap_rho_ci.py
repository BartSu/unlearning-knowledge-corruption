"""Bootstrap 95% CI of Spearman rho on LOO audit predictions (n=10)."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
PRED = ROOT / "audit" / "part2_audit_predictions.csv"
SUMMARY = ROOT / "audit" / "audit_summary.json"

N_BOOT = 10000
SEED = 0
LAYERS = ("L1_forget", "L2_locality", "L3_spillover")


def bootstrap_rho(y_true: np.ndarray, y_pred: np.ndarray,
                  n_boot: int, rng: np.random.Generator) -> dict:
    n = len(y_true)
    rhos = np.empty(n_boot)
    k = 0
    tries = 0
    while k < n_boot and tries < n_boot * 20:
        tries += 1
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        if np.unique(yt).size < 2 or np.unique(yp).size < 2:
            continue
        r, _ = spearmanr(yt, yp)
        if np.isnan(r):
            continue
        rhos[k] = r
        k += 1
    rhos = rhos[:k]
    point, _ = spearmanr(y_true, y_pred)
    return {
        "rho_point": float(point),
        "rho_mean": float(rhos.mean()),
        "rho_ci_low_95": float(np.percentile(rhos, 2.5)),
        "rho_ci_high_95": float(np.percentile(rhos, 97.5)),
        "n_boot_effective": int(k),
    }


def main():
    df = pd.read_csv(PRED)
    rng = np.random.default_rng(SEED)
    out = {}
    for layer in LAYERS:
        y = df[f"true_geo_{layer}"].values.astype(float)
        yhat = df[f"pred_geo_{layer}"].values.astype(float)
        out[layer] = bootstrap_rho(y, yhat, N_BOOT, rng)
        print(f"{layer:<14s}  ρ={out[layer]['rho_point']:+.3f}  "
              f"95% CI [{out[layer]['rho_ci_low_95']:+.3f}, "
              f"{out[layer]['rho_ci_high_95']:+.3f}]  "
              f"(n_boot={out[layer]['n_boot_effective']})")

    with open(SUMMARY) as f:
        summary = json.load(f)
    summary["bootstrap_rho_ci"] = {
        "n": int(len(df)),
        "n_boot": N_BOOT,
        "seed": SEED,
        "method": "percentile bootstrap over 10 LOO (true, predicted) pairs",
        "layers": out,
    }
    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote bootstrap_rho_ci to {SUMMARY}")


if __name__ == "__main__":
    main()
