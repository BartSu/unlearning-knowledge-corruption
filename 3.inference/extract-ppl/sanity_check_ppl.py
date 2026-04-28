"""Sanity-check base vs unlearned PPL distributions.

Reads ppl_long.parquet (produced by export_ppl_table.py) and asserts the basic
invariants we expect from a working GradAscent unlearn run:

  * L1 (forget train) geo-mean ppl ratio > 1   (unlearn must hurt forget set)
  * L1 mean log-ratio > L2 mean log-ratio > L3 (monotone decay)
  * Base PPL distributions on L1 vs L3 are not pathologically different
    (the base model is the same — any gap is sampling noise)
  * No NaN / non-positive PPL values

Prints a small distribution table plus pass/fail per check and exits non-zero
if any invariant fails.
"""

from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PARQUET = HERE / "ppl_long.parquet"


def summarize(name: str, s: pd.Series) -> dict:
    a = s.to_numpy()
    return {
        "name": name,
        "n": int(a.size),
        "base_or_log": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a)),
    }


def main() -> int:
    if not PARQUET.exists():
        print(f"ERROR: {PARQUET} not found — run export_ppl_table.py first", file=sys.stderr)
        return 2

    df = pd.read_parquet(PARQUET)
    failures: list[str] = []

    # 1. No NaN / non-positive
    for col in ("base_ppl", "unlearn_ppl"):
        bad = df[col].isna().sum() + (df[col] <= 0).sum()
        if bad:
            failures.append(f"{col}: {bad} NaN/non-positive rows")

    # 2. Per-layer geo-mean ratio
    print("=" * 68)
    print("  Per-layer log(ppl_unlearn / ppl_base)")
    print("=" * 68)
    layer_stats = {}
    for layer in ("L1", "L2", "L3"):
        sub = df[df["layer"] == layer]
        if sub.empty:
            failures.append(f"layer {layer} has no rows")
            continue
        lr = sub["log_ppl_ratio"].to_numpy()
        stats = {
            "n": int(lr.size),
            "geo_mean_ratio": float(math.exp(lr.mean())),
            "mean_log": float(lr.mean()),
            "median_log": float(np.median(lr)),
            "std_log": float(lr.std()),
            "pct_up_1_1": float((lr > math.log(1.1)).mean() * 100),
        }
        layer_stats[layer] = stats
        print(f"  {layer}  n={stats['n']:5d}  geo={stats['geo_mean_ratio']:.3f}x  "
              f"mean_log={stats['mean_log']:+.4f}  med_log={stats['median_log']:+.4f}  "
              f">1.1x: {stats['pct_up_1_1']:5.1f}%")

    # 3. Invariants
    print("\n" + "=" * 68)
    print("  Invariants")
    print("=" * 68)
    checks = []

    if "L1" in layer_stats:
        ok = layer_stats["L1"]["geo_mean_ratio"] > 1.0
        checks.append(("L1 geo-mean > 1 (unlearn hurts forget set)", ok))
        if not ok:
            failures.append("L1 geo-mean <= 1")

    if {"L1", "L2", "L3"} <= layer_stats.keys():
        m1 = layer_stats["L1"]["mean_log"]
        m2 = layer_stats["L2"]["mean_log"]
        m3 = layer_stats["L3"]["mean_log"]
        ok = m1 > m2 > m3
        checks.append(("L1 > L2 > L3 monotone decay (mean log-ratio)", ok))
        if not ok:
            failures.append(f"monotone decay violated: L1={m1:.4f} L2={m2:.4f} L3={m3:.4f}")

    # 4. Base PPL sanity: same base model, so base distributions across layers
    #    should overlap within a few x. Compare log(base_ppl) means.
    print("\n" + "=" * 68)
    print("  log(base_ppl) by layer (should be similar across layers)")
    print("=" * 68)
    base_means = {}
    for layer in ("L1", "L2", "L3"):
        sub = df[df["layer"] == layer]
        if sub.empty:
            continue
        lb = np.log(sub["base_ppl"].to_numpy().clip(min=1e-6))
        base_means[layer] = float(lb.mean())
        print(f"  {layer}  mean log(base_ppl) = {base_means[layer]:+.4f}  "
              f"(geo base_ppl = {math.exp(base_means[layer]):.2f})")
    if {"L1", "L3"} <= base_means.keys():
        gap = abs(base_means["L1"] - base_means["L3"])
        ok = gap < math.log(3.0)  # within 3x
        checks.append(("|log base_ppl L1 - L3| < log(3) (base model consistent)", ok))
        if not ok:
            failures.append(f"base PPL gap L1 vs L3 too large: {gap:.4f} nats")

    # 5. Unlearn increases PPL more often than it decreases (on L1)
    if "L1" in layer_stats:
        l1 = df[df["layer"] == "L1"]
        frac_up = (l1["log_ppl_ratio"] > 0).mean()
        ok = frac_up > 0.5
        checks.append((f"L1: >50% samples see ppl rise (got {frac_up*100:.1f}%)", ok))
        if not ok:
            failures.append(f"L1 frac_up = {frac_up:.3f}")

    for msg, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")

    print("\n" + "=" * 68)
    if failures:
        print(f"  SANITY CHECK FAILED ({len(failures)} issue(s))")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  SANITY CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
