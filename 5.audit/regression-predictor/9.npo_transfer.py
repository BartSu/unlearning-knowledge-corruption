"""NPO 30x30 cross-PPL → three-layer headline + audit transfer test.

Reads
-----
- ../../3.inference/extract-ppl/wikitext_cross_metrics_npo100_detail.json
   (per-sample base/unlearn loss+ppl for 30 NPO ckpt × 30 eval triplets)
- ../../4.feature-engineering/forget_set_geometry.csv
   (12-d forget-set features, already computed and trainer-agnostic)
- audit/part1_corruption_profile.csv
   (GradAscent ground-truth labels for n=100, used to train the RF auditor)

Writes
------
- audit/npo_headline.json   three-layer geo-mean for NPO and per-forget profile
- audit/npo_transfer.json   transfer rho/recall@k of GradAscent-trained RF on NPO labels
- prints summary table to stdout
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[2]
NPO_DETAIL = ROOT / "3.inference" / "extract-ppl" / "wikitext_cross_metrics_npo100_detail.json"
FORGET_GEOMETRY_CSV = ROOT / "4.feature-engineering" / "forget_set_geometry.csv"
GA_PROFILE_CSV = Path(__file__).resolve().parent / "audit" / "part1_corruption_profile.csv"
OUT_HEAD = Path(__file__).resolve().parent / "audit" / "npo100_headline.json"
OUT_TRANSFER = Path(__file__).resolve().parent / "audit" / "npo100_transfer.json"

LAYERS = ("L1_forget", "L2_locality", "L3_spillover")


def log_r(b: float, u: float) -> float:
    return math.log(max(u, 1e-6) / max(b, 1e-6))


def geo(a: np.ndarray) -> float:
    return float(math.exp(a.mean())) if len(a) else float("nan")


def build_per_sample(npo_detail: dict) -> pd.DataFrame:
    rows = []
    for r in npo_detail["results"]:
        m, e = r["model_triplet"], r["eval_triplet"]
        for split in ("train", "validation", "test"):
            base = r["base"].get(split, [])
            unl = r["unlearn"].get(split, [])
            if not base or not unl:
                continue
            for j, (b, u) in enumerate(zip(base, unl)):
                lr = log_r(b["ppl"], u["ppl"])
                if m == e and split == "train":
                    layer = "L1_forget"
                elif m == e and split in ("validation", "test"):
                    layer = "L2_locality"
                elif m != e and split == "test":
                    layer = "L3_spillover"
                else:
                    continue
                rows.append({
                    "forget_cluster": m, "eval_cluster": e, "split": split,
                    "sample_index": j, "layer": layer, "log_ppl_ratio": lr,
                })
    return pd.DataFrame(rows)


def build_per_forget_profile(per_sample: pd.DataFrame) -> pd.DataFrame:
    out = []
    for fc in sorted(per_sample["forget_cluster"].unique()):
        sub = per_sample[per_sample["forget_cluster"] == fc]
        prof = {"forget_cluster": fc}
        for layer in LAYERS:
            vals = sub.loc[sub["layer"] == layer, "log_ppl_ratio"].values
            prof[f"geo_{layer}"] = geo(vals)
            prof[f"n_{layer}"] = int(len(vals))
        out.append(prof)
    return pd.DataFrame(out)


def main():
    print("=" * 78)
    print("  NPO 30x30 — three-layer headline + audit transfer")
    print("=" * 78)

    with open(NPO_DETAIL) as f:
        detail = json.load(f)

    per_sample = build_per_sample(detail)
    print(f"per-sample rows: {len(per_sample):,}")
    print("layer counts:")
    for layer in LAYERS:
        n = (per_sample["layer"] == layer).sum()
        print(f"  {layer:<14s}  n={n}")

    # ---- NPO headline (pooled per-sample) ----
    print("\nNPO three-layer headline (pooled across 30 forget sets):")
    head = {}
    for layer in LAYERS:
        v = per_sample.loc[per_sample["layer"] == layer, "log_ppl_ratio"].values
        head[layer] = {
            "n": int(len(v)),
            "geo_mean_ratio": geo(v),
            "mean_log": float(v.mean()),
            "pct_up_10": float((v > math.log(1.1)).mean() * 100),
            "pct_up_2x": float((v > math.log(2.0)).mean() * 100),
        }
        print(f"  {layer:<14s}  geo={head[layer]['geo_mean_ratio']:.3f}x  "
              f">1.1x={head[layer]['pct_up_10']:.1f}%  >2x={head[layer]['pct_up_2x']:.1f}%  "
              f"(n={head[layer]['n']})")

    # ---- per-forget profile ----
    npo_prof = build_per_forget_profile(per_sample)
    print(f"\nNPO per-forget profile: {len(npo_prof)} forget sets")
    print(npo_prof.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    OUT_HEAD.write_text(json.dumps({
        "n_forget_sets": len(npo_prof),
        "headline": head,
        "per_forget_profile": npo_prof.to_dict(orient="records"),
    }, indent=2))
    print(f"\nWrote {OUT_HEAD}")

    # ---- audit transfer: RF trained on GradAscent labels, predict NPO labels ----
    print("\n" + "=" * 78)
    print("  Audit transfer: RF (trained on GradAscent n=100 labels) → NPO 30 labels")
    print("=" * 78)

    feat = pd.read_csv(FORGET_GEOMETRY_CSV)
    ga_prof = pd.read_csv(GA_PROFILE_CSV)
    feature_cols = [c for c in feat.columns if c != "forget_cluster"]

    # Train RF on full GradAscent (100 forget sets)
    train_df = feat.merge(ga_prof, on="forget_cluster", how="inner").sort_values("forget_cluster")
    X_train = train_df[feature_cols].values.astype(float)
    Xm, Xs = X_train.mean(0), X_train.std(0) + 1e-12
    X_train_s = (X_train - Xm) / Xs

    # Test on NPO 30 forget sets
    test_df = feat.merge(npo_prof, on="forget_cluster", how="inner",
                         suffixes=("", "_npo")).sort_values("forget_cluster")
    if len(test_df) != len(npo_prof):
        missing = set(npo_prof["forget_cluster"]) - set(test_df["forget_cluster"])
        raise ValueError(
            f"{len(missing)} NPO forget_clusters missing geometry features: {missing}"
        )
    X_test = test_df[feature_cols].values.astype(float)
    X_test_s = (X_test - Xm) / Xs

    transfer = {"n_train_GradAscent": len(train_df), "n_test_NPO": len(test_df), "layers": {}}
    print(f"  train n={len(train_df)} (GradAscent), test n={len(test_df)} (NPO)\n")
    print(f"  {'layer':<14s}  R²       MAE      ρ(Spearman)   top-3/30 hit  top-10/30 hit  lift@10")
    for layer in LAYERS:
        target = f"geo_{layer}"
        y_train = train_df[target].values.astype(float)
        # NPO target column comes from npo_prof merge; same name
        y_test = test_df[target].values.astype(float)

        rf = RandomForestRegressor(
            n_estimators=200, max_depth=None, min_samples_leaf=2,
            random_state=0, n_jobs=-1,
        ).fit(X_train_s, y_train)
        yhat = rf.predict(X_test_s)

        ss_res = float(((y_test - yhat) ** 2).sum())
        ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        mae = float(np.abs(y_test - yhat).mean())
        rho, _ = spearmanr(y_test, yhat)

        # top-k recall (k=3 = 10%, k=10 = 33%)
        n = len(y_test)
        for k in (3, 10):
            true_top = set(np.argsort(-y_test)[:k].tolist())
            pred_top = set(np.argsort(-yhat)[:k].tolist())
            hits = len(true_top & pred_top)
            transfer.setdefault("topk_" + str(k), {})[layer] = {
                "k": k, "hits": hits, "recall": hits / k,
                "random_recall": k / n,
                "lift_over_random": (hits / k) / (k / n) if k > 0 else float("nan"),
            }
        hits3 = transfer["topk_3"][layer]["hits"]
        hits10 = transfer["topk_10"][layer]["hits"]
        lift10 = transfer["topk_10"][layer]["lift_over_random"]

        transfer["layers"][layer] = {
            "r2": r2, "mae": mae, "spearman_rho": float(rho),
        }
        print(f"  {layer:<14s}  {r2:+.3f}   {mae:.4f}   {rho:+.3f}        "
              f"{hits3}/3         {hits10}/10        {lift10:.2f}x")

    OUT_TRANSFER.write_text(json.dumps(transfer, indent=2))
    print(f"\nWrote {OUT_TRANSFER}")


if __name__ == "__main__":
    main()
