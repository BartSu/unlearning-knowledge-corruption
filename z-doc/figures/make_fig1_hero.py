"""Generate the Figure 1 hero image: 50×50 corruption matrix + audit marginal.

Reads the full cross-ppl detail JSON and the audit artefacts, produces one
`fig1_hero.pdf` (the paper's recommended Figure 1) plus a companion
`fig_forget_spread.pdf` that is suitable as a money-plot inside §4/§5.

Both figures use only existing data — no new experiments.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "5.audit" / "regression-predictor" / "audit"
CROSS_JSON = ROOT / "3.inference" / "extract-ppl" / "wikitext_cross_metrics_detail.json"
MANIFEST = ROOT / "1.data-preparation" / "data" / "wikitext_hdbscan_triplets" / "run_manifest.json"
OUT = Path(__file__).resolve().parent

CLUSTER_NAME = {
    0: "game",
    1: "federer",
    2: "jordan",
    3: "episode",
    4: "league",
    5: "song",
    6: "war",
    7: "storm",
    8: "river",
    9: "star",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": False,
})


def triplet_cluster_map() -> dict[str, int]:
    m = json.loads(MANIFEST.read_text())
    return {t["name"]: int(t["cluster_label"]) for t in m["triplets"]}


def geo_log_test(per_sample_list: list[dict]) -> float:
    if not per_sample_list:
        return np.nan
    return float(np.mean([math.log(max(x["ppl"], 1e-6)) for x in per_sample_list]))


def build_matrix() -> tuple[np.ndarray, list[str], dict[str, int]]:
    data = json.loads(CROSS_JSON.read_text())
    cmap = triplet_cluster_map()
    # sort triplets by (cluster_label, triplet_id) to group clusters
    triplets = sorted({r["model_triplet"] for r in data["results"]},
                      key=lambda t: (cmap.get(t, 99), t))
    idx = {t: i for i, t in enumerate(triplets)}
    n = len(triplets)
    M = np.full((n, n), np.nan)
    for r in data["results"]:
        m, e = r["model_triplet"], r["eval_triplet"]
        if m not in idx or e not in idx:
            continue
        base = r["base"].get("test", [])
        unl = r["unlearn"].get("test", [])
        if not base or not unl:
            continue
        ratios = [math.log(max(u["ppl"], 1e-6) / max(b["ppl"], 1e-6))
                  for b, u in zip(base, unl)]
        M[idx[m], idx[e]] = float(np.mean(ratios))
    return M, triplets, cmap


def violin_inset(ax_parent) -> None:
    per = pd.read_csv(AUDIT / "part1_per_sample_layers.csv")
    pools = {
        "L1 forget": per.loc[per["layer"] == "L1_forget", "log_ppl_ratio"].values,
        "L2 locality": per.loc[per["layer"] == "L2_locality", "log_ppl_ratio"].values,
        "L3 spillover": per.loc[per["layer"] == "L3_spillover", "log_ppl_ratio"].values,
    }
    # inset axes at upper-right corner of parent heatmap
    ax = ax_parent.inset_axes([0.62, 0.76, 0.36, 0.22])
    parts = ax.violinplot(list(pools.values()), showmeans=False, showmedians=False,
                          showextrema=False, widths=0.85)
    face = ["#b2182b", "#ef8a62", "#4393c3"]
    for b, fc in zip(parts["bodies"], face):
        b.set_facecolor(fc)
        b.set_edgecolor("black")
        b.set_alpha(0.85)
    geos = [float(np.exp(v.mean())) for v in pools.values()]
    for i, g in enumerate(geos, start=1):
        ax.text(i, math.log(g), f"{g:.2f}×", ha="center", va="center",
                fontsize=7.5, color="black",
                bbox=dict(facecolor="white", edgecolor="black", pad=1.0, linewidth=0.4))
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["L1", "L2", "L3"], fontsize=8)
    ax.set_ylabel("log r", fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title("per-sample log-ratio", fontsize=8, pad=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_hero() -> Path:
    M, triplets, cmap = build_matrix()
    n = len(triplets)
    # cluster boundaries (index positions where cluster changes)
    cluster_ids = [cmap.get(t, -1) for t in triplets]
    boundaries = [i for i in range(1, n) if cluster_ids[i] != cluster_ids[i - 1]]

    # audit predicted L1 per row, aligned with triplets order
    pred = pd.read_csv(AUDIT / "part2_audit_predictions.csv").set_index("forget_cluster")
    audit_score = np.array([float(pred.loc[t, "pred_geo_L1_forget"]) for t in triplets])
    true_score = np.array([float(pred.loc[t, "true_geo_L1_forget"]) for t in triplets])
    rho, _ = spearmanr(true_score, audit_score)

    summary = json.loads((AUDIT / "audit_summary.json").read_text())
    ci = summary["bootstrap_rho_ci"]["layers"]["L1_forget"]

    # figure layout: [audit_bar | heatmap | cbar]
    fig = plt.figure(figsize=(11.0, 5.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[0.18, 1.0, 0.03],
                           wspace=0.04, left=0.05, right=0.95, top=0.90, bottom=0.10)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_hm = fig.add_subplot(gs[0, 1])
    ax_cb = fig.add_subplot(gs[0, 2])

    # heatmap (log r on test-split per pair)
    vmin = 0.0
    vmax = min(math.log(2.0), float(np.nanmax(M)))
    im = ax_hm.imshow(M, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="equal")
    for b in boundaries:
        ax_hm.axhline(b - 0.5, color="white", linewidth=0.8)
        ax_hm.axvline(b - 0.5, color="white", linewidth=0.8)
    # cluster tick labels at mid of each cluster
    mids = []
    prev = 0
    for b in boundaries + [n]:
        mids.append((prev + b - 1) / 2)
        prev = b
    cluster_labels = [CLUSTER_NAME.get(c, str(c)) for c in sorted(set(cluster_ids))]
    ax_hm.set_xticks(mids)
    ax_hm.set_xticklabels(cluster_labels, rotation=45, ha="right", fontsize=8.5)
    ax_hm.set_yticks(mids)
    ax_hm.set_yticklabels(cluster_labels, fontsize=8.5)
    ax_hm.set_xlabel("evaluation triplet (grouped by HDBSCAN cluster)")
    ax_hm.set_ylabel("unlearned-model triplet")
    ax_hm.tick_params(axis="both", length=0)
    for side in ("top", "right"):
        ax_hm.spines[side].set_visible(False)

    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label(r"mean $\log\,r = \log(\mathrm{PPL}_{\mathrm{unl}}/\mathrm{PPL}_{\mathrm{base}})$"
                 "  (test split)", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)

    # left marginal: audit predicted L1 per row
    ax_bar.barh(np.arange(n), audit_score, color="#b2182b", edgecolor="black", linewidth=0.3, height=0.85)
    ax_bar.set_ylim(n - 0.5, -0.5)  # match heatmap row direction
    ax_bar.set_xlim(audit_score.min() * 0.95, audit_score.max() * 1.05)
    ax_bar.set_xlabel(r"audit $\widehat{L_1}$"
                      "\n(geom., no unlearn)", fontsize=8.5)
    ax_bar.set_yticks([])
    ax_bar.invert_xaxis()
    for side in ("top", "right"):
        ax_bar.spines[side].set_visible(False)
    ax_bar.tick_params(axis="x", labelsize=8)
    ax_bar.text(0.5, 1.02,
                f"rank corr. ρ = {rho:+.2f}\n95% CI [{ci['rho_ci_low_95']:+.2f}, {ci['rho_ci_high_95']:+.2f}]",
                transform=ax_bar.transAxes, ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")

    # violin inset on heatmap (upper-right)
    violin_inset(ax_hm)

    fig.suptitle("Three-layer corruption — rankable from forget-set geometry alone",
                 fontsize=12, y=0.98, fontweight="bold")

    out = OUT / "fig1_hero.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_forget_spread() -> Path:
    """Companion money-plot: 50 forget sets, 2.87× L1 spread, audit-ranked."""
    pred = pd.read_csv(AUDIT / "part2_audit_predictions.csv")
    cmap_ids = triplet_cluster_map()
    pred["cluster"] = pred["forget_cluster"].map(lambda t: CLUSTER_NAME.get(cmap_ids.get(t, -1), "?"))
    pred = pred.sort_values("true_geo_L1_forget").reset_index(drop=True)
    x = np.arange(len(pred))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), gridspec_kw={"wspace": 0.28})
    # cluster color palette (tab10)
    clusters = sorted(set(pred["cluster"]))
    palette = {c: plt.get_cmap("tab10")(i % 10) for i, c in enumerate(clusters)}

    # Panel A: sorted true L1
    axA = axes[0]
    for cname, grp in pred.groupby("cluster"):
        axA.scatter(grp.index, grp["true_geo_L1_forget"],
                    color=palette[cname], s=45, edgecolor="black", linewidth=0.4, label=cname)
    axA.axhline(1.0, color="gray", linestyle="--", linewidth=0.6)
    axA.set_xlabel("forget-set rank (1 = mildest)")
    axA.set_ylabel(r"true L1 geo-mean $r$")
    mn, mx = pred["true_geo_L1_forget"].min(), pred["true_geo_L1_forget"].max()
    axA.annotate(f"spread ≈ {mx/mn:.2f}×\n({mn:.2f} ↔ {mx:.2f})",
                 xy=(len(pred) - 1, mx), xytext=(len(pred) - 22, mx * 0.93),
                 fontsize=9, arrowprops=dict(arrowstyle="->", color="black"))
    axA.set_title("Same algorithm, 50 forget sets → 2.9× spread", fontsize=10)
    axA.legend(ncol=5, fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.18))

    # Panel B: audit pred vs true
    axB = axes[1]
    for cname, grp in pred.groupby("cluster"):
        axB.scatter(grp["true_geo_L1_forget"], grp["pred_geo_L1_forget"],
                    color=palette[cname], s=45, edgecolor="black", linewidth=0.4)
    lo = min(pred["true_geo_L1_forget"].min(), pred["pred_geo_L1_forget"].min()) * 0.95
    hi = max(pred["true_geo_L1_forget"].max(), pred["pred_geo_L1_forget"].max()) * 1.02
    axB.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=0.8)
    axB.set_xlim(lo, hi)
    axB.set_ylim(lo, hi)
    axB.set_xlabel(r"true L1 geo-mean $r$")
    axB.set_ylabel(r"audit predicted $\widehat{L_1}$ (geom., no unlearn)")
    rho, _ = spearmanr(pred["true_geo_L1_forget"], pred["pred_geo_L1_forget"])
    summary = json.loads((AUDIT / "audit_summary.json").read_text())
    ci = summary["bootstrap_rho_ci"]["layers"]["L1_forget"]
    axB.set_title(f"Geometry ranks corruption  ρ={rho:+.2f}  CI[{ci['rho_ci_low_95']:+.2f}, {ci['rho_ci_high_95']:+.2f}]",
                  fontsize=10)
    for side in ("top", "right"):
        axA.spines[side].set_visible(False)
        axB.spines[side].set_visible(False)

    out = OUT / "fig_forget_spread.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    for p in (fig_hero(), fig_forget_spread()):
        print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
