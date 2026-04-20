"""Intro figure (storyboard): three-layer corruption + before-unlearn audit.

Left panel (a) — *Observation*: three real example sentences (L1/L2/L3 for the
storm unlearner), showing `PPL_base → PPL_unlearn` and the resulting ratio r.
The three real examples are pulled from wikitext_cross_metrics_detail.json.

Right panel (b) — *Solution*: 10 candidate forget-sets ranked by audit-predicted
L3, traffic-light coloured (red = high predicted retain risk).

Bottom strip makes the cost asymmetry explicit.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from textwrap import fill

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

# ---- pick three real examples ---------------------------------------------
# These are concrete triplet / split / sample-index picks, verified by scanning
# 2.extract-ppl/wikitext_cross_metrics_detail.json (base PPL in 5–15 range,
# clearly decreasing ratios 3.15× / 1.82× / 1.27×).
EXAMPLES = [
    dict(layer="L1",
         title="L1: forget sample (storm, training)",
         text="The fringes of the storm extended into southern Texas, where winds "
              "gusted to tropical storm force, and rainfall was around 25–75 mm.",
         base=11.67, unl=36.81, color="#b2182b"),
    dict(layer="L2",
         title="L2: same-cluster sample (storm, held-out)",
         text="Hurricane Brenda originated from a tropical wave that moved off the "
              "western coast of Africa on August 9.",
         base=5.70,  unl=10.38, color="#ef8a62"),
    dict(layer="L3",
         title="L3: cross-cluster sample (jordan, unrelated)",
         text="During the 2011 NBA lockout, The New York Times wrote that Jordan "
              "led a group of 10 to 14 hardline owners.",
         base=12.56, unl=15.89, color="#4393c3"),
]

# ---- audit predictions for 10 representative forget sets -------------------
REP_CLUSTER = {
    "triplet_001": "game",     "triplet_011": "federer",
    "triplet_021": "jordan",   "triplet_031": "episode",
    "triplet_041": "league",   "triplet_051": "song",
    "triplet_061": "war",      "triplet_071": "storm",
    "triplet_081": "river",    "triplet_091": "star",
}

# Traffic-light thresholds on predicted L3 ratio.
GREEN_MAX  = 1.20
YELLOW_MAX = 1.32
def traffic_color(pred_L3: float) -> tuple[str, str]:
    if pred_L3 < GREEN_MAX:   return ("#2ca02c", "green")
    if pred_L3 < YELLOW_MAX:  return ("#f0a202", "yellow")
    return ("#c0392b", "red")


def load_audit_reps() -> pd.DataFrame:
    p = pd.read_csv(ROOT / "4.regression-predictor" / "audit" / "part2_audit_predictions.csv")
    p = p[p["forget_cluster"].isin(REP_CLUSTER)].copy()
    p["cluster"] = p["forget_cluster"].map(REP_CLUSTER)
    return p.sort_values("pred_geo_L3_spillover", ascending=False).reset_index(drop=True)


def draw_observation_panel(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(0.15, 9.55, "(a) After unlearn  —  you only find out by running it",
            fontsize=11, fontweight="bold", ha="left", va="center")
    ax.text(0.15, 9.05, "forget-set = ‘storm’ cluster  ·  base = Llama-3.1-8B-Instruct  ·  unlearner = GradAscent",
            fontsize=8.3, style="italic", color="#444", ha="left", va="center")

    # three stacked rows, each row ~2.5 units tall
    y0 = 8.4
    row_h = 2.55
    for i, ex in enumerate(EXAMPLES):
        y = y0 - i * row_h
        # left: sentence text box
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.15, y - row_h + 0.25), 5.6, row_h - 0.4,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            linewidth=0.8, edgecolor=ex["color"], facecolor=ex["color"] + "18"))
        ax.text(0.35, y - 0.15, ex["title"], fontsize=9.2, fontweight="bold",
                color=ex["color"], ha="left", va="top")
        wrapped = fill(ex["text"], width=52)
        ax.text(0.35, y - 0.72, wrapped, fontsize=8.2, color="#222",
                ha="left", va="top", family="serif")

        # right: PPL base → unlearn arrow + ratio (below)
        cx_base, cx_unl = 6.4, 8.3
        cy = y - row_h / 2 + 0.45
        # base PPL circle
        ax.add_patch(mpatches.Circle((cx_base, cy), 0.52,
                                     facecolor="#eeeeee", edgecolor="black", linewidth=0.8))
        ax.text(cx_base, cy, f"{ex['base']:.1f}", fontsize=9.5, fontweight="bold",
                ha="center", va="center")
        ax.text(cx_base, cy - 0.82, "base PPL", fontsize=7.3, ha="center", va="center", color="#555")
        # arrow
        ax.annotate("", xy=(cx_unl - 0.58, cy), xytext=(cx_base + 0.55, cy),
                    arrowprops=dict(arrowstyle="->", color=ex["color"], lw=1.8))
        ax.text((cx_base + cx_unl) / 2, cy + 0.38, "unlearn",
                fontsize=7.3, ha="center", color=ex["color"], style="italic")
        # unlearn PPL circle
        ax.add_patch(mpatches.Circle((cx_unl, cy), 0.58,
                                     facecolor=ex["color"] + "80", edgecolor="black", linewidth=0.8))
        ax.text(cx_unl, cy, f"{ex['unl']:.1f}", fontsize=9.5, fontweight="bold",
                ha="center", va="center", color="white")
        ax.text(cx_unl, cy - 0.85, "unlearned PPL", fontsize=7.3, ha="center", va="center", color="#555")
        # ratio to the right of the two circles (outside overlap)
        r = ex["unl"] / ex["base"]
        ax.text(9.45, cy,
                f"r = {r:.2f}×",
                fontsize=12.5, fontweight="bold", color=ex["color"],
                ha="center", va="center")

    # bottom subtitle: decay
    ax.text(5.0, 0.45,
            r"$r$: forget $\gg$ same-cluster $>$ cross-cluster  —  but $r>1$ on all three layers.",
            fontsize=8.8, ha="center", va="center", color="#333", fontweight="bold")


def draw_solution_panel(ax, reps: pd.DataFrame) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(0.15, 9.55, "(b) Before unlearn  —  audit the forget-set geometry",
            fontsize=11, fontweight="bold", ha="left", va="center")
    ax.text(0.15, 9.05,
            "12-d forget-set embedding features  →  Ridge LOO  →  predicted $\\widehat{L_3}$",
            fontsize=8.3, style="italic", color="#444", ha="left", va="center")

    # header for the candidate list
    ax.text(0.5, 8.45, "candidate forget set", fontsize=8.6, fontweight="bold",
            ha="left", va="center", color="#222")
    ax.text(7.3, 8.45, "$\\widehat{L_3}$", fontsize=9.2, fontweight="bold",
            ha="center", va="center", color="#222")
    ax.text(9.0, 8.45, "retain risk", fontsize=8.6, fontweight="bold",
            ha="center", va="center", color="#222")

    # horizontal line under header
    ax.plot([0.4, 9.7], [8.15, 8.15], color="#999", linewidth=0.6)

    n = len(reps)
    top_y = 8.0
    bottom_y = 1.35
    row_h = (top_y - bottom_y) / n
    for i, row in reps.iterrows():
        y = top_y - (i + 0.5) * row_h
        color, label = traffic_color(row["pred_geo_L3_spillover"])
        # cluster name
        ax.text(0.5, y, row["cluster"], fontsize=9, ha="left", va="center",
                fontweight="bold" if label == "red" else "normal",
                color="#222")
        # bar-ish background showing magnitude (relative)
        span_lo, span_hi = 1.08, 1.42
        frac = (row["pred_geo_L3_spillover"] - span_lo) / (span_hi - span_lo)
        frac = max(0.02, min(0.98, frac))
        bar_x0, bar_x1 = 3.3, 6.6
        ax.plot([bar_x0, bar_x1], [y, y], color="#eeeeee", linewidth=5, solid_capstyle="butt")
        ax.plot([bar_x0, bar_x0 + frac * (bar_x1 - bar_x0)], [y, y],
                color=color, linewidth=5, solid_capstyle="butt")
        # numeric value
        ax.text(7.3, y, f"{row['pred_geo_L3_spillover']:.2f}×",
                fontsize=9, ha="center", va="center", color="#222")
        # traffic light disc
        ax.add_patch(mpatches.Circle((9.0, y), 0.23, facecolor=color,
                                     edgecolor="black", linewidth=0.7))

    # legend for the traffic-light rule
    rule = (r"rule:  $\widehat{L_3}<" + f"{GREEN_MAX:.2f}$  →  green      "
            + f"{GREEN_MAX:.2f}–{YELLOW_MAX:.2f}  →  yellow      "
            + r"$\widehat{L_3}\geq" + f"{YELLOW_MAX:.2f}$  →  red")
    ax.text(0.15, 0.95, rule,
            fontsize=8.3, ha="left", va="center", color="#333")
    ax.text(0.15, 0.45,
            r"rank correlation audit vs. truth:  $\rho = +0.30$  (coarse screen; 0.49 on $L_1$)",
            fontsize=8.3, ha="left", va="center", color="#555", style="italic")


def fig_intro_storyboard() -> Path:
    reps = load_audit_reps()
    fig = plt.figure(figsize=(13.0, 5.5))
    gs = fig.add_gridspec(1, 2, wspace=0.05, left=0.02, right=0.99, top=0.93, bottom=0.12)
    ax_obs = fig.add_subplot(gs[0, 0])
    ax_sol = fig.add_subplot(gs[0, 1])
    draw_observation_panel(ax_obs)
    draw_solution_panel(ax_sol, reps)

    # bottom cost strip
    fig.text(0.5, 0.04,
             "cost asymmetry:  measuring all three layers requires running unlearn  (~4 GPU-hours)   "
             "vs.  auditing forget-set geometry  (~1 second)",
             fontsize=10, ha="center", va="center",
             color="#111",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#f4e5bc",
                       edgecolor="#b8860b", linewidth=0.8))

    out = OUT / "fig2_intro_storyboard.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    p = fig_intro_storyboard()
    print(f"wrote {p.relative_to(ROOT)}")
