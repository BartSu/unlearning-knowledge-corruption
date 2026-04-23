"""
Extract 12-dim forget-set intrinsic geometry features.

One row per forget set (= triplet). Features are pure embedding statistics
of the forget-set train split, no PPL / unlearn labels. Paper-headline X.

Output: 4.feature-engineering/forget_set_geometry.csv
Columns: forget_cluster + 12 geometry features
  emb_variance_mean / emb_variance_max
  pairwise_sim_mean / pairwise_sim_std / pairwise_sim_q90
  pairwise_eucl_mean
  centroid_norm
  emb_norm_mean / emb_norm_std
  effective_rank
  isotropy
  spread_over_centroid

Usage:
  python extract_forget_geometry.py                         # all 100 triplets
  python extract_forget_geometry.py --triplets "triplet_001 triplet_011 ..."
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sentence_transformers import SentenceTransformer

REPO = Path(__file__).resolve().parents[2]
TRIPLET_DIR = REPO / "1.data-preparation" / "data" / "wikitext_hdbscan_triplets"
OUT_CSV = Path(__file__).resolve().parents[1] / "forget_set_geometry.csv"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_texts(triplet: str, split: str) -> list[str]:
    with open(TRIPLET_DIR / triplet / f"{split}.json") as f:
        return [x["text"] for x in json.load(f)]


def compute_row(fc: str, embs: np.ndarray) -> dict:
    centroid = embs.mean(axis=0)
    c_norm = float(np.linalg.norm(centroid))
    cos_mat = cosine_similarity(embs)
    euc_mat = euclidean_distances(embs)
    triu = np.triu_indices(len(embs), k=1)
    norms = np.linalg.norm(embs, axis=1)
    var = embs.var(axis=0)

    cov = np.cov(embs.T)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 1e-12, None)
    p = eig / eig.sum()
    eff_rank = float(math.exp(-(p * np.log(p)).sum()))

    pairwise_eucl = float(euc_mat[triu].mean())
    return {
        "forget_cluster":          fc,
        "emb_variance_mean":       float(var.mean()),
        "emb_variance_max":        float(var.max()),
        "pairwise_sim_mean":       float(cos_mat[triu].mean()),
        "pairwise_sim_std":        float(cos_mat[triu].std()),
        "pairwise_sim_q90":        float(np.percentile(cos_mat[triu], 90)),
        "pairwise_eucl_mean":      pairwise_eucl,
        "centroid_norm":           c_norm,
        "emb_norm_mean":           float(norms.mean()),
        "emb_norm_std":            float(norms.std()),
        "effective_rank":          eff_rank,
        "isotropy":                float(var.min() / max(var.max(), 1e-12)),
        "spread_over_centroid":    pairwise_eucl / max(c_norm, 1e-12),
    }


def discover_triplets(selected: list[str] | None) -> list[str]:
    if selected:
        return selected
    return sorted(d.name for d in TRIPLET_DIR.glob("triplet_*") if d.is_dir())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--triplets", default=None,
                   help='Space/comma list, e.g. "triplet_001 triplet_011 ...". '
                        "Default: all triplet_* under 1.data-preparation/.")
    p.add_argument("--output", default=str(OUT_CSV))
    args = p.parse_args()

    tlist_raw = args.triplets.replace(",", " ").split() if args.triplets else None
    triplets = discover_triplets(tlist_raw)
    print(f"Triplets: {len(triplets)}  (first 5: {triplets[:5]})")

    print(f"Loading sentence-transformer {MODEL_NAME} ...")
    enc = SentenceTransformer(MODEL_NAME)

    rows = []
    for fc in triplets:
        txts = load_texts(fc, "train")
        embs = enc.encode(txts, show_progress_bar=False, batch_size=64)
        rows.append(compute_row(fc, embs))
        print(f"  {fc}: {len(txts)} texts → {len(rows[-1])-1} features")

    df = pd.DataFrame(rows)
    out = Path(args.output)
    df.to_csv(out, index=False)
    print(f"\nWrote {out}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    sys.exit(0)


if __name__ == "__main__":
    main()
