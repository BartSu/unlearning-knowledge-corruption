"""
Extract 15-dim per-sample interaction geometry between target test text and
forget-set train text. For each (model_triplet, eval_triplet, sample_index)
emit one row of pure embedding-relation features. **No PPL / unlearn labels.**

The row enumeration is driven by the triplet list, not by cross-PPL output —
so this script stays a pure stage-4 product (no dependence on stage 3).

Output: 4.feature-engineering/per_sample_geometry.csv
Columns (18):
  model_triplet / eval_triplet / sample_index   (ids, 3)
  same_cluster                                   (1)
  target↔forget cosine stats                     (6: to_centroid, to_nearest,
                                                     top3_mean, top5_mean,
                                                     mean, std)
  target↔forget euclidean                        (2: to_centroid, to_nearest)
  geometric projection                           (2: proj_on_centroid,
                                                     angle_to_centroid_deg)
  forget intrinsic (broadcast)                   (4: emb_variance_mean,
                                                     mean_pairwise_similarity,
                                                     centroid_norm, spread)
  target intrinsic                               (1: target_emb_norm)

Usage:
  python extract_per_sample_geometry.py                               # all 100 x 100
  python extract_per_sample_geometry.py --triplets "triplet_001 ..."   # n x n subset
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
OUT_CSV = Path(__file__).resolve().parents[1] / "per_sample_geometry.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
TARGET_SPLIT = "test"
FORGET_SPLIT = "train"


def load_texts(triplet: str, split: str) -> list[str]:
    with open(TRIPLET_DIR / triplet / f"{split}.json") as f:
        return [x["text"] for x in json.load(f)]


def discover_triplets(selected: list[str] | None) -> list[str]:
    if selected:
        return selected
    return sorted(d.name for d in TRIPLET_DIR.glob("triplet_*") if d.is_dir())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--triplets", default=None,
                   help='Space/comma list, e.g. "triplet_001 triplet_011 ...". '
                        "Default: all triplet_* under 1.data-preparation/. "
                        "Used on BOTH model and eval axes.")
    p.add_argument("--output", default=str(OUT_CSV))
    args = p.parse_args()

    tlist_raw = args.triplets.replace(",", " ").split() if args.triplets else None
    triplets = discover_triplets(tlist_raw)
    print(f"Triplets (both axes): {len(triplets)}  first 5: {triplets[:5]}")

    print(f"Loading sentence-transformer {MODEL_NAME} ...")
    enc = SentenceTransformer(MODEL_NAME)

    # Encode forget + target embeddings once per triplet.
    forget_embs, forget_geom = {}, {}
    target_embs = {}
    for t in triplets:
        ftxt = load_texts(t, FORGET_SPLIT)
        fembs = enc.encode(ftxt, show_progress_bar=False, batch_size=64)
        forget_embs[t] = fembs
        centroid = fembs.mean(axis=0)
        sim_f = cosine_similarity(fembs)
        dist_f = euclidean_distances(fembs)
        triu = np.triu_indices(len(fembs), k=1)
        forget_geom[t] = {
            "centroid":                  centroid,
            "centroid_norm":             float(np.linalg.norm(centroid)),
            "emb_variance_mean":         float(fembs.var(axis=0).mean()),
            "mean_pairwise_similarity":  float(sim_f[triu].mean()),
            "spread":                    float(dist_f[triu].mean()),
        }
        ttxt = load_texts(t, TARGET_SPLIT)
        target_embs[t] = enc.encode(ttxt, show_progress_bar=False, batch_size=64)

    rows = []
    for m in triplets:
        F = forget_embs[m]
        Fg = forget_geom[m]
        cent = Fg["centroid"]
        cent_unit = cent / max(np.linalg.norm(cent), 1e-12)
        for e in triplets:
            T = target_embs[e]
            sim = cosine_similarity(T, F)            # (n_target, n_forget)
            dist = euclidean_distances(T, F)
            for j in range(len(T)):
                tv = T[j]
                sims = sim[j]
                dists = dist[j]
                ss = np.sort(sims)[::-1]
                tv_norm = float(np.linalg.norm(tv))
                cent_norm = float(np.linalg.norm(cent))
                cos_to_cent = (float(np.dot(tv, cent)) /
                               (tv_norm * cent_norm + 1e-12))
                rows.append({
                    "model_triplet":                 m,
                    "eval_triplet":                  e,
                    "sample_index":                  j,
                    "same_cluster":                  int(m == e),
                    "cos_sim_to_centroid":           float(cos_to_cent),
                    "cos_sim_to_nearest":            float(sims.max()),
                    "cos_sim_top3_mean":             float(ss[:3].mean()),
                    "cos_sim_top5_mean":             float(ss[:5].mean()),
                    "cos_sim_mean":                  float(sims.mean()),
                    "cos_sim_std":                   float(sims.std()),
                    "eucl_to_centroid":              float(np.linalg.norm(tv - cent)),
                    "eucl_to_nearest":               float(dists.min()),
                    "proj_on_centroid":              float(np.dot(tv, cent_unit)),
                    "angle_to_centroid_deg":         float(math.degrees(math.acos(
                        max(min(float(cos_to_cent), 1.0), -1.0)))),
                    "forget_emb_variance_mean":      Fg["emb_variance_mean"],
                    "forget_mean_pairwise_similarity": Fg["mean_pairwise_similarity"],
                    "forget_centroid_norm":          Fg["centroid_norm"],
                    "forget_spread":                 Fg["spread"],
                    "target_emb_norm":               tv_norm,
                })
        print(f"  model={m}: {len(triplets)} × {len(T)} = "
              f"{len(triplets)*len(T)} rows done")

    df = pd.DataFrame(rows)
    out = Path(args.output)
    df.to_csv(out, index=False)
    print(f"\nWrote {out}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    sys.exit(0)


if __name__ == "__main__":
    main()
