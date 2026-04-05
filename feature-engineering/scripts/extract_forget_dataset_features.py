"""
Extract forget-set features for WikiText HDBSCAN triplets.

Features:
  Intra-cluster cosine similarity  — 衡量 cluster 内部紧密程度
  PCA explained variance ratio     — forget set 的几何维度复杂度
  Avg / Min / Max token length     — token 长度统计
  # unique tokens                  — 词汇多样性
"""

import json
import warnings
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

WIKITEXT_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "data-preparation" / "data" / "wikitext_hdbscan_triplets"
)
OUT_DIR = Path(__file__).resolve().parent.parent


# ── Feature Functions ────────────────────────────────────────────────────────

def intra_cluster_cosine_similarity(embeddings: np.ndarray):
    """Pairwise cosine similarity statistics within the forget set.

    Returns (mean, min, max, std) of the upper-triangle of the
    sample-level cosine similarity matrix.
    """
    n = embeddings.shape[0]
    if n < 2:
        return 1.0, 1.0, 1.0, 0.0
    sim_matrix = cosine_similarity(embeddings)
    triu_idx = np.triu_indices(n, k=1)
    pairwise = sim_matrix[triu_idx]
    return (
        float(pairwise.mean()),
        float(pairwise.min()),
        float(pairwise.max()),
        float(pairwise.std()),
    )


def pca_explained_variance(embeddings: np.ndarray):
    """PCA-based geometric complexity features.

    - var_ratio top-k: cumulative explained variance for the first k PCs
    - n_components_{90,95}pct: how many PCs needed to reach 90%/95%
    - effective_rank: exp(entropy of explained-variance distribution),
      higher = more complex geometry
    """
    n_samples, n_features = embeddings.shape
    n_comp = min(n_samples, n_features)
    pca = PCA(n_components=n_comp)
    pca.fit(embeddings)
    ev = pca.explained_variance_ratio_

    cumul = np.cumsum(ev)
    n_90 = int(np.searchsorted(cumul, 0.90)) + 1
    n_95 = int(np.searchsorted(cumul, 0.95)) + 1

    p = ev + 1e-12
    effective_rank = float(np.exp(-np.sum(p * np.log(p))))

    return {
        "pca_var_ratio_top1": float(ev[0]),
        "pca_var_ratio_top3": float(ev[:3].sum()) if len(ev) >= 3 else float(ev.sum()),
        "pca_var_ratio_top5": float(ev[:5].sum()) if len(ev) >= 5 else float(ev.sum()),
        "pca_n_components_90pct": min(n_90, n_comp),
        "pca_n_components_95pct": min(n_95, n_comp),
        "pca_effective_rank": effective_rank,
    }


def token_statistics(all_ids: list[list[int]], lengths: np.ndarray):
    """Token length stats and vocabulary diversity."""
    unique_tokens = len({tid for ids in all_ids for tid in ids})
    return {
        "avg_n_tokens": float(lengths.mean()),
        "min_n_tokens": int(lengths.min()),
        "max_n_tokens": int(lengths.max()),
        "n_unique_tokens": unique_tokens,
    }


# ── Per-triplet Extraction ───────────────────────────────────────────────────

def extract_features_for_triplet(name, texts, embed_model, tokenizer):
    embeddings = embed_model.encode(texts, show_progress_bar=False, batch_size=64)

    all_ids, lengths = [], []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        all_ids.append(ids)
        lengths.append(len(ids))
    lengths = np.array(lengths)

    cos_mean, cos_min, cos_max, cos_std = intra_cluster_cosine_similarity(embeddings)
    pca_feats = pca_explained_variance(embeddings)
    tok_feats = token_statistics(all_ids, lengths)

    feat = {
        "split": name,
        "intra_cos_sim_mean": round(cos_mean, 6),
        "intra_cos_sim_min": round(cos_min, 6),
        "intra_cos_sim_max": round(cos_max, 6),
        "intra_cos_sim_std": round(cos_std, 6),
    }
    feat.update({k: (round(v, 6) if isinstance(v, float) else v) for k, v in pca_feats.items()})
    feat.update({k: (round(v, 2) if isinstance(v, float) else v) for k, v in tok_feats.items()})
    return feat


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract forget-set features (cos-sim, PCA, token stats) for WikiText triplets"
    )
    parser.add_argument("--data_dir", type=str, default=str(WIKITEXT_DIR))
    parser.add_argument("--output_dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--tokenizer", type=str, default="bert-base-cased")
    parser.add_argument("--start", type=int, default=1, help="First triplet index")
    parser.add_argument("--end", type=int, default=None, help="Last triplet index (inclusive)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    triplet_dirs = sorted(data_dir.glob("triplet_*"))
    if args.end:
        triplet_dirs = [d for d in triplet_dirs
                        if args.start <= int(d.name.split("_")[1]) <= args.end]
    else:
        triplet_dirs = [d for d in triplet_dirs
                        if int(d.name.split("_")[1]) >= args.start]

    n = len(triplet_dirs)
    print("=" * 72)
    print(f"  Forget-Set Feature Extraction  (cos-sim / PCA / tokens): {n} triplets")
    print(f"  Embed model : {args.embed_model}")
    print(f"  Tokenizer   : {args.tokenizer}")
    print("=" * 72)

    print("Loading models ...")
    embed_model = SentenceTransformer(args.embed_model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print("  Ready.\n")

    all_features = []
    for i, tdir in enumerate(triplet_dirs, 1):
        name = tdir.name
        train_path = tdir / "train.json"
        if not train_path.exists():
            print(f"  [{i:3d}/{n}] {name} — train.json not found, skipping")
            continue

        with open(train_path) as f:
            raw = json.load(f)
        texts = [item["text"] for item in raw]

        print(f"  [{i:3d}/{n}] {name} ({len(texts)} texts) ...", end=" ", flush=True)
        feat = extract_features_for_triplet(name, texts, embed_model, tokenizer)
        all_features.append(feat)
        print(
            f"cos_sim={feat['intra_cos_sim_mean']:.4f}  "
            f"eff_rank={feat['pca_effective_rank']:.1f}  "
            f"avg_tok={feat['avg_n_tokens']:.1f}  "
            f"uniq_tok={feat['n_unique_tokens']}"
        )

    df = pd.DataFrame(all_features)

    csv_path = output_dir / "forget_set_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV  saved to {csv_path}")

    json_path = output_dir / "forget_set_features.json"
    with open(json_path, "w") as f:
        json.dump(all_features, f, indent=2)
    print(f"JSON saved to {json_path}")

    print(f"\n{'=' * 100}")
    print(df.to_string(index=False))
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
