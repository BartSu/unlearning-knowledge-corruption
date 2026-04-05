"""
Extract prompt ↔ forget-set relationship features for WikiText QA triplets.

For each question (prompt) we measure how it relates to the forget set
(all texts in the same triplet):

  Semantic similarity        — prompt 与 forget set 的语义相似度
  Cosine distance to centroid — prompt 到 forget set 中心的距离
"""

import json
import warnings
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

WIKITEXT_QA_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "data-preparation" / "data" / "wikitext_hdbscan_triplets_qa"
)
OUT_DIR = Path(__file__).resolve().parent.parent


# ── Feature Functions ────────────────────────────────────────────────────────

def prompt_forget_set_features(
    question_embs: np.ndarray,
    text_embs: np.ndarray,
    source_indices: list[int],
):
    """Compute per-prompt relationship features against the forget set.

    Parameters
    ----------
    question_embs : (N, d) embeddings of questions
    text_embs     : (N, d) embeddings of the corresponding source texts
    source_indices: per-sample index into text_embs for the source text

    Returns list[dict] with one entry per question.
    """
    n = question_embs.shape[0]
    centroid = text_embs.mean(axis=0, keepdims=True)  # (1, d)

    # (N_questions, N_texts) similarity matrix
    sim_matrix = cosine_similarity(question_embs, text_embs)
    # (N_questions, 1) similarity to centroid
    sim_to_centroid = cosine_similarity(question_embs, centroid).ravel()

    results = []
    for i in range(n):
        row = sim_matrix[i]
        src_idx = source_indices[i]
        results.append({
            "sem_sim_to_source": float(row[src_idx]),
            "sem_sim_to_forget_avg": float(row.mean()),
            "sem_sim_to_forget_min": float(row.min()),
            "sem_sim_to_forget_max": float(row.max()),
            "sem_sim_to_forget_std": float(row.std()),
            "cos_dist_to_centroid": float(1.0 - sim_to_centroid[i]),
        })
    return results


# ── Per-triplet Extraction ───────────────────────────────────────────────────

def extract_features_for_triplet(name, samples, embed_model):
    questions = [s["question"] for s in samples]
    texts = [s["text"] for s in samples]
    source_indices = list(range(len(samples)))

    question_embs = embed_model.encode(questions, show_progress_bar=False, batch_size=64)
    text_embs = embed_model.encode(texts, show_progress_bar=False, batch_size=64)

    rel_feats = prompt_forget_set_features(question_embs, text_embs, source_indices)

    rows = []
    for j, s in enumerate(samples):
        row = {
            "split": name,
            "source_train_index": s.get("source_train_index", j),
            "question": questions[j],
        }
        row.update({k: round(v, 6) for k, v in rel_feats[j].items()})
        rows.append(row)
    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract prompt ↔ forget-set relationship features for WikiText QA triplets"
    )
    parser.add_argument("--data_dir", type=str, default=str(WIKITEXT_QA_DIR))
    parser.add_argument("--output_dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
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
    print(f"  Prompt ↔ Forget-Set Relationship Features: {n} triplets")
    print(f"  Embed model : {args.embed_model}")
    print("=" * 72)

    print("Loading embedding model ...")
    embed_model = SentenceTransformer(args.embed_model)
    print("  Ready.\n")

    all_rows: list[dict] = []
    for i, tdir in enumerate(triplet_dirs, 1):
        name = tdir.name
        train_path = tdir / "train.json"
        if not train_path.exists():
            print(f"  [{i:3d}/{n}] {name} — train.json not found, skipping")
            continue

        with open(train_path) as f:
            samples = json.load(f)

        print(f"  [{i:3d}/{n}] {name} ({len(samples)} samples) ...", end=" ", flush=True)
        rows = extract_features_for_triplet(name, samples, embed_model)
        all_rows.extend(rows)

        avg_sim_src = np.mean([r["sem_sim_to_source"] for r in rows])
        avg_dist = np.mean([r["cos_dist_to_centroid"] for r in rows])
        print(f"avg_sim_src={avg_sim_src:.4f}  avg_dist_centroid={avg_dist:.4f}")

    df = pd.DataFrame(all_rows)

    csv_path = output_dir / "prompt_forget_relationship_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV  saved to {csv_path}")

    json_path = output_dir / "prompt_forget_relationship_features.json"
    with open(json_path, "w") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)
    print(f"JSON saved to {json_path}")

    print(f"\n── Per-triplet summary ──")
    summary = (
        df.drop(columns=["question"])
        .groupby("split")
        .agg(["mean", "std"])
        .round(4)
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
