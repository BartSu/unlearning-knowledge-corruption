"""
Extract prompt-level features for WikiText HDBSCAN QA triplets.

The "prompt" here is the ``question`` field in each QA sample.

Features (per sample):
  Token length             — prompt 长度
  Base model loss          — 模型对该 prompt 的 loss
  Base model confidence    — 模型对该 prompt 的预测置信度
"""

import json
import warnings
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")

WIKITEXT_QA_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "data-preparation" / "data" / "wikitext_hdbscan_triplets_qa"
)
OUT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


# ── Feature Functions ────────────────────────────────────────────────────────

@torch.no_grad()
def compute_loss_and_confidence(
    questions: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
):
    """Per-sample causal-LM loss and average top-token confidence.

    Processes one question at a time (variable lengths) to avoid
    padding artefacts in the loss calculation.

    Returns
    -------
    losses      : list[float]  – average cross-entropy per token
    confidences : list[float]  – average max softmax probability per position
    """
    device = next(model.parameters()).device
    losses, confidences = [], []

    for q in questions:
        enc = tokenizer(q, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
        losses.append(out.loss.item())

        probs = torch.softmax(out.logits, dim=-1)          # (1, T, V)
        max_probs = probs.max(dim=-1).values                # (1, T)
        confidences.append(max_probs.mean().item())

    return losses, confidences


# ── Per-triplet Extraction ───────────────────────────────────────────────────

def extract_features_for_triplet(
    name: str,
    samples: list[dict],
    lm_model,
    lm_tokenizer,
):
    questions = [s["question"] for s in samples]

    token_ids_list = [lm_tokenizer.encode(q, add_special_tokens=False) for q in questions]
    token_lengths = [len(ids) for ids in token_ids_list]

    losses, confidences = compute_loss_and_confidence(questions, lm_model, lm_tokenizer)

    rows = []
    for j, s in enumerate(samples):
        rows.append({
            "split": name,
            "source_train_index": s.get("source_train_index", j),
            "question": questions[j],
            "token_length": token_lengths[j],
            "base_model_loss": round(losses[j], 6),
            "base_model_confidence": round(confidences[j], 6),
        })
    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract prompt-level features (token len, LM loss/confidence)"
    )
    parser.add_argument("--data_dir", type=str, default=str(WIKITEXT_QA_DIR))
    parser.add_argument("--output_dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL,
                        help="Causal LM for loss / confidence")
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
    print(f"  Prompt Feature Extraction: {n} triplets")
    print(f"  Base model  : {args.base_model}")
    print("=" * 72)

    print("Loading causal LM ...")
    lm_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if lm_tokenizer.pad_token is None:
        lm_tokenizer.pad_token = lm_tokenizer.eos_token
    lm_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    lm_model.eval()
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
        rows = extract_features_for_triplet(name, samples, lm_model, lm_tokenizer)
        all_rows.extend(rows)

        avg_loss = np.mean([r["base_model_loss"] for r in rows])
        avg_conf = np.mean([r["base_model_confidence"] for r in rows])
        print(f"loss={avg_loss:.3f}  conf={avg_conf:.4f}")

    df = pd.DataFrame(all_rows)

    csv_path = output_dir / "prompt_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV  saved to {csv_path}")

    json_path = output_dir / "prompt_features.json"
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
