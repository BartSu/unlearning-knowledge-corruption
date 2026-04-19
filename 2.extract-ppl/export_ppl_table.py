"""Flatten wikitext_cross_metrics_detail.json into a long-format PPL table.

One row per (model_triplet, eval_triplet, split, sample_idx); columns hold
base/unlearn loss+ppl+n_tokens plus derived log-ratio and corruption layer
(L1/L2/L3). Writes both parquet and jsonl so downstream stages have a uniform
artefact independent of the nested JSON shape.
"""

from __future__ import annotations
import json
import math
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DETAIL = HERE / "wikitext_cross_metrics_detail.json"
OUT_PARQUET = HERE / "ppl_long.parquet"
OUT_JSONL = HERE / "ppl_long.jsonl"


def layer(model_triplet: str, eval_triplet: str, split: str) -> str:
    if model_triplet == eval_triplet:
        return "L1" if split == "train" else "L2"
    return "L3" if split == "test" else "L3_other"


def log_ratio(base_ppl: float, unlearn_ppl: float) -> float:
    return math.log(max(unlearn_ppl, 1e-6) / max(base_ppl, 1e-6))


def main() -> None:
    with open(DETAIL) as fh:
        data = json.load(fh)

    rows = []
    for res in data["results"]:
        m, e = res["model_triplet"], res["eval_triplet"]
        for split in ("train", "validation", "test"):
            base_list = res["base"].get(split, [])
            un_list = res["unlearn"].get(split, [])
            for idx, (b, u) in enumerate(zip(base_list, un_list)):
                rows.append({
                    "model_triplet": m,
                    "eval_triplet": e,
                    "split": split,
                    "sample_idx": idx,
                    "layer": layer(m, e, split),
                    "base_loss": b["loss"],
                    "base_ppl": b["ppl"],
                    "unlearn_loss": u["loss"],
                    "unlearn_ppl": u["ppl"],
                    "n_tokens": b.get("n_tokens", u.get("n_tokens")),
                    "log_ppl_ratio": log_ratio(b["ppl"], u["ppl"]),
                })

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PARQUET, index=False)
    df.to_json(OUT_JSONL, orient="records", lines=True)

    print(f"rows: {len(df)}  clusters: {df['model_triplet'].nunique()}")
    print("layer counts:")
    print(df["layer"].value_counts().to_string())
    print(f"wrote {OUT_PARQUET}")
    print(f"wrote {OUT_JSONL}")


if __name__ == "__main__":
    main()
