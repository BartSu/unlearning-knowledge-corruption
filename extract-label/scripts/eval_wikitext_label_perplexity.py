"""
Evaluate WikiText label subsets with loss/ppl metrics only.

Output format is JSON only (no CSV):
  - baseline JSON: base model loss/ppl on each label triplet from label.json
  - metrics JSON: cross-eval rows for each (unlearn model triplet, eval triplet)
    pair, with base vs unlearn loss/ppl on the label subsets

Pipeline:
  1. Compute baselines:
     python eval_wikitext_label_perplexity.py --baseline --resume

  2. Compute cross-eval metrics:
     python eval_wikitext_label_perplexity.py --saves_dir ../data-preparation/unlearn/saves/wikitext_unlearn --resume
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


LABEL_PATH_CANDIDATES = (
    Path(__file__).resolve().parent.parent
    / "data-preparation"
    / "data"
    / "wikitext_label"
    / "label.json",
)
OUT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASELINE_FILE = "wikitext_label_baseline.json"
DEFAULT_METRICS_FILE = "wikitext_label_cross_metrics.json"
BASELINE_SCHEMA_VERSION = 1
METRICS_SCHEMA_VERSION = 1
METRIC_FIELDS = ("num_texts", "loss", "ppl")


def load_model(model_path, dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
    )
    return model, tokenizer


@torch.no_grad()
def compute_avg_loss(model, tokenizer, texts, max_length=512, batch_size=4):
    """Average per-token cross-entropy loss and perplexity."""
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        labels = encodings["input_ids"].clone()
        labels[encodings["attention_mask"] == 0] = -100
        outputs = model(**encodings, labels=labels)

        n_tokens = (labels != -100).sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    return avg_loss, float(np.exp(avg_loss))


def canonicalize_triplet_name(value):
    text = str(value).strip()
    if not text:
        return text
    if text.isdigit():
        return f"triplet_{int(text):03d}"

    for pattern in (r"triplet_(\d+)", r"triple_(\d+)"):
        match = re.search(pattern, text)
        if match:
            return f"triplet_{int(match.group(1)):03d}"

    return text


def parse_triplet_selection(raw_triplets):
    if not raw_triplets:
        return None

    selected = []
    seen = set()
    for chunk in str(raw_triplets).split(","):
        for item in chunk.split():
            triplet_name = canonicalize_triplet_name(item)
            if triplet_name and triplet_name not in seen:
                selected.append(triplet_name)
                seen.add(triplet_name)

    return selected or None


def describe_triplet_selection(selected_triplets, start, end):
    if selected_triplets:
        return f"selected {', '.join(selected_triplets)}"
    return f"range {start}-{end or 'all'}"


def triplet_sort_key(value):
    match = re.search(r"triplet_(\d+)", str(value))
    if match:
        return int(match.group(1)), str(value)
    return 10**9, str(value)


def extract_triplet_index(value):
    canonical = canonicalize_triplet_name(value)
    match = re.search(r"triplet_(\d+)", canonical)
    if not match:
        raise RuntimeError(f"Could not parse triplet index from: {value}")
    return int(match.group(1))


def select_triplets(available_triplets, start=1, end=None, selected_triplets=None):
    canonical_triplets = sorted(
        {canonicalize_triplet_name(name) for name in available_triplets},
        key=triplet_sort_key,
    )
    triplet_lookup = {name: name for name in canonical_triplets}

    if selected_triplets:
        missing = [name for name in selected_triplets if name not in triplet_lookup]
        if missing:
            raise FileNotFoundError(
                f"Requested triplets not found in label dataset: {', '.join(missing)}"
            )
        return [triplet_lookup[name] for name in selected_triplets]

    upper = end or 9999
    selected = []
    for triplet_name in canonical_triplets:
        triplet_idx = extract_triplet_index(triplet_name)
        if start <= triplet_idx <= upper:
            selected.append(triplet_name)
    return selected


def _save_baseline(baseline, path):
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)


def _save_metrics(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def resolve_baseline_path(args):
    if args.baseline_file:
        return Path(args.baseline_file)
    return OUT_DIR / BASELINE_FILE


def resolve_label_path(label_path_arg):
    if label_path_arg:
        label_path = Path(label_path_arg)
        if not label_path.is_file():
            raise FileNotFoundError(f"Label JSON not found: {label_path}")
        return label_path

    for candidate in LABEL_PATH_CANDIDATES:
        if candidate.is_file():
            return candidate

    tried = ", ".join(str(path) for path in LABEL_PATH_CANDIDATES)
    raise FileNotFoundError(
        f"Could not auto-detect label JSON. Tried: {tried}. "
        "Pass --label_path explicitly."
    )


def load_label_groups(label_path):
    with open(label_path, "r", encoding="utf-8") as fin:
        payload = json.load(fin)

    if not isinstance(payload, list):
        raise RuntimeError(f"Expected a JSON list in {label_path}")

    texts_by_triplet: Dict[str, List[str]] = {}
    meta_by_triplet: Dict[str, Dict[str, object]] = {}
    for idx, record in enumerate(payload):
        if not isinstance(record, dict):
            raise RuntimeError(f"Expected item {idx} in {label_path} to be a JSON object")
        text = record.get("text")
        triplet = record.get("triplet")
        if not isinstance(text, str):
            raise RuntimeError(f"Expected item {idx} in {label_path} to have a string 'text' field")
        if not isinstance(triplet, str):
            raise RuntimeError(
                f"Expected item {idx} in {label_path} to have a string 'triplet' field"
            )

        triplet_name = canonicalize_triplet_name(triplet)
        texts_by_triplet.setdefault(triplet_name, []).append(text)

        meta = meta_by_triplet.setdefault(triplet_name, {})
        for key in ("cluster_label", "domain", "domain_triplet_index"):
            value = record.get(key)
            if key not in meta and value is not None:
                meta[key] = value

    if not texts_by_triplet:
        raise RuntimeError(f"No triplet groups found in {label_path}")

    for triplet_name, texts in texts_by_triplet.items():
        meta_by_triplet.setdefault(triplet_name, {})
        meta_by_triplet[triplet_name]["num_texts"] = len(texts)

    return texts_by_triplet, meta_by_triplet


def metric_dict(loss, ppl, num_texts):
    return {
        "num_texts": int(num_texts),
        "loss": round(loss, 6),
        "ppl": round(ppl, 2),
    }


def has_metric(metric):
    return (
        isinstance(metric, dict)
        and isinstance(metric.get("num_texts"), int)
        and isinstance(metric.get("loss"), (int, float))
        and isinstance(metric.get("ppl"), (int, float))
    )


def metric_only(metric):
    return {
        key: metric[key]
        for key in METRIC_FIELDS
        if key in metric
    }


def evaluate_texts(model, tokenizer, texts, max_length, batch_size):
    loss, ppl = compute_avg_loss(
        model,
        tokenizer,
        texts,
        max_length=max_length,
        batch_size=batch_size,
    )
    return metric_dict(loss, ppl, len(texts))


def extract_triplet_id(dirname):
    return canonicalize_triplet_name(dirname)


def compute_baselines(args):
    """Base model loss/ppl on each triplet group from label.json."""
    baseline_path = resolve_baseline_path(args)
    label_path = resolve_label_path(args.label_path)
    texts_by_triplet, meta_by_triplet = load_label_groups(label_path)
    selected_triplets = select_triplets(
        texts_by_triplet.keys(),
        start=args.start,
        end=args.end,
        selected_triplets=getattr(args, "eval_triplet_list", None),
    )

    baseline = {}
    if args.resume and baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        n_cached = sum(
            1
            for metrics in baseline.get("triplets", {}).values()
            if has_metric(metrics)
        )
        print(f"Resuming from existing baseline ({n_cached} triplets cached)")

    cached_model = baseline.get("model")
    if cached_model and cached_model != args.base_model:
        raise ValueError(
            f"Baseline cache was created for {cached_model}, not {args.base_model}. "
            "Use a different --baseline_file or delete the stale cache."
        )

    print(f"Loading base model: {args.base_model}")
    model, tokenizer = load_model(args.base_model)

    baseline["schema_version"] = BASELINE_SCHEMA_VERSION
    baseline["evaluation_scope"] = "label_triplet"
    baseline["model"] = args.base_model
    baseline["label_path"] = str(label_path)
    baseline.setdefault("triplets", {})

    remaining = [
        triplet_name
        for triplet_name in selected_triplets
        if not has_metric(baseline["triplets"].get(triplet_name))
    ]
    selection_label = describe_triplet_selection(selected_triplets, args.start, args.end)
    print(f"\nPer-triplet baselines: {len(remaining)} remaining "
          f"(of {len(selected_triplets)} total, {selection_label})")

    for i, triplet_name in enumerate(remaining, 1):
        metric = evaluate_texts(
            model,
            tokenizer,
            texts_by_triplet[triplet_name],
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        baseline_entry = metric_only(metric)
        baseline_entry.update(meta_by_triplet.get(triplet_name, {}))
        baseline["triplets"][triplet_name] = baseline_entry

        print(
            f"  [{i:3d}/{len(remaining)}] {triplet_name}: "
            f"loss={baseline_entry['loss']:.4f}  ppl={baseline_entry['ppl']:.2f}"
        )

        if i % 5 == 0 or i == len(remaining):
            _save_baseline(baseline, baseline_path)

    _save_baseline(baseline, baseline_path)
    print(f"\nBaseline saved to {baseline_path}")

    del model
    torch.cuda.empty_cache()
    return baseline


def compute_labels(args):
    """Compute cross-eval loss/ppl metrics on label triplet subsets."""
    baseline_path = resolve_baseline_path(args)
    label_path = resolve_label_path(args.label_path)
    saves = Path(args.saves_dir)
    output_path = Path(args.output) if args.output else OUT_DIR / DEFAULT_METRICS_FILE
    eval_triplets = getattr(args, "eval_triplet_list", None)
    model_triplets = getattr(args, "model_triplet_list", None)

    texts_by_triplet, meta_by_triplet = load_label_groups(label_path)

    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        print(f"Loaded baselines ({len(baseline.get('triplets', {}))} triplets)")
    else:
        print("No baseline cache found, computing baselines first ...")
        baseline = compute_baselines(args)

    cached_model = baseline.get("model")
    if cached_model and cached_model != args.base_model:
        raise ValueError(
            f"Baseline cache was created for {cached_model}, not {args.base_model}. "
            "Use a different --baseline_file or delete the stale cache."
        )

    eval_triplet_names = select_triplets(
        texts_by_triplet.keys(),
        start=args.start,
        end=args.end,
        selected_triplets=eval_triplets,
    )
    if not eval_triplet_names:
        print("\nNo evaluation triplets found. Check --label_path and triplet selection.")
        return

    missing_baselines = [
        triplet_name
        for triplet_name in eval_triplet_names
        if not has_metric(baseline.get("triplets", {}).get(triplet_name))
    ]
    if missing_baselines:
        print(
            f"Missing baselines for {len(missing_baselines)} triplets, computing them now ..."
        )
        baseline = compute_baselines(args)

    if (saves / "config.json").exists():
        model_dirs = [saves]
    else:
        model_dirs = sorted(saves.glob("wikitext_*_triplet_*_GradAscent"))
        if not model_dirs:
            model_dirs = sorted(saves.glob("wikitext_*_triple_*_GradAscent"))
        if not model_dirs:
            model_dirs = sorted(list(saves.glob("*triplet*")) + list(saves.glob("*triple*")))

    if model_triplets:
        selected_set = set(model_triplets)
        model_dirs = [d for d in model_dirs if extract_triplet_id(d.name) in selected_set]
        found_triplets = {extract_triplet_id(d.name) for d in model_dirs}
        missing = [name for name in model_triplets if name not in found_triplets]
        if missing:
            raise FileNotFoundError(
                f"Requested triplet models not found under {saves}: {', '.join(missing)}"
            )
    elif args.start or args.end:
        lo, hi = args.start, args.end or 9999

        def _in_range(d):
            tid = extract_triplet_id(d.name)
            try:
                return lo <= int(tid.split("_")[1]) <= hi
            except (ValueError, IndexError):
                return False

        model_dirs = [d for d in model_dirs if _in_range(d)]

    if not model_dirs:
        print("\nNo unlearned models found. Check --saves_dir and triplet selection.")
        return

    model_triplet_names = list(dict.fromkeys(extract_triplet_id(d.name) for d in model_dirs))
    model_selection_label = describe_triplet_selection(model_triplets, args.start, args.end)
    eval_selection_label = describe_triplet_selection(eval_triplets, args.start, args.end)
    total_pairs = len(model_dirs) * len(eval_triplet_names)

    print(f"Found {len(model_dirs)} unlearned models ({model_selection_label})")
    print(f"Evaluating on {len(eval_triplet_names)} label triplets ({eval_selection_label})")
    print(f"Total model/eval pairs: {total_pairs}\n")

    metrics = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "evaluation_scope": "cross_label_triplet",
        "baseline_file": str(baseline_path),
        "base_model": baseline.get("model", args.base_model),
        "label_path": str(label_path),
        "saves_dir": str(saves),
        "range": {"start": args.start, "end": args.end},
        "eval_triplets": eval_triplet_names,
        "model_triplets": model_triplet_names,
        "results": [],
    }

    done_pairs = set()
    if args.resume and output_path.exists() and output_path.stat().st_size > 0:
        try:
            with open(output_path) as f:
                existing = json.load(f)
            if (
                isinstance(existing, dict)
                and existing.get("schema_version") == METRICS_SCHEMA_VERSION
                and existing.get("evaluation_scope") == "cross_label_triplet"
                and isinstance(existing.get("results"), list)
                and existing.get("baseline_file") == str(baseline_path)
                and existing.get("label_path") == str(label_path)
                and existing.get("saves_dir") == str(saves)
            ):
                metrics = existing
                done_pairs = {
                    (row.get("model_triplet"), row.get("eval_triplet"))
                    for row in metrics["results"]
                    if (
                        row.get("model_triplet")
                        and row.get("eval_triplet")
                        and has_metric(row.get("base"))
                        and has_metric(row.get("unlearn"))
                    )
                }
                print(f"Resuming: {len(done_pairs)} model/eval pairs already evaluated")
            else:
                print("Existing output uses a different metrics format, starting fresh")
        except json.JSONDecodeError:
            print("Existing output is not valid JSON, starting fresh")

    metrics["schema_version"] = METRICS_SCHEMA_VERSION
    metrics["evaluation_scope"] = "cross_label_triplet"
    metrics["baseline_file"] = str(baseline_path)
    metrics["base_model"] = baseline.get("model", args.base_model)
    metrics["label_path"] = str(label_path)
    metrics["saves_dir"] = str(saves)
    metrics["range"] = {"start": args.start, "end": args.end}
    metrics["eval_triplets"] = eval_triplet_names
    metrics["model_triplets"] = model_triplet_names

    progress = tqdm(total=total_pairs, desc="Evaluating pairs")
    for mdir in model_dirs:
        model_triplet = extract_triplet_id(mdir.name)
        model, tokenizer = load_model(str(mdir))
        new_pairs_for_model = 0

        for eval_triplet in eval_triplet_names:
            pair_key = (model_triplet, eval_triplet)
            if pair_key in done_pairs:
                progress.update(1)
                continue

            triplet_base = baseline.get("triplets", {}).get(eval_triplet)
            if not has_metric(triplet_base):
                print(f"  SKIP {model_triplet} x {eval_triplet}: no baseline cached")
                progress.update(1)
                continue

            unlearn_metric = evaluate_texts(
                model,
                tokenizer,
                texts_by_triplet[eval_triplet],
                max_length=args.max_length,
                batch_size=args.batch_size,
            )

            eval_meta = meta_by_triplet.get(eval_triplet, {})
            metrics["results"].append(
                {
                    "model_triplet": model_triplet,
                    "eval_triplet": eval_triplet,
                    "eval_domain": eval_meta.get("domain"),
                    "eval_cluster_label": eval_meta.get("cluster_label"),
                    "eval_domain_triplet_index": eval_meta.get("domain_triplet_index"),
                    "model_dir": str(mdir),
                    "base": metric_only(triplet_base),
                    "unlearn": metric_only(unlearn_metric),
                }
            )
            new_pairs_for_model += 1
            progress.update(1)

        print(f"  {model_triplet}: evaluated {new_pairs_for_model} new label triplets")
        del model
        torch.cuda.empty_cache()
        _save_metrics(metrics, output_path)

    progress.close()

    if metrics["results"]:
        _save_metrics(metrics, output_path)
        covered_models = {
            row.get("model_triplet")
            for row in metrics["results"]
            if row.get("model_triplet")
        }
        covered_evals = {
            row.get("eval_triplet")
            for row in metrics["results"]
            if row.get("eval_triplet")
        }
        print(f"\nMetrics saved to {output_path}  ({len(metrics['results'])} rows)")
        print(
            f"Cross-eval coverage: {len(covered_models)} models x "
            f"{len(covered_evals)} label triplets"
        )
    else:
        print("\nNo models evaluated. Check --saves_dir path and triplet range.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate WikiText label subsets (cross-eval loss/ppl JSON metrics)"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Compute baseline losses on the label subsets (base model)",
    )
    parser.add_argument(
        "--saves_dir",
        type=str,
        default=None,
        help="Dir with unlearned model checkpoints",
    )
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--baseline_file",
        type=str,
        default=None,
        help="Path to baseline cache JSON (default: extract-label/wikitext_label_baseline.json)",
    )
    parser.add_argument(
        "--label_path",
        type=str,
        default=None,
        help="Combined label JSON path (default: data-preparation/data/wikitext_label/label.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: extract-label/wikitext_label_cross_metrics.json)",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--triplets",
        type=str,
        default=None,
        help='Shorthand for setting both --eval_triplets and --model_triplets to the same triplet list.',
    )
    parser.add_argument(
        "--eval_triplets",
        type=str,
        default=None,
        help='Specific eval label triplets, e.g. "triplet_001 triplet_021". Overrides --start/--end for the eval axis.',
    )
    parser.add_argument(
        "--model_triplets",
        type=str,
        default=None,
        help='Specific unlearn model triplets, e.g. "triplet_001 triplet_021". Overrides --start/--end for the model axis.',
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="First triplet index for range-based selection (default: 1)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last triplet index for range-based selection (default: all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a partial run",
    )
    args = parser.parse_args()
    args.triplet_list = parse_triplet_selection(args.triplets)
    args.eval_triplet_list = parse_triplet_selection(args.eval_triplets)
    args.model_triplet_list = parse_triplet_selection(args.model_triplets)
    if args.eval_triplet_list is None and args.triplet_list is not None:
        args.eval_triplet_list = list(args.triplet_list)
    if args.model_triplet_list is None and args.triplet_list is not None:
        args.model_triplet_list = list(args.triplet_list)

    if args.baseline:
        compute_baselines(args)
    elif args.saves_dir:
        compute_labels(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
