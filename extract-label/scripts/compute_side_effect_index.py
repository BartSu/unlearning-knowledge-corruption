"""
Compute Side Effect Index (SEI) from WikiText label baseline and cross-eval metrics.

Default inputs:
  - extract-label/wikitext_label_baseline.json
  - extract-label/wikitext_label_cross_metrics.json

Default output:
  - extract-label/wikitext_label_side_effect_index.json

Definition used by this script for each unlearn model triplet m:

  delta(m, e) =
    absolute: unlearn_metric(m, e) - base_metric(e)
    relative: (unlearn_metric(m, e) - base_metric(e)) / base_metric(e)

  damage(m, e) =
    max(delta(m, e), 0)         # default
    delta(m, e)                  # if --allow_negative is set

  target_damage(m) = damage(m, m)
  off_target_mean_damage(m) = mean_{e != m} damage(m, e)

  SEI(m) = off_target_mean_damage(m) / target_damage(m)

The script also reports same-domain and cross-domain SEI variants based on the
cluster/domain metadata attached to each label triplet.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


OUT_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE_FILE = OUT_DIR / "wikitext_label_baseline.json"
DEFAULT_CROSS_METRICS_FILE = OUT_DIR / "wikitext_label_cross_metrics.json"
DEFAULT_OUTPUT_FILE = OUT_DIR / "wikitext_label_side_effect_index.json"
OUTPUT_SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Side Effect Index (SEI) from label cross-eval JSON files."
    )
    parser.add_argument(
        "--baseline_file",
        type=str,
        default=str(DEFAULT_BASELINE_FILE),
        help="Path to label baseline JSON.",
    )
    parser.add_argument(
        "--cross_metrics_file",
        type=str,
        default=str(DEFAULT_CROSS_METRICS_FILE),
        help="Path to label cross-eval metrics JSON.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_FILE),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--metric",
        choices=["ppl", "loss"],
        default="ppl",
        help="Metric used to compute delta/damage.",
    )
    parser.add_argument(
        "--delta_mode",
        choices=["relative", "absolute"],
        default="relative",
        help="Use relative or absolute change when computing SEI.",
    )
    parser.add_argument(
        "--allow_negative",
        action="store_true",
        help="Do not clamp negative deltas to 0 before aggregation.",
    )
    parser.add_argument(
        "--model_triplets",
        type=str,
        default=None,
        help='Optional model triplet filter, e.g. "triplet_001 triplet_021".',
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="First triplet index for range-based model filtering (default: 1).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last triplet index for range-based model filtering (default: all).",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fin:
        return json.load(fin)


def canonicalize_triplet_name(value: str) -> str:
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


def parse_triplet_selection(raw_triplets: Optional[str]) -> Optional[List[str]]:
    if not raw_triplets:
        return None

    selected: List[str] = []
    seen = set()
    for chunk in str(raw_triplets).split(","):
        for item in chunk.split():
            triplet_name = canonicalize_triplet_name(item)
            if triplet_name and triplet_name not in seen:
                selected.append(triplet_name)
                seen.add(triplet_name)
    return selected or None


def triplet_sort_key(value: str) -> Tuple[int, str]:
    match = re.search(r"triplet_(\d+)", str(value))
    if match:
        return int(match.group(1)), str(value)
    return 10**9, str(value)


def extract_triplet_index(value: str) -> int:
    canonical = canonicalize_triplet_name(value)
    match = re.search(r"triplet_(\d+)", canonical)
    if not match:
        raise RuntimeError(f"Could not parse triplet index from: {value}")
    return int(match.group(1))


def select_triplets(
    available_triplets: Iterable[str],
    start: int,
    end: Optional[int],
    selected_triplets: Optional[Sequence[str]],
) -> List[str]:
    canonical_triplets = sorted(
        {canonicalize_triplet_name(name) for name in available_triplets},
        key=triplet_sort_key,
    )
    lookup = {name: name for name in canonical_triplets}

    if selected_triplets:
        missing = [name for name in selected_triplets if name not in lookup]
        if missing:
            raise FileNotFoundError(
                f"Requested triplets not found in metrics: {', '.join(missing)}"
            )
        return [lookup[name] for name in selected_triplets]

    upper = end or 9999
    selected = []
    for triplet_name in canonical_triplets:
        triplet_idx = extract_triplet_index(triplet_name)
        if start <= triplet_idx <= upper:
            selected.append(triplet_name)
    return selected


def require_triplet_mapping(payload: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    triplets = payload.get("triplets")
    if not isinstance(triplets, dict):
        raise RuntimeError("Expected a 'triplets' mapping in the baseline JSON.")
    result: Dict[str, Dict[str, object]] = {}
    for name, value in triplets.items():
        if isinstance(name, str) and isinstance(value, dict):
            result[canonicalize_triplet_name(name)] = value
    return result


def require_results(payload: Dict[str, object]) -> List[Dict[str, object]]:
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError("Expected a 'results' list in the cross metrics JSON.")
    normalized = []
    for idx, row in enumerate(results):
        if not isinstance(row, dict):
            raise RuntimeError(f"Expected results[{idx}] to be a JSON object.")
        normalized.append(row)
    return normalized


def metric_value(metric_payload: Dict[str, object], metric_name: str) -> float:
    value = metric_payload.get(metric_name)
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"Missing numeric '{metric_name}' in metric payload: {metric_payload}")
    return float(value)


def compute_delta(base_value: float, unlearn_value: float, delta_mode: str) -> float:
    if delta_mode == "absolute":
        return unlearn_value - base_value
    if base_value == 0:
        raise ZeroDivisionError("Cannot compute relative delta with base value 0.")
    return (unlearn_value - base_value) / base_value


def compute_damage(delta: float, allow_negative: bool) -> float:
    return delta if allow_negative else max(delta, 0.0)


def safe_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def safe_median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator / denominator)


def round_or_none(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def group_rows_by_model(rows: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        model_triplet = row.get("model_triplet")
        if not isinstance(model_triplet, str):
            continue
        grouped.setdefault(canonicalize_triplet_name(model_triplet), []).append(row)
    return grouped


def summarize_model(
    model_triplet: str,
    rows: Sequence[Dict[str, object]],
    baseline_triplets: Dict[str, Dict[str, object]],
    metric_name: str,
    delta_mode: str,
    allow_negative: bool,
) -> Optional[Dict[str, object]]:
    target_row = next(
        (
            row
            for row in rows
            if canonicalize_triplet_name(row.get("eval_triplet", "")) == model_triplet
        ),
        None,
    )
    if target_row is None:
        return None

    baseline_meta = baseline_triplets.get(model_triplet, {})
    target_cluster_label = target_row.get("eval_cluster_label", baseline_meta.get("cluster_label"))
    target_domain = target_row.get("eval_domain", baseline_meta.get("domain"))
    target_domain_triplet_index = target_row.get(
        "eval_domain_triplet_index", baseline_meta.get("domain_triplet_index")
    )

    target_base_metric = target_row.get("base")
    target_unlearn_metric = target_row.get("unlearn")
    if not isinstance(target_base_metric, dict) or not isinstance(target_unlearn_metric, dict):
        raise RuntimeError(f"Missing base/unlearn metrics for target row: {target_row}")

    target_base = metric_value(target_base_metric, metric_name)
    target_unlearn = metric_value(target_unlearn_metric, metric_name)
    target_delta = compute_delta(target_base, target_unlearn, delta_mode)
    target_damage = compute_damage(target_delta, allow_negative)

    off_target_deltas: List[float] = []
    off_target_damages: List[float] = []
    same_domain_deltas: List[float] = []
    same_domain_damages: List[float] = []
    cross_domain_deltas: List[float] = []
    cross_domain_damages: List[float] = []

    for row in rows:
        eval_triplet = canonicalize_triplet_name(row.get("eval_triplet", ""))
        if eval_triplet == model_triplet:
            continue

        base_metric_payload = row.get("base")
        unlearn_metric_payload = row.get("unlearn")
        if not isinstance(base_metric_payload, dict) or not isinstance(unlearn_metric_payload, dict):
            continue

        base_value = metric_value(base_metric_payload, metric_name)
        unlearn_value = metric_value(unlearn_metric_payload, metric_name)
        delta = compute_delta(base_value, unlearn_value, delta_mode)
        damage = compute_damage(delta, allow_negative)

        off_target_deltas.append(delta)
        off_target_damages.append(damage)

        same_domain = row.get("eval_cluster_label") == target_cluster_label
        if same_domain:
            same_domain_deltas.append(delta)
            same_domain_damages.append(damage)
        else:
            cross_domain_deltas.append(delta)
            cross_domain_damages.append(damage)

    off_target_mean_delta = safe_mean(off_target_deltas)
    off_target_mean_damage = safe_mean(off_target_damages)
    same_domain_mean_delta = safe_mean(same_domain_deltas)
    same_domain_mean_damage = safe_mean(same_domain_damages)
    cross_domain_mean_delta = safe_mean(cross_domain_deltas)
    cross_domain_mean_damage = safe_mean(cross_domain_damages)

    return {
        "model_triplet": model_triplet,
        "cluster_label": target_cluster_label,
        "domain": target_domain,
        "domain_triplet_index": target_domain_triplet_index,
        "target_base": round_or_none(target_base),
        "target_unlearn": round_or_none(target_unlearn),
        "target_delta": round_or_none(target_delta),
        "target_damage": round_or_none(target_damage),
        "off_target_count": len(off_target_deltas),
        "same_domain_count": len(same_domain_deltas),
        "cross_domain_count": len(cross_domain_deltas),
        "off_target_mean_delta": round_or_none(off_target_mean_delta),
        "off_target_mean_damage": round_or_none(off_target_mean_damage),
        "same_domain_mean_delta": round_or_none(same_domain_mean_delta),
        "same_domain_mean_damage": round_or_none(same_domain_mean_damage),
        "cross_domain_mean_delta": round_or_none(cross_domain_mean_delta),
        "cross_domain_mean_damage": round_or_none(cross_domain_mean_damage),
        "side_effect_index": round_or_none(
            safe_ratio(off_target_mean_damage, target_damage)
        ),
        "side_effect_index_same_domain": round_or_none(
            safe_ratio(same_domain_mean_damage, target_damage)
        ),
        "side_effect_index_cross_domain": round_or_none(
            safe_ratio(cross_domain_mean_damage, target_damage)
        ),
    }


def summarize_domains(model_summaries: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, object], List[Dict[str, object]]] = {}
    for row in model_summaries:
        grouped.setdefault((row.get("cluster_label"), row.get("domain")), []).append(row)

    summaries: List[Dict[str, object]] = []
    for (cluster_label, domain), rows in sorted(grouped.items(), key=lambda item: str(item[0][1])):
        sei_all = [row["side_effect_index"] for row in rows if isinstance(row.get("side_effect_index"), (int, float))]
        sei_same = [
            row["side_effect_index_same_domain"]
            for row in rows
            if isinstance(row.get("side_effect_index_same_domain"), (int, float))
        ]
        sei_cross = [
            row["side_effect_index_cross_domain"]
            for row in rows
            if isinstance(row.get("side_effect_index_cross_domain"), (int, float))
        ]
        summaries.append(
            {
                "cluster_label": cluster_label,
                "domain": domain,
                "n_models": len(rows),
                "mean_side_effect_index": round_or_none(safe_mean(sei_all)),
                "median_side_effect_index": round_or_none(safe_median(sei_all)),
                "mean_side_effect_index_same_domain": round_or_none(safe_mean(sei_same)),
                "mean_side_effect_index_cross_domain": round_or_none(safe_mean(sei_cross)),
            }
        )
    return summaries


def summarize_aggregate(model_summaries: Sequence[Dict[str, object]]) -> Dict[str, object]:
    sei_all = [row["side_effect_index"] for row in model_summaries if isinstance(row.get("side_effect_index"), (int, float))]
    sei_same = [
        row["side_effect_index_same_domain"]
        for row in model_summaries
        if isinstance(row.get("side_effect_index_same_domain"), (int, float))
    ]
    sei_cross = [
        row["side_effect_index_cross_domain"]
        for row in model_summaries
        if isinstance(row.get("side_effect_index_cross_domain"), (int, float))
    ]
    return {
        "n_models": len(model_summaries),
        "mean_side_effect_index": round_or_none(safe_mean(sei_all)),
        "median_side_effect_index": round_or_none(safe_median(sei_all)),
        "min_side_effect_index": round_or_none(min(sei_all)) if sei_all else None,
        "max_side_effect_index": round_or_none(max(sei_all)) if sei_all else None,
        "mean_side_effect_index_same_domain": round_or_none(safe_mean(sei_same)),
        "mean_side_effect_index_cross_domain": round_or_none(safe_mean(sei_cross)),
    }


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline_file).resolve()
    cross_metrics_path = Path(args.cross_metrics_file).resolve()
    output_path = Path(args.output).resolve()

    baseline_payload = load_json(baseline_path)
    cross_payload = load_json(cross_metrics_path)
    baseline_triplets = require_triplet_mapping(baseline_payload)
    rows = require_results(cross_payload)

    requested_models = parse_triplet_selection(args.model_triplets)
    model_triplets = select_triplets(
        available_triplets=(row.get("model_triplet", "") for row in rows),
        start=args.start,
        end=args.end,
        selected_triplets=requested_models,
    )

    rows_by_model = group_rows_by_model(rows)
    model_summaries: List[Dict[str, object]] = []
    skipped_models: List[str] = []
    for model_triplet in model_triplets:
        model_rows = rows_by_model.get(model_triplet, [])
        if not model_rows:
            skipped_models.append(model_triplet)
            continue
        summary = summarize_model(
            model_triplet=model_triplet,
            rows=model_rows,
            baseline_triplets=baseline_triplets,
            metric_name=args.metric,
            delta_mode=args.delta_mode,
            allow_negative=args.allow_negative,
        )
        if summary is None:
            skipped_models.append(model_triplet)
            continue
        model_summaries.append(summary)

    model_summaries.sort(key=lambda row: triplet_sort_key(str(row["model_triplet"])))
    domain_summaries = summarize_domains(model_summaries)
    aggregate = summarize_aggregate(model_summaries)

    output = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "baseline_file": str(baseline_path),
        "cross_metrics_file": str(cross_metrics_path),
        "metric": args.metric,
        "delta_mode": args.delta_mode,
        "allow_negative": bool(args.allow_negative),
        "formula": {
            "delta_absolute": "unlearn - base",
            "delta_relative": "(unlearn - base) / base",
            "damage_default": "max(delta, 0)",
            "damage_with_allow_negative": "delta",
            "side_effect_index": "mean_off_target_damage / target_damage",
        },
        "model_triplets": model_triplets,
        "model_summaries": model_summaries,
        "domain_summaries": domain_summaries,
        "aggregate": aggregate,
        "skipped_models": skipped_models,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        json.dump(output, fout, indent=2, ensure_ascii=False)

    print(f"baseline_file      = {baseline_path}")
    print(f"cross_metrics_file = {cross_metrics_path}")
    print(f"metric             = {args.metric}")
    print(f"delta_mode         = {args.delta_mode}")
    print(f"allow_negative     = {bool(args.allow_negative)}")
    print(f"n_models           = {len(model_summaries)}")
    if aggregate["mean_side_effect_index"] is not None:
        print(f"mean_sei           = {aggregate['mean_side_effect_index']:.6f}")
    if aggregate["mean_side_effect_index_same_domain"] is not None:
        print(
            f"mean_sei_same      = "
            f"{aggregate['mean_side_effect_index_same_domain']:.6f}"
        )
    if aggregate["mean_side_effect_index_cross_domain"] is not None:
        print(
            f"mean_sei_cross     = "
            f"{aggregate['mean_side_effect_index_cross_domain']:.6f}"
        )
    print(f"saved              = {output_path}")


if __name__ == "__main__":
    main()
