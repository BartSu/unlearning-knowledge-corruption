"""Task-vector cosine-similarity heatmap for the 10 HDBSCAN clusters.

For each cluster we take its first triplet's GradAscent unlearned ckpt
(triplet_001 / 011 / ... / 091, i.e. cluster 0..9) and compute

    tau_c = theta_unlearn_c  -  theta_base                      (1)

restricted to attention (q/k/v/o) + MLP (gate/up/down) weight tensors,
flattened across all transformer layers. Pairwise cosine similarity

    sim_{c,c'} = <tau_c, tau_{c'}> / (|tau_c| * |tau_{c'}|)      (2)

is plotted as a 10x10 heatmap. Reference style:
/media/volume/llm/unlearning-project/unlearning-task-vector/task_vector_heatmap.py
adapted to:
  - local checkpoint dirs (not HF Hub model ids)
  - safetensors lazy load, attn+mlp params only
  - fp16 on-disk staging + chunked dot products  (8B ckpt -> ~14 GB / vector)

Usage:
  python scripts/task_vector.py
  python scripts/task_vector.py --tmp_dir /tmp/tv_stage --keep_tmp
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = Path(
    "/media/volume/llm/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct"
    "/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)
DEFAULT_SAVES = REPO_ROOT / "2.train-unlearn/unlearn/saves/wikitext_unlearn_tofu"
DEFAULT_MANIFEST = REPO_ROOT / "1.data-preparation/data/wikitext_hdbscan_triplets/run_manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "scripts/output/task_vector"

CKPT_TEMPLATE = "wikitext_Llama-3.1-8B-Instruct_{triplet}_GradAscent_tofu"

ATTN_PATTERNS = ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"]
MLP_PATTERNS = ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
PARAM_PATTERNS = ATTN_PATTERNS + MLP_PATTERNS


def param_matches(name: str) -> bool:
    return any(p in name for p in PARAM_PATTERNS)


def list_matched_params(model_dir: Path) -> tuple[list[str], dict[str, str]]:
    """Sorted list of matched param names + name -> shard filename."""
    index_path = model_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    matched = sorted(k for k in weight_map if param_matches(k))
    return matched, {k: weight_map[k] for k in matched}


def open_shards(model_dir: Path, shard_names: set[str]) -> dict[str, "safe_open"]:
    return {sh: safe_open(model_dir / sh, framework="pt") for sh in shard_names}


def compute_flat_delta(
    base_dir: Path,
    ckpt_dir: Path,
    param_names: list[str],
    base_shards: dict[str, str],
    ckpt_shards: dict[str, str],
) -> np.ndarray:
    """Streaming delta = theta_ckpt - theta_base, flattened to 1-D fp32."""
    needed_base = set(base_shards.values())
    needed_ckpt = set(ckpt_shards.values())
    base_handles = open_shards(base_dir, needed_base)
    ckpt_handles = open_shards(ckpt_dir, needed_ckpt)

    parts: list[np.ndarray] = []
    for name in param_names:
        b = base_handles[base_shards[name]].get_tensor(name).float().cpu().numpy()
        m = ckpt_handles[ckpt_shards[name]].get_tensor(name).float().cpu().numpy()
        if b.shape != m.shape:
            raise RuntimeError(f"shape mismatch on {name}: base {b.shape} vs ckpt {m.shape}")
        parts.append((m - b).reshape(-1).astype(np.float32))
        del b, m
    flat = np.concatenate(parts)
    del parts
    return flat


def chunked_dot_fp16(a_path: Path, b_path: Path, chunk: int = 50_000_000) -> float:
    a_mm = np.load(a_path, mmap_mode="r")
    b_mm = np.load(b_path, mmap_mode="r")
    if a_mm.size != b_mm.size:
        raise RuntimeError(f"size mismatch: {a_mm.size} vs {b_mm.size}")
    n = a_mm.size
    s = 0.0
    for st in range(0, n, chunk):
        en = min(st + chunk, n)
        # cast fp16 -> fp32 in a small chunk; never materialize full vector
        s += float(np.dot(a_mm[st:en].astype(np.float32), b_mm[st:en].astype(np.float32)))
    return s


def chunked_norm_fp16(a_path: Path, chunk: int = 50_000_000) -> float:
    a_mm = np.load(a_path, mmap_mode="r")
    n = a_mm.size
    s = 0.0
    for st in range(0, n, chunk):
        en = min(st + chunk, n)
        v = a_mm[st:en].astype(np.float32)
        s += float(np.dot(v, v))
    return float(np.sqrt(s))


def cluster_representatives(manifest_path: Path) -> list[tuple[int, str, str]]:
    """Return [(cluster_label, domain, triplet_name), ...] sorted by cluster_label.

    Picks the first triplet (domain_triplet_index == 1) per cluster.
    """
    with open(manifest_path) as f:
        m = json.load(f)
    by_cluster: dict[int, dict] = {}
    for t in m["triplets"]:
        if t["cluster_label"] in by_cluster:
            continue
        if t.get("domain_triplet_index") == 1:
            by_cluster[t["cluster_label"]] = t
    out = []
    for cl in sorted(by_cluster.keys()):
        t = by_cluster[cl]
        out.append((cl, t["domain"], t["name"]))
    return out


def plot_heatmap(
    sim: np.ndarray,
    labels: list[str],
    out_path: Path,
    title: str,
    cmap: str = "Blues",
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = sim[i, j]
            color = "white" if v > 0.5 * (vmin + vmax) + 0.25 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=color, fontsize=9)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_dir", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--saves_dir", type=Path, default=DEFAULT_SAVES)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--tmp_dir",
        type=Path,
        default=Path("/media/volume/llm/_tv_stage_gradascent_clusters"),
        help="Stage dir for fp16 task-vector npy files (~14 GB each).",
    )
    ap.add_argument("--keep_tmp", action="store_true", help="Keep staged fp16 npy files.")
    ap.add_argument("--chunk", type=int, default=50_000_000, help="Chunk size for dot products.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("Task-vector heatmap: 10 HDBSCAN clusters x GradAscent (cluster reps)")
    print(f"  base   : {args.base_model_dir}")
    print(f"  saves  : {args.saves_dir}")
    print(f"  manifest: {args.manifest}")
    print(f"  tmp    : {args.tmp_dir}")
    print(f"  out    : {args.output_dir}")
    print("=" * 64)

    reps = cluster_representatives(args.manifest)
    print("\nPer-cluster representatives (first triplet of each cluster):")
    for cl, dom, name in reps:
        print(f"  cluster {cl:>2}  {name}  domain={dom}")

    # Discover matched params from base model index
    base_param_names, base_shards = list_matched_params(args.base_model_dir)
    print(f"\nMatched params from base: {len(base_param_names)} tensors "
          f"({len(set(base_shards.values()))} shards)")

    # Stage task vectors to disk as fp16
    stage_paths: dict[str, Path] = {}
    norms: dict[str, float] = {}
    sizes: dict[str, int] = {}
    for cl, dom, name in reps:
        ckpt_dir = args.saves_dir / CKPT_TEMPLATE.format(triplet=name)
        if not ckpt_dir.exists():
            raise FileNotFoundError(ckpt_dir)
        _, ckpt_shards = list_matched_params(ckpt_dir)
        # Sanity: same param set
        if sorted(ckpt_shards.keys()) != base_param_names:
            missing = set(base_param_names) - set(ckpt_shards.keys())
            extra = set(ckpt_shards.keys()) - set(base_param_names)
            raise RuntimeError(
                f"param-name mismatch on {ckpt_dir}: missing {len(missing)}, extra {len(extra)}"
            )
        stage_path = args.tmp_dir / f"tau_cluster{cl:02d}_{name}.fp16.npy"
        if stage_path.exists():
            print(f"\n[cluster {cl}] reusing staged {stage_path.name}")
        else:
            print(f"\n[cluster {cl}] computing tau for {name} ({ckpt_dir.name})")
            flat32 = compute_flat_delta(
                args.base_model_dir,
                ckpt_dir,
                base_param_names,
                base_shards,
                ckpt_shards,
            )
            print(f"  dim={flat32.size:,}  ||tau||={float(np.linalg.norm(flat32)):.4e}")
            np.save(stage_path, flat32.astype(np.float16))
            del flat32
        stage_paths[name] = stage_path
        norms[name] = chunked_norm_fp16(stage_path, chunk=args.chunk)
        a_mm = np.load(stage_path, mmap_mode="r")
        sizes[name] = int(a_mm.size)
        del a_mm
        print(f"  staged ||tau||(fp16)={norms[name]:.4e}  size={sizes[name]:,}")

    # Pairwise cosine
    n = len(reps)
    labels = [f"c{cl}:{dom}" for cl, dom, _ in reps]
    short_labels = [f"c{cl}" for cl, _, _ in reps]
    name_order = [r[2] for r in reps]
    sim = np.eye(n, dtype=np.float64)
    print("\nComputing pairwise cosine ...")
    for i in range(n):
        for j in range(i + 1, n):
            d = chunked_dot_fp16(stage_paths[name_order[i]], stage_paths[name_order[j]],
                                 chunk=args.chunk)
            denom = norms[name_order[i]] * norms[name_order[j]]
            s = d / denom if denom > 0 else 0.0
            sim[i, j] = sim[j, i] = s
            print(f"  ({i},{j}) cluster{reps[i][0]}-cluster{reps[j][0]}: cos={s:+.4f}")

    print("\nCosine similarity matrix:")
    head = " " * 14 + " ".join(f"{l:>10}" for l in short_labels)
    print(head)
    for i, row in enumerate(sim):
        print(f"{short_labels[i]:>12}  " + " ".join(f"{v:10.3f}" for v in row))

    # Plots
    plot_full = args.output_dir / "task_vector_clusters_heatmap.png"
    plot_zoom = args.output_dir / "task_vector_clusters_heatmap_zoom.png"
    title = ("Task Vector Cosine Similarity\n"
             "(GradAscent unlearned model vs base, "
             "first triplet per HDBSCAN cluster, n=10)")
    plot_heatmap(sim, labels, plot_full, title=title, cmap="Blues", vmin=0.0, vmax=1.0)
    # Zoom view: re-scale to actual off-diagonal range to see structure
    mask = ~np.eye(n, dtype=bool)
    off = sim[mask]
    vmin_z = float(np.floor(off.min() * 100) / 100)
    vmax_z = float(np.ceil(off.max() * 100) / 100)
    plot_heatmap(
        sim, labels, plot_zoom,
        title=title + f"\n(zoom: off-diag in [{vmin_z:.2f}, {vmax_z:.2f}])",
        cmap="Blues", vmin=vmin_z, vmax=max(vmax_z, vmin_z + 0.01),
    )

    # JSON dump
    summary = {
        "base_model_dir": str(args.base_model_dir),
        "saves_dir": str(args.saves_dir),
        "n_clusters": n,
        "param_patterns": PARAM_PATTERNS,
        "n_matched_params": len(base_param_names),
        "task_vector_dim": sizes[name_order[0]],
        "cluster_reps": [
            {"cluster_label": cl, "domain": dom, "triplet": name,
             "norm_fp16": norms[name]}
            for (cl, dom, name) in reps
        ],
        "labels": labels,
        "cosine_similarity_matrix": sim.tolist(),
        "off_diag": {
            "mean": float(off.mean()),
            "min": float(off.min()),
            "max": float(off.max()),
            "std": float(off.std()),
        },
    }
    with open(args.output_dir / "task_vector_clusters_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMean off-diagonal cosine: {summary['off_diag']['mean']:+.4f}")
    print(f"Min / Max off-diag      : {summary['off_diag']['min']:+.4f} / "
          f"{summary['off_diag']['max']:+.4f}")
    print(f"\nWrote heatmap (full) -> {plot_full}")
    print(f"Wrote heatmap (zoom) -> {plot_zoom}")
    print(f"Wrote JSON summary   -> {args.output_dir / 'task_vector_clusters_summary.json'}")

    if not args.keep_tmp:
        shutil.rmtree(args.tmp_dir, ignore_errors=True)
        print(f"\nCleaned tmp dir: {args.tmp_dir}  (use --keep_tmp to retain)")


if __name__ == "__main__":
    main()
