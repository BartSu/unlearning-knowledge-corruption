# 1. Data Preparation

Stage 1 of the unlearning pipeline. Builds WikiText-103 triplet datasets used
downstream by unlearn (stage 2) and cross-PPL extraction (stage 3).

## Pipeline summary

WikiText-103 → filter → sentence embeddings → UMAP/PCA reduce → HDBSCAN
clustering (10 non-noise clusters) → per-cluster keyword labeling → triplet
sampling → 100 triplets (10 clusters × 10 triplets/cluster).

The upstream filtering / embedding / clustering code lives under
`open-unlearning/` and `scripts/` (see `data/wikitext_hdbscan_triplets/run_manifest.json`
for exact input paths and parameters). This README covers the final **triplet
dataset schema** consumed by stage 2 onward.

## File layout

```
1.data-preparation/
├── data/
│   └── wikitext_hdbscan_triplets/
│       ├── run_manifest.json        # provenance + per-triplet metadata
│       ├── triplet_001/
│       │   ├── train.json           # forget set        (50 texts)
│       │   ├── validation.json      # retain set        (50 texts)
│       │   └── test.json            # held-out probe    (50 texts)
│       ├── triplet_002/
│       │   └── …
│       └── triplet_100/
├── unlearn/
│   ├── wikitext_unlearn_sample.sh   # single-triplet smoke-test
│   ├── wikitext_unlearn_batch.sh    # batch unlearn (10 reps / 100 triplets)
│   └── eval_wikitext_perplexity.py  # baseline + cross-eval loss/PPL
└── open-unlearning/                 # vendored OpenUnlearning framework
```

## Triplet schema

Each `triplet_NNN/{train,validation,test}.json` is a JSON **list** of objects
with a single `text` field:

```json
[
  {"text": "Darden was drafted in the first round …"},
  {"text": "Chris Turner ( born September 8 , 1987 ) …"}
]
```

| split       | file              | role           | size |
|-------------|-------------------|----------------|------|
| train       | `train.json`      | forget set     | 50   |
| validation  | `validation.json` | retain set     | 50   |
| test        | `test.json`       | held-out probe | 50   |

All three splits are drawn from the **same HDBSCAN cluster** for a given
triplet (disjoint samples). Fixed sizes come from `triplet_generation` in
`run_manifest.json`:

```
forget_size = validation_size = test_size = 50
triplets_per_domain = 10        # 10 triplets per cluster
required_cluster_size = 150     # clusters smaller than this are skipped
seed = 42
```

## Triplet indexing

- 100 triplets: `triplet_001 … triplet_100`
- 10 HDBSCAN clusters (domains labeled by top keywords), 10 triplets each
- Cluster representatives used for the `n=10` audit: `triplet_0{01,11,21,…,91}`
  (the first triplet of each cluster)

Per-triplet metadata lives in `run_manifest.json → triplets[i]`:

```json
{
  "name": "triplet_001",
  "cluster_label": 0,
  "domain": "game_yard_tech",
  "domain_triplet_index": 1,
  "cluster_size": 245,
  "forget_size": 50,
  "validation_size": 50,
  "test_size": 50,
  "unused_cluster_samples_per_triplet": 95
}
```

## Downstream contract

Stage 2 (`unlearn/wikitext_unlearn_{sample,batch}.sh`) consumes
`train.json` (forget) and `validation.json` (retain) per triplet and writes
checkpoints to `unlearn/saves/wikitext_unlearn{_sample,_batch}/`.

Stage 3 (`2.extract-ppl/`) runs `eval_wikitext_perplexity.py` to compute
cross-PPL on all three splits for every (unlearn-checkpoint × eval-triplet)
pair.
