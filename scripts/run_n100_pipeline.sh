#!/bin/bash
# Full n=100 pipeline: 90 remaining unlearn → 100×100 cross-PPL → audit.
#
# Usage:
#   nohup ./scripts/run_n100_pipeline.sh > scripts/run_n100.log 2>&1 &
#   disown
#
# Monitor:
#   tail -f scripts/run_n100.log
#   cat  scripts/.n100_state           # STEP1_DONE / STEP2_DONE / STEP3_DONE / FAIL:<msg>
#   ls -lh 2.train-unlearn/unlearn/saves/wikitext_unlearn_tofu/ | wc -l
#   tail 3.inference/extract-ppl/cross_eval_100x100.log

set -uo pipefail

REPO=/media/volume/llm/unlearning-knowledge-corruption
STATE=$REPO/scripts/.n100_state
LOG_ROOT=$REPO/scripts

log() { echo "[$(date -Iseconds)] $*"; }
mark() { echo "$(date -Iseconds) $1" >> "$STATE"; }
fail() { echo "FAIL: $1 ($(date -Iseconds))" >> "$STATE"; log "FAIL: $1"; exit 1; }

echo "---- new run @ $(date -Iseconds) ----" >> "$STATE"

# Env
source /media/volume/llm/miniconda3/etc/profile.d/conda.sh
conda activate unlearning
export HF_HUB_CACHE=/media/volume/llm/huggingface/hub
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export CUDA_HOME=$HOME/fake_cuda

# Disk sanity
avail_gb=$(df --output=avail -BG $REPO | tail -1 | tr -dc '0-9')
log "disk available: ${avail_gb} GB (need ≥ 1500 for 90 new ckpts)"
if [ "${avail_gb:-0}" -lt 1500 ]; then
    fail "insufficient disk (${avail_gb}GB < 1500GB)"
fi

# ──────────────────────── STEP 1: 90 triplet unlearn ────────────────────────
log "STEP 1: remaining 90 triplet unlearn @ bs=8 gas=8"
cd $REPO/2.train-unlearn/unlearn
./wikitext_unlearn_tofu_aligned.sh --bs 8 --gas 8 > $LOG_ROOT/n100_step1.log 2>&1
RC=$?
n_ckpt=$(ls -d saves/wikitext_unlearn_tofu/wikitext_*_triplet_*_GradAscent_tofu/ 2>/dev/null | wc -l)
log "step 1 exit=$RC  ckpt count=$n_ckpt/100"
if [ "$n_ckpt" -lt 100 ]; then
    n_fail=$(grep -c fail logs_wikitext_unlearn_tofu/_summary.csv || echo 0)
    fail "only $n_ckpt/100 ckpts; $n_fail fails in summary.csv"
fi
mark "STEP1_DONE n_ckpt=$n_ckpt"

# ──────────────────── STEP 2: 100×100 cross-PPL + analyze ────────────────────
log "STEP 2: 100×100 cross-PPL"
cd $REPO/3.inference/extract-ppl

# Archive n=10 products before expanding (so analyze_corruption starts fresh on n=100)
mkdir -p legacy_n10_tofu
for f in wikitext_cross_metrics.json wikitext_cross_metrics_detail.json \
         corruption_summary.json ppl_long.parquet ppl_long.jsonl; do
    [ -e "$f" ] && mv "$f" legacy_n10_tofu/ 2>/dev/null || true
done
log "archived n=10 products to legacy_n10_tofu/"

# Extend baseline 50→100 (cross-eval would do this implicitly; doing explicit for clarity)
python eval_wikitext_perplexity.py --baseline \
    --data_dir $REPO/1.data-preparation/data/wikitext_hdbscan_triplets \
    --start 1 --end 100 --resume \
    > $LOG_ROOT/n100_step2_baseline.log 2>&1
RC=$?
log "  baseline extend exit=$RC"
[ $RC -eq 0 ] || fail "baseline extend rc=$RC"

# 100×100 cross-eval
python eval_wikitext_perplexity.py \
    --saves_dir $REPO/2.train-unlearn/unlearn/saves/wikitext_unlearn_tofu \
    --data_dir $REPO/1.data-preparation/data/wikitext_hdbscan_triplets \
    --start 1 --end 100 --batch_size 4 --resume \
    > $LOG_ROOT/n100_step2_cross.log 2>&1
RC=$?
n_rows=$(python3 -c "import json; print(len(json.load(open('wikitext_cross_metrics.json'))['results']))")
log "  cross-eval exit=$RC  rows=$n_rows (expect 10000)"
[ $RC -eq 0 ] || fail "cross-eval rc=$RC"
[ "$n_rows" -ge 10000 ] || fail "cross-eval only produced $n_rows rows"

python analyze_corruption.py > $LOG_ROOT/n100_step2_analyze.log 2>&1
python export_ppl_table.py   > $LOG_ROOT/n100_step2_export.log  2>&1
python sanity_check_ppl.py   > $LOG_ROOT/n100_step2_sanity.log  2>&1
mark "STEP2_DONE n_rows=$n_rows"

# ────────────── STEP 3: regen stage-4 X CSVs + audit pipeline ──────────────
log "STEP 3a: regenerate stage-4 geometry CSVs for n=100"
cd $REPO/4.feature-engineering/scripts
python extract_forget_geometry.py      > $LOG_ROOT/n100_step3a_forget.log     2>&1
python extract_per_sample_geometry.py  > $LOG_ROOT/n100_step3a_per_sample.log 2>&1
mark "STEP3a_DONE"

log "STEP 3b: audit pipeline (3→4→5→6)"
cd $REPO/5.audit/regression-predictor
rm -f geometry/corruption_geometry_features.csv   # force re-JOIN with n=100 labels
python 3.corruption_from_geometry.py > $LOG_ROOT/n100_step3b_3.log 2>&1
python 4.audit_experiments.py        > $LOG_ROOT/n100_step3b_4.log 2>&1
python 5.bootstrap_rho_ci.py         > $LOG_ROOT/n100_step3b_5.log 2>&1
python 6.heldout_r2_mae.py           > $LOG_ROOT/n100_step3b_6.log 2>&1
mark "STEP3_DONE"

# Final headline
python3 - <<'PY' >> "$STATE"
import json
c = json.load(open("/media/volume/llm/unlearning-knowledge-corruption/3.inference/extract-ppl/corruption_summary.json"))
a = json.load(open("/media/volume/llm/unlearning-knowledge-corruption/5.audit/regression-predictor/audit/audit_summary.json"))
print("HEADLINE n=100:")
for k in ("L1_forget","L2_locality_same_cluster","L3_cross_cluster_spillover"):
    v=c[k]; print(f"  {k:<30s} n={v['n']:>6} geo={v['geo_mean_ratio']:.3f}x pct>1.1={v['pct_up_10']:.1f}%")
ap = a["audit_predictor"]
for k in ("geo_L1_forget","geo_L2_locality","geo_L3_spillover"):
    v=ap[k]; print(f"  audit {k:<22s} R2={v['r2']:+.3f} rho={v['spearman_rho']:+.3f}")
PY

log "---- ALL DONE @ $(date -Iseconds) ----"
