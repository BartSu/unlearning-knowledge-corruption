#!/usr/bin/env bash
# End-to-end reproduce on a fresh clone.
# Does NOT re-run unlearn (requires GPU-hours); starts from the committed
# cross-metrics JSON and reproduces §9 of z-doc/README-CN.md steps 1–3.
#
# Usage:
#   scripts/e2e_fresh_clone.sh            # runs in-place
#   scripts/e2e_fresh_clone.sh --clone    # clones to /tmp and runs there

set -euo pipefail

if [[ "${1:-}" == "--clone" ]]; then
    SRC="$(git rev-parse --show-toplevel)"
    DST="$(mktemp -d)/fresh"
    echo "[e2e] cloning $SRC -> $DST"
    git clone "$SRC" "$DST"
    cd "$DST"
fi

echo "=== [1/3] 第一幕：三层 ground truth ==="
( cd 2.extract-ppl && python analyze_corruption.py )

echo "=== [2/3] 第二幕 A：per-sample 几何预测器 ==="
( cd 4.regression-predictor && python 3.corruption_from_geometry.py )

echo "=== [3/3] 第二幕 B：forget-set 审计（LOO） ==="
( cd 4.regression-predictor && python 4.audit_experiments.py )

echo "=== [fig] 重生成 slides 图 ==="
( cd z-doc/figures && python make_figures.py )

echo
echo "[e2e] ✅ reproduce chain completed."
echo "       artefacts:"
ls -la 2.extract-ppl/corruption_summary.json \
       4.regression-predictor/geometry/geometry_results.json \
       4.regression-predictor/audit/audit_summary.json \
       z-doc/figures/fig_*.pdf 2>&1 | sed 's/^/  /'
