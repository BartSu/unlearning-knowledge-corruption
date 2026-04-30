# CLAUDE.md —— 阶段 ⑤「Forget-set Audit」

Paper Act II 的方法论核心：**在不真跑 unlearn 的前提下**，用 forget set 的内禀几何去预测它被 unlearn 后会造成的三层 corruption (L1/L2/L3)。本目录的产物是所有 slides / README / paper 引用的 headline 数字。

根目录约定见 [`../CLAUDE.md`](../CLAUDE.md)；上游契约见 [`../3.inference/CLAUDE.md`](../3.inference/CLAUDE.md) 和 [`../4.feature-engineering/CLAUDE.md`](../4.feature-engineering/CLAUDE.md)。

---

## 目录职责

| 做什么 | 不做什么 |
|---|---|
| 用 forget-set 几何预测 L1/L2/L3 | 重新跑 unlearn 或 PPL |
| 从 ④ 读 X + 从 ③ 读 y → Ridge LOO / LOGO | 提特征（**阶段 ④ 已做**，这里只 JOIN） |
| 产 `audit_summary.json` —— paper headline | paper 叙事 / slides 图（属 `z-doc/`） |

## `regression-predictor/` —— 4 个编号脚本

脚本编号为 **3/4/5/6/7**（非 1-based）—— 保留历史编号以维持 git log 连续性；原 `1.training_data.py` / `2.train_rf.py` 随 262 维 ablation 一起在 2026-04-23 移除。**运行顺序**按编号递增：

| 脚本 | 产物 | 用途 |
|---|---|---|
| `3.corruption_from_geometry.py` | `geometry/{geometry_results.json, geometry_predictions.csv, corruption_geometry_features.csv}` | **并行视角**：LOGO-CV by `eval_triplet` 在 **16 维 per-sample** 几何上跑 Ridge / GB / RF → 对照 Ridge 的可学习性 |
| `4.audit_experiments.py` | `audit/{part1_*, part2_*, part3_*, audit_summary.json}` | **paper headline**：LOO by `forget_cluster` 在 **12 维 per-forget-set** 几何上跑 Ridge → L1/L2/L3 R²/ρ |
| `5.bootstrap_rho_ci.py` | 追加 `bootstrap_rho_ci` 到 `audit_summary.json` | Spearman ρ 95% CI（percentile, n_boot=10000, seed=0） |
| `6.heldout_r2_mae.py` | 追加 `heldout_r2_mae` 到 `audit_summary.json` | LOO held-out R²/MAE + LOO-mean baseline 对照 |
| `7.ranking_metrics.py` | 追加 `ranking_metrics` 到 `audit_summary.json` | top-k recall (k=5/10/20%) + NDCG@k + Kendall τ + pairwise concordance + bootstrap CI + random baseline lift。**Act II "粗筛预警" 的真考卷** |

## 上下游契约

**读 阶段 ④**（`X`，本阶段**不再内部 embed**）：
- `../4.feature-engineering/forget_set_geometry.csv` —— 12 维 per-forget-set 几何（`4.audit_experiments.py` 的 `part2_forget_features` 读）
- `../4.feature-engineering/per_sample_geometry.csv` —— 16 维 per-sample 几何（`3.corruption_from_geometry.py` 的 `build_features` 读，再 JOIN 上 label）

**读 阶段 ③**（`y`）：
- `../3.inference/extract-ppl/wikitext_cross_metrics_detail.json` —— 读 per-sample base/unlearn ppl 算 `log_ppl_ratio`

**读 阶段 ①**（仅 `4.audit_experiments.py` 的 `part3_coverage` 还在用）：
- `../1.data-preparation/data/wikitext_hdbscan_triplets/triplet_NNN/{train,validation}.json` —— retain-coverage 要 embed forget + validation 对比

**写（paper headline）**：
- `regression-predictor/audit/audit_summary.json` —— 顶层字段：
  - `layer_headline` (L1/L2/L3 geo-mean)
  - `audit_predictor` (LOO R²/ρ/r on 12 维 forget-set 几何)
  - `coverage_vs_spillover` (retain coverage 与 L3 spillover 相关性)
  - `bootstrap_rho_ci` (95% CI)
  - `heldout_r2_mae` (LOO R²/MAE + baseline 对照)
  - `ranking_metrics` (top-k recall / NDCG / Kendall τ + bootstrap CI + lift over random)

## 对 Claude 的具体要求

1. **`ROOT = parents[2]`**（仓库根），**不是 `parents[1]`**（=`5.audit/`）。脚本 3/4 已按此修好（2026-04-23），新增脚本写路径常量必须对齐。
2. **不在本阶段重新算 embedding / 几何**：所有 X 都来自阶段 ④ 的 CSV。`4.audit_experiments.py` 的 `part3_coverage` 是唯一例外（目前为了 validation vs forget 的覆盖度仍在内部 embed）—— 如果未来 retain-coverage 要扩展，优先把 embed 逻辑也搬到 ④，保持阶段 ⑤ 纯粹。
3. **旧产物不归档**：切换 unlearn 配置 / 扩 n 时直接覆盖 `audit/ geometry/`。要回看旧数字用 `git show <sha>:...`，不要在文件系统留副本。
4. **Paper narrative 用 12 维 forget-set 几何**：`4.audit_experiments.py` 产的 `audit/audit_summary.json` 里 `audit_predictor` 字段是 headline；`3.corruption_from_geometry.py` 的 16 维 per-sample 几何是**并行视角**的验证。汇报数字先报 4 的 Part 2。
5. **n=10 的 L3 R² 负值是预期**：bootstrap CI 跨零已证 LOO n=10 对 L3 信号量不够。**不是 bug**，是扩 n 的动机。`STATE.md` 里解释与 slides 一致。
6. **Ridge alpha 当前写死 1.0**：扩 n 时建议加 GridSearchCV (alpha ∈ [0.01..10])。当前 TODO，不阻塞。

7. **n=100 同簇 triplet 几何高度相似 → top-1 / top-3 不可靠**：`audit_summary.json["audit_predictor"]["top1_match"]` / `top3_overlap` 在 n=100 下都是 0，看起来"完全没找对"。但实际 ρ=0.49–0.84，top-10% recall lift 2–5×。**汇报 audit 能力时优先看 `ranking_metrics.layers.*.topk` 的 recall + lift_over_random + CI**，不要再把 top-1/top-3 单独拎出来当 headline；它在 100 个 forget set + 同簇邻居（如 #71/#72/#73）几何噪声下天然不稳。

## 已踩过的坑（留档）

1. **`parents[1]` 陷阱**（2026-04-23 修）：旧目录 `4.regression-predictor/` 在仓库根，`parents[1]` = 仓库根；现在 `5.audit/regression-predictor/`，`parents[1]` = `5.audit/`，`ROOT / "1.data-preparation"` 变成 `5.audit/1.data-preparation`（不存在）。**必须 `parents[2]`**。脚本 3/4 已修。

2. **`extract-ppl` 路径过期**：`ROOT / "2.extract-ppl" / "wikitext_cross_metrics_detail.json"` → 实际在 `3.inference/extract-ppl/`。脚本 3/4 已改。

3. **cached `corruption_geometry_features.csv` 静默污染**（2026-04-23 发现）：`3.corruption_from_geometry.py` 的 main 逻辑 "if csv exists: read; else: build"。切换 unlearn 配置（max_steps=2 → 5）时如果没清 cache CSV，会**静默**继续用旧 label 跑 RF，导出"看起来对但数字是旧配置的"结果。**修法**：切换 unlearn 配置前 `rm -f 5.audit/regression-predictor/geometry/corruption_geometry_features.csv`。同样对 `audit/` 下的 part2/part3 CSV（但这些是每次 overwrite 不用担心）。

4. **LOO n=10 下 Spearman ρ 对噪声敏感**：同一组 (true, pred) 改 1 个值 ρ 可能从 +0.6 跳到 +0.3。Bootstrap CI 是唯一可靠的稳健性报告方式，不要只引用点估 ρ。

5. **262 维 ablation 已移除**（2026-04-23）：原 `1.training_data.py` / `2.train_rf.py` 读 `4.feature-engineering/features.csv`（262 维 surface+几何混合）跑 RF baseline。paper 结论"12 维纯几何足够"后，该分支以及 `classifier-predictor/` 子目录整体删除。

## Smoke test（~3 min，纯数值 + 一次 sentence-transformer embed for coverage）

```bash
cd 5.audit/regression-predictor
source /media/volume/llm/miniconda3/etc/profile.d/conda.sh && conda activate unlearning
export HF_HUB_CACHE=/media/volume/llm/huggingface/hub \
       HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets \
       CUDA_HOME=$HOME/fake_cuda

# 先清 cache 避免旧数字污染
rm -f geometry/corruption_geometry_features.csv

python 3.corruption_from_geometry.py    # 16 维 LOGO-CV
python 4.audit_experiments.py           # 12 维 LOO → audit_summary.json
python 5.bootstrap_rho_ci.py            # 追加 CI
python 6.heldout_r2_mae.py              # 追加 held-out + baseline
python 7.ranking_metrics.py             # 追加 top-k recall / NDCG / lift

python3 -c "import json; s=json.load(open('audit/audit_summary.json')); print(sorted(s.keys()))"
```

期望（n=10 TOFU-aligned 实测 2026-04-23）：
- 6 个顶级 key：`audit_predictor / bootstrap_rho_ci / coverage_vs_spillover / heldout_r2_mae / layer_headline / ranking_metrics`
- audit_predictor L1 R²=+0.292 / L2=+0.523 / L3=−0.458
- bootstrap_rho_ci L1 [+0.08, +0.91] / L2 [+0.43, +1.00] / L3 [−0.51, +0.71]
- heldout audit 胜 LOO-mean baseline（baseline 三层都是 R²=−0.235）：L1 / L2 胜，L3 输（n=10 信号不足，扩 n=100 的动机）
- 3.corruption_from_geometry LOGO-CV（16 维）：mean_baseline R²=−0.017 / Ridge +0.225 / GB +0.215 / RF +0.175 / RF L3-only +0.016
