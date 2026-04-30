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

## `regression-predictor/` —— 8 个编号脚本

脚本编号为 **3/4/5/6/7/8/9/10**（非 1-based）—— 保留历史编号以维持 git log 连续性；原 `1.training_data.py` / `2.train_rf.py` 随 262 维 ablation 一起在 2026-04-23 移除。**运行顺序**按编号递增：

| 脚本 | 产物 | 用途 |
|---|---|---|
| `3.corruption_from_geometry.py` | `geometry/{geometry_results.json, geometry_predictions.csv, corruption_geometry_features.csv}` | **并行视角**：LOGO-CV by `eval_triplet` 在 **16 维 per-sample** 几何上跑 Ridge / GB / RF → 对照 Ridge 的可学习性 |
| `4.audit_experiments.py` | `audit/{part1_*, part2_*, part3_*, audit_summary.json}` | **paper headline (GradAscent)**：LOO over n=100 forget sets, **Random Forest**(n=200, min_leaf=2, seed=0) 在 **12 维 per-forget-set** 几何上 → L1/L2/L3 R²/ρ |
| `5.bootstrap_rho_ci.py` | 追加 `bootstrap_rho_ci` 到 `audit_summary.json` | Spearman ρ 95% CI（percentile, n_boot=10000, seed=0） |
| `6.heldout_r2_mae.py` | 追加 `heldout_r2_mae` 到 `audit_summary.json` | LOO held-out R²/MAE + LOO-mean baseline 对照 |
| `7.ranking_metrics.py` | 追加 `ranking_metrics` 到 `audit_summary.json` | top-k recall (k=5/10/20%) + NDCG@k + Kendall τ + pairwise concordance + bootstrap CI + random baseline lift |
| `8.alt_predictors.py` | 追加 `alt_predictors` 到 `audit_summary.json` | **paper §5.4 ablation**: Ridge / Lasso / GB 在同 12-d / 同 LOO 上对照 RF headline |
| `9.npo_transfer.py` | `audit/{npo_headline.json, npo_transfer.json, npo100_*.json}` | **informal**：用 GradAscent-trained RF 预测 NPO 30/100 forget set 三层（show transfer ρ, R² 不可比 due to scale mismatch） |
| `10.npo_audit.py` | `audit/npo_audit_summary.json` | **paper §5.5 cross-unlearner**：跟 4 同协议但 input 是 NPO 100 labels → 独立 NPO audit (NPO L1 ρ=0.66 / L2 ρ=0.82 / L3 ρ=0.51, 三层 CI 都 >0) |

## 上下游契约

**读 阶段 ④**（`X`，本阶段**不再内部 embed**）：
- `../4.feature-engineering/forget_set_geometry.csv` —— 12 维 per-forget-set 几何（`4.audit_experiments.py` 的 `part2_forget_features` 读）
- `../4.feature-engineering/per_sample_geometry.csv` —— 16 维 per-sample 几何（`3.corruption_from_geometry.py` 的 `build_features` 读，再 JOIN 上 label）

**读 阶段 ③**（`y`）：
- `../3.inference/extract-ppl/wikitext_cross_metrics_detail.json` —— 读 per-sample base/unlearn ppl 算 `log_ppl_ratio`

**读 阶段 ①**（仅 `4.audit_experiments.py` 的 `part3_coverage` 还在用）：
- `../1.data-preparation/data/wikitext_hdbscan_triplets/triplet_NNN/{train,validation}.json` —— retain-coverage 要 embed forget + validation 对比

**写（paper headline）**：
- `regression-predictor/audit/audit_summary.json` —— GradAscent 视角，顶层字段：
  - `layer_headline` (L1/L2/L3 geo-mean)
  - `audit_predictor` (LOO RF R²/ρ on 12 维 forget-set 几何)
  - `coverage_vs_spillover` (retain coverage 与 L3 spillover 相关性)
  - `bootstrap_rho_ci` (95% CI)
  - `heldout_r2_mae` (LOO R²/MAE + baseline 对照)
  - `ranking_metrics` (top-k recall / NDCG / Kendall τ + bootstrap CI + lift over random)
  - `alt_predictors` (Ridge / Lasso / GB ablation)
- `regression-predictor/audit/npo_audit_summary.json` —— NPO 平行视角（同结构）
- `regression-predictor/audit/npo100_headline.json` + `npo_transfer.json` —— 中间产物（NPO per-forget profile + GradAscent-RF transfer 对照）

## 对 Claude 的具体要求

1. **`ROOT = parents[2]`**（仓库根），**不是 `parents[1]`**（=`5.audit/`）。脚本 3/4 已按此修好（2026-04-23），新增脚本写路径常量必须对齐。
2. **不在本阶段重新算 embedding / 几何**：所有 X 都来自阶段 ④ 的 CSV。`4.audit_experiments.py` 的 `part3_coverage` 是唯一例外（目前为了 validation vs forget 的覆盖度仍在内部 embed）—— 如果未来 retain-coverage 要扩展，优先把 embed 逻辑也搬到 ④，保持阶段 ⑤ 纯粹。
3. **旧产物不归档**：切换 unlearn 配置 / 扩 n 时直接覆盖 `audit/ geometry/`。要回看旧数字用 `git show <sha>:...`，不要在文件系统留副本。
4. **Paper narrative 用 12 维 forget-set 几何 + Random Forest**：`4.audit_experiments.py` 产的 `audit/audit_summary.json` 里 `audit_predictor` 字段是 GradAscent headline；`3.corruption_from_geometry.py` 的 16 维 per-sample 几何是**并行视角**的验证；`8.alt_predictors.py` 是 ablation（Ridge / Lasso / GB 同协议对比 RF）；`10.npo_audit.py` 是 cross-unlearner（NPO 平行 audit）。汇报数字先报 4 的 RF headline。
5. **n=100 RF 下 L3 CI 严格 > 0**（n=10 时 R²=−0.46 是历史，已被 n=100 + RF 修复到 R²=+0.305 / ρ=0.59 [0.44, 0.71]）；ablation 表显示 even Ridge n=100 也能把 L3 R² 拉到 +0.215 / ρ=0.49 [0.33, 0.63]，所以 statistical power 是关键，model class 是次要。
6. **Random Forest hyperparam 当前写死 (n_estimators=200, min_samples_leaf=2, max_depth=None, random_state=0)**：未来改 hyperparam 须同步更新 `4.audit_experiments.py` / `8.alt_predictors.py` / `10.npo_audit.py` 三处保持协议一致。
7. **n=100 同簇 triplet 几何高度相似 → top-1 / top-3 不可靠**：`audit_summary.json["audit_predictor"]["top1_match"]` / `top3_overlap` 在 n=100 下都是 0，看起来"完全没找对"。但实际 ρ=0.51–0.87，top-10% recall lift 2–8×。**汇报 audit 能力时优先看 `ranking_metrics.layers.*.topk` 的 recall + lift_over_random + CI**，不要再把 top-1/top-3 单独拎出来当 headline；它在 100 个 forget set + 同簇邻居（如 #71/#72/#73）几何噪声下天然不稳。
8. **NPO 用 BS=2 GAS=32**（GradAscent 用 BS=8 GAS=8，effective_batch 都是 64）：NPO 含 reference model（多一个 8B params + activation），H100 80GB BS=4 在 step 4/5 OOM，BS=2 才稳。如果 cross-unlearner 加 GradDiff/RMU 也带 reference model，同样要降 BS。

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

python 3.corruption_from_geometry.py    # 16 维 LOGO-CV (并行视角)
python 4.audit_experiments.py           # GradAscent RF headline → audit_summary.json
python 5.bootstrap_rho_ci.py            # 追加 CI
python 6.heldout_r2_mae.py              # 追加 held-out + baseline
python 7.ranking_metrics.py             # 追加 top-k recall / NDCG / lift
python 8.alt_predictors.py              # 追加 alt_predictors (Ridge/Lasso/GB ablation)
python 10.npo_audit.py                  # 独立 NPO audit → npo_audit_summary.json (要求 NPO cross-PPL 已跑)

python3 -c "import json; s=json.load(open('audit/audit_summary.json')); print(sorted(s.keys()))"
```

期望（n=100 TOFU-aligned + RF headline 实测 2026-04-30）：
- audit_summary.json 7 个顶级 key：`alt_predictors / audit_predictor / bootstrap_rho_ci / coverage_vs_spillover / heldout_r2_mae / layer_headline / ranking_metrics`
- audit_predictor L1 R²=+0.301 / ρ=+0.560；L2 R²=+0.735 / ρ=+0.869；L3 R²=+0.305 / ρ=+0.592
- bootstrap_rho_ci L1 [+0.40, +0.69] / L2 [+0.81, +0.91] / L3 [+0.44, +0.71]（三层 CI 都 >0）
- heldout audit 胜 LOO-mean baseline（baseline 三层都是 R²=−0.020）：三层都胜
- ranking_metrics top-10% recall L1=2/10 (2×) / L2=6/10 (6×) / L3=3/10 (3×)
- npo_audit_summary.json 平行结构，L1 ρ=+0.66 / L2 ρ=+0.82 / L3 ρ=+0.51（三层 CI 都 >0）
