# CLAUDE.md —— 阶段 ⑤「Forget-set Audit」

Paper Act II 的方法论核心：**在不真跑 unlearn 的前提下，用 forget set 的内禀几何去预测它被 unlearn 后会造成的三层 corruption（L1/L2/L3）**。本目录的产物是所有 slides / README / paper 引用的 headline 数字。

根目录约定见 [`../CLAUDE.md`](../CLAUDE.md)；上游契约见 [`../3.inference/CLAUDE.md`](../3.inference/CLAUDE.md) 和 [`../4.feature-engineering/CLAUDE.md`](../4.feature-engineering/CLAUDE.md)。

---

## 目录职责

| 做什么 | 不做什么 |
|---|---|
| 用 forget-set 几何预测 L1/L2/L3 | 重新跑 unlearn 或 PPL |
| LOO / bootstrap CI / held-out R²/MAE 三件套 | 提特征（属阶段 ④） |
| 产 `audit_summary.json` —— paper headline | paper 叙事 / slides 图（属 `z-doc/`） |

## 子目录

- [`regression-predictor/`](regression-predictor/) —— **paper 主链**。Ridge / GB / RF on 连续 `log(ppl_ratio)` 或 per-forget-set geo-mean。
- [`classifier-predictor/`](classifier-predictor/) —— 并行实验分支：把 `ppl_ratio > 阈值` 二值化后用 RF 分类。**不在 paper headline 上**；当作 ablation 保留。

## `regression-predictor/` —— 脚本编号即运行顺序

| 脚本 | 产物 | 用途 |
|---|---|---|
| `1.training_data.py` | `training_data.csv` (5000 × 270) | JOIN `4.feature-engineering/features.csv` (X) + `3.inference/extract-ppl/wikitext_cross_metrics_detail.json` (y = `ppl_ratio`) |
| `2.train_rf.py` | `rf/{results.json, logo_predictions.csv, model_ppl.joblib}` | **旁路分支**：LOGO-CV RF on **262** features；用于对照"surface-heavy features 能做多好"，**不是 paper headline** |
| `3.corruption_from_geometry.py` | `geometry/{geometry_results.json, geometry_predictions.csv, corruption_geometry_features.csv}` | LOGO-CV Ridge/GB/RF on **16 维纯几何**（target↔forget + forget 内禀 + target 内禀） |
| `4.audit_experiments.py` | `audit/{part1_*, part2_*, part3_*, audit_summary.json}` | **paper 主 pipeline**：三层 profile + forget-set-level LOO audit (Ridge on 12 维 forget-set 几何) + retain coverage |
| `5.bootstrap_rho_ci.py` | 追加 `bootstrap_rho_ci` 到 `audit_summary.json` | Spearman ρ 95% CI（percentile, n_boot=10000, seed=0） |
| `6.heldout_r2_mae.py` | 追加 `heldout_r2_mae` 到 `audit_summary.json` | LOO held-out R²/MAE + LOO-mean baseline 对照 |

**典型运行顺序**：`1 → 3 → 4 → 5 → 6`。脚本 `2` 是旁路，不在 headline 路径。

## 上下游契约

**读**：
- `../4.feature-engineering/features.csv` —— 5000 × 262 features
- `../3.inference/extract-ppl/wikitext_cross_metrics_detail.json` —— 100 pair × 50 sample per-sample base/unlearn loss+ppl
- `../1.data-preparation/data/wikitext_hdbscan_triplets/triplet_NNN/` —— 只在 `3.corruption_from_geometry.py` 和 `4.audit_experiments.py` 里重新算 embedding 用

**写（paper headline）**：
- `regression-predictor/audit/audit_summary.json` —— 顶级字段：
  - `layer_headline` (L1/L2/L3 geo-mean)
  - `audit_predictor` (LOO R², ρ, r on 12 维 forget-set 几何)
  - `coverage_vs_spillover` (retain coverage 与 L3 spillover 相关性)
  - `bootstrap_rho_ci` (95% CI)
  - `heldout_r2_mae` (LOO R²/MAE + baseline 对照)

## 对 Claude 的具体要求

1. **`ROOT = parents[2]`**（仓库根），**不是 `parents[1]`**（=`5.audit/`）。脚本 1/3/4 已按此约定修好（2026-04-23），新增脚本写路径常量必须对齐。
2. **不归档旧 audit 产物**：切换 unlearn 配置 / 扩 n 时直接覆盖 `audit/ geometry/ rf/`。要回看旧数字用 `git show <sha>:...`，**不要在文件系统里留副本** —— 会让下游文档引用不知道指向哪版。
3. **Paper narrative 用 12/16 维纯几何，不是 262 feat**：脚本 `4.audit_experiments.py` 的 12 维 forget-set 几何是 paper headline；`3.corruption_from_geometry.py` 的 16 维是 per-sample 视角的验证；`2.train_rf.py` 的 262 维 RF 只是 ablation/旁路。汇报数字**先报 4 的 Part 2**。
4. **n=10 的 L3 R² 负值是预期**：bootstrap CI 跨零已证 LOO n=10 对 L3 信号量不够。**不是 bug**，是扩 n 的动机。`STATE.md` 里解释与 slides 一致。
5. **Ridge alpha 目前写死 1.0**：未来扩 n 时建议加 GridSearchCV (alpha ∈ [0.01..10])。当前 TODO，不阻塞。

## 已踩过的坑（留档）

1. **`parents[1]` 陷阱**（2026-04-23 修）：旧目录 `4.regression-predictor/` 在仓库根，`parents[1]` = 仓库根；现在 `5.audit/regression-predictor/`，`parents[1]` = `5.audit/`，`ROOT / "1.data-preparation"` 变成 `5.audit/1.data-preparation`（不存在）。**必须 `parents[2]`**。脚本 1/3/4 已修。

2. **`extract-ppl` 路径过期**：`ROOT / "2.extract-ppl" / "wikitext_cross_metrics_detail.json"` → 实际在 `3.inference/extract-ppl/`。脚本 3/4 已改。

3. **`feature-engineering` 路径过期**：`ROOT.parent / "feature-engineering" / "features.csv"` → 实际在 `4.feature-engineering/features.csv`，且 `ROOT.parent` 在新目录下是 `5.audit/` 不是仓库根。脚本 1 已改（新增 `REPO = ROOT.parents[1]`）。

4. **`training_data.csv` JOIN 静默丢行**：JOIN key = `(eval_triplet, sample_index)`。如果上游 `features.csv` 的 `sample_index` 语义漂移（例如从 test split 改到 train），merge 会丢样本或产 NaN 而不报错。建议在 `1.training_data.py` 加 `assert len(joined) == 100 * 50`。

5. **LOO n=10 下 Spearman ρ 对噪声敏感**：同一组 (true, pred) 改 1 个值 ρ 可能从 +0.6 跳到 +0.3。Bootstrap CI 是唯一可靠的稳健性报告方式，不要只引用点估 ρ。

6. **`cluster_features.csv` 可选依赖**：`analyze_corruption.py` 会尝试读（不在本目录），不存在时 silently skip。当前未生成，不阻塞，但日志里会有一行 `Errno 2: No such file ...`。

## Smoke test（~3 min，纯数值无 GPU，但 sentence-transformer 要 GPU + `HF_HUB_CACHE`）

```bash
cd 5.audit/regression-predictor
source /media/volume/llm/miniconda3/etc/profile.d/conda.sh && conda activate unlearning
export HF_HUB_CACHE=/media/volume/llm/huggingface/hub \
       HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets \
       CUDA_HOME=$HOME/fake_cuda
python 1.training_data.py
python 3.corruption_from_geometry.py
python 4.audit_experiments.py
python 5.bootstrap_rho_ci.py
python 6.heldout_r2_mae.py
python3 -c "import json; s=json.load(open('audit/audit_summary.json')); print(sorted(s.keys()))"
```

期望（n=10 TOFU-aligned 实测 2026-04-23）：5 个顶级 key；L1 R²=+0.292 / L2=+0.523 / L3=−0.458；ρ(L1)=+0.624 CI [+0.08,+0.91]；ρ(L2)=+0.842 CI [+0.43,+1.00]；audit L1/L2 胜 LOO-mean baseline（baseline 三层都是 R²=−0.235）。
