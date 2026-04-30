# PROGRESS.md - 具体的操作过程

## 任务目标
迭代推进 LLM 遗忘（unlearning）研究流水线：
**数据准备 → unlearn 模型 → 困惑度提取 → 几何分析 → forget-set audit**。
最终目标：把 unlearning benchmarking 重构为 *forget-set 审计*（从"每次都跑 unlearn"到"先粗筛预警、再真跑 top-k"）。

## 背景信息
- 仓库根目录：`/media/volume/llm/unlearning`
- 五个流水线阶段与编号目录的对应关系：
  1. **数据准备**：`1.data-preparation/` —— WikiText-103 → HDBSCAN 10 簇 → 100 triplets（每簇 10 个）
  2. **Unlearn 模型**：`2.train-unlearn/unlearn/` —— GradAscent，每 forget set 一个 checkpoint
  3. **困惑度提取**：`2.extract-ppl/` —— 10×10 cross-PPL，输出 L1/L2/L3 三层 ground truth
  4. **几何分析**：`3.feature-engineering/` + `4.regression-predictor/geometry/` —— forget-set 内禀几何特征（12 维）
  5. **Forget-set audit**：`4.regression-predictor/audit/` —— Ridge LOO，输出三层排序预测
- **研究定位**（与 `z-doc/slides.tex` 保持一致）：三层 corruption view 是 Act I 的观察结果；forget-set audit（"便宜的粗筛预警器"）是 Act II 的方法论提议。所有代码、文档、slides 的叙事都应围绕这两幕展开。

## 当前瓶颈
> 迁移到 [`STATE.md`](STATE.md)（关键数字 + 决策点 + 下一步，单一来源）。本文件专注「任务清单 + 迭代日志」。

## 文档同步规则
z-doc/ 当前两份对外材料：`paper/paper.tex` (EMNLP draft) + `slides/slides.tex` (英文 beamer)。两份引用同源数字，改 audit 关键数字（`5.audit/.../audit_summary.json` 或 `npo_audit_summary.json`）后须刷新两份。
- 任一 `.tex` 改后用 `~/bin/tectonic` 重编 `.pdf`，`.tex` + `.pdf` 一并 commit。
- 图源在 `z-doc/figures/`（真 PDF），paper.tex 用 `\graphicspath{{../figures/}}`，slides 同。改 audit 数字后用 `python z-doc/figures/make_figures.py` 重生图。

## 任务列表

### 阶段 1 — 数据准备（`1.data-preparation/`）
- [x] WikiText-103 → HDBSCAN 10 簇 → 100 triplets 已生成（`data/wikitext_hdbscan_triplets/triplet_001..100`）
- [x] 验证 `2.train-unlearn/unlearn/wikitext_unlearn_sample.sh` 能在小样本上端到端运行（`saves/wikitext_unlearn_sample/.../triplet_001_GradAscent/` 150/150 step、epoch 10/10、4 分片 safetensors + `trainer_state.json` 齐全）
- [x] 恢复或替换已删除的 `2.train-unlearn/unlearn/eval_wikitext_perplexity.py`（从 `c799c51~1` 恢复 698 行；路径更新为 `1.data-preparation/`；`py_compile` + `--help` 通过）
- [x] 在 `1.data-preparation/README.md` 中记录数据集 schema（triplet {train=forget, validation=retain, test=probe} × 50 条 `{"text": ...}`；100 triplet × 10 cluster 布局）

### 阶段 2 — Unlearn 模型（`2.train-unlearn/unlearn/`）✅ GradAscent + NPO 都 n=100
- [x] 跑 10 个代表 triplet（`triplet_0{01,11,21,…,91}`）的 GradAscent checkpoint（深度配置：max_steps=150 / epoch=10，见 `saves/wikitext_unlearn_sample/`）
- [x] batch 跑满 100 个 triplet 的 GradAscent checkpoint（**浅配置**：max_steps=2 / epoch=1 / train_loss ≈ −2.4 / ~38 s 每个；落在 `saves/wikitext_unlearn/triplet_001..100_GradAscent/`）
- [x] 跑 100 个 GradAscent ckpt 的 TOFU-aligned 主配置（max_steps=5 / epoch=5 / lr=1e-5 / BS=8 GAS=8 / 1h54min），落在 `saves/wikitext_unlearn_tofu/wikitext_*_GradAscent_tofu/`
- [x] 加第二个 unlearner NPO 100 ckpt（BS=2 GAS=32 effective_batch=64 / NPO 含 reference model H100 BS=4 OOM / 96s/triplet wall / 100 个总 ~2.7h），落在 `saves/wikitext_unlearn_tofu/wikitext_*_NPO_tofu/`，符号链接到 `saves/wikitext_unlearn_tofu_npo100/` 给 cross-PPL 用
- [ ] 加 GradDiff / RMU 验证审计器跨算法稳健性 (NPO 已 done, GradDiff/RMU 是 next)

### 阶段 3 — 困惑度提取（`2.extract-ppl/`）
- [x] 对 10 个 unlearn ckpt 跑 10×10 cross-PPL → `wikitext_cross_metrics_detail.json`
- [x] `analyze_corruption.py` 产出 L1/L2/L3 三层 ground truth → `corruption_summary.json`（已扩到 n=50：5 triplet/cluster × 10 cluster，L1 n=2500 / L2 n=5000 / L3 n=122500；geo 2.126 / 1.491 / 1.283）
- [x] 以统一格式（parquet + jsonl）保存逐样本 PPL 长表（`export_ppl_table.py` → `ppl_long.{parquet,jsonl}`，**15000 行对应旧 n=10 快照**，layer ∈ {L1,L2,L3,L3_other}）
- [x] base vs unlearned PPL 分布 sanity-check 脚本（`sanity_check_ppl.py`，四项 invariant 在 n=10 快照上全 PASS：L1 geo=1.762x，L1>L2>L3 单调，base PPL 跨层一致，L1 100% 样本 ppl 上升）
- [ ] 重跑 `export_ppl_table.py` + `sanity_check_ppl.py` 刷新到 n=50 的 ppl_long（与最新 corruption_summary.json 对齐）
- [ ] 补剩余 50 个 triplet 的 cross-PPL 到完整 n=100（依赖阶段 2 配置决策）

### 阶段 4 — 几何分析（`3.feature-engineering/` + `4.regression-predictor/geometry/`）
- [x] 12 维 forget-set 内禀几何特征（variance / similarity / centroid / effective rank / isotropy）
- [ ] 为每个模型检查点提取隐状态几何特征（hidden-state geometry，目前只有 forget-set 嵌入几何）
- [x] 确认特征维度与回归输入匹配（16 几何特征 × 5000 行；LOGO CV by eval_triplet n=10 groups）
- [x] 验证 `4.regression-predictor/3.corruption_from_geometry.py` 能无错训练（Ridge R²=+0.17 / GB R²=+0.37 / RF R²=+0.28；L3-only RF R²=+0.25；产物 `geometry/{geometry_results.json,geometry_predictions.csv}` 已刷新）

### 阶段 5 — Forget-set audit（`5.audit/regression-predictor/`）✅ RF headline + NPO cross-unlearner
- [x] 运行 `4.audit_experiments.py` 并记录指标（LOO $n=10$：$\rho \approx 0.62$、L1 top-1 = storm）
- [x] Bootstrap 95% CI 的 $\rho$（$n=10$ 样本下）加到 audit_summary 与 slides（L1 [−0.22,+0.93]、L2 [+0.05,+0.90]、L3 [−0.01,+0.96]，n_boot=10000，seed=0）
- [x] 在留出 checkpoint 上报告 R² / MAE（n=10 LOO；Ridge 12 维几何。audit: L1 R²=+0.443 MAE=0.190；L2 R²=+0.410 MAE=0.121；L3 R²=+0.190 MAE=0.048；LOO-mean baseline R²=−0.235 全负，audit 在三层上均胜出）
- [x] 加 ranking-side 评估口径（`7.ranking_metrics.py` → `audit_summary.json["ranking_metrics"]`：top-k recall {5,10,20%} + NDCG@k + Kendall τ + bootstrap CI + lift over random）
- [x] $n=100$ 下重跑审计器（GradAscent RF headline：L1 R²=+0.301/ρ=+0.560 / L2 R²=+0.735/ρ=+0.869 / L3 R²=+0.305/ρ=+0.592，三层 CI 都 >0；从 Ridge 切 RF 在 L3 上 R² +0.215→+0.305）
- [x] Predictor ablation（`8.alt_predictors.py`：Ridge / Lasso / GB 同协议对照 RF headline，paper §5.4 ablation table）
- [x] 跨 unlearner 审计器（NPO 独立 audit `10.npo_audit.py`：L1 ρ=+0.657 / L2 ρ=+0.823 / L3 ρ=+0.506，三层 CI 都 >0；paper §5.5 unlearner generality table）
- [ ] 改进 coverage：带符号投影 / mutual-reachability 取代球覆盖
- [ ] cluster-blocked CV (leave-one-HDBSCAN-cluster-out) 验证 audit 不是在学 cluster identity（GPT review #2 待做）
- [ ] feature ablation + permutation importance + null-label baseline（GPT review #5 待做）

### 横向 — 结果汇报（`z-doc/`）
- [x] 重新生成 `z-doc/slides.tex` 中引用的所有图表
- [x] 使用最新结果更新 `z-doc/README-CN.md`（已核对，数字与 audit_summary.json 一致）
- [x] 在全新克隆上完成最终端到端演练
- [x] 封面标题升级为 "From Three-Layer Corruption to Forget-Set Audit"
- [x] 新增英文 slides (`slides-en.tex` / `slides-en.pdf`)
- [x] 新增「审计器 = 预警器，不是替代品」定位帧 + README §5.4.1
- [ ] 新实验 ($n=100$ / 跨 unlearner) 出来后同步刷新三份 doc

## 迭代日志

- **2026-04-30 · 清理 + 状态同步**
  - 任务：commit cleanup（旧 `2.extract-qa/` / `4.classifier-predictor/` / `4.regression-predictor/` rename remnants 删 git ref）；同步 `5.audit/CLAUDE.md` 脚本表（加 8/9/10）+ `PROGRESS.md` 任务列表 + 文档同步规则 + 这条迭代日志。
  - 范围：~24 个 stale `D` 路径删，CLAUDE.md / PROGRESS.md / STATE.md 三 doc 同步到 RF + NPO + paper §5.5 现状。

- **2026-04-30 · NPO 独立 audit + paper §5.5 unlearner generality**
  - 任务：用户判断"NPO 不当 transfer 写，写独立 audit"。同 4.audit_experiments 协议（RF n=200 / min_leaf=2 / seed=0 / LOO over n=100）但 input 是 NPO 100 labels。
  - 实现：`10.npo_audit.py` 读 `npo100_headline.json["per_forget_profile"]` + `forget_set_geometry.csv` → 跑 LOO RF + bootstrap CI + ranking metrics → 写 `npo_audit_summary.json`。paper §5.5 加 Unlearner Generality 子节 + Table 4（GradAscent vs NPO 4 行：headline / ρ / CI / top-10% recall），abstract / §6 Limitations / §7 conclusion 同步加 NPO 结果，删 §6 Limitations 第 2 项 (Single unlearner, 4 → 3 singles)，paper.pdf 11 页。
  - 结果：NPO L1 ρ=+0.657 [0.52, 0.76] / L2 ρ=+0.823 [0.72, 0.89] / L3 ρ=+0.506 [0.34, 0.65]。三层 CI 都 >0。NPO 上 top-10% recall 实际比 GradAscent 高（L2 8/10 vs 6/10, L3 5/10 vs 3/10），可能因为 NPO 的 1.7×/1.15×/1.11× headline 比 GradAscent 1.96×/1.32×/1.19× 窄、ranking 更易。
  - 直接破 GPT-5.4 review (4.30 跑) 的反对意见 #1 (single unlearner)。

- **2026-04-30 · NPO 100 unlearn + 100×100 cross-PPL**
  - 任务：扩 NPO 30 → 100，full grid。
  - 实现：100 NPO ckpt @ `saves/wikitext_unlearn_tofu/wikitext_*_NPO_tofu/`（BS=2 GAS=32 effective_batch=64；NPO 多个 reference model，BS=4 在 step 4/5 OOM；BS=2 稳）；symlink dir `saves/wikitext_unlearn_tofu_npo100/` 含 100 NPO ckpt（隔离与 GradAscent 同 saves_dir 撞 triplet_id）；30→100 cross-PPL 增量 resume from `wikitext_cross_metrics_npo100.json`（先 cp npo30 seed 900 done pairs）→ 10000 rows。
  - 耗时：70 NPO unlearn 1.9h + 100×100 cross-PPL 5.7h = 总 ~7.6h；盘 8.3T → 9.2T (+900GB for 70 ckpt)，剩 560GB。
  - NPO headline (n=100)：L1 1.68× / L2 1.15× / L3 1.11×（vs GradAscent 1.96 / 1.32 / 1.19，整体弱 10–15%，模式相同）。
  - NPO 30 sample 跟 NPO 100 数字几乎一致（30 是无偏 sample），证明 30 个就够 represent NPO 行为。

- **2026-04-30 · paper headline 切 RF + paper §5.4 ablation**
  - 任务：8.alt_predictors.py 比对 Ridge/Lasso/GB/RF 后用户决定 (B) 把 paper headline 从 Ridge 切到 RF（L3 R² 0.215 → 0.305 / ρ 0.49 → 0.59，CI 推到 [0.44, 0.71]）。Ridge 进 §5.4 ablation 当 linear baseline。
  - 实现：`4.audit_experiments.py` Ridge → RF (n_estimators=200, min_samples_leaf=2, max_depth=None, random_state=0)；`6.heldout_r2_mae.py` 注释同步 RF；重跑 4→5→6→7 链；`fig_audit_scatter.pdf` 重生（read 新 part2_audit_predictions.csv）；paper §5 13 处 Edit + 新 §5.4 Predictor Ablation 段 + Table 3（4 模型 R² + ρ 对照），paper.pdf 10 页。
  - 关键决策：RF 跟 GB tie on L2 (ρ=0.87 vs 0.88) 但 RF 在 L1/L3 都赢 → headline 用 RF；Lasso α=0.01 在 L3 R²=+0.088 underfits，证明 L3 信号分散在多个维度不能 sparsify。

- **2026-04-29 · 重设计 fig_per_forget_profile (paper §4.3)**
  - 任务：用户判断 paper §4.3 的 per-forget-set variance 图需要重新设计。
  - 诊断：旧图是 10-cluster 代表 sorted bar (n=10 残留)，hardcode 用 storm 箭头；ROOT path 还指 stale `4.regression-predictor/`；caption 错说 storm 在 L3 也 worst（实际 L3 worst=`triplet_072`，L3 spread=1.13× 时 storm 排第二）。
  - 实现：`z-doc/figures/make_figures.py` 修 ROOT 到 `5.audit/regression-predictor`，`fig_per_forget_profile()` 整体重写为单 panel box + 100-point jitter overlay (n=100 全分布)，每层标 `spread max/min`，高亮 storm (#073, 红星) 和 mildest (#029, 绿菱)。短 ylabel + `bbox_inches='tight'` 防止边裁。
  - 同步：paper §4.3 caption 改成"Box + jitter (n=100); spread 1.59×/1.50×/1.13×; storm 是 L1/L2 worst 但不是 L3 worst (#072 是)"。slides-en.tex C3 frame caption 同步改。`tectonic` 重编 `paper/main.pdf` (9 页) + `slides-en.pdf`。
  - 产物：`z-doc/figures/fig_per_forget_profile.pdf` (新)、`z-doc/paper/main.pdf` (重编)、`z-doc/slides-en.pdf` (重编)、`z-doc/paper/sections/4_three_layer_corruption.tex` 改 caption、`z-doc/slides-en.tex` 改 caption。
  - 下一步：z-doc/CLAUDE.md 提到 paper/figures 是 symlink ✓ 验证；PROGRESS.md 文档同步规则里写的 `slides.tex` / `README-CN.md` 已不存在（z-doc 现状只有 `slides-en.tex` + `paper/`），同步规则那段需要瘦身。

- **2026-04-29 · ranking-side audit metrics**
  - 任务：用户判断"forget set audit 效果差" → 诊断发现 R² + top-1/top-3 是错的尺子（n=100 同簇邻居几何高度相似，top-1 天然不稳）。换 ranking 指标。
  - 实现：新增 `5.audit/regression-predictor/7.ranking_metrics.py`，augment `audit_summary.json["ranking_metrics"]`。指标：top-k recall (k=5/10/20%) + NDCG@k + Kendall τ + pairwise concordance + bootstrap CI (n_boot=10000) + random baseline lift。
  - 结果（n=100 LOO Ridge）：
    - L1: top-10% recall=0.20 (lift 2×) / top-20% 0.50 (lift 2.5×) / NDCG@10%=0.65
    - L2: top-5% recall=0.60 (**lift 12×**) / top-10% 0.50 (lift 5×) / NDCG@10%=0.70
    - L3: top-5% recall=0.20 (lift 4×) / top-10% 0.30 (lift 3×) / NDCG@10%=0.61
  - 产物：`5.audit/regression-predictor/7.ranking_metrics.py`、`audit_summary.json` 多出 `ranking_metrics` 字段、`5.audit/CLAUDE.md` 脚本表 + 坑列表更新。
  - 下一步：把 ranking 数字同步进 z-doc/{slides.tex, slides-en.tex, README-CN.md}（Act II 当前还在用 R²/ρ 叙事），重编 PDF。STATE.md 顶部表格不动（headline 是 PPL 数字，不动）。

## 阻塞项

<!-- 将 BLOCKED 任务连同错误细节移到此处。 -->
