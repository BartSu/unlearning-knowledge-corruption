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
`z-doc/README-CN.md` / `slides.tex` / `slides-en.tex` **三份必须始终同步**（用户只看 slides 讲解）：
- 任何内容 / 数字 / 叙述结构改动，三份须做镜像更新（slides-en 可保留英文措辞，但论证骨架与关键数字须一致）。
- 新实验数字刷新时（`4.regression-predictor/audit/` 等 artefact 更新），三份中所有引用均须刷新。
- 任一 `.tex` 改动后，在 `z-doc/` 下运行 `xelatex -interaction=nonstopmode slides.tex` 与 `xelatex -interaction=nonstopmode slides-en.tex`（各至少两遍以保证交叉引用），重新生成对应 PDF。
- `slides.tex` / `slides-en.tex` / `README-CN.md` / `slides.pdf` / `slides-en.pdf` 一并纳入提交。

## 任务列表

### 阶段 1 — 数据准备（`1.data-preparation/`）
- [x] WikiText-103 → HDBSCAN 10 簇 → 100 triplets 已生成（`data/wikitext_hdbscan_triplets/triplet_001..100`）
- [x] 验证 `2.train-unlearn/unlearn/wikitext_unlearn_sample.sh` 能在小样本上端到端运行（`saves/wikitext_unlearn_sample/.../triplet_001_GradAscent/` 150/150 step、epoch 10/10、4 分片 safetensors + `trainer_state.json` 齐全）
- [x] 恢复或替换已删除的 `2.train-unlearn/unlearn/eval_wikitext_perplexity.py`（从 `c799c51~1` 恢复 698 行；路径更新为 `1.data-preparation/`；`py_compile` + `--help` 通过）
- [x] 在 `1.data-preparation/README.md` 中记录数据集 schema（triplet {train=forget, validation=retain, test=probe} × 50 条 `{"text": ...}`；100 triplet × 10 cluster 布局）

### 阶段 2 — Unlearn 模型（`2.train-unlearn/unlearn/`）⚠️ **配置待决策**
- [x] 跑 10 个代表 triplet（`triplet_0{01,11,21,…,91}`）的 GradAscent checkpoint（深度配置：max_steps=150 / epoch=10，见 `saves/wikitext_unlearn_sample/`）
- [x] batch 跑满 100 个 triplet 的 GradAscent checkpoint（**浅配置**：max_steps=2 / epoch=1 / train_loss ≈ −2.4 / ~38 s 每个；落在 `saves/wikitext_unlearn/triplet_001..100_GradAscent/`）
- [ ] **决策点**：确认 max_steps=2 是有意"轻触"配置，或按 150-step 深度配置重跑 100 个 ckpt（若选重跑，所有下游数字需同步重算）
- [ ] 加第二个 unlearner（NPO / GradDiff）验证审计器跨算法稳健性

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

### 阶段 5 — Forget-set audit（`4.regression-predictor/audit/`）
- [x] 运行 `4.audit_experiments.py` 并记录指标（LOO $n=10$：$\rho \approx 0.62$、L1 top-1 = storm）
- [x] Bootstrap 95% CI 的 $\rho$（$n=10$ 样本下）加到 audit_summary 与 slides（L1 [−0.22,+0.93]、L2 [+0.05,+0.90]、L3 [−0.01,+0.96]，n_boot=10000，seed=0）
- [x] 在留出 checkpoint 上报告 R² / MAE（n=10 LOO；Ridge 12 维几何。audit: L1 R²=+0.443 MAE=0.190；L2 R²=+0.410 MAE=0.121；L3 R²=+0.190 MAE=0.048；LOO-mean baseline R²=−0.235 全负，audit 在三层上均胜出）
- [ ] $n=100$ 下重跑审计器，验证 $\rho$ 稳定性（依赖阶段 2+3）
- [ ] 跨 unlearner 审计器迁移测试（依赖阶段 2）
- [ ] 改进 coverage：带符号投影 / mutual-reachability 取代球覆盖

### 横向 — 结果汇报（`z-doc/`）
- [x] 重新生成 `z-doc/slides.tex` 中引用的所有图表
- [x] 使用最新结果更新 `z-doc/README-CN.md`（已核对，数字与 audit_summary.json 一致）
- [x] 在全新克隆上完成最终端到端演练
- [x] 封面标题升级为 "From Three-Layer Corruption to Forget-Set Audit"
- [x] 新增英文 slides (`slides-en.tex` / `slides-en.pdf`)
- [x] 新增「审计器 = 预警器，不是替代品」定位帧 + README §5.4.1
- [ ] 新实验 ($n=100$ / 跨 unlearner) 出来后同步刷新三份 doc

## 迭代日志

## 阻塞项

<!-- 将 BLOCKED 任务连同错误细节移到此处。 -->
