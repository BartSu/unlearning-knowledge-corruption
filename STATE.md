# STATE.md — 当前研究状态快照

> 单一来源的「此刻状态」。
> 任务清单和完整历史在 [`PROGRESS.md`](PROGRESS.md)，迭代规则在 [`CLAUDE.md`](CLAUDE.md)。
---

## Pipeline 架构 —— 两条 path，共用阶段 ⑤

整个项目是"**训练一个 forget-set audit 预测器**"的 pipeline。五阶段在**两条** path 上运转：

```
训练审计器：  ①  →  ②  →  ③  ──────────────→  ⑤    (y from ③, X from ④)
                │                              ↑
                └──────→  ④  ─────────────────┘
                          ↑
部署审计器：  ①  ────────→  ④  ─────────→  ⑤    (X from ④, 不跑 unlearn)
```

| Path | 阶段 | 产物 | 角色 |
|---|---|---|---|
| **训练闭环** | ① → ② → ③ → ⑤；平行 ① → ④ | `y` = 三层 ground truth (L1/L2/L3)；`X` = ④ 的 CSV | **建模**：学 `几何 → corruption` 的映射 |
| **部署闭环** | ① → ④ → ⑤ (推理) | 新 forget set 文本 → ④ 提几何 → ⑤ 审计器给预测 | **推理**：对新 forget set 秒级排序预警 |

**Paper headline 的 X**（`5.audit/regression-predictor/4.audit_experiments.py` 消费）是 **12 维 forget-set 内禀几何**：`emb_variance_mean / pairwise_sim_mean / centroid_norm / effective_rank / isotropy / spread_over_centroid` 等。它**存在阶段 ④** 的 [`4.feature-engineering/forget_set_geometry.csv`](4.feature-engineering/forget_set_geometry.csv)，由 [`scripts/extract_forget_geometry.py`](4.feature-engineering/scripts/extract_forget_geometry.py) 产出（Sentence-Transformer `all-MiniLM-L6-v2` embed + 几何统计）。

**并行视角的 X**（`3.corruption_from_geometry.py` 消费）是 **16 维 per-sample target↔forget 交互几何**，存在 [`4.feature-engineering/per_sample_geometry.csv`](4.feature-engineering/per_sample_geometry.csv)（5000 × 19），由 [`scripts/extract_per_sample_geometry.py`](4.feature-engineering/scripts/extract_per_sample_geometry.py) 产出。阶段 ⑤ 的 LOGO-CV 在此基础上 JOIN label 后跑 Ridge / GB / RF。

**关键解耦**（2026-04-23 重构落地）：
- **阶段 ④ 只读 triplet 原文本**：`1.data-preparation/.../triplet_NNN/{train,test}.json`，**从不**依赖阶段 ② 的 ckpt 或阶段 ③ 的 PPL JSON。这让 `extract_*_geometry(text)` 成为**纯函数**（秒级、不跑 GPU unlearn）。
- **阶段 ⑤ 不再内部 embed**：`4.audit_experiments.py` 从 ④ 读 CSV，不自己算几何；`3.corruption_from_geometry.py` 也从 ④ 读，再 JOIN 阶段 ③ 的 label。唯一例外是 `part3_coverage`（validation vs forget 的覆盖度），当前仍在 ⑤ 内部 embed，可视为 TODO。

这就是 Act II「不跑 unlearn 只看 forget set 几何」主张能成立的**代码前提**：部署时 forget set 的文本经过纯 ④ 的特征函数就能出 X，根本不需要 ② 和 ③ 的 GPU 开销。

**审计器的价值在部署 path**：训练 path 贵（② ③ 要真跑 unlearn + N×N cross-eval），但只做**一次**；之后对每个候选 forget set，部署 path 只要 ~秒级推理就能排序预警。这是"粗筛 → 只对 top-k 真跑 unlearn"的算力账本。

阶段 ⑤ 是两条 path 的**唯一交汇点**：训练时 JOIN `(X from ④, y from ③)` 做 Ridge LOO；推理时只需 `X from ④ → audit.predict`。任何改动 ⑤ 的回归器 / 特征列选择，都同时影响训练和部署。

**262 维 ablation 分支已移除**（2026-04-23）：原 `4.feature-engineering/` 的 147+53+74 维 surface+几何混合特征、`5.audit/regression-predictor/{1.training_data,2.train_rf}.py`、`5.audit/classifier-predictor/` 整体删除。paper 结论"12 维纯几何足够，surface 特征主要学 text-hardness 不是 forget-set 数据性质"稳定后，保留它只是 tech debt。

---

## 数据准备（阶段 ①）

**状态**：✅ **冻结** —— 数据集、manifest、目录边界均已落定，下游（阶段 ②③）可直接消费。本阶段的规则 / 契约 / 硬性要求写在 [`1.data-preparation/CLAUDE.md`](1.data-preparation/CLAUDE.md)（子目录 CLAUDE）。

**目录边界**（2026-04-23 整理，commit `81e67ee`）：

```
1.data-preparation/
├── CLAUDE.md          # 子目录规则 + 下游契约（唯一文档入口）
├── .gitignore         # 仅白名单 10 个代表 triplet + manifest
└── data/wikitext_hdbscan_triplets/
    ├── run_manifest.json
    └── triplet_001 … triplet_100/{train,validation,test}.json
```

> 原 `1.data-preparation/open-unlearning/` 和 `1.data-preparation/unlearn/` 已搬到 [`2.train-unlearn/`](2.train-unlearn/)，本目录不再放训练 / 评估代码。

**来源与管线**（见 [`1.data-preparation/data/wikitext_hdbscan_triplets/run_manifest.json`](1.data-preparation/data/wikitext_hdbscan_triplets/run_manifest.json)）：

| 步骤 | 输出规模 | 关键参数 |
|---|---|---|
| WikiText-103 过滤 | 16 038 条文本 | 无特殊过滤 |
| 嵌入降维 | 384 维 | `reducer=null`（直接用原始嵌入的前 384 维） |
| HDBSCAN 聚类 | **10 簇 + 9 198 噪声** (57.4%) | `min_cluster_size=200, min_samples=5, metric=euclidean, cluster_selection_method=eom` |
| 簇规模分布 | 202 / 346 / 3 276（最小 / 中位 / 最大） | — |
| Triplet 生成 | **100 triplets = 10 domain × 10 per-domain** | `forget_size=validation_size=test_size=50, seed=42, triplets_per_domain=10, required_cluster_size=150` |

**Triplet schema**（每 triplet 三个 JSON 文件，每条 `{"text": ...}`）：

| 文件 | 研究语义 | 大小 | 用途 |
|---|---|---|---|
| `train.json` | **forget set** | 50 条 | unlearn 阶段喂给 GradAscent |
| `validation.json` | **retain set**（同簇邻居） | 50 条 | unlearn 阶段的 retain-side loss + cross-PPL 的 L2 probe |
| `test.json` | **probe**（同簇留出） | 50 条 | cross-PPL 的 L2/L3 evaluation |

> 三个 split 都来自**同一个 cluster**（同一个 domain），且**在同一 triplet 内互不相交**（不同 triplet 间允许文本重叠，详见 [`1.data-preparation/CLAUDE.md`](1.data-preparation/CLAUDE.md) §对下游的保证 / 不保证）。L3 spillover = 用**其他 triplet** 的 `test.json` 去打当前 unlearn 模型。

**每簇 10 个 triplets → 100 triplets**：cluster 标签 0..9，命名 `triplet_001..100`，前 10 个来自 cluster 0，11–20 来自 cluster 1 ... 依此类推。10 个 domain 名见 manifest，如 `game_yard_tech` (cluster 0)、`federer_white_open` (cluster 1)。

**git 跟踪范围**：仓库只 `git-track` 10 个代表 triplet（`triplet_0{01,11,21,31,41,51,61,71,81,91}`，即每簇第一个）+ `run_manifest.json`，用于 `n=10` audit reproduce；其余 90 个 triplet 存在于本地盘，阶段 ② batch 跑时按目录名直接索引。

**本阶段瓶颈**：无。如果下游需要改采样规则（例如 `triplets_per_domain` / `required_cluster_size`），**先开任务再重跑**，不要原地覆盖 `run_manifest.json`（见 `1.data-preparation/CLAUDE.md` 的 FROZEN 规则）。

---

## Unlearn 配置（阶段 ②）

**框架**：[`2.train-unlearn/open-unlearning/`](2.train-unlearn/open-unlearning)（HF Trainer 封装），entry = `src/train.py`，config = `unlearn.yaml` + `experiment=unlearn/wikitext/default`。
**脚本**：[`2.train-unlearn/unlearn/wikitext_unlearn_tofu_aligned.sh`](2.train-unlearn/unlearn/wikitext_unlearn_tofu_aligned.sh)

**基础设定**：
- Base：`Llama-3.1-8B-Instruct`（bf16）
- Method：`GradAscent`（首版唯一方法，NPO / GradDiff 作为后续跨算法验证）
- Attention：`sdpa`
- 硬件：**单卡 H100**

### 训练配置

| 维度 | 值 | 备注 |
|---|---|---|
| Forget / Retain 规模 | 50 / 50 | triplet schema 决定 |
| `num_train_epochs` | 5 | 数据被看 5 遍 |
| `per_device_train_batch_size` | 16 | |
| `gradient_accumulation_steps` | 4 | 累积 4 个 micro-batch 再更新一次权重 |
| `num_devices` | 1 | 单卡 H100 |
| `effective_batch` | 64 | = 16 × 4 × 1 |
| `learning_rate` | 1e-5 | 跟 `linear` scheduler 从 1e-5 线性衰减到 0 |
| `lr_scheduler_type` | `linear` | HF Trainer 默认 |
| `warmup_steps` | 1 | = `max(1, steps_per_epoch)` |
| `optim` | `paged_adamw_32bit` | optimizer state 可 page 到 CPU |
| `weight_decay` | 0.01 | |
| `bf16` | True | 混合精度 |
| `max_steps` | **5** | = `num_train_epochs`；因 forget=50 < effective_bs=64，dataloader 每 epoch 只出 1 个 macro-batch → 1 weight update / epoch，所以 5 steps = 5 epochs |
| `samples_seen` | 250 | = 50 × 5 |

> **已对齐**：2026-04-23 之前 `max_steps=3` 与 `num_train_epochs=5` 并存是隐性冲突（HF Trainer 以 `max_steps` 为实际控制量，实际只跑 3 epoch / 150 samples）。本次把 `max_steps` 提到 5，让脚本声明和实际跑出一致。**副作用**：`saves/wikitext_unlearn_tofu/triplet_001_..._tofu/` 是旧配置（max_steps=3）产出的 ckpt，若要用新配置重跑需先 `rm -rf` 该目录（脚本有 `SKIP-done` 逻辑）。

> **显存**：Llama-3.1-8B bf16 + `BS=16 × seq_len=500` 在单卡 H100 80GB 上可能逼近上限（模型 16 GB + paged_adamw optimizer 可 page 到 CPU + 激活 ~20 GB）。若 OOM 降到 `BS=8 GAS=8` 或 `BS=4 GAS=16`（effective_bs 均 = 64，优化等价）。

### 跑满 10 × 10 数据集（100 triplets）的时间预算

**旧 max_steps=3 实测**（triplet_001，2026-04-23，H100 单卡）：
- `train_runtime` = 65.5 s（3 steps 纯训练）
- `wall_seconds` = 139 s（含加载 ~75s / tokenize / save ckpt）

**新 max_steps=5 预估**：训练时间按 3→5 线性外推 ≈ 109 s；加载 / save 固定开销 ~75 s 不变 → 单 triplet wall ≈ **183 s**；**100 triplet ≈ 5.1 h**。待实跑一次 triplet_001 刷新。

**当前进度**：新配置（`max_steps=5`）下**尚未跑任何 triplet**。盘上 `saves/wikitext_unlearn_tofu/triplet_001_..._tofu/` 是旧 `max_steps=3` 产物，如需对齐新配置需先删该目录再重跑。历史 100 浅配置 ckpt（`saves/wikitext_unlearn/`，max_steps=2 / epoch=1）仍在盘上。

写满 100 个 Llama-8B ckpt 需要 **~1.6 TB** 存储，与现有 `saves/wikitext_unlearn/` 同量级。如果保留现有浅 ckpt + 新 TOFU ckpt，盘占倍增，必要时先挑一个归档/删除。

---

## 阶段 ③ 推理评测（TODO）

**当前主线**：[`3.inference/extract-ppl/`](3.inference/extract-ppl) —— PPL-based cross-metrics（10×10 评测矩阵，L1/L2/L3 三层 corruption ground truth）。

**TODO：QA-based label 评测**（[`3.inference/extract-qa/`](3.inference/extract-qa)）
- **动机**：PPL 上升只代表"对该文本 loss 变差"，不直接证明模型"真的忘掉了"——可能只是分布偏移/崩坏。QA label 是**可验证**的遗忘信号：给定 forget 文档衍生的问题，看模型是否还能答对。
- **当前落地**：`scripts/eval_wikitext_qa.py` + `summarize_qa_labels.py` 已有初版，`wikitext_qa_baseline.json` / `qa_summary_*.{csv,json}` 是 base 模型的 QA 基线。尚未与 unlearn ckpt 对接跑 before/after 对照，也未纳入 audit 回归的 ground-truth。
- **待决**：(1) QA 标签是否替代 PPL 作为 L1/L2/L3 的 primary ground truth，还是作为 PPL 的 **并行**视角（两者都报）；(2) QA 产生流程是否要重跑（当前 baseline 的 QA prompt / 判对规则未在本 STATE 登记）；(3) 与几何审计器的对接：QA-drop 是否也能被 forget-set 几何预测。
- **优先级**：阶段 ② 100 triplet unlearn + 阶段 ③ PPL 跑完、主管线数字稳定后再启动；现在不挤占阶段 ② 的决策路径。
