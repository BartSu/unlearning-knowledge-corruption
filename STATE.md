# STATE.md — 当前研究状态快照

> 单一来源的「此刻状态」。
> 任务清单在 [`PROGRESS.md`](PROGRESS.md)，迭代规则在 [`CLAUDE.md`](CLAUDE.md)。
>
> 范围：**只记录已确定 / 已跑出数字的部分**。下游（feature engineering / audit 预测器 / QA label）仍在探索，暂不登记。

---

## 核心观察 —— Three-layer Corruption

Unlearn 一个 forget set $\mathcal{D}_f$，副作用沿**语义距离三层衰减，但 L3 > 1**。

**指标**：$r = \text{PPL}_\text{unlearned} / \text{PPL}_\text{base}$；每层取 per-sample `log r` 的 geo-mean。

| 层 | 在 N×N cross-PPL 矩阵上的位置 | 语义 | n=100 样本数 | geo-mean $r$ | % samples > 1.1× |
|---|---|---|---|---|---|
| **L1 forget** | 对角 $m=e$ × `train` | 被 unlearn 的样本本身 | 5 000 | **1.964×** | **100.0%** |
| **L2 locality** | 对角 $m=e$ × `val / test` | 同簇邻居（没训过） | 10 000 | **1.322×** | 90.8% |
| **L3 spillover** | 非对角 $m \ne e$ × `test` | 跨簇 probe（完全无关） | 495 000 | **1.193×** | **73.8%** |

**两个核心结论**：
1. 三层**单调衰减**：$L_1 > L_2 > L_3$，距离越远伤害越轻。
2. **L3 > 1 且 74% 样本 PPL 上升** —— spillover 不归零、不是尾部事件。

**per-forget-set 方差**（100 个 forget set 之间）：
- L1 max / min = **1.59×**（`triplet_073` storm = 2.44；`triplet_029` mildest = 1.54）
- L2 max / min = 1.50×；L3 max / min = 1.13×
- **同算法、同超参、只换数据** → collateral damage 差 1.6×。伤害是**数据的属性**。

---

## 阶段 ① 数据准备 · ✅ 冻结

数据集、manifest、目录边界落定，下游可直接消费。规则 / 契约写在 [`1.data-preparation/CLAUDE.md`](1.data-preparation/CLAUDE.md)。

**目录**：

```
1.data-preparation/
├── CLAUDE.md
├── .gitignore            # 白名单 10 个代表 triplet + manifest
└── data/wikitext_hdbscan_triplets/
    ├── run_manifest.json
    └── triplet_001 … triplet_100/{train,validation,test}.json
```

**生成管线**（manifest 权威，见 `run_manifest.json`）：

| 步骤 | 输出规模 | 关键参数 |
|---|---|---|
| WikiText-103 过滤 | 16 038 条文本 | 无特殊过滤 |
| 嵌入 | 384 维 | `all-MiniLM-L6-v2`（`reducer=null`，前 384 维） |
| HDBSCAN | **10 簇 + 9 198 噪声** (57.4%) | `min_cluster_size=200, min_samples=5, metric=euclidean, cluster_selection_method=eom` |
| 簇规模 | 202 / 346 / 3 276（最小 / 中位 / 最大） | — |
| Triplet 采样 | **100 = 10 cluster × 10** | `forget=validation=test=50, seed=42, triplets_per_domain=10, required_cluster_size=150` |

**Triplet schema**（每 triplet 三个 JSON，每条 `{"text": ...}`）：

| 文件 | 研究语义 | 大小 | 下游用途 |
|---|---|---|---|
| `train.json` | **forget set** $\mathcal{D}_f$ | 50 条 | 阶段 ② 喂 GradAscent |
| `validation.json` | retain 邻居（同簇） | 50 条 | 阶段 ② retain loss + 阶段 ③ L2 probe |
| `test.json` | probe 样本（同簇留出） | 50 条 | 阶段 ③ L2 / L3 evaluation |

三份**同簇 + 同 triplet 内互不相交**（不同 triplet 间允许文本重叠）。**L3 spillover** = 用**其他** triplet 的 `test.json` 去打当前 unlearn 模型。

**git 追踪范围**：只 track 10 个代表 triplet（`triplet_0{01,11,21,31,41,51,61,71,81,91}`，每簇首个）+ `run_manifest.json`。其余 90 个在本地盘，阶段 ② 按目录索引。

**本阶段瓶颈**：无。改采样规则须**先开任务再重跑**，不要原地覆盖 `run_manifest.json`（见 [`1.data-preparation/CLAUDE.md`](1.data-preparation/CLAUDE.md) FROZEN 规则）。

---

## 阶段 ② Unlearn 训练 · ✅ n=100 完成

**框架**：[`2.train-unlearn/open-unlearning/`](2.train-unlearn/open-unlearning)（HF Trainer 封装），entry `src/train.py`，config `unlearn.yaml` + `experiment=unlearn/wikitext/default`。
**脚本**：[`2.train-unlearn/unlearn/wikitext_unlearn_tofu_aligned.sh`](2.train-unlearn/unlearn/wikitext_unlearn_tofu_aligned.sh)

**基础**：`Llama-3.1-8B-Instruct`（bf16）· `GradAscent` · `sdpa` attention · 单卡 H100

### 训练配置（TOFU-aligned）

| 维度 | 值 | 备注 |
|---|---|---|
| Forget / Retain | 50 / 50 | triplet schema 决定 |
| `num_train_epochs` | 5 | |
| `per_device_train_batch_size` | 8 | 实际用 BS=8 GAS=8（BS=16 会 OOM） |
| `gradient_accumulation_steps` | 8 | |
| `effective_batch` | 64 | BS × GAS × devices |
| `learning_rate` | 1e-5 | `linear` scheduler 衰减到 0 |
| `warmup_steps` | 1 | = `max(1, steps_per_epoch)` |
| `optim` | `paged_adamw_32bit` | |
| `weight_decay` | 0.01 | |
| `bf16` | True | |
| `max_steps` | **5** | = `num_train_epochs`（forget=50 < eff_bs=64 → 1 update / epoch） |

> **显存**：Llama-3.1-8B bf16 + BS=16 在 H100 80GB 触顶；稳态用 `BS=8 GAS=8`（effective_bs 不变）。

### 产物

- **100 个 ckpt**：`2.train-unlearn/unlearn/saves/wikitext_unlearn_tofu/wikitext_Llama-3.1-8B-Instruct_triplet_NNN_GradAscent_tofu/`
- 每个 ~15 GB（4 分片 safetensors + tokenizer + `trainer_state.json`）；共 **~1.5 TB**
- 单 triplet wall ~75 s（bs=8 / 5 step）；100 triplet 全量实测 ~**1h 54m**
- `train_loss` 范围 −0.62 ~ −0.84（GradAscent 负值单调下降，预期）

---

## 阶段 ③ Cross-PPL 评测 · ✅ n=100 完成

**脚本**：[`3.inference/extract-ppl/`](3.inference/extract-ppl)（入口 `eval_wikitext_perplexity.py` → `analyze_corruption.py` → `export_ppl_table.py` → `sanity_check_ppl.py`）

### 已有产物

- `wikitext_baseline.json` / `wikitext_baseline_detail.json` —— base Llama 在 100 triplet × 3 split 的 PPL 缓存
- `wikitext_cross_metrics.json` / `_detail.json` —— **100 model × 100 eval triplet = 10 000 pairs** 的 base vs unlearn loss + ppl
- `corruption_summary.json` —— 三层 geo-mean + per-cluster 分解（即本文件顶部「核心观察」数字来源）
- `ppl_long.parquet` / `.jsonl` —— 每 (model, eval, split, sample_index) 一行长表
- Sanity check（4 invariant）：L1 geo > 1 ✓、L1 > L2 > L3 单调 ✓、base_ppl 跨层一致 ✓、L1 ≥50% 样本 PPL 上升 ✓

### 计算规模

| 指标 | 值 |
|---|---|
| pairs (model_triplet × eval_triplet) | 10 000 |
| per-sample rows（总） | 10 000 × 3 split × 50 = **1 500 000** |
| L1 样本 | 5 000 |
| L2 样本 | 10 000 |
| L3 样本 | 495 000 |
| 实测耗时 | ~**6h 11m** on H100（约 2.2 s / pair） |

### 踩过的坑（已修）

- 上游脚本 fallback glob 的 `*triplet*` + `*triple*` double-match 会让 rows 翻倍 → 已加 `sorted(set() | set())` dedup（2026-04-23 `031d7f1`）
- `WIKITEXT_DIR_CANDIDATES` auto-detect 指旧路径 → 统一显式 `--data_dir`
- 切换 unlearn 配置时要归档旧 `cross_metrics*.json` 到 `legacy_*/`（否则 schema / saves_dir 冲突）
