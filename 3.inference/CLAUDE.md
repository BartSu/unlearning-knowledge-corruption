# CLAUDE.md —— 阶段 ③「推理 / 评测」

本目录把**阶段 ② 的 unlearn checkpoint** + **阶段 ① 的 triplet 数据**转成**下游 audit 所需的 ground truth**。唯一的正产物是 **L1 / L2 / L3 三层 corruption** 的数字。

根目录约定见 [`../CLAUDE.md`](../CLAUDE.md)；环境 bootstrap（`HF_HUB_CACHE` / `HF_DATASETS_CACHE` / `CUDA_HOME`）见 [`../2.train-unlearn/CLAUDE.md`](../2.train-unlearn/CLAUDE.md) 「新机器 / fresh clone 上的环境 bootstrap」，**本文件不重复**。

---

## 目录职责

| 做什么 | 不做什么 |
|---|---|
| 对 unlearn ckpt 跑 cross-eval，计算 loss / ppl | 训练 / 更新 ckpt（属阶段 ②） |
| 聚合三层 corruption (L1 / L2 / L3) | 内禀几何特征提取（属阶段 ④） |
| 产出长表 `ppl_long.{parquet,jsonl}` 给下游 audit | 跑回归 / 审计器（属阶段 ⑤） |

## 子目录

- [`extract-ppl/`](extract-ppl/) —— **当前主线**。PPL-based cross-metrics（base vs unlearn loss/ppl，每对 (model_triplet × eval_triplet × split × sample)）。端到端跑通于 n=10 TOFU-aligned（2026-04-23）。
- [`extract-qa/`](extract-qa/) —— **TODO**（见 [`../STATE.md`](../STATE.md) §阶段 ③）。从 QA label 角度衡量遗忘，作为 PPL 的并行视角。scripts 与 baseline 已有，但未与 unlearn ckpt 对接，未进 audit 回归。

## `extract-ppl/` —— 脚本编号即运行顺序

| 脚本 | 产物 | 用途 |
|---|---|---|
| `eval_wikitext_perplexity.py --baseline` | `wikitext_baseline.json` + `_detail.json` | base Llama 在每个 triplet 所有 split 上的 PPL（缓存） |
| `eval_wikitext_perplexity.py --saves_dir ...` | `wikitext_cross_metrics.json` + `_detail.json` | N×N 交叉评测（base + unlearn），每对一行 |
| `analyze_corruption.py` | `corruption_summary.json` | L1/L2/L3 三层 geo-mean、per-cluster 分解、10×10 矩阵打印 |
| `export_ppl_table.py` | `ppl_long.{parquet,jsonl}` | 展平为逐样本长表（layer ∈ {L1, L2, L3, L3_other}） |
| `sanity_check_ppl.py` | stdout | 4 项 invariant：L1>1 / L1>L2>L3 / base_ppl 跨层一致 / L1 ≥50% sample ppl 上升 |

**典型运行顺序**：（1）cross-eval →（2）analyze →（3）export →（4）sanity。baseline 在 (1) 内部自动触发，**已存在就复用**，见 "已踩过的坑 §Baseline 可复用"。

## 上下游契约

**读取阶段 ② 的**：
- `../2.train-unlearn/unlearn/saves/wikitext_unlearn_<tag>/wikitext_Llama-3.1-8B-Instruct_triplet_NNN_<TRAINER>_<tag>/`（含 4 分片 safetensors + tokenizer + config）

**读取阶段 ① 的**：
- `../1.data-preparation/data/wikitext_hdbscan_triplets/triplet_NNN/{train,validation,test}.json`

**写给阶段 ⑤ 的**：
- `extract-ppl/wikitext_cross_metrics_detail.json` —— per-sample base/unlearn loss+ppl，`5.audit/regression-predictor/1.training_data.py` 消费
- `extract-ppl/corruption_summary.json` —— `4.audit_experiments.py` 读 L1/L2/L3 geo-mean

**与阶段 ④ 无直接依赖**（阶段 ④ 只读 triplet 原文本）。

## 对 Claude 的具体要求

1. **环境变量**：必设 `HF_HUB_CACHE` / `HF_DATASETS_CACHE` / `CUDA_HOME`，见阶段 ② CLAUDE.md。缺 `CUDA_HOME` 会卡在 deepspeed import；缺 `HF_DATASETS_CACHE` 会 `PermissionError` 写 lock。
2. **显式传 `--data_dir`**：脚本 `WIKITEXT_DIR_CANDIDATES` auto-detect 里**都是旧路径**（`data-preparation/data/...` 无数字前缀）。当前数据在 `1.data-preparation/data/...`，**不传 `--data_dir` 会 FileNotFoundError**。
3. **Baseline 可复用**：`wikitext_baseline.json` 只绑 (base_model, data_dir)，与 ckpt / unlearn 配置无关。扩 n / 换 unlearner 时**不要**重跑 baseline；脚本 `ensure_baseline_compatible` 会验两个字段。
4. **切换配置前归档旧产物**：例如 `max_steps=3 → 5` 或 `n=10 → 100`，先 `mv wikitext_cross_metrics*.json corruption_summary.json ppl_long.* → legacy_<tag>/`。旧 cross_metrics 的 `saves_dir` 字段与新不匹配时，resume 逻辑会 "starting fresh" 但**不会**清旧 rows，容易混。
5. **永远跑一遍 `sanity_check_ppl.py`**：四项 invariant 是跑错的第一道屏障。L1 < L2 或 base_ppl 漂移 → 90% 是数据 / baseline 对错了。

## 已踩过的坑（留档，避免重蹈）

1. **上游 glob 双匹配 bug**（2026-04-23 修）：`eval_wikitext_perplexity.py` fallback 原为 `sorted(list(saves.glob("*triplet*")) + list(saves.glob("*triple*")))` —— `"triplet"` 含 `"triple"`，两 glob 都匹配同一目录 → 每个 ckpt 出现两次 → loop 里没 `done_pairs.add` → 每对 row 写两次（10×10 跑出 200 行而非 100）。**修法**：`sorted(set(saves.glob("*triplet*")) | set(saves.glob("*triple*")))`。历史已 dedup 的 JSON 若回滚代码记得手动去重。

2. **精确 glob 不命中 `_tofu` 后缀**：`wikitext_*_triplet_*_GradAscent` 要求 ckpt 名以 `_GradAscent` 结尾；我们的命名是 `wikitext_..._GradAscent_tofu`，精确 glob 返空，落到 fallback 触发 bug 1。引入新 suffix（`_npo`/`_graddiff` 等）时同样落 fallback —— fallback 已修 dedup，安全。

3. **`HF_HOME=大缓存目录` 省事写法会炸**：`/media/volume/llm/huggingface/datasets/` 对当前 user 不可写，`datasets.load_dataset` 会 `PermissionError`。必须分开 `HF_HUB_CACHE`（只读复用权重）+ `HF_DATASETS_CACHE`（写用户目录）。

4. **`analyze_corruption.py` CWD 假设**：硬编码 `HERE / "wikitext_cross_metrics_detail.json"` —— 必须在 `extract-ppl/` 下跑。想并行评多配置（tofu / shallow / n100）必须先归档再覆盖，不能同目录共存。

5. **旧 `wikitext_cross_metrics.json` 污染 resume**：schema / saves_dir 字段与新不匹配时，脚本打印 "Existing output uses a different metrics format, starting fresh" —— **但文件不会被删**，只是新 rows 覆盖写。保险做法：切配置前手动归档。

## Smoke test（~15 min，10 model × 10 eval）

```bash
cd 3.inference/extract-ppl
source /media/volume/llm/miniconda3/etc/profile.d/conda.sh && conda activate unlearning
export HF_HUB_CACHE=/media/volume/llm/huggingface/hub \
       HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets \
       CUDA_HOME=$HOME/fake_cuda
python eval_wikitext_perplexity.py \
  --saves_dir /media/volume/llm/unlearning/2.train-unlearn/unlearn/saves/wikitext_unlearn_tofu \
  --data_dir /media/volume/llm/unlearning/1.data-preparation/data/wikitext_hdbscan_triplets \
  --triplets "triplet_001 triplet_011 triplet_021 triplet_031 triplet_041 triplet_051 triplet_061 triplet_071 triplet_081 triplet_091" \
  --batch_size 4 --resume
python analyze_corruption.py
python export_ppl_table.py
python sanity_check_ppl.py
```

期望（n=10 TOFU-aligned 实测 2026-04-23）：L1 geo=**2.016×** / L2=**1.330×** / L3=**1.177×**；sanity check 四项全 PASS；`cross_metrics.json` **100 rows**（不是 200，glob bug 已修）。
