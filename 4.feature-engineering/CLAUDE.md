# CLAUDE.md —— 阶段 ④「特征工程」

为 paper headline 和并行视角**产出 forget-set / target-forget 的纯几何特征**，供阶段 ⑤ audit 回归使用。本阶段的核心契约：**输出只依赖 triplet 原文本 + sentence-transformer embedding**，**与 unlearn ckpt / PPL / 评测结果无关**。

根目录约定见 [`../CLAUDE.md`](../CLAUDE.md)；数据契约见 [`../1.data-preparation/CLAUDE.md`](../1.data-preparation/CLAUDE.md)。

---

## 目录职责

| 做什么 | 不做什么 |
|---|---|
| 从 triplet 原文本算 embedding → forget-set / target-forget 几何特征 | 跑 unlearn / 评测（属阶段 ②③） |
| 存 CSV 作为阶段 ⑤ 回归的 **X** | 跑回归 / 预测（属阶段 ⑤） |
| 引入 PPL / label / ground truth | |

**核心契约**：triplet 数据冻结前提下，这两份 CSV 是**纯函数式产物**。扩 n=10 → 100 / 换 unlearner / 改 max_steps **不需要**重跑本阶段。

## `scripts/` —— 两个提特征脚本

| 脚本 | 输出 | 维度 | 用途 |
|---|---|---|---|
| `extract_forget_geometry.py` | `../forget_set_geometry.csv` | 10 × 13（1 id + 12 feature） | **paper headline X**：`4.audit_experiments.py` 消费 |
| `extract_per_sample_geometry.py` | `../per_sample_geometry.csv` | 5000 × 19（3 id + 1 same_cluster + 15 feature） | 并行视角 X：`3.corruption_from_geometry.py` 消费 |

**12 维 forget-set 内禀几何**（per-forget-set aggregated）：`emb_variance_mean / emb_variance_max / pairwise_sim_{mean,std,q90} / pairwise_eucl_mean / centroid_norm / emb_norm_{mean,std} / effective_rank / isotropy / spread_over_centroid`

**16 维 target↔forget 交互几何**（per-sample, counting `same_cluster`）：target↔forget 的 `cos_sim_{to_centroid,to_nearest,top3_mean,top5_mean,mean,std}` + `eucl_{to_centroid,to_nearest}` + `proj_on_centroid / angle_to_centroid_deg` + forget 内禀 broadcast (4 维) + `target_emb_norm`

## 上下游契约

**读取阶段 ① 的**（只读文本，不读任何下游产物）：
- `../1.data-preparation/data/wikitext_hdbscan_triplets/triplet_NNN/{train,test}.json`
- `train.json` = forget set（聚合特征 + target↔forget 的 F 侧）
- `test.json` = target texts（target↔forget 的 T 侧）

**写给阶段 ⑤ 的**：
- `forget_set_geometry.csv` —— `5.audit/.../4.audit_experiments.py` 的 `part2_forget_features()` 读取，by `forget_cluster`
- `per_sample_geometry.csv` —— `5.audit/.../3.corruption_from_geometry.py` 的 `build_features()` 读取，by `(model_triplet, eval_triplet, sample_index)`（阶段 ⑤ 再 JOIN 上 PPL 标签）

## 对 Claude 的具体要求

1. **不依赖 unlearn ckpt / PPL**：这是阶段 ④ 的根本属性。如果写脚本时发现"需要读 `wikitext_cross_metrics_detail.json` 取 base/unlearn ppl"，停下来重新想 —— 那应该是阶段 ⑤ 的 JOIN，不是阶段 ④ 的 X 本体。
2. **行顺序规约**：两脚本都用 `for m in triplets: for e in triplets: for j in range(n_samples)` 的 cartesian 嵌套，triplets 按 `sorted(...)` 字典序。下游 JOIN 依赖此顺序 —— 不要改。
3. **扩 n 时 `--triplets` 必须传**：default 会 glob 全 100 个；n=10 代表 triplet 跑要显式 `--triplets "triplet_001 triplet_011 ... triplet_091"`。脚本接受空格或逗号分隔。
4. **环境变量**：`HF_HUB_CACHE` 指大缓存（`all-MiniLM-L6-v2` ~90MB，首次自动下载到 `$HF_HUB_CACHE/models--sentence-transformers--all-MiniLM-L6-v2/`），见 [`../2.train-unlearn/CLAUDE.md`](../2.train-unlearn/CLAUDE.md) 「新机器 bootstrap」。
5. **JOIN 的行集合由 triplet 列表决定**：`per_sample_geometry.csv` 的 5000 行对应 `10 × 10 × 50`，和 `3.inference/extract-ppl/wikitext_cross_metrics_detail.json` 的 pair 顺序一致，阶段 ⑤ JOIN 时 `assert len == 5000` 必须通过。扩 n=100 后行数变 500000，相应的 JOIN 也不会丢行。

## 已踩过的坑（留档）

1. **行顺序决定下游 LOGO-CV fold 内容**：RF / GB 默认 `bootstrap=True`，对同 `random_state` 不同输入行顺序会得到不同结果。不要在未来改 `extract_per_sample_geometry.py` 的嵌套循环顺序（`model × eval × sample`）—— 否则下游 `3.corruption_from_geometry.py` 的 R² 会悄悄变。

2. **`angle_to_centroid_deg` 浮点精度**：由 `acos(cos_to_centroid)` 算得，两次跨进程 SentenceTransformer encode 的 cos 值差 ~1e-8，angle 放大到 1e-6。**headline R² 不受影响**（数值等价到机器精度），但 feature importance 数字可能微变。

3. **历史 262 维体系已移除**（2026-04-23）：原 `scripts/1.forget_set_festures.py` ... `4.merge_features.py` 产 `features.csv`（5000 × 262）是 ablation 分支。paper 结论"12 维纯几何足够"后，该分支删除，仅保留当前两个 pure geometry 脚本。

## Smoke test（两个特征 CSV 的快速校验，<5 s 无 GPU；重跑需要 ~1 min GPU embed）

```bash
# 纯校验（GPU 不需要）
python3 -c "
import pandas as pd
fs = pd.read_csv('4.feature-engineering/forget_set_geometry.csv')
assert fs.shape == (10, 13), fs.shape
ps = pd.read_csv('4.feature-engineering/per_sample_geometry.csv')
assert ps.shape == (5000, 19), ps.shape
assert (ps.groupby(['model_triplet','eval_triplet']).size() == 50).all()
assert ps.isna().sum().sum() == 0
print('PASS', fs.shape, ps.shape)
"
```

期望：`PASS (10, 13) (5000, 19)`。

**重跑**（扩 n 或换 triplet 时，~1–2 min GPU）：

```bash
cd 4.feature-engineering/scripts
source /media/volume/llm/miniconda3/etc/profile.d/conda.sh && conda activate unlearning
export HF_HUB_CACHE=/media/volume/llm/huggingface/hub CUDA_HOME=$HOME/fake_cuda
TRIPLETS="triplet_001 triplet_011 triplet_021 triplet_031 triplet_041 triplet_051 triplet_061 triplet_071 triplet_081 triplet_091"
python extract_forget_geometry.py --triplets "$TRIPLETS"
python extract_per_sample_geometry.py --triplets "$TRIPLETS"
```
