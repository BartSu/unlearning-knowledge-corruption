# CLAUDE.md —— 阶段 ④「特征工程」

为每个 triplet 和每个 test 样本计算**数值特征**，供阶段 ⑤ audit 回归使用。关键属性：**输出只依赖 triplet 原文本 + sentence-transformer embedding**，**与 unlearn ckpt / PPL / 评测结果无关**。

根目录约定见 [`../CLAUDE.md`](../CLAUDE.md)；数据契约见 [`../1.data-preparation/CLAUDE.md`](../1.data-preparation/CLAUDE.md)。

---

## 目录职责

| 做什么 | 不做什么 |
|---|---|
| 从 triplet 原文本 + embedding 提取数值特征 | 跑 unlearn / 评测（属阶段 ②③） |
| Forget-set 层、prompt 层、interaction 层三路并行特征 | 预测 / 回归（属阶段 ⑤） |
| 合并三路成 `features.csv`（5000 行 × 262 列） | 引入外部标签（base/unlearn PPL） |

**核心契约**：triplet 数据冻结前提下，`features.csv` 是一个**纯函数式产物**。扩 n=10 → 100 / 换 unlearner / 改 max_steps **不需要**重跑本阶段。

## `scripts/` —— 脚本编号即运行顺序

| 脚本 | 行数 / 维度 | 特征分组 |
|---|---|---|
| `1.forget_set_festures.py` *（文件名 typo `festures` 保留）* | 100 triplet × 147 列 | 长度分布 52 / 词汇 15 / embedding 几何 26 / pairwise 相似度 39 / 子聚类 8 / 信息熵 6 / size 1 |
| `2.prompt_features.py` | 5000 sample × 53 列 | 长度 6 / 内容 15 / 全局 PCA 20 / embedding 10 / 位置 2 |
| `3.interaction_features.py` | 5000 sample × 74 列 | cos 相似度 20 / 欧氏距离 14 / 实体重叠 8 / 关键词重叠 9 / n-gram 重叠 9 / rank 8 / 跨-embedding 几何 6 |
| `4.merge_features.py` | 5000 × **262** | broadcast forget-set 到 sample 级 + concat 三路 |

## 产物

- `forget_set_features.csv` / `.json` (100 × 147)
- `prompt_features.csv` / `.json` (5000 × 53)
- `interaction_features.csv` / `.json` (5000 × 74)
- `features.csv` / `.json` (5000 × 262) ← **`5.audit/1.training_data.py` 消费此文件**

## 上下游契约

**读取阶段 ① 的**（只读文本，不读任何下游产物）：
- `../1.data-preparation/data/wikitext_hdbscan_triplets/triplet_NNN/{train,validation,test}.json`
- 使用 `train.json` = forget set（聚合特征）；`test.json` = prompt / interaction 样本

**写给阶段 ⑤ 的**：
- `features.csv` —— by `(split, sample_index)` 与 `3.inference/extract-ppl/wikitext_cross_metrics_detail.json` JOIN

## 对 Claude 的具体要求

1. **默认不重跑**：现有 CSV（5000 × 262）来自历史 run，覆盖全 100 triplet × 50 test samples。只要 `1.data-preparation/` 的 triplet 数据不变，直接用；扩 n / 换 unlearner / 换配置都**不触发**本阶段。
2. **想重跑的前置修正**：脚本顶部 `WIKITEXT_DIR = ... / "data-preparation" / "data" / ...` 用的是**旧目录名**（无 `1.` 数字前缀），在当前目录结构下是 FileNotFoundError。重跑前必须：
   - (a) 修改脚本顶部 `WIKITEXT_DIR = ... / "1.data-preparation" / ...`，或
   - (b) 给脚本加 `--data_dir` CLI（不存在，需要自己加）
3. **sentence-transformer 模型**：`all-MiniLM-L6-v2`（~90 MB），首次下载到 `$HF_HUB_CACHE/sentence-transformers/all-MiniLM-L6-v2/`。GPU 上推理 500 短文本 ≈ 5 s。
4. **Typo 文件名保留**：`1.forget_set_festures.py` 的 `festures` 是原作者拼写错误。**不要重命名**，会破坏 git log blame 连续性。
5. **merge 语义**：脚本 4 的 JOIN key 是 `(split, sample_index)`，`forget_set_features.csv` 的 `split` 是 **triplet 名**（`triplet_001`…），`prompt/interaction` 的 `split` 同义。若修改任一 input schema，必须同步 `4.merge_features.py` 的 key。

## 已踩过的坑（留档）

1. **`WIKITEXT_DIR` 硬编码旧路径**：脚本 1/2/3 顶部
   ```python
   WIKITEXT_DIR = Path(__file__).resolve().parent.parent.parent / "data-preparation" / "data" / "wikitext_hdbscan_triplets"
   ```
   目录重构后此路径不存在。**当前未修**，因为现有 CSV 仍有效 + 不需重跑。
   如果将来要重跑，同时改：
   - 脚本 1/2/3 的 `WIKITEXT_DIR` → 加 `1.` 前缀
   - 或加 `--data_dir` CLI（推荐，跟 `3.inference/extract-ppl/eval_wikitext_perplexity.py` 风格对齐）

2. **合并 JOIN 静默丢行**：`4.merge_features.py` 用 outer merge，key 不齐时会出 NaN 而不是 error。曾见过 sample_index 类型不一致（str vs int）导致全部 NaN。修法：merge 后 assert `len(merged) == len(prompt) == len(interaction)` 并 `features.isna().sum().sum() == 0`。

3. **`all-MiniLM-L6-v2` cache 路径**：用 `HF_HUB_CACHE` 即可复用；如果系统同时设了 `TRANSFORMERS_CACHE` 会优先后者，可能下载到奇怪位置。仅设 `HF_HUB_CACHE` 最干净。

## Smoke test（现有 CSV 仍有效的快速验证，<5 s 无 GPU）

```bash
cd 4.feature-engineering
python3 -c "
import pandas as pd
df = pd.read_csv('features.csv')
assert df.shape == (5000, 262), f'shape {df.shape}'
assert df['split'].nunique() == 100, f\"triplets {df['split'].nunique()}\"
assert (df.groupby('split').size() == 50).all(), 'per-triplet count'
assert df.isna().sum().sum() == 0, 'NaN present'
print('PASS', df.shape, df['split'].nunique(), 'triplets')
"
```

期望：`PASS (5000, 262) 100 triplets`。

**完整重跑**（需先修 `WIKITEXT_DIR`，预估 30–60 min，GPU 推理 embedding 主导）：
```bash
cd 4.feature-engineering/scripts
python 1.forget_set_festures.py
python 2.prompt_features.py
python 3.interaction_features.py
python 4.merge_features.py
```
