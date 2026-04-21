# LLM Unlearning —— 三层 Corruption + Forget-Set 审计

> **一句话**：不跑 unlearn，只看 forget set 几何 ⇒ 排序级副作用预警。
>
> Llama-3.1-8B-Instruct · GradAscent · WikiText-103 · HDBSCAN × 10 簇 · $n=50$ forget set。

---

## 🎯 Presentation 速查（把下面这几个链接打开就够讲）

### 讲解材料（带我走一遍）

| 材料 | 用途 | 路径 |
|---|---|---|
| 🎞️ **中文 slides (PDF)** | 演讲主用，29 页 | [`z-doc/slides.pdf`](z-doc/slides.pdf) |
| 🎞️ **英文 slides (PDF)** | 给英文听众 | [`z-doc/slides-en.pdf`](z-doc/slides-en.pdf) |
| 📄 **Deep-dive README (中文)** | 讲稿底稿，所有数字出处 | [`z-doc/README-CN.md`](z-doc/README-CN.md) |
| 📎 **slides 源文件 (tex)** | 想改 slides | [`z-doc/slides.tex`](z-doc/slides.tex) / [`z-doc/slides-en.tex`](z-doc/slides-en.tex) |
| 🖼️ **Hero 图（一图看懂）** | 讲第一张图 | [`z-doc/figures/fig1_hero.pdf`](z-doc/figures/fig1_hero.pdf) |
| 📚 **起点论文 (Dang+ ACL'24)** | 方法论来源 | [`z-doc/a-curious-case.pdf`](z-doc/a-curious-case.pdf) |

### 关键数字（被问就翻这几个文件）

| 数据 | 路径 |
|---|---|
| 三层 corruption ground truth | [`2.extract-ppl/corruption_summary.json`](2.extract-ppl/corruption_summary.json) |
| 审计器 LOO 结果（$\rho$、$R^2$、CI） | [`4.regression-predictor/audit/audit_summary.json`](4.regression-predictor/audit/audit_summary.json) |
| 每 forget set 的三层 profile | [`4.regression-predictor/audit/part1_corruption_profile.csv`](4.regression-predictor/audit/part1_corruption_profile.csv) |
| 12 维几何审计特征表 | [`4.regression-predictor/audit/part2_forget_features.csv`](4.regression-predictor/audit/part2_forget_features.csv) |
| 50×50 cross-ppl 明细 | [`2.extract-ppl/wikitext_cross_metrics_detail.json`](2.extract-ppl/wikitext_cross_metrics_detail.json) |

### 项目元信息

| 文档 | 内容 |
|---|---|
| [`PROGRESS.md`](PROGRESS.md) | 五阶段进度 + 迭代日志 + 当前瓶颈 |
| [`AGENT.md`](AGENT.md) | Ralph Loop 任务规范 |
| [`1.data-preparation/README.md`](1.data-preparation/README.md) | 数据 schema（triplet × 100） |

---

## 🎤 怎么讲这个项目（20 分钟版）

故事分 **两幕**，跟着 slides 的 section 走即可：

### 开场（3 min）—— 定义 + 一句话贡献

1. **三层定义帧**（slides p3）：同心圆图 + `r = PPL_unlearned / PPL_base`
   - L1 = 想遗忘的（被 unlearn 那批）
   - L2 = 连带忘的（同簇、没训过）
   - L3 = 不该忘却被波及（别的簇）
2. **一句话贡献**：三层衰减但**不归零** —— **L1 2.13× → L2 1.49× → L3 1.28×**；且 **80.3% 的"无关"样本 PPL 被抬高**。
3. **动机**：换个 forget set，副作用差近 4×；每跑一次 unlearn 要数 GPU-hour ⇒ 能不能不 fine-tune 就预测？

### 第一幕（7 min）—— 三层真的存在（C1 + C2 + C3）

- **C1**（严格递减）：2.13 > 1.49 > 1.28，不是乱分箱
- **C2**（L3 不归零）：80.3% 跨簇样本 PPL 真被抬高，不是噪声
- **C3**（forget-set 依赖）：同一算法，换 forget set，L1 差近 3×（1.42 → 4.07）
  - ⇒ 单个 forget set 给 unlearner 打分 **稳定性不足**
  - ⇒ profile 是 forget set 的函数，**值得审计**

配图：`fig_three_layer_decay.pdf`、`fig_per_forget_profile.pdf`。

### 第二幕（8 min）—— 审计：仅凭 forget set 几何排序副作用

1. **审计问题**：输入 = 12 维 forget set 几何（方差 / 相似度 / centroid / 集中度）；输出 = L1/L2/L3 风险分数。**不看 eval 文本、不跑 unlearn**。
2. **审计结果**（LOO, $n=50$, Ridge）：
   - L1 $\rho = +0.49$，CI [+0.22, +0.68] ✅
   - L2 $\rho = +0.54$，CI [+0.29, +0.72] ✅
   - L3 $\rho = +0.30$，CI [−0.01, +0.55] ⚠️ 边缘
3. **定位帧**（关键！）：审计器 **= 便宜的粗筛预警器，不是替代品**。
   - 能做：排序、粗筛、省算力
   - 不能做：绝对数值、给 unlearner 打分、省掉最终 unlearn
   - 用法：100 个候选 → 审计排序 → 真跑 top-$k$ 验证
4. **对照**：几何 16 维 $R^2 = +0.36$ vs. 文本表面 261 维 $R^2 = +0.14` —— 特征少 16×，$R^2$ 高 2.5×。
5. **诚实负结果**：球覆盖（retain in forget radius）$\rho = -0.27$ 方向反了 ⇒ 改用带符号投影 / mutual-reachability。

### 收尾（2 min）—— Limitations + Future Work

**四个「单」** 对应 **四个解法**：
- 单 base → 加模型家族
- 单 unlearner (GradAscent) → 加 NPO / GradDiff
- $n=50$ → $n=100$（100 triplet 已全部 unlearn，待扩 cross-eval）
- 单指标 PPL → QA accuracy（`2.extract-qa/` 已就位）

---

## 🗺️ 五阶段流水线（目录 = 阶段）

```
1.data-preparation/     ① 数据：WikiText-103 → HDBSCAN 10 簇 → 100 triplets
 └─ unlearn/            ② Unlearn：GradAscent × Llama-3.1-8B-Instruct（已跑 50/100）
2.extract-ppl/          ③ 困惑度：50×50 cross-PPL ⇒ L1/L2/L3 ground truth
3.feature-engineering/  ④ 几何：12 维 forget-set 内禀特征
4.regression-predictor/ ⑤ 审计：Ridge LOO 排序预测
 ├─ audit/              └─ 主结果 artefact
 └─ geometry/           └─ per-sample 对照实验
z-doc/                  📄 slides / README / figures（讲解材料）
```

**当前瓶颈**：阶段 ② —— 100 triplet 只跑了前 50 个 unlearn，扩到 100 是继续收紧 L3 CI 的必要前提（详见 `PROGRESS.md`）。

---

## ⚡ 快速复现主结果

```bash
# 0. Unlearn + cross-ppl（已 commit 产物，可跳过）
#    ⇒ 2.extract-ppl/wikitext_cross_metrics_detail.json

# 1. 第一幕：三层 ground truth
cd 2.extract-ppl && python analyze_corruption.py
# ⇒ corruption_summary.json

# 2. 第二幕 A：per-sample 几何预测器（对照 261 维基线）
cd ../4.regression-predictor && python 3.corruption_from_geometry.py
# ⇒ geometry/geometry_results.json

# 3. 第二幕 B：forget-set 审计（LOO, 4 parts）
python 4.audit_experiments.py
# ⇒ audit/audit_summary.json

# 4. 重编 slides PDF（若改过 tex）
cd ../z-doc && xelatex -interaction=nonstopmode slides.tex && xelatex -interaction=nonstopmode slides.tex
```

---

## 🎒 预想问答（面试 / QA 环节）

| Q | 要点 |
|---|---|
| 为什么不直接跑 unlearn 评估？ | 每跑一次 GradAscent 要几 GPU-hour；100 候选 × 每个 50×50 cross-eval 算不起 |
| $n=50$ 会不会太小？ | L1/L2 的 bootstrap CI 已排除 0（显著）；L3 仍边缘，这正是扩到 $n=100$ 的动机 |
| 审计器 $R^2$ 才 +0.22，是不是很差？ | 定位是**排序级粗筛预警器**，不是打分器；胜过 LOO-mean baseline ($R^2=-0.04$) 且 MAE 全线更低 |
| 为什么几何比文本表面强？ | 16 维 vs 261 维拿到 2.5× 的 $R^2$；但不能断言「corruption 就是几何现象」—— 单 base / 单 unlearner 下谨慎表述 |
| L3 的 80.3% 是不是噪声？ | 3.6% 的跨簇样本 PPL 直接翻倍（`pct_up_2x`），量级上不是随机波动 |
| 和 influence function 什么区别？ | influence 是 model-centric（看模型梯度）；我们是 data-centric（看 forget-set 几何）+ 三层风险分解 |
| 对哪个算法 generalize？ | 目前只在 GradAscent 上验证；NPO / GradDiff 的稳健性是 Future Work #2 |

---

## 📮 研究定位（一段话，可直接念）

> 本工作把 LLM unlearning 的 benchmarking 问题重构成一个 **forget-set 审计** 问题。我们在 Llama-3.1-8B-Instruct × GradAscent 设定下观察到 unlearning 的副作用沿语义距离呈三层衰减（L1 forget / L2 同簇 locality / L3 跨簇 spillover），但 L3 **并未归零**：80.3% 的跨簇样本 PPL 被抬高。作为方法论提议，我们用 12 维 forget set 几何特征在 LOO（$n=50$）下对这三层做排序级预测，L1/L2 的 Spearman $\rho$ 的 95% bootstrap CI 已排除 0。审计器定位为**便宜的粗筛预警器** —— 在正式 unlearn 前用 1 秒的几何分析，给出红 / 黄 / 绿三档风险提示。
