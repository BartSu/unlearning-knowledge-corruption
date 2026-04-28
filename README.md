# LLM Unlearning — 三层 Corruption + Forget-Set 审计

> 不跑 unlearn，只看 forget set 几何 ⇒ 副作用预警。
> Llama-3.1-8B-Instruct · GradAscent · WikiText-103 · HDBSCAN × 10 簇 · $n=100$ forget set。

---

## 🗂️ 项目元文档（三份关键文件）

| 文件 | 作用 |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | **迭代规则** —— agent 每次迭代的读写约定、任务选择规则、安全约束。 |
| [`STATE.md`](STATE.md) | **当前研究状态快照** —— 瓶颈、关键数字、下一步决策点。 |
| [`PROGRESS.md`](PROGRESS.md) | **任务清单 + 迭代日志** —— 五阶段流水线的任务目标、目录映射、文档同步规则、历次迭代记录。 |

> 职责分工：`CLAUDE.md` 是**要遵守的条约**，`STATE.md` 是**现在的项目进展**，`PROGRESS.md` 是**都干了啥**。

---

## 🎞️ 对外材料

所有对外叙事产物在 [`z-doc/`](z-doc/) 下，详细约定见 [`z-doc/CLAUDE.md`](z-doc/CLAUDE.md)。

### Slides（讲解 / 答辩）

| 材料 | 用途 | 路径 |
|---|---|---|
| 📝 Slides 源文件 | beamer / metropolis theme，编辑用 | [`z-doc/slides-en.tex`](z-doc/slides-en.tex) |

### Paper（EMNLP draft）

| 材料 | 用途 | 路径 |
|---|---|---|
| 📄 Paper PDF | 当前 draft 编译产物（9 页，ACL-like 双栏） | [`z-doc/paper/main.pdf`](z-doc/paper/main.pdf) |
