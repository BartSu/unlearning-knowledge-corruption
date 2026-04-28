# CLAUDE.md —— `z-doc/`「对外 presentation」

本目录放所有**给人看**的叙事材料：英文 Beamer slides、EMNLP paper draft、图表脚本、related work 登记簿。**不在任何 pipeline 的数据 path 上**，单纯 sink。

根目录约定见 [`../CLAUDE.md`](../CLAUDE.md)。

---

## 目录职责

| 做什么 | 不做什么 |
|---|---|
| 承载 `slides-en.tex` + `paper/` EMNLP draft 两份对外叙事 | 重新跑实验 / 产数字（属阶段 ①–⑤） |
| 承载 `figures/*.py` 从上游 artefact 生成 PDF 图 | 引入新的数据源（图必须 trace 回 artefact） |
| 编译 `slides-en.pdf` / `paper/main.pdf` 提交入 git | 只管 `.tex` 不管 `.pdf`（两者必须一致） |

## 文件清单

| 路径 | 内容 |
|---|---|
| `slides-en.tex` / `slides-en.pdf` | 英文 Beamer slides（讲解用，metropolis theme） |
| `paper/main.tex` + `paper/sections/*.tex` | **EMNLP paper draft**（article 双栏 ACL-like 排版；提交前可换官方 `acl.sty`） |
| `paper/references.bib` | paper 的引用库（只放正文真正 `\citep` 的条目，详见 paper hygiene 段） |
| `paper/main.pdf` | paper draft 编译产物 |
| `paper/figures/` | symlink 到 `../figures/`，paper 与 slides 共享图源 |
| `figures/make_figures.py` | 从 `5.audit/regression-predictor/audit/` 读 CSV 生成 fig_three_layer_decay / fig_per_forget_profile / fig_audit_scatter |
| `figures/make_fig1_hero.py` | 生成 `fig1_hero.pdf`（一图看懂帧用） |
| `figures/make_fig2_intro_storyboard.py` | 生成 `fig2_intro_storyboard.pdf` |
| `cite/cite.md` | **Related work 登记簿**（见下节 §cite 规则） |
| `cite/*.pdf` | 相关文献的 PDF 正本（与 `cite.md` 条目一一对应） |

## 编译方式（决定性信息）

### 首选：tectonic（本仓库默认）

**为什么**：single binary、按需从 CTAN 拉缺失 package、不碰系统 TeX、可在不同机器一致复现。装过一次 `~/bin/tectonic` 即可用。

```bash
cd z-doc/
~/bin/tectonic slides-en.tex            # slides
cd paper && ~/bin/tectonic main.tex     # paper draft
```

**特点**：
- 一次调用自动完成多次 LaTeX pass（不需要人工 xelatex 跑两次）
- 缺包自动下载（不会像系统 TeX 那样 "package not found"）
- 首次编译 paper 会拉 times / natbib / booktabs 等，~30 s；之后 <10 s

### 备选：系统 xelatex（若 tectonic 不可用）

```bash
xelatex -interaction=nonstopmode slides-en.tex
xelatex -interaction=nonstopmode slides-en.tex   # 双跑保证 TOC 引用
```

依赖：`apt install texlive-xetex texlive-fonts-extra`。注意**跨机一致性不如 tectonic**（系统包版本漂移可能让同一 .tex 渲染略有差异）。

### Overleaf（远端备选）

Overleaf 编辑器选 **XeLaTeX**（slides）或 **pdfLaTeX**（paper）。字体和 package 托管在云端，无需本地环境。PDF 要手动下回本地再 commit。

## 对 Claude 的具体要求

1. **改完 .tex 必须重编 .pdf 并一起 commit**。`.tex` 和 `.pdf` 不一致会让 fresh clone 看到的 PDF 与叙事脱节。
2. **slides 与 paper 数字必须同源**：`slides-en.tex` 与 `paper/sections/*.tex` 引用的所有关键数字（L1/L2/L3、$\rho$、$R^2$、CI 等）都必须 trace 回 `5.audit/regression-predictor/audit/audit_summary.json` 与 `3.inference/extract-ppl/corruption_summary.json`。任一上游 artefact 更新，两份叙事须同步刷新。
3. **数字更新触发重生成图**：`5.audit/.../audit/*.csv` 或 `3.inference/extract-ppl/corruption_summary.json` 变了之后，`figures/make_figures.py` 要重跑生成新 PDF，然后 slides + paper 重编。
4. **paper bib 卫生**：`paper/references.bib` 只放 `paper/sections/*.tex` 真正 `\citep`/`\citet` 到的条目，不要塞死条目。新增引用前先到 [`cite/cite.md`](cite/cite.md) 查是否已登记；若是直接影响 framing 的文献，新增条目时同步给 cite 登记簿加一条三段结构。
5. **overfull 可接受阈值**：metropolis theme（slides）对 `\vbox` 最多溢出 ~20 pt 视觉不明显（被页脚进度条遮盖）；> 50 pt 就会顶出框外看到半行，需要 trim。paper 双栏的 hbox underfull/overfull 通常来自长 URL 或参考文献，可忽略。
6. **不要改 `figures/*.pdf` 直接**：它们是 `make_*.py` 的产物，改源脚本重跑。
7. **paper draft 排版**：当前 `paper/main.tex` 用 `\documentclass{article}` 自包含 ACL-like 双栏布局；正式提交 EMNLP 时下载 `acl.sty` 替换 preamble，section 内容无需动。

## `cite/` —— Related Work 登记簿

**目的**：集中记录**真正影响 framing 或实现**的文献。每条包含 full citation + summary + 对本工作的启发或使用 + positioning（我们与他们的差别）。广度扫描 / 闲读文献**不进**这里。

**当前已登记**（全文见 [`cite/cite.md`](cite/cite.md)）：

| PDF | 对本工作的角色 |
|---|---|
| `a-curious-case.pdf` | **视角来源**：data-first 预测下游行为（训练数据 → 对抗鲁棒性）→ 启发我们的 forget-set audit |
| `knowledge_hole.pdf` | **现象来源**：unlearn 留下的 adjacent / latent knowledge holes → 对应我们的 L2/L3 locality+spillover |
| `open-unlearning.pdf` | **Code base**：`2.train-unlearn/open-unlearning/` vendored fork 的上游 |

**维护规则**（cite.md 尾部有详细版）：
1. 新增条目 **必须**：full citation + PDF 下到 `cite/` + 三段结构（Summary / How it inspires or uses us / What we differ）
2. **不删条目** —— 若某篇后来不再 relevant，加 `**Note (YYYY-MM-DD)**: no longer cited because ...`，留 audit trail
3. 广度 / scan 类文献进 paper draft 的 Related Work 段，不塞这里

## 已踩过的坑（留档）

1. **本机无 xelatex 但有 tectonic**（2026-04-24 发现）：换机后 `xelatex` 命令不在 PATH 里，但 `~/bin/tectonic` 已装。不要白费力气试装 TeX Live 全家桶；**tectonic 够用**。

2. **`figures/` 产物带时间戳 / 数字的文件覆盖要小心**：`fig1_hero.pdf` / `fig2_intro_storyboard.pdf` / `fig_audit_scatter.pdf` 等在 n=50 → n=100 换配置时会**静默覆盖**；要追溯某版数字对应的图，用 `git show <sha>:z-doc/figures/fig_*.pdf` 回查。

3. **`make_figures.py` 硬编码 `5.audit/.../audit/*.csv` 路径**：如果目录重构（历史上已发生 `4.regression-predictor` → `5.audit/regression-predictor`），脚本要同步改路径，否则图拿到**旧 artefact** 就 silently 过期。

4. **`slides-en.tex` 的 `\institute{...}` 字段**：写实验环境（base model + unlearner + dataset），数字或 base model 改动时要同步。它会显示在 title page。

5. **`paper/figures/` 是 symlink**：指向 `../figures/`，paper 与 slides 共享图源。move / rename `figures/` 时必须同步重建 symlink，否则 paper 编译会丢图。

6. **paper bib 里的 `[VERIFY]` 占位**：`references.bib` 中 LLM 生成但未在 DBLP/CrossRef 核对的条目以 `% [VERIFY]` 标注；提交前必须逐条核对真实出处或删去对应 `\citep`。

## Smoke test（~30 s）

```bash
cd z-doc
~/bin/tectonic slides-en.tex > /tmp/tectonic-slides.log 2>&1
pdfinfo slides-en.pdf | grep Pages

cd paper
~/bin/tectonic main.tex > /tmp/tectonic-paper.log 2>&1
pdfinfo main.pdf | grep Pages   # 期望 ~9 页（n=100 EMNLP draft）
```

若两份 PDF 都生成且无 `error:` 输出，编译成功。Overfull / underfull warnings 正常（见对 Claude 要求 §5）。
