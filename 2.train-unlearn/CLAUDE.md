# CLAUDE.md —— 阶段 ②「Unlearn 训练」

本目录的唯一产物是 **每个 forget set 一个 unlearn checkpoint**（`unlearn/saves/...`），供阶段 ③ cross-PPL 评测消费。

根目录约定见 [`../CLAUDE.md`](../CLAUDE.md)。数据契约见 [`../1.data-preparation/CLAUDE.md`](../1.data-preparation/CLAUDE.md)。本文件补充阶段 ② 专有的**代码来源、环境约束和下游契约**。

---

## 目录职责

| 做什么 | 不做什么 |
|---|---|
| 用 OpenUnlearning 框架 unlearn Llama-3.1-8B-Instruct | 修改数据集（属阶段 ①） |
| 为每个 triplet 生成一份 checkpoint | 计算 cross-PPL（属阶段 ③） |
| 记录训练日志（`logs_wikitext_unlearn*`） | 自行升级框架依赖版本 |

## `open-unlearning/` —— 上游 vendored fork

**来源**：[locuslab/open-unlearning](https://github.com/locuslab/open-unlearning/tree/main) ·  License: MIT · 以 vendored 形式内嵌（非 git submodule，直接拷贝源码）。

**角色**：提供 `src/train.py` + Hydra 配置（`configs/`）+ Trainer 抽象（`src/trainer/unlearn/`）。GradAscent / NPO / GradDiff 等方法直接复用上游实现，不自行重写。

### 🚨 环境配置必须遵循上游

按上游 `open-unlearning/README.md` 的 Quickstart 部分走，**不要自己挑版本**：

```bash
conda create -n unlearning python=3.11
conda activate unlearning
cd 2.train-unlearn/open-unlearning
pip install ".[lm-eval]"
pip install --no-build-isolation flash-attn==2.6.3
```

关键版本钉死在 `open-unlearning/requirements.txt`（`torch==2.4.1`, `transformers==4.51.3`, `accelerate==0.34.2`, `deepspeed==0.15.4`, `hydra-core==1.3` 等）。

**硬性要求**：
- 升级任何依赖版本前，先确认上游最新 main 是否已升级，**优先同步上游**而不是本地 patch。
- 本地 Python 必须 ≥ 3.11（上游 `python_requires=">=3.11"`）。
- Flash-attn 不在 `requirements.txt` 里，必须手动装 `2.6.3`（上游 Quickstart 指定），否则 `attn_implementation=sdpa` 虽可退回但训练速度会下降。

### vendored fork 的维护规则

- 优先**不改** `open-unlearning/src/` 下的核心代码（`train.py`, `trainer/`, `data/`, `evals/`）—— 要改先问：能否通过 Hydra config override / 新 Trainer 子类 / 外层 wrapper 脚本达成同样目的？改上游代码会失去和 upstream `git diff` 同步的能力。
- 本仓库对 `open-unlearning/` 的任何本地改动，都应在 commit message 或 PROGRESS.md 里标注"local patch to upstream"，方便以后合并上游更新。
- `open-unlearning/saves/` 和 `open-unlearning/logs/` 已被 `.gitignore`（上游运行会自己生成），不要 track。

## `unlearn/` —— 本地 wrapper 脚本

OpenUnlearning 的启动命令偏长（Hydra override 一堆），`unlearn/*.sh` 是预配好的 shell wrapper：

| 脚本 | 用途 | 配置要点 |
|---|---|---|
| `wikitext_unlearn_sample.sh` | 单 triplet smoke-test（验证端到端能跑） | 深度配置（`max_steps=150`, `epoch=10`） |
| `wikitext_unlearn.sh` | 单 triplet 浅配置 | `max_steps=2`, `epoch=1`（历史 batch 用的浅配置，已知偏浅，见 STATE.md 阶段 ② 决策点） |
| `wikitext_unlearn_batch.sh` | 循环 100 个 triplet 的批量浅 unlearn | 同上浅配置 |
| `wikitext_unlearn_tofu_aligned.sh` | **当前主配置**，对齐 TOFU 默认超参 | `max_steps=5, epoch=5, bs=16, gas=4, lr=1e-5, optim=paged_adamw_32bit`（见 STATE.md 阶段 ② 训练配置表） |

PPL 评测不在本目录 —— 由阶段 ③（`../3.inference/extract-ppl/`）负责；本阶段只产 checkpoint。

### 重要的日志约定

- `logs_wikitext_unlearn_sample/` —— sample 脚本产生
- `logs_wikitext_unlearn_batch/` —— batch 脚本产生
- `logs_wikitext_unlearn_tofu/` —— TOFU-aligned 主配置产生
- `logs_*_bak/` —— 旧实验归档，不要删，但也不要当最新结果引用
- `batch_run.log` / `run.pid` —— 当前正在跑的 batch 的运行态

## 上下游契约

**读取阶段 ① 的**（每个 triplet）：
- `../1.data-preparation/data/wikitext_hdbscan_triplets/triplet_NNN/train.json` → forget set（给 GradAscent forget loss）
- `../1.data-preparation/data/wikitext_hdbscan_triplets/triplet_NNN/validation.json` → retain set（给 retain-side loss）
- `test.json` **本阶段不读**（属阶段 ③ 的 cross-PPL 探针）

**写入给阶段 ③ 的**：
- `unlearn/saves/wikitext_unlearn_<tag>/wikitext_Llama-3.1-8B-Instruct_triplet_NNN_GradAscent/`
  - 4 分片 safetensors + `trainer_state.json` + tokenizer 文件
  - 单 ckpt ≈ 16 GB（bf16）；100 ckpt ≈ 1.6 TB（盘占见 STATE.md 警告）

## 对 Claude 的具体要求

1. **环境冲突以上游为准**：`open-unlearning/requirements.txt` 是权威，不要随意 pin 到更新版本的 `torch` / `transformers`。
2. **训练超参的权威位置**：Hydra config（`open-unlearning/configs/experiment/unlearn/wikitext/default.yaml` + CLI override）。改超参前先看 STATE.md 阶段 ② 的配置表是否需要同步更新。
3. **训练运行长耗时**：batch 跑 100 triplet ~1.5–2 h；不要在一个迭代里等到跑完，开 `run_in_background` + 日志监控。
4. **checkpoint 盘占极大**（16 GB × 100 = 1.6 TB），跑新配置前先确认盘够；必要时归档旧 `saves/wikitext_unlearn_*/`，不要随手 `rm -rf`。
5. **不要改上游源码绕过问题**：Hydra 找不到配置、Trainer 报错 —— 优先看 override、看 `open-unlearning/docs/hydra.md`、看上游 issue，最后才是本地 patch。
6. **max_steps 配置歧义**（见 STATE.md 阶段 ② 当前决策点）：写新脚本前先看 `wikitext_unlearn_tofu_aligned.sh` 为准，浅配置 `wikitext_unlearn*.sh` 是历史遗留。

---

## 🧭 新机器 / fresh clone 上的环境 bootstrap（必读）

同一份代码在**不同物理机器**上跑会踩的坑。fresh clone、换机器、换云实例时先读这节，省 30 分钟。

### Miniconda 与 `unlearning` env

- Miniconda 已装在 `/media/volume/llm/miniconda3`。**base env 和 `unlearning` env 均为只读**（属于另一个 user），禁止 `pip install` / `conda install`（会 `Permission denied`）。
- `unlearning` env 已预装：`python=3.11.14` / `torch=2.4.1+cu121` / `transformers=4.45.1` / `flash_attn=2.6.3` / `deepspeed=0.15.4` / `hydra-core=1.3.0` / `datasets=3.0.1` / `accelerate=0.34.2` / `bitsandbytes=0.44.1`。注意 `transformers=4.45.1` 低于上游 `requirements.txt` 的 `4.51.3`，但已验证兼容（历史 100 ckpt + 当前 smoke test 均跑通）。
- 激活：`source /media/volume/llm/miniconda3/etc/profile.d/conda.sh && conda activate unlearning`。

### 跑 `wikitext_unlearn_tofu_aligned.sh` 前**必设**的 3 个环境变量

```bash
export HF_HUB_CACHE=/media/volume/llm/huggingface/hub        # 复用已下载的 Llama-3.1-8B-Instruct（~15 GB，snapshot 0e9e39f...）
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets   # datasets 要写 lock；HF_HUB_CACHE 目录对当前 user 不可写
export CUDA_HOME=$HOME/fake_cuda                             # deepspeed 0.15.4 import-time 会跑 $CUDA_HOME/bin/nvcc -V，系统无 CUDA toolkit 时必须 shim
```

三者缺任一都会 fail（已分别验证 fail 模式）。**不要**直接设 `HF_HOME=/media/volume/llm/huggingface` 省事 —— 那会让 datasets cache 也落到只读目录，仍会 `PermissionError`。

### 已踩过的坑（留档，避免重蹈）

1. **Hydra struct override 三态**：
   - `trainer.args.X=v` → 要求 `X` 已在 struct 里；否则 `Key not in struct`。
   - `+trainer.args.X=v` → 要求 `X` **不**在 struct 里；否则 `already at`。
   - `++trainer.args.X=v` → 不管在不在都 force set。
   `configs/trainer/finetune.yaml` 默认 `args` struct 里**没有** `max_steps` / `warmup_steps`，**有** `lr_scheduler_type`（继承自 `TrainingArguments`）。wrapper 脚本里三者一律用 `++`，不要混用，避免未来改 struct 时互相拉扯。

2. **deepspeed 0.15.4 import-time 的 CUDA_HOME 检查**：`import deepspeed` 触发 `FPQuantizerBuilder.is_compatible()` → `installed_cuda_version()`，无 `CUDA_HOME/bin/nvcc` 时**直接 raise** `MissingCUDAException`（不是 return False，设计问题）。单卡 GradAscent 实际不走 deepspeed 编译路径，只是 import-time 卡住。**不要**装系统级 CUDA toolkit（要 sudo，~3 GB），用一个 shim 就够：

   ```bash
   mkdir -p $HOME/fake_cuda/bin
   cat > $HOME/fake_cuda/bin/nvcc <<'EOF'
   #!/bin/bash
   case "$1" in
       -V|--version)
           echo "nvcc: NVIDIA (R) Cuda compiler driver"
           echo "Cuda compilation tools, release 12.1, V12.1.105"
           ;;
       *) echo "fake nvcc shim" >&2; exit 1 ;;
   esac
   EOF
   chmod +x $HOME/fake_cuda/bin/nvcc
   ```

   未来升级到 deepspeed ≥ 0.16（若上游同步）可能不再需要 shim；验证方式：`CUDA_HOME= python -c "import deepspeed"` 不再 raise。

3. **HF cache 权限分离**：`/media/volume/llm/huggingface/hub/` 权重目录可**读**（snapshot 完整：4 分片 safetensors + tokenizer），但 `/media/volume/llm/huggingface/datasets/` 属于另一个 user，`datasets.load_dataset` 想写 `.lock` 时 `PermissionError`。固定方案：`HF_HUB_CACHE` 指大盘只读缓存、`HF_DATASETS_CACHE` 指用户目录。

### Smoke test（~140–160 s，单 triplet）

fresh env / 换机后先跑这个确认链路通，再动 batch：

```bash
cd /media/volume/llm/unlearning/2.train-unlearn/unlearn
source /media/volume/llm/miniconda3/etc/profile.d/conda.sh && conda activate unlearning
export HF_HUB_CACHE=/media/volume/llm/huggingface/hub \
       HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets \
       CUDA_HOME=$HOME/fake_cuda
./wikitext_unlearn_tofu_aligned.sh --triplet triplet_001 --single
```

期望产物：`saves/wikitext_unlearn_tofu/wikitext_Llama-3.1-8B-Instruct_triplet_001_GradAscent_tofu/` 下 4 分片 safetensors + tokenizer + `trainer_state.json`（`global_step=5`，`train_loss ≈ -0.6`，GradAscent 负值单调下降符合预期）。失败看 `logs_wikitext_unlearn_tofu/<task>.log`。
