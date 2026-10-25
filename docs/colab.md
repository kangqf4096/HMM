# Colab：OPT-1.3B 与 arXiv

Git 仓库保存共享代码、配置和说明；Colab 拉取仓库后准备资源并执行实验。现有目录结构继续使用，权重、数据、缓存和完整运行输出均不提交 Git。本说明使用 `configs/colab_arxiv.json`；原 `text_smoke.json` / `text_pilot.json` 保留 Qwen 服务器实验设置。

## 第一次连接

在 Colab 选择 GPU 运行时。新建 Notebook 即可，核心逻辑仍由仓库脚本执行。先在一个单元格运行：

```python
%cd /content
!git clone https://github.com/kangqf4096/HMM.git opt-hmm
%cd /content/opt-hmm
```

如为私有仓库，先通过自己的认证方式取得访问权，不把凭证写入共享 Notebook。已有 checkout 时，保存需要的修改后再同步分支，无需重复 clone。Colab 只能拉到已经提交并推送的文件；本地尚未提交的改动不会自动同步。

以下使用项目内独立 Python 环境，避免依赖 Notebook 内已经导入的 PyTorch。将整个代码块放在一个单元格运行；`%%bash` 必须是单元格第一行：

```bash
%%bash
set -euo pipefail
cd /content/opt-hmm
source scripts/env.sh
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[inference]"
.venv/bin/python scripts/test.py
.venv/bin/python scripts/run.py check-runtime
```

项目固定 vLLM 0.19.1；让其安装兼容的 PyTorch 依赖。后续通过 `.venv/bin/python` 执行脚本。每个 `%%bash` 单元格都要重新 `cd` 并 `source scripts/env.sh`，因为不同单元格的 shell 环境不共享。临时文件和 pip/Hugging Face/vLLM 缓存均留在项目内。

`check-runtime` 核验真实 vLLM 版本、CUDA、GPU 和采样适配器。模型家族受支持不等于当前会话已经通过实际生成验收；安装或检查失败时先处理报错，再运行生成。

## 准备模型和数据

```bash
%%bash
set -euo pipefail
cd /content/opt-hmm
source scripts/env.sh
.venv/bin/python scripts/download_model.py
.venv/bin/python scripts/run.py prepare --config configs/colab_arxiv.json --output data/ax0
```

`download_model.py` 从 [facebook/opt-1.3b](https://huggingface.co/facebook/opt-1.3b/tree/3f5c25d0bc631cb57ac65913f76e22c2dfb61d62) 下载固定版本的 PyTorch 权重、配置和 tokenizer 到 `models/opt-1.3b/`，只下载一份所需权重，不同时下载 TensorFlow/Flax 版本。PyTorch 权重约 2.63 GB，实际 GPU 显存需求还包括 KV cache 和运行开销。重复执行下载命令会利用下载状态；不要求额外文件哈希校验。来源版本保存为 `opt_hmm_source.json`，并写入准备清单及后续生成记录。

`prepare` 下载 [ccdv/arxiv-summarization](https://huggingface.co/datasets/ccdv/arxiv-summarization) 的固定 `document` 版本，使用 `article` 正文，继续原有的整篇去重和角色分组。它不是从 arXiv 下载 PDF，也不使用摘要作 prompt。当前实现会缓存完整 document 配置，不是流式小样本下载：下载约 3.61 GB，生成 Arrow 数据约 7.55 GB。模型、数据缓存合计应至少预留约 15 GB，另给 Python 依赖、pip 缓存和实验输出留空间；不需要预留整个 arXiv PDF 库的空间。

准备结束得到 `data/ax0/prompts.jsonl` 和 `manifest.json`。这一步使用 tokenizer，无需加载 GPU 模型进行生成。当前配置是 smoke 规模：92 篇独立论文，50 token prompt，128 token continuation；它用于流程验收，正式实验应另建配置扩大规模。

## 生成与评价

```bash
%%bash
set -euo pipefail
cd /content/opt-hmm
source scripts/env.sh
.venv/bin/python scripts/run.py generate --config configs/colab_arxiv.json --prompts data/ax0 --output runs/ax0-gen
.venv/bin/python scripts/run.py evaluate --input runs/ax0-gen --output runs/ax0-eval
```

主要阅读 `runs/ax0-eval/report.md`。输出目录必须为空；准备或生成成功后不必重跑前一步。需要另一次完整实验时换短名称 `ax1`。更换模型后重新 prepare、generate、evaluate，不能复用 Qwen 的 tokenized prompts、发射密度或阈值。

若需要此前的枢轴概率诊断，在 generate 命令加上 `--pivot-diagnostics`，并在生成结束后另一个进程执行：

```bash
.venv/bin/python scripts/run.py diagnose-pivots --input runs/ax0-gen --output runs/ax0-diag
```

## 持久保存

Colab 会话的 `/content` 文件随虚拟机回收而消失；共享 Notebook 也不会共享其虚拟机中的文件。需要保留的准备数据、结果和可选的模型副本放到 Drive 等持久存储。高频运行使用 `/content/opt-hmm` 的本地文件；保存或恢复时复制到/从 Drive。

先在 Python 单元格挂载：

```python
from google.colab import drive
drive.mount('/content/drive')
```

然后将指定的已完成运行复制保存：

```bash
%%bash
set -euo pipefail
cd /content/opt-hmm
mkdir -p /content/drive/MyDrive/opt-hmm/ax0
cp -a data/ax0 /content/drive/MyDrive/opt-hmm/ax0/prompts
cp -a runs/ax0-gen /content/drive/MyDrive/opt-hmm/ax0/generation
cp -a runs/ax0-eval /content/drive/MyDrive/opt-hmm/ax0/evaluation
```

以上保存命令用于第一次归档到新目录；已存在的归档使用新名称，避免重复嵌套复制。也可以将模型和 `.cache/hf/` 另行复制保留以节省下次下载；并非必需，原始公开资源可按固定版本重新取得。恢复准备数据时，模型仍放回 `/content/opt-hmm/models/opt-1.3b`，以满足现有模型路径检查。

Git 中保留实验配置、代码版本和整理后的小型图表/报告即可。完整 `runs/` 仍由 Git 忽略。准备清单和生成 metadata 已记录代码 commit、实际包版本、模型/数据来源和配置；有未提交修改时，commit 本身不能代表完整的运行代码，因此共享实验前应先提交对应代码。

参考：[Colab 文件与运行时说明](https://research.google.com/colaboratory/faq.html)、[Hugging Face 定向下载](https://huggingface.co/docs/huggingface_hub/guides/download)、[vLLM 0.19.1 支持模型](https://docs.vllm.ai/en/v0.19.1/models/supported_models/)。本地 CPU 检查不能替代 Colab 上的实际下载和 GPU smoke。
