# Colab：OPT-1.3B 与 arXiv

Git 仓库保存共享代码、配置和说明；Colab 准备资源并执行实验。使用 `configs/colab_arxiv.json`。现有目录结构继续使用，模型、数据、缓存和完整输出均由 Git 忽略。

仓库提供可直接上传到 Colab 的 [Notebook](../notebooks/colab_arxiv.ipynb)。推送代码后，也可从 [Colab 打开仓库中的 Notebook](https://colab.research.google.com/github/kangqf4096/HMM/blob/main/notebooks/colab_arxiv.ipynb)，按界面要求授权访问私有 GitHub 仓库。下面给出同样的单元格，适用于手动创建的 Notebook。

## 第 1 步：启用 GPU

Colab 菜单“运行时 → 更改运行时类型”选择 GPU。运行 `!nvidia-smi` 检查显卡。

## 第 2 步：克隆或更新私有仓库

当前 HMM 为私有仓库。SSH 地址 `git@github.com:kangqf4096/HMM.git` 需要在 Colab 配置 SSH 密钥；这里采用同一仓库的 HTTPS 地址和只读 personal access token。

在 GitHub Settings → Developer settings → Personal access tokens → Fine-grained tokens 创建令牌：Resource owner 选择 `kangqf4096`；Repository access 选择 Only select repositories → HMM；Repository permissions 中 Contents 设为 Read-only。也可从[预填页面](https://github.com/settings/personal-access-tokens/new?name=Colab-HMM&target_name=kangqf4096&contents=read&expires_in=7)进入后选择仓库。

将下格当作普通 Python 单元格运行。运行时在隐藏输入框粘贴 token 并按回车。首次 clone；已有 checkout 时拉取当前分支。token 只通过本次 Git 子进程的环境配置使用，禁用 credential helper，结束后清理临时变量。项目文件、Notebook 源码、remote URL 和 Git 凭据文件中均不保存 token。

```python
import base64
import os
import subprocess
from getpass import getpass
from pathlib import Path

def github_sync(root="/content/opt-hmm"):
    """Clone or fast-forward the current branch, prompting for a token each time."""
    project = Path(root).resolve()
    repo = "https://github.com/kangqf4096/HMM.git"
    if (project / ".git").is_dir():
        branch = subprocess.check_output(
            ["git", "-C", str(project), "branch", "--show-current"], text=True).strip()
        if not branch:
            raise RuntimeError("Checkout is detached. Switch to the intended branch before pulling.")
        command = ["git", "-C", str(project), "pull", "--ff-only", repo, branch]
    else:
        command = ["git", "clone", repo, str(project)]
    token = getpass("GitHub token（每次输入，不保存）: ").strip()
    if not token:
        raise ValueError("Token is empty.")
    auth = base64.b64encode(f"kangqf4096:{token}".encode()).decode()
    git_env = os.environ.copy()
    git_env.update({
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
        "GIT_CONFIG_VALUE_0": f"Authorization: Basic {auth}",
        "GIT_CONFIG_KEY_1": "credential.helper",
        "GIT_CONFIG_VALUE_1": "",
    })
    try:
        subprocess.run(command, env=git_env, check=True)
    finally:
        git_env.pop("GIT_CONFIG_VALUE_0", None)
        token = auth = None
    print("代码已同步：", project)
    return project

PROJECT = github_sync()
os.chdir(PROJECT)
```

函数定义可保留在 Notebook 中，里面没有令牌值。后续运行 `github_sync()` 时每次重新输入 token；不是把 token 保存在项目内再靠 `.gitignore` 忽略。如报 403/Repository not found，检查 token 的仓库选择、Contents 权限和所属账号的访问权。

## 第 3 步：安装并检查环境

整块放入一个单元格，`%%bash` 位于第一行：

```bash
%%bash
set -euo pipefail
cd /content/opt-hmm
source scripts/env.sh
if ! python3 -m venv .venv; then
    python3 -m pip install virtualenv
    python3 -m virtualenv .venv
fi
.venv/bin/python -m pip install -e ".[inference]"
.venv/bin/python scripts/test.py
.venv/bin/python scripts/run.py check-runtime
```

项目固定 vLLM 0.19.1，并使用其兼容的 PyTorch 依赖；后续由 `.venv/bin/python` 执行。首次安装可能较慢。`check-runtime` 核验真实版本、CUDA、GPU 和采样适配器，检查成功后继续。

## 第 4 步之前：加载进度工具

```python
import runpy

tools = runpy.run_path("/content/opt-hmm/scripts/colab_tools.py")
run_step = tools["run_step"]
github_sync = tools["github_sync"]
```

第 4 步开始使用 Python 单元格调用 `run_step`。它采用 `subprocess.Popen`、合并 stdout/stderr 的 PIPE、Python `-u` 和独立的管道读取线程；按字节读取，处理 `\r` 与 `\n`，避免只用 readline 时等不到进度条刷新。IPython 每约 0.5 秒更新同一个 HTML 输出区域，保留最近日志及完整日志文件。

Colab 前端仍可能合并更新；这不能保证逐 token 实时显示。没有可靠总量的阶段显示活动条和耗时，已知数量的阶段显示实际计数。百分比属于当前阶段，不把已完成子任务或耗时冒充整个流程的总进度。失败会抛出异常并保留日志；点击停止时尝试结束该步骤的进程组。

## 第 4 步：下载 OPT-1.3B

```python
model_log = run_step(
    "scripts/download_model.py",
    title="第 4 步：下载 OPT-1.3B", log_name="model",
)
```

从 [facebook/opt-1.3b](https://huggingface.co/facebook/opt-1.3b/tree/3f5c25d0bc631cb57ac65913f76e22c2dfb61d62) 下载固定版本的 PyTorch 权重、配置和 tokenizer 到 `models/opt-1.3b`。只下载所需的一份权重，不同时取 TF/Flax 版本。权重约 2.63 GB；实际 GPU 显存还包括 KV cache 和运行开销。下载阶段复用 Hugging Face 报告的进度，重跑复用下载状态，不要求额外文件哈希校验。

## 第 5 步：下载并准备 arXiv

```python
prepare_log = run_step(
    "scripts/run.py", "prepare",
    "--config", "configs/colab_arxiv.json",
    "--output", "data/ax0",
    title="第 5 步：准备 arXiv", log_name="prepare",
)
```

使用 [ccdv/arxiv-summarization](https://huggingface.co/datasets/ccdv/arxiv-summarization) 的固定 document 版本和 article 正文，不下载 PDF，不使用摘要作提示词。当前实现缓存完整配置：约 3.61 GB 下载和约 7.55 GB Arrow 数据，之后才抽样；不是流式小样本下载。模型与数据至少预留约 15 GB，另给 Python 依赖、安装缓存和输出留空间。

数据下载/Arrow 处理显示 datasets 报告的进度，抽样显示已选论文数。输出 `data/ax0/prompts.jsonl` 和 `manifest.json`；当前为 92 篇独立论文、50 token prompt、128 token continuation 的 smoke 配置。只准备资源时，到此即可。模型和数据来源版本记录在准备清单与后续 metadata 中。

## 第 6 步：生成并评价

```python
generation_log = run_step(
    "scripts/run.py", "generate",
    "--config", "configs/colab_arxiv.json",
    "--prompts", "data/ax0",
    "--output", "runs/ax0-gen",
    title="第 6 步：生成实验文本", log_name="generate",
)
```

```python
evaluation_log = run_step(
    "scripts/run.py", "evaluate",
    "--input", "runs/ax0-gen",
    "--output", "runs/ax0-eval",
    title="第 6 步：校准与评价", log_name="evaluate",
)
```

生成进度复用 vLLM 的真实请求进度，随后按请求核验采样审计并整理输出。评价显示实际完成的 null 校准/推断数量，以及配对 bootstrap 的方法数量。

输出目录必须为空。某一步成功后不必重跑前一步；另一次实验换短名称 ax1。更换模型后重新 prepare、generate、evaluate，不能复用 Qwen 的 tokenized prompts、发射密度或阈值。原 text_smoke/text_pilot 配置保留服务器 Qwen 设置。

若需要枢轴诊断，在生成参数中增加 `--pivot-diagnostics`；生成结束后执行：

```python
run_step("scripts/run.py", "diagnose-pivots", "--input", "runs/ax0-gen", "--output", "runs/ax0-diag",
         title="枢轴诊断", log_name="diagnose")
```

## 第 7 步：查看报告

```python
from pathlib import Path
from IPython.display import Markdown, display

display(Markdown(Path("/content/opt-hmm/runs/ax0-eval/report.md").read_text(encoding="utf-8")))
```

完整日志在 tmp/model.log、tmp/prepare.log、tmp/generate.log、tmp/evaluate.log；同一步重跑时追加。Notebook 只展示最近日志，文件保存全部内容。

## 第 8 步：保存实验

Colab 的 /content 会随虚拟机回收而消失；分享 Notebook 不会分享其虚拟机文件。下格打包实验输入与结果并下载：

```python
import subprocess
from google.colab import files

subprocess.run([
    "tar", "-czf", "/content/ax0.tar.gz", "-C", "/content/opt-hmm",
    "data/ax0", "runs/ax0-gen", "runs/ax0-eval",
], check=True)
files.download("/content/ax0.tar.gz")
```

也可挂载 Google Drive 后复制到 MyDrive 中。模型与完整语料缓存可选备份，也可按固定版本重新取得；不要把 .venv 当成可搬运环境归档。恢复准备数据时模型仍放回 /content/opt-hmm/models/opt-1.3b，以满足现有模型路径检查。

## 后续 pull

在实验步骤结束后，运行普通 Python 单元格：

```python
PROJECT = github_sync()
```

每次重新输入 token。它通过 HTTPS 更新当前分支，使用 --ff-only；分支分叉或更新会覆盖本地修改时停止，并保留 Git 的报错。不会执行 reset/clean。Git 忽略的模型、数据、缓存和结果保留，pull 不会让它们重新下载。

更新后重跑“加载进度工具”单元格；pyproject.toml 等依赖声明有变化时重跑第 3 步。Notebook 函数里没有 token，本地 Windows 仓库的 remote 不受 Colab 操作影响。

若正在使用旧 Notebook、还没有 github_sync 定义，可直接粘贴第 2 步的完整代码；已有 checkout 时它会 pull。Colab 只能取得已提交并推送的代码；本地未提交改动不会自动同步。

参考：[Colab 文件与运行时](https://research.google.com/colaboratory/faq.html)、[Git 进程环境配置](https://git-scm.com/docs/git-config)、[Git pull](https://git-scm.com/docs/git-pull)、[Python subprocess](https://docs.python.org/3.12/library/subprocess.html)、[IPython 显示更新](https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html)。本地 CPU 检查不能替代 Colab 上的实际下载、GPU 运行与前端显示验收。
