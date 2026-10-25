# Colab：从持久目录运行 OPT-1.3B 与 arXiv

可直接上传 [Notebook](../notebooks/colab_arxiv.ipynb)，也可将下列单元格逐个复制到 Colab。

| 内容 | 默认位置 |
|---|---|
| 项目代码、配置、模型、普通 arXiv 分片、输入输出、日志 | `/content/drive/MyDrive/opt-hmm` |
| Python 环境、pip/vLLM 缓存、进程临时文件 | `/content/opt-hmm` |

相对资源路径由 opt_hmm.paths 统一按项目根目录解析，不依赖 Notebook 当前工作目录；Colab 中遗留的 /content 或 /content/opt-hmm 项目变量会自动纠正为持久目录。原始语料默认直接读取 Drive 中的普通 Parquet 文件；不会因为 OPT_HMM_RUNTIME 指向本地运行时就改读本地语料副本。

Git 管理共享代码，Google Drive 保存项目、权重、语料缓存、实验输入、输出和日志，Colab 提供计算。先在菜单“运行时 → 更改运行时类型”选择 GPU，然后从第一个单元格开始执行。

仓库需要先推送最新代码；GitHub token 每次通过隐藏输入框输入，不写入 Notebook、项目文件、remote URL 或凭据缓存。

项目固定在 `/content/drive/MyDrive/opt-hmm`。Python 环境和进程临时文件使用 `/content/opt-hmm`，模型和语料默认直接从 Drive 项目目录加载，可在新会话重建。Drive 目录中的文件持续保留，但内存和正在执行的程序不会随着挂载恢复。

## 第 1 步：先挂载 Google Drive
运行下格，按界面提示授权。确认打印的持久目录包含 `/drive/MyDrive/`。后续所有代码使用 PROJECT 变量；仅执行 mount 而继续向原 `/content/opt-hmm` 写文件不会使旧目录持久化。

```python
from google.colab import drive
from pathlib import Path
import os

drive.mount("/content/drive")
assert Path("/content/drive/MyDrive").is_dir(), "Drive 尚未挂载成功"

PROJECT = Path("/content/drive/MyDrive/opt-hmm")
RUNTIME = Path("/content/opt-hmm")
os.environ["OPT_HMM_PROJECT"] = str(PROJECT)
os.environ["OPT_HMM_RUNTIME"] = str(RUNTIME)

print("持久项目目录：", PROJECT)
print("本次运行环境：", RUNTIME)
```

```python
import subprocess
subprocess.run(["nvidia-smi"], check=True)
```

## 第 2 步：克隆或更新 Drive 中的私有仓库
创建 [GitHub 只读 token](https://github.com/settings/personal-access-tokens/new?name=Colab-HMM&target_name=kangqf4096&contents=read&expires_in=7)：选择 `kangqf4096`、Only select repositories → HMM、Contents → Read-only。

首次运行 clone 到 PROJECT；下次挂载后运行同一格，会拉取当前分支。每次运行时粘贴 token 并回车，使用 `pull --ff-only`，不保存凭据。Colab 只会取得已经推送的文件。

```python
import base64
import os
import subprocess
from getpass import getpass
from pathlib import Path

COLAB_PROJECT = "/content/drive/MyDrive/opt-hmm"
COLAB_RUNTIME = "/content/opt-hmm"
CHECKOUT_ROOT = Path(COLAB_PROJECT)

def on_colab():
    return os.name == "posix" and Path("/content").is_dir()

def project_root(root=None):
    selected = root or os.environ.get("OPT_HMM_PROJECT")
    if on_colab():
        if not selected or str(selected).rstrip("/") in ("/content", COLAB_RUNTIME):
            selected = COLAB_PROJECT
    project = Path(selected or CHECKOUT_ROOT).expanduser().resolve()
    if project.as_posix().startswith("/content/drive/") and not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("先运行 drive.mount('/content/drive')，再使用持久项目目录。")
    return project

def github_sync(root=None):
    """Clone or fast-forward the current branch, prompting for a token each time."""
    project = project_root(root)
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
os.environ["OPT_HMM_PROJECT"] = str(PROJECT)
os.chdir(PROJECT)
print("当前项目目录：", PROJECT)
```

## 加载进度和复制工具
克隆成功后运行。后续 pull 更新辅助代码后，重新运行这一格。

```python
import importlib
import os
import runpy
import sys
from pathlib import Path

PROJECT = Path("/content/drive/MyDrive/opt-hmm")
assert Path("/content/drive/MyDrive").is_dir(), "请先挂载 Google Drive"
assert (PROJECT / "scripts/colab_tools.py").is_file(), "请先将新版仓库克隆或更新到 Drive"
os.environ["OPT_HMM_PROJECT"] = str(PROJECT)
os.environ["OPT_HMM_RUNTIME"] = "/content/opt-hmm"
os.chdir(PROJECT)

# Reload updated package code if this notebook previously imported it before pull.
sys.path.insert(0, str(PROJECT / "src"))
for name in list(sys.modules):
    if name == "opt_hmm" or name.startswith("opt_hmm."):
        del sys.modules[name]
importlib.invalidate_caches()
tools = runpy.run_path(str(PROJECT / "scripts/colab_tools.py"))
run_step = tools["run_step"]
github_sync = tools["github_sync"]
copy_runtime_assets = tools["copy_runtime_assets"]
print("项目目录：", tools["project_root"]())
print("Python 环境：", Path("/content/opt-hmm/.venv/bin/python"))
```

## 当前会话已有下载文件时：复制到 Drive（仅首次切换需要）
如果旧 `/content/opt-hmm` 中已经有模型、数据或实验结果，在运行中的下载/生成结束后执行下格。它复制 `models/`、`data/`、`runs/`、`.cache/hf/` 和顶层日志，保留源文件，跳过 Drive 中已存在的文件；不会复制旧 `.git` 或 `.venv`。这是复制步骤，复制大文件时当前文件需要一段时间才完成。

已有完整 OPT 准备输入在移到 Drive 后仍可按记录的模型 repo/revision 复用；缺少模型来源记录或换了模型版本时需在新目录重新 prepare。未完成的输出目录仍需另取新名字。

如果是全新的会话，没有旧资源，跳过此格即可。务必等复制完成后再断开原会话。

```python
migration = copy_runtime_assets("/content/opt-hmm", root=PROJECT)
```

## 第 3 步：安装本次运行环境
依赖安装到运行时 `/content/opt-hmm/.venv`，源代码和持久资源仍在 Drive。避免将大量环境文件与 vLLM 进程通信文件放进 Drive 挂载目录。新会话需要重新安装环境，权重和数据可复用。

整块作为一个单元格执行，`%%bash` 位于第一行。它继承第 1 步设置的项目路径。

```bash
%%bash
set -euo pipefail
export OPT_HMM_PROJECT=/content/drive/MyDrive/opt-hmm
export OPT_HMM_RUNTIME=/content/opt-hmm
cd "$OPT_HMM_PROJECT"
source scripts/env.sh

if ! python3 -m venv "$OPT_HMM_RUNTIME/.venv"; then
    python3 -m pip install virtualenv
    python3 -m virtualenv "$OPT_HMM_RUNTIME/.venv"
fi
"$OPT_HMM_RUNTIME/.venv/bin/python" -m pip install -e ".[inference]"
"$OPT_HMM_RUNTIME/.venv/bin/python" scripts/test.py
"$OPT_HMM_RUNTIME/.venv/bin/python" scripts/run.py check-runtime
```

## 第 4 步：下载 OPT-1.3B 到 Drive
模型写入 PROJECT 下的 `models/opt-1.3b`，固定版本权重约 2.63 GB。显示 Hugging Face 报告的下载进度；已有资源会复用下载状态。

```python
model_log = run_step(
    "scripts/download_model.py",
    title="第 4 步：下载 OPT-1.3B 到 Drive", log_name="model", root=PROJECT,
)
```

## 第 5 步：从 Drive 普通 arXiv 分片抽样
Parquet 普通文件位于 PROJECT/data/arxiv/数据版本/document 下，加载器默认直接读取这里；OPT_HMM_RUNTIME 仅指定 Python 环境和进程临时目录，不会切换资源加载路径。

优先复用持久目录的普通 Parquet 文件，其次从旧 Hub 缓存中的链接目标找回真实 blob；没有可用文件时才下载为持久目录中的普通文件。数据约 3.61 GB，不再默认创建运行时工作副本；旧缓存和旧工作副本保留。

抽样仍以全部 215913 篇论文的官方行顺序及相同随机种子进行，整篇去重后选取 92 篇。tmp/arxiv-files.json 中 archive 和 work 默认都是 Drive 目录。已有完整 ax0 输入可以复用；新实验用 ax1 等短名称，未完成目录不会自动删除。只准备资源时到此即可。

```python
if all((PROJECT / "data/ax0" / name).is_file()
       for name in ("manifest.json", "prompts.jsonl")):
    print("已有完整的 data/ax0，将复用；生成前会核对配置及模型来源版本。")
else:
    prepare_log = run_step(
        "scripts/run.py", "prepare", "--dataset-loader", "parquet_file",
        "--config", "configs/colab_arxiv.json", "--output", "data/ax0",
        title="第 5 步：准备 arXiv 到 Drive", log_name="prepare", root=PROJECT,
    )
```

## 第 6 步：生成并评价，输出直接写入 Drive
第 4 步起，子进程采用 PIPE 和 Python `-u`，独立线程处理日志中的换行和回车刷新，约每 0.5 秒更新同一个面板。Colab 前端仍可能合并刷新，不能保证逐 token 即时显示。

有可靠数量时显示当前阶段的真实进度；加载模型等阶段显示活动条和耗时。完整日志直接写入 PROJECT/tmp。当前 ax0 的完整输出已存在时跳过，不把旧结果算作新实验；需要新实验时统一修改输入和输出名字。

```python
if all((PROJECT / "runs/ax0-gen" / name).is_file()
       for name in ("metadata.json", "sources.jsonl", "observations.jsonl")):
    print("已有 ax0 的生成输出，继续评价；新实验请改用 ax1 等新目录。")
else:
    generation_log = run_step(
        "scripts/run.py", "generate",
        "--config", "configs/colab_arxiv.json", "--prompts", "data/ax0",
        "--output", "runs/ax0-gen",
        title="第 6 步：生成实验文本", log_name="generate", root=PROJECT,
    )
```

```python
if all((PROJECT / "runs/ax0-eval" / name).is_file()
       for name in ("report.md", "report.json", "predictions.jsonl")):
    print("已有 ax0 的评价报告，可直接查看；重新评价请使用新的输出目录。")
else:
    evaluation_log = run_step(
        "scripts/run.py", "evaluate",
        "--input", "runs/ax0-gen", "--output", "runs/ax0-eval",
        title="第 6 步：校准与评价", log_name="evaluate", root=PROJECT,
    )
```

## 第 7 步：查看 Drive 中的报告

```python
from IPython.display import Markdown, display

display(Markdown((PROJECT / "runs/ax0-eval/report.md").read_text(encoding="utf-8")))
```

## 第 8 步：可选下载归档
模型、语料缓存、输入、结果和日志已经直接写入 Drive，无需结束时再搬运。也可以用下格将本次实验输入和输出打包到 Drive 下 `runs/ax0.tar.gz`，并下载一份到本机。Notebook 本身可通过“文件 → 在云端硬盘中保存副本”保存；Git 中保留共享版本。

```python
import subprocess
from google.colab import files

archive = PROJECT / "runs/ax0.tar.gz"
subprocess.run([
    "tar", "-czf", str(archive), "-C", str(PROJECT),
    "data/ax0", "runs/ax0-gen", "runs/ax0-eval",
], check=True)
files.download(str(archive))
```

## 后续 pull
每次新会话先挂载 Drive，再运行第 2 步；同一会话更新可运行下格。每次重新输入 token。模型、数据、缓存和输出被 Git 忽略，正常 pull 会保留它们。本地 Windows 仓库的 remote 不受影响。

更新后重跑“加载进度和复制工具”；依赖声明变化时重跑第 3 步。请在当前实验结束后更新代码。

```python
PROJECT = github_sync(PROJECT)
```

## 查看完整日志
日志在 PROJECT/tmp 下，不受运行时本地临时目录清理影响；同一步重跑时追加。子进程失败会抛出异常并保留日志，点击停止时尝试结束该步骤的进程组。

```python
print((PROJECT / "tmp/prepare.log").read_text(encoding="utf-8")[-12000:])
```

## Drive 中旧 Hub snapshot 的文本链接
此前的错误缓存打开后是 ../../blobs/... 文本，没有 PAR1 头尾。加载器会优先从实际 blob 找回字节，保存成 PROJECT/data/arxiv/版本/document 下的普通文件，并默认直接读取持久目录。

拉取更新后重新运行“加载进度和复制工具”，清除旧的路径变量和已导入包。相对配置、模型、输入输出及日志全部按 Drive 项目解析；即使旧 OPT_HMM_PROJECT 值为 /content 或 /content/opt-hmm，在 Colab 中也会转回持久项目。Python 命令位于运行时 .venv 并不表示资源从运行时加载。

repair_arxiv 使用相同的普通文件准备流程，报告 tmp/arxiv-cache.json 列出实际读取路径，不做整文件哈希，也不删除旧缓存。

```python
repair_log = run_step(
    "scripts/repair_arxiv.py", "--config", "configs/colab_arxiv.json",
    title="检查并修复 arXiv 缓存", log_name="arxiv-fix", root=PROJECT,
)
```

参考：[Colab 文件与 Drive 挂载](https://research.google.com/colaboratory/faq.html)。本地路径测试不代替真实挂载和 GPU 验收。
