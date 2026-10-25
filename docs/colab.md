# Colab：先挂载 Drive，再运行 OPT-1.3B 与 arXiv

可直接上传 [Notebook](../notebooks/colab_arxiv.ipynb)，也可以将下列单元格逐个复制到 Colab。

| 内容 | 位置 |
|---|---|
| Git checkout、模型、普通 arXiv 原始分片、输入/输出、日志 | `/content/drive/MyDrive/opt-hmm`，持久保存 |
| Python 虚拟环境、语料工作副本、pip/vLLM 缓存、进程临时目录 | `/content/opt-hmm`，每次会话可重建 |

模型和实验资料持久存入 Drive；arXiv 普通原始分片放在 data/arxiv/数据版本/。读取时准备运行时本地副本，不依赖 Drive 上的 Hub snapshot 符号链接。

Git 管理共享代码，Google Drive 保存项目、权重、语料缓存、实验输入、输出和日志，Colab 提供计算。先在菜单“运行时 → 更改运行时类型”选择 GPU，然后从第一个单元格开始执行。

仓库需要先推送最新代码；GitHub token 每次通过隐藏输入框输入，不写入 Notebook、项目文件、remote URL 或凭据缓存。

项目固定在 `/content/drive/MyDrive/opt-hmm`。Python 环境、进程临时文件和原始语料的可重建工作副本使用 `/content/opt-hmm`，可在新会话重建。Drive 目录中的文件持续保留，但内存和正在执行的程序不会随着挂载恢复。

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

def project_root(root=None):
    project = Path(root or os.environ.get("OPT_HMM_PROJECT", "/content/drive/MyDrive/opt-hmm")).resolve()
    if project.as_posix().startswith("/content/drive/") and not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("先运行第 1 步 drive.mount('/content/drive')，再使用 Drive 项目目录。")
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

PROJECT = github_sync(PROJECT)
os.chdir(PROJECT)
```

## 加载进度和复制工具
克隆成功后运行。后续 pull 更新辅助代码后，重新运行这一格。

```python
import runpy

tools = runpy.run_path(str(PROJECT / "scripts/colab_tools.py"))
run_step = tools["run_step"]
github_sync = tools["github_sync"]
copy_runtime_assets = tools["copy_runtime_assets"]
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

## 第 5 步：准备普通 arXiv 分片，再从本地副本抽样
原始 Parquet 以普通文件持久保存在 PROJECT/data/arxiv/数据版本/document 下；运行时工作副本保存在 RUNTIME/.cache/arxiv/数据版本/document 下。数据量约 3.61 GB，Drive 和运行时各需要一份空间；旧 Hub/Arrow 缓存不自动删除。

程序优先复用已准备的普通文件，再尝试从旧 Hub 缓存的文本链接中找回真实 blob；没有可用数据时才下载到本地普通文件，然后复制到 Drive，复制显示字节进度。后续 ParquetFile 只读取本地工作副本，避免直接读取 Drive 的 Hub snapshot 链接。新会话从 Drive 普通文件恢复本地副本即可。

抽样仍以全部 215913 篇官方论文的相同行顺序和随机种子进行，整篇去重后选择 92 篇，保留 source_split/source_row。不会重新构建完整 Arrow 语料；报告 tmp/arxiv-files.json 记录旧链接实际指向的内容、采用的源路径及本地/Drive 普通文件路径。

已有完整 ax0 输入可以继续使用；新的实验用 ax1 等短名称，未完成目录不会自动删除。只准备资源时到此即可。

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

## 已确认的故障：Drive 中的 Hub snapshot 读到了文本链接

本次实际输出显示 train-00000-of-00015.parquet 虽为符号链接，但打开后大小只有 79 字节，内容是 `../../blobs/9b/...`，没有 PAR1 头尾。这确认了当前缓存路径没有读到真实的 Parquet 文件。先前“数据已坏”和“仅是 datasets fragment 扫描”的判断都不足以解释该证据；更换读取 API 不会把文本链接变成数据。

当前第 5 步先准备普通文件：优先沿受限的旧缓存 blob 路径找回已有真实数据，找不到时下载到运行时的 local_dir；再复制普通字节文件到 Drive 的 data/arxiv/版本目录。读取时使用本地工作副本，Drive 继续作为持久存储。旧缓存保留；不需要重新下载模型，不做整文件哈希校验。

拉取代码后重新加载进度工具，再重跑第 5 步即可，无需先运行旧的缓存重下流程。下面 repair_arxiv 入口也已改为同一普通文件准备流程，可单独执行并查看 tmp/arxiv-cache.json；它不再把修复结果只写回 Drive 中的 Hub 链接缓存。

新版应出现“复制 arXiv 到本地工作目录”“保存普通 arXiv 分片到 Drive”“读取 arXiv 行组元数据”“抽取独立论文”等阶段。文件已经就绪时跳过复制。即使 Colab 断开，本地工作副本消失，Drive 中原始数据、实验输入和结果仍保留。

```python
repair_log = run_step(
    "scripts/repair_arxiv.py", "--config", "configs/colab_arxiv.json",
    title="检查并修复 arXiv 缓存", log_name="arxiv-fix", root=PROJECT,
)
```

参考：[Hub 缓存与 local_dir 下载](https://huggingface.co/docs/huggingface_hub/package_reference/file_download)、[Colab Drive 读写建议](https://research.google.com/colaboratory/faq.html)、[ParquetFile](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html)。本地使用真实 PyArrow 26.0.0 验证了文本链接恢复、普通文件保存和读取、重复运行及抽样一致性；尚需用户的 Colab 会话验收。
