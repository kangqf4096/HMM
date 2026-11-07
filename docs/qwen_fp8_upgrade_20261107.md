# RTX 4090 24GB：Qwen3-14B-FP8 的最小升级方案

推荐目标是 vLLM 0.8.5 + PyTorch 2.6.0/CUDA 12.4 + Transformers 4.51.3，继续启用 V0。版本依据上游依赖和模型说明核对；本地 Windows 只能执行 CPU 回归，远程 GPU 的实际验收按下文执行。

## 已知现状

L2a 的生成元数据记录：Python 3.10.16，vLLM 0.8.4/V0，PyTorch 2.6.0+cu124，CUDA runtime 12.4，Transformers 4.51.3，NumPy 2.2.6，datasets 4.8.5，PyArrow 25.0.1，huggingface-hub 0.36.2；GPU 为 RTX 4090，驱动报告可用总量约 23.52 GiB。Tokenizers、Triton、xFormers、Accelerate 的旧实际版本未记录，不能根据配置范围推定。

项目原配置为 Python >=3.10,<3.13，vLLM ==0.8.4，Transformers >=4.51.1,<4.56，datasets >=3,<5，PyArrow >=21。更新后允许 vLLM 0.8.4/0.8.5；其他范围保留。

候选固定版本清单见 [requirements/qwen_fp8.txt](../requirements/qwen_fp8.txt)。其中 PyTorch 2.6.0、TorchAudio 2.6.0、TorchVision 0.21.0、xFormers 0.0.29.post2 来自 vLLM 0.8.5 官方 CUDA 依赖；Triton 3.2.0 与 PyTorch 2.6.0 对应。FP8 回放补充 Accelerate 1.6.0，Tokenizers 固定 0.21.1。

## 先同步本次补丁

本机项目根目录执行（Linux scp 语法；Windows OpenSSH 也可执行这条相对路径命令）：

```bash
scp -P 37375 tmp/q3patch.tar.gz root@ssh-cn-xibei2.ebcloud.com:/root/blockdata/HMM/
```

服务器执行：

```bash
cd /root/blockdata/HMM
tar -xzf q3patch.tar.gz
```

补丁只包含运行时兼容代码、FP8 回放加载、L2 入口可选参数、固定版本清单、检查脚本和测试，不包含原 configs/text_l2.json、实验数据或权重。

## 在新环境安装

先在上一轮实际使用的 Python 环境中执行。不要直接对 conda base 或旧 .venv 运行升级。

```bash
cd /root/blockdata/HMM
source scripts/env.sh
mkdir -p tmp/q3
python -m pip freeze > tmp/q3/old-freeze.txt
python -c 'import sys,torch; print(sys.executable); print(sys.version); print(torch.__version__, torch.version.cuda)'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

python -m venv .venv-q3
source .venv-q3/bin/activate
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install --only-binary=vllm,torch,xformers \
  -r requirements/qwen_fp8.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install --no-deps -e .
python -m pip check
export VLLM_USE_V1=0
python scripts/run.py check-runtime
python -m pip freeze > tmp/q3/new-freeze.txt
```

`.venv-q3` 是独立环境，没有继承旧环境的 site-packages；这能避免旧 torch 扩展被混用。代价是需要安装该环境内的 GPU 依赖，占用额外磁盘并可能下载数 GB。首次安装不要与生成任务并发占用 GPU。`--only-binary` 确保这些大组件没有预编译 wheel 时明确失败，不会悄悄开始长时间源码编译。

使用官方兼容的 CUDA 12.4 预编译 wheel，不需要先安装新的系统 CUDA Toolkit；当前驱动能运行 torch 2.6.0+cu124 已有实际证据。若下载镜像缺少 wheel，可以保持同一版本清单改用可访问的 PyPI 源，而不是取消版本限制。

每次新登录后：

```bash
cd /root/blockdata/HMM
source .venv-q3/bin/activate
source scripts/env.sh
export VLLM_USE_V1=0
```

## `load_qkv_weight` 越界：先确认实际环境

若加载 Qwen3 FP8 时在 `parameter.py` 的 `load_qkv_weight` / `narrow` 报
`IndexError: start out of range`（例如维度为 56，偏移却为 5120），这是
[vLLM 官方记录的 0.8.4 FP8 加载缺陷](https://github.com/vllm-project/vllm/issues/17327)。
[0.8.5 的修复](https://github.com/vllm-project/vllm/pull/17318)会把 QKV 的权重偏移转换为 FP8 scale 的块偏移。
原 OPT 非量化加载不经过这条路径，所以旧实验能跑不代表新模型能使用同一版本。

在发生错误的同一个终端确认 Python、版本和导入位置：

```bash
python -c 'import sys,vllm; print(sys.executable); print(vllm.__version__); print(vllm.__file__)'
```

如果显示 `/root/miniconda3/lib/...` 和 `0.8.4`，检查是否尚未激活已经安装好的 `.venv-q3`。
若新环境尚未安装，按上文「在新环境安装」完成安装。切换后确认实际版本是 `0.8.5`，再运行：

```bash
python scripts/run.py check-runtime --model models/opt-1.3b
python scripts/check_fp8.py models/opt-1.3b
```

L2 GPU 入口现会读取本地模型的 `config.json`，在加载权重前拒绝 Qwen3 FP8 + vLLM 0.8.4；
这也适用于 `models/opt-1.3b` 软链接入口。普通 OPT 仍可使用 0.8.4。
之前的失败若已留下非空的 `*-gen` 输出目录，重跑时需先将失败目录改名保留，或使用新的 run-name。

回退时退出新环境并激活原先使用的 conda 环境或旧 `.venv`；本方案没有对它们安装或卸载包。

## 通过 hf-mirror 匿名下载

在新环境中执行，不需要 Hugging Face 登录。模型先放在独立目录，避免与原 OPT 的 pytorch_model.bin 或 tokenizer 文件混合。

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_XET=1

python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

repo = "Qwen/Qwen3-14B-FP8"
revision = HfApi().model_info(repo, token=False).sha
folder = Path("models/qwen3-14b-fp8")
snapshot_download(
    repo_id=repo, revision=revision, token=False,
    local_dir=str(folder), max_workers=4,
    allow_patterns=["*.json", "*.safetensors", "vocab.json", "merges.txt", "LICENSE*", "README*"],
)
(folder / "opt_hmm_source.json").write_text(json.dumps({
    "repo_id": repo, "revision": revision,
    "url": f"https://huggingface.co/{repo}/tree/{revision}"
}, indent=2) + "\n", encoding="utf-8")
print("Downloaded:", folder.resolve(), revision)
PY
```

该 revision 用于记录模型来源，不是要求额外执行文件哈希校验。下载中断后重跑会复用已下载文件。

## 检查两个 GPU 阶段

```bash
python scripts/check_fp8.py models/qwen3-14b-fp8
```

脚本先加载 vLLM，在同 prompt 上各生成一份 W/U，检查每步 V0 回调与水印赢家；该进程退出释放 GPU 后，再用 Transformers FP8 回放验证概率有限且归一。默认只生成 128 token；可以添加 `--tokens 500` 检查本实验长度。结果保留在项目 `tmp/q3-*/`，正文文件为 samples.txt。

两个步骤通过才表明当前服务器具备基本运行条件；它不检验自然语言质量、不保证长时间运行无 OOM，也不证明 vLLM 与 Transformers 的 FP8 数值完全一致。正式实验前仍应读取 W/U 正文，检查重复循环是否改善。

## 保留旧模型入口，使用原 L2 流程

模型检查通过后，可以给 Qwen 保留 `models/opt-1.3b` 这个目录入口名；配置和清单仍记录真实模型来源。第一次切换时执行以下单个 bash 块，它会拒绝覆盖已有备份或链接：

```bash
(
  set -e
  test -d models/qwen3-14b-fp8
  test ! -L models/opt-1.3b
  test ! -e models/opt-1.3b.opt-backup
  mv models/opt-1.3b models/opt-1.3b.opt-backup
  ln -s qwen3-14b-fp8 models/opt-1.3b
)
```

生成独立配置，保留原 text_l2.json：

```bash
python - <<'PY'
import json
from pathlib import Path

config = json.loads(Path("configs/text_l2.json").read_text(encoding="utf-8"))
source = json.loads(Path("models/qwen3-14b-fp8/opt_hmm_source.json").read_text(encoding="utf-8"))
config.update(model="models/opt-1.3b", model_repo=source["repo_id"],
              model_revision=source["revision"], max_num_seqs=1, enforce_eager=True)
Path("configs/text_l2_q3.json").write_text(json.dumps(config, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
PY

CONFIG=configs/text_l2_q3.json DTYPE=bfloat16 WORKERS=10 \
PROGRESS=runs/l2q1-progress.json WEIGHTS=runs/l2q1-weights.json \
bash scripts/l2_full.sh models/opt-1.3b l2q1
```

`dtype=bfloat16` 是非量化部分和激活的 dtype；模型配置中的 FP8 权重量化仍保留，不是把 14B 全部展开为 BF16。先用单并发和 eager 减少 4090 显存压力，再根据实际运行决定是否提高并发。

流程仍是原 L2 各阶段，但换 tokenizer 后必须重新 prepare、generate、build、fit，不能接着使用 l2a 的 OPT token IDs、f1、阈值或结果目录。仅替换权重不保证消除上下文派生水印的重复循环；本轮质量检查仍是必要的。

## 版本依据

- [Qwen 官方 FP8 模型说明](https://huggingface.co/Qwen/Qwen3-14B-FP8)：Transformers >=4.51.0，vLLM >=0.8.5。
- [vLLM 0.8.5 CUDA 依赖](https://github.com/vllm-project/vllm/blob/v0.8.5/requirements/cuda.txt)：torch/torchaudio 2.6.0、torchvision 0.21.0、xformers 0.0.29.post2。
- [vLLM 0.8.5 公共依赖](https://github.com/vllm-project/vllm/blob/v0.8.5/requirements/common.txt)：Transformers >=4.51.1、Tokenizers >=0.21.1、HF Hub >=0.30。
- [vLLM 0.8.5 安装说明](https://docs.vllm.ai/en/v0.8.5/getting_started/installation/gpu.html)：预编译 CUDA 12.4 路径与二进制依赖匹配。
- [Transformers 4.51.3 FP8 环境检查](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/quantizers/quantizer_finegrained_fp8.py)：要求 Accelerate、GPU capability >=8.9，包含 RTX 4090。
