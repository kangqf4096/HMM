# GPU 实验运行手册（单卡 / OPT-1.3B）

适用：`~/blockdata/HMM`，vLLM 0.8.4 镜像（Python 3.10 / CUDA 12.4），单卡 FP16。
**模型固定为 OPT 系列，默认 `facebook/opt-1.3b`**（论文传统）；不混用 Qwen / Llama。
所有命令在仓库根目录执行；`source scripts/env.sh` 会自动 `cd` 到项目根并把缓存、临时目录固定在项目内。

约定：

- `MODEL`：项目内模型目录，默认 `models/opt-1.3b`（相对项目根，`--model` 会按项目根解析）。
- `TAG`：本轮短名（字母数字），决定 `data/<...>-$TAG`、`runs/<...>-$TAG-*` 的目录名。输出目录必须为空，重跑换 TAG。
- 代码改动走 git（只有文本，体积小）；**权重、数据、实验产物一律不走 git**：权重用 scp，数据在远程本地用 `prepare` 重算。

```bash
set -o pipefail            # 让 `python ... | tee` 保留 python 的失败退出码
cd ~/blockdata/HMM
source .venv/bin/activate
source scripts/env.sh
export MODEL=models/opt-1.3b
export TAG=ax01
mkdir -p tmp runs/logs models
```

---

## 0. 先确认远程有没有 OPT 权重

```bash
ls -lh ~/blockdata/HMM/models/opt-1.3b 2>/dev/null || echo "NO_MODEL"
```

需要看到 `config.json / tokenizer_config.json / vocab.json / merges.txt / pytorch_model.bin`（约 2.6 GB）。

### 没有权重：远程直接下载

```bash
cd ~/blockdata/HMM
source .venv/bin/activate
source scripts/env.sh
export HF_ENDPOINT=https://hf-mirror.com     # 直连 huggingface.co 时去掉

python scripts/download_model.py --config configs/text_smoke.json
ls -lh models/opt-1.3b
```

验证：

```bash
python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('models/opt-1.3b', local_files_only=True); print(t.vocab_size)"
```

- `download_model.py` 只接受 `facebook/opt-1.3b` 的固定 revision，下完会写 `models/opt-1.3b/opt_hmm_source.json`。
- 中断后重跑同一条命令会续传（HF 缓存保留已下完的分片）。
- 镜像也慢时 `unset HF_ENDPOINT` 走直连；完全不通外网才退回 scp：本机 `tar -czf opt-1.3b.tgz -C <权重目录> opt-1.3b`，`scp -P 37375 opt-1.3b.tgz root@ssh-cn-xibei2.ebcloud.com:~/blockdata/HMM/models/`，远程解包。

### 代码改动用 git

已提交并推送：`e0f9c79 固定 OPT 系列模型并更新实验文档`。远程执行：

```bash
cd ~/blockdata/HMM
git pull --ff-only
git log --oneline -2
```

若远程有未提交改动导致被拒，先 `git status` / `git stash` 处理，不要 `git reset --hard`。

---

## 1. 上机自检

```bash
nvidia-smi
df -h .
python scripts/test.py
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python scripts/run.py check-runtime | tee "tmp/check-runtime-$TAG.json"
```

`check-runtime` 必须返回 `vllm=0.8.4`、`engine=V0`、`devices` 非空，并完成 V0 callable 自检。失败先别往下跑。

提示符出现 `(.venv) (base)` 是正常的：conda base 在 shell 里，`.venv` 叠在它上面（`--system-site-packages`，所以镜像的 vLLM/torch 仍可见）。确认三件事就够了：

```bash
which python                      # 必须是 ~/blockdata/HMM/.venv/bin/python
python -c "import sys, vllm, torch; print(sys.prefix, vllm.__version__, torch.__version__)"
```

`sys.prefix` 是项目内 `.venv` 即可；报错栈里出现 `/root/miniconda3/lib/python3.10/...` 也只是因为 venv 复用 conda 的解释器和标准库，不代表用错了 Python。

---

## 2. 现在就能跑：CPU 模拟（不需要 GPU）

```bash
python scripts/run.py simulate --config configs/sim_smoke.json --output "runs/sim-smoke-$TAG"
python scripts/run.py simulate --config configs/sim_main.json  --output "runs/sim-main-$TAG"
python scripts/run.py simulate --config configs/sim_smoke.json --scenario island      --segment-length 20 --output "runs/island-$TAG"
python scripts/run.py simulate --config configs/sim_smoke.json --scenario hole        --segment-length 20 --output "runs/hole-$TAG"
python scripts/run.py simulate --config configs/sim_smoke.json --scenario strong_weak                    --output "runs/weak-$TAG"
python scripts/run.py simulate --config configs/sim_smoke.json --signal-p 1 0                            --output "runs/noinfo-$TAG"
```

每个目录产出 `report.md / report.json / metrics.jsonl / thresholds.json / reliability.json`。
`sim_main` 是 200 篇测试 + 2000/2000 null，最接近正式结论。

---

## 3. 数据准备（按目录复用，与 TAG 无关）

```bash
python scripts/run.py prepare --config configs/text_smoke.json --model "$MODEL" --output "data/smoke-$TAG"
```

- **提示词目录按 `data/<名字>` 复用，不看 `TAG`**。换了 `TAG` 不要重新 prepare：把 `--prompts` 指回已有的那个目录即可（`--output`/`--prompts` 是同一个目录的两种叫法）。只有换模型、换语料、改 `prompt_tokens` 或角色数量时才必须重跑。
- 首次下载 `ccdv/arxiv-summarization` document（约 3.6 GB）到 `.cache/hf`，之后复用。
- 下载失败：`export HF_ENDPOINT=https://hf-mirror.com` 后重跑；语料已在缓存时可用 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` 完全离线重算。
- 产出 `data/smoke-$TAG/{prompts.jsonl,manifest.json}`；`selected_articles` 应为 92。
- 中途失败留下的半成品目录必须先 `rm -rf` 再重跑（输出目录必须为空）。`prepare` 一进函数就建目录，所以**空目录就是上次失败留下的**，可以直接 `rmdir`。

---

## 4. smoke：流程验收（128 token，96 个请求）

```bash
python scripts/run.py generate --config configs/text_smoke.json --model "$MODEL" \
  --prompts "data/smoke-$TAG" --output "runs/smoke-$TAG-gen" --gpu-memory 0.6 \
  2>&1 | tee "runs/logs/smoke-$TAG-gen.log"

python scripts/run.py evaluate --input "runs/smoke-$TAG-gen" --output "runs/smoke-$TAG-eval"
```

验收点：

- `runs/smoke-$TAG-gen/metadata.json`：`requests=96`，`model` 是 `models/opt-1.3b`；审计通过（seed/context/pivot 不一致会直接报错）。
- 记下 `generation_seconds`：**每请求秒数 ≈ generation_seconds / 96**，用来估后面几轮。
- `runs/smoke-$TAG-eval/report.md`：5 种方法的 IoU/F1 + null any-FP（32 篇 null，区间很宽，只看是否明显偏离 0.05）。

> 1.3B 的模型计算很快，瓶颈是每 token 的 Python Gumbel 处理器；96×128 通常 2–10 分钟。

---

## 5. 枢轴分布诊断（复用同一份 prompts，需重新 generate）

```bash
python scripts/run.py generate --config configs/text_smoke.json --model "$MODEL" \
  --prompts "data/smoke-$TAG" --output "runs/piv-$TAG-gen" --pivot-diagnostics \
  --gpu-memory 0.6 2>&1 | tee "runs/logs/piv-$TAG-gen.log"

# 必须等上一个 vLLM 进程完全退出：diagnose 会用 Transformers 再加载同一模型
python scripts/run.py diagnose-pivots --input "runs/piv-$TAG-gen" --model "$MODEL" \
  --output "runs/piv-$TAG-diag" 2>&1 | tee "runs/logs/piv-$TAG-diag.log"
```

看：

- `runs/piv-$TAG-diag/report.md`：E[-log Y|P]=S2、PIT 距离、编辑后 P 的 TV、vLLM/HF 概率重算误差；
- `runs/piv-$TAG-diag/localization/report.md`：原有 5 方法 + 3 种条件发射的 IoU 与 null 误报。

诊断失败但生成成功时**不要重新生成**，换个诊断目录重跑最后一条即可。

---

## 6. pilot 预实验（500 token）

```bash
python scripts/run.py prepare  --config configs/text_pilot.json --model "$MODEL" --output "data/pilot-$TAG"
python scripts/run.py generate --config configs/text_pilot.json --model "$MODEL" \
  --prompts "data/pilot-$TAG" --output "runs/pilot-$TAG-gen" --gpu-memory 0.6 \
  2>&1 | tee "runs/logs/pilot-$TAG-gen.log"
python scripts/run.py evaluate --input "runs/pilot-$TAG-gen" --output "runs/pilot-$TAG-eval"
```

- 请求数应为 **384**（32 校准 + 16 开发×2 + 32 测试×2 + 128 null 校准 + 128 null 测试）。
- 看 `report.md` 的 any-FP 是否接近 0.05、IoU 是否随长度提升。

---

## 7. 正式规模（2000 null 校准 / 2000 null 测试，可选）

先确认磁盘 ≥ 40 GB（模型 2.6 + 语料 3.6 + 产物约 5）。

生成配置（= pilot 的 500 token 设置 + 2000/2000 null）：

```bash
python - <<'PY'
import json
config = json.load(open("configs/text_pilot.json"))
config["roles"]["null_cal"] = 2000
config["roles"]["null_test"] = 2000
json.dump(config, open("configs/text_formal.json", "w"), indent=2)
PY
```

```bash
python scripts/run.py prepare  --config configs/text_formal.json --model "$MODEL" --output "data/formal-$TAG"
python scripts/run.py generate --config configs/text_formal.json --model "$MODEL" \
  --prompts "data/formal-$TAG" --output "runs/formal-$TAG-gen" --gpu-memory 0.6 \
  2>&1 | tee "runs/logs/formal-$TAG-gen.log"
python scripts/run.py evaluate --input "runs/formal-$TAG-gen" --output "runs/formal-$TAG-eval"
```

- 请求数应为 **4128**，生成 token 约 2.264 M（null 按 550 生成长度）。
- 时间 ≈ smoke 每请求秒数 × 4128 × 4.3；1.3B 上主要仍取决于逐 token 的 Python 处理器，通常 1–4 小时。

---

## 8. 建议执行顺序

| 顺序 | 任务 | 需要 GPU | 预计 |
|---|---|---|---|
| 1 | 6 个 CPU 模拟 | 否 | 10–40 分钟 |
| 2 | `check-runtime` | 是 | 1–2 分钟 |
| 3 | `prepare` smoke | 否（CPU+网络） | 5–20 分钟 |
| 4 | smoke generate + evaluate | 是 | 2–10 分钟 |
| 5 | pivot generate + diagnose | 是 | 5–20 分钟 |
| 6 | pilot prepare + generate + evaluate | 是（prepare 不用） | 20–60 分钟 |
| 7 | formal（可选） | 是 | 1–4 小时 |

长任务建议放 tmux，避免 SSH 断线：

```bash
tmux new -s opt
# 断线后：tmux attach -t opt
```

---

## 9. 每轮清理（`docs/storage_budget.md`）

```bash
# generate 自检通过后（trace 已无消费者）
rm -rf runs/<name>-gen/trace
rm -f  runs/<name>-gen/sources.jsonl      # 要做 diagnose 就先留着

# evaluate 成功后
rm -f runs/<name>-eval/predictions.jsonl

# diagnose-pivots 成功后
rm -f runs/<name>-diag/positions.jsonl
rm -rf runs/<name>-diag/localization
```

保留：`observations.jsonl`、`metrics.jsonl`、`report.*`、`density.json`、`thresholds.json`、`reliability.json`。

---

## 10. 结果检查清单

- [ ] `check-runtime` 通过（vLLM 0.8.4 / V0 / CUDA / devices）
- [ ] `metadata.json` 的 `model` 以 `models/opt-1.3b` 结尾，`key_id` 为 `public-research-v1`
- [ ] `metadata.json` 的 `key` 与本轮实际使用的密钥一致（研究阶段明文写入；部署前改用 `key_fingerprint`，见 `docs/reproducibility_and_secrets_20261101.md`）
- [ ] `metadata.json` 的 `enforce_eager` 为 `false`（即模型前向走 CUDA graph）；与上一轮不一致时先确认原因再比耗时
- [ ] smoke `requests=96`、pilot `requests=384`、formal `requests=4128`
- [ ] smoke/pilot `report.md`：null any-FP 不是 0 或 1，IoU 有数值
- [ ] pivot `report.md`：先看 vLLM vs HF 重算误差，再解释条件发射增益
- [ ] 所有轮次 `metadata.json` 的 `config.tokens` 与配置一致

---

## 11. 常见故障

| 现象 | 处理 |
|---|---|
| `CUDA out of memory` | 降 `--gpu-memory 0.5`；1.3B 正常不会 OOM，先查是不是别的进程占卡 |
| `Local model directory does not exist` | 第 0 节没做或路径写错；`--model` 是相对项目根解析的 |
| `Tokenized prompts belong to a different model` | prepare 和 generate 的 `--model` 不一致；用同一路径重跑 prepare |
| `Output is not empty` | 换新 `TAG`，或 `rm -rf` 该输出目录后重跑 |
| HF 下载失败 | `export HF_ENDPOINT=https://hf-mirror.com` |
| vLLM 版本不对 | 只能用镜像自带 0.8.4；不要 `pip install vllm` 覆盖 |
| `diagnose-pivots` OOM / 报缺 CUDA | 确认上一个 vLLM 进程已退出，再单独跑诊断 |
| CUDA graph 捕获失败 / 显存不够建图 | 在配置里设 `"enforce_eager": true` 回退，并在轮次记录里注明；不要用环境变量临时切换 |
| 换图后耗时反而变长 | 先看是不是这一轮开了 `--pivot-diagnostics`；逐 token 的 Python 处理通常比前向更贵 |

---

## 12. 时间外推公式

```bash
python - <<'PY'
import json
m = json.load(open("runs/smoke-<TAG>-gen/metadata.json"))
print(m["generation_seconds"], m["generation_seconds"]/96)
PY
```

- pilot 预估 ≈ `(秒/请求) × 384 × 3.9`（500/128 token）
- formal 预估 ≈ `(秒/请求) × 4128 × 4.3`（平均生成长度约 548/128）
