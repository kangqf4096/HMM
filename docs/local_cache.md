# blockdata 模型与内存双批缓存（2026-11-12）

服务器只有网络存储；`/root` 是 overlay，不再当作本地缓存盘。
项目与最终结果保留在 `/root/blockdata/HMM`，模型默认位于
`/root/blockdata/models/qwen3-14b-fp8`。实际目录若不同，用 `--model` 指定。
生成双批缓存使用 `/dev/shm/cache/g-*/b*`；压力转存目录为
`/root/blockdata/cache/g-*/b*`。

## 同步与模型迁移

程序、配置、脚本和说明均通过 GitHub 交付，服务器执行：

```bash
cd /root/blockdata/HMM
git pull --ff-only
source scripts/env.sh
# 仅当模型还在 overlay 且目标目录不存在时执行；已有 blockdata 模型直接复用。
bash scripts/local_model.sh /root/models/qwen3-14b-fp8 /root/blockdata/models/qwen3-14b-fp8
df -h /root/blockdata /dev/shm
```

迁移脚本解引用符号链接，先复制到目标旁 `.part`，成功后改名，保留旧模型，
待 smoke 成功后自行处理旧副本。若网络盘上已经有完整权重，直接使用它，无需重复复制。
迁移后仍以模型来源记录 `opt_hmm_source.json` 匹配已经准备的 prompts。

## 100 GB cgroup smoke

使用 **100,000,000,000 字节（十进制 100 GB）** 上限。
`scripts/cache_smoke.py` 要求 Linux cgroup v2：若当前 cgroup 的 `memory.max`
已不高于这个值，直接使用；否则尝试创建受限子组并设置 `memory.swap.max=0`。
没有 memory controller 委派权限且当前限额过大时，加载模型前退出，
需要在实验平台把容器/作业的内存硬限额设为上述值，或提供可写的受委派 cgroup。
脚本不会修改父 cgroup 的限额，也不会仅凭轮询声称实现硬上限。

复用已有 L2 prompts，例如 `data/l2q1`（换成实际目录）：

```bash
cd /root/blockdata/HMM
source scripts/env.sh
python scripts/cache_smoke.py --prompts data/l2q1 --output runs/memsmoke
# 另一个终端查看生成日志：
tail -f runs/memsmoke/run.log
```

默认使用正式 `configs/text_l2.json` 的模型、500 token 长度、并发、CUDA graph
和 chunk；只取训练侧前 `generate_chunk + 1` 个请求，默认 201，检验满批与跨批回传。
保留正式 chunk 大小，不用两个短请求替代批次内存测试。
这是生成/缓存 smoke，不执行后续 HF replay、build 或 CPU 分析；其结论只覆盖本次配置。
800 token 的实验另运行：

```bash
python scripts/cache_smoke.py --config configs/text_l2_L800.json \
  --prompts data/l2L800 --output runs/mem800
```

已有输出目录会拒绝覆盖。日志与结果保存在 `runs/memsmoke/`：
`run.log`、`memory.jsonl`、`memory.json`、`gen/`，成功才生成 `passed.json`。
统计使用 cgroup `memory.current`，包括进程、子进程、tmpfs 和文件缓存，
而不是只看主 Python RSS。`memory.events` 的前后值用于判断 OOM/oom_kill 是否增加；
已有 cgroup 的 `memory.peak` 可能含本次测试之前的历史，`peak_sampled_bytes` 为本次轮询峰值。

## 先释放缓存，再停止

默认达到 **90 GB** 时暂停 trace 写入，将本次会话的两批缓存转存到
`/root/blockdata/cache`，然后用原路径的符号链接继续读写；后续批次直接写网络盘。
不足双批预算的 `/dev/shm` 容量也会触发同一路径，不会因为常见的小 shm 挂载直接失败。
所有 trace 写入、缓存转存、核验读取和批次提交共用会话级 flock。
转存时先完成磁盘副本，再释放 tmpfs，避免删除仍在写入的审计记录。
网络写出后 fsync，并请求 Linux 丢弃对应文件页缓存；实际回收受内核和文件系统影响。
压力转存发生一次后不再回切内存，避免反复抖动。

**97 GB** 为 smoke 的提前停止线，**100 GB** 由 cgroup 硬限制兜底。
若现有限额更低，脚本自动降低转存/停止阈值。模型加载阶段没有双批缓存可释放；
若主要占用来自 eager 权重/运行时，转存不能保证避免 OOM。
转存和网络写出需要时间，轮询不能保证在瞬时峰值前完成；必须保留 cgroup 硬限制。
异常时保留未提交批次和 `.part`，不自动清空整个 `/dev/shm/cache`，也不影响其他会话。
进程被 cgroup 强杀时不会自动恢复，先检查日志与残留，再换输出目录重跑。

可在正式 smoke 后单独降低阈值来验证转存机制（此命令会很早切到网络盘，
因此不能用来代表默认内存缓存的峰值）：

```bash
python scripts/cache_smoke.py --prompts data/l2q1 --output runs/memspill --spill-gb 0.001
```

正式 `l2_gpu.sh` 和长度实验已经接入相同的 90 GB 缓存转存逻辑；
**100 GB 的硬限额必须由平台配置，或通过本 smoke 的受限子组提供**。
普通 `l2_gpu.sh` 不负责改变平台 cgroup 限额。

## 权重 eager 与结果回传

项目固定 vLLM 0.8.4/0.8.5 V0，保留 callable logits processor。
这些版本没有新版 `--safetensors-load-strategy eager` 参数。
项目通过其自定义 loader 扩展，逐个 safetensors 分片完整读入 CPU 内存后解析，
实现 eager 读取；保留 vLLM 的分片筛选、模型加载和 FP8 后处理，不升级到 V1。
所有三处 `LLM(...)` 入口均使用此 loader；`.bin/.pt` 仍走原始 PyTorch loader。
这里的 eager 是**权重加载策略**，与 `enforce_eager`（是否关闭 CUDA graph）无关。
官方新版参数说明：<https://docs.vllm.ai/en/stable/configuration/engine_args/>。
0.8.5 接口依据：<https://github.com/vllm-project/vllm/blob/v0.8.5/vllm/model_executor/model_loader/loader.py>。

每批生成后校验 seed/context/chosen token，写 sources，再回传结果目录
`batches/bN`（先 `.part` 后改名），成功后释放本批缓存；最多两批同时存在。
全部完成后流式合并 `sources.jsonl`，后续 `l2-build` 接口不变。
种子仍为 `config.seed + 全局请求序号`，不会在批次边界重置。
预算仍为每请求 `max(1 MiB, tokens × 4096 B)`，另预留 1 GiB shm 空间；
这是估计，不是实际 cgroup 内存配额。网络盘可能比 overlay 更适合顺序读取，
实际收益由 smoke 的模型加载耗时与运行日志判断。
