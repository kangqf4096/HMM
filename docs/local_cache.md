# 网络数据盘与本地双批缓存（2026-11-08）

服务器：项目、原始数据与最终结果位于 `/root/blockdata/HMM`；模型位于
`/root/models/qwen3-14b-fp8`；生成缓存位于 `/root/cache/g-*/b*`。
这是 `/root/model` 之外的新目录 `/root/models`，不会覆盖已有 OPT 模型。

先同步补丁，然后在服务器执行：

```bash
cd /root/blockdata/HMM
bash scripts/local_model.sh
df -h /root /root/cache
source scripts/env.sh
python scripts/run.py l2-generate --config configs/text_l2.json \
  --prompts data/l2a --side train --output runs/l2local-train-gen
```

上例复用 `data/l2a`，请替换为已有 prompts 目录。旧运行目录不能复用。
要从 prepare 开始执行完整 GPU 链，可使用
`bash scripts/l2_gpu.sh /root/models/qwen3-14b-fp8 l2local`。

`local_model.sh` 复制实际文件（解引用符号链接），先写入目标旁的 `.part` 目录，
成功后改名；网络原模型保留作备份，不占本地盘。没有强制哈希校验。
拷贝前要求空闲容量至少为模型大小加 2 GiB；30 GB 是总容量，系统、环境、
现有 `/root/model/opt-1.3b.tgz` 等已经占用的部分必须扣除。

正式 `text_l2.json` 已启用 `generation_storage`，模型默认使用上述本地路径。
其他配置仍按原方式运行。缓存按 `l2.generate_chunk`（默认 200 个生成请求）划分，
与 `max_num_seqs` 并发数不同。一次模型加载服务全部批次。

生成只需要准备好的 `prompts.jsonl`，不再读取原始 Parquet。为保持角色分配和请求顺序，
prompt 清单只从网络读取一次到内存；后台线程将下一批实际需要的 job/prompt
写入本地缓存，与本批计算重叠，而不是重复拷贝整个数据集。
本批 trace 写入本地，生成后立即校验 seed/context/chosen token，写出本批 sources，
再同步回传到网络结果目录的 `batches/bN`（先 `.part` 后改名），成功后删除本批缓存。
当前批回传完成后才允许预取再下一批，最多两批同时存在。
每批 trace 保存在 `batches/bN/trace`；顶层 `sources.jsonl` 在全部成功后流式合并，
后续 `l2-build` 的输入接口不变。

空间检查预留 1 GiB，并按每请求 `max(1 MiB, tokens × 4096 B)` 估算缓存预算。
500 token、200 请求时两批预算约 0.76 GiB，另加 1 GiB 余量；实际大小可能不同。
这些是保守估计和启动检查，不是文件系统配额。回传失败保留本地批和远端 `.part`；
下次不会自动删除或续跑它们，检查并归档后再使用新输出目录重跑。

随机种子保持 `config.seed + 全局请求序号`，不在每批重置；chunk 边界保持原配置。
生成的 `generation_seconds` 在缓存模式下包含预取等待、核验和回传时间。
`l2-build` 和 CPU 分析仍采用现有的内存处理及网络盘批量输出流程，
没有逐 token 网络 trace 写入；它们未改造成双批缓存。

仅同步代码到服务器不会自动搬迁服务器模型，必须执行上述迁移脚本。
