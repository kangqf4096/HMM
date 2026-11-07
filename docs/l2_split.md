# L2 分开运行 GPU 与 CPU 阶段

在项目根目录、已激活的 Python 环境下运行以下 Linux 命令。

## 1. GPU 阶段

```bash
bash scripts/l2_gpu.sh models/opt-1.3b l2a
```

执行运行时检查、prompt 准备、训练侧生成与回放构建、测试侧生成与回放构建。
训练侧先构建，以确定测试侧 `alt_top` 场景使用的 `L_top`。
此阶段结束后即可释放 GPU。

### 完整阶段恢复

重复执行相同命令即可接着运行，使用同一个 run-name、配置和 `DTYPE`：

```bash
bash scripts/l2_gpu.sh /root/models/qwen3-14b-fp8 l2a
```

脚本分别检查 prompt 准备、训练生成、训练构建、测试生成、测试构建。
生成阶段要求 `sources.jsonl` 和最终 `metadata.json` 完整存在；构建阶段要求
`positions.jsonl`、`articles.jsonl` 和最终 `manifest.json` 完整存在。
已完成的阶段显示 `[skip]`，直接复用已有结果，包括此功能加入前生成的完整结果。
跳过时也会补齐总进度文件中该阶段的计数，已有计时会保留。

未完成的阶段显示 `[restart]`，将原输出目录改名为同一位置的 `.old1`、`.old2` 等，
然后从头执行该阶段；旧目录中的批次和 trace 都会保留。
例如训练生成已完成、训练构建中断，重跑会跳过训练生成，只重跑训练构建并继续后续阶段。
恢复粒度是完整阶段：生成中断时整个训练或测试生成阶段重跑，构建中断时整个对应构建阶段重跑。

恢复会核对保存的配置、模型来源、训练/测试侧及生成 dtype；不一致时停止并要求新 run-name。
同一固定模型的目录搬迁及本地缓存位置变化允许复用。若上游阶段重新执行，依赖它的构建阶段也会重新执行。
所有旧目录均保留，不会自动清理 `.oldN`。

## 2. CPU 阶段

```bash
WORKERS=10 bash scripts/l2_cpu.sh l2a
```

依次执行机制检查、训练拟合与 dev 选择、校准和冻结、一次测试评价、cap 敏感性分析、进度权重导出。
脚本将 `CUDA_VISIBLE_DEVICES` 设为空，不检查 GPU 运行时，也不加载语言模型。
`WORKERS` 传给拟合和测试评价的 SeedBS 进程池，不代表所有 CPU 步骤都会使用 10 个进程。

测试侧数据可按实验计划 §6 提前生成与构建。机制检查、拟合与参数选择仅读取训练侧；
`l2-fit` 写出 `runs/l2a-fit/frozen.json` 后，才开始读取测试侧进行评价。

如果 CPU 阶段换机器执行，在相同版本的项目中复制以下目录，保持相对路径：

```text
runs/l2a-train-pos/
runs/l2a-test-pos/
```

这两个目录必须完整，包含 `manifest.json`、`positions.jsonl`、`articles.jsonl`。
CPU 阶段不需要模型权重、`data/l2a/` 或两侧 `*-gen/` 目录。
另复制 `runs/l2_progress.json` 可保留 GPU 阶段的总进度和计时；不复制也能运行 CPU 阶段。

## 一次跑完与参数

原命令仍可使用，现在按 GPU → CPU 顺序执行：

```bash
WORKERS=10 bash scripts/l2_full.sh models/opt-1.3b l2a
```

GPU 脚本沿用 `CONFIG`、`DTYPE`、`PROGRESS` 环境变量；CPU 脚本沿用 `WORKERS`、`PROGRESS`、`WEIGHTS`。
两步的 run-name 和 `PROGRESS` 应保持一致，默认分别为 `l2a` 和 `runs/l2_progress.json`。
输出目录命名沿用原流程。GPU 脚本按上述完整阶段恢复；CPU 脚本仍要求阶段输出为空。
