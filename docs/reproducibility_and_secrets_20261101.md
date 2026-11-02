# 复现边界、运行时选择与密钥处理

日期：2026-11-01。范围：代码/数据/产物的传输边界，`metadata.json` 的密钥字段，以及 CUDA graph 与 eager 的取舍。
本文区分**当前已实现**与**部署前才实施**两类内容；未实施的部分在文中明确标出，不能当成已经生效的约束。

---

## 1. 三条边界

| 内容 | 通道 | 理由 |
|---|---|---|
| 代码（`src/ configs/ scripts/ tests/ docs/ notebooks/`） | git | 需要版本、diff、可回滚；体量小 |
| 论文与本地文献（`papers/`） | git | 附录与可复现性需要；目前约 31 MB，是仓库体积的主要部分 |
| 数据、模型、产物（`data/ models/ runs/ .cache/ tmp/`） | **不进 git** | 单轮正式实验就是 GB 级；`.gitignore` 已挡住这五项 |

数据与产物走远端本地重算：语料由 `prepare` 重新下载/抽样，模型用 `download_model.py` 或从旧机器 scp，结果留在 `runs/` 不回传仓库。

**远端出网受限时的替代**：不要退回逐文件 scp。用

    git bundle create hmm.bundle HEAD
    # scp 单个 bundle 过去
    git fetch hmm.bundle main:incoming && git merge incoming

这样仍保留提交历史与 diff，只多一次文件传输。

`GPU实验运行手册` 早期写的是"代码改动直接 scp"，与本节冲突；以本节为准。

---

## 2. `metadata.json` 的密钥字段

### 2.1 当前实现（研究阶段）

`generate` 把实际密钥**明文**写入 `metadata.json` 的 `key` 字段，同时保留：

- `key_id`：配置里的标识（默认 `public-research-v1`）；
- `public_research_key`：是否为内置公开研究密钥。

明文写入的理由是复现优先：自定义 `OPT_HMM_KEY` 的轮次不再依赖任何外部记录（shell history、笔记、口头传递）就能重跑和对齐。研究阶段没有发布方法，也没有对抗性使用者，泄漏的现实风险低于"半年后无法判断两轮是不是同一把 key"。

### 2.2 泄漏面的准确描述

不同产物的敏感度并不相同：

| 字段 | 泄漏的能力 | 范围 |
|---|---|---|
| 逐位置 `seed`（在 `sources.jsonl`、`trace/`、`observations.jsonl` 的 `verification_seeds`） | 复现该位置的随机向量，进而重放/伪造**这一段文本**的水印 | 被该文本锁死 |
| `key`（在 `metadata.json`） | 对**任意新上下文**计算 seed 与随机向量 | 无界 |

因此"密钥不落盘"并不能让已跑过的文本变成不可重放——`seed` 已经够了。研究阶段保留 `seed` 是为了审计；真要收窄泄漏面，应当删 `sources.jsonl` 与 `trace/`，这与存储预算的清理建议一致，比隐藏 `key` 更有效。

### 2.3 部署前必须改（方案已定，**未实施**）

把 `key` 换成 **key commitment**：

    key_fingerprint = sha256(key.encode("utf-8")).hexdigest()[:16]

- 不可反推，但足以判断"两轮是否同一把 key"、支撑日志与工单对齐；
- 需要对外证明某次运行确实用了某 key 时，改用它揭示或零知识式承诺，而不是现在这样就地哈希；
- 与 `key_id` 并存：`key_id` 是人读标签，`key_fingerprint` 是机器对齐标签。

实施时的改动点：`generate.py` 的 metadata 写入处、`README` 协议小节、以及新增一条"研究模式 / 部署模式"的开关说明。**在此之前，任何对外产物都应视为携带明文密钥。**

---

## 3. CUDA graph 与 eager

### 3.1 语义

vLLM V0 的 CUDA graph 只捕获**模型前向**。逐请求的 `GumbelRequest` 在 sampler 内被调用，位于捕获图之外。因此：

- `enforce_eager` 不影响随机量、枢轴量、选择规则或 trace 内容；
- 它是运行时选项，**不是协议的一部分**。早期文档把"使用 eager 模式"写进协议描述，属于把实现选择误当约束，已更正。

### 3.2 当前配置

- `configs/*.json` 显式写 `"enforce_eager": false`，默认即开启 CUDA graph；
- `metadata.json` 记录 `enforce_eager`、`max_num_seqs`、`gpu_memory_utilization`，便于区分运行时不同的两轮；
- 捕获失败（某些算子不可捕获、显存不足）应在配置里改回 `true`，并在轮次记录里注明；不要靠环境变量临时切换。

### 3.3 预期收益与真正的瓶颈

按 token 计的 Python 工作量才是主导项：每步要生成整个词表的随机向量、做一次 H2D 拷贝、写一行 trace；开 `--pivot-diagnostics` 时还要把整个词表取回 CPU 计算概率摘要，每步一次同步。相对而言，OPT-1.3B 在 3090 上、batch ≤ 8、序列 ≤ 550 的前向开销不大。

因此优先级是：

1. 诊断只在该开的轮次开（收益最大，且无风险）；
2. 需要更满的前向时再调 `max_num_seqs`（会改变批组成，末位数值可能变化，需重跑校准）；
3. CUDA graph（几乎零成本，但只省前向那一段）；
4. 把 Uniform 生成搬到 GPU——**会改协议，不做**。

### 3.4 换图之后的验收

同一 `seed`、同一 `prompts` 目录下，eager 轮与 graph 轮的 `observations.jsonl` 应当逐 token 相同。不同就说明有数值翻转，必须记录差异条数与原因，不能直接混用两轮的阈值。

`generate` 自带的审计（每个 step 校验 `seed/context/active`，启用水印处再校验 `chosen` 与 `selected_y`）会拦住任何串扰：捕获期若误触发处理器，trace 行数或内容必然对不上，第一次 smoke 就会报错。

---

## 4. 待部署时再处理（清单）

- [ ] `metadata.json`：`key` → `key_fingerprint`；
- [ ] 文档：研究模式与部署模式的措辞分离，README 与运行手册同步；
- [ ] 产物：`sources.jsonl`、`trace/` 从"默认保留"改为"默认删除"，仅在显式审计模式下保留；
- [ ] `OPT_HMM_KEY` 的来源（环境变量）改为受限读取，避免出现在进程列表与 shell history；
- [ ] 若要对外声称"密钥未泄漏"，需要重新评估 PRNG 与 BLAKE2b 的按键派生是否满足该声明，当前明确不是密码学主张。
