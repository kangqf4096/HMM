# 云服务器存储预算：OPT-1.3B + arXiv 正式规模

日期：2026-10-31（第三版）。适用：迁出 Colab、自有云服务器、`dataset_loader=parquet_file`、
正式规模 null 校准/测试各 2000 篇、**不留 wheel 缓存**、不分批。

第三版修正了前两版对文本产物字节数的低估：前两版用 `runs/smoke1` 的仿真记录标定（只有 7 个字段），
真实 `generate` 记录带 `token_ids/origin/o/z/mask/verification_seeds`，实测 **92.7 B/token**；
`positions.jsonl` 实测 **597 B/位置**。结论（100 GB）不变，但各产物体积和“最小保留”要改。

## 1. 结论

| 方案 | 磁盘 | 前提 |
|---|---:|---|
| 最低可用 | **70 GB**（单盘） | 每轮跑完删 trace/sources/predictions/positions |
| 推荐 | **100 GB**（单盘，或 40 系统 + 60 数据） | 5 轮正式 + 1 轮完整诊断，留 20% 余量 |
| 全保留不清理 | 130 GB | 所有轮次的 trace/sources/predictions/positions 都留下 |

实测标定点（脚本：`tmp/measure_records.py`、`tmp/measure_positions.py`）：

| 量 | 实测值 |
|---|---:|
| `observations.jsonl` | 92.7 B/token（水印/纯 null/编辑记录几乎相同） |
| `sources.jsonl` | 85.7 B/token（无诊断）/ 394 B/token（含 `probability_trace`） |
| `trace/*.jsonl` 单步 | 495 B（开诊断、active）/ 171 B（不开诊断） |
| `predictions.jsonl` | 27.2 B/token/方法 |
| `positions.jsonl` | 597 B/位置 |
| `metrics.jsonl` | ~1.1 KB/行 |

单次正式实验：4,128 个生成请求、2.264 M 生成 token；12,320 条记录、6.16 M 记录 token。

## 2. 中间产物：是什么、谁消费、何时可删

| 产物 | 是什么 | 消费者 | 正式规模 | 可删时机 |
|---|---|---|---:|---|
| `data/ax*/manifest.json` | 数据版本、抽样规则、角色分配、模型 repo/revision、每篇来源 split/row | generate 校验 | KB | 可长期留 |
| `data/ax*/prompts.jsonl` | 每篇 50 个 prompt token + group/role | generate | <10 MB | 可长期留（复用） |
| `*-gen/metadata.json` | 冻结协议、模型路径与来源、vLLM/驱动版本、词表与屏蔽 id、耗时、lengths | **evaluate 必读** | KB | 随 gen 目录留 |
| `*-gen/trace/*.jsonl` | 采样器逐步审计：step、active、context、seed、chosen、selected_y、概率摘要（s2/s3/pmax/熵/有效词表）、cdf/log_pdf。证明“实际采样 = 密钥驱动的 Gumbel-max” | 仅 generate 末尾自检 | **1.11 GB**（诊断）/ 0.39 GB | **generate 返回后立即删** |
| `*-gen/sources.jsonl` | 每个请求的原始事实：原始 token、prompt、哪些位置启用水印、每位置上下文、每位置 seed、生成时概率 | 仅 `diagnose-pivots` | 0.89 GB / 0.19 GB | 不做诊断就立即删 |
| `*-gen/observations.jsonl` | 评价输入：最终 token、origin 映射、o（是否保留原文）、z（水印耦合真值）、mask、枢轴量 y、逐位置 seed、编辑动作 | **evaluate 唯一输入** | **0.57 GB** | 评价后建议留（可免 GPU 重算） |
| `*-eval/density.json` | f1 的 50 箱直方图密度 + 样本数 + 伪计数 | 人看/诊断 | KB | 永久 |
| `*-eval/thresholds.json` | 每方法 × 每 length_key 的纯 null any-FP 阈值（value/count/rank/alpha） | 复核 | KB | 永久 |
| `*-eval/metrics.jsonl` | 逐篇×方法：epsilon、rho、tp/fp/fn/tn、IoU/F1、any_fp、Brier、片段指标、耗时 | 论文表格 | 0.11 GB | 建议留 |
| `*-eval/predictions.jsonl` | 逐篇×方法逐 token 分数、二值标签、区间 | 仅画 `example.svg` 与自定义复核 | **1.34 GB**（8 方法） | report 写完即删 |
| `*-eval/reliability.json` | 分数分箱的校准曲线（mean_probability vs signal_fraction） | 论文 | KB | 永久 |
| `*-eval/report.json/md`、`example.svg` | 汇总表、配对 bootstrap | 论文 | 几十 KB | 永久 |
| `*-diag/positions.jsonl` | 逐位置重放：y、生成时 s2 等摘要、eval/source 的 CDF 与 log pdf、TV、seed 是否保留 | 只供诊断报告 | **3.68 GB** | 报告写出后删 |
| `*-diag/replay_agreement.jsonl` | vLLM 生成时概率 vs HF 重算的 s2/pmax/cdf/log_pdf 绝对差 | 只供诊断报告 | 几 MB | 同上 |
| `*-diag/localization/` | 用 8 个方法重跑整套评价（结构同 `*-eval/`） | 只供诊断报告 | ~1.45 GB | 同上 |

关键事实（已核对代码）：`evaluate` 只读 `metadata.json` + `observations.jsonl`，不读 trace/sources；
`generate` 返回前已把 trace 逐行读完做 seed/context 校验，之后无消费者；`predictions.jsonl`
只在 `run_evaluation` 内部用于渲染 `example.svg`。

## 3. 单轮体积与三级保留策略

| 策略 | 保留内容 | 单轮 | 5 轮正式 |
|---|---|---:|---:|
| A 最小 | observations + metrics + 报告 | **0.70 GB** | 3.5 GB |
| B 中等 | A + predictions | 2.0 GB | 10 GB |
| C 全保留 | B + trace + sources + positions + localization | **9.2 GB** | 46 GB |

生产链（含诊断）拆解：generate 2.6 GB + evaluate 1.45 GB + diagnose 5.2 GB ≈ 9.2 GB。
推荐 A 用于常规轮次，只给定稿那一轮留 C。

## 4. 不分批

分批会改变 vLLM 采样种子（`seed = config.seed + len(jobs)` 用调用内序号）和编辑 RNG 流，
产物只能当作新的独立运行；而它压的峰值本来只有约 2.6 GB，删产物省的是每轮 8.5 GB。
需要分批只在抢占式实例/单次 GPU 时限场景，改动见上一版说明。

## 5. 修订后的预算（100 GB 方案）

| 项目 | GB |
|---|---:|
| OS + 驱动 + apt 缓存 | 18 |
| `.venv`（vLLM + torch + CUDA 库） | 10 |
| pip 缓存（装完清） | 3 |
| OPT-1.3B | 2.6 |
| arXiv Parquet | 3.6 |
| 5 轮正式（A）+ 1 轮完整诊断（C） | 3.5 + 9.2 = 12.7 |
| 生成阶段峰值（trace+sources+observations） | 2.6 |
| tmp / 日志 / vLLM 编译缓存 | 4 |
| 小计 | **53.9** |
| 20% 余量 | 10.8 |
| **合计** | **约 65 GB** |

70 GB 能跑，100 GB 从容。

## 6. 每轮清理命令

```bash
# generate 返回后（trace 已自检完）
rm -rf runs/ax00-gen/trace                      # 1.11 GB
rm -f  runs/ax00-gen/sources.jsonl              # 0.89 GB；要做 diagnose 则留到诊断结束

# evaluate 成功后
rm -f runs/ax00-eval/predictions.jsonl          # 1.34 GB；复核时重跑 evaluate 即可

# diagnose-pivots 成功后
rm -f  runs/ax00-diag/positions.jsonl           # 3.68 GB
rm -rf runs/ax00-diag/localization              # 1.45 GB

python -m pip cache purge
```

## 7. 模型固定为 OPT

本项目只使用 OPT 系列（论文传统），默认 `facebook/opt-1.3b`（FP16 约 2.6 GB，已计入上表）。换规模时只在 OPT 内更换：opt-125m / 350m 更小，opt-2.7b（FP16 约 5.2 GB）在 24 GB 卡上仍然宽裕。不需要为其他模型族预留磁盘。
