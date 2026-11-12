# L2 文本长度稳健性：冻结 f1 对长度有多敏感（300 / 500 / 800）

**问题。** 冻结在 500 token 上的 f1（stage 1 的 hist200、`fe` 的 e' 核网格、选定的 cap 与带宽 h）换到别的文本长度后，还够不够用？
只看当前最优方案 `fe`（`pd_fe`、`pe_fe`），并列 `seedbs` 作参照。

## 两组

| 组 | 做法 |
|---|---|
| **A 重训** | 每个长度各自完整跑一遍现有流程（`l2_cpu.sh`）：f1、cap、h、a00/a11/π1、τ、SeedBS 全部在该长度的训练侧重新拟合与选择。 |
| **B 沿用** | f1（stage 1、核网格）、cap、h 取自 500 长度的 A 组，**不再选择**；只在新长度的训练侧用同一个 EM 重估 a00/a11/π1，在新长度的 `null_cal` 上重标 τ。SeedBS 与密钥、`L_top` 取该长度的 A 组。 |

500 长度只有 A 组（B 在 500 上就是 A）。τ 在两组都在各自长度重新校准，所以误报反映新长度上的校准质量，不是 τ 的跨长度迁移。

## 口径

- **退化文章全部剔除，不留带退化的版本。** 规则同 `tmp/refilter.py`：只看带水印信号的位置（truth = 1；无水印文章取全部位置），
  单个 1/200 箱占比 > 0.3 且至少 50 个位置即为退化（循环文本反复出现同一个 (上下文, token)，枢轴值就只有一个）。
  按 prompt 组整组剔除，**任一长度、任一文章命中，三个长度都剔除**，所以三个长度的文章集完全相同，只差长度。评价侧只扫描水印文本和 null，U 文本不在规则内。
- **参照基准是“剔除退化后在 500 上重拟合”**（`l2q1f`），不是原 `l2q1` 的冻结参数。
- **e' 覆盖检查**：某个 e' 档位的位置数不足 300 时，若该档所有文章的 e' 都小于各自的真实水印占比 `eps_true`，则豁免，不中断（配置 `l2.coverage.waive_underestimated`，
  三份新配置打开，`text_l2.json` 默认关）。空档不豁免；核网格某个格点在 3 个带宽内没有任何训练位置时，`KernelGrid.fit` 仍会报错，那是需要人来决定的情形。
- 评价侧与 500 长度是同一批 prompt、同一生成的前缀；500 的评价结果此前已被看过，本研究全部是描述性的，不据此回头改冻结规则。

## 运行（服务器，Linux）

```bash
cd /root/blockdata/HMM
source .venv-q3/bin/activate && source scripts/env.sh
git pull
df -h /root /root/cache

bash scripts/l2_len.sh gpu /root/models/qwen3-14b-fp8   # GPU：800 生成、前缀校验、派生 300、两个长度的 build
WORKERS=4 bash scripts/l2_len.sh cpu                    # CPU：退化扫描、过滤、A×3、B×2、报告
```

脚本按完整步骤续跑：已完成的步骤显示 `[skip]`，未完成的输出挪到 `<名>.old1`、`.old2` 后重做。`BASE`（默认 `l2q1`）是已有的 500 token 运行。
每个运行单独用 `runs/<run>_progress.json`，不共用总进度文件。

| 步骤 | 命令 | 输出 |
|---|---|---|
| 复制 prompt 清单 | `l2-derive prompts` | `data/l2L800`、`data/l2L300`（选篇与 `tokens` 无关，清单与 `l2q1` 相同） |
| 生成 800（GPU，`/root/cache` 双批缓存） | `l2_gpu.sh`（`CONFIG=configs/text_l2_L800.json`） | `runs/l2L800-{train,test}-gen`、`-pos` |
| 前缀校验 | `l2-derive prefix` | `runs/l2len-prefix-{train,test}.json`：800 生成的前 500 token 是否与 `l2q1` 逐 token 相同 |
| 派生 300（不生成） | `l2-derive sources`，再 `l2_gpu.sh` | `runs/l2L300-*`：把 800 的 sources 截成前 300 token，只重做 build（熵重放要 GPU） |
| 退化扫描 | `l2-degenerate` | `runs/l2len-degenerate.json` |
| 过滤 | `l2-filter` | `runs/l2q1f-*-pos`、`l2L800f-*-pos`、`l2L300f-*-pos` |
| A 组 | `l2_cpu.sh <run>f` | `runs/<run>f-{mechanism,fit,test,cap}` |
| B 组 | `l2-reuse`，再 `l2-test` | `runs/<run>f-reuse`（`frozen.json`、`reuse_report.json`）、`runs/<run>f-reuse-test` |
| 报告 | `l2-lengths` | `runs/l2len-report/report.md`、`report.json` |

## 报告怎么读（先于结果写定，不设数值红线）

1. **IoU**：B − A（同长度、按 prompt 配对）；fe − seedbs（跨长度可比的量，因为切点数与 SeedBS 最小段长 2.5×B 都随长度变）；A(长度) − A(500)。
2. **细化指标**：recall、precision、F1、段召回、Rand index、真值/预测切点数；后验健康指标（τ、Setting 1 的位置 recall 与假切点、饱和位置、null any-FP）；转移参数 a00/a11/π1。
3. **f1 本身**（训练侧纯水印文章）：长度之间与位置窗口之间的 hist200 距离（JS 经文章二分的噪声校正）、P(Y≥0.99)/P(Y≥0.9) 的配对差、e' 的分布与较参考长度的偏移、fe 核网格逐 e 的 JS 与 n_eff。
   位置窗口默认按 token 位置 150、300、500 切最长文本（`--window-edges`）。150 这个边界来自开跑前对 `l2q1` 训练侧 398 篇纯水印文章的预览：
   前约 150 个 token 的信号更强（P(Y≥0.99) 0.085 对 0.057，噪声校正后 JS 0.0021，区间不含 0），150–300 与 300–500 之间几乎没有差别（校正后 JS 0.00007）。
   这是探索性的、只在 500 token 内观察到；若这个模式在更长的文本上延续，短文本的池化 f1 里早期位置占比更大，
   f1 随长度的变化就主要是早期与后期两种密度的混合比例变化。这正是本研究要检验的。

区间是按 prompt 配对的 bootstrap 95% 区间；`*` 表示区间不含 0。不同长度的绝对分数不直接比较。

## 要知道的

- **GPU 时间**：l2q1 训练侧生成用了 34086 s。生成量与长度成正比，800 约为其 1.6 倍，训练侧约 15 小时，加评价侧约 6 小时。
  `max_num_seqs=1` 不要改：并发会让数值路径变化，前缀校验可能因此失败。前缀校验失败不算错误，500 仍用原 `l2q1`，但"300 是 500 的前缀"的理由就要重新看。
- **磁盘**：双批缓存每请求预算 `max(1 MiB, tokens × 4096 B)`，800 token 约 3.3 MB，两批 200 请求约 1.3 GB，另有 1 GiB 预留。生成中断会留下 `/root/cache/g-*/b*`，要手工清理。
- **A 组的 CPU 时间**最长：三个长度各做一次完整拟合与 SeedBS 选择。把 `runs/l2q1-cache` 拷到 `runs/l2q1f-cache` 可复用 500 长度的 SeedBS 预计算（脚本已处理）。
- **进度条**：剔除后文章数少于配置里的角色篇数，而各阶段的总量按配置算，所以百分比可能到不了 100%，只是显示问题。
- **null 文章也被剔除**，any-FP 只对非退化文本成立。
- 本地验证：`python -m pytest tests/test_l2_lengths.py -q`（toy 语言模型端到端：derive → build → filter → A → B → 报告）。注意 `.gitignore` 忽略了 `tests/`。
