# L2 补充实验：Beta 自适应 f1(e′) 与文献基线

在已完成的 L2 运行（如 `l2q1`）上追加。不改动已冻结的 `fe`、`m400`、`seedbs`，它们的逐篇结果直接取自参照运行的 `metrics.jsonl`，所以所有差值都是同一批测试文章上的 prompt 配对 bootstrap。训练侧先拟合并写出 `frozen_extra.json`，之后才读测试侧，只跑一次。

## 1 Beta 自适应（回答 `汇报.md`：参数族没做 e′ 自适应所以差很多？）

四个臂组成 2×2：

| | 固定 f1 | e′ 自适应 f1(e′) |
|---|---|---|
| hist200 | `m400` | `fe` |
| Beta | `bm` | `fb` |

- `fb`：与 `fe` 用同一套 e′ 核权重（网格、h 网格、cap 网格、PD/PE 全部照旧，h 与 cap 由 dev 选），只把每个格点上的加权直方图换成加权矩估计的 Beta(a, b)，再按箱平均成 hist200 形式接入 HMM。
- `bm`：用 `m400` 的 400 篇纯水印位置拟合一个固定 Beta，和 `m400` 只差"族"。
- 读法：`fe − m400` 是核自适应对 hist 的收益，`fb − bm` 是对 Beta 的收益；报告里给 `(fe − m400) − (fb − bm)` 的配对区间。`fb − fe` 直接回答"自适应的 Beta 能否追上非参核"。
- e′ 仍由冻结的 hist200 做 ε_opt，不是 Beta 下的估计；这一点两个族相同，所以不影响对比，但也意味着本实验不检验"Beta 下的 ε 估计"。
- 矩估计在箱中点上取均值与方差（方差减去箱宽项），没有用极大似然。不做 MLE 的原因：先看最简单的参数族能否追上，追不上再换估计方式。

## 2 文献基线（只用 pivot 序列 Y_t，p_t = 1 − Y_t）

核对过的原文位置：`docs/2608.14906v2` 算法 1 与补充材料 C；`papers/text/2025_jrssb_robust_edits.txt` 式 (5)–(8)；`papers/text/2511.06645_adaptive_segment.txt` B.5；`papers/text/2308.00113v2_three_bricks.txt` 式 (7)。

| 名称 | 方法 | 与论文的差别 |
|---|---|---|
| `spot_opt` | SPOT-plugin：u∈[0.005, 0.98]，M = max(20, ⌈3.5·ln(n)^1.5⌉)，η = 0；ε̂ 用冻结 stage-1 hist200 的 ε_opt（cap 在 dev 上选） | 不做"无 u 满足时回退最大 u"（回退会让每篇 null 文章平均多标约 1 个位置，任何阈值都救不回来）；λ 不由 C/ln n 手调，而是校准 τ |
| `spot_ini` | 同上，ε̂ 用论文模拟里的尾部计数估计 clip((P_n(Y>0.85) − 0.15)/0.85, 0, 1) | 同上 |
| `win_trgof` | 居中窗口（B+1 个 token）里的 Tr-GoF 统计 n·S⁺(s)，s∈{1（Berk–Jones）, 2（HC）}，c⁺ = 1/m | 检测统计量改作定位用，B 与 s 在 dev 上选；窗口分数 x 映射为 x/(1+x) |
| `win_ars` | 居中窗口里 −log(1−Y) 之和，换成精确 Gamma(m, 1) 上尾概率的 −log，窗口在文本两端变短时仍在同一零分布尺度上 | 同上 |

SPOT 的得分写成逐位置分数 1/(1 + min_{u: Y_t>τ(u)} T̂(u))，"分数 ≥ 1/(1+λ)" 与算法 1 的标记集合完全一致（`tests/test_l2_baselines.py` 逐位置核对），因此可以直接放进 `null_cal` 校准框架。

未纳入：AOL（Zhao 等 2024，需要其在线平滑算法的官方实现，文献库里没有）、WISER（Bonnerjee 等，`segment` 文献，需要外部实现与上千组调参）。

## 3 两个口径

- **文档级（项目主协议）**：`null_cal` 上规则 M 校准 τ，使 any-FP ≤ α = 0.05。所有方法都做得到，各方法的 τ 都是 `null_cal` 的一个阶统计量。
- **逐位置（_tok 后缀）**：τ 取 `null_cal` 全部位置分数的 1 − α 分位，被标记位置占比 ≤ 5%。这是更宽松的口径，SPOT 论文用的就是这种（经验 FPR 上限下的最优参数）。HMM 方法沿用冻结参数，只重新校准 τ；基线在该口径下独立地在 dev 上选配置。`seedbs` 不在此口径内（其分数不是逐位置的）。两个口径下都同时报告文档级 any-FP 与逐位置误报，不要拿 τ 不同的两行直接比。

## 4 运行（CPU，Linux）

```bash
python scripts/run.py l2-extra --frozen runs/l2q1-fit --train runs/l2q1-train-pos \
  --input runs/l2q1-test-pos --reference runs/l2q1-test --output runs/l2q1-extra
```

输出：`report.md`、`report.json`、`frozen_extra.json`（每个方法的口径、配置、`null_cal` 上的 any-FP 与逐位置误报、阈值）、`fit_extra.json`（dev 选择表）、`metrics.jsonl`（新方法的逐篇结果）。

运行前会核对：训练侧 stage-1 计数与参照拟合一致、当前代码复现参照的 `fe` 核密度（最大偏差 ≤ 1e-6）、参照测试运行与本测试侧是同一批文章。任何一条不满足都直接报错，不继续。

结果是一次运行，不据此回头改 `fe` 的冻结参数。
