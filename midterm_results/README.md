# 中期报告实验结果索引与撰写导航

本目录汇总了截至目前已完成、可直接用于撰写中期报告的实验报告与图表（已剔除数百 MB 级的逐位置原始数据，总大小不到 1 MB）。`clone` 本仓库后，直接按下面顺序阅读对应目录即可获取全部文字结论、数据表格与插图。

> **口径提示**：主线报告请以 `01`–`06` 目录的结果为准（其中 Qwen3-14B-FP8 主实验采用 `05_l2q1_qwen3_main/` 和 `06_l2q1_extra_baselines/` 的结果）。`99_archive_l2q1f/` 为后续剔除循环退化样本后的内部复核存档，暂不作为本次中期报告主表口径。

---

## 一、推荐的中期报告结构与素材对应

| 报告章节 | 对应目录 | 核心说明的科学问题 | 最核心可用数据 / 结论 |
|---|---|---|---|
| **1. 理论与枢轴性质验证** | [`01_pivot_and_clip/`](01_pivot_and_clip/) | 水印枢轴量 $Y_t$ 的解析分布、为什么需要似然比截断（cap/clip） | 验证了 $E[-\log Y \mid P] = S_2$；说明无界似然比在编辑后容易过拟合尖峰，适当截断（如 `clip_ln3` 在 `replace_0.1` 达 **0.872 IoU**，高于无截断的 0.571）能显著提升稳健性 |
| **2. 基础连续性验证（L0）** | [`02_l0_opt1.3b/`](02_l0_opt1.3b/) | 引入 HMM 状态连续性先验相比逐位置独立检测到底有没有用 | OPT-1.3B（128 token，1,050 篇，4 场景）：`switch` 场景下 `opt_hmm` 相比逐点 `independent` 提升 **+0.295 IoU**（直方图，0.895 vs 0.600）/ **+0.346 IoU**（Beta，0.890 vs 0.544） |
| **3. 转移结构与发射分布系统消融（L1v2）** | [`03_l1v2_opt1.3b/`](03_l1v2_opt1.3b/) | 比较 30 种配置：全局双参数转移矩阵 $A$ vs 逐篇 $\rho$、PD vs PE、25/200 箱直方图 vs ECDF400 vs 7 种参数族 vs SeedBS | OPT-1.3B（128 token，240 测试 + 400 null）：`pd_hist200`（**0.822**）与 `pd_rho_hist25`（**0.830**）大幅领先 `independent`（**0.630**）和 `seedbs`（**0.669**）；似然比检验支持两状态保持率非对称（$a_{00} > a_{11}$，$p < 0.006$） |
| **4. 长序列与密集切换压力测试（L2a，OPT-1.3B）** | [`04_l2a_opt1.3b/`](04_l2a_opt1.3b/) | 500 token 跨密钥（K1→K2）迁移下，论文标准场景与高频交替切换场景的性能 | OPT-1.3B（500 token，1,600 测试）：`pd_m400` 论文块达 **0.945 IoU**；**密集交替块达 0.777 IoU**，大幅领先 `SeedBS`（0.431，**+0.346**）；`m400` 相比 `m100` 证明增加纯水印校准样本（400 vs 100）稳定带来正增益 |
| **5. 现代大模型主实验（L2q1，Qwen3-14B-FP8）** | [`05_l2q1_qwen3_main/`](05_l2q1_qwen3_main/) | 换用 Qwen3-14B-FP8（500 token，跨密钥 K1→K2）后，自适应发射 `fe` 相比固定直方图 `m100/m400` 与 `SeedBS` 的表现 | **自适应 `fe` 全面胜出**：论文块 `pd_fe/pe_fe` 达 **0.797 / 0.810**（`m400` 为 0.511，`SeedBS` 为 0.670）；密集交替块 **0.397 / 0.393**；R2 编辑 **0.713**；R0 编辑 **0.315**；`null_test` any-FP 分别为 **4.3% / 3.3%**（守住 5% 线） |
| **6. 外部文献基线全面对比（L2q1-extra）** | [`06_l2q1_extra_baselines/`](06_l2q1_extra_baselines/) | 与已发表的同类检测/定位方法全面横向对比（文档级 5% 误报口径 & 逐位置口径） | 在 Qwen3-14B-FP8 上对比 **13 个方法**（含 `SeedBS-NOT`、`SPOT-plugin`、`SPOT`、`窗口 Tr-GoF`、`窗口 Aaronson 和`、`Beta 核自适应`）；文档级口径下 `pd_fe`/`pe_fe` **四块全部排名第一**，且所有方法 `C1` 重校准误报检验均通过 |
| **7. 后续工作计划（报告结尾可用）** | 见主库 `docs/` | 中期之后将推进的方向 | ① 长短文本长度稳健性验证（300 / 500 / 800 token，见 [`docs/l2_length.md`](../docs/l2_length.md)）；② 退化循环过滤与门控+位置级误报解耦（见 `99_archive_l2q1f/` 与 [`docs/plan_gate_position_20261116.md`](../docs/plan_gate_position_20261116.md)） |

---

## 二、各目录文件明细

### 1. `01_pivot_and_clip/` —— 机理诊断与截断实验（OPT-1.3B）
- [`piv_diag_report.md`](01_pivot_and_clip/piv_diag_report.md)：真实模型下 $Y_t$ 的条件律、HF/vLLM 概率重算误差与编辑后分布偏移诊断。
- [`localization_report.md`](01_pivot_and_clip/localization_report.md)：基础定位方法与 `raw_analytic` / `s2_histogram` 等条件发射对照表。
- [`localization_example.svg`](01_pivot_and_clip/localization_example.svg)：单篇样本定位曲线示例图。
- [`pivot_ax01_localization.png`](01_pivot_and_clip/pivot_ax01_localization.png)：枢轴定位效果汇总图（可直接插图）。
- [`clipv2_report.md`](01_pivot_and_clip/clipv2_report.md)：不同对数密度截断阈值（`clip_ln2` / `3` / `4` / `6` / `8`）的定位对比表。
- [`clipv2_example.svg`](01_pivot_and_clip/clipv2_example.svg)：截断对比示例图。
- *配套理论文档（已在主库 `docs/`）*：[`docs/pivot_law_derivation_20261025.md`](../docs/pivot_law_derivation_20261025.md)、[`docs/pivot_law_20261025/relations.png`](../docs/pivot_law_20261025/relations.png)。

### 2. `02_l0_opt1.3b/` —— L0 基础连续性验证（OPT-1.3B，128 token）
- [`report.md`](02_l0_opt1.3b/report.md)：L0 主报告（含 `full_wm`、`switch`、`resample_0.1`、`resample_0.3` 四个场景 × 2 种发射 × 5 种方法的 IoU、P/R/F1、$\hat\varepsilon$ 误差、片段召回与边界误差、null any-FP）。
- [`report_nocap.md`](02_l0_opt1.3b/report_nocap.md)：无截断 `beta_nocap` 附录对比。
- [`diag.md`](02_l0_opt1.3b/diag.md)：离线重校准分布、等误报工作点排名与 $\rho$ 边缘化敏感性分析。
- *配套判读文档*：[`docs/l0_review_20261103.md`](../docs/l0_review_20261103.md)。

### 3. `03_l1v2_opt1.3b/` —— L1v2 转移矩阵与 30 方法大消融（OPT-1.3B，128 token）
- [`report.md`](03_l1v2_opt1.3b/report.md)：完整 30 方法评测报告（含 6 个定位场景、`full_wm`、`null_test`、EM 转移矩阵 $a_{00}/a_{11}$ 显著性检验及参数族全面对比）。
- *配套判读文档*：[`docs/l1v2_review_20261105.md`](../docs/l1v2_review_20261105.md)。

### 4. `04_l2a_opt1.3b/` —— L2a 论文设定与密集切换（OPT-1.3B，500 token，K1→K2）
- [`analysis.md`](04_l2a_opt1.3b/analysis.md)：L2a 详尽解读（含 `m400 − m100` 校准样本量增益表、不同切换点数 $K=1\sim41$ 下 HMM vs SeedBS 对比）。
- [`overview.png`](04_l2a_opt1.3b/overview.png) / [`overview.svg`](04_l2a_opt1.3b/overview.svg)：**L2a 定位与误报总览图**（推荐直接放入报告）。
- [`test_report.md`](04_l2a_opt1.3b/test_report.md)：L2a 完整测试指标（IoU、Rand index、切换点检出比、误报与次终点）。
- [`curves.svg`](04_l2a_opt1.3b/curves.svg)：随切换点数 $K$ 变化的性能曲线图。
- [`fit_report.md`](04_l2a_opt1.3b/fit_report.md)：训练侧拟合与参数选择报告。
- [`cap_sweep.md`](04_l2a_opt1.3b/cap_sweep.md)：截断超参 `cap` 扫描曲线结果（可用于消融讨论）。
- [`mechanism.md`](04_l2a_opt1.3b/mechanism.md)：R2 熵选择性编辑机制验证记录。

### 5. `05_l2q1_qwen3_main/` —— L2q1 主实验（Qwen3-14B-FP8，500 token，K1→K2）
> 本组实验最适合作为中期报告的**核心主实验结果**。
- [`test_report.md`](05_l2q1_qwen3_main/test_report.md)：包含 `pd_fe` / `pe_fe` / `pd_m400` / `pe_m400` / `pd_m100` / `pe_m100` 及多档 `SeedBS` 的全部主终点（IoU、Rand index、按 $K$ 曲线）、配对 bootstrap 差值、`null_test` 误报及转移矩阵参数估计（`pd_fe` 下 $a_{00}=0.9835, a_{11}=0.9934, p=0.0000$）。
- [`curves.svg`](05_l2q1_qwen3_main/curves.svg)：随切换点数 $K$ 变化的性能曲线图（推荐直接放入报告）。
- [`fit_report.md`](05_l2q1_qwen3_main/fit_report.md)：Qwen3 训练侧 $e'$ 覆盖、参数选择与冻结报告。
- [`cap_sweep.md`](05_l2q1_qwen3_main/cap_sweep.md)：`cap` 敏感性扫描结果。
- [`mechanism.md`](05_l2q1_qwen3_main/mechanism.md)：训练侧机制检验记录。

### 6. `06_l2q1_extra_baselines/` —— L2q1 补充基线实验（Qwen3-14B-FP8）
> 本组实验最适合放在主实验之后作为**与文献已有 SOTA 方法的横向对比**。
- [`extra_baselines_report.md`](06_l2q1_extra_baselines/extra_baselines_report.md)：在统一设置下对比 13 个方法：
  - 我方方案：`pd_fe`、`pe_fe`、`pd_m400`、`pe_m400`、`pd_m100`、`pe_m100`、Beta 核自适应（`pd_fb`/`pe_fb`）、固定 Beta（`pd_bm`/`pe_bm`）
  - 文献基线：`SeedBS-NOT`（Li 等 2024）、`SPOT-plugin`、`SPOT`（尾部计数 $\hat\varepsilon$）、`窗口 Tr-GoF`、`窗口 Aaronson 和`
  - 同时给出文档级 `any-FP ≤ 0.05` 口径和逐位置 FPR `≤ 0.05` 口径两套结果及配对 bootstrap 区间。

### 7. `99_archive_l2q1f/` —— 内部归档：剔除退化样本后的 Qwen3 复核（仅备查）
- 包含 [`test_report.md`](99_archive_l2q1f/test_report.md)、[`curves.svg`](99_archive_l2q1f/curves.svg)、[`fit_report.md`](99_archive_l2q1f/fit_report.md)、[`cap_sweep.md`](99_archive_l2q1f/cap_sweep.md)、[`mechanism.md`](99_archive_l2q1f/mechanism.md)。
- **说明**：该版本剔除了存在长周期循环退化的生成文章并重拟合了模型；由于 `pd_fe` 的 `τ` 与部分场景表现与 `l2q1` 存在口径差异，中期报告暂统一以 `05_l2q1_qwen3_main/` 为准，本目录仅作仓库留档与后续 L3 改进（[`docs/plan_gate_position_20261116.md`](../docs/plan_gate_position_20261116.md)）的参照。

---

## 三、中期报告核心数据速查表（均源自上述报告原文）

### 表 A：Qwen3-14B-FP8 主实验与外部基线（文档级 any-FP ≤ 0.05 口径，IoU [95% CI]）
来源：[`05_l2q1_qwen3_main/test_report.md`](05_l2q1_qwen3_main/test_report.md) & [`06_l2q1_extra_baselines/extra_baselines_report.md`](06_l2q1_extra_baselines/extra_baselines_report.md)

| 方法 | 论文块（Setting 2–4） | 密集交替块 | R2 编辑块 | R0 编辑块 | `null_test` any-FP |
|---|---:|---:|---:|---:|---:|
| **`pd_fe`（本文自适应方案）** | **0.797** [0.777, 0.815] | **0.397** [0.386, 0.406] | **0.713** [0.670, 0.753] | **0.315** [0.299, 0.330] | **0.043** (4.3%) |
| **`pe_fe`（本文自适应方案）** | **0.810** [0.787, 0.830] | 0.393 [0.380, 0.403] | 0.693 [0.650, 0.733] | 0.306 [0.288, 0.322] | **0.033** (3.3%) |
| `pd_fb`（Beta 核自适应） | 0.712 [0.680, 0.741] | 0.337 [0.311, 0.361] | 0.521 [0.479, 0.565] | 0.241 [0.219, 0.262] | 0.050 (5.0%) |
| `pd_m400`（固定 hist200） | 0.511 [0.463, 0.557] | 0.227 [0.196, 0.257] | 0.248 [0.198, 0.299] | 0.147 [0.125, 0.169] | 0.033 (3.3%) |
| `seedbs`（SeedBS-NOT） | 0.670 [0.646, 0.691] | 0.334 [0.311, 0.356] | 0.698 [0.654, 0.741] | 0.284 [0.266, 0.300] | 0.060 (6.0%) |
| `win_ars`（窗口 Aaronson 和） | 0.587 [0.542, 0.629] | 0.185 [0.164, 0.205] | 0.291 [0.239, 0.341] | 0.159 [0.138, 0.181] | 0.033 (3.3%) |
| `win_trgof`（窗口 Tr-GoF） | 0.504 [0.455, 0.553] | 0.144 [0.125, 0.163] | 0.237 [0.182, 0.290] | 0.136 [0.116, 0.157] | 0.052 (5.2%) |
| `spot_opt`（SPOT-plugin） | 0.375 [0.332, 0.414] | 0.163 [0.136, 0.189] | 0.254 [0.201, 0.307] | 0.163 [0.136, 0.191] | 0.040 (4.0%) |
| `spot_ini`（SPOT 原版估计） | 0.141 [0.119, 0.165] | 0.088 [0.070, 0.106] | 0.099 [0.067, 0.134] | 0.082 [0.065, 0.100] | 0.048 (4.8%) |

### 表 B：OPT-1.3B 跨密钥实验 L2a（500 token，IoU [95% CI]）
来源：[`04_l2a_opt1.3b/analysis.md`](04_l2a_opt1.3b/analysis.md) & [`04_l2a_opt1.3b/test_report.md`](04_l2a_opt1.3b/test_report.md)

| 方法 | 论文块（Setting 2–4） | 密集交替块 | R2 编辑块 | R0 编辑块 |
|---|---:|---:|---:|---:|
| `pd_m400` | 0.945 [0.921, 0.966] | **0.777** [0.749, 0.801] | **0.744** [0.700, 0.784] | **0.557** [0.530, 0.582] |
| `pd_m100` | 0.940 [0.914, 0.961] | 0.757 [0.727, 0.782] | 0.704 [0.655, 0.751] | 0.520 [0.491, 0.547] |
| `pd_fe` | **0.950** [0.924, 0.971] | 0.750 [0.725, 0.771] | 0.721 [0.674, 0.768] | 0.523 [0.500, 0.546] |
| `seedbs`（dev 选择版） | 0.926 [0.905, 0.944] | 0.431 [0.417, 0.445] | 0.627 [0.572, 0.680] | 0.321 [0.307, 0.334] |
