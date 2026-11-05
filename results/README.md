# results/ — 本轮结果索引

L0 全量第一轮（1,050 篇，4 场景 × 2 发射 × 5 方法）的可提交产物。
**结论 → 文件 → 复现命令**；原始逐位置数据不进版本库（`.gitignore` 已排除 `runs/`、`data/`）。

## 1. 结论与出处

| 结论 | 文件 | 位置 |
|---|---|---|
| 预注册判据下 **8 格全灭**；四场景主表（IoU / P / R / F1 / ε误差 / any-FP） | `l0full0-eval_report.md` | §主表（full_wm / switch / resample_0.1 / resample_0.3） |
| **连续性有效**：switch 上 opt_hmm 比 independent 高 **+0.295** [0.263, 0.329]（beta +0.346） | `l0full0-eval_report.md`、`l0full0-eval2_report.md` | §连续性增益判定 |
| **逐篇估 ε / 逐篇选 ρ 无可检出增益**：对 fixed_e / fixed_rho / window 的 Δ 多在 ±0.02 且 CI 跨 0 | 同上 | §连续性增益判定 |
| **P 规则救不了那两格**：`IoU(P)` 8 格全不达标（`fraction_tau = 0.5` 不是等误报工作点） | `l0full0-eval2_report.md` | 主表 `IoU(P)` 列 |
| **`any-FP ≤ α` 单点硬判有约 42% 先天失败率**（`P(FP>α) = 0.395–0.445`） | `l0full0-diag.md` | §2 重校准 |
| τ 处无并列，**最大原子在 0**（ε̂=0 时 max γ 恒为 0；beta 400 篇 null 仅 35 个不同取值） | `l0full0-diag.md` | §3 |
| **等误报工作点下 opt_hmm 在 4/8 格最优**（resample_0.3/beta +0.118） | `l0full0-diag.md` | §4 |
| **ρ 不可识别**：边缘化权重近均匀 → 维持 argmax | `l0full0-diag.md` | §5 |
| **cap=25 对 beta 造原子、对 hist 不生效**；nocap 尾部失控（`max f1 = 459730`） | `l0full0-eval-nocap_report.md` | 头部与 §发射形状 |
| 冻结参数（τ / rank / 发射 / cap） | `../frozen.json` | 全文 |
| 判读与预注册修订 R1–R5 | `../docs/l0_review_20261103.md` | §1 / §4 / §8 |

## 2. 复现命令（服务器，仓库根目录）

```bash
python scripts/test.py

# 用已生成的 positions 重评，约 1 分钟
python scripts/run.py l0-evaluate --input runs/l0full0-pos --output runs/l0full0-eval2

# 离线诊断：阈值复核 / 重校准误报 / τ 原子 / 等误报工作点 / ρ 变体，约 1–2 分钟
python scripts/l0_diag.py --positions runs/l0full0-pos \
  --frozen runs/l0full0-eval/frozen.json --output runs/l0full0-diag
```

`l0_diag.py` 在重算的 τ 与 `frozen.json` 不一致时会以退出码 2 报错，兼作"报告数字与代码一致"的检查。

## 3. 原始数据

`runs/l0full0-pos/positions.jsonl`（逐位置 Y / truth / edited / split，29 MB）不进版本库，从服务器取：

```bash
scp -P 37375 root@ssh-cn-xibei2.ebcloud.com:~/blockdata/HMM/runs/l0full0-pos/positions.jsonl .
```

sha256 与文件清单见本地那次抓取的 `result/MANIFEST.tsv` 与 `result/SOURCES.tsv`。

## 4. 生成环境（`l0full0-metadata.json`）

| 项 | 值 |
|---|---|
| 生成时代码 | `86c2c36`（当前仓库 HEAD 是 `b804275`，在其后 3 个提交；结论判读用的是后者） |
| seed | `20261015` |
| 模型 | `facebook/opt-1.3b` @ `3f5c25d0bc631cb57ac65913f76e22c2dfb61d62` |
| 数据集 | `ccdv/arxiv-summarization` / `document` @ `240aaf1a969b3f8cd0ade6986bfad0cd730ee288` |
| 运行时 | python 3.10.16 · vLLM 0.8.4 · torch 2.6.0 · numpy 2.2.6 |
| 规模 | 候选 215,913 篇 → 检查 1,056 → 选中 **1,050**；续写 128 token，prompt 50 token |
| 耗时 | 生成 111.96 s（单卡 RTX 3090，FP16，`enforce_eager=false`） |
| 审计 | "Every active chosen token, pivot, context and seed verified." |

## 5. 文件说明

| 文件 | 内容 |
|---|---|
| `l0full0-eval_report.md` | 第一轮评价（cap=25，histogram + beta） |
| `l0full0-eval-nocap_report.md` | beta_nocap 附录（尾部失控，不入主表） |
| `l0full0-eval2_report.md` / `.json` | 同一批 positions 用新代码重评，新增 `IoU(P)` 列 |
| `l0full0-diag.md` / `.json` | 离线诊断五节 |
| `l0full0-metadata.json` | 生成元数据（上表即摘自它） |
