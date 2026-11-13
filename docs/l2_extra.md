# L2 补充臂：Beta 自适应 f1(e′)（`fb`）与 SPOT 基线

在已完成的 L2 运行（如 `l2q1`）上追加，不改动已冻结的 `fe`、`m400`、`seedbs`。

- **fb**：与 `fe` 用同一套 e′ 核权重（h 网格、cap 网格、kernel_grid、PD/PE 全部照旧由 dev 选），但每个格点的 f1 是加权矩估计的 Beta(a,b)，再按箱平均成 hist200 形式。回答"光滑参数族能否代替非参核"。
- **spot**：SPOT-plugin（`docs/2608.14906v2`，Algorithm 1）。ε̂ 用冻结 hist200 的 ε_opt；尾部网格 u∈[0.005,0.98]，M=max(20,⌈3.5·ln(n)^1.5⌉)，η=0；常数 C 在 dev 上选。两个口径：`spot`（null_cal 文档级 any-FP ≤ α，可能无可行 C）与 `spot_fpr`（null_cal 逐位置误报 ≤ α）。与论文的差别：无阈值满足准则时输出空（论文回退到最大 u）。不经过后验 τ，不是对论文协议的复现。
- 其余论文基线（AOL、WISER）需要外部实现与上千组调参，本轮未接入。

## 运行（CPU，Linux）

```bash
python scripts/run.py l2-extra --frozen runs/l2q1-fit --train runs/l2q1-train-pos \
  --input runs/l2q1-test-pos --reference runs/l2q1-test --output runs/l2q1-extra
```

输出 `report.md`（IoU、配对差 fb−fe、fb−m400、spot−fe/seedbs、null any-FP）、`frozen_extra.json`、`fit_extra.json`。fb 与 spot 先在训练侧冻结再读测试侧；参照方法的逐篇结果取自 `--reference/metrics.jsonl`，所以差值是同一批文章的配对 bootstrap。结果为一次运行，不据此回头改 `fe` 的冻结参数。
