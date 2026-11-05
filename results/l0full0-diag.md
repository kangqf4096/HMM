# L0 诊断（离线重放）

- 重校准抽样次数：200；α = 0.05；fraction_tau = 0.5
- 只重算阈值与决策；不重新拟合发射、不重新生成文本。

## 1. 阈值复核

| 规则 | tau | rank | 与 frozen 一致 |
|---|---:|---:|---|
| histogram|independent | 0.479883 | 381 | 是 |
| histogram|fixed_e | 0.171388 | 381 | 是 |
| histogram|opt_hmm | 0.596822 | 381 | 是 |
| histogram|fixed_rho | 0.139253 | 381 | 是 |
| histogram|window | 0.478844 | 381 | 是 |
| beta|independent | 0.147132 | 381 | 是 |
| beta|fixed_e | 0.095279 | 381 | 是 |
| beta|opt_hmm | 0.147132 | 381 | 是 |
| beta|fixed_rho | 0.002206 | 381 | 是 |
| beta|window | 0.447284 | 381 | 是 |

## 2. null 误报：单阈值 vs 重校准

| 规则 | FP(null_cal) | FP(null_test) | 重校准 p5 | 中位 | p95 | P(FP>α) |
|---|---:|---:|---:|---:|---:|---:|
| histogram|independent | 0.0475 | 0.0550 | 0.0275 | 0.0450 | 0.0750 | 0.405 |
| histogram|fixed_e | 0.0475 | 0.0550 | 0.0275 | 0.0475 | 0.0750 | 0.395 |
| histogram|opt_hmm | 0.0475 | 0.0575 | 0.0250 | 0.0500 | 0.0751 | 0.435 |
| histogram|fixed_rho | 0.0475 | 0.0525 | 0.0275 | 0.0475 | 0.0750 | 0.400 |
| histogram|window | 0.0475 | 0.0425 | 0.0300 | 0.0500 | 0.0701 | 0.445 |
| beta|independent | 0.0475 | 0.0825 | 0.0275 | 0.0475 | 0.0801 | 0.410 |
| beta|fixed_e | 0.0475 | 0.0450 | 0.0275 | 0.0500 | 0.0750 | 0.445 |
| beta|opt_hmm | 0.0475 | 0.0850 | 0.0275 | 0.0488 | 0.0801 | 0.420 |
| beta|fixed_rho | 0.0475 | 0.0875 | 0.0299 | 0.0500 | 0.0800 | 0.425 |
| beta|window | 0.0475 | 0.0300 | 0.0275 | 0.0475 | 0.0725 | 0.405 |

## 3. 原子诊断（τ 附近的重复取值）

| 规则 | 角色 | 不同取值 | τ 处原子 | ≥τ | >τ | 最大原子占比 |
|---|---|---:|---:|---:|---:|---:|
| histogram|independent | null_cal | 199 | 1 | 20 | 19 | 0.505 |
| histogram|independent | null_test | 188 | 1 | 22 | 22 | 0.532 |
| histogram|fixed_e | null_cal | 400 | 1 | 20 | 19 | 0.003 |
| histogram|fixed_e | null_test | 400 | 1 | 22 | 22 | 0.003 |
| histogram|opt_hmm | null_cal | 199 | 1 | 20 | 19 | 0.505 |
| histogram|opt_hmm | null_test | 188 | 1 | 23 | 23 | 0.532 |
| histogram|fixed_rho | null_cal | 199 | 1 | 20 | 19 | 0.505 |
| histogram|fixed_rho | null_test | 188 | 1 | 21 | 21 | 0.532 |
| histogram|window | null_cal | 400 | 1 | 20 | 19 | 0.003 |
| histogram|window | null_test | 400 | 1 | 17 | 17 | 0.003 |
| beta|independent | null_cal | 35 | 1 | 20 | 19 | 0.915 |
| beta|independent | null_test | 51 | 1 | 33 | 33 | 0.875 |
| beta|fixed_e | null_cal | 400 | 1 | 20 | 19 | 0.003 |
| beta|fixed_e | null_test | 400 | 1 | 18 | 18 | 0.003 |
| beta|opt_hmm | null_cal | 35 | 1 | 20 | 19 | 0.915 |
| beta|opt_hmm | null_test | 51 | 1 | 34 | 34 | 0.875 |
| beta|fixed_rho | null_cal | 35 | 1 | 20 | 19 | 0.915 |
| beta|fixed_rho | null_test | 51 | 1 | 35 | 35 | 0.875 |
| beta|window | null_cal | 400 | 1 | 20 | 19 | 0.003 |
| beta|window | null_test | 400 | 1 | 12 | 12 | 0.003 |

## 4. 等误报工作点（按 null_test 的 max 分位取阈值）

| 规则 | full_wm | resample_0.1 | resample_0.3 | switch |
|---|---:|---:|---:|---:|
| histogram|independent | 0.975 | 0.688 | 0.429 | 0.600 |
| histogram|fixed_e | 0.989 | 0.787 | 0.502 | 0.888 |
| histogram|opt_hmm | 0.957 | 0.774 | 0.470 | 0.893 |
| histogram|fixed_rho | 0.995 | 0.761 | 0.481 | 0.885 |
| histogram|window | 0.996 | 0.672 | 0.315 | 0.881 |
| beta|independent | 0.999 | 0.675 | 0.449 | 0.592 |
| beta|fixed_e | 0.995 | 0.746 | 0.470 | 0.878 |
| beta|opt_hmm | 0.990 | 0.794 | 0.588 | 0.900 |
| beta|fixed_rho | 0.998 | 0.682 | 0.453 | 0.757 |
| beta|window | 0.997 | 0.675 | 0.336 | 0.869 |

## 5. rho 选择方式对比

### histogram

| 变体 | tau | any-FP | full_wm | resample_0.1 | resample_0.3 | switch |
|---|---:|---:|---:|---:|---:|---:|
| argmax | 0.5968 | 0.0575 | 0.964 | 0.786 | 0.486 | 0.895 |
| marginal | 0.4476 | 0.0500 | 0.983 | 0.814 | 0.524 | 0.897 |
| fixed_0.5 | 0.7052 | 0.0525 | 0.953 | 0.733 | 0.394 | 0.685 |
| fixed_0.8 | 0.5585 | 0.0475 | 0.979 | 0.825 | 0.476 | 0.860 |
| fixed_0.95 | 0.1393 | 0.0525 | 0.996 | 0.756 | 0.494 | 0.877 |
| fixed_0.99 | 0.0101 | 0.0700 | 0.997 | 0.682 | 0.442 | 0.821 |

marginal 在 null_cal 上的平均 rho 权重：`[0.185, 0.154, 0.133, 0.13, 0.131, 0.133, 0.134]`（网格 [0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]）

### beta

| 变体 | tau | any-FP | full_wm | resample_0.1 | resample_0.3 | switch |
|---|---:|---:|---:|---:|---:|---:|
| argmax | 0.1471 | 0.0850 | 0.996 | 0.763 | 0.558 | 0.890 |
| marginal | 0.0380 | 0.0775 | 0.997 | 0.684 | 0.437 | 0.841 |
| fixed_0.5 | 0.0807 | 0.0850 | 0.998 | 0.696 | 0.480 | 0.629 |
| fixed_0.8 | 0.0221 | 0.0850 | 0.998 | 0.688 | 0.435 | 0.672 |
| fixed_0.95 | 0.0022 | 0.0875 | 0.998 | 0.661 | 0.395 | 0.677 |
| fixed_0.99 | 0.0001 | 0.0925 | 0.998 | 0.645 | 0.382 | 0.657 |

marginal 在 null_cal 上的平均 rho 权重：`[0.145, 0.143, 0.142, 0.142, 0.142, 0.143, 0.143]`（网格 [0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]）

