# Real-model pivot diagnostics

Read alongside localization/report.md and report.json.

All added methods use original prompts. Source alignment is an evaluation oracle.

## Actual sampler: E[-log Y | P] = S2(P)

| Role | S2 interval | Positions | Mean X | Predicted mean X | PIT KS distance |
|---|---|---:|---:|---:|---:|
| calibration | [0.0, 0.05] | 283 | 0.02786 | 0.02593 | 0.03981 |
| calibration | [0.05, 0.2] | 372 | 0.11179 | 0.10988 | 0.03521 |
| calibration | [0.2, 0.5] | 157 | 0.29777 | 0.32459 | 0.04522 |
| calibration | [0.5, 0.8] | 85 | 0.71836 | 0.64243 | 0.06699 |
| calibration | [0.8, 0.95] | 48 | 0.83464 | 0.88153 | 0.10319 |
| calibration | [0.95, 1.0000001] | 47 | 0.85126 | 0.98077 | 0.08867 |
| dev | [0.0, 0.05] | 156 | 0.02996 | 0.02554 | 0.06388 |
| dev | [0.05, 0.2] | 220 | 0.09777 | 0.10554 | 0.05370 |
| dev | [0.2, 0.5] | 121 | 0.34324 | 0.30564 | 0.07051 |
| dev | [0.5, 0.8] | 58 | 0.39743 | 0.64133 | 0.19069 |
| dev | [0.8, 0.95] | 63 | 0.45961 | 0.90602 | 0.39472 |
| dev | [0.95, 1.0000001] | 134 | 0.64652 | 0.99195 | 0.30961 |
| test | [0.0, 0.05] | 412 | 0.02253 | 0.02511 | 0.03843 |
| test | [0.05, 0.2] | 509 | 0.09539 | 0.10818 | 0.03492 |
| test | [0.2, 0.5] | 290 | 0.28764 | 0.32008 | 0.04735 |
| test | [0.5, 0.8] | 119 | 0.50185 | 0.63723 | 0.12419 |
| test | [0.8, 0.95] | 83 | 0.61911 | 0.88000 | 0.29514 |
| test | [0.95, 1.0000001] | 91 | 0.52391 | 0.98188 | 0.47169 |

## Original versus edited history (test documents)

| Variant | Seed preserved | Positions | Mean TV(Pgen, Peval) | Max TV | Mean CDF difference at Y |
|---|---|---:|---:|---:|---:|
| delete_0.1 | False | 286 | 0.49776 | 0.99960 | 0.09289 |
| delete_0.1 | True | 610 | 0.13359 | 0.92768 | 0.03504 |
| insert_0.1 | False | 318 | 0.45041 | 0.99966 | 0.07354 |
| insert_0.1 | True | 678 | 0.12108 | 0.90576 | 0.04103 |
| replace_0.1 | False | 296 | 0.53099 | 0.99996 | 0.10620 |
| replace_0.1 | True | 602 | 0.15764 | 0.92114 | 0.04644 |
| replace_0.3 | False | 536 | 0.65800 | 0.99986 | 0.12302 |
| replace_0.3 | True | 159 | 0.34058 | 0.96627 | 0.11628 |
| switch | True | 992 | 0.00000 | 0.00000 | 0.00000 |
| watermarked | True | 992 | 0.00000 | 0.00000 | 0.00000 |

## Replay accuracy

HF replay is an approximation to vLLM; check these errors before interpreting conditional emission comparisons.

- s2_abs_difference: mean=0.00043574687590339304, max=0.00920279120811024
- pmax_abs_difference: mean=0.0006100927849317331, max=0.007929236501716508
- cdf_abs_difference: mean=0.0003222986762604147, max=0.005262793022480339
- log_pdf_abs_difference: mean=0.0012430278776076706, max=0.03097356587494371

No IID token-level p-values. Document-group residuals and full per-position diagnostics are in report.json/positions.jsonl.
Each added localization method reruns the complete independent null calibration at the original length/mask.
