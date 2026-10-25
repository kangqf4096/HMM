# 当前配置下的真实模型枢轴分布实验

本轮沿用 `configs/text_smoke.json`，模型和生成参数均不改变：Qwen2.5-Coder-3B-Instruct、Scientific Papers arXiv 正文、50 token prompt、128 token continuation、m=4、T=1、单卡 FP16、vLLM 0.19.1。继续使用 8 篇发射校准、4 篇开发、8 篇测试、32 篇 null 校准、32 篇 null 测试、8 篇编辑池文章，合计 92 篇独立文章、96 个原始生成请求。

本实验的目的是测量统一经验备择发射是否失配，以及已知更多生成条件能带来多大改善。较小 smoke 样本用于流程验证和初步诊断；本轮不更换模型、语料或人为扩大数据。

## 服务器运行

在已经激活 vLLM 0.19.1 环境、更新了本仓库代码的 Linux 服务器上，从仓库根目录执行：

```bash
source scripts/env.sh
python -m pip install -e ".[inference]"
bash scripts/pivot_smoke.sh /home/MCEC_ADAS/kdz/models/Qwen2.5-Coder-3B-Instruct piv0
```

已有匹配的 `data/arxiv0` 提示词可以复用，第三个参数指定目录：

```bash
bash scripts/pivot_smoke.sh /home/MCEC_ADAS/kdz/models/Qwen2.5-Coder-3B-Instruct piv0 data/arxiv0
```

复用会检查数据集版本、模型路径、prompt 长度、上下文宽度和角色数量。新建提示词目录仍复用项目内 Hugging Face 缓存；不因新 run 名称主动重下已有缓存。若输出已存在且非空，换一个短 run 名称，如 piv1。脚本不清理旧结果。

脚本依次运行本地可用测试、服务器真实 vLLM adapter 自检、准备或复用提示词、生成带概率诊断的文本，以及单独进程中的概率重算和定位评价。vLLM 生成进程退出后才用 Transformers 加载相同模型，避免同时驻留两份模型。所有缓存和临时文件位于项目内。

旧生成结果没有新增的采样前概率记录，无法补出实际采样器的这一审计；可以复用提示词，但本轮需重新 generate。新增选项 `--pivot-diagnostics` 不消耗新随机数，不更改已有 Gumbel 选择规则。

## 分步命令与失败后继续

```bash
python scripts/run.py prepare --config configs/text_smoke.json --output data/piv0
python scripts/run.py generate --config configs/text_smoke.json --prompts data/piv0 --output runs/piv0-gen --pivot-diagnostics
python scripts/run.py diagnose-pivots --input runs/piv0-gen --output runs/piv0-diag
```

如果生成已成功，只在后续诊断失败，直接以新的诊断输出目录重跑最后一条命令即可，无需重新生成：

```bash
python scripts/run.py diagnose-pivots --input runs/piv0-gen --output runs/piv0-diag1
```

后续若明确切到现有 pilot 配置，可以在 prepare/generate 中同时使用 `configs/text_pilot.json`；本轮的一键脚本固定使用 smoke。

## 记录与比较内容

生成回调在屏蔽特殊 token 后、覆盖为 Gumbel 赢家之前，读取真实 logits，按原温度计算 P。每步保存 S2、S3、pmax、Shannon 熵、有效词表大小；启用水印的位置另保存选中 token 的基础概率、解析 CDF F1,P(Y) 和 log 密度。它们同时保存在 `trace/` 和 `sources.jsonl`，不保存整个词表概率向量。

随后通过同一个模型对原始完整历史和编辑后的历史分别做 teacher forcing，计算各位置 P。重算使用相同温度及词表屏蔽，但 Transformers 与 vLLM 的实现和浮点结果可能不同。因此报告单独给出生成时实际概率摘要/CDF/log 密度与重算值的差异，先看这个误差，再解释条件模型的定位效果。

评价保留原有五种方法，额外比较三种发射输入，均使用相同的比例估计、rho 选择、HMM 和独立 null 阈值校准：

| 方法 | 发射与可用信息 |
|---|---|
| 原有 `opt_hmm` | 纯水印校准文章的全局直方图，检测只使用最终枢轴量 |
| `s2_histogram` | 使用模型重算的 S2，按预设区间分组拟合直方图；只使用 calibration 文章，固定 50 个伪样本向全局分布收缩，空组回退全局 |
| `model_context` | 对最终文本加原始 prompt 重算 P，再计算解析 f1,P(Y) |
| `source_alignment` | 对所有有 origin 的位置使用其原始历史 P，无论 Z 是否为 1；插入/替换且无 origin 的位置回退最终历史 P |

三个额外方法都使用了原始 prompt。`source_alignment` 还使用了真实 origin 映射，属于实验 oracle；其所有位置均按同一条规则选择发射，不读取 Z 来选择发射。Z 只用于最后评价和按存活位置分组的诊断。不能将这些方法的效果直接当成无需 prompt、无需原始对齐的检测性能。

S2 区间固定为 [0,.05)、[.05,.2)、[.2,.5)、[.5,.8)、[.8,.95)、[.95,1]。本轮不按 dev/test 结果调区间。极小解析密度的 log 值下限为 -700，截断次数单独记录。

## 查看输出

主要看：

- `runs/piv0-diag/report.md`：真实采样器的 E[-log Y|P]=S2 检查、按 S2 分组的 PIT 距离、编辑后 P 的 TV 距离、seed 保留与否分组、HF/vLLM 概率重算误差。
- `runs/piv0-diag/localization/report.md`：原有五种方法和新增三种发射的 IoU/F1、独立 null 文档误报。
- `runs/piv0-diag/report.json`：每篇残差、存活信号的条件构成、条件得分相对边缘得分的差异，以及各方法使用的信息与限制。
- `runs/piv0-diag/localization/metrics.jsonl`、`reliability.json`：比例误差、Brier、概率可靠性与逐篇定位指标。
- `runs/piv0-diag/positions.jsonl`：逐位置概率摘要、原始/最终条件密度、TV、CDF 差异、是否使用 origin 回退。

PIT/KS 均为描述性距离，不给独立 token p 值。留出 test 文章和按原始文章分组的结果用于判断，而不是以校准集拟合程度代替泛化。纯 null 文档阈值仍不自动构成混合文本 FDR 保证。

## 本地验证范围

Windows 本机已通过 45 项 CPU 测试，包含概率计算不修改输入、极端尾部 log 密度、原有五方法与外部密度比入口的等价性，以及整条诊断/额外定位/null 校准流程。翻转评价标签不会改变发射输入。测试中的模型重算使用可控的三 token 模型，不是 Qwen 实机结果。

服务器的 `check-runtime` 另会使用真实 torch/vLLM 检查启用概率记录前后输出 logits 完全一致，并检查采样前 S2 记录。实际 Qwen/CUDA 生成和 Transformers 重算尚需运行上述服务器命令完成。
