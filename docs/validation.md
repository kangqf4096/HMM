# 本地验证记录

日期：2026-10-15。环境：Windows，Python 3.12.14，NumPy；具体 NumPy 版本和全部结果保存在 validation.json。

完成的检查：

- 33 项 unittest 全部通过。包含 n<=12 穷举状态路径核对前向似然、逐点后验和 Viterbi；rho=0 与独立混合模型一致；长序列缩放、比例端点/不可辨识、CDF 模拟、PRNG 跳转、插入/删除/替换的上下文恢复、片段匹配、分组泄漏和缺失长度校准拒绝。
- Python 全模块编译通过；CLI 帮助正常。
- bash -n 对 env.sh / smoke.sh 的语法检查通过。
- 三个完整模拟运行均已生成密度、阈值、逐篇指标、概率可靠性、配对区间和 SVG 诊断图。
- 原始研究文档未改动；结果、缓存和临时文件全部在新项目目录内。

可在 Linux 用以下命令复现同样的数值流程：

    python scripts/test.py
    python scripts/run.py simulate --config configs/sim_smoke.json --output runs/sim0
    python scripts/run.py simulate --config configs/sim_smoke.json --scenario island --segment-length 20 --test-documents 8 --output runs/island0
    python scripts/run.py simulate --config configs/sim_smoke.json --signal-p 1 0 --test-documents 8 --output runs/noinfo0

## Markov 小试结果

配置为 n=128、epsilon=0.3、rho=0.95、P=(0.5,0.5)，10000 个独立发射校准枢轴，64 篇 null 校准、64 篇独立 null 测试、24 篇混合测试。随机生成的 24 篇中有 3 篇实际全为 0，主要 IoU/F1 只统计其余 21 篇，未删除原始全 0 轨迹。

| 方法 | 含信号篇均值 IoU | 独立 null any-FP |
|---|---:|---:|
| independent | 0.1628 | 0/64 |
| fixed_e | 0.3546 | 5/64 |
| opt_hmm | 0.3205 | 2/64 |
| fixed_rho | 0.3553 | 2/64 |
| window | 0.0905 | 3/64 |
| oracle | 0.3495 | 3/64 |

这些结果只作为流程跑通的证据。null 误报估计区间较宽，各方法经验误报并不相同；样本量也很小。不能由该表声称相同实际误报率下存在确定的定位提升，更不能声称按篇 OPT 比例优于固定比例。

固定 20-token 水印岛和理论 P=(1,0) 的无信息对照也已经完成。后者 oracle 使用准确的 f1=f0，输出“不可辨识”并不标段；估计密度版本仍经完整的独立 null 校准，不向检测器泄露理论不可辨识标签。

## 2026-10-17 arXiv 数据源替换验证

本地 Windows / Python 3.12.14 验证：38 项 unittest 全部通过；新增检查覆盖完整论文的角色隔离、正文与摘要字段选择、跨来源 split 的正文去重、短/空文本过滤、固定 seed 抽样复现、提前停止分词及 token 池长度、数据源版本写入清单，以及在 GPU 启动前拒绝旧语料清单。准备流程测试使用小型内存语料和替代 tokenizer，不代表已经完成真实语料下载或模型分词。

同日增加服务器迁移前的清理脚本后，共 40 项 unittest 全部通过。额外验证清理 WikiText 缓存、下载文件、解压数据及旧提示词，保留其他数据集/模型/实验结果，可重复运行，并在删除前拒绝解析到项目外的目标。清理测试仅使用项目 tmp 内创建的小型目录。

2026-10-15 的 33 项测试及上方模拟数值为历史记录，validation.json 保留当时的结果。

## 仍需在服务器完成的验收

本机没有执行真实 arXiv 全量下载、实际模型分词或 CUDA 生成，以下步骤需在服务器运行：

1. 在 Linux 的实际 vLLM 0.19.1 环境运行 check-runtime，验证 CUDA 和真实 AdapterLogitsProcessor 的 token 引用、批次交换及请求移除行为。
2. prepare 下载 ccdv/arxiv-summarization 的 document Parquet 配置、使用实际模型 tokenizer，核对论文容量和冻结角色；所有输出使用新的 arxiv 名称。
3. generate 完成逐 token 审计：每个实际水印赢家必须与返回 token、上下文、seed 和枢轴量一致。
4. evaluate 查看真实 null 均匀性/相关性、重复上下文比例、实际 any-FP、IoU 和可靠性，才能判断该基础方案在当前代码模型上是否适用。

第一轮采用优先可检查的 CPU PRNG + GPU logits 计算 + 逐 token 日志实现，GPU 吞吐尚未测量。
