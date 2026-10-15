# OPT-HMM 水印片段定位实验

这是现有基础方案的可运行 v0：从连续枢轴量估计整篇水印信号比例，再用两状态 HMM 计算逐 token 后验，并输出连续片段。定位目标是“相对于指定密钥仍保留水印耦合的位置”。

默认远程环境是 Linux、RTX 3090 24GB、单卡、FP16、vLLM **0.19.1**，模型位置：

    /home/MCEC_ADAS/kdz/models/Qwen2.5-Coder-3B-Instruct

该代码模型先用于流程和协议验收。其结果只说明该模型及该数据配置下的情况；后续可换通用文本模型。检测算法不需要调用 LLM；LLM 只用于生成实验数据。

## 快速运行（Linux）

将整个仓库复制到服务器，在仓库根目录运行。所有临时文件、缓存和输出都留在项目内。新输出目录必须为空，重跑请换一个短名称。

如果已有安装好 vLLM 0.19.1 的环境，先激活它，再运行：

    source scripts/env.sh
    python -m pip install -e ".[inference]"
    python scripts/test.py
    python scripts/run.py check-runtime

如果需要新环境，可先使用 Python 3.12：

    python3.12 -m venv .venv
    source .venv/bin/activate
    source scripts/env.sh
    python -m pip install -e ".[inference]"

不要单独安装另一个版本的 PyTorch 覆盖 vLLM 的依赖；由 vLLM 0.19.1 的安装包选择兼容依赖。驱动/运行时是否可用由实际服务器检查决定。

先跑无需 GPU 的模拟：

    python scripts/run.py simulate --config configs/sim_smoke.json --output runs/sim0

再跑真实模型小试，可以一条脚本顺序完成准备、生成和评价：

    bash scripts/smoke.sh /home/MCEC_ADAS/kdz/models/Qwen2.5-Coder-3B-Instruct text0

等价的分步命令：

    python scripts/run.py prepare --config configs/text_smoke.json --output data/text0
    python scripts/run.py generate --config configs/text_smoke.json --prompts data/text0 --output runs/text0-gen
    python scripts/run.py evaluate --input runs/text0-gen --output runs/text0-eval

模型默认路径已写入配置，也可以为 prepare 和 generate 同时传入相同的 --model 路径。安装包还提供 opt-hmm 命令，与 python scripts/run.py 相同。

## 首轮数据与规模

选择 [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) 的 wikitext-2-raw-v1。官方卡片给出的该子集下载大小约 4.72 MB，适合先验收；没有下载整个 C4 或 WikiText-103。

程序将行恢复成完整文章，以一级文章标题分组，并对官方 train/validation/test 中的重复标题去重，然后将文章随机分配给实验角色。这些官方 split 仅作为数据来源记录，实验角色独立重新划分。每篇文章只生成一个提示词，禁止同一文章跨角色；标题和二级标题不会分别当作独立文档。

| 配置 | 发射校准 | 开发 | 测试 | null 阈值校准 | null 独立测试 | 编辑 token 池 | 续写长度 |
|---|---:|---:|---:|---:|---:|---:|---:|
| text_smoke.json | 8 | 4 | 8 | 32 | 32 | 8 | 128 |
| text_pilot.json | 32 | 16 | 32 | 128 | 128 | 8 | 500 |

这些数量都是独立文章数。smoke 共需要 92 篇合格文章，生成 96 个原始请求；每篇测试/开发文章有完整水印、按块切换、10%/30% 替换、10% 插入、10% 删除六种条件。null 文本生成到所需最长长度，再在相同原始文章内裁剪出各个最终长度；同一长度下每篇文章只参与一次阈值校准。

开发集当前只输出诊断和逐篇指标，基础参数预先固定；代码不会根据测试标签自动选箱数或窗口。若调整开发参数，需要新配置和独立测试运行。

smoke 和 pilot 是流程验收/预实验，32 或 128 篇 null 不足以精确验证 5% 的真实误报率。扩大正式实验时，建议将独立 null 校准和测试各增加到 2000；WikiText-2 可能没有足够独立文章。程序会明确报容量不足。此时主动选择 wikitext-103-raw-v1 并使用新配置/输出，而不是复用一篇文章凑样本。

## 模型、水印与观测协议

- 使用本地目录进行离线推理，数据集从 Hugging Face 下载。未使用远程商业 LLM API。
- 采用原始英文文章前缀续写，不套 chat template。默认提示词 50 token、上下文宽度 m=4、温度 T=1。
- tokens 表示生成的完整 continuation 长度。检测器不接收原 prompt，最前 m 个位置为 unavailable。128 token 文本通常有 124 个可评价位置；500 token 文本有 496 个。插入/删除后按最终长度重新处理。
- 上下文 token IDs 与密钥经 BLAKE2b 生成 128-bit seed，显式 PCG64 生成整个词表的均匀随机数；选中的随机数就是枢轴量。生成和检测使用相同实现，标量验证通过 PRNG advance 直接找到对应 token 的随机数。
- 为避免分母很小，Gumbel-max 用等价的 argmax(logits/T - log(-log(U))) 实现。选定后只保留赢家的有限 logit，避免 vLLM 再做一次随机选择。
- 水印和普通采样使用相同的特殊 token / 无 tokenizer 对应 ID 屏蔽规则，不使用 top-k/top-p 截断、重复惩罚或额外 logit bias，固定长度并忽略 EOS 停止。
- 默认公开研究密钥只为可复现。可以设置 OPT_HMM_KEY（1..64 UTF-8 字节）并更改配置的 key_id；实际 key 不写入结果 metadata。协议中的哈希是种子生成算法，不是文件完整性校验。

seed 的伪随机独立性、相同上下文重复时的依赖，以及编辑后是否回到均匀 null，仍然是需要用真实数据检查的近似。PCG64 不是密码学随机数生成器；这里固定它用于可复现的研究协议。

## 现有算法和输出

共同使用独立纯水印文章拟合的正值、归一化直方图密度。默认 f0=1；按篇比例最大化边际复合似然，包含 e=0 和 e=1 的端点。

比较方法包括：

- independent：rho=0 的独立混合分类。
- fixed_e：e=0.5，按篇选 rho 的 HMM。
- opt_hmm：按篇估计 e，再按前向似然选 rho。
- fixed_rho：按篇估计 e，rho 固定为 0.95。
- window：局部平均 log 密度比，固定 0.5 prior 的 logistic 分数。
- oracle：仅在平稳 Markov 模拟中，使用真实发射和真实生成参数；同样重新校准 null 阈值。

rho 候选固定为 0、0.5、0.8、0.9、0.95、0.98、0.99。前向/后向数值缩放；Viterbi 提供实现和正确性检查，主输出使用后验阈值。缺失位置使用中性发射跨过，最后不可输出，也不能直接连接两侧片段。没有补洞、删短段或强制匹配总比例。

每种方法、每个最终长度及 mask 都独立校准 T=max(score) 的阈值，完整重跑比例拟合和 rho 选择，严格大于阈值才输出。样本不足以取得所需秩时阈值为 1，保守地不输出。该约束是纯 null 文档的 any-FP；不提供混合文本 FDR 保证。

生成结果：

- manifest.json / prompts.jsonl：文章分组、提示词、各角色、模型和数据来源。
- sources.jsonl：原始 token、生成上下文、启用水印位置及原始 seed，供评价使用。
- trace/*.jsonl：生成时实际执行的逐步审计；每个水印赢家都要与返回 token、验证 seed 和枢轴量一致。
- observations.jsonl：最终 token、origin 映射、编辑记录、O/Z 标签、可评价 mask 和枢轴量。
- metadata.json：冻结协议、模型路径、版本、词表支持、推理和加载耗时。

评价结果：

- report.md / report.json：按条件定位结果、null any-FP 与 Wilson 区间、按文章分组的配对 bootstrap。
- density.json / thresholds.json：拟合发射与各方法阈值。
- metrics.jsonl / predictions.jsonl：逐篇指标、分数、标签和区间。
- reliability.json：概率分箱可靠性统计。窗口分数不是 HMM 后验。
- example.svg：固定选取第一篇测试文本的真值、分数和标签示例，不挑最好样本。

区间全部为从 0 开始、右端不含的 token 偏移。主要效果指标是含信号文档的 token 集合 IoU，空输出计 0；纯 null 单列。辅助报告 F1、微平均 F1、FDP、片段召回/边界误差、Brier 和比例误差。片段配对先最大化匹配数，再最大化总 IoU，IoU 门槛 0.5。多个编辑版本按原文文章分组重采样。

## 后续实验与模型替换

较大模拟：

    python scripts/run.py simulate --config configs/sim_main.json --output runs/sim-main0
    python scripts/run.py simulate --config configs/sim_smoke.json --scenario island --segment-length 20 --output runs/island0
    python scripts/run.py simulate --config configs/sim_smoke.json --scenario hole --segment-length 20 --output runs/hole0
    python scripts/run.py simulate --config configs/sim_smoke.json --scenario strong_weak --output runs/weak0
    python scripts/run.py simulate --config configs/sim_smoke.json --signal-p 1 0 --output runs/noinfo0

pilot 使用相同的三步，只把配置替换为 configs/text_pilot.json，并选择新的 data/runs 名称。

以后更换受 vLLM 支持的标准文本模型，通常只需把 prepare 和 generate 的 --model 都指向新位置；tokenizer 与词表会从模型读取。需要重新准备提示词、生成校准/测试样本和计算阈值；旧模型的 density/threshold 不得复用。可能还需调整 dtype、max_model_len、max_num_seqs 和 --gpu-memory。需要自定义代码、特殊聊天模板或多模态输入的模型不在这项直接替换保证内。

当前没有实现条件发射、非齐次 HMM、HSMM、自由 Baum-Welch 或外部 SPOT/变点基线。它们需要在基础结果明确后分别加入。

## 已验证范围与来源

本地 CPU 验证记录见 [docs/validation.md](docs/validation.md)。Windows 本机没有 vLLM/PyTorch 和模型权重，真实 CUDA 生成与实际安装适配器验收由服务器运行 check-runtime 和 smoke.sh 完成。不能把本地模拟通过解释为真实水印协议已通过 GPU 实验。

接口依据 [vLLM 0.19.1 custom logits processors](https://docs.vllm.ai/en/v0.19.1/features/custom_logitsprocs/) 与 [该版本源码](https://github.com/vllm-project/vllm/tree/v0.19.1/vllm/v1/sample/logits_processor)。采用 V1 AdapterLogitsProcessor，初始化时注册类，通过 SamplingParams.extra_args 传递每个请求的设置；没有使用旧版 SamplingParams.logits_processors API。

基础方案来自父目录的研究 note、《问题收敛与算法 v0》和 experiment_plan_v1.md；原始文档未改动。数据缓存、模型、生成日志和实验结果被 Git 忽略，仓库只保存代码、配置、文档与小型验证记录。
