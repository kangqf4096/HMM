# 零假设、备择分布与上下文密钥：项目核对及论文笔记

整理日期：2026-10-25。结论以当前仓库实现和文末列出的原始论文为依据。本文区分实际实现、论文的理想随机模型，以及尚未验证的近似。

关于备择对模型概率的依赖、熵/碰撞概率/KL 的解析关系、经验边缘分布的适用性和失配反例，见后续 [数学推导与数值核对](pivot_law_derivation_20261025.md)。

## 1. 当前项目是不是需要备择分布？

**是：当前 OPT-HMM v0 需要估计水印状态下枢轴量的发射密度 f1。它不需要检测时调用 LLM，也不需要知道每个位置完整的下一词分布 P_t。**

代码依据：

- `src/opt_hmm/experiment.py` 的 `run_evaluation` 从独立纯水印校准文章收集 Y，再调用 `Histogram.fit`。
- `src/opt_hmm/emissions.py` 的 `Histogram` 通过正伪计数拟合积分为 1 的直方图密度。
- `src/opt_hmm/detector.py` 的 `Detector.infer` 把密度值作为 g(Y)=f1(Y)/f0(Y)；当前 f0=1。
- `src/opt_hmm/proportion.py` 最大化边际复合对数似然 sum_t log(1-e+e*g(Y_t))，包含 e=0、e=1 真端点。
- HMM 再以两状态发射比和状态转移结构计算逐位置后验；它的成立需要发射与状态模型具有代表性。
- 最终是否输出片段，另用独立纯 null 文档校准最大分数阈值。这主要控制纯 null 文档上“至少输出一个假阳性位置”的事件；不自动给出混合文本 FDR 或后验概率准确性。

因此“需要知道备择”应表述为“需要一个校准估计的水印枢轴密度”，不能表述为“已知每个位置真实备择分布”。实际的备择随 P_t 变化，代码用一个共同 f1 近似它们的总体分布；模型、温度、语料或文本类型变化都可能造成失配。

| 任务或方法 | 设定误报阈值需要什么 | 是否需要事先获得完整 f1 |
|---|---|---|
| KGW 绿名单计数检验 | 零假设下的绿 token 概率 gamma 与有效计数的分布/校准 | 通常不需要 |
| Gumbel 得分和、均匀性检验、Tr-GoF | 零假设下的枢轴量或统计量分布 | 运行这类检验通常不需要；分析功效仍需描述备择类 |
| 简单假设之间的 Neyman-Pearson 似然比检验 | 两个指定分布或可计算密度比 | 需要相应的两分布信息 |
| 原始 OPT 的初始估计量 | 已知 F0，以及备择在零附近相对 F0 消失等结构 | 不需要完整 f1，但有结构假设、偏差和方差代价 |
| 原始 OPT 的 refined/optimal 估计量 | 额外纯水印样本估计平均备择 CDF / 密度比 | 使用估计的备择信息 |
| 当前 OPT-HMM v0 | 独立纯水印校准的 f1，加独立 null 阈值校准 | 需要估计 f1 |
| SPOT | 已知零假设尾概率、观测尾计数和水印存活比例估计 | 核心定位规则不需要完整逐位置备择密度；比例估计的选择可引入额外备择信息 |

原始 OPT 的第 3.3 节（PDF 第 8-10 页）明确区分初始、修正和最优估计量：初始估计量利用 F1(x)/F0(x) 在 x→0 时消失；修正与最优估计量利用额外纯水印枢轴样本估计平均备择。当前仓库使用边际复合似然求比例，不能直接声称逐项复现了论文的固定点迭代或继承了全部理论。

SPOT v2 第 3.2 节（PDF 第 8-10 页）用“预期 null 尾部数量 / 实际尾部数量”选择阈值，避免估计完整异质备择。其 plug-in 比例估计如果选用 OPT 的 optimal 版本，仍会使用该估计器所需的信息。SPOT 的理论限定于论文指定的枢轴规则类、假设和渐近设置，不能无条件推广到当前 HMM。

## 2. 零假设是哪一个？

文档层面：

- H0：没有针对**当前指定密钥**的存活水印证据。
- H1：至少有一些位置保留了该密钥的水印耦合。

位置层面：

- H0,t：Z_t=0，当前位置不保留该密钥下的水印耦合。
- H1,t：Z_t=1，当前位置仍保留水印生成时的耦合。

H0 不能简单等同于“人写的”。它可以包括未加水印的 LLM 输出、其他密钥的输出，以及原先来自水印 LLM 但相关耦合已被编辑破坏的位置。跨密钥独立性、编辑后是否真的回到 null，都需要满足相应随机模型或由实验检查。

当前协议构造每个词表坐标的随机数 U_{t,w}，观测枢轴量为 Y_t=U_{t,w_t}。理想模型给出：

\[
H_{0,t}:\quad Y_t\sim \operatorname{Unif}(0,1),\quad f_0(y)=1.
\]

若 w_t 由 Gumbel-max 生成，即 w_t=argmax_w log(U_{t,w})/P_{t,w}，则

\[
H_{1,t}:\quad F_{1,P_t}(y)=\sum_{w:P_{t,w}>0}P_{t,w}y^{1/P_{t,w}},
\qquad
f_{1,P_t}(y)=\sum_{w:P_{t,w}>0}y^{1/P_{t,w}-1}.
\]

这里 **P_t 是 token 的下一词概率分布，f1 是枢轴量 Y 的密度**，两者是不同对象。当前检测器拟合后者。

这些公式来自统计框架论文及 Gumbel-max 理想随机模型，也与 `emissions.py` 的 `theoretical_cdf` / `theoretical_density` 一致。对于实际固定密钥和上下文哈希，均匀性、跨位置独立性不能逐字当作无条件精确事实：相同上下文会复用随机向量，当前 PRNG 也只是固定的研究协议。

一个例子：词表只有两个 token，P=(1/2,1/2)。无耦合时 Y~U(0,1)；有水印时选中 U1、U2 中较大的一个，所以 F1(y)=y²，f1(y)=2y。有水印的 Y 更容易靠近 1，但取值范围与 null 重叠，单个位置通常不能百分之百区分。

当前 `protocol.py` 还区分 O（直接保留自启用水印的原输出）与 Z（O 成立且最终前 m 个 token 与生成上下文相同）。一个 token 可以 O=1、Z=0。因此项目定位的是存活水印耦合，不能把结果解释为精确的人类/AI 作者来源标签。

## 3. 哪些论文用上下文生成逐步密钥或随机量？

常见结构是固定主密钥 K，再派生每步 seed：

\[
s_t=\operatorname{PRF}_K(c_t),\qquad U_{t,w}=G(s_t,w),
\]

其中 c_t 可以取前一个 token、前 m 个 token、全部前缀或上下文语义编码；G 可以由 PRNG 向量生成或按坐标求值实现。检测者从待测文本重算 c_t、s_t 与相应分数。主密钥一般不需要每个 token 更换。

| 路线与论文 | 如何使用上下文/密钥 | 对本项目的意义 |
|---|---|---|
| KGW，Kirchenbauer 等，ICML 2023 | 基础版本由前一个 token 的哈希初始化 RNG，形成绿/红名单；密钥化及更宽窗口是常见扩展 | 理解上下文自同步与绿名单计数 |
| Aaronson/Gumbel-max；Three Bricks，Fernandez 等，2023；Li 等统计框架，AoS 2025 | 前 h 个 token 与主密钥经哈希得到每步 seed，生成词表随机向量，确定性选择赢家 | 与当前研究协议最接近；Three Bricks 第 II-D、III-C 节直接讨论密钥多样性和重复上下文 |
| Unbiased Watermark，Hu 等，ICLR 2024 | context code 可以是全部前缀或末 m 个 token；hash(context code,key) 生成 watermark code；重复 code 时跳过重加权 | 明确区分单步无偏、整段/多次请求无偏及历史去重 |
| DiPmark，Wu 等，ICML 2024 | 上下文与私钥派生随机重排/水印随机量，再做保分布重加权 | 说明上下文派生与采样机制是可分开的选择 |
| SynthID-Text，Dathathri 等，Nature 2024 | 局部上下文与 key 派生 seed，对候选 token 计算多层 g 值，进行 tournament sampling；可使用重复上下文屏蔽 | 固定主密钥并不要求输出全程确定；采样算法仍可保留额外随机性 |
| SemaMark，Ren 等，NAACL Findings 2024 | 用上下文的语义表示及离散化值稳定地决定词表分区 | 对改写更稳健的另一类上下文编码，不能直接套用当前 Gumbel 枢轴密度 |
| Unigram-Watermark，Zhao 等，ICLR 2024 | 固定词表分组，不依赖逐步文本上下文 | 对照：固定分组并非不能用；有鲁棒性、质量和统计依赖的取舍 |
| Kuditipudi 等，Robust Distortion-free Watermarks，2023/2024 | 预共享随机序列，可随机循环移位；检测时对文本与随机序列做对齐；另讨论 hash 版本 | 对照：持续推进随机流也是有效路线，代价是同步和多请求复用管理 |
| WaterPool，Huang、Wan，NAACL 2025 | 生成时保留更大的随机 key 采样空间，用语义检索恢复 key | 不能将所有语义方法都归成“语义哈希生成 seed”；WaterPool 主要改变 key 恢复方式 |

“Gumbel-max”是采样/嵌入方式，“局部上下文哈希”是逐步随机量的生成与恢复方式。两者应分开描述；同一种采样可以配上下文哈希或外部随机序列。

## 4. 同一个随机密钥不停迭代，会不会完全确定？

需要区分以下情况：

| 实现 | 是否每步得到同一随机向量 | 结论 |
|---|---|---|
| 每步重新 RNG(K)，直接取第一组随机数 | 是 | 容易产生重复偏向；但采用随机 categorical 采样的绿名单方案仍可有输出随机性。Unigram 是固定分组的正式设计，不能一概判无效 |
| RNG(K) 只初始化一次，状态持续前进 | 否；依次输出 U1、U2、… | 正常的伪随机流。全序列由 seed 决定，不表示不同位置取同一数；检测必须恢复序号、偏移或做对齐 |
| 每步 RNG(PRF_K(末 m 个 token)) | 新上下文通常给出不同向量；相同上下文给出相同向量 | 易从局部文本恢复；可在后续窗口完全匹配原上下文时重新同步；重复上下文造成随机量复用 |
| 每篇加入 nonce/随机偏移 | 不同请求可以用不同随机量 | 能改善跨请求多样性；nonce/偏移必须可恢复、可检索、随文传递或纳入搜索，并重新校准完整检测过程 |

**若使用完全由密钥派生随机量的 Gumbel-max，固定主密钥、完整 prompt、模型、参数和确定性计算后，整个 continuation 确实是确定的。** 同一个 seed 持续推进的随机流也一样可复现。上下文哈希本身不会给相同 prompt 增加新的运行时熵。

这不是水印失效的充分条件。统计分布是对模型中被当作随机的 key/随机向量、不同样本等对象定义的；“条件于所有随机输入后是确定的”与“在随机模型下有分布”完全相容。需要额外区分多次相同 prompt 的输出多样性、密钥复用后的联合分布，以及单次随机量边际上的无偏。不能把单步 Gumbel-max 恒等式直接提升为固定 key 下无限请求的整体无偏保证。

Kuditipudi 等第 2.2 节明确讨论 deterministic `generate` 的重复输出，并用随机 shift 或避免重用序列元素来管理。Unbiased Watermark 第 4.2.2 节及附录 C 讨论 context code history；SynthID 的 Repeated context masking 也专门处理相同 seed 的重复偏向。检测阶段对重复上下文/重复 context-token 去重，与生成阶段跳过水印是不同操作，不能混为一谈。

## 5. 对当前实现的具体判断

当前 `protocol.py` 已经采用固定主密钥 + 最近 m 个 token 的密钥化 BLAKE2b + PCG64，而非每步重置成同一个 seed。默认 m=4。`vllm_processor.py` 选定 Gumbel 赢家后屏蔽其他 logits，确保后续不再随机改选。

因此不用为了“固定主密钥”本身更换协议。当前值得实证检查的是：

1. 独立纯水印校准 f1 对测试语料及各类编辑位置是否具有代表性；固定 f1 的后验可靠性和比例误差如何。
2. 重复上下文及 context-token 的比例、同 prompt 跨请求重复程度，以及与 null 偏离、定位误差的关系。
3. 编辑破坏上下文后，Z=0 的位置是否近似均匀；按独立文档而非 token 评估误报与不确定性。
4. 把生成时重复上下文跳过、检测时去重、SPOT 等作为后续独立变体/基线评估。改变生成规则将影响 enabled/Z、发射校准和阈值，不宜直接无声替换现有协议。

没有在这次文献整理中修改水印生成、检测或校准算法。

## 6. 可离线使用的原始资料

文献总索引：[papers/README.md](../papers/README.md)。机器索引：[papers/index.json](../papers/index.json)。PDF 在 `papers/pdf/`，逐页抽取文本在 `papers/text/`；原本已在 `docs/` 的定位论文保持原位置，只登记索引。

直接对应本次问题的原文：

- [KGW](https://proceedings.mlr.press/v202/kirchenbauer23a.html)：生成算法、零假设和绿名单检验。
- [Three Bricks](https://arxiv.org/abs/2308.00113)：II-D Key Management、III-B/C 检验与重复分数处理。
- [Unbiased Watermark](https://arxiv.org/abs/2310.10669)：4.2.2 Context code、附录 C 的重复上下文讨论。
- [Kuditipudi 等](https://arxiv.org/abs/2307.15593)：2.1 协议与随机化检验、2.2 随机移位和多请求设置。
- [统计框架](https://arxiv.org/abs/2404.01245)：零假设枢轴性质与 Gumbel-max 备择分布。
- [Tr-GoF](https://arxiv.org/abs/2411.13868)：编辑混合模型及不依赖精确备择的自适应检测。
- [OPT](https://arxiv.org/abs/2506.22343)：3.2 可识别性、3.3 三类估计量，不能混淆信息要求。
- [SPOT v2](https://arxiv.org/abs/2608.14906v2)：3.2 尾阈值扫描、附录实现中的 plug-in 比例估计。
- [SynthID-Text](https://www.nature.com/articles/s41586-024-08025-4)：Random seed generator、Repeated context masking，以及补充材料 G。
- [SemaMark](https://aclanthology.org/2024.findings-naacl.40/)：3.1 语义编码与词表划分。
- [DiPmark](https://arxiv.org/abs/2310.07710)、[Unigram](https://arxiv.org/abs/2306.17439)、[WaterPool](https://aclanthology.org/2025.naacl-long.209/)：不同 key 模块及采样机制的对照。

这是一组针对本次问题的代表性原始论文，不是所有水印方法的穷尽综述。后续先搜索本地缓存；只有出现资料缺口、明确需要新版本或新问题时再联网补充。
