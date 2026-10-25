# 水印论文缓存与阅读索引

整理日期：2026-10-25。主题：零假设与备择分布、上下文派生密钥、伪随机采样的确定性、重复上下文和编辑后同步。

当前索引共 18 份 PDF：16 篇研究论文，其中 SPOT 保留两个版本，另含 SynthID 补充材料。新增下载 12 份，从父目录复用 4 份，登记仓库已有 2 份；PDF 合计约 34 MB。已检查 PDF 可解析性、页数、全文缓存及本页与研究笔记中的本地链接，并渲染核对关键公式所在页面。

本次问题的完整结论及代码对应关系见 [研究笔记](../docs/watermark_hypotheses_and_keys_20261025.md)。

后续讨论的 [解析关系与边缘分布失配推导](../docs/pivot_law_derivation_20261025.md) 给出 Gumbel 枢轴与 S2/S3、熵和 KL 的联系，以及使用统一经验发射的适用边界。

## 使用方式

- `pdf/`：从 arXiv、PMLR、ACL Anthology、Nature 等原始来源保存的论文，或从仓库父目录复用的已有论文。
- `text/`：逐页抽取文本，每页以 `=== PDF PAGE n ===` 标记。适合检索；公式、图表和排版以 PDF 为准。
- `index.json`：标题、原始来源、实际下载 URL（可用时）、本地路径、版本（PDF 可识别时）、页数、大小及获取方式。使用相对路径，复制到 Linux 后可继续使用。
- 原本就在 `docs/` 的 SPOT v1/v2 只登记已有位置，没有再次下载。
- 父目录已有的 4 篇水印论文复制进本 Git 仓库，使其可随仓库迁移；没有重新联网获取它们。原文件保留。
- 本目录应作为文献缓存纳入版本管理。下载版不保证是在线最新版本，实际版本以索引与 PDF 标识为准；本次整理也没有刷新已有论文。

在 Linux 的仓库根目录离线检索：

```bash
rg -n -i 'null|alternative|pivotal' papers/text
rg -n -i 'context code|repeated context|key management|shift-generate' papers/text
rg -n -i 'identifiability|optimal weight|surviving fraction' papers/text
```

先查此索引、研究笔记和已有全文，再决定是否需要新检索。无需执行文件哈希校验。

## 推荐阅读顺序

1. **零假设与项目 f1**：统计框架 → Tr-GoF → OPT 第 3.3 节 → 项目 `docs/algorithm.md`。
2. **上下文密钥与复用**：Three Bricks 的 II-D、III-C → Unbiased Watermark 的 4.2.2、附录 C → SynthID 的 Repeated context masking。
3. **另一种随机流路线**：Robust Distortion-free Watermarks 第 2.1、2.2 节。
4. **不先估计完整异质备择的定位**：SPOT v2 的 3.2 节；其比例估计器也需要单独核对信息要求。
5. **扩展设计**：DiPmark、SemaMark、Unigram、WaterPool。

## 论文目录

| ID | 论文与发表/版本信息 | 本地 PDF | 原始来源 | 关注点 |
|---|---|---|---|---|
| kgw | A Watermark for Large Language Models；ICML 2023 | [PDF](pdf/2023_icml_kgw.pdf) | [PMLR](https://proceedings.mlr.press/v202/kirchenbauer23a.html) | 前 token 哈希、绿名单、零假设检验 |
| bricks | Three Bricks to Consolidate Watermarks for Large Language Models；2023，arXiv v2 | [PDF](pdf/2308.00113v2_three_bricks.pdf) | [arXiv](https://arxiv.org/abs/2308.00113) | 上下文 key、多样性与鲁棒性、重复分数 |
| robust | Robust Distortion-free Watermarks for Language Models；2023/2024，arXiv v3 | [PDF](pdf/2307.15593v3_distortion_free.pdf) | [arXiv](https://arxiv.org/abs/2307.15593) | 随机序列、随机 shift、对齐、随机化检验 |
| unbiased | Unbiased Watermark for Large Language Models；ICLR 2024，arXiv v2 | [PDF](pdf/2310.10669_unbiased.pdf) | [arXiv](https://arxiv.org/abs/2310.10669) | context code、历史去重、无偏重加权 |
| dipmark | A Resilient and Accessible Distribution-Preserving Watermark for Large Language Models；ICML 2024，arXiv v2 | [PDF](pdf/2310.07710v2_dipmark.pdf) | [arXiv](https://arxiv.org/abs/2310.07710) | 上下文派生随机量与保分布采样 |
| synthid | Scalable watermarking for identifying large language model outputs；Nature 2024 | [PDF](pdf/2024_nature_synthid.pdf) | [Nature](https://www.nature.com/articles/s41586-024-08025-4) | 多层 tournament、重复上下文屏蔽 |
| synthid_supplement | 同上，Supplementary Information；56 页 | [PDF](pdf/2024_synthid_supplement.pdf) | [出版商补充材料](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-024-08025-4/MediaObjects/41586_2024_8025_MOESM1_ESM.pdf) | A 检测、G 各层级 non-distortion |
| semamark | A Robust Semantics-based Watermark for Large Language Model against Paraphrasing；NAACL Findings 2024 | [PDF](pdf/2024_naacl_semamark.pdf) | [ACL](https://aclanthology.org/2024.findings-naacl.40/) | 语义上下文编码、稳定词表分区 |
| christ | Undetectable Watermarks for Language Models；2023，arXiv v1 | [PDF](pdf/2306.09194v1_undetectable.pdf) | [arXiv](https://arxiv.org/abs/2306.09194) | PRF、密钥、重复查询与不可区分性理论 |
| unigram | Provable Robust Watermarking for AI-Generated Text；ICLR 2024 | [PDF](pdf/2306.17439_unigram.pdf) | [arXiv](https://arxiv.org/abs/2306.17439) | 固定词表分组的对照，不能一概判无效 |
| waterpool | WaterPool: A Language Model Watermark Mitigating Trade-Offs among Imperceptibility, Efficacy and Robustness；NAACL 2025 | [PDF](pdf/2025_naacl_waterpool.pdf) | [ACL](https://aclanthology.org/2025.naacl-long.209/) | 随机 key 空间与语义检索恢复 key |
| framework | A Statistical Framework of Watermarks for Large Language Models: Pivot, Detection Efficiency and Optimal Rules；AoS 2025 出版版，复用已有 | [PDF](pdf/2025_aos_statistical_framework.pdf) | [DOI](https://doi.org/10.1214/24-AOS2468) / [arXiv](https://arxiv.org/abs/2404.01245) | 枢轴、f0、异质备择、最优检测 |
| trgof | Robust detection of watermarks for large language models under human edits；JRSS-B 2025 出版版，复用已有 | [PDF](pdf/2025_jrssb_robust_edits.pdf) | [DOI](https://doi.org/10.1093/jrsssb/qkaf056) / [arXiv](https://arxiv.org/abs/2411.13868) | 编辑混合、Tr-GoF、适应未知信号与编辑水平 |
| opt | Optimal Estimation of Watermark Proportions in Hybrid AI-Human Texts；2025，arXiv v1，复用已有 | [PDF](pdf/2506.22343v1_proportions.pdf) | [arXiv](https://arxiv.org/abs/2506.22343) | 比例可识别性、初始/修正/最优估计量 |
| segment | Segmenting Watermarked Texts From Language Models；2024，arXiv v1，原有 `1.pdf` 复用 | [PDF](pdf/segment_existing.pdf) | [arXiv](https://arxiv.org/abs/2410.20670) | 随机化检验与变点分段 |
| adaptive | Adaptive Testing for Segmenting Watermarked Texts From Language Models；2025，arXiv v1 | [PDF](pdf/2511.06645_adaptive_segment.pdf) | [arXiv](https://arxiv.org/abs/2511.06645) | 自适应得分、prompt/NTP 估计失配与分段 |
| localization_v1 | Optimal Watermark Localization in Mixed-Source Large Language Model Texts；2026，已有 v1 | [已有 PDF](../docs/2608.14906v1_Optimal_Watermark_Localization.pdf) | [arXiv v1](https://arxiv.org/abs/2608.14906v1) | 早期定位理论；保留作版本比较 |
| localization_v2 | 同上；2026，已有 v2 | [已有 PDF](../docs/2608.14906v2_Optimal_Watermark_Localization.pdf) | [arXiv v2](https://arxiv.org/abs/2608.14906v2) | SPOT、自适应尾阈值与定位边界 |

目录包含针对本次问题的代表性资料，不是穷尽综述。arXiv 首发年与正式会议/期刊年份可能不同；表中按对应版本说明。
