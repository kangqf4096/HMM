# 标准密钥方案、已知模型下的备择分布：性质清单与适用边界

日期：2026-10-27。范围：固定主密钥、上下文派生每步 seed、Gumbel-max 采样，且检测端已知真实的下一词分布 \(P_t\)。本文把这类方案下“备择分布到底有哪些性质、哪些性质不能想当然”整理成一张清单；推导细节和校准失配反例见 [Gumbel 枢轴分布与模型概率](pivot_law_derivation_20261025.md)，零假设与密钥路线见 [零假设、备择分布与上下文密钥](watermark_hypotheses_and_keys_20261025.md)。

本文不是新的文献贡献，也不声称真实 LLM 已满足全部理想假设。它的用途是：在写生成方法、检测器和实验设计时，能直接查到“这个性质成不成立、依赖什么、怎么测”。

---

## 0. 方案与假设

### 0.1 标准密钥方案

词表 \(\mathcal V\)，下一词分布 \(P_t=(P_{t,w})_{w\in\mathcal V}\)。固定主密钥 \(K\)，上下文 \(c_t\)（例如最近 \(m\) 个 token 或完整前缀），每步 seed

\[
s_t=\operatorname{PRF}_K(c_t),
\qquad
U_{t,w}=G(s_t,w)\in(0,1)
\]

为伪随机向量。Gumbel-max 选择

\[
X_t=\arg\max_{w\in\mathcal V}\frac{\log U_{t,w}}{P_{t,w}},
\qquad
Y_t=U_{t,X_t}.
\]

等价写法：令 \(E_{t,w}=-\log U_{t,w}\sim\operatorname{Exp}(1)\)，\(T_{t,w}=E_{t,w}/P_{t,w}\sim\operatorname{Exp}(P_{t,w})\)，则 \(X_t=\arg\min_w T_{t,w}\)，\(Y_t=U_{t,X_t}\)。

### 0.2 零假设与备择

- \(H_{0,t}\)：当前位置没有针对密钥 \(K\) 的水印耦合，\(X_t\sim P_t\)，\(Y_t\sim\mathrm{Unif}(0,1)\)，\(f_0(y)=1\)。
- \(H_{1,t}\)：\(X_t\) 是上述 Gumbel-max 的赢家，\(Y_t=U_{t,X_t}\) 服从本文讨论的备择分布。

### 0.3 理想假设（必须显式写出）

1. 词表有限，\(P_{t,w}\ge0\)，\(\sum_w P_{t,w}=1\)；\(P_{t,w}=0\) 的坐标不参与求和。
2. 给定 \(P_t\)，\(U_{t,w}\) 独立且均匀；\(P_t\) 在 \(U_t\) 产生前确定，并与 \(U_t\) 独立。
3. 不同位置的 \(U_t\) 独立；同一上下文重复出现时是否复用随机向量，是另一个层次的问题。
4. “已知模型”指 \(P_t\) 是生成时的真实下一词分布，不含估计误差、温度/top-p 复现误差或编辑导致的前缀差异。

固定主密钥、完整 prompt、模型参数和确定性计算后，实际输出可以是完全确定的；这不与“在随机模型下有分布”矛盾，但不能把重复运行同一个 prompt 当作独立重复抽样。以下性质都是**理想独立随机向量模型下的逐位置边际性质**；联合分布、重复上下文和真实模型失配要单独检查。

---

## 1. 基本分布

### 1.1 CDF 与密度

\[
F_{1,P}(y)=\sum_{w:P_w>0}P_w\,y^{1/P_w},
\qquad
f_{1,P}(y)=\sum_{w:P_w>0}y^{1/P_w-1},
\qquad 0<y<1.
\]

这是统计框架论文 Lemma 3.1 的 Beta 混合形式；推导：\(T=\min_w T_w\sim\operatorname{Exp}(1)\)，赢家 \(W\sim P\) 与 \(T\) 独立，\(Y=U_W\)，故 \(Y\mid W=w\sim\operatorname{Beta}(1/P_w,1)\)。

### 1.2 形状

- 支撑为 \((0,1)\)；连续（除 \(P_w=1\) 的退化端点）。
- \(f_{1,P}(y)\) 随 \(y\) **非递减**：每个指数 \(1/P_w-1\ge0\)。
- \(F_{1,P}(y)\) 是凸函数：\(F''_{1,P}(y)=\sum_w (1/P_w-1)y^{1/P_w-2}\ge0\)。
- \(F_{1,P}(0)=0\)，\(F_{1,P}(1)=1\)。
- \(f_{1,P}(1)=|\{w:P_w>0\}|\)；若存在 \(P_w=1\)，则 \(f_{1,P}(0)=1\)，否则 \(f_{1,P}(0)=0\)。
- 与 \(f_0(y)=1\) 相比，\(f_{1,P}\) 在 \(y\) 大的一侧更重；但两者支撑相同，单点仍可能不可区分。

### 1.3 退化与弱信号

- \(P=(1,0,\dots,0)\)：\(F_{1,P}(y)=y\)，\(f_{1,P}(y)=1\)，与 null 完全相同；没有可辨识信号。
- \(P\) 接近点质量：\(S_2\approx1\)，信号很弱；比例估计与“没有水印”容易混淆。
- \(P\) 在 \(K\) 个 token 上均匀：\(Y\sim\operatorname{Beta}(K,1)\)，信号随 \(K\) 增大而增强（见 2.4）。

---

## 2. 矩、集中度、熵

### 2.1 全部正整数阶矩

令 \(X=-\log Y\)。对整数 \(k\ge1\)，

\[
\boxed{\mathbb E[X^k\mid P]=k!\sum_w P_w^{k+1}=k!\,S_{k+1}(P)},
\]

其中 \(S_r(P)=\sum_w P_w^r\)。因此

\[
\mathbb E[X\mid P]=S_2(P),
\qquad
\operatorname{Var}(X\mid P)=2S_3(P)-S_2(P)^2.
\]

零假设下 \(X\sim\operatorname{Exp}(1)\)，\(\mathbb E_0[X]=1\)。所以均值间隔为

\[
1-S_2(P).
\]

\(S_2\) 是“两次独立抽样选中同一 token”的概率（碰撞概率），与二阶 Rényi 熵 \(H_2(P)=-\log S_2(P)\) 对应，有效词表大小约为 \(1/S_2(P)\)。

另一个可用恒等式：对 \(a\ge0\)，

\[
\mathbb E[Y^a\mid P]=\sum_w\frac{P_w}{1+aP_w}.
\]

### 2.2 同一 \(S_2\) 不够决定整个备择

\(S_2\) 只给出 \(X\) 的均值。反例：

\[
P=\left(\frac12,\frac14,\frac14\right),\qquad
Q=\left(\frac5{12},\frac5{12},\frac16\right),
\]

两者 \(S_2=3/8\)，但 \(S_3\) 分别为 \(5/32\) 与 \(43/288\)，故 \(\operatorname{Var}(X)\) 分别为 \(11/64\) 与 \(91/576\)，CDF 也不同。因此：

- 测 \(S_2\) 适合解释“平均信号强弱”；
- 要做似然比、尾部阈值或 HMM 发射，至少还要 \(S_3\)，或者直接使用 \(P_t\)；
- 不能把“按 \(S_2\) 分箱的经验分布”说成“条件分布”，它仍是箱内的加权平均。

### 2.3 熵与 KL 的解析关系

以 \(f_0=1\) 为参照，

\[
D_{\mathrm{KL}}(f_{1,P}\Vert f_0)
=\int_0^1 f_{1,P}(y)\log f_{1,P}(y)\,dy.
\]

有

\[
\boxed{
H(P)-1+S_2(P)
=D_{\mathrm{KL}}(f_{1,P}\Vert f_0)+I(W;Y\mid P)
}
\]

从而

\[
0\le D_{\mathrm{KL}}(f_{1,P}\Vert f_0)
\le H(P)-1+S_2(P).
\]

若 \(P\) 在 \(K\) 个 token 上均匀，则 \(I(W;Y\mid P)=0\)，等号成立：

\[
D_{\mathrm{KL}}(f_{1,P}\Vert f_0)=\log K-1+\frac1K.
\]

注意：

- 这是“枢轴分布与均匀 null 的 KL”，不是 \(D_{\mathrm{KL}}(P\Vert\mathrm{Unif})\)；
- \(H(P)\) 单独不能恢复完整 \(f_1\)；它只给上界；
- 上界对集中分布可能很松（例如 \(P=(0.99,0.01)\) 时上界约 0.0362，实际约 0.0022）。

### 2.4 “更分散则更强”的严格版本

固定 \(0<y<1\)，令 \(a=-\log y\)，\(\phi(p)=p e^{-a/p}\)。则

\[
\phi''(p)=a^2p^{-3}e^{-a/p}\ge0,
\]

所以 \(F_{1,P}(y)=\sum_w\phi(P_w)\) 是 Schur 凸函数。若 \(P\) 在主序意义下比 \(Q\) 更集中，则

\[
F_{1,P}(y)\ge F_{1,Q}(y)\quad\text{对所有 }y,
\]

即更分散的 \(P\) 使 \(Y\) 一阶随机占优地更靠近 1，信号更强。

限定：

- 只对可按主序比较的 \(P,Q\) 成立；
- 不能仅凭两个分布的 Shannon 熵大小就断言所有尾部、所有检测指标同序；
- “更分散”不等于“熵更大”；主序是更强的条件。

### 2.5 单调似然比

因为 \(f_{1,P}(y)\) 非递减、\(f_0(y)=1\)，所以

\[
\frac{f_{1,P}(y)}{f_0(y)}=f_{1,P}(y)
\]

非递减。对单点检验，Neyman–Pearson 最优拒绝域是“\(Y\) 大于某个阈值”；不同 \(P\) 下阈值和证据强度不同，但排序方向一致。这个性质支持“按 \(Y\) 从大到小看”的直觉，但不保证混合分布或 HMM 联合定位的排序不变。

---

## 3. 尾部、形状与数值敏感性

### 3.1 \(y\to0\)

由最小 \(P_w\) 主导：

\[
f_{1,P}(y)\sim y^{1/P_{\min}-1},
\qquad
F_{1,P}(y)\sim P_{\min}\,y^{1/P_{\min}}.
\]

若 \(P_{\min}\) 很小，指数很大，密度在 0 附近极平；若 \(P_{\min}\) 接近 1，则接近均匀。

### 3.2 \(y\to1\)

\[
f_{1,P}(1)=|\{w:P_w>0\}|,
\qquad
F_{1,P}(1-\varepsilon)
\approx 1-|\{w:P_w>0\}|\,\varepsilon
+\frac{\varepsilon^2}{2}\sum_w\left(\frac1{P_w}-1\right).
\]

靠近 1 的行为由整个概率谱决定，不只由 \(S_2\)。

### 3.3 CDF 稳定不等于密度比稳定

对共同词表上的 \(P,Q\)，

\[
\sup_{0\le y\le1}|F_{1,P}(y)-F_{1,Q}(y)|
\le \operatorname{TV}(P,Q)
=\frac12\sum_w|P_w-Q_w|.
\]

证明思路：\(F_{1,P}=\sum_w\phi(P_w)\)，\(\phi'(p)=e^{-a/p}(1+a/p)\in[0,1]\)，沿 \(P,Q\) 之间的线段积分，并用 \(\sum_w(P_w-Q_w)=0\) 把常数改进到 \(\mathrm{TV}\)。

限定：

- 这只约束 CDF，不约束逐点密度、密度比或 log 密度比；
- 尾部、小概率区、接近 0 的密度估计可能更敏感；
- 少量文本编辑未必只轻微改变 \(P\)，需实测 \(P_t^{\mathrm{gen}}\) 与 \(P_t^{\mathrm{eval}}\) 的 TV。

---

## 4. 条件于被选中 token 的细化

除了边际 \(f_{1,P}(y)\)，还有条件模型

\[
Y\mid W=w,P\sim\operatorname{Beta}(1/P_w,1),
\qquad
f_1(y\mid W=w,P)=\frac1{P_w}y^{1/P_w-1}.
\]

对应的 log 得分是

\[
\ell(y,w,P)=-\log P_w+\left(\frac1{P_w}-1\right)\log y.
\]

要点：

- 这是“已知被选中 token \(w\)”后的条件密度，比边际 \(f_{1,P}\) 更细；
- 它**不是**边际密度，也不能直接替代 HMM 的 \(f_1(y)\)；
- 若要做完整联合检验，还需要 null 下的 token 模型；不能把 \(P\) 同时当作人类文本的 token 分布；
- 在检测端 \(w\) 是观测到的，所以这个条件得分可用；但它对 \(P_w\) 的估计误差、对 \(y\) 的尾部都很敏感。

---

## 5. 多位置、混合与 HMM

### 5.1 独立性与联合分布

理想模型下，若不同位置的 \(U_t\) 独立，则 \(Y_t\) 逐位置独立，联合密度是乘积。实际协议中：

- 固定主密钥 + 最近 \(m\) 个 token 哈希：相同上下文会复用同一随机向量；
- 上下文重叠、重复短语、编辑后的前缀重合，都会造成 \(Y_t\) 之间的依赖；
- 因此 HMM 的乘积似然是**工作近似**，不是精确联合密度；null 文档的整篇误报必须按文档校准，不能把 token 当独立重复。

### 5.2 条件混合

设 \(C\) 为上下文/生成条件，\(Q_1\) 为水印位置的条件构成，则平均备择为

\[
\bar f_1(y)=\int f_{1,P(c)}(y)\,dQ_1(c).
\]

性质：

- 每个 \(f_{1,P}\) 非递减，混合后仍非递减；似然比排序方向一致。
- 但不同条件的证据强度、阈值、后验数值不同；混合分布不等于任一条件分布。
- KL 混合分解：

\[
\mathbb E_C D_{\mathrm{KL}}(f_{1,P(C)}\Vert f_0)
-D_{\mathrm{KL}}(\bar f_1\Vert f_0)
=\mathbb E_C D_{\mathrm{KL}}(f_{1,P(C)}\Vert\bar f_1)
=I(C;Y\mid Z=1).
\]

右边是“观察 \(C\) 相比只观察 \(Y\) 多出来的信息”。异质性本身不会让边缘方法失效；它让边缘方法损失信息，而 **\(Q_1^{\mathrm{cal}}\ne Q_1^{\mathrm{test}}\)** 会另外造成失配。

### 5.3 校准迁移与比例混淆

若校准样本来自 \(Q_1^{\mathrm{cal}}\)，测试来自 \(Q_1^{\mathrm{test}}\)，则

\[
\bar f_1^{\mathrm{cal}}\ne\bar f_1^{\mathrm{test}}
\]

即使校准样本无限大也不解决。一个边界例子：校准 \(f_{1,\mathrm{strong}}=2y\)；测试中所有位置都启用水印，但只有比例 \(a\) 的位置使用强条件 \(P=(1/2,1/2)\)，其余为近确定性的弱条件 \(P=(0.999,0.001)\)。测试总体近似为

\[
f_{\mathrm{test}}(y)=a\,2y+(1-a)\cdot1.
\]

用 \((1-\varepsilon)+\varepsilon\cdot2y\) 拟合，会得到 \(\hat\varepsilon\approx a\)，而实际启用水印比例是 1。总体分箱 MLE 在 \(a=0.5\) 时约为 0.5019；这是分布失配，不是样本量问题。

对只用 \(X=-\log Y\) 均值的比例估计，偏差可直接写出：

\[
\mathbb E[X]=1-e+e\,s_{\mathrm{test}},
\qquad
\hat e_{\mathrm{moment}}
=e\frac{1-s_{\mathrm{test}}}{1-s_{\mathrm{cal}}}.
\]

当前仓库用复合似然而非这个均值估计器；这个式子只用来说明“比例”和“信号强弱”的混淆机制。

### 5.4 HMM 还需要序列信息

即使整体边缘完全匹配，HMM 仍可能把“持续的低信息水印段”解释成 null 段：它需要的是给定状态后的序列联合结构，不是一条全局边缘曲线。若高/低信息条件成段持续，或与 \(Z\) 的转移相关，应考虑：

- 按 \(S_2/S_3\) 分层的条件发射；
- 依赖已观测历史的混合密度；
- 显式的强弱状态/半马尔可夫 duration；
- 或在不可辨识时 abstain。

---

## 6. 零分布与检验

### 6.1 零分布

理想模型下，\(H_0\) 时 \(X_t\sim P_t\) 且 \(U_t\) 与 \(X_t\) 独立，故

\[
Y_t\sim\mathrm{Unif}(0,1),
\qquad f_0(y)=1.
\]

注意：

- 这是逐位置边际；固定 key/上下文复用时，边际仍可 Uniform，但联合不独立；
- 编辑后若 key 不再匹配，\(H_0\) 仍可能成立；检测端不要用“key 对不上”直接判有水印；
- null 文档的 any-FP 要按文档校准，不能把 token 当独立重复。

### 6.2 最优检验与功效

已知 \(P_t\) 和备择 \(f_{1,P_t}\) 时，单点 Neyman–Pearson 最优统计量是似然比

\[
\Lambda_t=\frac{f_{1,P_t}(Y_t)}{f_0(Y_t)}=f_{1,P_t}(Y_t).
\]

因为 \(f_{1,P_t}\) 非递减，最优拒绝域是 \(Y_t\ge c\)。功效由

\[
D_{\mathrm{KL}}(f_{1,P_t}\Vert f_0)
\]

控制；多个位置时用 \(\sum_t\log f_{1,P_t}(Y_t)\)（独立近似）或 HMM 的联合似然。注意：

- 这是“已知 \(P_t\)”的 oracle 功效，不是实际检测器可达的功效；
- 实际检测器若用 \(P_t^{\mathrm{eval}}\ne P_t^{\mathrm{gen}}\)，功效会下降；
- 若只用经验 \(\bar f_1\)，功效取决于 \(Q_1^{\mathrm{cal}}\) 与 \(Q_1^{\mathrm{test}}\) 的匹配程度。

### 6.3 PIT 是诊断，不是 null p 值

把 \(Y_t\) 代入 \(F_{1,P_t}\) 得到的 \(F_{1,P_t}(Y_t)\) 在 \(H_1\) 下应近似 Uniform；在 \(H_0\) 下 \(F_{0}(Y_t)=Y_t\) 才是 Uniform。不要把“水印模型 PIT 的 KS 距离”当成 null p 值；它检验的是“观测是否像水印模型”，不是“是否有水印”。

---

## 7. 常见误读

1. **“备择分布就是 Uniform 的某种平移”**：不是；它是 \(P\) 的整个概率谱决定的 Beta 混合。
2. **“熵越大信号越强”**：只在主序意义下严格；Shannon 熵单独不够，且同一 \(S_2\) 可有不同 \(S_3\)。
3. **“\(S_2\) 可以决定完整 \(f_1\)”**：不能；\(S_2\) 只给均值，\(S_3\) 给方差，完整 CDF 需要整个谱或 \(P_t\)。
4. **“CDF 接近，密度比也接近”**：不保证；尾部/小概率区可能很敏感。
5. **“条件于 token 的密度就是边际密度”**：不是；前者是 Beta 条件模型，后者是混合。
6. **“固定 key 下重复运行是独立抽样”**：不是；同一 prompt/上下文可能完全确定或复用随机向量。
7. **“经验 \(f_1\) 可以跨语料/跨编辑迁移”**：只在同一总体、同一 \(Q_1\) 下成立；否则是指标偏差，不是方差。
8. **“HMM 边缘匹配就够了”**：不够；持续低信息段、转移相关、编辑选择都可能让联合后验错。
9. **“弱信号段也是水印段，所以 HMM 应该能定位”**：弱到不可辨识时，任何检测器都没有信息；应 abstain，而不是靠 \(\rho\) 造证据。
10. **“Gumbel-max 一定无损”**：理想独立 Uniform 下保边际；PRF 有偏、top-p 截断、数值实现错误都会破坏。

---

## 8. 可测量清单

| 量 | 公式/定义 | 说明 | 怎么测 |
|---|---|---|---|
| \(S_2\) | \(\sum_w P_w^2\) | 碰撞概率；\(E[X]\) | 生成时记录 \(P_t\) 或代理模型 |
| \(S_3\) | \(\sum_w P_w^3\) | 方差 \(2S_3-S_2^2\) | 同上 |
| \(H(P)\) | \(-\sum_w P_w\log P_w\) | KL 上界的一部分 | 同上 |
| \(p_{\max}\) | \(\max_w P_w\) | 集中度/弱信号代理 | 同上 |
| \(F_{1,P}(y)\) | \(\sum_w P_w y^{1/P_w}\) | 备择 CDF | 解析/网格 |
| \(f_{1,P}(y)\) | \(\sum_w y^{1/P_w-1}\) | 备择密度 | 解析/网格 |
| \(E[X]\) | \(S_2\) | 均值诊断 | 样本均值 vs 理论 |
| \(\operatorname{Var}(X)\) | \(2S_3-S_2^2\) | 方差诊断 | 样本方差 vs 理论 |
| \(D_{\mathrm{KL}}\) | \(\int f_1\log f_1\) | 单点功效上界 | 数值积分/解析 |
| \(\mathrm{TV}(P,Q)\) | \(\frac12\sum|P-Q|\) | CDF 稳定界 | 生成 vs 重算 |
| \(\sup_y|F_P-F_Q|\) | CDF 差 | 备择稳定性 | 网格 |
| PIT KS | \(F_{1,P_t}(Y_t)\) 的均匀性 | 水印模型诊断 | 留出文章 |
| \(S_2\) 分层 PIT | 按 \(S_2\) 箱 | 检查条件失配 | 分箱 |
| key BER/GAR | 复述后 key 是否复现 | key 稳定性 | 复述/翻译 |
| any-FP | null 文档至少一个输出 | 误报控制 | 独立 null 文档 |

---

## 9. 与现有文档的关系

- [pivot_law_derivation_20261025.md](pivot_law_derivation_20261025.md) 给出本文大部分公式的推导、CDF 稳定界、校准失配反例和数值核对；本文是“性质清单”版本，便于写方法/实验时查阅。
- [watermark_hypotheses_and_keys_20261025.md](watermark_hypotheses_and_keys_20261025.md) 给出 H0/H1、密钥路线和当前实现边界。
- [pivot_experiment_20261025.md](pivot_experiment_20261025.md) 给出按 \(S_2\) 分层、解析 \(P_t\)、全局经验三种发射的实验计划。
- 项目主文档 [问题设置.docx](../../问题设置.docx) 和 [experiment_plan_v1.md](../../experiment_plan_v1.md) 目前仍以“全局经验 \(f_1\)”为主，尚未系统写入本文的 \(S_2/S_3\) 性质；若这条线成为主线，应把本文压缩成一节并入主计划。

## 参考

- Li, X., Ruan, F., Wang, H., Long, Q., Su, W. J. (2025). A Statistical Framework of Watermarks for Large Language Models: Pivot, Detection Efficiency and Optimal Rules. *Annals of Statistics* 53(1), 322–351. 本地 PDF：`../papers/pdf/2025_aos_statistical_framework.pdf`。
- Li, X., Liu, X., Li, G. (2025). Optimal Estimation of Watermark Proportions. 本地 PDF：`../papers/pdf/2506.22343v1_proportions.pdf`。
- SPOT v2: Optimal Watermark Localization in Mixed-Source Large Language Model Texts. 本地 PDF：`../papers/pdf/2608.14906v2_Optimal_Watermark_Localization.pdf`。
- 项目内推导与数值核对：[pivot_law_derivation_20261025.md](pivot_law_derivation_20261025.md)、[results.json](pivot_law_20261025/results.json)。
