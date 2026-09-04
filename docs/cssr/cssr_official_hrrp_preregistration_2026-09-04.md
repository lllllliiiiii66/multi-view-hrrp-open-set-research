# CSSR 机制解剖与官方语义 pCSSR-HRRP 基线预注册

> 日期：2026-09-04
>
> 阶段：P3 独立机制审计与官方语义基线验证
>
> 实验：official_cssr_hrrp_pilot_v1
>
> 分支：codex/cssr-mechanism-official-baseline
>
> 起点提交：105c313f436f20e57c6157e08e0afd737556302e
>
> 状态：仅预注册；尚未生成本实验的阶段 A 结果、smoke 或 pilot 结果

## 1. 研究问题与停止边界

本任务回答两个相互独立的问题。

### 1.1 阶段 A：现有 D1/D2 的身份依赖机制

只使用已经封存的 D0/D1/D2 checkpoint、manifest、特征和逐样本结果，解释：

1. D2 为什么能改善 N1/N4 和 DDG 双向吸收；
2. D2 为什么使 N2 的 MARVEL CRANE 明显退化；
3. 失败主要来自 raw reconstruction、activation scale、known reference width，还是两视角均值；
4. adapter 是否进一步压低了 R2 表示的有效维度；
5. 官方 S2/S3 是否能缓解旧 D1/D2 只依赖重构分数时的身份不稳定。

阶段 A 不重新训练、不选模、不修改旧产物，也不进入阶段 B 的模型、参数、分数或结果标签。

### 1.2 阶段 B：官方语义 pCSSR 是否适配当前 HRRP

实现一维 HRRP 版本 OFFICIAL_SEMANTICS_PCSSR_1D，尽量保持固定官方源码的模型、损失与评分语义，并与匹配的 linear control 及冻结 R2 强基线比较。

本任务仅为单 fold、单 seed、三个 surrogate identity pair 的 pilot。无论结果标签为何：

- confirmation_allowed = false；
- automatic_followon_authorized = false；
- final_unknown_test_authorized = false。

任务在 pilot 聚合、审计、结果报告和建议形成后停止。不得自动进入 confirmation、最终 unknown、偶数角 test 或另一个方法实验。

## 2. 既有结论保护

本任务不得改写以下已经封口的结论。

1. ms_mean_head_factorial_surrogate_v1 只证明多尺度 backbone 在 source-known surrogate 协议下具有稳定收益；R2_MS_MEAN_CE 是当前强基线，但尚未通过最终 7-known/3-unknown 与偶数角 test。
2. fg_mv_cssr_frozen_r2_v1 的正式标签仍是 no_cssr_signal。它检验的是冻结 R2 上的 PCSSR_CORE_1D 与旧 B0–B4 分数，不是本任务包含 S1/S2/S3 的完整官方语义基线。
3. fg_mv_cssr_e2e_redesign_v2 继续保持：
   - pilot_status = hard_failed_incomplete；
   - pilot_gate = not_evaluated；
   - selected_method = null。
4. 旧 N4-Q2 的 100 倍规则确实由加权项 0.5 L_rel 触发，但历史运行没有保存触发 epoch 和原始梯度。后续 5-epoch 审计没有复现该现象，因此真实原因仍无法判断；不得称为误报。
5. fg_mv_cssr_decoupled_audit_v3 的标签继续是 decoupled_cssr_failed。D2 改善 DDG 双向吸收，但使 N2 MARVEL CRANE 的 AUROC 相对 D0 下降 45.56 个百分点；confirmation 未运行。
6. 新任务是独立官方语义基线，不是对旧 D2 的补完、修复、恢复或翻案，也不能外推为所有类别重构方法的结论。

RESEARCH_CONTEXT.md 本任务不得修改。任何可能的主上下文更新只能在任务结束后作为提案交给用户。

## 3. 预注册时间顺序与不可变性

执行顺序固定为：

1. 从起点提交创建独立分支；
2. 在产生任何阶段 A 数值结果前，先单独提交本预注册；
3. 实现并冻结官方 oracle、完整配置、阶段 A、阶段 B、结果标签和审计代码，在首次运行前提交；
4. 提交产生结果的完整代码，并记录提交 SHA-256；
5. 运行本地完整测试、GPU 专项测试、Python compile、配置校验和 git diff --check；
6. 运行阶段 A 只读审计；
7. 运行 N1 smoke；
8. smoke 全部通过后，运行 12 项 pilot；
9. 聚合、逐样本反算、checkpoint 重放和全量哈希；
10. 写入两个结果报告并停止。

阶段 B 的实现、配置、评分和 gate 必须在阶段 A 首次运行前提交并哈希绑定。阶段 A 的任何发现均不得修改阶段 B。

如果代码在产生任何性能后发生变化，旧结果不得与新代码混用；修复只能新建产物目录并重新完成相应前置审计。

## 4. 固定数据与角色

统一固定：

| 项目 | 值 |
|---|---|
| angle_fold | 0 |
| R2_seed | 20260830 |
| official_cssr_seed | 20260906 |
| pilot_pairs | N1、N4、N2 |
| development angles | 奇数角 |
| final even-angle test | 不生成、不读取 |
| final 3 unknown | 不生成、不读取 |

三个 pair 继续使用既有、封存的类别定义和 manifest：

| Pair | Surrogate identities |
|---|---|
| N1 | DDG-112、迷你好望角型散货船 |
| N4 | DDG-1000、集装箱船达飞罗尔多夫级 |
| N2 | 油气轮 MARVEL CRANE、迷你好望角型散货船 |

每个 pair 有五个 train-known 类：

- 每类 144 个唯一 train-known 底层样本，共 720 个；
- 每类 36 个唯一 known-calibration 底层样本，共 180 个；
- 两个 surrogate identity 各 36 个唯一底层样本，共 72 个；
- 正式 pair 评价复用既有每类或每 surrogate identity 500 对的 manifest 和顺序。

任何底层 sample ID 及其派生版本不得跨 train、known calibration 或 surrogate role。训练和训练统计不得使用 known calibration 或 surrogate unknown。

## 5. 产物与源码身份

### 5.1 封存 D0/D1/D2 来源

阶段 A 只读取起点提交对应的完整远端产物：

- 实验 fg_mv_cssr_decoupled_audit_v3；
- phase stage_b_pilot；
- pair N1、N4、N2；
- 方法 D0、D1、D2。

必须先核对：

- phase success、artifact manifest 和逐文件 SHA-256；
- checkpoint bitwise SHA-256；
- pair manifest、unique-base manifest 与标签顺序；
- R2 checkpoint、logits、预测和源码身份；
- D1/D2 adapter、AE、参考分布和数值设置。

如果缺少中间量，只允许从封存 checkpoint 和原 manifest 只读重算。重算前必须验证全部模型处于 eval、数值设置一致，且所有已有输出逐元素一致。不得覆盖旧文件。

### 5.2 官方源码 oracle

官方参考固定为：

- 论文：Class-Specific Semantic Reconstruction for Open Set Recognition；
- arXiv：2207.02158；
- 仓库：xyzedd/CSSR；
- commit：d5a99e91f310ec274c7bfe5796fb270719a07ab3。

官方源码不提交到本研究仓库。oracle 根目录是运行时必填的环境依赖绝对路径；本机当前参考 checkout 为 /private/tmp/cssr-official-d5a99e91，但该绝对路径不作为跨主机协议身份。

每次测试必须同时验证 git commit 和以下文件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| methods/cssr.py | 0d23558c6a3cc4bf068036502a8ab43ee6278aecd91d96741f7375a142d9c5a3 |
| methods/cssr_ft.py | 31244f194d91f6cab0bdf34eb14a0ed3b58f25b6c49a44042bb96baa9977fb16 |
| configs/basic.json | 672375c6838004ae604509ba57098c7fefd17b6ac0f38e7c955fc8c09ba3192a |
| configs/pcssr.json | 353b0768cc6ee60ac76c110a22da8bdb5c15179260d4abeb2f43fee422d24c6b |
| configs/pcssr/cifar10.json | ce5c7187cab1d8a7387526e459dc21c257f407e15e2304a91f618a8d8d34b0ab |
| configs/pcssr/imagenet.json | 170b8b7f86a2bde8fd409feaa96edfbfbd4226cc7ed9d1a564db8ca8a783b505 |

路径、commit 或任一文件哈希不符时，官方差分测试硬失败。

## 6. 阶段 A：逐样本机制解剖

### 6.1 统计单位

底层样本统计一律按 sample ID 去重，每个唯一 base 只计一次。pair-level 统计只在明确标记的两视角分析中使用，不得用 pair multiplicity 重复加权底层样本。

对 D1、D2 的每个唯一 train-known、known-calibration 与 surrogate 样本、每个视角 \(v\)、每个类别 AE \(k\)，保存：

\[
E_{i,v,k}=\operatorname{mean}_{c,t}|U_{i,v}-A_k(U_{i,v})|,
\]

\[
M_{i,v}=\operatorname{mean}_{c,t}|U_{i,v}|,
\]

\[
r_{i,v,k}=\frac{E_{i,v,k}}{M_{i,v}+10^{-8}}.
\]

类别参考分布完全复用封存 D1/D2 的定义：

- 每类由其 36 个唯一 true-class known-calibration base 的对应类别 \(r_k\) 构成；
- known-calibration 查询按 sample ID leave-one-base-out；
- train-known 与 surrogate 查询不 leave-one-out；
- \(p=(1+\#\{r_{\mathrm{ref}}\ge r_{\mathrm{query}}\})/(n_{\mathrm{ref}}+1)\)；
- ties 使用大于等于；
- \(a=-\log(p+10^{-8})\)。

保存全部类别的 E、M、r、p、a，而不是只保存 predicted-class 的最终分数。

### 6.2 两视角分解

同时保存：

- 冻结 R2 pair 融合预测 \(\hat y_{\mathrm{R2}}\)；
- 每个视角、每个类别的 E/M/r/p/a；
- predicted-class E/M/r/p/a；
- 两视角均值、差、最小值和最大值；
- 每个视角的最低重构类别 \(\arg\min_k r_{i,v,k}\)；
- 两个视角最低重构类别是否相同；
- D0 类别条件 MLS。

predicted-class known p95 固定为对应类别 36 个原始参考 \(r_k\) 的 NumPy linear quantile：

\[
q_{0.95}=\operatorname{quantile}(r_{\mathrm{ref},k},0.95,\text{method=linear}).
\]

视角接受固定为 \(r_{i,v,\hat y_{\mathrm{R2}}}\le q_{0.95}\)。报告两个都接受、只接受一个、两个都拒绝的比例。

Pearson 与 Spearman 分别对两个视角的 predicted-class E、M、r、a 报告；主要 view-aggregation 讨论以 a 为准。AUROC 分别报告 view1、view2、两视角算术均值，不以这些结果选择新的聚合。

### 6.3 身份诊断

重点报告：

- N2 / MARVEL CRANE；
- N2 / 迷你好望角型散货船；
- N1 / DDG-112；
- N4 / DDG-1000。

每个身份报告 raw E、M、r、p、a，两视角 AUROC、均值 AUROC、相关性、接受组合比例，以及 predicted-class reference 的 count、mean、population std、median、IQR、p90、p95、p99。

IQR 固定为 NumPy linear quantile 的 q75−q25。所有 std 使用 population std，即 ddof=0。

以下解释不是互斥标签，也不设事后性能门槛：

- raw-error overlap：未知 E 与 predicted-class known E 重叠；
- activation-scale effect：E 不低但较大 M 使 r 明显降低；
- broad-reference effect：predicted-class reference 的分散程度和高分位较宽；
- view-aggregation effect：单视角异常证据在算术均值后明显减弱。

只报告分布、反事实视角结果和证据强弱，不强行输出唯一原因。

### 6.4 AE 交叉重构与过泛化

D1、D2 分别构造 E、r、a 的 identity-by-AE 矩阵。行覆盖五个 train-known identity、五个 known-calibration identity 和两个 surrogate identity；列为五个类别 AE。

最低重构 AE 固定为 \(\arg\min_k r_k\)。一个 AE “吞噬”某身份仅表示它在该身份中取得最高的最低重构占比，不等同于因果结论。

每个 AE 保存：

- own_known_median_r；
- other_known_median_r；
- surrogate_median_r；
- specificity_ratio = other_known_median_r / (own_known_median_r + 1e-12)；
- open_acceptance_rate = surrogate r 低于或等于 own-known p95 的比例；
- best_ae_share_across_all_nonown_identities。

own-known reference 固定使用该类 36 个 known-calibration unique bases；other-known 和 surrogate 先在每个 identity 内计算，再对 identity 等权平均，避免身份样本量造成权重偏差。best-AE share 也先按 identity 计算再等权平均。

### 6.5 Z 与 U 的表示几何

几何统计分别在以下互不混合的等样本 identity 池计算：

1. train-known：5 类 × 144 unique bases；
2. known calibration：5 类 × 36 unique bases；
3. surrogate：2 类 × 36 unique bases；
4. evaluation：known calibration 与 surrogate 合并，7 类 × 36 unique bases。

不得按 pair multiplicity 加权。对特征 \(T\in\mathbb R^{N\times C\times L}\)，构造位置观测矩阵：

\[
V=\operatorname{reshape}(\operatorname{transpose}(T),[N L,C]).
\]

使用 population channel mean 对 V 中心化。奇异值 \(s_j\) 的解释率为：

\[
\pi_j=s_j^2/\sum_l s_l^2.
\]

entropy effective rank 固定为：

\[
\operatorname{erank}=\exp\left(-\sum_{\pi_j>0}\pi_j\log\pi_j\right).
\]

另保存：

- 每个 base 的 Frobenius norm 的算术均值；
- 每通道 population variance；
- 前 10 个奇异值及累计解释率；
- 每个 base 的 \(\|U-Z\|_F/(\|Z\|_F+10^{-12})\)，以及 identity 等权均值；
- 每类中心为该类全部 base-position 行的通道均值；
- within scatter 为各类位置到本类中心的平均平方欧氏距离，再对类别等权平均；
- between distance 为所有不同 known 类中心的无序两两平方欧氏距离均值；
- Fisher ratio = between / (within + 1e-12)；
- known-calibration 最近中心分类使用 train-known 中心和每个 base 对中心的平均位置平方距离；
- surrogate 到各 train-known 中心的平均位置平方距离。

主报告必须分别给出 Z、D1-U、D2-U；不能只依据 U 的绝对 effective rank 判断 adapter 导致压缩。

### 6.6 官方 S1/S2/S3 对旧 D1/D2 的 post-hoc 诊断

该部分只用于机制解释，performance_gate_eligible=false。

对 D1、D2 分别使用其自身 pCSSR 单视角概率预测，对 raw train-known 单视角样本按该预测类别分组建立模板。测试 pair 的共同预测类别由两个视角的 pCSSR 概率算术均值后 argmax 得到；两个视角都按这一共同类别打分。

S1 严格使用官方配置表达式 R[0]/R[1]/R[1] 的数组语义：

- R[0] 是预测类别的 clipped reconstruction logit；
- R[1] 是逐位置的通道绝对激活均值；
- 先逐位置相除，再对位置平均；
- 不增加 paper-aligned 改写或额外权重。

S2 严格按固定官方 commit：

- 建模板时使用 abs(feature)；
- 对每个预测类别计算通道均值；
- 跨类别逐通道除以该通道类别和；
- 测试打分使用 raw signed feature；
- 官方源码中被注释掉的测试时 abs 不启用。

S3 严格使用官方 G_p_pro，p=8；模板从 raw train-known、按自身预测类别建立；测试计算共同预测类 Gram 模板与样本 Gram 的 elementwise product sum。

模板构建后，每个唯一 train-known HRRP 产生四个确定性增强版本。增强施加在该封存单元自己的 global-scalar-z-score HRRP 输入上，然后重新前向得到 U；不得直接扰动已保存 U。每个版本只含：

- gain 服从 Uniform[0.9,1.1]；
- additive Gaussian noise 的 std 为 0.02。

不允许 shift、crop、reverse、角度扰动或 surrogate unknown。阶段 A 增强使用第 8.3 节的 SHA-256 派生规则，material 固定为
`cssr_identity_mechanism_score_norm_v1|20260906|pair_id|fold_0|sample_id|variant|purpose`；`variant`
为 1–4，`purpose` 只能是 `gain` 或 `noise`。material 不含 D1/D2 方法 ID，因此同一 pair 的 D1/D2 共享完全相同的输入增强。

S1、S2、S3 的标准化统计使用全部四倍 augmented train-known 分数、float64 accumulation、population mean/std。要求 std 有限且大于 1e-12；标准化严格为：

\[
\widetilde S_j=(S_j-\mu_j)/(\sigma_j+10^{-8}).
\]

full score 固定为 \(\widetilde S_1+\widetilde S_2+\widetilde S_3\)，权重均为 1。报告 S1、S2、S3 和 full 的 AUROC、OSCR、FPR95，不修改旧 D1/D2，也不进入阶段 B gate。

若某预测类别没有 raw train-known 样本，或统计非有限，该模型的阶段 A post-hoc 诊断标为 failed；不得用补零模板伪造结果。阶段 A 其他已完成分析与阶段 B 仍按各自独立前提执行。

## 7. 阶段 B：模型与官方语义

### 7.1 共同 backbone

O1–O4 都从对应 pair 的同一个已审计 R2 epoch-100 checkpoint strict-load：

- stem；
- stage1；
- stage2；
- stage3。

输出特征图固定为 \(Z\in\mathbb R^{128\times76}\)。不加载或使用原 R2 的 global average/max projection、128 维融合 feature 或线性 CE head。

同一 pair 的 O1–O4 encoder 初始 state，包括参数和 BN buffers，必须 bitwise 完全一致。

### 7.2 pCSSR head

O2/O4 使用：

- 五个独立类别 AE；
- 每个 AE 为 Conv1d(128,64,kernel_size=1,bias=false)、Tanh、Conv1d(64,128,kernel_size=1,bias=false)；
- L1 reconstruction error；
- reconstruction logit 为 \(-0.1\sum_c|A_k(z)-z|\)；
- clip 到 [-100,100]；
- 对每个位置在五类上 SoftMax；
- 对位置概率做算术平均；
- 分类损失为正确类平均概率的负对数；
- 不并行保留 CE head。

### 7.3 Matched linear control 的证据名称

O1/O3 使用固定官方 LinearClassifier 原语：

- Conv1d(128,5,kernel_size=1,bias=false)；
- 输出乘 gamma=0.1；
- 逐位置 SoftMax 后对位置平均；
- 使用同一个概率分类损失。

这里的 gamma=0.1 与 softmax_avg 是本交接任务冻结的 matched-control 配置。固定官方 linear 配置实际使用 gamma=1 与 avg_softmax，因此 O1/O3 只能称：

> 使用官方 LinearClassifier 原语、与本任务 pCSSR 匹配的 linear control。

不得称 O1/O3 为官方 linear.json、官方 linear 配置或官方 linear benchmark 的复刻。

### 7.4 一维适配与官方复现边界

OFFICIAL_SEMANTICS_PCSSR_1D 保留官方 head、分类和 S1/S2/S3 语义，但存在明确的一维项目适配：

- Conv2d 1×1 改为等价 Conv1d 1×1；
- 图像 backbone 改为预训练 HRRP 多尺度 backbone；
- 图像增强改为第 8 节的 HRRP-safe 增强；
- 输入采用 train-known global scalar z-score；
- 两视角只在推理时作对称概率与分数平均；
- 训练日程使用本预注册固定的 40-epoch 受控协议。

因此本任务不宣称直接复现官方图像 benchmark，只称官方数学和固定代码语义的一维 HRRP 适配。

## 8. 输入、增强与训练

### 8.1 Global scalar z-score

每个 pair 仅用其 720 个 train-known 原始模型输入的全部有限元素计算：

\[
\mu=\operatorname{mean}(x),\qquad
\sigma=\sqrt{\operatorname{mean}((x-\mu)^2)}.
\]

这是一个跨样本、跨 601 位置的 global scalar population mean/std，ddof=0，float64 accumulation。要求 \(\sigma\) 有限且大于 1e-12。所有角色使用：

\[
x_z=(x-\mu)/(\sigma+10^{-12}).
\]

保存 \(\mu,\sigma\)、输入样本 ID、数组哈希和统计重算审计。known calibration 与 surrogate 不参与拟合。

### 8.2 训练样本与增强

O1–O4 都使用 720 个唯一单视角 train-known bases；每个样本每 epoch 恰好一次，drop_last=false，共 6 个 optimizer updates。

每个 epoch 对每个样本生成：

\[
x_{\mathrm{aug}}=g x_z+\epsilon,
\]

其中 \(g\sim U[0.9,1.1]\)，\(\epsilon\sim N(0,0.02^2)\)，噪声对 601 个位置独立。不得使用 shift、reverse、crop、angle interpolation、角度标签或 padding mask。

同一 phase、pair、epoch 的 O1–O4 使用完全相同的：

- base 顺序；
- batch 边界；
- 每个 sample ID 的 gain；
- 每个位置的 noise 数组。

增强数组按 sample ID 绑定，不受模型执行顺序影响。保存每 epoch 的顺序、seed、gain SHA-256、noise SHA-256 和组合输入摘要哈希。

### 8.3 随机数派生

所有派生 seed 固定使用：

1. UTF-8 编码的竖线分隔 material；
2. SHA-256；
3. digest 前 8 bytes；
4. big-endian unsigned integer；
5. NumPy PCG64 生成数据顺序、gain 和 noise。

共享训练日程 material 固定为：

official_cssr_hrrp_schedule_v1|20260906|phase|pair_id|fold_0|epoch|purpose

purpose 只允许 base_order、gain、noise。material 不含 method ID，因此 O1–O4 共享完全相同训练日程。PCG64 生成的 gain 和 noise 以 float64 构造增强输入，在进入模型前统一 cast 为 float32。

模型初始化使用 PyTorch deterministic generator：

- linear head：official_cssr_hrrp_linear_init_v1|20260906|pair_id；
- pCSSR AE：official_cssr_hrrp_pcssr_init_v1|20260906|pair_id。

O1/O3 的 linear head 初始 state 完全一致；O2/O4 的 AE 初始 state 完全一致。保存 state hash 和 RNG state。CUDA deterministic algorithms 开启，TF32、cuDNN benchmark 关闭。

### 8.4 优化器与学习率

O1–O4 固定：

| 参数 | 值 |
|---|---|
| optimizer | SGD |
| momentum | 0.0 |
| nesterov | false |
| batch size | 128 |
| epochs | 40 |
| early stopping | false |
| formal checkpoint | epoch 40 |
| gradient clipping | none |
| head base lr | 0.05 |
| head weight decay | 1e-4 |
| encoder base lr | 0.005 |
| encoder weight decay | 5e-4 |
| warmup epochs | 2 |
| milestones | epoch 25、35 |
| decay | 0.1 |

O1/O2 的 encoder 在 40 epoch 内始终 requires_grad=false 且 eval，参数及 BN buffers 必须 bitwise 不变。

O3/O4：

- epoch 1–5：encoder requires_grad=false 且 eval，只训练 head；
- epoch 6–40：encoder requires_grad=true 且 train，BN affine 可训练，BN running buffers 正常更新；
- epoch 6 的 encoder lr 直接使用当时 schedule 的 base lr 0.005，不补做前两轮 warmup；
- encoder weight decay 作用于全部可训练 encoder 参数，BN running buffers 不属于 optimizer。

每 epoch 有 6 次 update，因此 warmup 共 12 次 update。令 warmup 内第 \(q\) 次 update 为 \(q=1,\ldots,12\)，该次 optimizer step 使用：

\[
\operatorname{lr}(q)=\operatorname{base\_lr}\times q/12.
\]

即第 12 次 update 达到 base lr。epoch 3–24 使用 base lr；epoch 25 首 batch 起使用 0.1×base lr；epoch 35 首 batch起使用 0.01×base lr。scheduler 在每个 optimizer step 前设置该 step 的 lr。

不得根据 train、known calibration 或 surrogate 性能选择 epoch、调整学习率或延长训练。

### 8.5 每轮诊断

每 epoch 保存：

- train loss、single-view train Accuracy；
- known-calibration single-view Accuracy；
- known-calibration pair Accuracy；
- NLL、Brier、ECE；
- feature norm、head 或 AE norm；
- pCSSR true-class 与 nearest-wrong reconstruction gap；
- O3/O4 encoder 参数 drift 与 BN buffer drift；
- 实际学习率、样本顺序和增强哈希。

known calibration 只用于诊断，不进入训练、epoch 选择、模板或 score 标准化。

## 9. 方法、统计与分数

### 9.1 五个方法

| ID | 方法 | 训练 |
|---|---|---|
| O0_R2_CC_MLS | 封存 R2 + 类别条件 MLS | 不训练，原样复用 |
| O1_OFFICIAL_LINEAR_FT | 冻结 encoder + matched linear head | 只训练 head |
| O2_OFFICIAL_PCSSR_FT | 冻结 encoder + official-semantics pCSSR head | 只训练 AE |
| O3_OFFICIAL_LINEAR_E2E | matched linear，epoch 6 起端到端 | head + encoder |
| O4_OFFICIAL_PCSSR_E2E | pCSSR，epoch 6 起端到端 | AE + encoder |

### 9.2 Raw train templates

epoch-40 checkpoint 固定后，用无增强的 720 个 unique train-known 单视角样本建立每个模型自己的统计。

O2/O4 按模型自身 single-view pCSSR 概率预测类别分组：

- 一阶模板使用 abs(feature)；
- 先对组内样本和位置平均；
- 再跨类别逐通道除以类别和；
- Gram 模板使用同一 abs(feature) 和官方 G_p_pro，p=8；
- 每一预测类别必须至少有一个样本。

测试 S2 使用 raw signed feature，官方源码中被注释的测试时 abs 不启用。

若任一预测类别为空、类别和产生零除、模板非有限，则该训练单元硬失败。

### 9.3 Augmented-train score normalization

每个 train-known base 使用与训练相同的 gain/noise 家族产生四个确定性版本，并在原始 z-score HRRP 上增强后重新前向。独立 material 固定为
`official_cssr_hrrp_score_norm_v1|20260906|pair_id|fold_0|sample_id|variant|purpose`；`variant`
为 1–4，`purpose` 只能是 `gain` 或 `noise`。material 不含 method ID，因此 O2/O4 使用相同的标准化增强。PCG64 输出在进入模型前按第 8.3 节转为 float32。

对 S1、S2、S3 分别使用 float64、ddof=0 计算 population mean/std。要求全部有限且 std > 1e-12。标准化为：

\[
\widetilde S_j=(S_j-\mu_j)/(\sigma_j+10^{-8}).
\]

只使用 train-known；known calibration 和 surrogate 不参与模板或 mean/std。

### 9.4 两视角推理

对 pair 的两个视角：

\[
P_{\mathrm{pair}}(k)=\frac{P_1(k)+P_2(k)}2,\qquad
\hat y=\arg\max_kP_{\mathrm{pair}}(k).
\]

O1/O3 主 unknown score：

\[
u=-\max_kP_{\mathrm{pair}}(k).
\]

同时保存 max raw spatial-average logit 和 max pair probability，但不改变主分数。

O2/O4 对两个视角都按共同类别 \(\hat y\) 计算 S1、S2、S3：

\[
s_{\mathrm{pair}}=\frac12\sum_{v=1}^2
(\widetilde S_1^{(v)}+\widetilde S_2^{(v)}+\widetilde S_3^{(v)}),
\]

\[
u=-s_{\mathrm{pair}}.
\]

另报告 S1、S2、S3、S1+S2、S1+S3、S2+S3、full 和 pCSSR max pair probability；它们不用于搜索权重。

O0 保持原类别条件 MLS。所有 unknown score 统一为越大越未知。

视角交换后 pair probability、预测类别和全部 pair score 必须逐元素不变。

### 9.5 阈值

每个方法、每个 pair 的阈值仅由该方法完整 known-calibration pair 分数确定，使 known acceptance 为冻结评价器定义的 95%。必须复用仓库既有 quantile/tie 语义并由单元测试锁定。

surrogate identity 不参与阈值、模板、normalization、训练或 epoch 选择，只用于冻结后评价和预注册 pilot 标签。

## 10. 官方全链路差分

把一维输入重排为 \([B,C,L]\rightarrow[B,C,1,L]\)，与第 5.2 节固定官方源码 oracle 逐项比较：

1. 每类 AE reconstruction；
2. L1 reconstruction logits；
3. clip；
4. pixelwise probabilities；
5. softmax_avg；
6. classification loss；
7. 输入梯度；
8. 每个 AE encoder/decoder 梯度；
9. matched LinearClassifier 原语前向与梯度；
10. R[0]/R[1]/R[1]；
11. first-order prototype 与跨类别归一化；
12. 测试时 signed-feature S2；
13. G_p_pro，p=8；
14. Gram template；
15. S1、S2、S3 raw score；
16. augmented-train mean/std；
17. full integrated score。

float32 容差固定为 rtol=1e-5、atol=1e-6；float64 固定为 rtol=1e-9、atol=1e-11。不得因失败放宽容差。

差分失败时 stage_b_status=blocked_by_official_differential_failure，阶段 B 不得运行 smoke 或 pilot。阶段 A 与阶段 B 相互独立；只要阶段 A 的封存产物完整，阶段 A 可以继续并单独报告。

## 11. Smoke 与 pilot

### 11.1 Smoke

smoke 固定为：

- pair=N1；
- O1–O4 各 1 epoch，共 4 个训练任务；
- 使用完整 720 个 unique train-known bases；
- 6 个 optimizer updates；
- O1–O4 共享完全相同顺序和增强；
- 评价只从封存 N1 manifest 中按 pair ID 稳定排序后，各 known 类与各 surrogate identity 取前 2 对。

smoke 是 diagnostic_only，不保存或展示可用于方法比较的性能结论，也不进入任何 gate。只检查：

- 输入、前向、反向、optimizer 和 checkpoint 链路；
- 数值有限；
- train/calibration/surrogate/final-test 隔离；
- unique-base 每轮恰好一次；
- O1–O4 日程与增强哈希一致；
- 冻结 encoder 与 BN 不变；
- checkpoint bitwise replay；
- 两视角交换不变；
- 产物落盘、指标可反算和全量哈希。

任一 smoke 单元或 phase audit 失败时，不得运行 pilot。

### 11.2 Pilot

只运行：

- N1 / fold0 / seed20260906；
- N4 / fold0 / seed20260906；
- N2 / fold0 / seed20260906。

每个 pair 训练 O1–O4，共 12 个训练任务；O0 直接复用。不得运行额外 fold、seed 或 pair。

只有 12/12 训练单元成功、单元审计通过、聚合审计通过，才允许执行第 13 节的结果标签。

## 12. 指标与比较

每个方法、每个 pair 报告：

1. Known Accuracy；
2. Known Macro-F1；
3. AUROC；
4. OSCR；
5. FPR95；
6. KCCR；
7. URR；
8. KCCR/URR harmonic mean；
9. K+1 Macro-F1。

每个 surrogate identity 单独报告 AUROC、URR、FPR95 和 false-accept 去向，特别列出 DDG-112→DDG-1000、DDG-1000→DDG-112、MARVEL CRANE 和迷你好望角型散货船。

预注册比较：

- frozen-head CSSR effect：O2−O1；
- end-to-end CSSR effect：O4−O3；
- joint representation effect：O4−O2；
- strong-baseline comparison：O4−O0；
- score-integration effect：O2/O4 各自 full−S1-only。

所有平均 delta 先在同一 pair 内相减，再对 N1/N4/N2 作无权算术平均。正 pair 定义为 delta > 0；等于 0 不计为正。百分点门槛按原始 0–1 指标差计算，例如 +2 pp 为 +0.02。

不使用 bootstrap、置信区间或统计显著性决定标签。

## 13. Pilot 状态与确定性结果标签

### 13.1 硬失败优先

以下任一情况使 pilot_status=hard_failed_incomplete、pilot_gate=not_evaluated、selected_method=null：

- 12 项中任一训练任务失败、取消、缺失或未运行；
- checkpoint、manifest、数据隔离、共享日程或源码绑定审计失败；
- NaN/Inf、optimizer error 或参数非有限；
- O2/O4 任一 raw train 预测类别为空；
- score normalization std 不有限或不大于 1e-12；
- checkpoint 重放、逐样本指标反算或 artifact hash 不一致。

硬失败时取消未完成任务，不计算部分三-pair gate，不贴 official_cssr_no_signal 标签。

### 13.2 共同保护条件

对 CSSR full-score variant \(C\in\{O2,O4\}\)，定义：

- safe_identity(C)：所有六个 surrogate identity 的 AUROC 均不低于 0.40，且相对同 pair O0 的 AUROC 下降均不超过 0.10；
- safe_vs_O0(C)：平均 OSCR 不低于 O0、平均 KCCR 相对 O0 下降不超过 0.01、平均 FPR95 相对 O0 恶化不超过 0.02，并满足 safe_identity(C)；
- stable_vs(A,B,margin)：A−B 的三 pair 平均 AUROC 不低于 margin，且至少 2/3 pair delta > 0。

### 13.3 标签定义与优先级

标签按以下顺序只选第一个满足者：

1. official_cssr_strong_signal
   - stable_vs(O4,O3,0.02)；
   - stable_vs(O4,O0,0.01)；
   - safe_vs_O0(O4)。

2. official_cssr_method_signal_only
   - stable_vs(O4,O3,0.02)；
   - safe_vs_O0(O4)；
   - 但 stable_vs(O4,O0,0.01) 不满足。

3. official_cssr_ft_signal_only
   - O2−O1 平均 AUROC ≥0.02；
   - 至少 2/3 pair delta >0；
   - O2−O1 平均 OSCR ≥0；
   - O2−O1 平均 KCCR ≥−0.01；
   - O2−O1 平均 FPR95 ≤+0.02；
   - safe_identity(O2)；
   - O4 不满足前两个更高优先级标签。

4. official_cssr_score_integration_only
   - O2 或 O4 至少一个 full−S1-only 平均 AUROC ≥0.01；
   - 对该 variant 至少 2/3 pair delta >0；
   - 该 full variant 满足 safe_identity；
   - O2/O4 均未达到更高优先级；
   - 保存所有 qualifying_variants，不依据较高数值另选权重或方法。

5. official_cssr_no_signal
   - 完整 pilot 已通过审计，但以上标签均不满足。

“任一 identity AUROC <0.40 或相对 O0 下降 >0.10”按逻辑 OR 构成该 variant 的 identity catastrophe，使该 variant 不能获得 signal 标签；如果另一个 CSSR full variant 独立满足更高优先级条件，仍按优先级判定。该规则不允许删除困难 identity。

无论标签为何，selected_method 只表示报告中的候选建议，不授权训练或测试新单元；confirmation 和 final test 始终为 false。

## 14. 测试清单

### 14.1 阶段 A

至少覆盖：

1. D0/D1/D2 原 checkpoint strict-load；
2. 原逐样本结果精确复算；
3. E/M/r/p/a 手工样例及 ties；
4. 每个 base 只计一次；
5. view-level 到 pair-level 聚合；
6. AE 交叉重构矩阵；
7. reference quantile 与 p95；
8. Z/U entropy effective rank；
9. adapter residual ratio；
10. post-hoc S1/S2/S3 明确不进入 gate。

### 14.2 阶段 B

至少覆盖：

11. 官方 1×1 AE 前向与梯度差分；
12. matched official LinearClassifier 原语差分；
13. clip、softmax_avg 和 loss 差分；
14. S1 差分；
15. abs-train/signed-test first-order prototype 差分；
16. p=8 Gram 差分；
17. augmented-train mean/std 与集成差分；
18. O1/O2 encoder 初值完全相同；
19. O3/O4 encoder 初值完全相同；
20. O1–O4 encoder 初始参数与 BN buffers 完全相同；
21. head-only encoder 参数与 BN buffers 不变；
22. O3/O4 epoch1–5 冻结、epoch6 解冻并更新 BN；
23. unique-base 训练，不按 pair multiplicity 重复；
24. O1–O4 共享顺序、gain 和逐位置 noise；
25. surrogate unknown 不进入训练、统计或标准化；
26. final unknown/even-angle test 不生成；
27. pair probability 是两视角概率均值；
28. pair score使用同一共同预测类别下的两视角官方分数均值；
29. 交换视角不变；
30. 九项指标从逐样本结果精确反算；
31. 结果标签、优先级与 identity catastrophe 确定；
32. 任一单元硬失败导致 incomplete/not_evaluated；
33. 不自动授权 confirmation/final test；
34. global scalar z-score 只由 train-known 拟合；
35. warmup 和 milestone 在逐 update 层面精确匹配；
36. 官方 commit、六个文件哈希和 oracle 路径校验。

不得通过放宽容差、跳过 CUDA 测试或移除失败单元使 phase 通过。

## 15. 产物

阶段 A 使用独立目录：

artifacts/diagnostics/cssr_identity_failure_mechanism_v1/

至少保存：

- per_base_score_decomposition.npz；
- per_pair_view_decomposition.csv；
- ae_cross_reconstruction_raw.csv；
- ae_cross_reconstruction_normalized.csv；
- ae_specificity.json；
- reference_distribution_summary.csv；
- representation_geometry_z_vs_u.json；
- official_scores_on_d1_d2.csv；
- mechanism_audit.json；
- artifact_hashes.json。

阶段 B 的每个单元至少保存：

- resolved config、代码提交、配置哈希和环境；
- 官方 oracle commit/file hash 审计；
- 数据、split、unique-base、pair 与标签 manifest；
- z-score 统计及输入哈希；
- 初始化、每 epoch 顺序与增强哈希；
- 40-epoch 日志和 epoch-40 checkpoint；
- raw train prediction grouping、S1/S2/S3 模板与 normalization 参数；
- calibration/surrogate 逐样本概率、logits、各 score、unknown score、阈值、预测和标签；
- 九项指标与 identity false-accept 表；
- checkpoint replay、指标反算、视角交换和全量 artifact hash。

phase 聚合必须保存 12 项任务状态、O0–O4 完整表、全部预注册差值、gate 输入、唯一结果标签和授权字段。

原始数据、checkpoint、manifest、逐样本结果和日志不提交 Git；Git 只保存代码、配置、测试和报告。

## 16. 最终报告

结果文档固定为：

- docs/cssr/cssr_identity_failure_mechanism_audit_2026-09-04.md；
- docs/cssr/cssr_official_hrrp_results_2026-09-04.md。

机制报告回答：

1. MARVEL 失败主要对应哪几类已冻结证据；
2. 哪些 AE 最常成为 MARVEL 的最低重构类；
3. 这些 AE 是否也对其他 identity 低误差；
4. D1/D2 adapter 相对原 Z 是否降低 effective rank；
5. D2 为什么能改善 DDG 却损害 MARVEL；
6. 官方 S2/S3 是否缓解旧 D2 的身份依赖。

官方基线报告回答：

1. 官方全链路差分是否通过；
2. O0–O4 九项完整指标；
3. O2−O1；
4. O4−O3；
5. O4−O2；
6. O4−O0；
7. S1、S2、S3 与 full 的贡献；
8. pCSSR 自身 known 分类的实际 Accuracy、NLL、Brier 和 ECE；
9. MARVEL 与 DDG 问题是否仍存在；
10. 唯一结果标签；
11. 是否建议继续研究多视角 CSSR；
12. confirmation、最终 unknown 和偶数角 test 均未获自动授权。

“CSSR known 分类是否可靠”不设置额外事后 pass/fail 门槛，只报告冻结指标、跨 pair 范围和与 matched linear 的差异。

所有结论必须区分：

- 已确认的产物事实；
- 单 fold、单 seed surrogate pilot 的有限结论；
- 尚未验证的最终 unknown 泛化；
- 仅作机制解释的推断。

## 17. 禁止项

本任务禁止：

- 最终三类 unknown；
- 偶数角 test；
- ARPL、RCSSR；
- 伪未知、GAN、confusing samples；
- Transformer、attention、PMA；
- CSSR+CE 双头；
- 自定义 L_abs、L_sep、adapter 或 learned rejector；
- local-kernel AE；
- 类别条件 conformal p-value作为阶段 B 方法；
- 自定义 MLS/CSSR 融合；
- 第二个 fold或额外 seed；
- 根据阶段 A、smoke 或 pilot 修改网络、超参数、增强、分数、权重、阈值或 gate；
- 自动 confirmation、最终测试或新方法；
- 修改 RESEARCH_CONTEXT.md；
- 覆盖任何旧实验产物。

如果 pilot 得到局部正结果，只能按第 13 节输出预注册标签和下一步建议，不得在当前任务内扩大搜索或继续实验。
