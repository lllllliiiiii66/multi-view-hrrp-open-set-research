# 随机无序、无显式角度信息的多视 HRRP 开集识别：Baseline 方法选型报告

**检索与核查截止：2026-08-10**  
**目标场景：** 每个目标类别具有完整 360°、1°间隔仿真 HRRP；一次识别样本是同一目标的若干条随机、无序 HRRP；不提供显式姿态角；测试同时完成已知型号分类和未知型号拒识。

## 技术摘要

第一轮 baseline 不应按原论文最高指标堆叠，而应构成一条能逐项归因的链：

> 单视下限 → 多视晚融合 → 严格置换不变的集合特征融合 → 视角间交互 → EVT 校准 → 重构拒识 → 已发表直接多视开集方法。

据此，建议第一轮采用 **7 个实验配置、5 个主要实现模块**：

1. 单视共享 1D-CNN + Energy/MSP；
2. 多视逐条 1D-CNN + 平均 logit/概率 + 同一 Energy/MSP；
3. Deep Sets（共享编码器 + mean pooling）+ Energy/MSP；
4. 无位置、无角度编码的 Set Transformer + Energy/MSP；
5. Deep Sets + OpenMax；
6. Deep Sets + 类条件重构分数；
7. JDSR-OSR（固定 5 视）的独立历史直接基线。

前 6 个配置共用单视编码器、类别划分、集合采样和阈值协议；其中 MSP 与 Energy 是同一模型的零训练成本评分变体，不另算一套网络。JDSR 单独比较，不强求参数量相等，但必须使用相同底层 HRRP 划分和同一未知类协议。

该集合足以回答六个互补问题：多视数量本身是否有用；特征融合是否优于决策融合；严格集合不变是否必要；视角交互是否优于均值统计；EVT 是否优于普通置信度；重构证据是否提供互补；现代集合方法是否确实超过最接近的已发表直接方法。

## 1. 选型原则

### 1.1 纳入 baseline 的必要条件

主 baseline 至少应满足以下之一：

- 原任务就是 HRRP 目标身份开集识别，且拒识机制可迁移到集合级特征；
- 原任务是多视/多站 HRRP 联合分类，且能删除角度、位置或站点身份后用于无序集合；
- 原任务直接是多条 HRRP 联合输入、已知目标分类与真实未知目标拒识；
- 虽非 HRRP 论文，但给出处理无序集合的标准结构、公开实现和明确的置换不变性质。

以下情形不进入主排名：

- 依赖显式姿态角或训练样本角度排序；
- 依赖连续时间/角度顺序；
- “multi-modality”实际指类内多峰或少样本 episode，并非多条 HRRP 联合输入；
- 未知类在训练、阈值选择、早停或超参数选择中被使用；
- 输入任务是运动状态、未知子类聚类或 HRRP+ISAR 多模态，而非目标型号开集识别。

### 1.2 公平性原则

所有可共享的神经 baseline 使用同一 1D-HRRP 编码器、相同特征维数和相同训练预算。只在需要回答对应问题时改变一个部件：

| 对照 | 唯一变化 | 可归因结论 |
|---|---|---|
| 单视 vs 晚融合 | 视图数与分数平均 | 多视观测本身的收益 |
| 晚融合 vs Deep Sets | 决策层平均改为特征层 mean pooling | 学习集合级表征的收益 |
| Deep Sets vs Set Transformer | 均值统计改为无位置自注意交互 | 视角间互补/冲突建模的收益 |
| Deep Sets-Energy vs Deep Sets-OpenMax | 只换开集判决头 | EVT 尾部校准的收益 |
| Deep Sets-Energy vs Deep Sets-Reconstruction | 只增加重构证据 | 生成式边界的互补性 |
| Deep Sets/Set Transformer vs JDSR | 深度集合表示对比联合稀疏重构 | 相对最接近已发表直接方法的真实增益 |

## 2. 第一轮最小充分集合

### M0：Single-view CNN + Energy/MSP

- **输入：** 从集合中固定随机抽取 1 条 HRRP；训练和测试均单视。
- **判决：** 已知类取最大 logit；Energy 为主分数、MSP 为免费附加分数；阈值由已知验证集分位数确定。
- **回答：** 不使用多视信息时，分类和拒识的下限是多少？
- **必要性：** 没有这一项，就无法把多视收益与更大模型容量区分开。

### M1：Lundén-style late fusion + Energy/MSP

[Lundén 与 Koivunen（2016）](https://doi.org/10.1109/RADAR.2016.7485271)对各雷达通道分别做 DCNN 分类，再平均类别后验；最大融合概率低于阈值时拒识。其多站架构成立，但已核实验没有真实未见类别完整验证，因而这里只复现“逐视编码 + 等权决策融合”的结构思想，不宣称复现原论文 OSR 结果。[Aalto 官方记录](https://research.aalto.fi/fi/publications/deep-learning-for-hrrp-based-target-recognition-in-multistatic-ra/)概括了各通道局部 CNN、概率平均与全局最大概率阈值。

- **适配：** 所有视图共享同一个 1D-CNN；对 logit 或概率取均值；不建立固定站点分支。
- **顺序/角度：** 平均操作天然不依赖排列，不输入角度。
- **回答：** 只增加视图并平均，是否已经足够？
- **复现难度：** 低；但只能称 Lundén-style 变体，不能称精确复刻原仿真系统。

### M2：Deep Sets + Energy/MSP

[Deep Sets](https://papers.nips.cc/paper/6931-deep-sets.pdf)以共享元素映射后求和/均值，再用集合级网络输出；无位置或角度输入时具有严格置换不变性。作者提供了[官方代码](https://github.com/manzilzaheer/DeepSets)。

- **适配：** 每条 HRRP 经共享 1D-CNN 得到 (z_i)，用 masked mean 得到集合表征，再分类。
- **集合大小：** 天然支持可变 (V)，但训练时必须覆盖计划测试的视图数量范围。
- **回答：** 集合级特征学习是否优于逐视决策平均？可变视图数是否稳定？
- **复现难度：** 低，是本研究最重要的结构基准。

### M3：Set Transformer（无位置/角度编码）+ Energy/MSP

[Set Transformer](https://proceedings.mlr.press/v97/lee19d.html)用集合自注意建模元素之间的交互，并用 PMA 聚合；其[官方 PyTorch 实现](https://github.com/juho-lee/set_transformer)明确说明结构针对置换不变集合设计。

- **适配：** 共享 M2 的 HRRP 编码器，只把 masked mean 替换为 SAB/ISAB + PMA；不加入位置、角度、站点编号。
- **公平控制：** 调整隐藏维数，使参数量与 M2 在同一数量级；同时报告 FLOPs 和延迟。
- **回答：** 宽角随机视图之间的交互是否包含均值统计无法表达的互补信息？
- **复现难度：** 中；官方代码公开，领域适配工作量可控。

这一设计也得到 HRRP 闭集证据支持：郭帅等（2023）的[多站角度引导 Transformer](https://radars.ac.cn/article/doi/10.12000/JR23014)在 5 类实测飞机模拟 3 站数据上，普通无角度多站 Transformer 特征融合为 93.60%，完整角度引导模型为 96.90%；官方页面同时显示“资源附件(0)”。主模型依赖方位角，不能作为本场景主 baseline；其无角度分支和站间置换不变融合才是应复现的部分。

### M4：Deep Sets + OpenMax

[Chen 等（2019）](https://doi.org/10.1049/joe.2019.0706)把 CNN 激活到类中心的距离尾部拟合为 Weibull，并用 OpenMax 产生未知类别概率。论文在未知类由 1 增至 4 时报告 F1 约 92.50%、88.92%、84.43%、80.06%，但原文仅 4 页，数据与尾长等细节不足，未定位作者官方代码。

- **适配：** 对 M2 的集合级激活拟合类中心和 Weibull，不对每个视图先做 OpenMax 再平均。
- **阈值：** Weibull 只用训练集拟合；unknown probability 或校准分数阈值只用已知验证集确定。
- **回答：** 在完全相同的集合表征上，EVT 尾部校准是否优于 Energy/MSP？
- **复现难度：** 中；应报告尾长敏感性，不照搬论文未知类参与比较的工作点。

### M5：Deep Sets + class-conditional reconstruction

[Wan 等（2019）](https://link.springer.com/article/10.1186/s13634-019-0603-y)使用 CNN 编码器与反卷积解码器，以重构误差拒绝库外目标；实验以 3 类实测飞机为已知、XPATCH 仿真卡车为未知，最佳 AUROC 约 0.9662。原论文用 ROC 扫描阈值且已知/未知跨数据源，不能直接照搬其工作点。

- **适配：** 共享 M2 的集合表征；每类一个轻量解码头，或一个带类别条件的共享解码头；重构集合内各 HRRP 特征而非强制恢复视图顺序。
- **集合级分数：** 对视图重构误差使用 median 或 trimmed mean；mean 作为对照。
- **阈值：** 每类已知验证误差分位数，不使用测试未知。
- **回答：** 重构证据是否能拒绝 OpenMax/Energy 难以发现的未知？是否更容易被某个异常视图污染？
- **复现难度：** 中；这是迁移论文思想的公平变体，不是逐层精确复刻。

### M6：JDSR-OSR

[刘盛启等（2023）](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT221284?viewType=HTML)的 JDSR-OSR 是目前与目标场景最接近的已发表直接方法。它以固定 (J=5) 的多条 HRRP 为一次输入，共享候选类别块但允许各视角在块内选择不同原子；先以联合重构误差选类，再对匹配误差右尾和非匹配误差左尾分别拟合 GPD，并加权阈值判决。Situation-II 使用随机角度排列/组合，平均识别率约 0.816；但随机协议不等于模型已被证明严格置换不变。

- **原始协议：** MSTAR SAR 芯片反演 HRRP；17°训练、15°测试；BMP2、BTR70、T72 为已知，其余 7 类逐步加入未知；固定 5 视。
- **关键参数：** (J=5)、稀疏度 (K=2)、尾比例 (ho=0.7)、全局阈值 (delta_g=0.3)。
- **公开性：** 官方期刊页为“资源附件(0)”；题名、DOI和 GitHub 检索未定位代码、反演或分组脚本。
- **本研究复现：** 只在 (V=5) 上按核心算法复现；使用本研究 HRRP 原始划分重建字典，另做输入置换和随机宽角压力测试。
- **回答：** 现代深度集合方法相对直接已发表联合稀疏方法是否有真实提升？
- **复现难度：** 高，应与神经 baseline 分开报告实现不确定性和运行成本。

## 3. 三级清单

### 3.1 强烈推荐

| 方法 | 推荐方式 | 原因 | 第一轮作用 |
|---|---|---|---|
| Chen et al. 2019 CNN-OpenMax | 移植 OpenMax 到集合级特征 | 经典 HRRP-EVT，结构简单，可与 Energy 严格对照 | 开集头对照 |
| Wan et al. 2019 CNN-AE | 移植重构头，改为 known-only 阈值 | 与 EVT 技术路线互补 | 生成式拒识对照 |
| Lundén-style late fusion | 共享编码器 + 平均 logit/概率 | 最低成本检验“多视数量本身” | 多视晚融合下限 |
| Deep Sets | mean pooling，无角度/位置 | 严格置换不变、官方代码、可变 (V) | 主集合基线 |
| Set Transformer | 无位置/角度的 SAB/ISAB+PMA | 检验视角交互，官方代码 | 交互式集合基线 |
| 郭帅等 2023 无角度分支 | 只复现普通多站 Transformer 融合抽象 | HRRP 领域内的置换不变融合证据 | 支撑 M3 架构选择 |
| JDSR-OSR | 固定 5 视原理复现 | 唯一含随机角度情形的严格直接论文 | 已发表直接对照 |

“强烈推荐”不等于逐参数复刻整篇论文。OpenMax、CNN-AE、Lundén 和郭帅等方法在第一轮都应作为**受控移植或消融版本**；只有 JDSR 需要保留其直接联合稀疏与双尾 EVT 核心。

### 3.2 可选

| 方法 | 何时加入 | 不列入第一轮的原因 |
|---|---|---|
| Xia et al. 2023 Closed Classification Boundary | 需要比较非参数局部边界时 | kNN 存储/推理重，且与 OpenMax 问题部分重叠 |
| SDL-MEVB 2025 | M4 显示单变量 EVT 不足，且类内宽角多峰明显时 | GEV + copula + 原型约束耦合，代码与校准比例缺失 |
| PGR 2025 | 已有可信目标物理尺度、需专门研究 near-OOD 时 | 伪 OOD 与先验解耦复现代价高 |
| OSFSM 2025 | 第二轮研究 HRRP 专用编码器与过置信约束时 | 专用骨干不利于第一轮共享编码器公平比较 |
| JSR-OSR 2022 | 需要量化严格共同支撑在宽角集合中的失效时 | 与 JDSR 路线高度重叠，且论文明确指出大角差异会破坏假设 |
| CFRPL 2026 | 仅作为“允许角度标签”的 oracle | 核心是 30°角度组编码，与主场景冲突 |
| 完整角度引导 Transformer | 仅作为角度信息上界 | 使用本研究明确禁止的方位角输入 |

### 3.3 不建议复现

| 方法 | 原因 |
|---|---|
| MtCS-OSR 2024 | 与 JDSR 同属多视稀疏重构；连续小角假设冲突；Bayesian 求解成本高；关键阈值 (delta) 的确定规则未报告 |
| MFA-Net 2025 及 GRU/RNN 连续序列方法 | 依赖连续帧或时间/角度顺序，输入假设与随机无序集合相反 |
| MMPN 2022 | “multi-modality”是姿态导致的类内多峰，任务含 few-shot 新类支持，不是无支持未知拒识，也不是多条 HRRP 联合输入 |
| MHA-CoST 2026 | 单视、多任务且还区分未知子类；数据明确不共享，超出当前问题 |
| HRRPSeqNet 2025 | 标签是空间目标运动状态，输入是严格有序序列；阈值由开放实验选择，不能作为目标身份集合 baseline |
| Chain Coverage 等角度链方法 | 训练依赖角度排序，扩展到 360°随机宽角集合困难，且缺少现代公开实现与标准 OSR 指标 |

## 4. 逐线选型结论

### 4.1 单视 HRRP 开集识别线

应抽取的是**拒识机制**，不是整套专用骨干：

- OpenMax 代表 EVT/尾部校准；
- CNN-AE 代表重构/生成式证据；
- Closed Classification Boundary 代表非参数局部闭边界；
- SDL-MEVB 代表距离与方向的多变量极值边界；
- PGR 代表物理先验伪未知；
- OSFSM 代表 HRRP 专用编码与抑制过置信。

第一轮只保留 OpenMax 与重构，是因为二者机制互补、实现可与同一 Deep Sets 表征严格配对。其余方法应在发现明确失败模式后有条件加入，而不是以论文结果高为由整套堆入。

### 4.2 多视 HRRP 闭集线

应形成三级融合复杂度：

1. Lundén-style 后验平均：没有学习视角间关系；
2. Deep Sets mean pooling：学习集合级表征，但交互仅由聚合统计隐式表达；
3. Set Transformer：显式学习视角间注意关系。

郭帅等 2023 的普通多站 Transformer 消融为第 3 级提供 HRRP 领域内证据，但完整角度引导版本只能当 oracle。连续多帧 Transformer、GRU 和 RNN 不进入主 baseline，因为其优势可能来自顺序而非集合融合。

### 4.3 直接多视 HRRP 开集线

已核三篇严格直接论文为 JSR-OSR、JDSR-OSR 和 MtCS-OSR。三者都依赖固定 5 视、稀疏/压缩感知重构和极值统计；没有已定位的公开代码。第一轮只复现 JDSR，因为：

- 它放宽了 JSR 的完全共同原子假设；
- 它唯一报告 Situation-II 随机角度情形；
- MtCS 增加 Bayesian 求解成本，却没有报告可复现的阈值规则；
- 同时复现三者不会形成新的受控消融轴。

JSR 的最佳角色是第二轮“错误先验压力测试”：它可检验随机宽角集合是否确实破坏严格共同支撑。MtCS 现阶段只应保留为发展谱系证据。

## 5. 统一数据协议

### 5.1 先切分底层 HRRP，再组成集合

必须先按单条 HRRP 的角度索引划分 train/validation/test，再从各自池中独立采样集合。不能先从 360 条 HRRP 反复组成集合后随机划分，否则同一条底层 HRRP 会跨训练和测试重复出现，造成严重泄漏。

建议：

- 对每个已知类，将 360 个角度索引做全圆均匀分层后分为训练、验证、测试池；
- 相邻角度高度相关，主结果至少再报告一个“角度块留出”协议，例如连续扇区留出，以避免随机角度点切分过于容易；
- 未知类完全不参加训练和阈值选择，只从其测试池构造集合；
- 为每个集合记录底层角度索引、采样随机种子和排列随机种子，但这些角度不输入模型。

### 5.2 集合构造

- 主配置固定 (V=5)，与 JDSR 对齐；
- 泛化压力测试使用 (V\in\{1,3,5,8,16\})，训练时对 (V) 随机化；
- 每个集合内无放回抽样；允许不同集合复用底层 HRRP，但 train/validation/test 池绝不交叉；
- 测试每个集合至少评估 10 个随机排列。Deep Sets 和无位置 Set Transformer 的输出应在数值容差内不变；
- 另设“连续小角集合”和“全圆随机宽角集合”两个采样协议，以隔离视角相关性。

### 5.3 已知/未知类划分

- 至少 5 个固定类划分种子；
- 每个划分保持相同已知类训练预算；
- 报告 unknown-class leave-many-out，而不是只选一个容易未知类；
- 如果有可定义的目标族或几何相似度，再把未知分为 near-OOD 与 far-OOD；若没有可靠依据，只按类划分报告，不事后用结果给未知难度贴标签。

### 5.4 阈值与校准

主协议采用 known-only calibration：

- MSP/Energy：阈值取已知验证集分数的固定分位数，预先选定已知接受率；
- OpenMax：Weibull 只拟合训练集正确样本；tail size 在已知验证集上选择；
- 重构：每类已知验证误差分位数；
- JDSR：先报告论文固定参数，再加一组只用已知验证重构误差确定阈值的公平版本；
- 禁止使用测试未知类别确定阈值、权重、尾长、早停轮次或模型选择。

### 5.5 指标

至少报告：

- 已知类 closed-set accuracy / macro-F1；
- AUROC、AUPR-Unknown、FPR@95%TPR；
- OSCR；
- 固定 known acceptance 下的未知拒识率；
- (K+1) 类 macro-F1；
- 不同视角数、不同未知类划分和不同随机种子的均值与置信区间；
- 参数量、FLOPs、推理延迟、峰值显存；JDSR 另报字典大小和单样本求解时间。

不要把跨论文原始指标直接排名：各论文的未知类、数据源、阈值选择和开放度不同。所有等级判断均基于与目标场景的结构匹配、互补性和可复现性，而非原论文单点数值。

## 6. 实施顺序与停止规则

### 阶段 A：共享骨架

先实现 M0、M1、M2，并完成排列不变单元测试、底层 HRRP 泄漏检查和 known-only 阈值代码。若 M2 未能稳定超过 M1，应先排查集合采样、编码器容量和训练视角数覆盖，不急于增加复杂拒识头。

### 阶段 B：视角交互

加入 M3。若 Set Transformer 对 M2 没有稳定提升，检查注意力是否只学习到均匀权重，以及性能是否只在连续小角而非宽角随机集合改善。没有稳定增益时，不继续堆叠更复杂 Transformer。

### 阶段 C：拒识机制

在固定 M2 权重或固定训练配方上加入 M4、M5。只有当 OpenMax 或重构在多个未知类划分上稳定改善 OSCR/AUROC，且不显著牺牲已知分类，才进入后续组合模型。

### 阶段 D：历史直接基线

实现 M6 JDSR。先复现固定 5 视和两类角度协议，再与 M2/M3 的 (V=5) 子集比较。若反演/字典细节无法唯一确定，保留多套合理实现并报告区间，不把某一实现差异解释成方法结论。

## 7. 证据边界与复现风险

- 截至 2026-08-10，通过题名、DOI、作者与 GitHub 查询，未定位 JSR、JDSR、MtCS、CNN-OpenMax、CNN-AE、SDL-MEVB、PGR、OSFSM 或 CFRPL 的作者官方实现；“未定位”不等于绝对不存在。
- JDSR 与郭帅等论文的官方期刊页面均显示“资源附件(0)”；这只能证明期刊页没有附件，不能排除作者私下代码。
- Deep Sets 与 Set Transformer 有作者公开仓库，但都没有 HRRP 适配代码或目标数据。
- 多数 HRRP 实测数据不公开；直接论文使用 MSTAR SAR 反演 HRRP，原始 MSTAR 可获取不等于反演、归一化、角度分组和集合采样可复现。
- 原论文阈值常与未知测试数据、ROC 扫描或未报告的离群比例有关。主实验必须统一改为 known-only calibration，同时可附加“论文式阈值”结果，但二者不能混合解释。

## 8. 最终决策

第一轮建议立即实现的不是 7 套彼此独立的大模型，而是：

- 一个共享 1D-HRRP 编码器；
- 三种融合器：晚融合、mean Deep Sets、Set Transformer；
- 三种集合级开集分数：Energy/MSP、OpenMax、类条件重构；
- 一套独立 JDSR 复现。

这构成最小但充分的因果链。若资源必须进一步压缩，优先保留 M0、M1、M2、M4、M6；M3 与 M5 分别是第一个应补回的“交互”和“生成式拒识”分支。反之，即使资源充足，也不建议在第一轮同时复现 JSR、MtCS、PGR、OSFSM 和 CFRPL，因为它们会显著增加实现差异，却不能形成同等清晰的公平消融关系。

## 主要来源

- [Lundén & Koivunen, 2016, IEEE RadarConf](https://doi.org/10.1109/RADAR.2016.7485271)
- [Chen et al., 2019, CNN-OpenMax](https://doi.org/10.1049/joe.2019.0706)
- [Wan et al., 2019, CNN classification + reconstruction](https://link.springer.com/article/10.1186/s13634-019-0603-y)
- [Zaheer et al., 2017, Deep Sets](https://papers.nips.cc/paper/6931-deep-sets.pdf)；[官方代码](https://github.com/manzilzaheer/DeepSets)
- [Lee et al., 2019, Set Transformer](https://proceedings.mlr.press/v97/lee19d.html)；[官方代码](https://github.com/juho-lee/set_transformer)
- [Qu, Liu & Fu, 2022, JSR-OSR](https://doi.org/10.1088/1742-6596/2384/1/012012)
- [Xia, Wang & Liu, 2023, Closed Classification Boundary](https://www.mdpi.com/2072-4292/15/2/468)
- [郭帅等, 2023, 角度引导多站 Transformer](https://radars.ac.cn/article/doi/10.12000/JR23014)
- [刘盛启等, 2023, JDSR-OSR](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT221284?viewType=HTML)
- [Zhang et al., 2024, MtCS-OSR](https://doi.org/10.1049/icp.2024.1229)
- [Li et al., 2025, SDL-MEVB](https://doi.org/10.1109/TAES.2025.3527429)
- [Li et al., 2025, PGR](https://doi.org/10.1109/TAES.2025.3556812)
- [Li et al., 2025, OSFSM](https://doi.org/10.1109/TAES.2025.3575045)
- [Pan, Liang & Liao, 2026, CFRPL](https://doi.org/10.1016/j.patcog.2025.112565)

更细的逐篇输入假设、代码状态、复现难度与实验问题映射见同目录的 `baseline候选矩阵.csv`。
