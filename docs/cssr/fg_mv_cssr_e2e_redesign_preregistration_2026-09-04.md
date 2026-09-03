# FG-MV-CSSR v2 端到端重设计快速实验预注册

> 日期：2026-09-04
>
> 阶段：P3 独立快速机制验证
>
> 实验：`fg_mv_cssr_e2e_redesign_v2`
>
> 分支：`codex/fg-mv-cssr-e2e-redesign`
>
> 起点：`codex/fg-mv-cssr-frozen-r2`，提交 `df56bf1f85e21950c44226a2563eecf526481de2`
>
> 状态：在本实验任何 smoke、pilot 或 confirmation 性能产生前冻结

## 1. 研究问题和停止边界

本实验只回答：在当前最可靠的 `R2_MS_MEAN_CE` 基础上，类别重构是否必须通过端到端联合对齐、正确类绝对重构约束和 HRRP 距离向局部结构，才能形成稳定优于类别条件 MLS 的未知证据。

上一轮 `fg_mv_cssr_frozen_r2_v1` 的 `no_cssr_signal` 结论保持不变。该结论否定的是冻结 R2 后外挂 pCSSR-core 的具体实现，不外推为所有 CSSR 无效。本实验不继续调旧 B4，也不改变上一轮报告。

本轮是当前 CSSR 主线的最后一次有边界修正。若 pilot 没有合格候选，输出 `cssr_redesign_failed` 并停止 CSSR；若条件性 confirmation 失败，输出 `cssr_redesign_rejected` 并停止 CSSR。不得据结果修改损失、AE、epoch、分数或扩大搜索。

禁止项：最终三类 unknown、偶数角 test、ARPL、伪未知、GAN、Transformer、attention、skip connection、额外拒判器、分数融合、第二个 fold、额外 seed 和超参数搜索。`RESEARCH_CONTEXT.md` 不修改。

## 2. 冻结数据和实验单元

统一固定：

```text
angle_fold = 0
model_seed = 20260830
finetune_seed = 20260904
known_calibration_acceptance = 0.95
```

Pilot 只运行：

```text
N1 = DDG-112 / 迷你好望角型散货船
N4 = DDG-1000 / 集装箱船达飞罗尔多夫级
N2 = MARVEL CRANE / 迷你好望角型散货船
```

每个 pair 训练 Q1–Q4，共 `3 × 4 = 12` 个任务。Q0 不训练，直接复用同一单元的正式 R2 epoch-100 checkpoint。

仅当 pilot 选择出唯一 CSSR 候选时，才允许在 `N0/N3/N5/N6` 上运行 Q1 和该候选，共最多 8 个新训练任务；Q0 继续复用。不得自动运行其他 pair、fold 或 seed。

所有训练、重构参考分布和阈值只使用奇数角开发池中的 train-known 与 known calibration。surrogate unknown 只用于冻结后的评价；不得进入训练、参考分布、阈值、epoch 选择或任何分布拟合。最终 unknown 和偶数角 test 不生成 pair、特征或预测。

## 3. 方法矩阵

### Q0_FROZEN_R2_CC_MLS

- 不训练；严格加载对应单元的 R2 epoch-100 checkpoint。
- 多尺度共享 encoder、算术均值融合和线性 CE head 全部冻结。
- known prediction 为融合 CE logits 的 argmax。
- unknown score 为上一轮 B1 的预测类别条件 MLS。

### Q1_CE_FINETUNE_CONTROL

- 从与同单元 Q2–Q4 完全相同的 R2 checkpoint 初始化。
- 使用相同可训练层、动态 pair schedule、优化器、epoch、batch、学习率和正则化。
- 只优化融合 CE：`L = L_cls`。
- 主 unknown score 为微调后预测类别条件 MLS。

### Q2_E2E_REL_CSSR_1X1

- Q1 路径加每类独立的 1×1 类别 AE。
- `L = L_cls + 0.5 * L_rel`。
- 不含 `L_abs` 或 `L_sep`。
- 主 unknown score 为 guided reconstruction；类别条件 MLS 仅作诊断。

### Q3_E2E_ABSREL_CSSR_1X1

- 与 Q2 相同的 1×1 类别 AE。
- `L = L_cls + 0.5 * L_rel + 0.25 * L_abs + 0.5 * L_sep`。
- 主 unknown score为 guided reconstruction；类别条件 MLS 仅作诊断。

### Q4_E2E_ABSREL_CSSR_LOCAL3

- 与 Q3 完全相同，仅把每类 AE 固定为：

```text
Conv1d(128, 64, kernel_size=3, padding=1, bias=False)
Tanh
Conv1d(64, 128, kernel_size=3, padding=1, bias=False)
```

- 无 skip、归一化层、attention 或额外隐藏层。
- 主 unknown score为 guided reconstruction；类别条件 MLS 仅作诊断。

## 4. 初始化、冻结范围和训练配置

Q1–Q4 必须从同一 R2 state dict 初始化，epoch-0 encoder、projection 和 CE head 逐字节一致；旧 checkpoint 必须 `strict=True` 加载。原 R2 stem、stage1、stage2 冻结；只训练最后 residual stage、pooling 后 projection、CE head，以及 Q2–Q4 的类别 AE。不得改变 R2 encoder 架构或旧 state-dict key。

固定训练配置：

```text
optimizer = AdamW
epochs = 20
batch_size_pairs = 64
warmup_epochs = 2
scheduler = cosine
gradient_clip_norm = 5.0

lr_last_stage = 3e-5
lr_projection_and_ce_head = 1e-4
lr_autoencoders = 1e-3

weight_decay_last_stage = 5e-4
weight_decay_projection_and_head = 5e-4
weight_decay_autoencoders = 1e-4
dropout = 0.1
```

不 early stop；epoch 20 是唯一正式 checkpoint；epoch 0/5/10/15/20 只保存诊断，不参与选择。

## 5. 动态去重二视角配对

每个 epoch、每个 known 类从唯一 train-known 底层 HRRP 出发，使用由 `finetune_seed`、pair、fold、epoch 和 class 确定的显式 `epoch_seed` 构造确定性跨 15° frame derangement。每个底层 sample 在该 epoch 恰好出现一次于 view1、一次于 view2；每对必须来自不同 frame。

同一 pair/fold/seed 下 Q1–Q4 必须共享完全相同的 20 个 epoch schedule。保存每 epoch pair manifest、sample 使用次数、跨 frame 审计、单 epoch SHA-256 和完整 schedule SHA-256。任一约束无法满足时单元失败，不回退为固定 500 对。

## 6. 特征、AE 和损失

R2 的只读 `forward_feature_map` 返回最后 residual stage 后、pooling 前的单视角特征图。601 维输入的预期形状为 `[B,128,76]`；旧 `forward` 输出和 checkpoint key 不变。

融合分类：

\[
g=(h_1+h_2)/2,\qquad L_{cls}=CE(C(g),y).
\]

官方式相对 pCSSR 损失逐视角计算并平均：

\[
L_{rel}=\tfrac12[L_{pCSSR}(Z_1,y)+L_{pCSSR}(Z_2,y)].
\]

其语义严格保持现有已通过差分测试的负 L1 logits、`gamma=0.1`、clip 和 `softmax_avg` 顺序。

本项目新定义的绝对归一化重构误差为：

\[
r_{v,k}=\frac{\operatorname{mean}_{c,t}|Z_v-A_k(Z_v)|}{\operatorname{mean}_{c,t}|Z_v|+10^{-8}}.
\]

分母只使用一次全图平均绝对激活；它不是旧 pCSSR 的逐位置激活平方归一化，也不称为官方完整 CSSR 分数。

\[
L_{abs}=\tfrac12\sum_v r_{v,y},
\]

\[
L_{sep}=\tfrac12\sum_v\max(0,0.2+r_{v,y}-\min_{k\ne y}r_{v,k}).
\]

每 epoch 保存 `L_cls/L_rel/L_abs/L_sep`、正确类和最近错误类 `r`、margin、各损失对最后 stage 和 AE 的梯度范数。出现 NaN/Inf 立即失败；任一辅助损失梯度持续超过 CE 梯度 100 倍时失败并报告，不自动改权重。这里“持续”预注册为连续 3 个完整 epoch 的 epoch 均值均超过 100 倍；CE 梯度分母使用 `max(CE_norm, 1e-12)`，同时保存原始范数以便审计。

## 7. 过拟合诊断

Q1–Q4 每个 epoch 在 known calibration 上记录 Accuracy、Macro-F1、NLL、Brier、ECE、平均最大 logit、top1-top2 logit margin、单视角和融合 feature norm、CE head norm、相对 epoch-0 R2 的 feature drift，以及融合 logits 相对 epoch-0 R2 的 KL divergence。

这些量只用于解释 CE Accuracy 饱和后的置信度变化和 CSSR 辅助损失影响，不用于选择 epoch、方法或调参。ECE 固定为 15 个等宽置信度 bin；空 bin 不贡献加权和。NLL/Brier/ECE 基于各方法自身 fused softmax 概率。

## 8. 冻结后评分和阈值

所有方法的 known prediction 都由自身融合 CE logits argmax 给出。

Q0/Q1 主 unknown score：预测类别条件 MLS，分数越大越未知。

Q2–Q4 主 unknown score：对 known calibration 中真实类别为 `k` 的唯一单视角底层样本建立对应 `r_{v,k}` 参考分布。known calibration 自身评分按 `sample_id` leave-one-base-out。对当前视角：

\[
p_{v,k}=\frac{1+\#\{r\in R_k:r\ge r_{v,k}\}}{|R_k|+1},\qquad a_{v,k}=-\log(p_{v,k}+10^{-8}).
\]

融合预测类为 `y_hat` 时：

\[
u_{guided}=\tfrac12(a_{1,yhat}+a_{2,yhat}).
\]

两槽 reference 分布按各自槽位独立建立；交换两个视角时同时交换槽位 reference，融合预测和 guided score 必须不变。若同一 `sample_id` 在 calibration pair 中重复，leave-one-base-out 排除该底层样本的全部派生出现，而不是只排除当前 pair 行。

Q2–Q4 同时保存自身类别条件 MLS，但只作诊断，不能参与主 gate、加权或组合。每种方法的 threshold 仅由自身 known calibration 主分数按 95% known acceptance 确定。surrogate unknown 不进入 reference 或 threshold。

## 9. 指标和身份级分析

每种方法报告 Known Accuracy、Known Macro-F1、AUROC、OSCR、FPR95、KCCR、URR、KCCR/URR 调和平均和 K+1 Macro-F1。保存逐样本 logits、known prediction、unknown score、threshold、open-set prediction、标签和来源，九项指标必须可零误差反算。

每个 surrogate identity 单独报告 AUROC、URR、FPR95 和 false-accept 去向；另外报告 DDG-1000/DDG-112 双向吸收，以及 MARVEL CRANE 和迷你好望角型散货船的反向表现。

## 10. Pilot gate 和唯一候选选择

Q2、Q3、Q4 分别相对 Q1 判定。合格必须同时满足：

1. 三个 pair 平均 AUROC delta `>= +2.0 pp`；
2. 至少 `2/3` pair 的 AUROC delta 为正；
3. 平均 OSCR delta `>= 0`；
4. 平均 KCCR delta `>= -1.0 pp`；
5. 平均 FPR95 delta `<= +2.0 pp`；
6. 相对 Q0 的三 pair 平均 AUROC delta `>= +1.0 pp`；
7. 任一单独 surrogate identity AUROC 不低于 `40%`；
8. 任一 identity 相对 Q1 的 AUROC delta 不低于 `-10 pp`。

选择规则按复杂度优先：

- Q2 合格时，Q3/Q4 只有相对 Q2 的三 pair 平均 AUROC再提高至少 `2 pp` 才可取代；若 Q3、Q4 都达到取代门槛，先比较平均 AUROC，完全相等时选结构更简单的 Q3。
- Q2 不合格但 Q3 合格时，Q4 只有相对 Q3 的三 pair 平均 AUROC再提高至少 `2 pp` 才可取代。
- 仅 Q4 合格时选择 Q4。
- 多个候选在规则后仍完全并列时，按 `Q2 < Q3 < Q4` 选择更简单者。
- 全部不合格时输出 `cssr_redesign_failed`，禁止 confirmation。

唯一输出标签为：Q2=`e2e_alignment_signal`，Q3=`absolute_alignment_signal`，Q4=`local_structure_signal`，无候选=`cssr_redesign_failed`。

## 11. 条件性 confirmation

只有 pilot 选择出唯一 CSSR 候选时，才在 `N0/N3/N5/N6` 各运行 Q1 和所选候选，Q0 复用。成功必须同时满足：

1. 四个 pair 平均 AUROC delta 相对 Q1 `>= +2.0 pp`；
2. 至少 `3/4` pair AUROC delta 为正；
3. 平均 OSCR delta `>= 0`；
4. 平均 KCCR delta `>= -1.0 pp`；
5. 平均 FPR95 delta `<= +2.0 pp`；
6. 相对 Q0 的平均 AUROC delta `>= +1.0 pp`；
7. 无单独 identity AUROC低于 `40%`；
8. 无 identity 相对 Q1 AUROC下降超过 `10 pp`。

通过输出 `cssr_redesign_worth_full_validation`；失败输出 `cssr_redesign_rejected`。两者均保持 `final_unknown_test_authorized=false`，不自动运行最终测试、第二 fold 或额外 seed。

## 12. 实现、验证和产物

计划新增独立配置、模型、runner、测试和结果报告，不复用或覆盖上一轮结果目录：

```text
configs/experiments/cssr/fg_mv_cssr_e2e_redesign_v2.yaml
src/hrrp_osr/models/cssr_e2e_1d.py
src/hrrp_osr/training/fg_mv_cssr_e2e_redesign.py
tests/test_fg_mv_cssr_e2e_model.py
tests/test_fg_mv_cssr_e2e_protocol.py
tests/test_fg_mv_cssr_e2e_runner.py
docs/cssr/fg_mv_cssr_e2e_redesign_results_2026-09-04.md
artifacts/cssr/fg_mv_cssr_e2e_redesign_v2/  # 不提交 Git
```

运行顺序固定为：本地完整 pytest、Python compile、配置校验、`git diff --check`、GPU 专项测试、N1 极小 smoke、12 项 pilot、pilot 全量审计；只有 gate 通过后才运行最多 8 项 confirmation 及其审计。

每个单元保存 resolved config、源 R2 引用及哈希、epoch-wise pair schedule 与哈希、normalization、checkpoint、训练日志、梯度/置信度诊断、reference 分布、逐样本预测和分数、阈值、身份级错误分析、环境和全量文件哈希。phase 保存任务计划、launcher 状态、聚合表、gate/selection 决策、完整性审计和 `_PHASE_SUCCESS.json`。

本预注册中的附加消歧仅使实现可确定，不改变交接文档的研究假设、候选、训练权重、数据、gate 或停止边界。任何实现中发现的未决语义必须在产生新性能前写入本文件或独立勘误并提交；产生性能后不得补规则。
