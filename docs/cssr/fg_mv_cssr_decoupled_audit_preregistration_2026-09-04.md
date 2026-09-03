# CSSR 梯度病理审计与解耦式多视角类别语义重构预注册

> 日期：2026-09-04
>
> 阶段：P3 独立快速机制审计与验证
>
> 实验：`fg_mv_cssr_decoupled_audit_v3`
>
> 分支：`codex/fg-mv-cssr-decoupled-audit`
>
> 起点：`codex/fg-mv-cssr-e2e-redesign`，提交 `dca13380128d2ebce9ec18113b9c270842161b0d`
>
> 状态：在本实验任何梯度审计输出、smoke、pilot 或 confirmation 性能产生前冻结

## 1. 研究问题与不可变证据边界

本实验包含两个证据用途不同的阶段：

1. 阶段 A 只解释旧 `N4-Q2` 的 100 倍梯度门槛是由小 CE 分母、真实辅助梯度主导、方向冲突还是现有 5 轮证据无法区分；其结果不进入方法性能 gate。
2. 阶段 B 在完整冻结 R2 分类路径后，验证独立 CSSR 语义适配空间能否稳定优于冻结 R2 的类别条件 MLS，以及绝对重构与分离约束是否提供额外收益。

旧实验 `fg_mv_cssr_e2e_redesign_v2` 的结论永久保持：

```text
pilot_status = hard_failed_incomplete
pilot_gate = not_evaluated
selected_method = null
```

本轮不修改旧配置、代码语义、报告或产物，不把阶段 A 当作旧实验重跑成功，也不把阶段 B 用于补完旧 gate。`RESEARCH_CONTEXT.md` 不修改。

统一禁止：最终三类 unknown、偶数角 test、ARPL、伪未知、GAN、Transformer、attention、额外 seed、第二个 fold、超参数搜索以及未预注册的网络、损失或分数组合。

## 2. 固定数据、种子与任务规模

统一固定：

```text
angle_fold = 0
R2_seed = 20260830
audit_seed = 20260904
cssr_seed = 20260905
known_calibration_acceptance = 0.95
```

阶段 A：

```text
audit_pairs = [N1, N4]
audit_method = Q2_E2E_REL_CSSR_1X1
audit_epochs = 5
task_count = 2
```

阶段 B pilot：

```text
pilot_pairs = [N1, N4, N2]
methods = [D1_DECOUPLED_REL_CSSR, D2_DECOUPLED_ABSREL_CSSR]
task_count = 6
```

只有完整 pilot 经审计并选出唯一候选时，才允许在 `N0/N3/N5/N6` 各训练一次该候选，最多增加 4 项 confirmation。D0 不训练。不得自动重跑 Q1–Q4 全矩阵。

三个 pilot pair 固定为：

- N1：DDG-112 / 迷你好望角型散货船；
- N4：DDG-1000 / 集装箱船达飞罗尔多夫级；
- N2：油气轮 MARVEL CRANE / 迷你好望角型散货船。

## 3. 共同 R2 来源与数据隔离

所有单元严格加载相应 pair 的 `R2_MS_MEAN_CE` epoch-100 checkpoint，来源为已审计的 `ms_mean_head_factorial_surrogate_v1` 正式 confirmation：

```text
formal_code_commit = 62e318de82b4221b599e06b1166483673e9c1cd3
angle_fold = 0
initialization_seed = 20260830
checkpoint_selection = fixed_final_epoch
```

必须复核 checkpoint、源配置、pair manifest、标签顺序、归一化和既有 R2 logits 哈希。所有训练仅使用奇数角开发池的 train-known；known calibration 仅用于参考分布、阈值和冻结后诊断；surrogate unknown 仅用于冻结后评价。最终 unknown 与偶数角 test 不生成 pair、特征或预测。

## 4. 阶段 A：旧 Q2 梯度病理审计

### 4.1 原方法语义精确复用

阶段 A 严格复用旧 Q2 的：R2 checkpoint、1×1 类别 AE、AE 初始化、动态 pair schedule、优化器、各参数组学习率和 weight decay、可训练层、batch size、梯度裁剪、数值设置以及

```text
L = L_cls + 0.5 * L_rel
```

旧 v2 源配置 SHA-256 固定为 `5c227c00a7ac5a88c9bf5d66618964bc05c67f45c51c2a880731f6753626512e`。阶段 A 启动时必须同时核对：N1/N4 R2 checkpoint SHA-256 分别为 `a4f6fa3235fbb5cf74b712588a0318f614a05287adec4ee881820424cddbcbaa` / `169387ad7a87463110ac7a2cd45afd7dac49428538c93c84975162e425d94ff5`；N1/N4 epoch-0 R2 common-state hash 分别为 `3a7e74e89d11812877409d415c505c087a913e3b08098ab1fa583fa1151aeb07` / `67825e2bc143b32ed52beb5778e1190bc809269dfaedfb516a692eac64fa31f2`；Q2 AE 初始 hash 为 `4c3257678eaca1bd20ea2e97b09aeaf87fdd23c71ff65dcc1696dfe61963d6fb`。

前 5 轮 schedule hash 也必须逐轮等于旧产物：

| Pair | Epoch 1–5 schedule SHA-256 |
|---|---|
| N1 | `c2583a5ef3bea986a97fb14aac738e03a16ffc0f794a13ccc3951aaad4468922`；`3e89d9a8cc76685d8dead18614818cf5480b0f308c348e744e773b9e17d8f498`；`0e591a578934bbaaebf42437870b328a92923f5799975a43e1943272e9601b39`；`85a8bd03d420797559afba991931f83dbd3189ccc17d59f54fe6644729c0fcd6`；`b07a1347d955ca466736e570b370fcf719c16f2ed1aff404cd2f7871e53f53fe` |
| N4 | `c6090d55d3500feb6e578d29c5738e8696ddd87eb68d01207a8dc3f0d1acd6f1`；`9321d11bbc7d15ea3965d9fb3dfc84181f48877a972955696b8b1f35fa2ef582`；`82244ed649e5f4c4cb192feb1a227118889521729431945e968767f6465fac10`；`ec11524bcf16eb0001878de2a2d5cb82c6c11d9f95c7b636bb3973feb96f50f1`；`85673d1df05cc1190aadafcfc93d1782363288467ec27c28f7110f54dc06951a` |

`N1-Q2` 与 `N4-Q2` 都从各自原 R2 epoch-100 checkpoint 和 seed `20260904` 的 Q2 初始状态重新开始，不从旧训练中途恢复。旧 N4-Q2 在异常前没有成功 checkpoint，因此“精确复现旧 N4-Q2”只指相同 R2、Q2 初始状态、配置、数据和 epoch-wise schedule；不得声称恢复了不存在的 Q2 checkpoint。

audit-only 模式只关闭“连续 3 个 epoch 的 mean-of-batch-ratios 超过 100 后抛异常”，仍保存 `would_have_triggered_original_100x_gate` 和首次触发 epoch。NaN/Inf、optimizer error、参数非有限、源哈希变化、数据或 schedule 不一致仍立即失败。固定只运行 5 epoch；如果 5 轮内未触发，记录 `not_triggered_within_5_epochs=true`，不得增加 epoch。

### 4.2 每 batch 诊断

在共享最后 residual stage 的同一有序参数向量上计算：

```text
g_cls = grad(L_cls)
g_rel_raw = grad(L_rel)
g_rel_weighted = grad(0.5 * L_rel)
g_total = grad(L_cls + 0.5 * L_rel)
```

每 batch 原子保存：四个 L2 norm、`weighted_rel/ max(cls,1e-12)`、cosine、dot、裁剪前总梯度 norm、裁剪后估计 norm、裁剪 scale、是否裁剪、`L_cls`、`L_rel`、train Accuracy、CE 最大置信度均值，以及 last stage、projection、CE head、CSSR AE 四个参数组的总梯度 norm。

总梯度裁剪 scale 与 PyTorch 语义对齐：

```text
scale = min(1, clip_norm / (pre_clip_total_norm + 1e-6))
post_clip_estimated_norm = pre_clip_total_norm * scale
clipped = pre_clip_total_norm > clip_norm
```

若 `||g_cls|| * ||g_rel_weighted|| <= 1e-24`，cosine 保存为 `null`；dot 仍保存。聚合 cosine 时只使用定义良好的值，并额外报告 undefined fraction。

`L_rel` 对 projection 与 CE head 应为无梯度；绝对 norm `<=1e-12` 视为浮点零，超过该值即实现失败。额外诊断的 autograd 调用不得改变 forward、RNG、optimizer 或参数更新；审计模式与原训练语义需通过状态差分测试。

每完成一个 batch 都原子替换 `batch_diagnostics.jsonl` 的完整快照；发生任何异常时，先原子保存当前诊断和 `failure_state.json`，再抛出错误。

### 4.3 每 epoch 聚合

对 `weighted_rel/cls` 同时保存：

```text
A = mean(batch ratios)
B = mean(||g_rel_weighted||) / max(mean(||g_cls||), 1e-12)
C = sqrt(mean(||g_rel_weighted||^2)) /
    max(sqrt(mean(||g_cls||^2)), 1e-12)
```

另保存 ratio 的 min/median/p90/p95/max；p90/p95 固定使用 NumPy `quantile(method="linear")`；cosine 的 mean/median/positive fraction/negative fraction/undefined fraction；CE gradient `<1e-4/<1e-5/<1e-6/<1e-7/<1e-8` 的 batch fraction；clipping fraction；四个参数组的 epoch 前后相对更新

```text
||theta_after-theta_before|| / max(||theta_before||, 1e-12)
```

以及 known calibration Accuracy、NLL、ECE、平均最大 logit、平均单视角/融合 feature norm。原 100 倍规则仍只依据 A 连续 3 个 epoch `>100` 判定。

### 4.4 解释标签的冻结规则

阶段 A 的主标签解释 N4，N1 作为相同实现的对照并单独保存全部判据。以下阈值只用于生成解释标签，不进入阶段 B 或任何性能选择：

- `small_ce_denominator_evidence`：至少 50% batch 的 `||g_cls||<1e-4`；
- `robust_auxiliary_domination_evidence`：B 或 C 在至少 3 个 epoch `>100`；
- `large_absolute_auxiliary_evidence`：至少 3 个 epoch 的 weighted-relative batch median norm `>=5`；
- `frequent_clipping_evidence`：至少 3 个 epoch 的 clipping fraction `>=0.5`；
- `large_parameter_update_evidence`：任一 epoch 的 last-stage 相对更新 `>=0.01`；
- `strong_conflict_evidence`：至少 3 个 epoch 同时满足 cosine median `<=-0.25` 且 negative fraction `>=0.75`；
- `rapid_calibration_drift`：相对 epoch 0，Accuracy 下降至少 `2 pp` 或 NLL 增加至少 `0.1`。

标签顺序固定为：

1. 同时有 `small_ce_denominator_evidence`，且有 strong conflict、frequent clipping 或 rapid calibration drift：`mixed_gradient_conflict`；
2. 无 small-denominator evidence，且 robust domination 成立，同时 large-absolute、frequent clipping、large update、strong conflict 或 rapid drift 至少一项成立：`true_auxiliary_domination`；
3. small-denominator evidence 成立、robust domination 不成立、全程 clipping fraction `<0.1`、last-stage 相对更新始终 `<0.01` 且无数值异常：`ratio_denominator_collapse_likely`；
4. 其他情况：`inconclusive`。

若阶段 A 出现代码错误或非有限值，阶段 B 被阻断；否则无论解释标签是什么，阶段 B 都继续。输出独立报告 `docs/cssr/cssr_gradient_pathology_audit_2026-09-04.md`。

## 5. 阶段 B：解耦 CSSR 模型

### 5.1 冻结 R2 分类路径

R2 的 encoder、全部 BatchNorm 参数与 running buffers、projection 和 CE head 全部 `requires_grad=False` 并始终处于 eval mode。训练前后完整 R2 state hash 必须一致。

已知预测始终为：

```text
Z_v = R2.forward_feature_map(x_v)
h_v = R2 pooling + projection
g = (h_1 + h_2) / 2
logits_fused = CE_head(g)
y_hat = argmax(logits_fused)
```

CSSR 不改变 `h_v/g/logits_fused/y_hat`，也不改变 D0 的类别条件 MLS。D0、D1、D2 在同一 pair 上的 fused logits、known prediction 和 Known Accuracy/Macro-F1 必须逐元素一致。

### 5.2 共享 CSSR 专用适配器

两个视角共享唯一适配器：

```text
delta = Conv1d(128,64,kernel_size=3,padding=1,bias=False)
        -> GroupNorm(8,64,eps=1e-5,affine=True)
        -> GELU
        -> Conv1d(64,128,kernel_size=1,bias=False)
U = Z + 0.1 * delta
```

residual scale 固定、不可学习；无 view-specific 参数、angle、attention 或进入类别 AE 的额外 skip。交换两视角后逐视角输出相应交换，pair score不变。适配器只服务 CSSR，绝不回接 CE 路径。

### 5.3 类别特定局部 AE

每个已知类别独立拥有：

```text
Conv1d(128,32,kernel_size=3,padding=1,bias=False)
Tanh
Conv1d(32,128,kernel_size=3,padding=1,bias=False)
```

类别间不共享参数；无 skip、BatchNorm、LayerNorm、attention 或共享 decoder；重构对象为 `U`。

### 5.4 两个固定变体

`D1_DECOUPLED_REL_CSSR`：

```text
L = L_rel
```

保持已通过差分测试的 negative channel-sum L1 logits、`gamma=0.1`、clip `[-100,100]`、class softmax per position 后 spatial mean。

`D2_DECOUPLED_ABSREL_CSSR`：

```text
r_k = mean(|U-A_k(U)|) / (mean(|U|)+1e-8)
L_abs = mean(r_y)
L_sep = mean(max(0, 0.2 + r_y - min_{k!=y}(r_k)))
L = L_rel + 0.25 * L_abs + 0.5 * L_sep
```

不测试其他权重、margin、latent dimension、1×1 AE、多模态 AE、mixture、ARPL 或 MLS/CSSR 分数加权。

## 6. 阶段 B 的唯一单视角训练数据与顺序

阶段 B 不构造训练 pair。每个单元从 train-known 提取按 `(model_label,sample_id)` 排序的 720 个唯一底层样本，每类 144 条；每个 sample_id 每 epoch 恰好出现一次。pair multiplicity 不增加训练权重。

每 epoch 先按类别分别使用

```text
fg_mv_cssr_decoupled_single_view_class_v1|cssr_seed|pair_id|fold|epoch|model_label
```

的 SHA-256 前 8 字节按大端无符号整数初始化 `numpy.random.Generator(PCG64)`，对各类 144 条排序样本独立 permutation。再用

```text
fg_mv_cssr_decoupled_single_view_class_order_v1|cssr_seed|pair_id|fold|epoch
```

派生一次五类轮询起始顺序，并按该顺序循环交织五类的第 1、2、…、144 条样本。这样完整 epoch 五类各 144 条，任意完整 batch 的类别数最多相差 1；DataLoader 不再 shuffle。D1/D2 在同一 pair 上共享 20 个 epoch 的顺序与哈希。保存每 epoch manifest、sample usage、class count、class seed、class-order seed 和完整 schedule hash。

known calibration、surrogate unknown、最终 unknown 与偶数角 test 均不进入训练。

## 7. 初始化与固定训练配置

D1/D2 使用同一 `cssr_seed=20260905` 初始化 adapter 和 AE；同一 pair 的 epoch-0 adapter/AE state hash 必须一致。构造模型与 optimizer 后重新设置 CPU/CUDA RNG，以免构造消耗改变训练随机流。

```text
optimizer = AdamW
batch_size = 128
epochs = 20
warmup_epochs = 2
gradient_clip_norm = 5.0
lr_adapter = 3e-4
lr_autoencoders = 1e-3
weight_decay_adapter = 1e-4
weight_decay_autoencoders = 1e-4
early_stopping = false
formal_checkpoint = epoch 20
```

学习率在每个 epoch 更新前设置：epoch 1/2 的 factor 为 `0.5/1.0`；epoch 3–20 为 `0.5*(1+cos(pi*(epoch-3)/18))`，20 个 epoch 都执行正学习率更新，概念上的 0 位于 epoch 20 完成之后。

epoch 1–5 adapter `requires_grad=False` 且实际 lr 为 0，只训练类别 AE；epoch 6–20 adapter 解冻，使用全局 cosine factor 对应的 lr，与 AE 联合训练。adapter 参数可以预先存在于 optimizer 中，但冻结期不得产生梯度、weight decay 或参数变化。

不使用 known calibration 或 surrogate unknown 选择 epoch；epoch 20 是唯一正式 checkpoint。

## 8. 阶段 B 梯度、表示与硬失败监控

每 epoch 保存 `L_rel/L_abs/L_sep`、adapter/AE/total gradient norm、clipping fraction、adapter/AE 参数相对更新、CSSR train/known-calibration classification Accuracy、true-class `r`、nearest-wrong `r` 和 reconstruction margin。

每个 epoch 完成后，用按 `(model_label,sample_id)` 排序的完整 720 条 train-known 作为固定诊断人口计算 `U`；不使用当轮训练 permutation。保存：每样本 Frobenius norm 均值；在 sample×position 维度上的逐通道方差 min/mean/max；中心化矩阵 `[720*76,128]` 的奇异值能量摘要。effective rank 固定为

```text
p_i = s_i^2 / sum_j s_j^2
effective_rank = exp(-sum_i p_i * log(p_i + 1e-12))
```

并保存前 10 个能量占比。若完整集合 `max(channel_variance)<=1e-12`，判定 `U` 完全坍缩并硬失败。

本阶段不使用“辅助梯度/CE 梯度 100 倍”门槛。硬失败仅限：NaN/Inf、optimizer error、参数非有限、上述完全常数坍缩、checkpoint 无法严格重放、数据泄漏、manifest/schedule/源码哈希不一致。checkpoint 回放在同一已冻结 CUDA 数值环境中要求逐数组 bitwise exact。不得因性能差提前停止。

## 9. 冻结后推理与分数

known prediction 永远使用冻结 R2 的 `y_hat`。对 known calibration 中真实类别为 `k` 的唯一单视角底层样本建立一个由两个槽位共享的 `r_k` 参考分布；自身评分按 `sample_id` leave-one-base-out，pair multiplicity 不进入参考。

```text
p_v,k = (1 + count(reference_r_k >= r_v,k)) / (n_k + 1)
a_v,k = -log(p_v,k + 1e-8)
u_guided = 0.5 * (a_1,y_hat + a_2,y_hat)
```

每个方法 threshold 只由自己的 known calibration pair score 按 95% known acceptance 产生。surrogate unknown 不进入参考、阈值或分布拟合。

正式比较：

- D0：冻结 R2 的预测类别条件 MLS；
- D1：解耦 relative CSSR 的 `u_guided`；
- D2：解耦 abs+relative+separation CSSR 的 `u_guided`。

另保存 R2 全局 MLS `-max(fused_logits)` 作为背景诊断，不进入方法选择。禁止 independent-view best class、common-class search、max-view、top-2、JS divergence、learned rejector 或任何分数融合。

## 10. Smoke、pilot 指标与身份分析

在正式 pilot 前，N1 的 D1/D2 各运行 6 epoch，以同时覆盖 epoch 1–5 的 adapter 冻结路径和 epoch 6 的解冻路径；每轮仍使用全部 720 条唯一 train-known，known calibration 和 surrogate unknown 各类别只取冻结 manifest 顺序的前 2 对做链路评价。smoke 是 diagnostic，不进入 gate。

Pilot 固定运行 N1/N4/N2 × D1/D2 共 6 项。每种方法报告 Known Accuracy、Known Macro-F1、AUROC、OSCR、FPR95、KCCR、URR、KCCR/URR harmonic mean 和 K+1 Macro-F1。每个 surrogate identity 单独报告 AUROC、URR、FPR95 和全部 false-accept 去向。

## 11. Pilot gate 与唯一候选

D1、D2 分别相对 D0 判定。候选必须同时满足：

1. 三 pair 平均 AUROC delta `>=+2 pp`；
2. 至少 `2/3` pair AUROC delta 为正；
3. 平均 OSCR delta `>=0`；
4. 平均 KCCR delta `>=-1 pp`；
5. 平均 FPR95 delta `<=+2 pp`；
6. 任一 surrogate identity AUROC `>=40%`；
7. 任一 identity 相对同 pair D0 的 AUROC delta `>=-10 pp`；
8. N1 中 DDG-112 被吸收到 DDG-1000 的 false accept 数不高于 D0；
9. N4 中 DDG-1000 被吸收到 DDG-112 的 false accept 数不高于 D0。

选择顺序：D1 合格时优先 D1；D2 只有自身也合格且三 pair 平均 AUROC 比 D1 至少再高 `2 pp` 才取代 D1。D1 不合格而 D2 合格时选 D2；均不合格时输出 `decoupled_cssr_failed`。

标签固定为：

```text
D1 -> decoupled_relative_signal
D2 -> decoupled_absolute_alignment_signal
none -> decoupled_cssr_failed
```

不得依据结果修改网络、loss、训练时长、数据、分数或 gate。

## 12. 条件性 confirmation

仅在 pilot 产生审计通过的唯一候选时，在 `N0/N3/N5/N6` 各运行该候选一次；D0 直接复用。

成功必须同时满足：四 pair 平均 AUROC delta 相对 D0 `>=+2 pp`、至少 `3/4` pair 为正、平均 OSCR不下降、平均 KCCR下降不超过 `1 pp`、平均 FPR95恶化不超过 `2 pp`、任一 identity AUROC不低于 `40%`、任一 identity相对D0下降不超过 `10 pp`。

通过输出 `decoupled_cssr_worth_full_validation`，否则输出 `decoupled_cssr_rejected`。无论结果如何均保持 `final_unknown_test_authorized=false`，不得自动增加 fold、seed 或最终测试。

## 13. 审计、产物和运行顺序

新增独立配置、模型、两个 runner、测试和报告；产物使用 `artifacts/cssr/fg_mv_cssr_decoupled_audit_v3/`，不得覆盖 `fg_mv_cssr_frozen_r2_v1` 或 `fg_mv_cssr_e2e_redesign_v2`。

每个单元保存 resolved config、R2 来源及哈希、唯一 base 与评价 manifest、训练顺序及哈希、checkpoint、逐 batch/epoch 梯度与表示日志、参考分布、逐样本 logits/预测/unknown score/threshold、身份错误分析、环境和全量文件哈希。表格和 gate 必须能由逐样本预测精确重算。

固定顺序：

1. 创建并提交本预注册；
2. 实现独立配置、模型、runner 和测试；
3. 本地完整 pytest、Python compile、配置校验、`git diff --check`；
4. 4090 GPU 专项测试；
5. N4-Q2 与 N1-Q2 的 5 epoch 阶段 A 审计并生成独立报告；
6. 只有阶段 A 无代码错误或非有限值时，运行 N1 D1/D2 smoke 并完成审计；
7. smoke 通过后运行 6 项 pilot，完成 checkpoint 重放、指标反算和全量哈希审计；
8. 只有 pilot gate 选出唯一候选时，运行最多 4 项 confirmation；
9. 生成独立结果报告并停止。

任何性能产生后不得追加消歧、扩展候选或修改 gate。若实现前发现语义仍不唯一，必须先写预注册勘误并提交。

## 14. 解释边界

阶段 A 标签只解释旧 100 倍现象，不能证明旧 Q2 性能有效。阶段 B 只验证当前冻结模型、单一 fold、单一 seed 和 surrogate pair 上的解耦方案，不能外推最终三类 unknown、完整 CSSR 论文方法或所有类别重构方法。

如果解耦版本仍失败，正式表述为：

> 已排除共享 CE 末层梯度竞争，并为 CSSR 建立独立语义适配空间；该机制仍不能稳定超过类别条件 MLS，因此停止当前 CSSR 主线。

本任务结束后不得自动转向 ARPL、GMM、多原型或最终 test。
