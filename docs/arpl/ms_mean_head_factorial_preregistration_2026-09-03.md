# 多尺度 HRRP 骨干 × CE/ARPL 全因子简化确认实验预注册

> 日期：2026-09-03
>
> 阶段：P3 独立简化确认
>
> 实验：`ms_mean_head_factorial_surrogate_v1`
>
> 基线：`ccb30e18b4aa9e78e136ba330dddefb11aab11ae`
>
> 状态：运行前冻结；任何 smoke 或正式性能结果产生前创建

## 1. 唯一研究目的

本轮不继续优化 MV-RPFormer，也不修改或重启 M3–M6。只用一个严格的 2×2 因子设计回答：

1. 多尺度 HRRP 编码器是否相对浅层编码器产生跨新身份组合的稳定增益；
2. 在完全相同的多尺度编码器和算术均值融合下，ARPL 是否优于 CE。

上一轮 M3 Set Transformer、M4 分层逐视角 ARPL、M5/M6 伪未知拒判器路线的停止结论保持不变。上一轮 M2 只作为产生本轮问题的发现性证据，不能据此直接进入最终测试。

## 2. 四个冻结方法

| 方法 | 骨干 | 融合 | Head | Unknown score |
|---|---|---|---|---|
| R0_SHALLOW_MEAN_CE | `SharedHRRPEncoder1D` | 算术均值 | 线性 CE | 负最大原始 logit |
| R1_SHALLOW_MEAN_ARPL | `SharedHRRPEncoder1D` | 算术均值 | 单个 global ARPL | 负最大原始 ARPL logit |
| R2_MS_MEAN_CE | 冻结的 `HRRPMultiScaleResNet1D` | 算术均值 | 线性 CE | 负最大原始 logit |
| R3_MS_MEAN_ARPL | 冻结的 `HRRPMultiScaleResNet1D` | 算术均值 | 单个 global ARPL | 负最大原始 ARPL logit |

R3 必须与上一轮 M2 的 encoder、均值融合、global ARPL 和损失语义完全一致。多尺度网络的 stem、3/7/15 分支、残差、池化、projection、dropout 和 128 维输出均不得改变。

不得创建 SAB、PMA、attention、view-level head、reject token、learned rejector、mismatch/mixup 伪未知、confusing samples、GAN、新手工多视角分数或新网络结构。

同一 identity pair/fold/seed 中，R0/R1 的浅层 encoder 初始 state 必须相同，R2/R3 的多尺度 encoder 初始 state 必须相同。四个方法共享 pair manifest、样本顺序和 DataLoader 顺序；保存并审计初始 encoder state SHA-256。

## 3. 新 surrogate identity 设计

此前查看过的 `[0,1] [2,3] [4,5] [0,6] [1,5] [2,4] [3,6]` 全部禁止复用。正式冻结以下七组新 pair：

| Pair | Surrogate unknown | Train known |
|---|---|---|
| N0 | [0,2] | [1,3,4,5,6] |
| N1 | [2,5] | [0,1,3,4,6] |
| N2 | [3,5] | [0,1,2,4,6] |
| N3 | [1,3] | [0,2,4,5,6] |
| N4 | [1,6] | [0,2,3,4,5] |
| N5 | [4,6] | [0,1,2,3,5] |
| N6 | [0,4] | [1,2,3,5,6] |

七个类别各恰好出现两次。每个 pair 与 angle fold `0`、`4` 交叉，并对每个 pair/fold 使用种子 `20260830/20260831/20260832`：共 42 个实验单元、168 个正式方法任务。不得按结果删除或替换 pair。

## 4. 数据与训练协议

- 数据仅来自 7 个 source-known 类的奇数角开发池；每个视角来自不同 15° 帧，槽位规则为 `randomized_seeded`。
- pair 抽样随机流固定到“实验 × angle fold”，identity pair 只过滤 5-known/2-surrogate 类别角色；因此同一 fold 内、不同 identity pair 共享的源类别使用相同底层 pair 抽样，避免把 identity 差异与另一套抽样随机性混在一起。
- 每类 train-known、known calibration、surrogate unknown 各 500 pairs；smoke 为每类 10 pairs。
- 先切底层 HRRP，再分别造 pair；train 与评价底层样本不得重叠。
- `global_scalar_zscore` 只用当前 split 的唯一 train-known 底层 HRRP 拟合。
- surrogate unknown 不进入训练、归一化、阈值、checkpoint 或模型选择。
- 最终 3 个 unknown 类和偶数角 test 不得生成或运行。
- AdamW，学习率 `3e-4`，weight decay `1e-4`，batch size 64，5 epoch 线性 warmup 后 cosine decay，总计 100 epochs。
- 四种方法从第 1 到第 100 epoch 只训练各自 CE 或 ARPL 表示损失；无拒判 warmup 阶段、无数据增强、无 early stopping、无性能回退。
- 正式模型固定为 epoch 100，不根据 known 或 surrogate 指标选 epoch。
- ARPL 固定 `temperature=1.0`、`weight_pl=0.1`、`margin=1.0`、每类 1 个 reciprocal point、初始化标准差 0.1、radius 初值 0；head 参数不得搜索。

smoke 固定为 N0/fold0/seed20260830 的四种方法、每类 10 pairs、1 epoch。它只验证链路、有限数、隔离、交换不变、指标反算和产物封口；不设性能 gate。审计通过后不依据 smoke 数值修改协议，直接运行全部 168 项正式任务。

## 5. 评价与完整报告

known 预测为融合后 global logits 的 argmax。所有方法的 unknown score 都是负最大原始 logit，方向统一为越大越未知；阈值只由本模型 known calibration 分数按 95% 已知接受率确定。

每个方法、每个单元必须输出并可从逐样本预测精确反算：Known Accuracy、Known Macro-F1、AUROC、OSCR、FPR95、KCCR、URR、KCCR/URR 调和平均和 K+1 Macro-F1。

还必须报告每个 surrogate identity 的 AUROC/URR、其错误吸收的 known 类、42 个完整单元以及 pair/fold/seed 分组结果，不只给总均值。

## 6. Length/padding 诊断

每个 pair/fold 保存 original profile length、left/right padding、surrogate length 是否超出当前 train-known 原始长度的闭区间，以及 length-only AUROC。

预先定义：只有当两个 fold 中所有 surrogate 原始长度均处在各自 train-known 长度闭区间内时，该 identity pair 为 `length-safe`；任一 fold 越界即为 `length-risk`。完整集、safe 子集和 risk 子集同时报告，但该标签不进入 gate、不删除 pair、不改变输入或方法选择。

## 7. 预注册比较与统计单位

四个主比较：

- A：R3−R1，ARPL 条件下的 backbone 贡献；
- B：R2−R0，CE 条件下的 backbone 贡献；
- C：R3−R2，多尺度 backbone 上的 ARPL 贡献；
- D：R1−R0，浅层 backbone 上的 ARPL 贡献。

交互项为 `(R3−R2)−(R1−R0)`，只作机制解释。

主统计单位是 identity pair：每个 pair 先对 2 folds × 3 seeds 的 delta 求平均，再聚合 7 个 pair。另保留 42 个 unit-level delta、pair 内 seed 标准差和 fold 差异。以七个 pair-level delta 做固定种子 `20260903`、10000 次 percentile paired bootstrap 95% 区间；500 个组合样本不作为独立统计重复。

## 8. 冻结门槛

比较 A 和 B 分别判断 backbone 成功，均须同时满足：平均 AUROC delta `>= +3.0 pp`、至少 `6/7` pair 为正、平均 OSCR 不下降、平均 Known Accuracy 下降不超过 `0.5 pp`、平均 FPR95 恶化不超过 `2.0 pp`。

据 A/B 输出唯一标签：两者通过为 `backbone_general_success`；仅 A 为 `backbone_arpl_only`；仅 B 为 `backbone_ce_only`；均不通过为 `no_backbone_gain`。

多尺度 head 比较 C：若 R3−R2 同时满足平均 AUROC `>= +1.0 pp`、至少 `5/7` pair 为正、平均 OSCR 不下降、Known Accuracy 下降不超过 `0.5 pp`、FPR95 恶化不超过 `2.0 pp`，则 `ARPL_PREFERRED`。对 R2−R3 完全对称判断 `CE_PREFERRED`；两者均不满足则 `HEAD_INDETERMINATE`，不得因此调 head 参数。

候选建议固定为：

| Backbone 标签 | Head 标签 | 建议候选 |
|---|---|---|
| backbone_general_success | ARPL_PREFERRED | R3 |
| backbone_general_success | CE_PREFERRED 或 HEAD_INDETERMINATE | R2 |
| backbone_arpl_only | 任意 | R3 |
| backbone_ce_only | 任意 | R2 |
| no_backbone_gain | 任意 | none |

无论结果如何，`final_unknown_test_authorized=false`。本任务只能建议是否另行预注册最终测试，不能自动运行。

## 9. 产物、运行环境与停止边界

每项保存 resolved config、pair manifest/hash、初始化 hash、epoch-100 checkpoint、训练日志、特征/logits/scores、逐样本预测、九项指标、length/padding 诊断、环境、源码和产物哈希。聚合保存四个主比较、交互项、全部层级汇总、bootstrap 区间、吸收矩阵和最终机器判定。

正式运行固定使用 4×RTX 4090；每卡 4 个独立任务、峰值 16 并发，不使用 DDP、AMP、TF32 或 `torch.compile`。四种方法到物理 GPU 的映射按实验单元轮换，使每卡各承担 42 项且任何方法不固定绑定某一张卡。每任务 4 个 intra-op、1 个 inter-op 线程并启用确定性算法。

实现完成后依次通过完整 pytest、GPU 专项测试、Python compile、配置校验、`git diff --check` 和固定 smoke。随后一次性运行 168 项、聚合、逐样本反算和哈希审计，并写入独立结果报告。`RESEARCH_CONTEXT.md` 不修改，所有旧产物不覆盖。完成结果报告后停止。

实现审计记录：上述按 fold 共享 pair 抽样随机流的约束是在正式 GPU smoke 前由公平性静态审计补充；此前仅完成过一次本地链路预检，其性能数值作废且不进入任何判断，本补充不依据该数值。
