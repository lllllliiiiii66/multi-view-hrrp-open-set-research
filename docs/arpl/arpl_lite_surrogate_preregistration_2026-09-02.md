# ARPL-lite 多视角 HRRP surrogate OSR 预注册

> 日期：2026-09-02  
> 阶段：AMDR P0 结束后的新 P1 可行性诊断  
> 状态：运行前冻结；不查看最终3个 unknown 或偶数角 test

## 1. 唯一研究问题

在完全相同的轻量、置换不变双视角 backbone 和训练预算下，官方代码对齐的 reciprocal-point geometry 与 adaptive margin，是否比普通 Cross-Entropy + Maximum Logit Score 提供更好的已知分类与 surrogate unknown 排序折中？

本轮是最小复现与可行性诊断，不是最终方法，也不搜索架构或开集损失。

## 2. 数据与三个固定 surrogate splits

只使用当前7个 known identities 的奇数角开发池。每个 split 映射到一个固定 odd-angle calibration fold；train-known 的剩余144个底层角度用于训练，fold内36个角度用于 known calibration；surrogate unknown 只使用对应36个 calibration 角度进行报告，其余144个角度不生成模型输入。

类别顺序固定为：

```text
0 CVN77
1 DDG-1000
2 DDG-112
3 油气轮MARVEL CRANE
4 爱达魔都号
5 迷你好望角型散货船
6 集装箱船达飞罗尔多夫级
```

三个角色划分按类别顺序相邻、互不重复地取前6类作 holdout；由于三次共只有6个 surrogate 位置，第7类固定留在 train-known。该规则在运行前确定，不根据结果挑选：

| split | angle fold | train-known | surrogate-unknown |
|---|---:|---|---|
| S0 | 1 | 2,3,4,5,6 | 0,1 |
| S1 | 2 | 0,1,4,5,6 | 2,3 |
| S2 | 3 | 0,1,2,3,6 | 4,5 |

每类 train/known-calibration/surrogate-unknown 各500个跨15°帧双视角组合；槽位按显式种子随机定向，但模型必须置换不变。最终3个 unknown 类、偶数角 test、AMDR投影和角度标签均不生成模型输入。

## 3. 输入、backbone 与公平对照

- 原始处理 bundle 中的 dB HRRP 采用 `global_scalar_zscore`；均值和标准差只由当前 split 的 train-known 组合涉及的唯一底层样本拟合；
- 输入 `[B,2,601]`；两个视角共享 `SharedHRRPEncoder1D`；
- per-view feature `[B,2,128]`，fused feature 为视角均值 `[B,128]`；
- `CE_MLS`：线性分类头、CrossEntropy，unknown score=`-max_logit`；
- `ARPL_LITE`：每类一个 reciprocal point，`logit=L2/128-dot`，`temperature=1`，`weight_pl=0.1`，margin=1，可学习 radius；unknown score同样为 `-max_logit`；
- 两者共用相同 pair manifest、类别顺序、encoder结构、优化器、batch size、epoch预算、初始化种子和数据顺序规则。

## 4. 训练预算与唯一安全回退

主配置：单一初始化种子 `20260830`；AdamW，学习率 `1e-3`、weight decay `1e-4`；batch size 64；最多100 epochs；known calibration Accuracy 优先、Macro-F1 次级选择 checkpoint；patience 15；无增强、无 scheduler。

唯一安全回退预注册为学习率 `3e-4`，其他项目完全不变。只有出现 NaN/Inf 或优化器数值异常时才能触发，并对同一 split 的 CE_MLS 与 ARPL_LITE 一起重跑以保持公平；不得依据 surrogate AUROC、OSCR 或最终未知结果触发。

## 5. 开发、阈值和评价隔离

- surrogate unknown 不参与训练、归一化、checkpoint选择、early stopping、阈值或任何分布拟合；
- 最佳 epoch 只由 known calibration Accuracy/Macro-F1确定；
- threshold只由最佳 checkpoint的 known calibration unknown scores按95%已知接受率确定；
- surrogate unknown只在最佳 checkpoint冻结后推理一次并报告；
- 所有 unknown score 均为“越大越未知”。

每个 split/method 报告 known Accuracy、Macro-F1、AUROC、OSCR、FPR95、KCCR、URR、调和平均和K+1 Macro-F1，并聚合三个 split 的均值和总体标准差。

## 6. 执行顺序和停止规则

1. CPU shape/手算/梯度和协议单元测试；
2. 每类50对、最多3 epochs的 S0 diagnostic smoke；
3. smoke通过后，在 Merlin 运行S0–S2正式500对配置；
4. 保存并反算全部逐样本结果；
5. 完成结果报告后停止。

本轮禁止 GAN/confusing samples、CSSR、COSTARR、attention、Set Transformer、view-specific encoder、per-view reciprocal fusion、额外未知损失、大范围超参数搜索和最终 open-set test。

是否值得继续的判断只作为下一轮提案：若 ARPL_LITE 相比 CE_MLS 在多数预注册 split 上保持相近已知分类且带来稳定开集指标增益，可建议研究完整 ARPL+CS 或 per-view reciprocal evidence；否则停止扩展。不得在本轮结果后自动实现。
