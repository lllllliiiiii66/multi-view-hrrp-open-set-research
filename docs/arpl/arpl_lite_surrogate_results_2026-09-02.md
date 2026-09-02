# ARPL-lite 多视角 HRRP surrogate OSR 结果

> 预注册日期：2026-09-02  
> 完成日期：2026-09-03  
> 阶段：AMDR P0 结束后的新 P1 可行性诊断  
> 结论：ARPL-lite 未产生稳定开集收益，本轮停止，不进入 ARPL+CS 或逐视角 reciprocal evidence

## 1. 结论

在相同双视角 backbone、训练预算、初始化和 pair manifest 下，ARPL-lite 将三个 surrogate split 的平均 known Accuracy 从 98.67% 提高到 99.20%，但平均 AUROC 从 64.06% 降至 62.79%，平均 OSCR 从 63.69% 降至 62.66%。S1 明显改善，S0 指标混合，S2 明显退化，因此不能认为 reciprocal-point geometry 提供了稳定的未知排序增益。

按照运行前冻结的停止规则，本轮不建议继续完整 ARPL+confusing samples，也不建议直接扩展逐视角 reciprocal fusion。ARPL-lite 保留为已审计的参考实现，不作为当前主要开集方法基础。

该结论仅来自7个 source-known 类内部的3组 surrogate OSR 诊断；最终3个 unknown 类和偶数角 test 均未运行，不能写成最终 open-set 结果。

## 2. 已冻结且实际执行的设计

- 三组固定的 `5 train-known + 2 surrogate-unknown` 类别划分：S0/S1/S2；
- 仅使用奇数角开发池，每类每个数据角色500个跨15°帧双视角组合；
- 输入为 `[B,2,601]`，共享 `SharedHRRPEncoder1D`，两视角特征取均值；
- 公平比较 `CE_MLS` 与无 confusing samples 的 `ARPL_LITE`；
- 单一种子 `20260830`，AdamW、学习率 `1e-3`、batch size 64、最多100轮、patience 15；
- checkpoint 仅由 known calibration Accuracy、其次 Macro-F1 选择；
- 阈值仅由 known calibration 按95%已知接受率确定；
- surrogate unknown 仅在 checkpoint 冻结后推理，不进入训练、归一化、早停、阈值或分布拟合；
- 最终3个 unknown、偶数角 test、AMDR、GAN/confusing samples、CSSR、COSTARR 和逐视角 reciprocal fusion 均未使用。

## 3. 正式结果

### 3.1 分 split 结果

| Split | 方法 | Known Acc. | Known Macro-F1 | AUROC | OSCR | FPR95↓ | URR | KCCR/URR调和 | K+1 Macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | CE_MLS | 99.32% | 99.32% | 52.72% | 52.53% | 94.24% | 24.00% | 38.29% | 77.33% |
| S0 | ARPL_LITE | 99.96% | 99.96% | 54.85% | 54.84% | 84.12% | 20.10% | 33.18% | 76.25% |
| S1 | CE_MLS | 98.68% | 98.68% | 58.89% | 58.71% | 81.72% | 1.70% | 3.34% | 68.62% |
| S1 | ARPL_LITE | 99.00% | 99.00% | 65.70% | 65.56% | 69.76% | 24.60% | 39.06% | 77.59% |
| S2 | CE_MLS | 98.00% | 97.99% | 80.58% | 79.84% | 70.68% | 37.20% | 53.34% | 80.77% |
| S2 | ARPL_LITE | 98.64% | 98.64% | 67.81% | 67.59% | 87.88% | 20.10% | 33.15% | 77.23% |

### 3.2 三组均值

| 方法 | Known Acc. | Known Macro-F1 | AUROC | OSCR | FPR95↓ | URR | KCCR/URR调和 | K+1 Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CE_MLS | 98.67% | 98.66% | 64.06% | 63.69% | 82.21% | 20.97% | 31.66% | 75.57% |
| ARPL_LITE | 99.20% | 99.20% | 62.79% | 62.66% | 80.59% | 21.60% | 35.13% | 77.03% |
| ARPL−CE | +0.53 pp | +0.53 pp | −1.27 pp | −1.03 pp | −1.63 pp | +0.63 pp | +3.47 pp | +1.45 pp |

均值中的部分指标略有改善主要由S1贡献，并不构成跨 split 的稳定优势：ARPL 的调和分数在S0下降5.11 pp、S1上升35.72 pp、S2下降20.20 pp；AUROC在S0/S1提高2.13/6.82 pp，在S2下降12.77 pp。

## 4. 数值与协议审计

- 三组均使用主学习率 `1e-3`，没有触发数值回退；
- 未出现 NaN/Inf；视角交换后 fused feature 和 logits 的最大绝对差均为0；
- ARPL 最佳轮次为 S0/S1/S2：53/18/34，停止轮次为68/33/49；
- 最终 radius 为1.639/0.720/1.359；训练样本 true-class reciprocal distance 均值为0.516/0.460/0.484；
- 每组选择2500个 train-known、2500个 known calibration、1000个 surrogate-unknown 组合；训练与评价底层样本重叠为0；
- 最终 unknown 组合数为0，偶数角组合数为0，test pair和test feature均未生成；
- CE和ARPL逐组使用相同 pair manifest、类别顺序、backbone初始化、优化器、预算、初始化种子和数据顺序；
- 三组两种方法共21,000条逐样本预测，重新计算指标的最大绝对误差为0；
- 65个根产物哈希及每个方法目录8个哈希均重新核验一致。

## 5. 运行环境与产物

- 代码提交：`4693061bbeae3dc8f2ef856ecfeb18ac9f1ea6ca`；
- Merlin：8 CPU配额、32 GiB内存，Python 3.12.13，PyTorch 2.13.0+cpu；
- 正式训练用时497.43秒；
- 正式产物目录：`artifacts/arpl/arpl_lite_surrogate_osr_v1/full_surrogate_20260902_4693061`；
- pair manifest SHA-256：S0 `44c4c9ce...a710`，S1 `26aa9c13...d44e`，S2 `94757062...d7bd`；完整哈希保存在各产物目录；
- 环境记录中的 `git.dirty=true` 仅由未跟踪的 `artifacts/` 目录造成，运行代码本身位于上述固定提交。

运行前后实际检查：本地完整测试269项通过；Merlin新增ARPL专项测试11项通过；Python编译检查和 `git diff --check` 通过；S0 diagnostic smoke通过；正式S0–S2运行及独立产物反算通过。

## 6. 停止边界与下一步提案

本轮到此停止，不自动运行最终3个 unknown，不实现 ARPL+CS、逐视角 reciprocal fusion、CSSR、COSTARR 或新的融合结构。

若继续研究，建议先基于这次结果重新筛选下一种公认开集方法，预注册后仍使用相同 surrogate 协议与 CE_MLS 比较；不建议继续围绕当前 ARPL-lite 调参。`RESEARCH_CONTEXT.md` 本轮未修改，建议后续经用户确认后仅回写两点：AMDR P0 已 reject；ARPL-lite 在 surrogate OSR 上没有稳定优于 CE_MLS。
