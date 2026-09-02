# ARPL/CE 现有 checkpoint 多视角证据审计

> 日期：2026-09-03  
> 性质：S0–S2 post-hoc diagnostic；不作为确认性证据，不选择最终规则

## 1. 结论

ARPL loss 数学内核已经与官方实现完成逐前向、逐梯度等价验证；后续差异属于 HRRP backbone、训练协议和多视角适配问题，而不是 ARPL 核心公式错误。

旧 CE_MLS/ARPL_LITE checkpoint 中确实存在逐视角分歧信号，但没有一个单独证据跨 S0–S2 稳定优于 fused maximum-logit。平均AUROC最高的非 fused 诊断量是“两个视角预测类别是否不同”：CE为64.53%，ARPL为63.37%，仅比各自 fused score 高0.47/0.58个百分点，而且分 split 增减方向不一致。JS divergence 在S2较强，但S0/S1接近随机水平。

因此本审计只能支持继续执行预注册的固定 VIEW_AUX 与 F0–F3 development gate，不能据此直接指定某个 fusion rule。

## 2. 官方 differential equivalence

官方快照固定为 commit `3ede8b38e1cfb9d70e106cc19d563453110c36ab`：

- `Dist.py` SHA-256：`a05fc01c...4dc62`；
- `ARPLoss.py` SHA-256：`6dec41f0...27ec`；
- 原文件按MIT许可证保存在测试 fixture；仅将官方硬编码 `.cuda()` 在CPU测试中替换为同一tensor的no-op，未改变公式。

在完全相同的随机 features、labels、reciprocal points、radius、temperature 和 `weight_pl` 下，float32与float64均逐项通过：squared L2、dot、logits、classification loss、margin loss、total loss，以及 features、reciprocal points、radius 的梯度。容差分别为 `rtol=1e-5, atol=1e-6` 和 `rtol=1e-9, atol=1e-10`。

## 3. 旧 checkpoint 逐视角诊断

下表为三组平均AUROC：

| Unknown score | CE_MLS | ARPL_LITE |
|---|---:|---:|
| fused maximum-logit | 64.06% | 62.79% |
| worst-view maximum-logit | 56.72% | 53.85% |
| mean-view maximum-logit | 58.76% | 56.75% |
| view score gap | 43.56% | 42.62% |
| view prediction disagreement | **64.53%** | **63.37%** |
| JS divergence | 59.92% | 58.69% |
| feature cosine distance | 50.95% | 50.10% |
| feature squared-L2/维数 | 47.85% | 46.11% |
| fused feature norm | 49.03% | 45.44% |

主要不稳定性：

- CE prediction disagreement 的S0/S1/S2 AUROC为61.01%/58.83%/73.76%，对应 fused 为52.72%/58.89%/80.58%；
- ARPL prediction disagreement 为63.78%/60.97%/65.37%，对应 fused 为54.85%/65.70%/67.81%；
- JS在CE/ARPL的S2达到79.05%/72.80%，但S0/S1约为50%；
- score_fused 与 feature norm 在部分 split/角色存在较强负相关，说明 maximum-logit 受特征尺度影响，但相关强度和替代检测能力并不稳定。

## 4. ARPL 均值特征恒等式

对旧 S0/S1/S2 的全部 known calibration 和 surrogate unknown 特征验证：

\[
\frac{logits(f_1)+logits(f_2)}{2}
=logits\!\left(\frac{f_1+f_2}{2}\right)
+\frac{\lVert f_1-f_2\rVert^2}{4d}.
\]

附加项在类别间相同。float32最大绝对残差为 `1.04e-5`，在 `rtol=1e-5, atol=1e-6` 下全部通过。该恒等式说明ARPL的单视角平均logit相对fused logit只多出一个类别无关项，但 `max`、ECDF与跨视角取并集仍可能产生不同拒识排序。

## 5. Length/padding shortcut audit

只用各 split 的 train-known 元数据拟合“到最近训练支持值的标准化距离”，没有使用 surrogate unknown 调参。原始长度、左右padding及三者组合的AUROC完全相同：

| Split | AUROC |
|---|---:|
| S0 | 50.00% |
| S1 | 50.00% |
| S2 | 75.00% |
| 平均 | 58.33% |

S2 surrogate 中包含长度501的类别，而该 split 的train-known长度支持没有501，因此存在明显捷径风险。该分数只用于解释，不进入训练、ECDF或融合规则。

## 6. 协议和产物审计

- 旧根目录65个文件哈希全部复核一致；
- S0/S1/S2 pair manifest SHA-256与原正式实验一致；
- 只读取旧checkpoint、features和predictions，没有重新训练；
- 最终3个unknown与偶数角test均未使用；
- 诊断产物位于 `artifacts/arpl/arpl_mv_evidence_surrogate_v1/posthoc_audit_20260903_1ca0d61`。
