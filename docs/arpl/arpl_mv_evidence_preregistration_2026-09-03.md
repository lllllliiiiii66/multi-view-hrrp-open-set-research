# ARPL/CE 多视角差异感知开集证据实验预注册

> 日期：2026-09-03  
> 阶段：AMDR P0 已结束后的 P1 surrogate OSR 机制诊断  
> 状态：运行前冻结  
> 基线提交：`1ca0d611e0d39eb1ddcdf08ea66298f572599893`

## 1. 唯一研究问题

检验当前共享编码器加特征均值是否在开集拒识时掩盖单个异常视角：保持轻量、共享、置换不变的双视角 backbone，增加固定权重的逐视角分类约束，并在 known calibration 上统一分数量纲后，逐视角异常与视角分歧能否稳定改善 surrogate unknown 检测。

本轮不是 ARPL 参数搜索，不实现 ARPL+CS，也不引入新的复杂融合网络。

## 2. 证据边界

- 只使用7个 source-known 类的奇数角开发池；
- S0/S1/S2 已经查看，只能用于 development 和规则选择；
- 只有 development gate 通过，才允许运行此前未查看的 C0–C3 confirmation；
- 最终3个 unknown、偶数角 test、AMDR、GAN/confusing samples、CSSR、COSTARR、OpenMax、attention、Set Transformer、view-specific encoder和位置编码全部禁止；
- `RESEARCH_CONTEXT.md` 不修改，旧 ARPL-lite 产物不覆盖。

## 3. 阶段 A/B：运行前审计

先用官方 commit `3ede8b38e1cfb9d70e106cc19d563453110c36ab` 的原始 `Dist.py` 和 `ARPLoss.py`，对相同 features、labels、points、radius、temperature 与 `weight_pl` 逐项比较 squared L2、dot、logits、分类损失、margin损失、总损失及 features/points/radius 梯度。float32 使用 `rtol=1e-5, atol=1e-6`，float64 使用 `rtol=1e-9, atol=1e-10`。不通过则停止训练。

随后只读取现有 S0–S2 CE_MLS/ARPL_LITE checkpoint、features和预测，恢复逐视角 logits与固定诊断分数，验证 ARPL 均值特征恒等式，并审计 feature norm 相关性与 length/padding 捷径风险。该部分为 post-hoc diagnostic，不参与最终规则选择。

## 4. 两种新训练方式

共享编码器得到 `f1`、`f2` 和 `fused=(f1+f2)/2`，同一个 head 作用于三者，已知类别预测始终取 fused logits。

`CE_VIEW_AUX`：

\[
L=CE(H(fused),y)+\frac{0.5}{2}[CE(H(f_1),y)+CE(H(f_2),y)].
\]

`ARPL_VIEW_AUX`：fused feature 使用完整 ARPL-lite 分类与margin；两个单视角只增加共享 reciprocal head 的分类项，margin不重复施加：

\[
L=L_{ARPL}(fused,y)+\frac{0.5}{2}[CE(A(f_1)/T,y)+CE(A(f_2)/T,y)].
\]

`lambda_view=0.5` 固定且不搜索，不增加 consistency loss。训练预算、优化器、初始化、pair manifest、归一化、checkpoint选择与旧基线一致。

## 5. 固定证据规则

令 `u_f/u_1/u_2` 为 fused/view1/view2 的负最大原始 logit，`d_js` 为两个单视角 softmax分布的 Jensen–Shannon divergence。所有经验CDF仅在当前模型的 known calibration 上拟合；两个视角共用合并后的同一 per-view ECDF。

- `F0_FUSED = q_f`
- `F1_WORST_VIEW = max(q_1,q_2)`
- `F2_EVIDENCE_UNION = max(q_f,q_1,q_2)`
- `F3_DISAGREEMENT_AWARE = max(q_f,q_1,q_2,q_js)`

阈值仍只由各规则的 known calibration 分数按95%已知接受率确定。`mean(q1,q2)`只作诊断。

## 6. Development gate

在 S0–S2 上运行/复用 CE_MLS、CE_VIEW_AUX、ARPL_LITE、ARPL_VIEW_AUX。规则选择只使用两个 VIEW_AUX 模型形成的6个“方法×split”单元。

F1/F2/F3 相对同模型 F0 的准入条件同时为：至少4/6个 AUROC delta 为正、平均 AUROC delta至少 `+0.02`、平均 OSCR不下降。多个合格时依次按平均AUROC提升、平均OSCR提升、规则更简单选择。没有规则合格则停止，不生成 confirmation，不设计新公式。

## 7. Conditional confirmation

仅在 gate 通过时运行：

| Split | angle fold | train-known | surrogate-unknown |
|---|---:|---|---|
| C0 | 0 | 1,2,3,4,5 | 0,6 |
| C1 | 4 | 0,2,3,4,6 | 1,5 |
| C2 | 0 | 0,1,3,5,6 | 2,4 |
| C3 | 4 | 0,1,2,4,5 | 3,6 |

固定种子为 `20260830/20260831/20260832`；每类 train-known、known calibration、surrogate unknown均500对。四种模型只比较 F0 与 development 冻结的唯一规则，不在 confirmation 重新选择。

## 8. 去留规则

分别对 CE 与 ARPL 比较完整方法 `VIEW_AUX+selected fusion` 和对应 `fused baseline+F0`。成功必须同时满足：12个单元平均AUROC至少提高2个百分点、至少8个AUROC delta为正、平均OSCR不下降、平均known Accuracy下降不超过0.5个百分点、平均FPR95恶化不超过2个百分点。

最终只允许 `common_success`、`arpl_specific_success`、`ce_specific_success`、`no_stable_gain` 四种结论。本门槛是项目去留规则，不是统计显著性。

## 9. 停止边界

完成规定审计、development、条件性confirmation与结果报告后停止。不得自动运行最终unknown、偶数角test、ARPL+CS或新增融合公式。
