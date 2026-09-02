# ARPL/CE 多视角差异感知开集证据实验结果

> 日期：2026-09-03  
> 阶段：P1 surrogate OSR 机制诊断  
> 最终结论：`no_stable_gain`  
> 停止位置：development gate失败；未运行confirmation

## 1. 结论

当前逐视角证据融合没有形成稳定信号。F1、F2均使6个 development 单元的平均AUROC下降；F3平均仅提高0.06个百分点，且只有2/6单元为正，未达到预注册的“平均至少提高2个百分点、至少4/6为正、平均OSCR不下降”。因此没有冻结任何fusion rule，C0–C3 confirmation被协议阻止且未生成。

逐视角辅助训练本身呈现不同结果：CE_VIEW_AUX平均AUROC比CE_MLS下降4.55个百分点；ARPL_VIEW_AUX平均比ARPL_LITE提高8.51个百分点，但主要由存在length/padding捷径风险的S2贡献，且没有新split多种子确认，不能写成ARPL特定成功。

最终判断为 `no_stable_gain`：不建议将当前差异感知规则作为后续主方法，也不继续增加权重、温度、MLP或新score。

## 2. ARPL官方差分测试

通过。官方 commit `3ede8b38e1cfb9d70e106cc19d563453110c36ab` 与项目实现，在相同 features、labels、points、radius、temperature 和 `weight_pl` 下完成：

- squared L2、dot、logits逐项等价；
- classification、margin、total loss逐项等价；
- features、reciprocal points、radius梯度逐项等价；
- float32使用 `rtol=1e-5, atol=1e-6`，float64使用 `rtol=1e-9, atol=1e-10`。

因此：**ARPL loss 数学内核已经与官方实现完成逐前向、逐梯度等价验证；后续差异属于 HRRP backbone、训练协议和多视角适配问题，而不是 ARPL 核心公式错误。**

## 3. 现有checkpoint post-hoc审计

旧S0–S2未重新训练。平均AUROC最高的非fused单一证据为“两个视角预测类别是否分歧”：CE 64.53%、ARPL 63.37%，仅比各自fused score高0.47/0.58个百分点，且分split方向不一致。JS divergence主要在S2有效，S0/S1约为随机水平。

ARPL均值特征logit恒等式在全部旧特征上通过，float32最大绝对残差 `1.04e-5`。详细逐项结果见 `docs/arpl/arpl_mv_evidence_audit_2026-09-03.md`。

length/padding只用train-known元数据拟合，S0/S1/S2 AUROC为50%/50%/75%。S2包含训练支持中缺失的profile length，因此其显著提升可能部分依赖长度捷径；该诊断没有进入训练、ECDF或规则选择。

## 4. Development训练结果

### 4.1 F0下的逐视角辅助训练

三组均值：

| 模型 | Known Acc. | Known Macro-F1 | AUROC | OSCR | FPR95↓ | URR | K+1 Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CE_MLS | 98.67% | 98.66% | 64.04% | 63.67% | 82.25% | 20.93% | 75.56% |
| CE_VIEW_AUX | 99.56% | 99.56% | 59.50% | 59.34% | 91.84% | 33.87% | 80.77% |
| ARPL_LITE | 99.20% | 99.20% | 62.77% | 62.64% | 80.63% | 21.47% | 76.98% |
| ARPL_VIEW_AUX | 99.77% | 99.77% | 71.28% | 71.21% | 69.85% | 45.60% | 83.67% |

F0下的paired AUROC变化：

| Head | S0 | S1 | S2 | 三组均值 |
|---|---:|---:|---:|---:|
| CE_VIEW_AUX − CE_MLS | −2.56 pp | +1.32 pp | −12.40 pp | **−4.55 pp** |
| ARPL_VIEW_AUX − ARPL_LITE | −0.97 pp | +2.60 pp | +23.91 pp | **+8.51 pp** |

解释边界：CE逐视角辅助训练没有稳定收益；ARPL出现development-only正向信号，但只有2/3 split为正，S2贡献过大且存在length/padding风险，没有confirmation证据。

### 4.2 Evidence fusion gate

规则选择只使用 CE_VIEW_AUX/ARPL_VIEW_AUX × S0/S1/S2 共6个单元：

| 候选规则 | AUROC正向单元 | 平均AUROC delta | 平均OSCR delta | 合格 |
|---|---:|---:|---:|---|
| F1_WORST_VIEW | 0/6 | −5.58 pp | −5.59 pp | 否 |
| F2_EVIDENCE_UNION | 1/6 | −1.79 pp | −1.80 pp | 否 |
| F3_DISAGREEMENT_AWARE | 2/6 | +0.06 pp | +0.02 pp | 否 |

F3的唯一明显提升来自CE_VIEW_AUX的S2（AUROC +7.81 pp）；CE的S0/S1与ARPL的S0/S1均下降，ARPL的S2仅+0.04 pp。这不是跨head、跨split的稳定信号。

## 5. 三个问题的回答

1. **逐视角辅助训练是否有效？**  
   CE：否，平均AUROC下降4.55 pp。ARPL：development上平均提高8.51 pp，但高度依赖S2，尚未确认，不能判成功。

2. **证据融合是否有效？**  
   否。F1–F3没有任何规则通过development gate。

3. **完整方法是否有效？**  
   无法形成预注册完整方法，因为没有可冻结的fusion rule；按协议归为 `no_stable_gain`。

## 6. Confirmation与最终测试

- C0–C3：**未运行**；
- 三个confirmation seeds：**未运行**；
- common/ARPL-specific/CE-specific确认性判据：没有进入；
- 最终3个unknown：未使用；
- 偶数角test：未生成、未使用；
- ARPL+CS、CSSR、COSTARR、OpenMax、新融合公式：未实现。

## 7. 数值、复现与产物审计

- 12个development模型均使用主学习率 `1e-3`，没有数值回退；
- 未出现NaN/Inf；同一split四模型使用相同pair manifest和标签顺序；
- 重跑的6个CE_MLS/ARPL_LITE checkpoint状态哈希与旧正式checkpoint逐一完全一致；
- ARPL/ARPL_VIEW_AUX最佳checkpoint radius范围为0.720–1.829，active-margin比例为8.32%–100%，完整轨迹已保存；
- 三组每种模型各3500条评价预测，共42,000条，F0–F3全部指标零误差反算；
- development根目录101个产物哈希全部复核一致；
- 运行提交：`33ef66984b08624a2bd1e3ca0d27957d097fee37`；
- Merlin：8 CPU配额、32 GiB内存；development用时1216.74秒；
- 正式产物：`artifacts/arpl/arpl_mv_evidence_surrogate_v1/development_20260903_33ef669`；
- post-hoc产物：`artifacts/arpl/arpl_mv_evidence_surrogate_v1/posthoc_audit_20260903_1ca0d61`。

验证：本地完整pytest 282项通过；Merlin专项13项通过；Python编译、配置校验和 `git diff --check` 通过；S0 diagnostic smoke通过；旧产物65个哈希、development 101个哈希及全部预测反算通过。

## 8. 下一步建议与停止边界

本任务到此停止。不建议继续围绕 F1–F3 增加权重、温度或学习型融合器。若继续研究，应先回到方法筛选：选择另一种有明确开集机制与可靠出处、且能在相同 surrogate 协议下公平比较的方法；在新方案预注册前不运行最终unknown或偶数角test。

`RESEARCH_CONTEXT.md` 本轮未修改。若用户后续确认回写，建议仅加入：ARPL数学内核已完成官方逐梯度等价验证；当前多视角差异感知证据在development gate失败，confirmation未运行，结论为 `no_stable_gain`。
