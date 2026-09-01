# P0 AMDR local-KNN 图 lambda2 事后扩展诊断

## 目的与证据边界

本诊断回答：在 fold 0、fixed-initial D、local-KNN Gaussian 图和 `lambda_manifold=1` 不变时，把 `lambda_sparse` 从既有上界 10 继续增大，known calibration 分类效果是否提高，以及权重是否变得更稀疏。

这是查看过 fold 0 test 结果后，由用户明确要求开展的事后诊断。参数仍只由 known calibration 选择，搜索阶段不生成 test 特征、不计算 test 指标；但由于扩展决策发生在查看 test 之后，结果不能作为正式确认性证据，也不能据此反复调 fold 0 test。

## 冻结配置

- 基础配置：`pilot_fold0_amplitude_pruned_fixed_d_local_knn_v1.yaml`
- 固定：`lambda_manifold=1`、`l21_reweighting=fixed_initial`、local KNN `K=10`、分类 KNN `K=3`
- 搜索：`lambda_sparse=[10,15,20,30,50,100]`
- 选择：known calibration Accuracy 优先，Macro-F1 次优，再偏向较小参数
- 若最优值仍在 100 的边界，本轮停止，不继续自动扩大
- 选择完成前不生成 test 特征；若新值优于 10，仅固定该值做一次诊断 test

## 解释限制

当前 `fixed_initial` 只固定使用初始行权重矩阵，并不在每轮根据当前权重重新强化行稀疏。因此增大 `lambda_sparse` 首先会加强整体收缩，不保证产生严格为零的特征行。除了分类指标，还必须报告权重 Frobenius 范数、训练后剪枝行数和行范数分布，才能判断稀疏约束是否实际生效。

## 结果

| `lambda_sparse` | Calibration Accuracy | Macro-F1 | 权重范数 | 剪枝行数 |
|---:|---:|---:|---:|---:|
| 10 | 67.06% | 66.93% | 15.224 | 0 |
| 15 | 65.94% | 65.77% | 12.794 | 0 |
| 20 | 67.43% | 67.36% | 11.386 | 0 |
| 30 | 67.23% | 67.25% | 9.461 | 0 |
| 50 | **67.49%** | **67.38%** | 7.635 | 0 |
| 100 | 66.20% | 66.15% | 5.549 | 3 |

known calibration 选中非边界值 `lambda_sparse=50`。相对 `10`，calibration Accuracy 只提高 0.43 个百分点；权重整体持续缩小，但 `50` 下 1202 行仍无一行达到既定剪枝阈值。到 `100` 才剪掉 3 行，同时准确率下降，说明增大参数主要产生整体收缩，并未形成有效的强行稀疏选择。

固定 `lambda_sparse=50` 的一次诊断 test 结果：known Accuracy 61.14%、known Macro-F1 60.47%、AUROC 30.70%、OSCR 20.78%、未知拒识率 3.20%。相对先前 `lambda_sparse=10`，known Accuracy 提高 2.40 个百分点，AUROC 提高 2.92 个百分点，但开集表现仍明显不可用。

7 个已知类中 6 类准确率提高，主要是集装箱船达飞罗尔多夫级 `+6.8` 个百分点和 CVN77 `+4.8` 个百分点；DDG-112 下降 `4.0` 个百分点。该变化不是所有类别一致改善。

## 结论

本轮确认更强的正则在当前 fold 0 诊断中能改善闭集结果，`50` 优于 `10`；但不能据此得出“因为输入是 1202 维，所以 lambda2 必须更大”的一般结论。它没有让 fixed-initial D 产生实质特征剪枝，也没有解决未知分离问题。

由于这是查看 fold 0 test 后开展的事后扩展，停止继续追调该 fold。若后续保留这一方向，应在未查看的新 fold 上预先固定少量对照值验证 `10` 与 `50` 是否稳定，而不是继续扩大网格。

## 审计

- 参数搜索阶段记录 `test_features_materialized=false`、`test_metrics_used=false`。
- 搜索与固定测试使用同一配对清单，SHA-256 为 `4e16ce9e713b264ac481d01659c292f13db606c3aea6f2bbb4f77050ade0c171`。
- 固定测试 10 个产物哈希全部一致；8500 行逐样本预测反算指标与保存结果最大差异为 0。
- 结果回传压缩包 SHA-256 为 `f4a39b21fcb49dc3538a7635e7c520cdde427db4a5c0e4a93a3eacf8f777acf8`。
- 本文不并入 `RESEARCH_CONTEXT.md`，等待用户确认。
