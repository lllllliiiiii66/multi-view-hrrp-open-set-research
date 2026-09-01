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
