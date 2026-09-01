# P0 AMDR lambda2=50 固定 D / 动态 D 受控诊断

## 研究问题

在 fold 0、local-KNN Gaussian 图、`lambda_manifold=1`、`lambda_sparse=50` 和相同数据配对下，只把 `D` 从训练前固定改为每轮更新，能否让行稀疏机制真正发挥作用并改善 known calibration 分类？

本实验属于查看过 fold 0 test 后的事后 P0 诊断，不是正式性能证据。搜索阶段不生成 test 特征、不计算 test 指标。

## 冻结对照

- 固定项：数据、配对、初始化、输入变换、local KNN 图、迭代上限、停止条件、分类 KNN 和两个 lambda。
- 唯一变化：`l21_reweighting=fixed_initial` 对比 `update_each_iteration`。
- 两种策略均只在 3500 条 known calibration 对上评价。
- 除 Accuracy、Macro-F1 外，报告收敛状态、迭代数、权重范数、剪枝行数和行范数分布。

## 测试门槛

动态 `D` 只有同时满足以下条件才运行一次固定 test：

1. 在配置上限内正常收敛；
2. calibration Accuracy 至少比固定 `D` 高 `0.005`，即 0.5 个百分点；
3. calibration Macro-F1 不低于固定 `D`。

若任一条件不满足，停止该分支，不生成动态 `D` 的 test 特征。无论是否通过，均不根据 fold 0 test 继续调整参数。
