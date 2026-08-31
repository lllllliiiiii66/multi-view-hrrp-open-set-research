# P0 AMDR 固定 D 与动态 D 参数选择预注册

> 日期：2026-08-31
> 范围：新十类 HRRP 数据、现有确定性 7 known / 3 unknown 角色、fold 0 pilot。只比较固定 `D` 和动态 `D` 的闭集已知类表示；未知类与偶数角测试特征不参与参数选择。

## 1. 研究问题

在现有严格奇偶角划分下，分别为固定 `D` 和动态更新 `D` 选择合适的 `lambda_manifold/lambda_sparse` 后，两种 AMDR 表示的已知类泛化效果如何？

## 2. 固定不变项

- 基础配置：`configs/amdr/pilot_fold0_amplitude_pruned_v1.yaml`；
- 每类每个数据角色 500 对：known train 3,500、known calibration 3,500、固定 test 5,000；
- 峰值相对幅度 `/20`，601 维，随机但固定种子的视角槽位，训练后按 `1e-5` 行平方范数阈值剪枝；
- AMDR 初始化种子 `20260830`、最多 300 次、相对变化阈值 `3e-5`；
- KNN 固定 `K=3`，平方欧氏距离；本轮不同时选择 K、不加 PCA、不屏蔽同底层关系边。

## 3. 参数选择

两种 `D` 策略分别独立选择参数：

1. `fixed_initial`：按固定种子生成初始 `W`，只计算一次 `D`；
2. `update_each_iteration`：每轮根据上一轮 `W` 重新计算 `D`。

初始网格：

- `lambda_manifold ∈ {0.01, 0.1, 1, 10}`；
- `lambda_sparse ∈ {0.01, 0.1, 1, 10}`；
- 每种策略 16 组，共 32 组。

排序规则依次为：known calibration Accuracy 更高、Macro-F1 更高、`lambda_manifold` 更小、`lambda_sparse` 更小。选择期间不生成测试投影、不计算测试指标。

若入选参数位于网格边界，在查看测试集前只扩展一次相应方向；扩展后仍位于新边界则停止并报告参数范围尚未覆盖，不查看测试结果。

## 4. 最终比较

只有两种策略的参数选择都完成后，才各自在固定偶数角测试集评估一次。主要比较 known Accuracy 和 Macro-F1，并同时记录 calibration—test 差距、训练迭代数、最终 `W` 范数与 `alpha`。本轮仍标记为 diagnostic fold 0，不作为跨 fold 正式性能结论。
