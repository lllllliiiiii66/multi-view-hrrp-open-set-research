# P0 AMDR 类内局部 KNN 图与 lambda2 搜索预注册（fold 0 pilot）

## 研究问题

在保持当前数据、幅度输入、固定初始 `D`、`lambda1=1` 和 KNN 分类器不变时，将完整同类图替换为类内局部 KNN 图，能否改善 AMDR 的 known-only calibration 表现；在该图下，`lambda2` 的 1–10 范围内哪个值最好。

本实验属于 P0 diagnostic，不是原附件复现，也不是正式性能结果。

## 唯一方法改动

- 每个类别、每个视角只保留投影空间中最近的 10 个同类邻居；
- 使用行内局部尺度的高斯权重并按行归一化；
- `L2_distance` 语义为平方欧氏距离，因此高斯指数不再次平方该距离；
- 第一轮图由训练输入空间构建，避免单位矩阵初始化使第一轮流形项为零；
- 当前 pilot 使用逻辑稀疏的稠密类别块存储，只隔离方法效果，不把存储格式变化混入准确率对照。

## 固定项

- fold 0，7 known / 3 unknown；
- 每类 train/calibration/test 各 500 对；
- 奇数角 train/calibration，偶数角固定 test；
- 随机有序槽位及配对种子保持不变；
- 幅度变换与 601 维噪声补齐保持不变；
- 固定初始 `D`，`lambda1=1`，分类 KNN `K=3`；
- 不采用测试集早停或最佳迭代选择。

## 参数选择

只使用 known calibration，搜索：

```text
lambda2 = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
```

按 calibration Accuracy、Macro-F1、较小 `lambda2` 的顺序选择。搜索阶段不生成 test 投影，不计算 test 指标。该范围由用户明确指定；若最佳值位于 1 或 10，只报告边界事实，不自动扩大范围。

## 最终评估

选定 `lambda2` 后冻结独立配置，只执行一次偶数角 test。主要闭集对照为：

- 完整同类图、固定 `D`、`lambda1=lambda2=1`：已知类 test Accuracy 57.94%；
- 局部 KNN 图、`lambda2=1`：用于隔离图结构改动；
- 局部 KNN 图、calibration 选定 `lambda2`：用于评估图结构与正则强度的组合效果。

任何 test 结果均不得反向修改图邻居数、`lambda2`、预处理或配对规则。
