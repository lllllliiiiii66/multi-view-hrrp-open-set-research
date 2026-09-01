# P0 AMDR 封口实验预注册

> 日期：2026-09-01
>
> 阶段：P0
>
> 状态：运行前冻结。本文不修改 `RESEARCH_CONTEXT.md`，不得根据结果追加候选或改变门槛。

## 1. 唯一研究问题

在严格底层 HRRP 隔离协议下，一个方法定义清楚、数值稳定、接近师兄附件实际实现的 AMDR，是否值得继续作为多视角开集识别的主要表示基础？

本轮不寻找 fold 0 上的最高准确率，不实现或评价任何复杂开集机制。fold 0、未知类和偶数角 test 均不参与运行、选择或确认。

## 2. 冻结方法

- 输入：`power_db_to_peak_relative_amplitude_v1`，即逐条 `10 ** ((x_db - max(x_db)) / 20)`；
- 两个固定有序视角槽位，`slot_order=randomized_seeded`；
- 每个组合来自不同 15°角度帧；
- `graph_neighborhood=complete_same_class_inverse_distance_v1`；
- `graph_same_base_policy=allow_same_base_v1`；
- `l21_reweighting=fixed_initial`；
- `objective_scaling=sample_class_mean_v1`；
- 不使用 PCA；训练后行剪枝阈值为 0；
- KNN：`k=3`、平方欧氏距离；
- 每类 train/calibration 各 500 对；
- `max_iterations=300`、`minimum_iterations=3`、`tolerance=3e-5`、`numerical_epsilon=1e-10`、`solve_ridge=1e-10`、初始化种子 `20260830`；
- 不生成偶数角 test 配对、特征、预测或指标。

固定 `D` 是接近师兄附件实际代码的稳定 **code-reference** 版本，不是论文声称的逐轮 `l2,1` 重加权实现。

## 3. 唯一允许的参数候选

候选来自已有完整图、固定 `D`、legacy 参数区域到 normalized 目标的等价尺度映射：

- `lambda_manifold ∈ {10,100,1000}`；
- `lambda_sparse = 2.857142857142857e-4`，即 `1/3500`。

不因边界、准确率或收敛结果扩展候选。

## 4. 阶段 A：fold 1–3 全局选择

每个 fold 分别运行三个候选，只使用 known train 和 known calibration。候选只有同时满足以下条件才有全局准入资格：

1. fold 1、2、3 全部 `converged=True`；
2. 三个 fold 的每次运行均有 `min(alpha) >= 0.05`。

合格候选依次按以下规则选择唯一全局配置：

1. 三 fold 平均 calibration Accuracy 更高；
2. 三 fold 平均 calibration Macro-F1 更高；
3. `lambda_manifold` 更小；
4. `lambda_sparse` 更小。

如果没有合格候选，P0 closure 直接判定为 `reject`，不运行 fold 4，不改变迭代上限、参数、图、`D`、PCA 或输入。

## 5. 阶段 B：fold 4 一次性确认

只使用阶段 A 选出的唯一配置，在此前未参与选择的 fold 4 运行一次 AMDR。同时使用完全相同的 pair manifest、train/calibration 底层样本、槽位、幅度输入、标签顺序和 KNN 设置，运行：

`Raw two-view concatenation + KNN`，其中 `X_raw=concatenate([view1,view2], axis=1)`。

两者只比较 fold 4 known calibration Accuracy 和 Macro-F1，不生成 test 特征。

## 6. 去留规则

### Primary

同时满足：

- fold 4 AMDR 收敛；
- `min(alpha) >= 0.05`；
- `AMDR Accuracy - Raw Accuracy >= 0.02`；
- `AMDR Macro-F1 - Raw Macro-F1 >= 0`。

### Auxiliary

没有达到 Primary，但同时满足：

- fold 4 AMDR 收敛；
- 无视角塌缩；
- Accuracy 差值严格大于 `-0.02`；
- Macro-F1 差值严格大于 `-0.02`。

### Reject

阶段 A 无合格候选，或 fold 4 不收敛/视角塌缩，或 Accuracy/Macro-F1 任一项比 Raw 低至少 0.02。2 个百分点是工程决策门槛，不解释为统计显著性。

## 7. 实现、产物和审计

配置：`configs/amdr/p0_closure_known_only_v1.yaml`。运行入口必须具有独立 `p0_closure` 命名，不覆盖 fold 0 诊断目录。

保存：每个 fold/candidate 的 resolved config、pair manifest 与 SHA-256、训练日志、模型、收敛状态、迭代数、`alpha`、权重范数、calibration Accuracy/Macro-F1；阶段 A 聚合表；fold 4 AMDR/Raw 逐样本预测与比较；最终 `primary/auxiliary/reject` 决策；环境、Git 状态和全产物哈希。

代码必须保持历史选择语义：只有配置显式设置 `require_converged: true` 时才排除未收敛候选；历史未设置配置仍按原排序。

## 8. 停止规则

完成 pytest、配置校验、`git diff --check` 和 fold 1–4 规定运行后停止。不进入 P1，不运行 fold 0，不生成偶数角 test，不再寻找更高 AMDR 数字。

