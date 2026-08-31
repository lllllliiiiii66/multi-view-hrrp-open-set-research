# P0 AMDR 参数选择与固定/更新 D 对照结果（fold 0 pilot）

## 结论

之前 fold 0 的 46.03% 闭集准确率主要由参数配置不合适造成，固定 `D` 只是次要影响。保持逐轮更新 `D` 不变，仅将 `lambda_manifold` 和 `lambda_sparse` 从 `0.01/0.01` 调整为只由 known train/calibration 选出的 `1/1`，偶数角 known test Accuracy 提高到 56.77%；固定初始 `D` 后进一步提高到 57.94%。因此，本轮已经排除“更新 `D` 是主要故障来源”的判断。

本轮属于 P0 diagnostic pilot，不是正式性能结果。它回答的是：在相同 fold 0 数据和幅度输入下，参数选择与 `D` 更新方式分别对 AMDR 闭集表现有多大影响。

## 核心结果

| 方法 | 参数 `lambda_manifold/lambda_sparse` | Calibration Accuracy | Test known Accuracy | Test known Macro-F1 |
|---|---:|---:|---:|---:|
| 原配置：逐轮更新 `D` | `0.01/0.01` | 63.06% | 46.03% | 44.57% |
| 调参后：逐轮更新 `D` | `1/1` | 71.00% | 56.77% | 56.05% |
| 调参后：固定初始 `D` | `1/1` | 72.31% | **57.94%** | **57.18%** |
| 原始幅度特征直接 1-NN（事后参照，不参与选参） | 不适用 | 74.34% | 58.46% | 未记录 |

已经确认：

- 仅调整参数、仍逐轮更新 `D`，Test known Accuracy 提高 **10.74 个百分点**；这是本轮最大增量。
- 在相同选定参数下，固定 `D` 比逐轮更新 `D` 再提高 **1.17 个百分点**，影响明显小于参数选择。
- 固定 `D` 的 57.94% 与原始特征直接 1-NN 的 58.46% 只差 **0.52 个百分点**。AMDR 不再出现此前相对简单 KNN 大幅退化的异常。
- 两种 `D` 策略都由 known-only calibration 选中 `1/1`，且选定值不在搜索边界，不需要扩展本轮参数范围。

## 核心原因

旧配置的两个正则参数均为 `0.01`，投影矩阵范数达到 341.58，训练集留一准确率为 100%，但偶数角测试只有 46.03%，表现出明显的训练拟合过强。调参后，两种策略的投影矩阵范数都下降到约 34，闭集泛化同步恢复。因此，当前证据支持“正则尺度过小是此前性能崩塌的主要原因”。

固定 `D` 的额外收益较小，说明逐轮更新 `D` 并非根本错误；它会改变特征筛选过程，但在合理参数下不会单独导致此前约 12 个百分点的损失。

## 证据边界

尚未验证：

- 当前只运行了 fold 0、单一配对种子，不能据此判断固定 `D` 在其他 fold 中一定更好。
- Calibration 到偶数角 test 仍有约 14 个百分点的下降，说明奇数角 calibration 对偶数角泛化的估计仍偏乐观。
- 本轮只解决闭集表现异常，没有解决开集拒判。固定/更新 `D` 的 AUROC 分别为 44.61%/45.00%，均低于 50%，不能据此宣称已有有效开集能力。
- 原始幅度直接 1-NN 是此前查看 fold 0 test 后得到的事后参照，只用于判断 AMDR 是否异常退化，未参与本轮参数选择。

本轮参数搜索只使用 known train/calibration。搜索阶段没有生成 test 投影特征，也没有计算 test 指标；选定参数后才分别执行一次固定 `D` 和逐轮更新 `D` 的偶数角 test。两次最终评估复用了同一份配对清单，清单 SHA-256 为 `4e16ce9e713b264ac481d01659c292f13db606c3aea6f2bbb4f77050ade0c171`。

## 建议执行

1. 暂定固定初始 `D`、`lambda_manifold=1`、`lambda_sparse=1` 为 P0 的候选闭集基线；它在本轮略优，而且规则更简单。
2. 下一次只做一个独立 fold 的确认实验：直接沿用该配置，不重新查看或反向调整 test。如果固定 `D` 仍不低于逐轮更新 `D`，再冻结为 P1 的 AMDR 表示。
3. 冻结后进入 P1 `AMDR + Thresholded KNN`，严格只用 known calibration 定阈值；不要继续围绕 fold 0 追加参数以追求更高 test 数字。

## 技术记录

- 参数搜索配置：`configs/amdr/pilot_fold0_d_strategy_parameter_selection_v1.yaml`
- 固定 `D` 配置：`configs/amdr/pilot_fold0_amplitude_pruned_fixed_d_selected_v1.yaml`
- 更新 `D` 配置：`configs/amdr/pilot_fold0_amplitude_pruned_dynamic_d_selected_v1.yaml`
- 参数搜索产物：`artifacts/amdr/parameter_selection/fold0_d_strategy_v1/`
- 最终评估产物：`artifacts/amdr/d_strategy_evaluation/fold0_fixed_d_selected_v1/`、`artifacts/amdr/d_strategy_evaluation/fold0_dynamic_d_selected_v1/`
- 代码版本：`e6fc2ebd66668b3c0242a897bc49409ec9467850`
- Merlin 环境检查：相关配置测试 13 项通过；完整 AMDR 相关测试 130 项通过。
- 结果回传压缩包 SHA-256：`a7de2bd2da4540d0c678b7a53be1fe341774e49694b9459b8800ea9fafd7e5d6`
- `RESEARCH_CONTEXT.md` 未修改。
