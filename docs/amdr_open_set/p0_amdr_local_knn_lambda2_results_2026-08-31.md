# P0 AMDR 局部图与 lambda2 诊断结果（fold 0 pilot）

## 结论

局部 KNN 图本身没有改善 AMDR；`lambda2=10` 只把闭集准确率小幅推到 58.74%，但开集 AUROC 降到 27.78%。因此，当前不建议用局部图替换完整类内图，也不建议在已查看的 fold 0 上继续扩展 `lambda2>10`。

本轮属于 P0 diagnostic pilot，不是正式性能结果。参数选择只使用 known calibration，测试未知类未参与训练、选择或阈值拟合。

## 核心结果

| 配置 | `lambda2` | Calibration Accuracy | Test known Accuracy | Test known Macro-F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| 完整类内图，固定 `D` | 1 | 72.31% | 57.94% | 57.18% | 44.61% |
| 局部 KNN 图，固定 `D` | 1 | 65.20% | 55.97% | 55.38% | 24.94% |
| 局部 KNN 图，known-only 校准选中 | 10 | 67.06% | **58.74%** | **57.97%** | 27.78% |

已经确认：

- 相同 `lambda2=1` 下，局部图比完整图低 1.97 个百分点，说明局部图本身没有带来收益。
- 将局部图的 `lambda2` 从 1 调到 10 后提高 2.77 个百分点，最终只比完整图高 0.80 个百分点，即 3500 条 known 测试对多判对 28 条。
- 58.74% 只比原始幅度直接 1-NN 的事后参照 58.46% 高 0.28 个百分点，AMDR 仍没有形成明显优势。
- 局部图选中项 AUROC 比完整图低 16.83 个百分点，闭集微增没有转化为更好的未知分离。

## 10 组 known calibration 结果

| `lambda2` | Accuracy | Macro-F1 | 迭代数 |
|---:|---:|---:|---:|
| 1 | 65.20% | 64.82% | 89 |
| 2 | 64.11% | 63.90% | 106 |
| 3 | 65.63% | 65.40% | 60 |
| 4 | 65.77% | 65.60% | 80 |
| 5 | 65.29% | 65.17% | 57 |
| 6 | 65.11% | 65.04% | 58 |
| 7 | 65.89% | 65.75% | 83 |
| 8 | 65.91% | 65.79% | 60 |
| 9 | 65.63% | 65.46% | 163 |
| 10 | **67.06%** | **66.93%** | 119 |

`lambda2=10` 位于搜索上界，但曲线并非单调上升。按照预注册规则只报告边界，不自动扩展；fold 0 测试已经查看，继续追调会增加事后调参风险。

## 类别变化

相对完整图 `1/1`，局部图选中项的主要增益来自集装箱船达飞罗尔多夫级（+6.2 个百分点）和爱达魔都号（+5.2 个百分点）；CVN77 下降 6.2 个百分点，DDG-1000 下降 2.4 个百分点。总体小幅上升来自类别间得失抵消，不是各类别一致改善。

## 审计与证据边界

- 10 个候选均使用 3500 条 known calibration 对，选择产物记录 `test_features_materialized=false`、`test_metrics_used=false`。
- 三组对照使用同一配对清单，SHA-256 为 `4e16ce9e713b264ac481d01659c292f13db606c3aea6f2bbb4f77050ade0c171`。
- 两组新结果均通过 `artifact_hashes.json` 校验；Accuracy 和 Macro-F1 已从逐样本预测独立反算并完全一致。
- 一次部分排序加速运行改变了等距离样本的邻居选择，已判定无效并排除。最终结果恢复稳定完整排序，选中项在 119 轮收敛，校准指标与参数搜索记录完全一致。
- 当前只有 fold 0 和单一配对种子，无法判断 0.80 个百分点是否稳定。

## 建议执行

1. 当前继续保留完整图、固定 `D`、`lambda1/lambda2=1/1` 作为 P1 候选闭集基线。
2. 不在 fold 0 继续扩展 `lambda2>10`；若仍需确认局部图，只在未查看 fold 上预先固定比较 `lambda2=1` 与 10。
3. 优先进入固定表示下的 P1 `Thresholded KNN`；局部图保留为负结果和消融参考。

## 技术记录

- 局部图实现提交：`9f8ae8ab1c839693dc1ec76ab641d47e8d2d7e13`
- 选中配置：`configs/amdr/pilot_fold0_amplitude_pruned_fixed_d_local_knn_lambda2_selected_v1.yaml`
- 参数选择产物：`artifacts/amdr/parameter_selection/fold0_local_knn_lambda2_v1/`
- 最终评估产物：`artifacts/amdr/d_strategy_evaluation/fold0_local_knn_lambda2_1_exact_v1/`、`artifacts/amdr/d_strategy_evaluation/fold0_local_knn_lambda2_selected_exact_v1/`
- 结果回传压缩包 SHA-256：`6a3a6676db7286a8cc078f1049f08a260270bf2be0ab5dbf15f8224a6a21ca7c`
- 本地完整测试：238 项通过。
- `RESEARCH_CONTEXT.md` 未修改。
