# P0 `amdr_research_v1` 完整收敛诊断

> 日期：2026-08-30
>
> 环境：Merlin，cgroup 8 CPU / 32 GiB
>
> 范围：fold 0、每类每个角色 100 对、训练 `n=700`，诊断性结果，不是正式开集实验。

## 1. 固定条件

- Git commit：`f63adba4f8da241ebf5efba128abe507e3382b08`；
- 输入：`power_db_to_peak_relative_power_v1`；
- 算法：`amdr_research_v1`；
- `lambda_manifold=lambda_sparse=0.01`，`K=3`；
- 相对状态变化阈值 `3e-5`，最多 300 次；
- latest checkpoint 每 5 次保存，并在收敛时额外保存；
- OpenBLAS/OMP/MKL/NumExpr 线程数显式设为 8。

## 2. 收敛与资源结果

模型在第 159 次停止，停止原因为 `converged_tolerance`。相对状态变化从 `43.1716` 降到 `2.19147e-5`，低于预注册阈值。记录的联合目标从 `245.5545` 降到 `39.7339`，158 次相邻变化全部不大于 0，没有观察到目标上升。

最后一轮目标分项为：回归 `0.56455`、流形 `8.59392`、稀疏 `30.57543`；视角权重为 `[0.456166, 0.543834]`。

进程内墙钟约 `10.93 s`，峰值 RSS 约 `232.41 MiB`。环境记录确认有效资源限额为 8 CPU / 32 GiB，运行时 Git 工作树干净。

## 3. 恢复验证

从第 159 次收敛 checkpoint 恢复后没有新增迭代。恢复运行与原运行的以下 SHA-256 完全一致：

- `model.npz`：`0cafb59f3798437c60e517aa37bbca7949a29fa6a8d1ee58b52d12750289ebe7`；
- `projections.npz`：`6cb71c7cebbd3be91a73c89a9bf56eed63f62737a73e379b3f38f84613d258fc`；
- `predictions.csv`：`b66ef1ca365563a1787da2d0896e109d55455514124b02a5d23d5d0e945c1e61`。

## 4. 研究解释与下一步

本次结果证明当前算法版本能够在 `n=700` 上稳定收敛、目标下降并可靠恢复，可以进入每类 500 对、训练 `n=3500` 的 fold 0 pilot。

诊断性开集指标仍然较弱：known Accuracy 约 `0.4086`、AUROC 约 `0.5004`、URR 为 `0.07`。这些数值只用于暴露当前状态，不用于事后调整本次 pilot 的正则参数、K 或阈值规则。pilot 继续使用完全相同的 `0.01/0.01`、`K=3` 和 95% known calibration 接受率，只改变组合数量。
