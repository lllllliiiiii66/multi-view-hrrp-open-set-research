# P0 AMDR 终止、checkpoint 与 Merlin 执行建议

> 日期：2026-08-30
>
> 阶段：P0
>
> 状态：可恢复 checkpoint 已实现并通过 smoke；Merlin 已连接并完成首次远程核验。

## 1. 三个容易混淆的概念

师兄的 `train.m` 并非完全没有停止机制。代码在第 3 次迭代之后检查 \(W\)、第一个视角的 \(S\)、\(T\) 和 \(\alpha\) 的变化量之和；低于 \(3\times10^{-5}\) 时提前结束，否则最多运行 300 次。这是“优化收敛停止”，不是根据 validation 性能的神经网络式早停。

本项目分别处理：

1. **收敛停止：需要。** Python AMDR 保留 `minimum_iterations + tolerance + max_iterations` 三个明确参数，并记录 `converged_tolerance` 或 `max_iterations` 终止原因。
2. **按 calibration Accuracy 早停：当前不加。** AMDR 是交替闭式优化，而不是按 epoch 训练的神经网络。用 calibration Accuracy 选择迭代次数会额外改变原方法，并把同一 calibration 同时用于 \(K\)、阈值和迭代选择。
3. **“最佳” checkpoint：当前不加。** 没有经预注册的独立目标可用来定义“最佳”。正式模型使用满足收敛条件或达到最大迭代时的最终状态。

## 2. 已补充的 latest checkpoint

远程任务中断恢复是工程可靠性问题，不应与模型选择混为一谈。当前实现新增 `latest_only_atomic_replace` checkpoint：

- 保存已完成迭代数、\(\widetilde W\)、\(\alpha\)、调整后回归目标 \(T\)、两个视角的全部类内 \(S\) 块和历史日志；
- 保存视角维度、样本数、训练视角值与顺序 SHA-256、标签顺序 SHA-256 和恢复关键模型配置；
- 先写临时文件，再原子替换 `checkpoint_latest.npz`，避免中断留下半个文件；
- 只保留最新状态，不按性能排名，避免大量迭代文件占用磁盘；
- 恢复时必须使用相同数据顺序和恢复关键配置；`max_iterations` 可增大，其他关键参数不能静默变更。

恢复命令必须写入新的输出目录，不覆盖中断运行：

```bash
.venv/bin/python -m hrrp_osr.amdr.smoke \
  --config configs/amdr/smoke_checkpoint_v1.yaml \
  --bundle-root data/processed/hrrp_10class_theta83_hh_padding_v1 \
  --output artifacts/amdr/smoke_checkpoint_v1/fold_0_resumed \
  --resume-from artifacts/amdr/smoke_checkpoint_v1/fold_0/checkpoint_latest.npz
```

## 3. 实际验证

- 合成数据先运行 2 次、保存 checkpoint，再恢复到第 5 次；迭代历史、\(\widetilde W\) 和 \(\alpha\) 与一次性运行 5 次逐位一致。
- 真实 HRRP fold 0 smoke 生成了 958 KiB 的 `checkpoint_latest.npz`，状态为完成 3 次迭代。
- 从该 checkpoint 恢复后，`model.npz` 和 `projections.npz` 的 SHA-256 分别与未中断运行完全一致。
- checkpoint smoke 仍然在 3 次上限时未收敛；这次修改不会把其变成正式性能实验。

## 4. Merlin 8 CPU / 32 GB 的使用边界

**当前 smoke 不需要 Merlin。** \(n=700\) 的 3 次本地运行只需数秒；本次记录的峰值 RSS 约为 233 MiB。继续在本地做快速单元测试和小型收敛诊断更方便。

**pilot 建议使用 Merlin。** 每类 500 对时 \(n=3500\)，类内 \(S\) 块本体约占 27 MiB，32 GB 内存有充足裕量。更重要的是 Merlin 可以提供独立、可记录的 Linux CPU 环境，适合实测单次迭代时间、峰值 RSS、checkpoint 写入时间和收敛次数。

**\(n=14000\) 的首轮主实验应优先放在 Merlin，但必须先通过 pilot。** 两个视角、7 类、每类 2000 样本时，仅类内 \(S\) 块原始 `float64` 存储约为 427 MiB；旧状态副本、距离、拉普拉斯中间量和输入矩阵会进一步推高峰值。32 GB 从容量上大概足够，但运行时间和 checkpoint I/O 需要用 pilot 实测，不能只按公式估计。

建议 checkpoint 间隔为：smoke 每 1 次，pilot 初始每 5 次，主实验初始每 10 次；后两者必须根据实测 I/O 时间再冻结。NumPy/SciPy 线性代数线程数建议与 8 CPU 对齐，并把实际 BLAS 后端、线程环境变量和峰值 RSS 写入运行产物。

## 5. 上 Merlin 之前的前置条件

1. 当前 AMDR 新代码仍是本地未提交文件，必须先审查、commit 并 push 到私有 GitHub；
2. 原始数据和处理后 bundle 不进 Git，应单独安全传到 Merlin，并重新核对三个 bundle SHA-256；
3. 在 Merlin 新建独立 Python 环境，记录 Python、NumPy、SciPy、BLAS、CPU、线程数和操作系统；
4. 先复现 \(n=700\) checkpoint smoke，哈希和数值差异通过后，再执行 \(n=3500\) pilot。

上述建议记录的是撰写时的待执行状态；随后已按本记录第 6 节完成 Merlin 连接与远程核验。

## 6. Merlin 首次实际核验

2026-08-30 已经通过用户提供的 SSH 入口连接 Merlin。宿主机显示 128 个逻辑 CPU 和 503 GiB 内存，但当前开发机 cgroup 限额实际为 8 CPU 和 32 GiB，与用户描述一致。

远程代码使用 GitHub 已推送分支 `codex/amdr-p0-smoke` 的 commit `0a9b791`；由于 Merlin 没有 GitHub HTTPS/SSH 凭据，通过经 SHA-256 校验的单分支 Git bundle 建立标准 Git 工作树，没有输入或复制密码/令牌。处理后 HRRP bundle 已经存在于 Merlin，其 profiles、manifest 和 bundle 三个 SHA-256 与配置完全一致。

远端使用 Python 3.12.13、NumPy 2.5.2、SciPy 1.18.0 和 OpenBLAS 0.3.34，线程数显式设为 8。提交快照上的 113 项测试全部通过；\(n=700\) checkpoint smoke 墙钟约 1.75 秒、进程内记录峰值 RSS 约 220 MiB。远端所有离散预测与指标同本地一致，只有阈值出现约 \(4.1\times10^{-11}\) 的跨 BLAS 浮点差异。远端 checkpoint 恢复后，模型、投影和预测产物与远端未中断运行的 SHA-256 完全一致。
