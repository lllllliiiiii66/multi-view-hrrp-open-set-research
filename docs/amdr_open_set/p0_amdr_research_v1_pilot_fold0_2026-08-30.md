# P0 `amdr_research_v1` fold 0 pilot

> 日期：2026-08-30
>
> 环境：Merlin，cgroup 8 CPU / 32 GiB
>
> 范围：fold 0、每类每个角色 500 对；训练与 calibration 各 3,500 对，test 5,000 对。结果为诊断性 pilot，不是正式 P1 主结果。

## 1. 固定条件

- Git commit：`aa98812ba7a952ec7c6ca42dd92bfed057a2b03c`；
- 输入：`power_db_to_peak_relative_power_v1`；
- 算法：`amdr_research_v1`；
- `lambda_manifold=lambda_sparse=0.01`，`K=3`；
- 相对状态变化阈值 `3e-5`，最多 300 次；
- 阈值只由当前 fold 的已知 calibration 按 95% known 接受率确定；
- 本次没有根据未知 test 表现调整正则参数、K 或阈值规则。

## 2. 优化、资源与恢复

模型在第 157 次达到 `2.97918e-5` 并停止。联合目标从 `8068.1425` 单调降到 `140.9949`，156 次相邻变化全部不大于 0。最后目标分项为：回归 `44.4185`、流形 `8.6980`、稀疏 `87.8784`；视角权重为 `[0.497860, 0.502140]`。

进程内墙钟约 `49.75 s`，峰值 RSS 约 `645.56 MiB`，checkpoint 约 12 MiB。配对审计全部通过，train/calibration/test 的底层样本使用次数分别只相差 1 次。

从第 157 次 checkpoint 恢复后没有新增迭代，以下产物 SHA-256 与原运行完全一致：

- `model.npz`：`277577954b46386010d3c92fe6feff91d04367ba3cfd114a14a45e70a75ccf41`；
- `projections.npz`：`834ccb16bc88d505f1136dbb034eee31248222030333a190f69de8cf1514f3ef`；
- `predictions.csv`：`599a79209098c8aa7ac8370a0bbd0d9a17a080e8723022043968e7d4b48349ff`。

远程主产物已复制到本地受 `.gitignore` 保护的 `artifacts/amdr/merlin/pilot_fold_0/`，复制后上述三个哈希再次一致。

## 3. 诊断性识别结果

- calibration known Accuracy：`0.5503`；
- test known Accuracy / Macro-F1：`0.3969 / 0.3809`；
- AUROC / OSCR / FPR95：`0.3334 / 0.1623 / 0.9769`；
- known acceptance / unknown rejection：`0.9137 / 0.0213`；
- KCCR/URR 调和分数：`0.0404`。

因此，pilot 在优化稳定性和资源可行性上通过，但 `AMDR + mean-KNN-distance` 的开集分离在当前 fold 明显失败，不能据此直接扩大到每类 2,000 对的主实验。

## 4. 失败机制证据

calibration 已知分数中位数为 `5.821`，test 已知为 `8.774`，test 未知反而只有 `4.251`。未知样本并没有获得更大的 KNN 距离，所以 AUROC 小于 0.5 和极低 URR 是数据几何的直接结果，而不是评价器把分数方向写反。

投影范数进一步支持这一点：训练集、test 已知和 test 未知的中位数分别约为 `5.674`、`7.053` 和 `5.446`。未知样本整体落得更接近训练投影尺度，部分已知偶数角样本反而偏离更远。

未知类平均原始长度为 241，已知类约为 515.3；两视角总补齐量均值分别为 720 和 171.4。test 分数与补齐量相关系数约为 `-0.202`，即补齐更多的样本倾向于获得更小而非更大的未知分数。该相关性只说明长度/补齐与失败共同关联，不能在没有控制实验时写成因果结论。

只在已知 calibration 上检查 `K=1/3/5/7/9` 后，最高 Accuracy 来自 `K=5` 的 `0.5511`，而 `K=3` 为 `0.5503`，差异只有约 0.086 个百分点。没有计算其他 K 的未知 test 结果，因此当前证据不支持把失败归因于 K=3。

## 5. 当前决策

暂不运行 `n=14000` 主实验。下一项最小工作应保持当前参数不变，在 pilot 尺度检查其他 fold/初始化种子是否重复出现“未知投影靠近训练、已知偶数角偏离”的现象；或者先预注册一个只使用已知 calibration 的距离诊断协议。未完成这一步前，不引入先进开集方法，也不修改 AMDR 表示学习目标。
