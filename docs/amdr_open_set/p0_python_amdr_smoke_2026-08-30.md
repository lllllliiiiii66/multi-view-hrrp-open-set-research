# P0 Python AMDR 诊断性 smoke 验证记录

> 日期：2026-08-30
>
> 阶段：P0
>
> 性质：诊断性 smoke，不是正式 P1 实验，不用于宣称识别性能。

## 1. 要回答的问题

本次 smoke 只验证一条最小链路是否可运行、可审计和可重现：十类原始 HRRP 的 601 维噪声补齐产物 → 奇偶角度隔离 → 5-fold known-only 校准 → 有序跨帧二视角组合 → Python AMDR 投影 → 平方欧氏 KNN → 仅基于已知校准样本的 95% 接受率阈值 → open-set 指标与完整产物落盘。

它不回答“AMDR 的正式闭集性能如何”或“Thresholded KNN 是否有效”。

## 2. 实现和配置

- 数据与组合：`src/hrrp_osr/amdr/data.py`；
- AMDR 投影与 KNN：`src/hrrp_osr/amdr/model.py`；
- 可审计运行入口：`src/hrrp_osr/amdr/smoke.py`；
- 版本化配置：`configs/amdr/smoke_v1.yaml`；
- 针对性测试：`tests/test_amdr_data.py` 和 `tests/test_amdr_model.py`。

本次使用 fold 0、7 known / 3 unknown，train/calibration/test 每类各 100 个二视角组合，AMDR 最多 3 次迭代，KNN 固定 \(K=3\)。输入转换为逐条 profile 的相对功率：

\[
x_{\mathrm{rel}}=10^{(x_{\mathrm{dB}}-\max x_{\mathrm{dB}})/10}.
\]

该转换仅用于 smoke 数值稳定性，不是已冻结的正式预处理。Python 实现是根据论文和附件代码转写的 smoke 版，尚不称为论文数值复现。

运行命令：

```bash
.venv/bin/python -m hrrp_osr.amdr.smoke \
  --config configs/amdr/smoke_v1.yaml \
  --bundle-root data/processed/hrrp_10class_theta83_hh_padding_v1 \
  --output artifacts/amdr/smoke_v1/fold_0
```

## 3. 验证结果

数据与协议检查全部通过：

- 5 个奇数角度 fold 均为 36 个角度，每个 fold 覆盖 24 帧；
- train 700 对、calibration 700 对、test 1000 对，共 2400 对；
- train/calibration 只含 7 个已知类，test 含 10 类；
- train/calibration 只使用奇数角度，test 只使用偶数角度；
- 两个视角来自不同帧，无重复无序底层样本对；
- train/calibration、train/test、calibration/test 的底层 sample ID 交集均为 0；
- 底层样本使用次数的最大与最小差均不超过 1；
- AMDR 投影、视角权重和距离均为有限数，且两个 `alpha` 之和为 1；
- 9 个主要产物的 SHA-256 校验通过，指标可由 `predictions.csv` 以 \(10^{-12}\) 容差独立反算；
- 从同一配置和数据重跑后，模型、投影、组合 manifest、预测、指标和训练日志等 7 个核心产物哈希完全一致。

针对新 AMDR 模块的 5 项单元测试通过；全仓库测试为 `215 passed`，`git diff --check` 无空白错误。在测试过程中，初版 fold 分配被检出为 36/37/36/35/36，修正后已严格为 36/36/36/36/36，这一问题没有进入最终 smoke 产物。

## 4. 诊断数值（不作性能结论）

运行完成 3 次迭代，但末次更新量仍为 240.018，未达到 \(3\times10^{-5}\) 的停止阈值。最终视角权重为 0.50265/0.49735。

smoke 产生的诊断数值为：known Accuracy 0.3914、AUROC 0.3619、OSCR 0.1662、FPR95 0.9614、URR 0.0333。这些数值受 100 对/类、单 fold、仅 3 次迭代、固定未选择的 \(\lambda\) 与 \(K\)、smoke-only 归一化以及未完成公式对齐的实现共同影响，不能用来判断 AMDR 或 KNN 开集机制的实际能力。

## 5. 产物与边界

本地诊断产物位于 `artifacts/amdr/smoke_v1/fold_0/`，包括解析后配置、环境、组合 manifest 及审计、投影矩阵和 `alpha`、各 split 投影、逐样本预测、阈值、指标、迭代日志和产物哈希。该目录受忽略规则保护，不提交 Git。

当前仍有五项关键限制：

1. 只运行 fold 0，未覆盖 5-fold 结果和多随机种子不确定性；
2. 只有 100 对/类和 3 次 AMDR 迭代，优化未收敛；
3. \(\lambda_1\)、\(\lambda_2\) 和 \(K\) 是 smoke 固定值，未执行 known-only 选择；
4. 逐 profile 相对功率变换尚未冻结为正式预处理；
5. Python AMDR 是可测试转写，但尚未完成与论文公式及 MATLAB 小型参考输出的逐步数值对齐。

此外，smoke 保留了类内精确关系块，但还没有把 AMDR 重写为分批或在线优化。在 7 个已知类下，100/500/2000 对每类将使单次训练的 \(n\) 分别为 700/3500/14000；因此本次结果不能被解读为“已经解决 \(n\times n\) 规模问题”。

## 6. 下一项最小工作

先核对并冻结 dB 数值域到 AMDR 输入的正式处理，再确认 Python AMDR 的更新顺序、\(l_{2,1}\) 重加权、视角权重、停止条件和对照 fixture。这两项通过后，再把同一协议扩到每类 500 对的 pilot；不直接运行 2000 对主实验，也不提前进入 P2。
