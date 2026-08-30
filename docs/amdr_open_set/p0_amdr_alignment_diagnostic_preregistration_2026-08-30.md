# P0 AMDR 三项参考对齐累计诊断预注册

> 日期：2026-08-30
>
> 范围：fold 0、每类每个数据角色 500 对；仅诊断现有 `amdr_research_v1` 与参考论文/Matlab 的三项实现差异，不修改开集机制，不作为正式 P1 结果。

## 1. 要回答的问题

现有功率输入、随机槽位、未剪枝 pilot 的已知类 test Accuracy 为 `0.3969`。本诊断依次检查：

1. 将功率 dB 恢复为峰值相对线性幅度是否改善闭集识别；
2. 在幅度输入基础上，补回 Matlab `sum(W.^2,2)<1e-5` 的训练后行剪枝是否继续改善；
3. 在前两项基础上，将两个视角固定为按角度、样本 ID 升序的槽位是否继续改善。

## 2. 累计变体

| 变体 | 输入 | 行剪枝 | 槽位 |
|---|---|---|---|
| 既有 pilot | `10 ** ((x_db-max)/10)` | 无 | 显式种子随机翻转 |
| A | `10 ** ((x_db-max)/20)` | 无 | 与既有 pilot 相同 |
| A+P | 同 A | 平方行范数 `<1e-5` 置零 | 与既有 pilot 相同 |
| A+P+O | 同 A | 同 A+P | 角度、样本 ID 升序 |

A、A+P 与既有 pilot 使用完全相同的配对生成协议和种子。A+P+O 继续选择相同的无序底层样本对，只改变两个端点进入 `view1/view2` 的方向。

除表中累计变化外，四组均保持 fold 0、7 known/3 unknown、每类每角色 500 对、`lambda_manifold=lambda_sparse=0.01`、K=3、相对收敛阈值 `3e-5`、最多 300 轮以及 known-only calibration 阈值规则不变。

## 3. 判读规则

主要比较 calibration Accuracy、test known Accuracy 和 known Macro-F1。开集指标继续保存，但三项修改的首要判定不依据未知 test 指标。若某一步没有改善，不据此删除原始产物或事后改变候选定义；进一步原因分析另建诊断。

配置：

- `configs/amdr/pilot_fold0_amplitude_v1.yaml`；
- `configs/amdr/pilot_fold0_amplitude_pruned_v1.yaml`；
- `configs/amdr/pilot_fold0_amplitude_pruned_canonical_v1.yaml`。
