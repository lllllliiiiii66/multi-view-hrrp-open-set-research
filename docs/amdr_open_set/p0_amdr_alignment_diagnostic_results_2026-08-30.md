# P0 AMDR 三项参考对齐累计诊断结果

> 日期：2026-08-30
>
> 代码：`58dff0436e30eb88ef0b9b599449f5944bd83da3`
>
> 环境：Merlin，cgroup 8 CPU / 32 GiB
>
> 范围：fold 0、每类每个数据角色 500 对；诊断结果，不是正式 P1 主结果。

## 1. 受控比较

三项按预注册顺序累计：A 为峰值相对幅度，P 为 Matlab `sum(W.^2,2)<1e-5` 训练后行剪枝，O 为按角度、样本 ID 升序的确定性槽位。

| 变体 | calibration Accuracy | test known Accuracy | known Macro-F1 | AUROC | 剪枝行数 |
|---|---:|---:|---:|---:|---:|
| 既有功率 pilot | 0.5503 | 0.3969 | 0.3809 | 0.3334 | 0 |
| A | 0.6306 | 0.4603 | 0.4457 | 0.2956 | 0 |
| A+P | 0.6306 | 0.4603 | 0.4457 | 0.2956 | 216 |
| A+P+O | 0.5174 | 0.4517 | 0.4365 | 0.3036 | 235 |

A 相比既有 pilot 的 calibration Accuracy、test known Accuracy 和 known Macro-F1 分别提高约 8.03、6.34 和 6.48 个百分点。七个已知类别的 test Accuracy 均提高，其中 CVN77、DDG-112 和达飞罗尔多夫级分别由 0.404/0.734/0.350 提高到 0.530/0.828/0.454。

A+P 复用 A 的第 164 轮已收敛 checkpoint，只增加训练后剪枝。216/1202 个权重行被置零，但 calibration/test 分类指标与 A 完全相同，因此该剪枝不是当前低闭集准确率的原因。

A+P+O 使用与 A 完全相同的 12,000 个无序底层样本对，只改变端点所在槽位。它在第 158 轮收敛，test known Accuracy 比 A+P 低约 0.86 个百分点，calibration Accuracy 低约 11.31 个百分点。确定性槽位没有带来改善；按照当前主协议，它只保留为参考对齐诊断，主协议继续使用显式种子随机定向。

## 2. 证据边界

幅度输入改善了闭集分类，但没有改善未知分离：AUROC 从既有 pilot 的 0.3334 降到 0.2956，URR 从 0.0213 降到 0.0113。因此本结果只支持“功率/幅度语义是闭集性能问题之一”，不支持“幅度输入解决了开集识别”。

即使采用幅度输入，test known Accuracy 仍只有 0.4603；MARVEL CRANE、爱达魔都号和迷你好望角型散货船仍只有 0.252/0.234/0.242。三项参考差异不足以解释全部性能缺口，当前仍不能将低结果直接归因于开集拒判。

这组三项是在观察 fold 0 既有 pilot 后开展的诊断，不覆盖原始功率 pilot，也不据此把 fold 0 重新定义为正式主结果。是否将幅度语义并入后续 `paper_aligned` 配置，应依据论文输入定义和代码级对齐决定，而不是依据 test unknown 指标选择。

## 3. 可审计性与资源

- 既有 pilot、A、A+P 的 `pair_manifest.csv` SHA-256 均为 `4e16ce9e713b264ac481d01659c292f13db606c3aea6f2bbb4f77050ade0c171`；
- A+P+O 与 A 的无序配对集合完全相同，差集均为 0，且全部槽位满足确定性升序；
- A 运行约 52.19 秒、峰值 RSS 638.87 MiB；A+P 复用 checkpoint 后约 3.20 秒；A+P+O 约 49.03 秒、峰值 RSS 719.97 MiB；
- 本地完整测试 `223 passed`，Merlin 当前已提交测试集 `120 passed`；
- 三个本地结果目录中的 `artifact_hashes.json` 共 29 个登记文件全部复核一致。

本地产物：`artifacts/amdr/merlin/alignment_diagnostics/`，受 `.gitignore` 保护，不提交版本库。

## 4. 当前最小结论

峰值相对幅度应进入下一轮 AMDR `paper_aligned` 代码确认候选；Matlab 行剪枝无需作为性能修复重点；确定性槽位不进入冻结的随机定向主协议。下一项仍应是代码级的一轮更新对照和 Matlab-compatible/paper-aligned 版本边界确认，而不是扩大 fold、样本规模或引入开集机制。
