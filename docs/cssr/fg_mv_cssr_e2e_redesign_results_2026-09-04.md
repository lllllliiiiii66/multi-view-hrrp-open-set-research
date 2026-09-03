# FG-MV-CSSR v2 端到端重设计快速实验结果

> 日期：2026-09-04
>
> 阶段：P3 独立快速机制验证
>
> 实验：`fg_mv_cssr_e2e_redesign_v2`
>
> 分支：`codex/fg-mv-cssr-e2e-redesign`
>
> 产生性能的冻结代码提交：`e42ec8b5dec20d7d94e9a2774f6a9353f8aed8cd`
>
> 配置 SHA-256：`5c227c00a7ac5a88c9bf5d66618964bc05c67f45c51c2a880731f6753626512e`
>
> 冻结运行：`angle_fold=0`，`R2_seed=20260830`，`finetune_seed=20260904`
>
> 最终状态：`pilot_status=hard_failed_incomplete`，`pilot_gate=not_evaluated`

## 1. 明确结论

本次正式 pilot 没有完成：计划的 12 个训练单元中，4 个完成并通过单元审计，`N4-Q2` 触发预注册的训练稳定性硬门槛，3 个并行单元随即取消，另 4 个没有启动。

触发项已经确定为 Q2 唯一的加权辅助项 `0.5 L_rel`。它相对分类损失 `L_cls` 在最后 residual stage 上的梯度比连续 3 个完整 epoch 超过 100。预注册要求这种情况立即失败并报告、不得修改权重挽救，因此停止剩余任务是合规处置。

由此得到：

```text
pilot_status = hard_failed_incomplete
pilot_gate = not_evaluated
selected_method = null
confirmation_allowed = false
final_unknown_test_authorized = false
```

本轮没有形成一个经三组 pair 完整验证的 CSSR 候选，当前 CSSR 主线因正式 pilot 的训练稳定性硬失败而停止。`cssr_redesign_failed` 是预注册中“完成全部候选 gate 后均不合格”的标签，本轮没有执行 gate，因此不复用该标签。

4 个成功单元只能作为非决策性的局部证据：Q1 的 CE 微调在 N1 上使 AUROC 提高 `8.37 pp`，在 N4 上却降低 `7.72 pp`；N1 的 Q3 比 Q1 高 `1.54 pp` AUROC，但 URR 低 `5.30 pp`，且 N4-Q2 已发生稳定性失败。这不足以证明端到端对齐、绝对约束或局部卷积结构中的任何一项稳定有效。

## 2. 冻结设计与执行边界

本轮严格沿用预注册的三个 pilot pair：

- N1：DDG-112 / 迷你好望角型散货船；
- N4：DDG-1000 / 集装箱船达飞罗尔多夫级；
- N2：MARVEL CRANE / 迷你好望角型散货船。

方法固定为：

| 方法 | 定义 | 主 unknown score |
|---|---|---|
| Q0 | 冻结 R2，不训练 | 类别条件 MLS |
| Q1 | 仅 CE 有限微调 | 类别条件 MLS |
| Q2 | Q1 + `0.5 L_rel`，1×1 AE | guided reconstruction |
| Q3 | Q2 + `0.25 L_abs + 0.5 L_sep`，1×1 AE | guided reconstruction |
| Q4 | Q3 的 AE 改为 kernel-3 Conv1d | guided reconstruction |

Q1–Q4 固定训练 20 epoch，不早停、不按性能选 epoch。每个 epoch 使用 720 对动态、跨帧、无重复无序对训练样本；同一 pair 下四种方法共享完全相同的 epoch-wise pair schedule。surrogate unknown 只用于冻结后评价，不进入训练、参考分布、阈值或 epoch 选择。

本轮没有生成或使用最终三类 unknown、偶数角 test 或 ARPL，也没有启动第二 fold、额外 seed 或事后调参。`RESEARCH_CONTEXT.md` 未修改。

## 3. 实施与审计链

实验遵循“预注册—消歧—实现—测试—smoke—pilot”的顺序，所有性能均由提交 `e42ec8b5dec20d7d94e9a2774f6a9353f8aed8cd` 产生。主要新增文件为：

- `configs/experiments/cssr/fg_mv_cssr_e2e_redesign_v2.yaml`；
- `docs/cssr/fg_mv_cssr_e2e_redesign_preregistration_2026-09-04.md`；
- `src/hrrp_osr/models/cssr_e2e_1d.py`；
- `src/hrrp_osr/training/fg_mv_cssr_e2e_redesign.py`；
- `tests/test_fg_mv_cssr_e2e_model.py`；
- `tests/test_fg_mv_cssr_e2e_protocol.py`；
- `tests/test_fg_mv_cssr_e2e_runner.py`；
- `tests/test_fg_mv_cssr_e2e_cuda.py`。

运行前检查：

- 本地完整测试：`409 passed, 5 skipped`；跳过项均为本机没有 CUDA；
- 4090 容器 CUDA 专项测试：`4 passed`；
- 4090 容器完整测试：`414 passed`；
- Python compile、配置校验、任务计划校验和 `git diff --check`：通过；
- 配置、模型、runner、导入链及关键依赖均进入源码哈希绑定。

GPU smoke 在 N1 的 Q1–Q4 上各运行 1 epoch，4 个单元均完成；phase audit 对 checkpoint 重放、指标重算、pair schedule、标签顺序及产物哈希全部通过。smoke 只验证链路，不进入任何性能判断。

正式 pilot 中，4 个成功单元分别通过独立 `audit-unit`；摘要包内保留文件的 SHA-256 全部匹配，`partial_metrics.json` 与各单元指标文件精确一致。由于本机摘要包主动省略 checkpoint、逐样本预测和大型 manifest，本报告对部分指标只称“指标与日志层复核通过”；完整产物仍保留在 GPU 容器。

## 4. 正式 pilot 完成状态

| Pair | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| N1 | 完成、审计通过 | 完成、审计通过 | 完成、审计通过 | 因 N4-Q2 硬失败取消（无结果） |
| N4 | 完成、审计通过 | **训练稳定性硬失败** | 因 N4-Q2 硬失败取消（无结果） | 未启动 |
| N2 | 因 N4-Q2 硬失败取消（无结果） | 未启动 | 未启动 | 未启动 |

共计：

- 完成且审计通过：4/12；
- 训练稳定性硬失败：1/12；
- 已启动后取消：3/12；
- 未启动：4/12。

取消或未启动的单元没有性能结论，不能写成方法效果差。由于 12 项不完整，phase aggregate、pilot gate、候选选择和 `_PHASE_SUCCESS.json` 均没有生成。

## 5. 硬失败的直接证据

Q2 的正式损失为：

```text
L = L_cls + 0.5 * L_rel
```

因此 N4-Q2 抛出的“weighted auxiliary gradient exceeded the frozen 100x stability limit”只能来自 `0.5 L_rel`，不是 Q3/Q4 才有的绝对重构或分离项。运行器只有在加权辅助项与 CE 对最后 residual stage 的每 epoch 梯度比连续 3 个完整 epoch 超过 100 时才抛出该异常。

异常在当轮训练记录落盘前抛出，所以现有失败日志没有保存三个触发 epoch 和各自原始梯度值。本轮只能确认“连续 3 个完整 epoch 超过 100”，不能补写具体 epoch 或比例。失败日志 SHA-256 为：

`c3d71032eb2a5b81fe90c80cdbbdca9ad7021fc8609cb4c321dbf0bdad95d966`

这是运行可观测性上的缺口，不改变本次硬失败判定。若未来开展独立实验，建议先让异常路径在抛出前原子保存失败 epoch、损失分量和原始梯度；不得用该改动重启或挽救本轮 pilot。

## 6. 已完成单元的部分指标

以下均为百分数，FPR95 越低越好。Q0 是相同 pair 的冻结 R2 参考；表中没有的组合均未形成可审计成功产物。

| Pair | 方法 | Known Acc | Known Macro-F1 | AUROC | OSCR | FPR95↓ | KCCR | URR | H | K+1 Macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N1 | Q0 | 98.08 | 98.084 | 86.96 | 85.90 | 64.04 | 94.52 | 63.60 | 76.04 | 88.39 |
| N1 | Q1 | 98.72 | 98.717 | 95.33 | 94.43 | 26.16 | 94.48 | 80.90 | 87.16 | 92.03 |
| N1 | Q2 | 98.68 | 98.676 | 91.91 | 90.86 | 33.76 | 94.04 | 60.20 | 73.41 | 87.27 |
| N1 | Q3 | 98.64 | 98.636 | 96.87 | 95.63 | 15.32 | 93.96 | 75.60 | 83.79 | 90.42 |
| N4 | Q0 | 99.32 | 99.321 | 83.01 | 82.74 | 60.88 | 94.56 | 31.50 | 47.26 | 79.34 |
| N4 | Q1 | 99.20 | 99.202 | 75.29 | 75.11 | 76.36 | 94.60 | 23.30 | 37.39 | 77.18 |

受控差值：

| 比较 | AUROC | OSCR | FPR95↓ | KCCR | URR | H | K+1 Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| N1 Q1−Q0 | +8.37 | +8.53 | −37.88 | −0.04 | +17.30 | +11.13 | +3.64 |
| N4 Q1−Q0 | −7.72 | −7.63 | +15.48 | +0.04 | −8.20 | −9.87 | −2.17 |
| N1 Q2−Q1 | −3.42 | −3.57 | +7.60 | −0.44 | −20.70 | −13.76 | −4.76 |
| N1 Q3−Q2 | +4.96 | +4.77 | −18.44 | −0.08 | +15.40 | +10.38 | +3.15 |
| N1 Q3−Q1 | +1.54 | +1.20 | −10.84 | −0.52 | −5.30 | −3.38 | −1.61 |

这些差值说明：

1. Q1 的额外 CE 微调明显依赖 pair，不能只凭 N1 宣称普遍改善；
2. Q2 在唯一完成的 N1 上低于 Q1，且在 N4 训练时硬失败，不能支持端到端相对重构已有效；
3. Q3 在 N1 上比 Q2 好，但同时加入了 `L_abs` 和 `L_sep`，不能把变化单独归因于其中一个；它相对 Q1 的 AUROC增益也只有 `1.54 pp`，且固定阈值下的 URR、H 和 K+1 Macro-F1更低；
4. Q4 没有任何正式成功单元，局部 Conv1d AE 完全无法评价。

这里不计算三 pair 均值、正向 pair 数或任何部分 gate，也不绘制汇总图，以免不完整结果造成“候选已经比较完”的误解。

## 7. 身份级错误与 DDG 相互吸收

### 7.1 身份级指标

单元格为 `AUROC / URR / FPR95`，均为百分数。

| Pair | Surrogate 身份 | Q0 | Q1 | Q2 | Q3 |
|---|---|---:|---:|---:|---:|
| N1 | DDG-112 | 74.24 / 27.2 / 72.28 | 90.88 / 61.8 / 36.84 | 83.82 / 20.4 / 39.76 | 93.76 / 51.2 / 18.84 |
| N1 | 迷你好望角型散货船 | 99.68 / 100 / 0.64 | 99.78 / 100 / 0.44 | 100.00 / 100 / 0.00 | 99.98 / 100 / 0.04 |
| N4 | DDG-1000 | 77.48 / 22.8 / 74.96 | 62.38 / 7.2 / 87.84 | — | — |
| N4 | 集装箱船达飞罗尔多夫级 | 88.53 / 40.2 / 31.72 | 88.20 / 39.4 / 30.92 | — | — |

N4 的 Q1 相对 Q0 整体退化主要由 DDG-1000 驱动：其身份级 AUROC 降低 `15.10 pp`。N1 中 Q3 相对 Q1 的 DDG-112 AUROC只提高 `2.88 pp`，迷你好望角提高 `0.20 pp`；这仍只是单一 pair 的局部现象。

### 7.2 false accept 去向

每个 surrogate 身份各 500 条：

- N1 DDG-112 被接收为 DDG-1000：Q0/Q1/Q2/Q3 分别为 `364/191/398/244`；
- N1 迷你好望角型散货船：Q0/Q1/Q2/Q3 均为 `0`；
- N4 DDG-1000 被接收为 DDG-112：Q0/Q1 分别为 `386/464`；
- N4 达飞罗尔多夫级被误接收：Q0 为 `CVN77:23 / MARVEL CRANE:44 / 爱达魔都号:232`，共 299；Q1 为 `CVN77:14 / MARVEL CRANE:43 / 爱达魔都号:246`，共 303。

因此 DDG 双向相互吸收没有得到“同时缓解”的证据：N1 的 Q2/Q3 相对 Q1 反而放过更多 DDG-112，N4 的 CSSR 单元没有成功结果。N4-Q1 还使 `DDG-1000→DDG-112` 从 386 增至 464。当前已完成身份中没有 AUROC 低于 40%，但 pilot 不完整，不能据此判断完整身份门槛。

N2 未完成，所以 MARVEL CRANE 与 N2 迷你好望角型散货船的反向表现无法评价。

## 8. 已知类稳定性、置信度与梯度诊断

### 8.1 已知类稳定性

4 个成功训练单元的 known calibration Accuracy 均为 `98.64%–99.20%`，相对各自 Q0 的绝对变化不超过 `0.64 pp`。这说明已完成单元没有明显破坏已知类分类，但不能外推到失败或未运行的 N4-Q2、N2、Q4。

### 8.2 置信度变化

| 单元 | NLL 变化 | ECE 变化 | 平均最大 logit 变化 | top1−top2 margin 变化 | 融合特征 norm 变化 | CE head norm 变化 |
|---|---:|---:|---:|---:|---:|---:|
| N1-Q1 | −0.00981 | −0.00042 | +0.47998 | +0.78616 | +0.11848 | +0.14895 |
| N1-Q2 | −0.01012 | −0.00103 | +0.45838 | +0.75067 | +0.10035 | +0.15156 |
| N1-Q3 | −0.01185 | −0.00087 | +0.46874 | +0.76912 | +0.11374 | +0.15149 |
| N4-Q1 | −0.00230 | −0.00449 | +0.63126 | +1.13064 | +0.25325 | +0.16901 |

训练后 logit、margin、融合特征 norm 和 head norm 都上升，但 NLL/ECE 没有同步恶化，因此不能认定完成单元出现了明确的概率校准恶化。N4-Q1 的 logit 和 margin 增长最明显，AUROC却下降 `7.72 pp`，说明“分类器更自信”不等于未知排序更好。

N1 的 Q2/Q3 相比 Q1 没有显示辅助损失明显抑制或加剧过度自信；差异很小，且缺少 N4/N2 完整对照。置信诊断按预注册只用于解释，未参与 epoch 或方法选择。

### 8.3 已完成单元的梯度比例

- N1-Q2：`relative` 的每 epoch batch 比值均值最大值为 `75.25232`，出现在 epoch 19；
- N1-Q3：`relative=40.82868`、`absolute=22.47617`、`separation=24.46231`；
- 4 个完成单元的最大连续违规计数均为 0。

这些数值是“每 epoch 内 batch 比值的均值再取 20 epoch 最大”，不是单 batch 最大值。它们不能抵消 N4-Q2 已经触发的连续三轮 100 倍硬失败。

## 9. Pilot gate、候选和 confirmation

预注册要求 Q2、Q3、Q4 各自具有 N1/N4/N2 三个 pair 的完整结果，才能计算三 pair 均值、至少 2/3 正向、身份下限和其他 gate。当前：

- Q1 缺 N2；
- Q2、Q3 缺 N4 和 N2；
- Q4 没有成功单元。

所以本轮不能执行 pilot gate，不能选择 `e2e_alignment_signal`、`absolute_alignment_signal` 或 `local_structure_signal` 中任何一个，也不能把局部结果拼成 `cssr_redesign_failed` 的“全候选指标失败”版本。

N4-Q2 的硬失败使 12 单元完整审计与 pilot gate 无法完成，因此本次 pilot 停止、其余在运行任务取消；confirmation 所需的“完整且审计通过的 pilot + 唯一候选”均不存在，N0/N3/N5/N6 的最多 8 项 confirmation 没有获得授权，也没有运行。结合“最后一次有边界修正”的预注册边界，当前 CSSR 主线不再继续。

## 10. 产物与哈希

4090 容器上的完整产物：

```text
/root/hrrp-runs/fg_mv_cssr_e2e_redesign_v2/smoke_gpu_e42ec8b/
/root/hrrp-runs/fg_mv_cssr_e2e_redesign_v2/pilot_gpu_e42ec8b/
```

本机长期保存的 summary-only 镜像（254 个文件，不含 checkpoint、逐样本预测和大型 manifest）：

```text
/Users/bytedance/Desktop/科研空间/artifacts/results/fg_mv_cssr_e2e_redesign_v2/pilot_gpu_e42ec8b_partial_summary/
/Users/bytedance/Desktop/科研空间/artifacts/results/fg_mv_cssr_e2e_redesign_v2/fg_mv_cssr_e2e_redesign_e42ec8b_partial_summary.tar.gz
```

摘要压缩包 SHA-256：

`068c134855f75826128ff7c406a821b06a98f424a4e6769cf428e0c41bebc938`

smoke phase：

- `_PHASE_SUCCESS.json` SHA-256：`59efe9c8546cad5f964e311c0f3e269142b13b0fba4c2f11693134d93d5eca68`；
- phase summary SHA-256：`440a3154ad6c03f26877521bcfaaf9a8c520f34c56af740a77937a243532b92b`；
- 成功封印内 artifact hash manifest SHA-256：`a16f3de4bfb70d1e27252160481b070809f9cc6210261dcf77b76a1d48dcb6ea`。

正式成功单元 `_SUCCESS.json` SHA-256：

| 单元 | SHA-256 |
|---|---|
| N1-Q1 | `f62b9e1179d7d1cc1992fe307363891d05f18d1ef14f42650902be079622ec7f` |
| N1-Q2 | `eb77f8963905eeb4259927731bb5f3d0cc8a1b792b38171a539b8a96ad1b3719` |
| N1-Q3 | `f6d53598fc5188413a61ccfcfd635023c8249f631ce33732130ac8dfd45abd8b` |
| N4-Q1 | `7fc5d725444552acb2bc3f392b27fc9cae89ac6facdfa77acc62130ed2561a32` |

对应四份单元审计日志 SHA-256 依次为：

```text
N1-Q1  6e51753a797d21a1f96583bf9a998937fb3d15604f3c8a99b2eb6f6e280e2b08
N1-Q2  7a4a26d6dcf123de55a4e3e3d5b9ad99f344d7704aa7c549cc534307e3eb349f
N1-Q3  bd157e6f7938c675b469bac41beb1e3f7c233b649f1f272d0af4fc5025c0c5ae
N4-Q1  f3dcd707fb76212939eec55bea6b07c4bebdc58b98c637ff6d298bb751c8bfdd
```

原始数据、checkpoint、manifest、逐样本预测、结果和日志均未提交 Git。

## 11. 按交接任务的 14 项要求答复

1. **12 项 pilot 是否完整：**否；4 项成功并审计通过、1 项训练稳定性硬失败、3 项取消、4 项未启动。
2. **Q0–Q4 的完整指标：**无法提供，因为 pilot 未完成；第 6 节只列可审计的部分结果，Q4 无正式成功结果。
3. **Q1 相对 Q0：**N1 AUROC `+8.37 pp`，N4 `−7.72 pp`；额外 CE 微调的开集影响明显依赖 pair，尚无普遍改善结论。
4. **Q2 相对 Q1：**N1 AUROC `−3.42 pp`，N4-Q2 触发相对重构梯度硬失败；端到端相对重构没有形成稳定有效证据。
5. **Q3 相对 Q2：**只有 N1 可比，AUROC `+4.96 pp`；由于 Q3 同时加入绝对重构和分离约束，不能拆分归因，也不能外推。
6. **Q4 相对 Q3：**无法评价；Q4 没有正式成功单元。
7. **选出哪个标签：**没有选出标签，`selected_method=null`；pilot gate 未执行，因此不能使用预注册中代表完整 gate 失败的 `cssr_redesign_failed`。
8. **是否运行最多 8 项 confirmation：**否；未获授权。
9. **新 pair 上是否通过：**不适用；N0/N3/N5/N6 均未运行。
10. **DDG 双向相互吸收是否同时缓解：**没有证据支持；N1 CSSR 相对 Q1 放过更多 DDG-112，N4 CSSR 无成功结果。
11. **是否仍出现某身份接近完全反向的 AUROC：**已完成结果中最低身份 AUROC为 N4-Q1 的 DDG-1000 `62.38%`，未接近完全反向；但 pilot 不完整，不能回答全部方法和 pair。
12. **NLL/ECE/logit/feature norm 是否显示过度自信：**logit、margin 和 norm 上升，但 NLL/ECE没有同步恶化；不能确认明确的概率校准恶化。N4-Q1 说明置信锐化可能伴随开集 AUROC下降。
13. **是否建议继续 CSSR：**不建议继续当前 CSSR 主线，也不建议事后改权重、损失、AE 或移除困难 pair 来挽救本轮。若未来重启，必须作为新的研究假设和独立预注册任务。
14. **最终 unknown、偶数角 test 和 ARPL：**均未运行、未生成、未实例化，也不得由本轮自动授权。

## 12. 停止边界与下一步

本任务在训练稳定性硬失败报告后停止，不运行 confirmation，不进入最终 unknown，不做调参或替代实验。

下一项最小工作不是继续跑 CSSR，而是让研究决策回到方法层面：保留本轮为“端到端相对重构的训练稳定性存在跨 pair 不一致，现有局部性能不足以证明泛化”的负面对照；随后由用户决定是否回到已验证的类别条件 MLS 路线，或为另一类开集机制建立全新的、独立预注册。任何新方向都不得复用本轮的局部 N1 数字作为候选选择依据。
