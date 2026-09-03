# CSSR 梯度病理审计结果

> 日期：2026-09-04
>
> 阶段：P3 独立诊断审计
>
> 实验：`fg_mv_cssr_decoupled_audit_v3`，阶段 A
>
> 运行代码提交：`eb17466ff41efaf15f555c545da4ce207f8ddb96`
>
> 配置 SHA-256：`b67f84dda0754b9b628ce046beb1b02bc8d7e15e0764bb03889bc6865ece5f7c`
>
> 最终标签：N4 `inconclusive`；N1 对照 `inconclusive`

## 1. 明确结论

N1-Q2 与 N4-Q2 的 5 epoch 梯度审计均完整运行并通过独立复核，每个单元保存了 `60` 个 batch 的诊断，batch 到 epoch 的统计重算完全一致。两者都没有在 5 轮内再次触发旧的“连续 3 个 epoch 比例超过 100”门槛，也没有代码错误、非有限值或审计失败。

因此，**现有证据无法确定旧 N4-Q2 的 100 倍门槛究竟由什么造成**。本轮只能确认：在冻结的相同起点、初始化、前 5 轮动态 schedule 和训练设置下，该现象没有重现；同时没有观察到小 CE 梯度分母、持续的真实辅助梯度主导、异常大的辅助绝对梯度、频繁裁剪、过大的末层更新、长期强负向梯度冲突或快速 calibration 漂移。N1 和 N4 的冻结标签都只能是 `inconclusive`。

这不是对旧硬失败的推翻，也不能写成旧门槛“误报”。旧实验 `fg_mv_cssr_e2e_redesign_v2` 继续保持：

```text
pilot_status = hard_failed_incomplete
pilot_gate = not_evaluated
selected_method = null
```

阶段 A 只用于解释训练现象，不进入 CSSR 方法性能 gate，也不证明旧 Q2 有效。

## 2. 冻结设计

审计严格复用旧 Q2 的 R2 checkpoint、1×1 类别 AE、初始化、动态 pair schedule、优化器、学习率、可训练层、batch size、裁剪、数值设置和损失：

```text
L = L_cls + 0.5 * L_rel
```

唯一行为差异是：审计模式不因旧 100 倍规则立即抛异常，而是继续固定运行 5 epoch，并保存 `would_have_triggered_original_100x_gate`。NaN/Inf、optimizer error、参数非有限、数据或 schedule 不一致仍会立即失败。审计对象固定为 N4-Q2，N1-Q2 作为同方法对照；每轮 `12` 个 batch，每个单元共 `60` 个 batch。

冻结来源包括：

| 项目 | N1 | N4 |
|---|---|---|
| R2 checkpoint SHA-256 | `a4f6fa3235fbb5cf74b712588a0318f614a05287adec4ee881820424cddbcbaa` | `169387ad7a87463110ac7a2cd45afd7dac49428538c93c84975162e425d94ff5` |
| epoch-0 R2 common-state SHA-256 | `3a7e74e89d11812877409d415c505c087a913e3b08098ab1fa583fa1151aeb07` | `67825e2bc143b32ed52beb5778e1190bc809269dfaedfb516a692eac64fa31f2` |

两单元共享的 Q2 AE 初始状态 SHA-256 为 `4c3257678eaca1bd20ea2e97b09aeaf87fdd23c71ff65dcc1696dfe61963d6fb`。旧 v2 配置 SHA-256 为 `5c227c00a7ac5a88c9bf5d66618964bc05c67f45c51c2a880731f6753626512e`。

## 3. 实际梯度结果

下表均为共享最后 residual stage 上 `60` 个 batch 的统计。`weighted CSSR` 指 `grad(0.5 L_rel)`；更新量为每个 epoch 前后参数相对变化的最大值。

| 统计 | N1-Q2 | N4-Q2 |
|---|---:|---:|
| CE 梯度 norm，min / median / max | `0.001222 / 0.005234 / 0.070141` | `0.000543 / 0.001988 / 0.020950` |
| weighted CSSR 梯度 norm，min / median / max | `0.031106 / 0.067556 / 0.253858` | `0.048372 / 0.084613 / 0.210972` |
| 裁剪前总梯度 norm，min / median / max | `0.055900 / 0.128968 / 0.487060` | `0.077585 / 0.169797 / 0.423339` |
| CE 与 weighted CSSR cosine，min / median / max | `-0.191756 / 0.061639 / 0.501689` | `-0.170743 / 0.075443 / 0.244614` |
| 实际发生裁剪 | `0/60` | `0/60` |
| `CE gradient < 1e-4` 的 batch 比例 | 每轮均为 `0` | 每轮均为 `0` |
| last residual stage 最大相对更新 | `0.001864` | `0.002110` |
| calibration Accuracy 范围 | `98.60%–98.84%` | `99.08%–99.20%` |
| calibration NLL 范围 | `0.04515–0.05029` | `0.02561–0.02741` |

所有 CE 梯度都高于 `1e-4`；因此本轮没有“小 CE 分母”证据。weighted CSSR 梯度绝对值有限，没有接近冻结的“大绝对梯度”判据；总梯度也从未触发裁剪。末层每轮相对更新最大为 N1 的 `0.186%` 和 N4 的 `0.211%`，均低于异常阈值 `1%`。

四个参数组在任一单 epoch 中的最大相对更新如下。冻结的“异常更新”判据只针对共享 last residual stage，不适用于学习率和角色不同的 projection、CE head 与 CSSR AE，后三组数值仅作完整记录。

| 参数组最大单轮相对更新 | N1-Q2 | N4-Q2 |
|---|---:|---:|
| last residual stage | `0.001864` | `0.002110` |
| projection | `0.004079` | `0.004997` |
| CE head | `0.009899` | `0.011640` |
| CSSR AE | `0.085152` | `0.091518` |

epoch 0 时，N1 的 calibration Accuracy/NLL/ECE 为 `98.08% / 0.05920 / 0.00915`，N4 为 `99.32% / 0.02787 / 0.00828`。随后 5 轮中，N1 Accuracy 为 `98.60%–98.84%`、NLL 为 `0.04515–0.05029`；N4 Accuracy 为 `99.08%–99.20%`、NLL 为 `0.02561–0.02741`。相对 epoch 0 均未达到 Accuracy 下降 `2 pp` 或 NLL 增加 `0.1` 的快速漂移条件。epoch 0 的平均最大 logit / 融合 feature norm 分别为 N1 `5.54683 / 10.65590`、N4 `5.73445 / 11.14747`；完整逐轮诊断保留在受哈希保护的产物中。

部分 batch 的 cosine 为负，说明局部存在方向相反，但两个单元的总体中位数均为正，且没有达到“至少 3 个 epoch 中位数不高于 `-0.25` 且负值比例至少 `75%`”的强冲突条件。因此只能说**未观察到持续强冲突**，不能说两个损失在所有 batch 都同向。

## 4. 三种梯度比例与旧 100 倍规则

旧规则只依据每个 epoch 的 `mean of batch ratios`，并要求连续 3 轮超过 `100`。本轮结果为：

| Epoch | N1 mean of batch ratios | N4 mean of batch ratios |
|---:|---:|---:|
| 1 | `18.3229` | `35.3504` |
| 2 | `12.7418` | `37.7218` |
| 3 | `13.1233` | `55.3308` |
| 4 | `18.7277` | `82.2360` |
| 5 | `22.9705` | `61.0204` |

| 跨 5 轮最大值 | N1-Q2 | N4-Q2 |
|---|---:|---:|
| mean of batch ratios | `22.9705` | `82.2360` |
| ratio of mean norms | `17.6489` | `60.7966` |
| RMS ratio | `16.4532` | `54.0982` |

N4 的相对比例明显高于 N1，但三种口径都没有超过 `100`，更没有连续 3 轮超过 `100`。因此：

```text
would_have_triggered_original_100x_gate = false
not_triggered_within_5_epochs = true
first_trigger_epoch = null
```

“未在 5 轮内触发”只覆盖预注册窗口。不得由此推断第 6–20 轮也不会触发，也不得追加 epoch 来追查旧现象。

## 5. 冻结判据逐项结论

| 判据 | N1 | N4 | 本轮含义 |
|---|---|---|---|
| 小 CE 分母 | 否 | 否 | 没有大量 batch 的 CE 梯度接近零 |
| 稳健辅助主导 | 否 | 否 | ratio-of-means / RMS 未连续达到冻结阈值 |
| 辅助绝对梯度过大 | 否 | 否 | weighted CSSR 绝对 norm 未达异常尺度 |
| 频繁 clipping | 否 | 否 | 两单元均 `0/60` |
| 参数更新过大 | 否 | 否 | last-stage 最大更新均低于 `1%` |
| 持续强梯度冲突 | 否 | 否 | 有局部负 cosine，但未形成冻结规则所指的长期强冲突 |
| calibration 快速漂移 | 否 | 否 | Accuracy/NLL 未达到冻结恶化门槛 |
| 数值异常 | 否 | 否 | 无 NaN/Inf、optimizer error 或非有限参数 |
| 最终标签 | `inconclusive` | `inconclusive` | 5 轮证据不足以解释旧 N4 触发原因 |

阶段 B 的放行条件只是“阶段 A 没有代码或数值失败”，不是“旧 Q2 已被证明稳定”。因此本阶段审计通过后允许继续独立的解耦 CSSR 验证，但阶段 B 必须按自己的性能 gate 单独作结论。

## 6. 完整性审计与证据边界

已经确认：

- N1/N4 的 R2 checkpoint、旧 logits、pair manifest、标签顺序、Q2 初始状态和前 5 轮 schedule 均与冻结来源一致；
- 每轮 720 条训练 pair、12 个 batch，两个单元各保存 5 轮、60 个 batch 和 25 项受哈希保护的单元产物；
- batch 诊断到 epoch 聚合的独立重算完全一致；
- projection 与 CE head 没有接收到 `L_rel` 的非零梯度；冻结前缀保持不变，训练参数均有限；
- 本阶段是 `diagnostic_only`，`performance_gate_eligible=false`；
- 没有生成或使用最终三类 unknown、偶数角 test，也没有用 surrogate unknown 作训练或方法选择。

本轮没有确认：

- 旧 N4-Q2 在原 20 epoch 进程中为何会出现连续三轮超过 100；
- 该现象是否只在第 6 轮以后出现，或是否来自当前未被 5 轮窗口覆盖的运行状态；
- 旧 Q2 的开集性能是否有效；阶段 A 不是性能实验。

正式环境为 Python `3.12.3`、PyTorch `2.7.0a0+7c8ec84dab.nv25.03`、NumPy `1.26.4` 和 NVIDIA GeForce RTX 4090；确定性算法开启，TF32 与 cuDNN benchmark 关闭。

## 7. 产物与哈希

GPU 容器完整产物根目录：

```text
/root/hrrp-runs/fg_mv_cssr_decoupled_audit_v3_eb17466/
```

阶段 A 聚合目录为 `<实验根>/stage_a_gradient/`，单元位于：

```text
stage_a_gradient/gradient_pathology_audit/{N1,N4}/fold_0/seed_20260904/Q2_E2E_REL_CSSR_1X1/
```

本机另保存了不覆盖旧实验的 summary 镜像：

```text
/Users/bytedance/Desktop/科研空间/artifacts/results/fg_mv_cssr_decoupled_audit_v3/fg_mv_cssr_decoupled_audit_v3_eb17466/
```

阶段 A 封口文件：

- `_PHASE_SUCCESS.json` SHA-256：`79bb0aa3c7b047756d1d828d8758cd16d8ca288afbe8843c3ef7953b79519503`；
- `gradient_pathology_audit_summary.json` SHA-256：`5772aa4e84f3d9e86b03ebe070f1203467db876c3602ae7e3c77d95f103c5cfa`；
- `artifact_hashes.json` SHA-256：`f5d2f11e2d3c14928a4345e303dace311e27b52b49c0651051f395a3cd4f3203`。

原始数据、checkpoint、manifest、逐 batch 诊断、结果和日志保留在容器独立目录，不提交 Git；Git 只保存代码、冻结配置、测试与本报告。`RESEARCH_CONTEXT.md` 和旧 v2 结果均未修改。
