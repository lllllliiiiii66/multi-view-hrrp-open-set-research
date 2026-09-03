# CSSR 梯度病理审计与解耦式多视角类别语义重构结果

> 日期：2026-09-04
>
> 阶段：P3 独立快速机制审计与验证
>
> 实验：`fg_mv_cssr_decoupled_audit_v3`
>
> 分支：`codex/fg-mv-cssr-decoupled-audit`
>
> 产生性能的冻结代码提交：`eb17466ff41efaf15f555c545da4ce207f8ddb96`
>
> 配置 SHA-256：`b67f84dda0754b9b628ce046beb1b02bc8d7e15e0764bb03889bc6865ece5f7c`
>
> 冻结运行：`angle_fold=0`，`R2_seed=20260830`，`audit_seed=20260904`，`cssr_seed=20260905`
>
> 最终状态：`decoupled_cssr_failed`；`selected_method=null`；confirmation 未运行且 `not_evaluated`

## 1. 明确结论

阶段 A 的 N1/N4 梯度审计和阶段 B 的 6 项 pilot 均已完整运行并通过独立产物审计。旧 N4-Q2 的 100 倍门槛在冻结的 5 epoch 审计窗口内没有复现，因此其历史触发原因仍然**无法判断**；当前审计只排除了本次运行中的“小 CE 分母、持续真实辅助梯度主导、强梯度冲突、频繁裁剪、异常参数更新和快速 calibration 漂移”，不能把旧硬失败改写为误报或把旧 Q2 恢复为有效结果。

解耦 pilot 中，D1 相对 D0 的三 pair 平均 AUROC 下降 `0.49 pp`，只有 `1/3` pair 为正。D2 平均 AUROC 提高 `2.05 pp`，并同时改善 DDG 双向吸收，但 N2 的 MARVEL CRANE 身份 AUROC 从 `92.84%` 降至 `47.27%`，下降 `45.56 pp`，远超预注册允许的 `10 pp`。因此 D1、D2 均未通过 pilot gate，唯一机器决定为：

```text
pilot_gate = decoupled_cssr_failed
selected_method = null
confirmation_allowed = false
automatic_followon_authorized = false
confirmation_status = not_evaluated
final_unknown_test_authorized = false
```

**已排除共享CE末层梯度竞争，并为CSSR建立独立语义适配空间；该机制仍不能稳定超过类别条件MLS，因此停止当前CSSR主线。**

该结论只适用于本轮固定的 adapter、类别局部 AE、D1/D2 损失和 guided score，不外推为“所有类别重构方法无效”。

## 2. 冻结设计与证据边界

本轮严格执行预注册，不根据结果修改网络、损失、数据、训练时长、分数或 gate。

| 项目 | 冻结定义 |
|---|---|
| 数据范围 | fold 0 的 source-known 奇数角开发池；最终 3 unknown 和偶数角 test 未生成、未使用 |
| R2 来源 | 已审计 `R2_MS_MEAN_CE` epoch-100，正式代码提交 `62e318de82b4221b599e06b1166483673e9c1cd3` |
| R2 分类路径 | encoder、BN 参数及 running buffers、projection、CE head 全部冻结并始终 `eval` |
| Pilot pair | N1、N4、N2 |
| D0 | 冻结 R2 + 类别条件 MLS；直接复用，不训练 |
| D1 | 共享 CSSR adapter + 每类独立 kernel-3 AE；仅 `L_rel` |
| D2 | D1 + `0.25 L_abs + 0.5 L_sep`，margin 固定 `0.2` |
| CSSR 训练数据 | 每单元 5 类 × 144 条唯一 train-known 单视角底层样本，共 720 条；每样本每 epoch 恰好一次 |
| CSSR 训练 | AdamW，20 epoch；adapter 在 epoch 1–5 冻结、6–20 解冻；epoch 20 为唯一正式 checkpoint |
| 已知预测 | 始终使用冻结 R2 的融合 CE `argmax`；D0/D1/D2 不改变已知预测 |
| CSSR unknown score | 对 R2 提出的同一身份逐视角验证，两个视角异常度取均值；越大越未知 |
| 阈值 | 各方法只由 known calibration 按 95% known acceptance 确定 |
| 排除项 | 无 ARPL、伪未知、GAN、Transformer、attention、额外 seed、第二 fold、分数融合或事后调参 |

旧实验 `fg_mv_cssr_e2e_redesign_v2` 的状态保持不变：

```text
pilot_status = hard_failed_incomplete
pilot_gate = not_evaluated
selected_method = null
```

阶段 A 是独立解释性审计，阶段 B 是新结构验证；两者均未用于“补完”旧实验。`RESEARCH_CONTEXT.md` 未修改。

## 3. 运行、检查与审计链

运行前后的检查均通过：

- 本地完整测试：`491 passed, 10 skipped`；跳过项仅因本机无 CUDA；
- 4×RTX 4090 容器完整测试：`501 passed`，CUDA 专项实际执行；
- Python compile、配置校验和 `git diff --check`：通过；
- 阶段 A 两个单元、smoke 两个单元、pilot 六个单元均完成；
- 阶段 A、smoke、pilot 的 aggregate 与独立 audit 均通过；
- checkpoint 重放、指标反算、pair/标签顺序、R2 来源与源码哈希绑定均通过；
- pilot gate 失败后没有启动 N0/N3/N5/N6 confirmation。

正式 GPU 环境为 Python `3.12.3`、PyTorch `2.7.0a0+7c8ec84dab.nv25.03`、NumPy `1.26.4` 和 4×NVIDIA GeForce RTX 4090；确定性算法开启，TF32 与 cuDNN benchmark 关闭。

## 4. 阶段 A：梯度审计结果

N1/N4 均固定运行 5 epoch，每 epoch 12 batch，共 60 batch；原 Q2 的 R2 checkpoint、1×1 类别 AE、初始化、动态 pair schedule、优化器、学习率、损失 `L_cls + 0.5 L_rel`、可训练范围、batch size 和裁剪设置均保持不变，仅增加诊断记录。

| 诊断 | N1-Q2 | N4-Q2 |
|---|---:|---:|
| `||g_cls||` min / median / max | 0.001222 / 0.005234 / 0.070141 | 0.000543 / 0.001988 / 0.020950 |
| `||g_rel_weighted||` min / median / max | 0.031106 / 0.067556 / 0.253858 | 0.048372 / 0.084613 / 0.210972 |
| clip 前总梯度 min / median / max | 0.055900 / 0.128968 / 0.487060 | 0.077585 / 0.169797 / 0.423339 |
| cosine min / median / max | -0.191756 / 0.061639 / 0.501689 | -0.170743 / 0.075443 / 0.244614 |
| A：各 epoch mean-of-batch ratio | 18.32 / 12.74 / 13.12 / 18.73 / 22.97 | 35.35 / 37.72 / 55.33 / 82.24 / 61.02 |
| B：ratio-of-mean-norms 最大值 | 17.65 | 60.80 |
| C：RMS ratio 最大值 | 16.45 | 54.10 |
| `||g_cls||<1e-4` batch | 0/60 | 0/60 |
| 实际 clipping batch | 0/60 | 0/60 |
| last-stage 单 epoch 最大相对更新 | 0.001864 | 0.002110 |
| calibration Accuracy 范围 | 98.60%–98.84% | 99.08%–99.20% |
| calibration NLL 范围 | 0.04515–0.05029 | 0.02561–0.02741 |
| 原 100× gate 在 5 轮内触发 | 否 | 否 |
| 冻结规则标签 | `inconclusive` | `inconclusive` |

已确认：两组都没有小 CE 分母证据、持续 100 倍的稳健辅助主导、绝对辅助梯度异常、频繁裁剪、异常参数更新、强负向冲突、快速 calibration 漂移或非有限值。少量 batch 的 cosine 为负不等于长期冲突；两组中位数均为正，且未达到冻结的冲突判据。

尚未验证：旧 N4-Q2 为何在原 20 epoch 流程中连续三轮超过 100。冻结审计只允许 5 epoch，且历史触发没有在这一窗口重现；因此不能判断触发发生在更晚 epoch、来自未保存的历史运行状态，还是其他尚未定位的因素。

## 5. Diagnostic smoke

N1-D1/D2 各运行 6 epoch，完整覆盖 adapter 冻结的 epoch 1–5 和首次解冻的 epoch 6。两个单元均完成，checkpoint 重放精确、R2 前后状态不变、D1/D2 schedule hash 一致，aggregate 与独立 audit 均通过。smoke 仅验证链路，不进入性能 gate，也不支持任何方法效果结论。

## 6. Pilot 的 D0/D1/D2 完整结果

以下为每个 pair 的正式结果，单位为百分比；FPR95 越低越好，其余指标越高越好。阈值保留原始分数尺度，因此 D0 的 MLS 阈值与 D1/D2 的 CSSR 阈值不可横向比较。

| Pair | 方法 | Known Acc. | Known F1 | AUROC | OSCR | FPR95 ↓ | KCCR | URR | H | K+1 F1 | threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N1 | D0 | 98.080 | 98.084 | 86.962 | 85.901 | 64.040 | 94.520 | 63.600 | 76.037 | 88.393 | 0.964361 |
| N1 | D1 | 98.080 | 98.084 | 85.895 | 84.857 | 52.320 | 94.480 | 55.600 | 70.004 | 86.650 | 2.138333 |
| N1 | D2 | 98.080 | 98.084 | 95.969 | 94.628 | 18.760 | 94.280 | 75.900 | 84.097 | 90.769 | 2.138333 |
| N4 | D0 | 99.320 | 99.321 | 83.005 | 82.740 | 60.880 | 94.560 | 31.500 | 47.257 | 79.342 | 0.953252 |
| N4 | D1 | 99.320 | 99.321 | 85.749 | 85.321 | 64.160 | 94.480 | 47.900 | 63.571 | 83.437 | 2.061258 |
| N4 | D2 | 99.320 | 99.321 | 93.769 | 93.289 | 27.760 | 94.760 | 72.800 | 82.341 | 89.903 | 2.085653 |
| N2 | D0 | 98.560 | 98.564 | 87.207 | 86.971 | 59.200 | 94.960 | 49.900 | 65.422 | 84.564 | 0.964000 |
| N2 | D1 | 98.560 | 98.564 | 84.049 | 83.030 | 83.560 | 93.800 | 55.200 | 69.500 | 84.746 | 2.061258 |
| N2 | D2 | 98.560 | 98.564 | 73.587 | 72.778 | 92.480 | 93.840 | 51.800 | 66.752 | 83.903 | 2.061258 |

D0/D1/D2 在同一 pair 上的 Known Accuracy、Known Macro-F1、R2 fused logits 和预测逐元素一致，确认 CSSR 没有进入或改变冻结分类路径。

R2 全局 MLS 仅作背景，不参与选择：N1/N4/N2 的 AUROC 分别为 `77.35% / 80.16% / 89.86%`，OSCR 为 `77.16% / 79.98% / 89.34%`，FPR95 为 `37.72% / 42.36% / 43.68%`。

## 7. 受控差值与 pilot gate

### 7.1 D1、D2 相对 D0

| 候选 | 平均 ΔAUROC | AUROC 正 pair | 平均 ΔOSCR | 平均 ΔKCCR | 平均 ΔFPR95 | 最低 identity AUROC | 最差 identity ΔAUROC | Pilot gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| D1−D0 | -0.494 pp | 1/3 | -0.801 pp | -0.427 pp | +5.307 pp | 68.098% | -24.739 pp，N2 MARVEL | 不通过 |
| D2−D0 | +2.050 pp | 2/3 | +1.695 pp | -0.387 pp | -15.040 pp | 47.272% | -45.564 pp，N2 MARVEL | 不通过 |

D1 未达到平均 AUROC、正向 pair、OSCR、FPR95 和 identity 稳定性门槛，不能说明“解耦相对重构空间”稳定有效。

D2 达到平均 AUROC、正向 pair、OSCR、KCCR、FPR95 和最低 identity AUROC 门槛，但违反“任一 identity 相对 D0 不得下降超过 10 pp”：N2 MARVEL 从 `92.836%` 降到 `47.272%`。因此绝对重构与分离约束显示了强烈但 pair/identity 依赖的局部作用，不能认定为稳定有效。

### 7.2 D2 相对 D1

D2 相对 D1 的三 pair 平均 AUROC 增加 `2.544 pp`，N1/N4/N2 分别为 `+10.075 / +8.020 / -10.462 pp`；平均 OSCR 增加 `2.496 pp`，平均 FPR95 改善 `20.347 pp`，但 N2 方向明显相反。该对照支持“绝对项与分离项会显著改变结果”，不支持“它们已产生稳定的额外收益”。由于 D2 同时加入 `L_abs` 与 `L_sep`，本实验也不能拆分两者的单独贡献。

## 8. Identity 结果与错误吸收

每格为 `AUROC / URR / FPR95`，均为百分比。

| Pair | Surrogate identity | D0 | D1 | D2 |
|---|---|---:|---:|---:|
| N1 | DDG-112 | 74.244 / 27.2 / 72.28 | 72.069 / 11.2 / 59.80 | 92.079 / 51.8 / 26.84 |
| N1 | 迷你好望角型散货船 | 99.680 / 100.0 / 0.64 | 99.720 / 100.0 / 0.56 | 99.860 / 100.0 / 0.28 |
| N4 | DDG-1000 | 77.479 / 22.8 / 74.96 | 82.582 / 44.4 / 67.44 | 95.231 / 78.2 / 18.52 |
| N4 | 集装箱船达飞罗尔多夫级 | 88.531 / 40.2 / 31.72 | 88.915 / 51.4 / 46.64 | 92.307 / 67.4 / 46.28 |
| N2 | 油气轮 MARVEL CRANE | 92.836 / 62.8 / 24.84 | 68.098 / 10.4 / 96.04 | 47.272 / 3.6 / 97.04 |
| N2 | 迷你好望角型散货船 | 81.578 / 37.0 / 65.68 | 100.000 / 100.0 / 0.00 | 99.902 / 100.0 / 0.56 |

DDG 双向相互吸收相对 D0 的结果为：

| 方向（每身份 500 条） | D0 | D1 | D2 |
|---|---:|---:|---:|
| N1：DDG-112 → DDG-1000 | 364 | 444 | **241** |
| N4：DDG-1000 → DDG-112 | 386 | 278 | **109** |

D1 只改善第二个方向，却把第一个方向从 364 恶化到 444；D2 将两个方向分别降到 241 和 109，确认两向同时改善。但这不能抵消 N2 MARVEL 的严重退化。

其余 false accept 去向：

- N1 迷你好望角：D0/D1/D2 均为 0；
- N4 达飞罗尔多夫：D0 为 `CVN77:23 / MARVEL:44 / 爱达魔都:232`，共 299；D1 为 `51 / 72 / 120`，共 243；D2 为 `37 / 46 / 80`，共 163；
- N2 MARVEL：D0 为 `CVN77:75 / 爱达魔都:25 / 达飞罗尔多夫:86`，共 186；D1 为 `131 / 148 / 169`，共 448；D2 为 `157 / 150 / 175`，共 482；
- N2 迷你好望角：D0 的 315 个 false accept 全部流向 DDG-1000，D1/D2 均为 0。

因此 D2 的主要问题不是平均指标不足，而是对不同 identity 的作用方向不稳定：它显著改善 N1/N4 和 DDG 互吸，却几乎完全放过 N2 MARVEL。

## 9. 训练行为与完整性检查

六个训练单元均固定完成 20 epoch，没有性能早停：

| 单元 | 最终 train CSSR Acc. | 最终 cal CSSR Acc. | `L_rel` | `L_abs` 诊断 | `L_sep` 诊断 |
|---|---:|---:|---:|---:|---:|
| N1-D1 | 88.89% | 80.00% | 0.7044 | 1.5782 | 0.05680 |
| N1-D2 | 91.39% | 82.78% | 0.7866 | 0.8677 | 0.04335 |
| N4-D1 | 95.00% | 88.89% | 0.5063 | 1.4958 | 0.02350 |
| N4-D2 | 95.97% | 90.56% | 0.5394 | 0.7879 | 0.01815 |
| N2-D1 | 83.06% | 79.44% | 0.7542 | 1.3811 | 0.10961 |
| N2-D2 | 83.75% | 80.56% | 0.7937 | 0.8014 | 0.08218 |

这些 CSSR 内部分类数值只用于训练诊断，不替代正式 OSR 指标，也不能解释为方法已学到可泛化的拒识空间。

已经确认：adapter 在 epoch 1–5 的梯度与更新均为 0，epoch 6 后梯度和更新均为正；六项最大 clipping fraction 均为 0；`U` 未坍缩为常数，最终 effective rank 约为 `3.45–4.55`；R2 参数和 BN buffers 训练前后哈希一致。D1/D2 在同一 pair 上的初始状态哈希相同：

`d88b8401d02a26b15df933b586b5e0161758e6fd75982bf87a065a5fa00c63b8`

D1/D2 的 20 epoch schedule hash 逐 pair 一致；D0 确认为复用且未训练；六个 checkpoint 均 bitwise exact 重放；九项指标均从逐样本预测零误差反算。

## 10. 产物、哈希与停止边界

完整远端产物与日志位于：

```text
/root/hrrp-runs/fg_mv_cssr_decoupled_audit_v3_eb17466/stage_a_gradient/
/root/hrrp-runs/fg_mv_cssr_decoupled_audit_v3_eb17466/stage_b_smoke/
/root/hrrp-runs/fg_mv_cssr_decoupled_audit_v3_eb17466/stage_b_pilot/
/root/hrrp-run-logs/fg_mv_cssr_decoupled_audit_v3_eb17466/
```

冻结 R2 来源位于：

```text
/root/hrrp-runs/ms_mean_head_factorial_surrogate_v1/confirmation_gpu_62e318d/
```

本机 summary-only 镜像与压缩包位于：

```text
/Users/bytedance/Desktop/科研空间/artifacts/results/fg_mv_cssr_decoupled_audit_v3/fg_mv_cssr_decoupled_audit_v3_eb17466/
/Users/bytedance/Desktop/科研空间/artifacts/results/fg_mv_cssr_decoupled_audit_v3/fg_mv_cssr_decoupled_audit_v3_eb17466_summary.tar.gz
```

压缩包 SHA-256 为 `7ee8f00fc1a8d710239fef95b2caec9645e8f5cd69e7296de27d42c15b073699`。本机镜像不含 checkpoint、逐样本大表和大型 manifest；完整产物仍保留在远端，checkpoint 重放与逐样本指标反算的证据来自远端已通过的独立 audit。

关键封印与汇总哈希：

| Phase / 文件 | SHA-256 |
|---|---|
| gradient audit `_PHASE_SUCCESS.json` | `79bb0aa3c7b047756d1d828d8758cd16d8ca288afbe8843c3ef7953b79519503` |
| gradient audit summary | `5772aa4e84f3d9e86b03ebe070f1203467db876c3602ae7e3c77d95f103c5cfa` |
| gradient audit artifact manifest | `f5d2f11e2d3c14928a4345e303dace311e27b52b49c0651051f395a3cd4f3203` |
| smoke `_PHASE_SUCCESS.json` | `0bdbc7837b8021704996c5ce0a595ff3b2548f612eba3d41b9fc99964e6ef426` |
| smoke phase summary | `b885a34f99f0a16eb0111508a2f6df54b7a8bc376558b7d61526b77fdb91557a` |
| smoke artifact manifest | `52d1d4b742c39577a2b9f8797085616feabe086f615000928932576b500b4574` |
| pilot `_PHASE_SUCCESS.json` | `e34f465a6881de68ae6e7c6c314c86b6b76f49cf8a9cad2ed166b2ac4c170c1c` |
| pilot phase summary | `bb35c8a22decd2e68d1601faeee9181583ab053ea9455838d522a11144d99f5f` |
| pilot gate | `b73daaedf419cf8704f3feeef99efe94293a0e4115799d79eed7f83ba4d92cc2` |
| pilot artifact manifest | `881e8c598c4ab9a6e53016a936009f73a23a969c155b91aa3ac12a88c64c237d` |

Pilot gate 已失败，因此严格停止：没有运行最多 4 项 confirmation，没有生成或使用最终 3 unknown、偶数角 test，没有运行 ARPL，也没有进入额外 seed、fold、网络或损失尝试。

## 11. 按交接任务的 13 项答复

1. **梯度审计是否完整：**是。N1/N4 各 5 epoch、60 batch，逐 batch、逐 epoch、异常路径、模型状态和产物哈希均保存并通过独立审计。
2. **N4 原 100×门槛由什么造成：**无法判断。原 100×现象在冻结的 5 epoch 内没有复现；只能排除本次运行中的小 CE 分母、持续真实辅助主导、强冲突、裁剪、异常更新和快速漂移，不能称其为误报。
3. **CE 梯度是否接近零：**否。N1/N4 的最小值分别为 `0.001222/0.000543`，所有 120 个 batch 均不低于 `1e-4`。
4. **CSSR 绝对梯度、clipping 和参数更新是否异常：**本次 5 轮内没有。weighted CSSR 梯度最大值为 `0.2539/0.2110`，两组均 `0/60` batch 裁剪，last-stage 单轮相对更新最大 `0.186%/0.211%`。
5. **CE 与 CSSR 梯度是否冲突：**没有持续强冲突证据。两组 cosine 中位数为正，未达到冻结的强冲突门槛。
6. **6 项 pilot 是否完整：**是，`N1/N4/N2 × D1/D2` 全部完成并通过 aggregate、checkpoint 重放、指标反算和独立 audit。
7. **D0/D1/D2 完整结果：**见第 6 节九项指标全表；identity 结果和 false accept 去向见第 8 节。
8. **D1 是否说明解耦空间有效：**否。平均 AUROC `-0.49 pp`，只有 `1/3` pair 为正，并违反多项 gate。
9. **D2 是否说明绝对重构对齐有效：**只显示明显局部作用，不显示稳定有效。平均 AUROC `+2.05 pp`，但 N2 MARVEL 下降 `45.56 pp`，因此 gate 失败。
10. **DDG 双向相互吸收是否同时改善：**D1 否；D2 是，`364→241` 和 `386→109`，但整体仍因 N2 身份退化失败。
11. **是否进入 4 项 confirmation：**否；pilot 没有选出唯一合格候选，`confirmation_allowed=false`。
12. **Confirmation 是否通过：**`not_evaluated`。它未获授权且未运行，不能写成 confirmation 失败。
13. **是否继续 CSSR：**不继续当前 CSSR 主线；保留为经过审计的负结果和局部机制证据，不自动进入 P4、ARPL、最终测试或其他新方法。

## 12. 当前决定

本实验在预注册停止边界处完成。当前不执行任何后续性能实验；如果未来研究另一类重构机制，必须提出新的、边界明确的研究假设并独立预注册，不能依据本轮局部 N1/N4 提升事后改 gate。
