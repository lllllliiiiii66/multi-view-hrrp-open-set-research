# MV-RPFormer surrogate OSR 正式 GPU confirmation 结果

> 日期：2026-09-03
>
> 阶段：P3（面向开集的多视角表示学习）
>
> 状态：GPU development 及 confirmation 已完成，完整性审计通过；M6 未通过预注册主方法门槛
>
> 证据范围：7 个 source-known 类的奇数角 surrogate OSR；不是最终 7-known/3-unknown 或偶数角 test 结果

## 1. 直接结论

正式 GPU confirmation 共完成 `C0–C3 × 3 seeds × M0–M7 = 96` 个任务，聚合、逐样本反算、完整性审计和最终判定均成功。按照运行前冻结的规则：

- `m6_main_method_success = false`；
- `arpl_specific_success = true`；
- `freeze_m6_for_final_test = false`；
- `final_unknown_test_authorized = false`。

因此，**当前完整 MV-RPFormer（M6）不能冻结为最终方法，也不得运行最终 3 个未知类或偶数角 test**。失败原因不是闭集分类能力不足，而是相对 M4 的未知排序增益很小且不稳定：平均 AUROC 仅提高 `0.45 pp`，只有 `7/12` 单元为正，同时 known Accuracy 平均下降 `1.03 pp`、FPR95 平均恶化 `3.15 pp`。

本轮最强且跨单元最一致的正向结果来自更简单的 **M2：多尺度编码器 + mean pooling + global ARPL**。M2 相对 M1 在 `12/12` 单元提高 AUROC，平均提高 `9.18 pp`，FPR95 平均降低 `20.95 pp`；其 confirmation 平均 AUROC 为 `76.22%`，高于 M6 的 `64.25%`。按预注册的组件对照定义，这支持保留 M2 所代表的多尺度 backbone 路线，但现有 Set Transformer、分层 ARPL 和伪未知拒判器的组合没有带来稳定的进一步收益。

## 2. 冻结设计与实际运行

| 项目 | 实际值 |
|---|---|
| 原始科学预注册 | commit `69a1d3e82756dd781e785ddca1c85c3d1ee4037b` |
| GPU 兼容源码基线 | commit `2ee8f0d6b8f437786677055d4bc73c41891029b9` |
| 实际运行快照 | commit `1778d0385f371698f4f283b850ae13f7672dd151`，clean detached HEAD |
| 配置 | `configs/experiments/arpl/mv_rpformer_surrogate_v1.yaml` |
| 配置 SHA-256 | `66fe6c9fa556f2fcba1a5325163d28268570c243018e7a41be99e051a7c7ec23` |
| 数据 profiles SHA-256 | `2dd92282c125f0f677cf1f2dfce828781c8ba4385cf9ae552c4a2c56033c3f5b` |
| 数据 manifest SHA-256 | `748b9f30629c3b3cbe66c6a1dac30863fdab2d81a214e46d8bc3ef7c6022a08a` |
| 数据 bundle SHA-256 | `79176091b5b745e5df5957f1bb4ade7304c1403a4a32e64622e012db67d5a0c5` |
| 训练环境 | 4 × NVIDIA GeForce RTX 4090；PyTorch `2.7.0a0+7c8ec84dab.nv25.03`；CUDA `12.8`；NumPy `1.26.4` |
| 调度 | 每卡 4 个独立任务，全机峰值 16 并发；每任务 4 个 intra-op、1 个 inter-op 线程 |
| development | `S0–S2 × seed 20260830 × M0–M7 = 24/24` |
| confirmation | `C0–C3 × 3 seeds × M0–M7 = 96/96` |

GPU 兼容修改只有两项：将无参数的 `AdaptiveMaxPool1d(1)` 改为前向和并列最大值梯度语义相同、可确定性反向传播的 `torch.max(...).values`；在 NumPy 1.x 环境下为梯形积分使用等价的 `np.trapz` 回退。两项都在产生正式结果前完成并测试，没有改变模型参数、训练预算、损失、阈值或指标定义。详细边界见 `docs/arpl/mv_rpformer_gpu_runtime_amendment_2026-09-03.md`。

注意：phase 根目录 `environment.json` 中的 `device=cpu` 只表示聚合程序在 CPU 上执行；96 个训练任务的统一运行设备由 `confirmation_integrity.json.execution_runtime` 审计为 `cuda / NVIDIA GeForce RTX 4090`。

## 3. 完整性与本地复核

远端 phase audit 已确认：

- 24 个 development 和 96 个 confirmation 方法产物全部存在；
- 所有方法源码哈希一致并通过验证；
- 12 个 confirmation 单元均使用相同运行环境；
- 同一 split/seed 的八种方法共享真实 pair manifest、标签和预测顺序；
- M6 与 M7 使用相同伪未知调度；
- permutation audit 与 pseudo-pair audit 全部通过；
- NPZ 逐样本预测交叉检查和指标精确反算全部通过；
- `development_performance_gate_used=false`；
- `final_unknown_used=false`，`even_angle_test_used=false`。

本地又从 `metrics_by_unit.csv` 独立复算：96 行、每种方法 12 个单元；9 组比较的 7 项指标共 189 个 mean/std/positive-count 字段与 `comparison_summary.json` 完全一致。重新执行预注册门槛后得到 `main=false`、`ARPL-specific=true`、`freeze=false`，与 `confirmation_decision.json`、`comparison_summary.json` 和最终 JSON 三处记录完全一致。

完成凭证：

- final JSON SHA-256：`db384e30e7e63cb4958e0786c67817ee20933be25e97b66671eb0ced04cef832`；
- final paired-deltas SHA-256：`319ff39baa21431bc1b4ce2acb30f7fd949bc4161743a5578c63fb0d30d66555`；
- final success SHA-256：`dc53c469a04abf92858fcf77ad3bd4cacdf5d223459946fe75ab8198f650f680`。

## 4. 八种方法的正式结果

全部数值为 12 个 confirmation 单元的 `mean ± population std`，单位为百分比；FPR95 越低越好，其余指标越高越好。

| 方法 | Known Acc. | Known F1 | AUROC | OSCR | FPR95 ↓ | URR | K+1 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 95.99 ± 1.59 | 95.99 ± 1.59 | 67.22 ± 9.52 | 65.71 ± 8.89 | 80.40 ± 9.80 | 26.15 ± 8.44 | 76.51 ± 2.40 |
| M1 | 96.85 ± 1.05 | 96.85 ± 1.06 | 67.05 ± 13.08 | 66.01 ± 12.56 | 74.70 ± 12.07 | 25.04 ± 8.66 | 76.67 ± 2.66 |
| **M2** | **99.05 ± 0.41** | **99.05 ± 0.41** | **76.22 ± 15.22** | **76.00 ± 15.16** | **53.75 ± 17.87** | **34.27 ± 21.89** | **81.03 ± 5.41** |
| M3 | 98.64 ± 0.71 | 98.64 ± 0.71 | 63.85 ± 16.12 | 63.61 ± 15.99 | 74.99 ± 12.91 | 22.00 ± 13.64 | 77.60 ± 3.39 |
| M4 | 98.94 ± 0.62 | 98.94 ± 0.62 | 63.80 ± 22.19 | 63.71 ± 22.15 | 72.50 ± 17.93 | 25.27 ± 16.69 | 78.64 ± 4.21 |
| M5 | 98.06 ± 0.76 | 98.04 ± 0.77 | 63.51 ± 25.89 | 63.31 ± 25.79 | 71.15 ± 20.45 | 24.60 ± 18.33 | 78.21 ± 4.75 |
| **M6** | 97.91 ± 0.84 | 97.89 ± 0.85 | 64.25 ± 23.07 | 64.02 ± 22.98 | 75.65 ± 19.32 | 28.77 ± 17.82 | 79.58 ± 4.53 |
| M7 | 98.69 ± 0.94 | 98.68 ± 0.95 | 59.08 ± 25.01 | 58.85 ± 24.95 | 79.93 ± 15.40 | 28.36 ± 18.29 | 79.31 ± 4.74 |

方法含义：M0/M1 是浅层共享 CNN 的 CE/ARPL 基线；M2 换成多尺度编码器但仍用均值融合；M3 加入 Set Transformer；M4 再加入独立逐视角 ARPL；M5 加 mismatch-only 拒判器；M6 同时使用 mismatch 和 coherent mixup；M7 与 M6 结构相同，但 ARPL heads 换成 CE heads。

指标覆盖说明：冻结配置的 `report_metrics` 只包含表中的 7 项，没有输出项目通用规则还要求的 KCCR（已知类被正确分类且接受的比例）及 KCCR/URR 调和平均。本轮不从不完整的汇总包事后补算，也不改变预注册判定；这两项不属于 M6 gate，因此不影响当前去留结论。任何后续新预注册或最终 test 都必须在运行前把二者纳入配置、逐样本反算和完整性审计。

## 5. 预注册门槛逐项判定

### 5.1 M6 相对 M4：主方法门槛

| 条件 | 原始实际值 | 门槛 | 判定 |
|---|---:|---:|---|
| 平均 ΔAUROC | `+0.0044896667`（`+0.45 pp`） | `>= +0.02` | **失败** |
| AUROC 正向单元 | `7/12` | `>= 8/12` | **失败** |
| 平均 ΔOSCR | `+0.0031327833`（`+0.31 pp`） | `>= 0` | 通过 |
| 平均 ΔKnown Accuracy | `-0.0102666667`（`-1.03 pp`） | `>= -0.01` | **失败** |
| 平均 ΔFPR95 | `+0.0315`（恶化 `3.15 pp`） | `<= +0.02` | **失败** |

五项必须同时满足，因此 `m6_main_method_success=false`。Known Accuracy 只比边界多下降约 `0.027 pp`，但 AUROC 增益、正向单元数和 FPR95 也同时未达标，结论不依赖单一临界舍入。

### 5.2 M6 相对 M7：ARPL-specific 门槛

| 条件 | 原始实际值 | 门槛 | 判定 |
|---|---:|---:|---|
| 平均 ΔAUROC | `+0.0517462833`（`+5.17 pp`） | `>= +0.01` | 通过 |
| AUROC 正向单元 | `9/12` | `>= 7/12` | 通过 |

因此 `arpl_specific_success=true`：在相同双路径结构和伪未知调度下，ARPL heads 明显优于 CE heads。但冻结 M6 需要主门槛和 ARPL-specific 同时通过，所以最终仍为 `freeze_m6_for_final_test=false`。

`final_unknown_test_authorized=false` 是本轮停止边界，不应误解为又一次独立性能失败。

## 6. 全部受控比较

下表均为左方法减右方法的 12 单元平均差，单位为百分点；ΔFPR95 为负代表改善。`AUROC 正向`只统计严格 `ΔAUROC>0` 的单元。

| 比较（左−右） | ΔKnown Acc. | ΔKnown F1 | ΔAUROC ± SD | AUROC 正向 | ΔOSCR | ΔFPR95 | ΔURR | ΔK+1 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2−M1 Backbone | +2.19 | +2.19 | **+9.18 ± 5.43** | **12/12** | +9.99 | -20.95 | +9.22 | +4.37 |
| M3−M2 Transformer | -0.41 | -0.41 | **-12.38 ± 10.86** | 1/12 | -12.40 | +21.24 | -12.27 | -3.43 |
| M4−M3 Hier. ARPL | +0.30 | +0.30 | -0.04 ± 8.47 | 5/12 | +0.10 | -2.49 | +3.27 | +1.04 |
| M5−M4 Mismatch | -0.88 | -0.90 | -0.30 ± 9.62 | 7/12 | -0.40 | -1.35 | -0.67 | -0.42 |
| M6−M5 Mixup | -0.15 | -0.15 | +0.75 ± 5.72 | 6/12 | +0.71 | +4.50 | +4.17 | +1.37 |
| M6−M4 Full rejector | -1.03 | -1.04 | +0.45 ± 9.77 | 7/12 | +0.31 | +3.15 | +3.50 | +0.94 |
| M6−M7 ARPL-specific | -0.77 | -0.79 | **+5.17 ± 4.89** | **9/12** | +5.17 | -4.28 | +0.41 | +0.27 |
| M6−M1 ARPL base | +1.06 | +1.04 | -2.80 ± 11.67 | 4/12 | -1.99 | +0.95 | +3.72 | +2.92 |
| M6−M0 CE base | +1.92 | +1.90 | -2.97 ± 18.00 | 7/12 | -1.69 | -4.75 | +2.62 | +3.07 |

## 7. M4/M6/M7 的 12 单元稳定性

| 单元 | M4 AUROC | M6 AUROC | M6−M4 | M7 AUROC | M6−M7 | M6 Known Acc. | M6 FPR95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0/20260830 | 72.40% | 80.06% | +7.66 pp | 74.36% | +5.71 pp | 98.64% | 68.28% |
| C0/20260831 | 78.58% | 85.12% | +6.54 pp | 83.91% | +1.20 pp | 97.64% | 60.64% |
| C0/20260832 | 67.38% | 86.98% | +19.60 pp | 78.48% | +8.50 pp | 98.76% | 51.28% |
| C1/20260830 | 22.67% | 24.29% | +1.62 pp | 16.07% | +8.22 pp | 97.28% | 100.00% |
| C1/20260831 | 22.29% | 28.26% | +5.97 pp | 13.96% | +14.30 pp | 99.16% | 88.72% |
| C1/20260832 | 39.63% | 33.77% | -5.86 pp | 33.88% | -0.11 pp | 96.32% | 96.00% |
| C2/20260830 | 79.67% | 66.03% | -13.65 pp | 60.24% | +5.78 pp | 98.00% | 68.80% |
| C2/20260831 | 58.92% | 55.17% | -3.75 pp | 55.90% | -0.73 pp | 97.28% | 96.88% |
| C2/20260832 | 74.52% | 58.21% | -16.31 pp | 46.72% | +11.49 pp | 97.12% | 91.64% |
| C3/20260830 | 77.10% | 82.35% | +5.26 pp | 85.15% | -2.79 pp | 99.20% | 86.68% |
| C3/20260831 | 90.14% | 82.44% | -7.70 pp | 76.55% | +5.89 pp | 97.72% | 58.76% |
| C3/20260832 | 82.34% | 88.36% | +6.02 pp | 83.72% | +4.64 pp | 97.84% | 40.08% |

M6−M4 的 AUROC 增益按 identity split 平均为 C0 `+11.27 pp`（3/3 为正）、C1 `+0.58 pp`（2/3）、C2 `-11.24 pp`（0/3）、C3 `+1.19 pp`（2/3）；按随机种子平均则只有 `+0.22/+0.26/+0.86 pp`。M6 的绝对 AUROC 按 split 为 C0 `84.05%`、C1 `28.77%`、C2 `59.80%`、C3 `84.38%`。因此当前差异主要随 surrogate identity 组合改变，而不是由某一个随机种子主导。

C1 的失败具有系统性：M3/M4/M5/M6/M7 的三 seed 平均 AUROC 分别为 `39.03%/28.20%/21.37%/28.77%/21.30%`，M2–M7 在 C1 三个种子的 URR 均为 0。M6 的跨单元 AUROC 标准差为 `23.07 pp`，远大于相对 M4 的平均增益 `0.45 pp`。这些结果表明本轮表现随 surrogate identity 组合大幅变化，但不能单凭现有结果判定具体是类别相似、表示、融合、阈值还是伪未知构造所致。

## 8. 结果解释

### 8.1 已经确认

1. **多尺度 backbone 有稳定价值。** M2−M1 的 AUROC 在 12/12 单元为正，同时 known Accuracy、OSCR、FPR95 和 K+1 F1 都改善；这是本轮最清晰的正向组件证据。
2. **当前 Set Transformer 融合没有带来收益。** M3−M2 平均 AUROC 下降 `12.38 pp`，仅 1/12 单元为正；这否定的是当前冻结实现和协议下的增益，不是所有注意力或集合模型。
3. **逐视角 ARPL 单独近似中性。** M4−M3 平均 AUROC 为 `-0.04 pp`、5/12 为正，不能声称稳定有效。
4. **现有伪未知拒判器没有稳定超过 M4。** M5−M4 近似中性偏负；M6−M5 的 mixup 增量只有 `+0.75 pp` 且 6/12 为正；完整 M6−M4 只有 `+0.45 pp`。
5. **ARPL 不是本轮失败的简单原因。** 在结构和伪未知完全相同的 M6/M7 对照中，ARPL-specific 门槛通过。
6. **闭集准确率与开集排序明显脱钩。** M4–M7 的 known Accuracy 约为 `98%–99%`，但 AUROC 只有 `59%–64%` 且波动很大；继续只优化闭集 Accuracy 不能解决问题。

### 8.2 尚未验证的原因解释

以下是由结果支持、但尚不能写成因果结论的解释：

- M3 之后增加的模型复杂度可能没有在现有独立底层样本量和 surrogate identity 变化下学到可迁移的未知边界；
- mismatch/mixup 伪未知可能更容易被模型识别为生成机制本身，而不等价于完整、连贯的新目标身份；
- C1/C2 与 C0/C3 的巨大差异提示未知难度主要受具体身份组合和相似类别影响，而不是单一全局阈值或平均分数能够概括；
- M2 的优势可能来自多尺度 HRRP 表征本身，而非更复杂的多视角交互。

要区分这些解释，应先只分析现有产物中的类别级分数分布、C1/C2 误差、伪未知与真实 surrogate unknown 的分数关系、注意力/token 和拒判器输出；不应立刻开启新参数搜索。

## 9. 当前决策与下一步边界

1. 停止把 M6 当作待微调即可进入最终测试的候选；不运行最终未知类或偶数角 test。
2. 等待 Merlin CPU confirmation 作为独立复核。GPU 结果已在运行前指定为正式结果，不能根据两台机器谁更好而择优，也不能拼接任务。
3. M2 是本轮最值得保留的简化候选，但当前预注册只授权按 M6 gate 冻结最终方法，不能事后把 M2 自动提升为最终模型。
4. 下一次实验前应先决定：是围绕 M2 建立一个新的、简化的固定表示路线，还是先对现有 M3–M6 产物做机制诊断后停止该复杂分支。任何新路线都需要独立预注册和停止规则。

## 10. 原始证据索引

原始运行产物不提交 Git，保存在：

`artifacts/arpl/mv_rpformer_surrogate_v1/gpu_confirmation_1778d03/`

关键文件：

- `confirmation_gpu_1778d03/metrics_by_unit.csv`：96 行绝对指标；
- `confirmation_gpu_1778d03/paired_deltas.csv`：全部逐单元配对差值；
- `confirmation_gpu_1778d03/comparison_summary.json`：九组比较及聚合；
- `confirmation_gpu_1778d03/confirmation_integrity.json`：正式审计；
- `confirmation_gpu_1778d03/confirmation_decision.json`：预注册判定；
- `final_result_gpu_1778d03.json` 与 `.success.json`：最终结果和完成凭证；
- `mv_rpformer_gpu_summary_1778d03.tar.gz`：从容器下载的只读汇总包，SHA-256 为 `0ded98fdf5060585bb7665e5f9fc3fea7fbec0243f406f67a8d528213fa67f3f`。

`RESEARCH_CONTEXT.md` 本轮未修改；其中阶段状态已落后，是否回写需用户另行确认。
