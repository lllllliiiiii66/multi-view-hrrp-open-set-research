# 多尺度 HRRP 骨干 × CE/ARPL 全因子简化确认结果

> 日期：2026-09-03
>
> 阶段：P3 独立简化确认
>
> 实验：`ms_mean_head_factorial_surrogate_v1`
>
> 状态：GPU smoke 与 168 项正式 confirmation 已完成，远端另行执行的内置 phase audit 和本地汇总级独立复核均通过
>
> 证据范围：7 个 source-known 类奇数角开发池上的 surrogate identity OSR；不是最终 7-known/3-unknown 或偶数角 test 结果

## 1. 直接结论

正式 confirmation 的 `42 个实验单元 × 4 个方法 = 168` 项训练全部完成，`168/168` 成功、`0` 失败。预注册的样本共享、标签/预测顺序、成对初始化、九项指标反算、源码哈希、GPU 分配和最终测试隔离审计全部通过。

按运行前冻结的门槛：

- A（R3−R1，多尺度骨干在 ARPL 下的贡献）通过；
- B（R2−R0，多尺度骨干在 CE 下的贡献）通过；
- backbone 标签为 `backbone_general_success`；
- R3−R2 不通过 ARPL 优先门槛；反向 R2−R3 虽然平均 AUROC 提高 `7.46 pp`，但只有 `4/7` identity pair 为正，未达到 `5/7`，因此 head 标签严格保持 `HEAD_INDETERMINATE`；
- 机器规则推荐 `R2_MS_MEAN_CE`，即“冻结多尺度编码器 + 算术均值融合 + 线性 CE head”；
- `separate_final_test_preregistration_recommended=true`，但 `final_unknown_test_authorized=false`。

最重要的科学结论是：**多尺度 HRRP 表示相对浅层表示有跨 head、跨 fold、跨 seed 的明确增益；当前证据没有证明 ARPL head 在同一多尺度骨干上优于更简单的 CE head。** R2 的 42 单元平均 Known Accuracy 为 `98.92%`、AUROC 为 `86.04%`、OSCR 为 `85.78%`、FPR95 为 `34.39%`。

但 R2 仍不是已经验证的最终开集方法：其平均 URR 只有 `39.33%`，并且 DDG-1000、DDG-112 两个 surrogate identity 的 URR 都为 `0%`。这表明当前固定 95% known-acceptance 阈值存在明显的身份依赖失败，不能把高平均 AUROC 外推为最终未知类性能。

## 2. 冻结设计与实际运行

| 项目 | 实际值 |
|---|---|
| 任务来源基线 | `ccb30e18b4aa9e78e136ba330dddefb11aab11ae` |
| 正式运行代码 | `62e318de82b4221b599e06b1166483673e9c1cd3`，分支 `codex/ms-mean-head-factorial`，远端运行时 clean |
| 配置 | `configs/experiments/arpl/ms_mean_head_factorial_surrogate_v1.yaml` |
| 配置 SHA-256 | `c11daa6e2e5a7d7b72bc36840e60fc871f332c4fc85652636c729aa2eba14c71` |
| 数据 profiles SHA-256 | `2dd92282c125f0f677cf1f2dfce828781c8ba4385cf9ae552c4a2c56033c3f5b` |
| 数据 manifest SHA-256 | `748b9f30629c3b3cbe66c6a1dac30863fdab2d81a214e46d8bc3ef7c6022a08a` |
| 数据 bundle SHA-256 | `79176091b5b745e5df5957f1bb4ade7304c1403a4a32e64622e012db67d5a0c5` |
| Surrogate 设计 | N0–N6 共 7 个新 identity pair；每类恰出现两次 |
| 单元设计 | `7 pairs × 2 folds × 3 seeds = 42` |
| 方法设计 | R0–R3 四方法，共 168 个正式训练任务 |
| 每单元数据 | 每个 train-known 类 500 对、每个 known calibration 类 500 对、每个 surrogate 类 500 对 |
| 正式环境 | 4 × NVIDIA GeForce RTX 4090；CUDA 12.8；PyTorch `2.7.0a0+7c8ec84dab.nv25.03`；NumPy `1.26.4` |
| 调度 | 每卡 4 个 worker，峰值 16 并发；每卡 42 项；每方法在各卡 10/11 项 |
| 训练 | AdamW，`3e-4`，100 epochs，epoch 100 固定为正式模型；无 early stopping、无性能选点 |
| 开集分数 | `-max(raw logits)`，越大越未知；阈值仅由本模型 known calibration 按 95% known acceptance 得到 |
| 排除项 | 无伪未知、无拒判器、无 attention；未生成最终 3 unknown 或偶数角 test |

四个方法严格为：

| 方法 | 骨干 | 融合 | Head |
|---|---|---|---|
| R0_SHALLOW_MEAN_CE | `SharedHRRPEncoder1D` | 算术均值 | 线性 CE |
| R1_SHALLOW_MEAN_ARPL | `SharedHRRPEncoder1D` | 算术均值 | global ARPL |
| R2_MS_MEAN_CE | `HRRPMultiScaleResNet1D` | 算术均值 | 线性 CE |
| R3_MS_MEAN_ARPL | `HRRPMultiScaleResNet1D` | 算术均值 | global ARPL |

正式 GPU smoke 只运行 N0/fold0/seed20260830 的四种方法、每类 10 对、1 epoch。它只用于验证链路与审计，性能没有进入 gate，也没有据此修改协议。

## 3. 完整性审计

远端正式产物完成后，另行执行 `audit --phase confirmation`，结果为 `status=passed`。已确认：

- 42 个实验单元、168 个训练任务全部存在并通过单项审计；
- launcher 为 `168` 成功、`0` 失败，4 张 GPU 各承担 42 项；
- 同一单元四方法共享 pair manifest、预测顺序和 DataLoader 顺序；
- R0/R1、R2/R3 的对应骨干初始化哈希一致；
- 全部九项指标均从逐样本预测精确反算；
- 全部方法源码哈希、正式 CUDA 运行约束一致；
- 无 pseudo unknown 路径；最终 unknown、偶数角 test 均被排除；
- `bootstrap_used_for_gate=false`、`length_padding_used_for_gate=false`、`final_unknown_test_authorized=false`。

phase 根目录的 `environment.json` 写有 `device=cpu`，仅表示最终聚合程序在 CPU 上执行；`integrity.json.formal_runtime_contract` 已确认 168 个训练任务统一运行在 RTX 4090/CUDA 上。

下载到本地的是 summary-only 审阅包。其 22 个成员与远端汇总文件逐字节一致，归档 SHA-256 为：

`39ebd7f008b60652ff1b981579bc3c85b564bb46f9421836be6525c0840ba2a2`

本地已独立复算 168 行总体均值、35 个 pair-level delta、210 个 unit-level delta、五组比较、bootstrap 和全部 gate，均与保存结果一致。本地包不含 168 个 checkpoint 与逐样本目录，因此逐样本九指标的再次反算以远端另行执行并通过的内置 phase audit 和全树哈希清单为证据。

## 4. R0–R3 完整指标

下表为 42 个正式单元的算术均值，单位为百分比。FPR95 越低越好，其余越高越好。KCCR 是“已知样本既分类正确又未被拒绝”的比例；H 是 KCCR 与 URR 的调和平均。

| 方法 | Known Acc. | Known F1 | AUROC | OSCR | FPR95 ↓ | KCCR | URR | H | K+1 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 shallow+CE | 96.61 | 96.62 | 67.37 | 65.79 | 74.32 | 92.77 | 31.01 | 43.55 | 78.12 |
| R1 shallow+ARPL | 97.13 | 97.13 | 67.58 | 66.40 | 70.35 | 93.21 | 28.74 | 40.67 | 77.66 |
| **R2 MS+CE** | **98.92** | **98.92** | **86.04** | **85.78** | **34.39** | **94.73** | 39.33 | 51.97 | 81.93 |
| R3 MS+ARPL | 98.75 | 98.75 | 78.57 | 78.25 | 47.01 | 94.62 | **40.33** | **52.72** | **82.06** |

R3 的平均 URR、H 和 K+1 F1 略高于 R2，但 R2 的 AUROC、OSCR、FPR95 和 Known Accuracy 更好。方法选择不按某个总体均值临时决定，而严格使用第 6 节的 pair-level 预注册规则。

## 5. 全因子受控比较

下表均为左方法减右方法的 7 个 pair-level 平均差，单位为百分点。每个 pair 先对 2 folds × 3 seeds 求均值；AUROC bootstrap 以 7 个 identity pair 为单位、固定 seed 20260903、10,000 次，仅作描述，不进入 gate。

| 比较 | ΔKnown Acc. | ΔKnown F1 | ΔAUROC | AUROC 正 pair | AUROC 95% CI | ΔOSCR | ΔFPR95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A：R3−R1，ARPL 下骨干 | +1.62 | +1.62 | **+10.99** | **6/7** | `[+5.63,+17.52]` | +11.85 | -23.34 |
| B：R2−R0，CE 下骨干 | +2.31 | +2.30 | **+18.66** | **6/7** | `[+10.86,+24.81]` | +19.99 | -39.93 |
| C：R3−R2，多尺度上 ARPL | -0.17 | -0.17 | **-7.46** | 3/7 | `[-13.98,-1.41]` | -7.53 | +12.62 |
| D：R1−R0，浅层上 ARPL | +0.52 | +0.51 | +0.21 | 3/7 | `[-4.80,+5.43]` | +0.61 | -3.96 |
| Interaction：C−D | -0.69 | -0.68 | -7.67 | 2/7 | `[-13.42,-1.92]` | -8.14 | +16.59 |

| 比较 | ΔKCCR | ΔURR | ΔH | ΔK+1 F1 |
|---|---:|---:|---:|---:|
| A | +1.40 | +11.59 | +12.05 | +4.40 |
| B | +1.96 | +8.32 | +8.42 | +3.81 |
| C | -0.11 | +1.00 | +0.75 | +0.12 |
| D | +0.44 | -2.28 | -2.88 | -0.46 |
| Interaction | -0.55 | +3.27 | +3.63 | +0.59 |

## 6. 预注册 gate 与机器决定

### 6.1 Backbone

Backbone gate 同时要求：平均 ΔAUROC `>=3 pp`、AUROC 正 pair `>=6/7`、ΔOSCR `>=0`、ΔKnown Accuracy `>=-0.5 pp`、ΔFPR95 `<=+2 pp`。

| Gate | ΔAUROC | 正 pair | ΔOSCR | ΔKnown Acc. | ΔFPR95 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| A：R3−R1 | +10.99 ✓ | 6/7 ✓ | +11.85 ✓ | +1.62 ✓ | -23.34 ✓ | **通过** |
| B：R2−R0 | +18.66 ✓ | 6/7 ✓ | +19.99 ✓ | +2.31 ✓ | -39.93 ✓ | **通过** |

因此 backbone 唯一标签为 `backbone_general_success`。多尺度收益不是只在 ARPL 或 CE 的某一个 head 下出现。

### 6.2 Head

Head gate 同时要求：平均 ΔAUROC `>=1 pp`、AUROC 正 pair `>=5/7`，并满足同样三个保护条件。

| 方向 | ΔAUROC | 正 pair | ΔOSCR | ΔKnown Acc. | ΔFPR95 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| R3−R2，ARPL 优先 | -7.46 ✗ | 3/7 ✗ | -7.53 ✗ | -0.17 ✓ | +12.62 ✗ | 不通过 |
| R2−R3，CE 优先 | +7.46 ✓ | **4/7 ✗** | +7.53 ✓ | +0.17 ✓ | -12.62 ✓ | 不通过 |

反向 CE gate 只有“正 pair 数”一项未达标，但预注册要求五项全部满足，因此不能事后改写为 `CE_PREFERRED`，正式标签是 `HEAD_INDETERMINATE`。在 `backbone_general_success + HEAD_INDETERMINATE` 下，冻结规则选择结构更简单的 `R2_MS_MEAN_CE`。

## 7. Identity-pair 稳定性

下表为 7 个 pair 的 AUROC delta，单位为百分点。

| Pair | A：R3−R1 | B：R2−R0 | C：R3−R2 | D：R1−R0 | Interaction |
|---|---:|---:|---:|---:|---:|
| N0 | +9.75 | +29.26 | -6.72 | +12.79 | -19.50 |
| N1 | +11.63 | +22.41 | -22.65 | -11.86 | -10.78 |
| N2 | -0.70 | -1.38 | +0.55 | -0.12 | +0.67 |
| N3 | +6.95 | +22.38 | -16.12 | -0.68 | -15.44 |
| N4 | +11.80 | +21.67 | -9.66 | +0.21 | -9.87 |
| N5 | +10.07 | +11.57 | +1.39 | +2.90 | -1.51 |
| N6 | +27.45 | +24.73 | +0.95 | -1.77 | +2.71 |

A/B 都只在 N2 小幅为负，另外 6 个 pair 为正；因此 backbone 成功不是由单一 pair 驱动。按 fold 汇总，A 为 `+10.68/+11.30 pp`，B 为 `+18.32/+19.00 pp`；按三个 seed 汇总，A 为 `+14.71/+7.47/+10.79 pp`，B 为 `+20.98/+16.39/+18.62 pp`，方向全部一致。42 个 unit 中，A 有 `38/42` 为正，B 有 `39/42` 为正。

R2 的绝对 AUROC 在 fold 0/4 为 `84.87/87.20%`，在三个 seed 上为 `87.15/84.64/86.32%`；整体没有由单一 fold 或 seed 支撑，但 pair 内仍存在下节所示的明显差异。

Head 差异明显依赖 identity：R2 在 N0/N1/N3/N4 上胜出，R3 只在 N2/N5/N6 上小幅胜出。C 按两个 fold 为 `-7.79/-7.14 pp`，按三个 seed 为 `-6.58/-7.90/-7.92 pp`，总体方向稳定，但只有 4/7 pair 支持 CE，因此按冻结规则仍是 head indeterminate。

## 8. 最困难 pair 与身份级失败

按四个方法的平均 AUROC，从难到易依次为：N1、N4、N3、N0、N6、N5、N2。下表同时给出候选 R2 的绝对表现。

| Pair | Surrogate identities | 四方法平均 AUROC | 四方法平均 URR | R2 AUROC | R2 URR |
|---|---|---:|---:|---:|---:|
| N1 | DDG-112 / 迷你好望角型散货船 | **61.49** | 12.46 | 81.32 | **1.27** |
| N4 | DDG-1000 / 集装箱船达飞罗尔多夫级 | **64.22** | 31.81 | **77.42** | 30.60 |
| N3 | DDG-1000 / 油气轮 MARVEL CRANE | **66.19** | 31.60 | 81.59 | 38.15 |
| N0 | CVN77 / DDG-112 | 72.64 | 24.31 | 85.75 | 31.40 |
| N6 | CVN77 / 爱达魔都号 | 78.84 | 33.68 | 91.41 | 54.02 |
| N5 | 爱达魔都号 / 集装箱船达飞罗尔多夫级 | 87.87 | 47.93 | 92.59 | 61.47 |
| N2 | 油气轮 MARVEL CRANE / 迷你好望角型散货船 | 92.98 | 62.18 | 92.19 | 58.42 |

R2 的身份级结果进一步显示：

| Surrogate identity | mean AUROC | AUROC 范围 | URR |
|---|---:|---:|---:|
| CVN77 | 90.90 | 88.31–94.08 | 56.83 |
| DDG-1000 | **65.04** | 49.33–74.76 | **0.00** |
| DDG-112 | **78.45** | 69.24–84.98 | **0.00** |
| 油气轮 MARVEL CRANE | 94.56 | 91.79–97.41 | 69.67 |
| 爱达魔都号 | 92.39 | 87.81–94.92 | 58.92 |
| 迷你好望角型散货船 | 88.36 | 80.59–97.08 | 28.17 |
| 集装箱船达飞罗尔多夫级 | 92.56 | 88.38–96.57 | 61.73 |

错误吸收具有清晰结构：R2 未拒绝的 DDG-1000 样本 `6000/6000` 全部被判为 DDG-112；未拒绝的 DDG-112 样本 `6000/6000` 全部被判为 DDG-1000。迷你好望角型散货船的 4310 个未拒绝样本中，4153 个（`96.36%`）被吸收到 DDG-1000。R2 的全部 25,481 个 false accepts 中，流向 DDG-1000 或 DDG-112 的合计比例为 `64.00%`。

这已经确认“当前表示 + 最大 logit 分数 + 95% known acceptance 阈值”对部分身份存在系统性失败，但尚不能区分原因究竟是原始数据/标签相似、表示学习、分数定义还是阈值校准。N1 中 R2 的 AUROC 为 `81.32%`、URR 却只有 `1.27%`，也说明排序能力与固定阈值下的实际拒判率不能混为一谈。

## 9. Length/padding 诊断

按预注册的“两个 fold 中所有 surrogate 原始长度均落在 train-known 闭区间内”定义，N0–N6 全部为 `length-safe`，`length-risk` 为空。因此：

- length-safe 子集与全部 7 pair 的方向完全相同；
- 没有 length-risk pair，无法比较 safe/risk 两个子集是否一致；
- 所有 fold 的 `surrogate_outside_support_count=0`，没有“长度超出训练支持范围”的直接捷径证据；
- 但 N1、N2 的 length-only AUROC 为 `0.75`，其余为 `0.50`。这表示长度在支持区间内部仍可能携带身份信息，不能据此宣称 padding/长度完全无影响。

Length 结果只作解释，没有删除 pair、改变输入或参与 gate。

## 10. 已确认、尚未验证与当前决定

### 已经确认

1. 在这 7 个新 surrogate identity pair 上，多尺度 HRRP 骨干相对浅层骨干在 CE 和 ARPL 两条受控对照都达到预注册成功门槛。
2. 增益跨两个 angle fold 和三个初始化 seed 保持同方向，且在 6/7 identity pair 为正。
3. ARPL 在浅层骨干上基本中性；在多尺度骨干上也没有达到预注册的 ARPL 优先标准。
4. R2 是本轮规则下唯一推荐候选；选择它来自预先冻结的规则，不是事后挑最高数字。
5. DDG-1000 与 DDG-112 存在系统性相互吸收，当前阈值对这两个 surrogate identity 的 URR 为 0。

### 尚未验证

1. R2 对最终 3 个真实 unknown 类和偶数角 test 的性能；本轮未生成或使用对应 final-test pair、特征或预测，它们未进入训练、校准或评价。
2. DDG-1000/DDG-112 失败的具体因果原因。
3. 换用其他开集分数、阈值或拒判机制是否能解决身份依赖失败；本轮没有授权这些改动。
4. 本轮描述性 bootstrap 区间不是统计显著性声明，也没有进入机器 gate。

### 当前决定与下一步

按预注册规则，建议为 `R2_MS_MEAN_CE` **另行设计并冻结一次最终 7-known/3-unknown 测试协议**。若用户另行批准，唯一候选应保持本轮 R2 的多尺度骨干、算术均值融合、CE head、训练参数、`-max(raw logit)` 分数和 known-only 95% acceptance 阈值，不得根据最终 unknown 调参。

本轮没有授权该最终测试，故保持：

```text
backbone_label = backbone_general_success
head_label = HEAD_INDETERMINATE
recommended_candidate = R2_MS_MEAN_CE
separate_final_test_preregistration_recommended = true
final_unknown_test_authorized = false
```

## 11. 证据索引与哈希

本机 summary-only 产物位于：

`artifacts/results/ms_mean_head_factorial_surrogate_v1/confirmation_gpu_62e318d/`

关键文件：

- `metrics_by_unit.csv`：168 行四方法绝对指标，SHA-256 `132f98c1c489e8337747dac9cbe976d01bfc3ef1aa3cd9696589a8c33b735d9e`；
- `paired_deltas.csv`：35 行 pair-level 差值，SHA-256 `1b82dbd3d1abc1de71ee3c4ea9ad2b48b6e69a20cbe374e823d989749e0b74f0`；
- `unit_deltas.csv`：210 行 unit-level 差值，SHA-256 `55193b2f8301b9a254b29ec72ad4845a234b4d48ac7041e9fb046139ddca322d`；
- `comparison_summary.json`：全部聚合与 bootstrap，SHA-256 `fa16ef71e480ca9cbfb6c80cd9879e6729e6c987ba2fc0dcd04c126e96ec1ba2`；
- `factorial_decision.json`：冻结 gate 判定，SHA-256 `0ab8bbdb52192ce404293c2321a1410e1748eaa726ceabeaba8e30b705df83f7`；
- `integrity.json`：正式完整性审计，SHA-256 `5cefc52ee23e9c88a10b99762e49f9f716f34384dd574d5f0f9ed0706212b4c0`；
- `_PHASE_SUCCESS.json`：phase 成功封口，SHA-256 `066fac0b8fa729c6a653bb069a8570e947f03ff33dd288e1a227e22705d57489`；
- 汇总归档 `ms_mean_head_factorial_confirmation_gpu_62e318d_summary.tar.gz`：SHA-256 `39ebd7f008b60652ff1b981579bc3c85b564bb46f9421836be6525c0840ba2a2`。

完整的 168 项 checkpoint、逐样本预测和训练日志继续保存在 GPU 容器的独立正式目录中，没有提交 Git，也没有覆盖旧实验。

`RESEARCH_CONTEXT.md` 本轮未修改。若后续需要回写，建议只提案加入：P3 全因子确认已完成；backbone 为 general success、head 为 indeterminate、R2 为待独立预注册最终测试的唯一候选；最终 unknown/even-angle test 仍未授权。
