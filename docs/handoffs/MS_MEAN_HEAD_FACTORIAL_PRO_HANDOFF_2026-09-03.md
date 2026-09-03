# 多尺度 Mean × CE/ARPL 全因子确认：ChatGPT Pro 分析交接

> 日期：2026-09-03
>
> 用途：供 ChatGPT Pro 在不改代码、不启动新实验的前提下，判断是否应为唯一候选 R2 另行预注册最终测试
>
> 当前机器结论：`backbone_general_success + HEAD_INDETERMINATE`，推荐 `R2_MS_MEAN_CE`，最终测试尚未授权

## 一、请先完整阅读

按以下顺序阅读仓库文件：

1. `AGENTS.md`
2. `RESEARCH_CONTEXT.md`
3. `docs/arpl/mv_rpformer_preregistration_2026-09-03.md`
4. `docs/arpl/mv_rpformer_results_2026-09-03.md`
5. `docs/handoffs/MV_RPFORMER_PRO_HANDOFF_2026-09-03.md`
6. `docs/arpl/ms_mean_head_factorial_preregistration_2026-09-03.md`
7. `docs/arpl/ms_mean_head_factorial_results_2026-09-03.md`
8. `configs/experiments/arpl/ms_mean_head_factorial_surrogate_v1.yaml`

`RESEARCH_CONTEXT.md` 是受保护主上下文，本轮没有修改，其中阶段状态可能落后；本次 P3 的最新证据以第 6、7 项和正式汇总产物为准。

## 二、希望 Pro 回答的问题

请在严格区分“已经确认、尚未验证、建议执行”的前提下回答：

1. 这次结果是否足以停止浅层 backbone，并把 R2 作为唯一表示候选？
2. 应如何解释“R2−R3 平均 AUROC 为 `+7.46 pp`、两个 fold 和三个 seed 都同方向，但只有 `4/7` identity pair 为正”，以及为什么正式 head 标签仍必须是 `HEAD_INDETERMINATE`？
3. DDG-1000/DDG-112 的 URR 均为 0、相互吸收 100%，是否足以阻止进入最终测试，还是应作为最终测试前已知风险保留？
4. 是否建议为 R2 另行预注册一次最终 7-known/3-unknown、偶数角 test？如果建议，请只给出唯一冻结协议、一次性成功/停止规则和可报告的结论边界；不要提出参数搜索。
5. 如果不建议进入最终测试，请说明还缺少哪一个不使用最终 unknown、且不训练新模型的最小诊断；不要重新打开 M3–M6、伪未知、attention 或新 head 路线。

不要把 surrogate identity 结果外推为最终 3 个 unknown 类的性能，不要把描述性 bootstrap 写成显著性检验，也不要建议查看最终 unknown 后再调阈值。

## 三、为什么有这次独立实验

上一轮 MV-RPFormer confirmation 中，复杂的 Set Transformer、分层 ARPL 和伪未知拒判器没有形成稳定增益，但更简单的 M2“多尺度编码器 + mean + global ARPL”表现较好。由于上一轮缺少“多尺度编码器 + mean + CE”关键对照，不能判断收益来自多尺度骨干还是 ARPL。

本轮因此只做严格 2×2 因子设计：

| 方法 | 骨干 | 融合 | Head |
|---|---|---|---|
| R0 | shallow | mean | CE |
| R1 | shallow | mean | global ARPL |
| R2 | frozen multiscale | mean | CE |
| R3 | frozen multiscale | mean | global ARPL |

四方法统一使用 `-max(raw logits)` 作为 unknown score，阈值只由 known calibration 按 95% known acceptance 确定。没有拒判器、伪未知、attention、PMA、view-level head 或超参数搜索。

## 四、冻结证据边界

- 数据只来自 7 个 source-known 类的奇数角开发池。
- 新 N0–N6 七个 surrogate identity pair，每类恰出现两次；不复用上一轮已查看的 pair。
- 每个 pair 与 angle fold 0/4、seed 20260830/31/32 完全交叉。
- `7 × 2 × 3 = 42` 个实验单元，每单元四方法，共 168 项。
- 每个 train-known、known calibration、surrogate 类各 500 pairs；surrogate 不参与训练、归一化、阈值或 checkpoint。
- 正式模型固定 epoch 100；没有 early stopping 或按性能选点。
- 本轮未生成或使用最终 3 个 unknown 类及偶数角 test 对应的 final-test pair、特征或预测，它们未进入训练、校准或评价。

正式代码 commit 为 `62e318de82b4221b599e06b1166483673e9c1cd3`，配置 SHA-256 为 `c11daa6e2e5a7d7b72bc36840e60fc871f332c4fc85652636c729aa2eba14c71`。运行环境为 4 × RTX 4090、峰值 16 并发。

## 五、完整性结论

- 168/168 任务成功、0 失败；4 张卡各 42 项。
- 同一单元四方法共享 pair manifest、样本/预测/DataLoader 顺序。
- R0/R1 与 R2/R3 分别共享对应骨干初始状态。
- 九项指标已从逐样本预测精确反算。
- 伪未知路径不存在，final unknown 与 even-angle test 均排除。
- 远端另行执行的内置 `audit --phase confirmation` 返回 `status=passed`。
- 本地又独立复算总体均值、五组比较、35 个 pair delta、210 个 unit delta、bootstrap 和 gate，全部一致。

本地为 summary-only 镜像；完整 checkpoint、逐样本预测和训练日志留在 GPU 容器，并由全树 artifact hash manifest 封存。

## 六、四方法正式结果

42 个单元均值，单位为百分比；FPR95 越低越好，其余越高越好。

| 方法 | Known Acc. | Known F1 | AUROC | OSCR | FPR95 ↓ | KCCR | URR | H | K+1 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | 96.61 | 96.62 | 67.37 | 65.79 | 74.32 | 92.77 | 31.01 | 43.55 | 78.12 |
| R1 | 97.13 | 97.13 | 67.58 | 66.40 | 70.35 | 93.21 | 28.74 | 40.67 | 77.66 |
| **R2** | **98.92** | **98.92** | **86.04** | **85.78** | **34.39** | **94.73** | 39.33 | 51.97 | 81.93 |
| R3 | 98.75 | 98.75 | 78.57 | 78.25 | 47.01 | 94.62 | **40.33** | **52.72** | **82.06** |

R3 在 URR/H/K+1 F1 上略高，R2 在 AUROC/OSCR/FPR95/Known Accuracy 上明显更好。方法选择严格按 pair-level 预注册 gate，不按某一总体均值临时选择。

## 七、四个主比较与机器 gate

单位为百分点；AUROC CI 是 7 个 identity pair 的 10,000 次描述性 paired bootstrap，不参与 gate。

| 比较 | ΔAUROC | 正 pair | AUROC 95% CI | ΔOSCR | ΔKnown Acc. | ΔFPR95 | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| A：R3−R1 | +10.99 | 6/7 | `[+5.63,+17.52]` | +11.85 | +1.62 | -23.34 | 通过 |
| B：R2−R0 | +18.66 | 6/7 | `[+10.86,+24.81]` | +19.99 | +2.31 | -39.93 | 通过 |
| C：R3−R2 | -7.46 | 3/7 | `[-13.98,-1.41]` | -7.53 | -0.17 | +12.62 | 不通过 |
| D：R1−R0 | +0.21 | 3/7 | `[-4.80,+5.43]` | +0.61 | +0.52 | -3.96 | 无单独 gate |
| Interaction：C−D | -7.67 | 2/7 | `[-13.42,-1.92]` | -8.14 | -0.69 | +16.59 | 仅解释 |

A、B 五个条件全部通过，得到 `backbone_general_success`。

ARPL 方向 R3−R2 明确不通过。反向 R2−R3 的平均 AUROC、OSCR、Known Accuracy、FPR95 四项均通过，但 AUROC 正 pair 只有 `4/7 < 5/7`，因此按冻结规则不能写成 `CE_PREFERRED`，正式标签为 `HEAD_INDETERMINATE`。在 general + indeterminate 下，预注册规则选择更简单的 R2。

## 八、稳定性与身份异质性

Backbone 贡献在聚合层面稳定：

- A 的 fold 0/4 AUROC 增益为 `+10.68/+11.30 pp`；三个 seed 为 `+14.71/+7.47/+10.79 pp`；38/42 单元为正。
- B 的 fold 0/4 为 `+18.32/+19.00 pp`；三个 seed 为 `+20.98/+16.39/+18.62 pp`；39/42 单元为正。
- A/B 都只在绝对表现已经很高的 N2 小幅为负，其余 6 个 pair 为正。

Head 贡献依赖身份：

- R2 在 N0/N1/N3/N4 的全部 6/6 单元优于 R3；
- R3 在 N2/N5/N6 上只有小幅 pair 均值优势；
- R3−R2 的 fold 0/4 为 `-7.79/-7.14 pp`，三个 seed 为 `-6.58/-7.90/-7.92 pp`，但 pair-level 仍只有反向 4/7，故不能越过预注册一致性门槛。

按四方法平均 AUROC，最困难的 pair 为：

1. N1（DDG-112 + 迷你好望角）：AUROC `61.49%`；
2. N4（DDG-1000 + 达飞罗尔多夫级）：`64.22%`；
3. N3（DDG-1000 + MARVEL CRANE）：`66.19%`。

候选 R2 在 N4/N1/N3 的 AUROC 分别为 `77.42/81.32/81.59%`。N1 的 R2 AUROC 已有 `81.32%`，但 URR 仅 `1.27%`，显示排序指标和当前阈值下的实际拒判存在明显脱节。

## 九、必须关注的身份级失败

R2 的身份级 AUROC/URR：

| Identity | AUROC | URR |
|---|---:|---:|
| CVN77 | 90.90 | 56.83 |
| DDG-1000 | **65.04** | **0.00** |
| DDG-112 | **78.45** | **0.00** |
| MARVEL CRANE | 94.56 | 69.67 |
| 爱达魔都号 | 92.39 | 58.92 |
| 迷你好望角型散货船 | 88.36 | 28.17 |
| 达飞罗尔多夫级 | 92.56 | 61.73 |

R2 未拒绝的 DDG-1000 `6000/6000` 全被判为 DDG-112；未拒绝的 DDG-112 `6000/6000` 全被判为 DDG-1000。迷你好望角的 4310 个 false accepts 中，4153 个（`96.36%`）被吸收到 DDG-1000。R2 全部 25,481 个 false accepts 中，流向 DDG-1000/DDG-112 的合计比例为 `64.00%`。

这证明错误集中在少数身份关系，而不是均匀小误差；但现有汇总不能证明原因是数据/标签、真实 HRRP 相似性、表示、分数还是阈值。

## 十、Length/padding 边界

N0–N6 全部满足预注册 `length-safe` 定义，两个 fold 的 surrogate 越界数均为 0；`length-risk` 为空。safe 子集与全体完全重合，因此方向相同只是集合相同，不能声称 safe/risk 两类均一致。

N1、N2 的 length-only AUROC 为 `0.75`，其余 pair 为 `0.50`。可以写“没有 out-of-support 长度风险”，不能写“没有长度信号”或“模型没有利用长度”。Length 没有参与 gate。

## 十一、正式决定与停止边界

```text
backbone_label = backbone_general_success
head_label = HEAD_INDETERMINATE
recommended_candidate = R2_MS_MEAN_CE
separate_final_test_preregistration_recommended = true
final_unknown_test_authorized = false
```

若 Pro 建议进入下一步，候选只能是当前 R2：冻结的 `HRRPMultiScaleResNet1D`、算术均值融合、线性 CE、当前 100-epoch 训练协议、`-max(raw logits)` 分数和 known-only 95% acceptance 阈值。不得重新打开 R3 或调整 head/阈值后再比较，也不得查看最终 unknown 后修改协议。

无论 Pro 如何建议，当前任务都不得自动运行最终 3 unknown 或偶数角 test；需要用户对一份新的最终测试预注册明确批准。

## 十二、原始证据

本机 summary-only 目录：

`artifacts/results/ms_mean_head_factorial_surrogate_v1/confirmation_gpu_62e318d/`

关键哈希：

- 汇总归档：`39ebd7f008b60652ff1b981579bc3c85b564bb46f9421836be6525c0840ba2a2`；
- `summary.json`：`33f768d17126794976a38383085dbc433b6c6e88ab9743c7031bca50126e5ed5`；
- `integrity.json`：`5cefc52ee23e9c88a10b99762e49f9f716f34384dd574d5f0f9ed0706212b4c0`；
- `comparison_summary.json`：`fa16ef71e480ca9cbfb6c80cd9879e6729e6c987ba2fc0dcd04c126e96ec1ba2`；
- `factorial_decision.json`：`0ab8bbdb52192ce404293c2321a1410e1748eaa726ceabeaba8e30b705df83f7`。

完整 168 项产物保留在 GPU 容器独立目录，不提交 Git。`RESEARCH_CONTEXT.md` 未修改。
