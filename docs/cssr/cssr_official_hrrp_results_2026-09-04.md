# 官方语义 CSSR-HRRP 基线结果

> 日期：2026-09-04
>
> 阶段：P3 独立官方语义基线验证（Stage B）
>
> 实验：`official_cssr_hrrp_pilot_v1`
>
> 分支：`codex/cssr-mechanism-official-baseline`
>
> 产生结果的代码提交：`2f960b9a09b69ba74a8ecb254d0e6567deee4c3b`
>
> 配置 SHA-256：`11a92b15ca21f6b025f1fce0dfcb6411f3fa53ef1d26f22693c73cf5d46fa188`

## 1. 直接结论

pCSSR 核心实现对固定官方源码的差分已经通过，项目定义的两视角聚合也通过独立 NumPy 参考检查；但 N1 smoke 只完成 `2/4` 个训练单元：O1、O3 成功，O2、O4 均因 raw train 预测类别 3 为空而触发预注册硬失败。因此 `12` 项 pilot 没有启动，本任务不能给出 O0–O4 性能比较、CSSR signal/no-signal 标签或继续研究多视角 CSSR 的科学结论。

已经确认：

1. 固定官方源码的 pCSSR 前向、损失、梯度、S1/S2/S3 和标准化在 RTX 4090 上通过 float32 与 float64 差分；项目定义的两视角聚合通过独立 NumPy 参考检查。
2. O1 与 O3 的单元训练、隔离审计和 checkpoint 精确重放通过。
3. O2 与 O4 都在建立官方分数模板时抛出同一错误：`official score template prediction class 3 is empty`。
4. 该错误正是预注册的硬失败条件；smoke 汇总为 `hard_failed_incomplete`，decision 为 `diagnostic_smoke_failed`。
5. 最终三类 unknown 和偶数角 test 均未生成或读取；confirmation 与自动后续实验均未获授权。

尚未验证：官方语义 pCSSR 在当前 HRRP 数据上的开集性能、已知类分类可靠性、MARVEL/DDG 身份问题，以及端到端训练相对冻结训练或 R2 强基线的收益。

## 2. 方法与证据边界

本实验是把固定官方 pCSSR 的核心模型、损失和评分语义适配到一维 HRRP 特征，不是复现官方图像数据集结果。官方参考固定为：

- 论文：Class-Specific Semantic Reconstruction for Open Set Recognition；
- 仓库：`xyzedd/CSSR`；
- commit：`d5a99e91f310ec274c7bfe5796fb270719a07ab3`。

源码没有复制进本仓库。运行时先核对 commit 和以下六个文件：

| 官方文件 | SHA-256 |
|---|---|
| `methods/cssr.py` | `0d23558c6a3cc4bf068036502a8ab43ee6278aecd91d96741f7375a142d9c5a3` |
| `methods/cssr_ft.py` | `31244f194d91f6cab0bdf34eb14a0ed3b58f25b6c49a44042bb96baa9977fb16` |
| `configs/basic.json` | `672375c6838004ae604509ba57098c7fefd17b6ac0f38e7c955fc8c09ba3192a` |
| `configs/pcssr.json` | `353b0768cc6ee60ac76c110a22da8bdb5c15179260d4abeb2f43fee422d24c6b` |
| `configs/pcssr/cifar10.json` | `ce5c7187cab1d8a7387526e459dc21c257f407e15e2304a91f618a8d8d34b0ab` |
| `configs/pcssr/imagenet.json` | `170b8b7f86a2bde8fd409feaa96edfbfbd4226cc7ed9d1a564db8ca8a783b505` |

O1/O3 是为当前 HRRP 协议冻结的 matched linear controls，使用 `gamma=0.1` 和两视角 softmax 概率均值；它们不是官方 `linear.json` 配置，也不能作为官方图像 benchmark 数字。

## 3. 冻结候选

| ID | 定义 | smoke 状态 |
|---|---|---|
| O0_R2_CC_MLS | 封存 R2 + 类别条件 MLS | smoke 不训练；原计划在完整 pilot 中复用 |
| O1_OFFICIAL_LINEAR_FT | 冻结 encoder + matched linear head | 成功；单元审计通过；checkpoint 精确重放 |
| O2_OFFICIAL_PCSSR_FT | 冻结 encoder + official-semantics pCSSR head | 失败；预测类别 3 为空 |
| O3_OFFICIAL_LINEAR_E2E | matched linear，epoch 6 起端到端 | 成功；单元审计通过；checkpoint 精确重放 |
| O4_OFFICIAL_PCSSR_E2E | pCSSR，epoch 6 起端到端 | 失败；预测类别 3 为空 |

smoke 固定为 N1、完整 `720` 个 unique train-known bases、每个方法 `1` epoch 和 `6` 次 optimizer update。它只检查训练与审计链路，不用于方法性能比较，所以本报告不展示 O1/O3 的 smoke 指标。

## 4. 官方差分与验证结果

正式 GPU oracle 在 NVIDIA GeForce RTX 4090 上通过：

- float32：`passed`，固定容差 `rtol=1e-5, atol=1e-6`；
- float64：`passed`，固定容差 `rtol=1e-9, atol=1e-11`；
- pCSSR 与 matched linear 的前向、loss、参数和输入梯度通过；
- clip 边界、softmax average、S1、abs-train/signed-test S2、`G_p_pro(p=8)`、S3、augmented-train mean/std 与 full score 通过；
- 项目定义的两视角概率、共同预测类、八种 score 组合和“越大越未知”的方向通过独立 NumPy 参考检查；
- 两次确定性重复逐字段一致；
- CUDA 确定性算法开启，TF32、AMP 和 cuDNN benchmark 关闭。

oracle 文件 SHA-256 为 `82fafa77188027d5a4529f5b6eb15e1bdc19a17c41963210df54c5b14139ae50`。

代码检查记录：

| 环境 / 提交 | 检查 | 结果 |
|---|---|---|
| 本机 / `2f960b9` | 完整 pytest | `563 passed, 10 skipped` |
| GPU / `55c5965` | 完整 pytest | `571 passed, 2 skipped` |
| GPU / `2f960b9` | 修复相关专项测试 | `16 passed` |
| GPU / `2f960b9` | 正式 CUDA oracle | `passed` |
| 本机 / `2f960b9` | Python compile、配置加载与 `git diff --check` | `passed` |

`55c5965` 的完整 GPU 测试发生在两行 Stage A surrogate sentinel 修复之前；修复后的相关测试与正式 oracle 已在最终结果提交上重跑。该修复只把 Stage A 审计期望标签从 `-1` 对齐为封存 manifest 的实际模型标签 `5`，没有改变 Stage B 模型、配置、分数或 gate。

## 5. Smoke 硬失败

| 任务 | 运行结果 | 审计结果 | checkpoint replay | 结论 |
|---|---|---|---|---|
| N1 / O1 | success | passed | exact | 链路通过 |
| N1 / O2 | exception | 无完整单元目录 | not available | 类别 3 预测分组为空，硬失败 |
| N1 / O3 | success | passed | exact | 链路通过 |
| N1 / O4 | exception | 无完整单元目录 | not available | 类别 3 预测分组为空，硬失败 |

O2/O4 的一致错误发生在训练结束后建立官方统计量时：

```text
_fit_official_statistics -> build_official_score_templates
ValueError: official score template prediction class 3 is empty
```

这里的“类别为空”是指：一轮 pCSSR 训练后，用模型对 raw train-known 样本预测并按预测类别分组时，没有样本被预测为模型类别索引 `3`。它不是训练数据缺类；N1 的五个真实训练类均存在。日志和已通过的 oracle 也没有显示 GPU 故障、NaN/Inf 或官方语义差分错误。

预注册明确规定，O2/O4 任一 raw train 预测类别为空即硬失败，且任一 smoke 单元失败都禁止 pilot。执行结果因此是：

```text
smoke.status = hard_failed_incomplete
smoke.decision = diagnostic_smoke_failed
smoke.planned_unit_count = 4
smoke.successful_unit_count = 2
smoke.gate = null
pilot.execution = not run
pilot_gate = not evaluated
selected_method = null
```

其中 `pilot.execution` 和“not evaluated”是对执行状态的报告性说明；没有伪造一个未生成的 pilot 状态产物。失败后的 smoke phase audit 本身通过，证明 `2/4` 成功、`2/4` 缺失和硬失败封印均被如实记录，不代表四个训练单元全部通过。

## 6. 预注册问题的回答状态

| 预注册问题 | 状态 | 原因 |
|---|---|---|
| O0–O4 九项完整指标 | not evaluated | pilot 未运行；smoke 指标禁止用于性能比较 |
| O2−O1 frozen-head CSSR effect | not evaluated | O2 smoke 硬失败且无三-pair pilot |
| O4−O3 end-to-end CSSR effect | not evaluated | O4 smoke 硬失败且无三-pair pilot |
| O4−O2 joint representation effect | not evaluated | 两个 pCSSR 单元均未通过 smoke |
| O4−O0 strong-baseline comparison | not evaluated | pilot 未运行 |
| S1、S2、S3 与 full 的贡献 | not evaluated | 正式 score 模板未完整建立 |
| pCSSR known Accuracy/NLL/Brier/ECE | not evaluated | pilot 未运行；smoke 不作性能报告 |
| MARVEL 与 DDG 问题是否仍存在 | not evaluated | N2/N4 pilot 未运行 |
| 唯一 CSSR 结果标签 | unassigned | 不满足“完整 pilot 已通过审计”的贴标签前提 |
| 是否继续研究多视角 CSSR | 无科学结论 | 当前结果只回答执行链路，不回答方法性能 |

因此不能把本次结果写成 `official_cssr_no_signal`，也不能声称 pCSSR 优于、劣于或等于 matched linear 或 O0。

## 7. 授权与停止状态

```text
confirmation_allowed = false
automatic_followon_authorized = false
final_unknown_test_authorized = false
final_unknown_used = false
even_angle_test_used = false
```

本任务按预注册在失败报告后停止。没有运行 N1/N4/N2 × O1–O4 的 `12` 项、`40` epoch pilot，没有进入 confirmation，没有访问最终 unknown 或偶数角 test，也没有根据 smoke 修改网络、超参数、模板规则或 gate。`RESEARCH_CONTEXT.md` 未修改。

## 8. 结果产物与哈希

完整运行产物保留在 GPU 容器的独立目录：

```text
/root/hrrp-runs/official_cssr_hrrp_pilot_v1_2f960b9/
```

本机保存 summary-only 镜像：

```text
/Users/bytedance/Desktop/科研空间/artifacts/results/official_cssr_hrrp_pilot_v1/official_cssr_hrrp_pilot_v1_2f960b9_summary/
```

关键文件：

| 文件 | SHA-256 |
|---|---|
| `oracle/official_cssr_oracle.json` | `82fafa77188027d5a4529f5b6eb15e1bdc19a17c41963210df54c5b14139ae50` |
| `smoke/task_plan.json` | `d15af45581636e1d918368cc7ef45d15055e6523177f66dd01515758050b48f4` |
| `smoke/phase_summary.json` | `663b5671402f98ef6c4e670a17f9de6ff6d46bbd834689e2447aa28a4b91e1cd` |
| `smoke/task_audit.csv` | `cd4c206334bd17e0155f88787429240941f9054d3e67393836b953724211af9f` |
| `smoke/artifact_hashes.json` | `93ab2d711fa5faced383accdc88f903588a7a86065bff2d8ae4a00db199a5d42` |
| `smoke/_PHASE_INCOMPLETE.json` | `524a149806adb807ea8719724839177cd2a1034c01d6a8edc8364d7c12b34f01` |
| `logs/smoke_O2.log` | `b948abf222728ffe034f751584752cfdd95b889058d2e096822d22a54fe6f49a` |
| `logs/smoke_O4.log` | `b948abf222728ffe034f751584752cfdd95b889058d2e096822d22a54fe6f49a` |

summary 归档 SHA-256 为 `e9c9c0d1d8da37085a9a086a4766a126788c2aeb2c38ae645479d5cfbb07a06d`。原始数据、checkpoint、manifest、逐样本预测和日志不提交 Git。

## 9. 下一步建议

当前没有可冻结到 confirmation 的 CSSR 配置，也不应从本次 smoke 继续设计多视角创新。若要再次验证官方语义 pCSSR，必须另开任务并在看任何新性能前重新预注册，明确改变的是 smoke 训练预算还是空预测类别处理；本次失败状态和原始产物必须原样保留，不能覆盖或重新解释。
