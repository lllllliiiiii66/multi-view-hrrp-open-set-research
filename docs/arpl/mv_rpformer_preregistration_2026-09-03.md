# MV-RPFormer surrogate OSR 预注册

> 日期：2026-09-03
> 阶段：P3（在前序固定表示与手工多视角证据未形成稳定增益后，验证新的 open-set-oriented 多视角表示）
> 状态：运行前冻结
> 基线提交：`69a1d3e82756dd781e785ddca1c85c3d1ee4037b`

## 1. 唯一研究问题

检验“是否存在一个已知类别能够同时解释所有视角”能否被双路径、置换不变的集合模型学到，并通过类别条件拒判稳定提升 surrogate unknown 检测。上一轮 `arpl_mv_evidence_surrogate_v1` 的 F1–F3 手工规则与 `no_stable_gain` 结论保持不变，本轮不修改或覆盖其代码、配置、报告和产物。

## 2. 证据边界

- 只使用 7 个 source-known 类的奇数角开发池；S0–S2 用于 development，尚未运行的 C0–C3 用于 confirmation。
- 最终 3 个 unknown、偶数角 test、角度、视角序号和位置编码均不进入模型、训练、选择或评价。
- surrogate unknown 只用于 development/confirmation 的开放集评价，不参与训练、伪未知生成、checkpoint 选择或参数修改。
- development 只作稳定性和消融诊断，不设性能 gate；只要数值、置换、反算和泄漏审计通过，就按冻结配置运行 confirmation。
- C0–C3 上不改架构、损失权重、伪未知比例、学习率、epoch 或 checkpoint 规则。
- length/padding 风险只记录，不作为停止或删 split 的依据。

## 3. 冻结模型

共享多尺度一维残差编码器接收 `[B,601]`：31 点大卷积核 stem，32/64/128 三个阶段，每阶段含 3/7/15 三个标准一维卷积分支、残差、GELU 与 `dropout=0.1`；全局平均池化与最大池化拼接后投影到 128 维。两个视角严格共享参数。标准卷积分支在冻结前的同机受控计时中比深度可分离版本快约 5 倍，因此冻结为正式实现；该计时只用于工程实现选择，不作为方法性能证据。

集合路径使用一个 pre-LayerNorm SAB（128 维、4 heads、FFN 256、dropout 0.1、无位置编码），保留两个上下文化 token `z1,z2`；双 seed PMA 输出有固定语义的 `g_cls,g_rej`。不增加输出 SAB。

全局和逐视角 ARPL head 分别拥有独立 reciprocal points 和 radius，均使用已经通过官方逐前向/逐梯度差分验证的定义。分层表示损失固定为：

```text
L_repr = L_global + 0.5 * 0.5 * (L_view(z1,y) + L_view(z2,y))
```

不增加特征一致性损失。

拒判器输入 `g_rej` 与冻结的类别条件证据。为同时满足“包含两个视角对预测类的支持”和严格置换不变，两个支持值按数值降序排列后输入；另外输入其 mean/min/std、全局 unknown score、两视角 softmax JS、`||z1-z2||²/128`、分类 PMA 各 head 对两个视角的熵再跨 head 平均，以及全局 top1-top2 margin。预测类始终由 global logits 的 argmax 决定。

## 4. 伪未知与损失

伪未知只从当前 split 的 train-known 构造。mismatch 使用 anchor 的 view1 与不同类别 partner 的 view2；coherent mixup 在编码后对两个视角使用同一 `lambda ~ Beta(2,2)`，并限制到 `[0.3,0.7]`。M6/M7 各使用 50% mismatch 和 50% mixup；M5 作为 mismatch-only 消融，为维持真实/伪未知数量 1:1，全部伪未知均为 mismatch。

```text
L_rej = BCE(real,0) + 0.5*BCE(mismatch,1) + 0.5*BCE(mixup,1)
L_total = L_repr + 1.0*L_rej + 0.1*KL(uniform || softmax(global_logits_pseudo))
```

M5 没有 mixup 项时，使用 `BCE(real,0)+BCE(mismatch,1)`。伪未知采样保存确定性摘要哈希、类别不同审计、来源角色和 mixup lambda 范围，不保存或使用 surrogate/final unknown。

## 5. 方法矩阵与公平性

Development 和 confirmation 均运行 M0–M7，以便完整回答全部受控比较：

| ID | 定义 |
|---|---|
| M0 | 当前浅层共享 CNN + mean + CE MLS |
| M1 | 当前浅层共享 CNN + mean + ARPL-lite |
| M2 | 新多尺度编码器 + mean + global ARPL |
| M3 | 新编码器 + Set Transformer + global ARPL |
| M4 | M3 + 独立 view-level 完整 ARPL |
| M5 | M4 + dual-token rejector + mismatch |
| M6 | M4 + dual-token rejector + mismatch + coherent mixup |
| M7 | 与 M6 同结构和伪未知，但 global/view head 改为两个独立 CE head |

同一 split/seed 的八个方法共享完全相同的真实 pair manifest、归一化、标签和预测顺序。M0/M1 保留历史架构，但不复用旧的 early-stop/best-calibration checkpoint；全部方法在本任务中按相同优化器和固定第 100 epoch checkpoint 重跑，保证 checkpoint 规则一致。

## 6. 冻结训练与评价

AdamW，学习率 `3e-4`，weight decay `1e-4`，每个 batch 含 64 个真实 known，并在拒判阶段另生成 64 个 pseudo，总计 100 epochs；5 epochs 线性 warmup 后 cosine decay。前 30 epochs 只训练 `L_repr`，第 31 epoch 起加入拒判和 uniform 损失。每 epoch 保存 known calibration 诊断指标，但不据此选模型、早停或改变 confirmation；正式比较唯一使用 epoch 100。

S0/S1/S2 使用种子 20260830。C0–C3 使用 20260830/20260831/20260832，共 12 个 confirmation 单元。未知分数统一为越大越未知；无 rejector 的模型用 global `-max(logit)`，M5–M7 用 `p_unknown`。阈值只由本模型的 known calibration 按 95% 已知接受率确定。

Merlin 正式运行固定为每个进程 4 个 PyTorch intra-op 线程、1 个 inter-op 线程，最多同时运行 2 个独立方法任务；线程数和源码 SHA-256 随每个模型产物保存。并行只发生在互不共享文件的 split/seed/method 目录之间。

## 7. 固定判断规则

M6 主方法成功须同时满足：相对 M4 的 12 单元平均 AUROC 至少 `+0.02`、至少 8/12 为正、平均 OSCR 不下降、known Accuracy 平均下降不超过 `0.01`、FPR95 平均恶化不超过 `0.02`。

ARPL-specific 还要求 M6 相对 M7 平均 AUROC 至少 `+0.01` 且至少 7/12 为正。组件结论按 M2−M1、M3−M2、M4−M3、M6−M4、M6−M5、M6−M7、M6−M1/M0 分别报告，不把不同负结果合并为一句“方法失败”。

## 8. 验收与停止边界

训练前必须通过配置、ARPL官方差分、模型 shape/梯度、置换不变、伪未知隔离、warmup 和指标反算测试。运行后保存 resolved config、pair manifest/hash、伪未知审计、epoch-100 checkpoint、注意力和各层 token、global/view logits、head 参数与 radius 轨迹、预测、指标、paired delta、环境和哈希。

完成 development、强制 confirmation 和结果报告后停止。不得自动运行最终 7-known/3-unknown 或偶数角 test；是否冻结 M6 只按上述预注册门槛决定。
