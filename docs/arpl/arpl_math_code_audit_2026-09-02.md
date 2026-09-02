# ARPL 论文—官方代码数学审计

> 日期：2026-09-02  
> 阶段：AMDR P0 结束后的新 P1 可行性诊断  
> 状态：实现前冻结  
> 官方仓库：`gary23ai/ARPL`，审计 commit `3ede8b38e1cfb9d70e106cc19d563453110c36ab`

## 1. 结论

本项目第一轮实现采用官方 `ARPLoss` 的实际定义，不依据示意图重写符号：每类一个 reciprocal point，类别 logit 等于逐维平均平方 L2 距离减去点积；训练用 `CrossEntropy(logits / temperature)`，再加可学习 radius 的 margin ranking loss；推理以每行最大原始 logit 作为 knownness，项目接口取其相反数作为“越大越未知”的分数。

本轮不实现 GAN/confusing samples，因此名称固定为 `ARPL_LITE`，不能写成完整 `ARPL+CS`。

## 2. 官方代码事实

### 2.1 Reciprocal points 和距离

`loss/Dist.py:7-17`：reciprocal points 参数形状为
`[num_classes * num_centers, feature_dim]`，随机初始化为 `0.1 * randn`。官方默认 `num_centers=1`。

`loss/Dist.py:19-36`：

```text
dist_l2(x, p) = sum_j (x_j - p_j)^2 / feature_dim
dist_dot(x, p) = x^T p
```

随后 reshape 为 `[batch, num_classes, num_centers]` 并对 centers 求均值。本项目冻结每类一个 reciprocal point，不扩展多中心。

### 2.2 ARPL logits

`loss/ARPLoss.py:19-25` 明确给出：

```text
logits = dist_l2 - dist_dot
classification_loss = cross_entropy(logits / temperature, labels)
```

因此 logit 越大越支持对应类别；它不是“离正原型越近分数越高”的普通原型分类。实现和手算测试必须保留减号、L2 除以特征维数以及原始 logits 的温度位置。

### 2.3 Radius 与 margin loss

`loss/ARPLoss.py:10-16,27-32`：

```text
d_known = mean_j (x_j - p_y,j)^2
loss_margin = MarginRankingLoss(margin=1)(radius, d_known, target=+1)
loss_total = loss_cls + weight_pl * loss_margin
```

PyTorch 该项等价于 `max(0, d_known - radius + 1)` 的 batch 均值。`radius` 是从 0 初始化的单个可学习标量；`weight_pl` 和 `temperature` 的官方命令行默认分别为 `0.1` 和 `1.0`（`osr.py:30-40`）。

### 2.4 训练和推理

- `core/train.py` 同时优化 backbone 和 loss module 参数，即 reciprocal points 与 radius 必须进入优化器；
- `loss/ARPLoss.py:24` 在无标签推理时返回原始 logits；
- `core/test.py:25-27` 用原始 logits 的 `argmax` 分类；
- `core/test.py:52-54` 对 known/unknown 的每行 logits 取最大值，并将其作为 knownness 送入 OOD 评价；
- 本项目统一 unknown score 为 `-max(logits)`，保证“越大越未知”。

### 2.5 RPL、ARPL_LITE 与 ARPL+CS 边界

- 官方 `RPLoss.py` 只使用 L2 reciprocal distance，并以 MSE 约束距离和 radius；其返回推理 logits 还经过 Softmax；
- 官方 `ARPLoss.py` 加入 `L2 - dot` 方向项和 adaptive margin constraint；
- `osr.py --loss ARPLoss` 且不加 `--cs` 对应本项目的 `ARPL_LITE`；
- `--cs` 还会引入 generator、discriminator、辅助 BN 和 fake loss，才属于 `ARPL+CS`；本轮禁止实现。

## 3. HRRP 适配中的必要改动

| 项目 | 官方实现 | 本项目适配 | 是否改变数学定义 |
|---|---|---|---|
| 输入/backbone | 图像分类网络 | `[B,2,601]`，共享 `SharedHRRPEncoder1D` 后 mean pooling | 否，只替换特征提取器 |
| 特征维数 | 128 | 128 | 否 |
| 已知类别数 | 数据集决定 | 每个 surrogate split 为 5 | 否 |
| reciprocal points | 每类1个，`0.1*randn` | 相同 | 否 |
| target device | `torch.ones(...).cuda()` | `torch.ones_like(d_known)` | 否，修复 CPU/device 硬编码 |
| API | criterion 接收网络特征 | 独立 head 接收 fused feature，并显式返回诊断量 | 否 |
| unknown score | 最大 logit 为 knownness | `-max_logit` | 仅统一方向 |
| GAN/CS | 可选 | 关闭 | 本项目仅称 ARPL_LITE |

不加入特征归一化、余弦归一化、额外 MLP、per-view reciprocal points、伪未知损失或真实未知数据。

## 4. 实现验收条件

1. 用二维手算样例逐项验证 L2/维数、dot 和 `L2-dot`；
2. 验证 margin loss 等于 `max(0,d_known-radius+1)`；
3. reciprocal points、radius 和共享 encoder 均可获得有限梯度；
4. CPU 可运行，代码中不得硬编码 `.cuda()`；
5. 推理最大 logit 增大时，统一 unknown score 必须减小；
6. 两视角交换不改变 per-view 集合、fused feature 和 logits。

## 5. 证据边界

论文解释 reciprocal points 和 bounded open space 的理论动机；上述精确数值定义以审计 commit 的官方代码为实现证据。当前尚未在 HRRP 上得到任何 ARPL 结果，不能把论文或视觉数据集结果外推为本项目结论。

来源：

- 论文：https://arxiv.org/abs/2103.00953
- 官方代码：https://github.com/gary23ai/ARPL

