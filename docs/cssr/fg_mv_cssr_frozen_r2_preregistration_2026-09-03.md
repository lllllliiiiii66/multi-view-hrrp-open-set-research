# 冻结 R2 + 逐视角类别语义重构快速实验预注册

> 日期：2026-09-03
>
> 阶段：P3 独立快速机制验证
>
> 实验：`fg_mv_cssr_frozen_r2_v1`
>
> 起点：`codex/ms-mean-head-factorial` 分支结果提交 `edb05062d07be1984067f91759d6029cd9c0bf9a`
>
> 状态：实现与任何本实验 smoke、pilot 性能产生前冻结

## 1. 唯一研究问题与停止边界

上一轮已经确认多尺度 HRRP 骨干在 CE 和 ARPL 两种 head 下均稳定优于浅层骨干，并按预注册规则得到唯一简化候选 `R2_MS_MEAN_CE`。本轮不重新回答 backbone 或 CE/ARPL 选择问题，只验证：

> R2 融合分类器提出的同一个已知类别，若不能分别解释两个单视角的语义特征图，能否据此更可靠地拒绝 surrogate unknown？

R2 的多尺度 encoder、算术均值融合、线性 CE head、normalization、pair manifest、类别顺序和 epoch-100 checkpoint 全部冻结。本轮不重新训练或微调 R2，只训练每类一个小型重构头。

禁止 ARPL、rCSSR、reciprocal points、Set Transformer、attention、PMA、逐视角分类 head、伪未知、mismatch、mixup、GAN、confusing samples、学习型拒判器、分数加权、超参数搜索和最终测试。最终 3 个 unknown 类及偶数角 test 的 pair、特征和预测不得生成或使用。

本轮最多训练 3 个 pilot CSSR head；仅当预注册 pilot gate 通过时，再训练 4 个 confirmation CSSR head。无论结果如何，都不得自动进入最终测试、端到端 CSSR 或 CSSR+ARPL。

## 2. 冻结 R2 与输入特征

每个实验单元加载上一轮正式 GPU confirmation 中相同 `pair_id / fold_0 / seed_20260830 / R2_MS_MEAN_CE` 的 epoch-100 checkpoint，并验证：

- checkpoint 来自实验 `ms_mean_head_factorial_surrogate_v1`；
- method 为 `R2_MS_MEAN_CE`，angle fold 为 0，初始化 seed 为 20260830；
- 原配置 SHA-256 为 `c11daa6e2e5a7d7b72bc36840e60fc871f332c4fc85652636c729aa2eba14c71`；
- 原 pair manifest、normalization、类别顺序、checkpoint 和 R2 输出均与已封存产物一致；
- 新代码可 `strict=True` 加载旧 state dict；所有 R2 参数均为 `requires_grad=False`。

R2 的正式预测保持：

\[
Z_v = \operatorname{feature\_map}(E(x_v))\in\mathbb R^{128\times L},
\]

\[
h_v=\operatorname{projection}([\operatorname{avg}(Z_v),\operatorname{max}(Z_v)]),
\quad g=(h_1+h_2)/2,
\]

\[
\operatorname{logits}_{\rm fused}=W_{\rm CE}g+b,
\quad \hat y=\arg\max_k\operatorname{logits}_{{\rm fused},k}.
\]

只读接口 `forward_feature_map` 返回最后一个多尺度残差 stage 之后、global average/max pooling 之前的特征图。对 601 维输入，预期形状为 `[B,128,76]`。接口不得新增参数、改变原 `forward` 输出或改变旧 checkpoint 的 state-dict key。

CSSR 只重构该单视角语义图，不重构原始 601 维 HRRP，也不重构完全池化后的 128 维向量。已知类别预测始终使用冻结 R2 的 `\hat y`；CSSR 只提供 unknown score。

## 3. 官方 CSSR 证据与实现边界

官方参考固定为：

- repository：`https://github.com/xyzedd/CSSR`；
- commit：`d5a99e91f310ec274c7bfe5796fb270719a07ab3`；
- `methods/cssr.py` SHA-256：`0d23558c6a3cc4bf068036502a8ab43ee6278aecd91d96741f7375a142d9c5a3`；
- `methods/cssr_ft.py` SHA-256：`31244f194d91f6cab0bdf34eb14a0ed3b58f25b6c49a44042bb96baa9977fb16`；
- `configs/basic.json` SHA-256：`672375c6838004ae604509ba57098c7fefd17b6ac0f38e7c955fc8c09ba3192a`；
- `configs/pcssr.json` SHA-256：`353b0768cc6ee60ac76c110a22da8bdb5c15179260d4abeb2f43fee422d24c6b`；
- `configs/rcssr.json` SHA-256：`af40084644b4794559403f91e9d43a3008420df78484d87ec825e6d48b3d6f68`。

本轮名称固定为 `PCSSR_CORE_1D`。它只复用官方 pCSSR 的类别特定重构、负 L1 logit、clip、`softmax_avg` 和 scale-normalized reconstruction score 核心，不实现完整 pCSSR 的 feature prototype、Gram score、训练分数标准化或多分数集成，因此不得表述为完整 CSSR 复现。

官方固定提交中，pCSSR 从其自身预测类计算开放集分数且没有 epsilon。本项目把同一核心重构量扩展到所有类别，再用外部冻结 R2 的预测类构造 B4，并增加固定 epsilon 保证零激活时有限；这些均明确属于本项目的一维、多视角类别条件扩展。

## 4. `PCSSR_CORE_1D` 的冻结数学定义

每个已知类别 \(k\) 拥有一个互不共享参数的自动编码器：

```text
Conv1d(128, 64, kernel_size=1, bias=False)
Tanh
Conv1d(64, 128, kernel_size=1, bias=False)
```

无 skip、无额外隐藏层、无 attention、decoder 无激活。令：

\[
e_{v,k,t}=\sum_c\left|A_k(Z_v)_{c,t}-Z_{v,c,t}\right|,
\]

\[
q_{v,k,t}=\operatorname{clip}(-\gamma e_{v,k,t},-100,100),
\qquad \gamma=0.1.
\]

官方 `softmax_avg` 顺序固定为先在每个位置沿类别维 softmax，再沿位置平均：

\[
P_{v,k}=\frac{1}{L}\sum_t
\frac{\exp(q_{v,k,t})}{\sum_j\exp(q_{v,j,t})},
\]

\[
\mathcal L_{\rm cssr}=-\frac1B\sum_i\log P_{i,y_i}.
\]

训练损失不是“先平均 logit 再 softmax”，也不是逐位置交叉熵的平均。

按官方 `R[0]/R[1]/R[1]` 的逐位置归一化顺序，定义：

\[
a_{v,t}=\frac1{128}\sum_c|Z_{v,c,t}|,
\]

\[
\rho_{v,k}=-\frac1L\sum_t
\frac{q_{v,k,t}}{\max(a_{v,t},\epsilon)^2}
=\frac1L\sum_t
\frac{\min(\gamma e_{v,k,t},100)}{\max(a_{v,t},\epsilon)^2},
\qquad \epsilon=10^{-8}.
\]

`rho` 越大表示越不符合类别 \(k\)。归一化必须逐位置完成后再平均，不得改成先平均分子或分母。

## 5. 官方核心差分测试

任何 CSSR 训练前，使用同一输入、类别 AE 权重、标签、gamma、L1、clip 和 `softmax_avg`，将一维输入 `[B,C,L]` 转为官方 Conv2d 核心的 `[B,C,1,L]`，逐项比较：

- reconstruction；
- 类别重构 logits；
- class probabilities；
- classification loss；
- 输入 feature 梯度；
- 每个 encoder/decoder 权重梯度。

容差固定为：float32 `rtol=1e-5, atol=1e-6`；float64 `rtol=1e-9, atol=1e-11`。另验证 clip 饱和、零激活 epsilon、类别 AE 独立性、softmax 与空间平均顺序、逐位置幅度归一化顺序。核心差分失败即停止，不得训练。

## 6. CSSR 训练样本与参考分布

每个实验单元继续使用上一轮同一 bundle、normalization 和 pair manifest。

### 6.1 AE 训练集

只从 `train_known` manifest 中提取唯一底层 `sample_id`：每个底层 HRRP 恰出现一次，pair 重复不增加样本，也不作为权重。输入只使用该单视角经过原 R2 normalization 和冻结 encoder 得到的 feature map。

不使用 known calibration、surrogate unknown、最终 unknown 或偶数角 test 训练 AE。类别 batch 在各类数量不等时按固定、可审计的轮转规则尽量平衡；若各类数量相同则每轮恰好使用每个唯一样本一次。保存 unique-base manifest、顺序哈希和 feature-map 哈希。

### 6.2 类别条件重构参考分布

对 known calibration 中真实类别为 \(k\) 的唯一单视角底层样本，计算其 \(\rho_{v,k}\)，建立类别 \(k\) 的参考分布。pair multiplicity 不进入参考权重，surrogate unknown 不进入任何参考分布。

对任一单视角样本定义：

\[
p_{v,k}=\frac{1+\#\{r\in R_k:r\ge\rho_{v,k}\}}{|R_k|+1},
\qquad a_{v,k}=-\log(p_{v,k}+10^{-8}).
\]

`a` 越大表示越不符合该类别。known calibration 自身评分时，对真实类别对应的参考分布按 `sample_id` leave-one-base-sample-out；因为参考分布已按底层样本去重，同一 base 在多个 pair 中不会重复进入或重复删除。其他类别参考分布不存在同一 sample ID，不作额外删除。

### 6.3 类别条件 MLS 基线

令融合最大 logit 为 \(m=\max_k\operatorname{logits}_{{\rm fused},k}\)，MLS 非一致性为 \(r_{\rm mls}=-m\)。对 known calibration 中真实类别为 \(k\) 且 R2 正确预测为 \(k\) 的 pair 建立 \(R_k^{\rm MLS}\)。按 R2 预测类 \(\hat y\) 定义：

\[
u_{\rm B1}=\frac{1+\#\{r\in R_{\hat y}^{\rm MLS}:r\le r_{\rm mls}\}}
{|R_{\hat y}^{\rm MLS}|+1}.
\]

这是预测类别条件的非一致性经验分位数，越大越异常；不再施加第二个可调变换。

known calibration pair 对自身所属且正确预测的参考分布采用 leave-one-pair-out；不按 surrogate 结果改变参考集。若任一预测类别没有非空 known-only 参考分布，实验直接失败。

## 7. 五个固定评分方法

所有方法的 known 类预测完全相同，均为冻结 R2 的 `\hat y`；只改变越大越未知的 score：

| 方法 | Unknown score |
|---|---|
| `B0_GLOBAL_MLS` | \(u=-\max(\operatorname{logits}_{\rm fused})\) |
| `B1_CLASS_CONDITIONAL_MLS` | 第 6.3 节的预测类条件 MLS 异常分数 |
| `B2_INDEPENDENT_VIEW_CSSR` | \(u=\frac12\sum_v\min_k a_{v,k}\) |
| `B3_COMMON_CLASS_CSSR` | \(u=\min_k\frac12\sum_v a_{v,k}\) |
| `B4_FUSION_GUIDED_CSSR` | \(u=\frac12\sum_v a_{v,\hat y}\) |

同时保存 `k_common=argmin_k mean_v a_v,k` 供诊断；它不替换 R2 正式类别预测。

每个方法的 threshold 分别只由该方法的 known calibration pair score 按现有精确秩规则达到目标 95% known acceptance；surrogate unknown 不参与 threshold。禁止 max-view、手工加权、MLS/CSSR 融合、top-2 混合或新 MLP。

## 8. 冻结训练配置与训练动态审计

R2 只读；只优化类别 AE：

```text
optimizer = AdamW
learning_rate = 1e-3
weight_decay = 1e-4
batch_size = 128
epochs = 30
scheduler = none
initialization_seed = 20260903
early_stopping = false
augmentation = none
formal_checkpoint = epoch 30
```

每 epoch 保存 train CSSR loss、train CSSR classification Accuracy、known calibration CSSR Accuracy、正确类重构误差、最小错误类重构误差、二者 margin 和 AE 权重 norm。known calibration 仅作诊断，不参与训练、checkpoint 或超参数选择。即使出现早期 100% train Accuracy 或 calibration loss 恶化，仍固定使用 epoch 30。

训练前只读审计所选 R2 日志：train Accuracy 首次达到 95%/99% 的 epoch、known calibration Accuracy 平台、train loss，以及 epoch 30/50/70/100 的变化。calibration loss/NLL、最大 logit、feature norm 若原日志未记录，明确写为“现有产物无法判断”，不得为补日志重训 R2。

## 9. Pilot 与条件 confirmation

固定 `angle_fold=0`、`R2_seed=20260830`、`CSSR_seed=20260903`。

Pilot 恰含：

- P0=N1：DDG-112 / 迷你好望角型散货船；
- P1=N4：DDG-1000 / 集装箱船达飞罗尔多夫级；
- P2=N2：MARVEL CRANE / 迷你好望角型散货船。

每个 pair 只训练一个 `PCSSR_CORE_1D`，共 3 项。主比较为 B4−B1，同时报告 B4−B0/B2/B3。

B3 或 B4 相对 B1 的 gate 均同时要求：

1. 三个 pair 平均 AUROC delta `>= +2.0 pp`；
2. 至少 `2/3` pair 的 AUROC delta 为正；
3. 平均 OSCR delta `>= 0`；
4. 平均 KCCR delta `>= -1.0 pp`；
5. 平均 FPR95 delta `<= +2.0 pp`。

Pilot 信号按固定顺序唯一决定：

1. B4 通过且 B4 平均 AUROC 不低于 B3：`fusion_guided_signal`，主规则 B4；
2. 否则，B3 通过且 B3 平均 AUROC 高于 B4：`common_class_signal`，主规则 B3；
3. 其他所有未覆盖、冲突或未达门槛情况保守记为 `no_cssr_signal` 并停止。

第 3 条只使用户给出的三分类规则在任何边界情况下都有唯一结果，不改变任一 gate。

仅当得到前两个信号时，在未参与选择的 N0/N3/N5/N6 上各训练一个新 CSSR head，共最多 4 项；不重复 pilot、不增加 fold 或 seed。confirmation 主规则相对 B1 必须同时满足：平均 AUROC delta `>=+2.0 pp`、至少 `3/4` pair 为正、平均 OSCR 不下降、平均 KCCR 下降不超过 1.0 pp、平均 FPR95 恶化不超过 2.0 pp。失败即停止 CSSR 路线；通过只能说明“值得后续完整验证”。

## 10. 评价、错误分析与产物

B0–B4 均输出 Known Accuracy、Known Macro-F1、AUROC、OSCR、FPR95、KCCR、URR、KCCR/URR 调和平均、K+1 Macro-F1。因为预测类完全复用 R2，五方法 Known Accuracy 与 Macro-F1 必须逐位一致。

必须单独分析 DDG-1000、DDG-112、迷你好望角型散货船：各方法 AUROC/URR/FPR95、false-accept 吸收类、DDG-1000/DDG-112 相互吸收、R2 正确但 CSSR 拒绝的 known、R2 高置信错误吸收但 CSSR 成功拒绝的 surrogate、B3 `k_common` 与 R2 `y_hat` 不一致时的行为，以及重构误差随原始方位角的变化。角度只作事后解释，不进入模型或 threshold。

每单元保存：resolved config；R2 checkpoint 路径引用与 SHA-256；原 pair manifest 及 SHA-256；unique-base manifest；feature-map shape/hash；CSSR epoch-30 checkpoint；训练日志；AE norm；类别条件参考分布；逐样本 R2 fused logits、`y_hat`、每视角每类别 `rho/p/a`、B0–B4 score、threshold、reject flag、标签、预测、sample/pair ID；九指标；错误分析；环境、源码和产物哈希。

全部九指标必须从逐样本文件零误差反算。审计还必须确认旧 R2 logits 与预测逐值一致、训练/参考/评价 base ID 角色隔离、pair 重复不改变 unique-base 数量、R2 全冻结、无 ARPL、无 pseudo、无最终 unknown/偶数角 test。

## 11. 运行顺序与最终结论边界

严格顺序：

1. 官方核心差分与完整本地 pytest；
2. Python compile、配置校验、`git diff --check`；
3. P0/N1 极小 smoke：这里的 P0 指 pilot 第一个任务、即 `N1/fold_0/R2 seed 20260830`，固定每个已知类取 2 个唯一 train base、每个评价角色每类取前 2 个 pair、CSSR 训练 1 epoch；只验证链路和审计，性能不参与决策；
4. 三项 pilot；
5. 聚合并按冻结 gate 产生唯一信号；
6. 只有非 `no_cssr_signal` 才运行四项 confirmation；
7. 聚合、逐样本反算、哈希审计并写独立结果报告。

最终只能得出以下之一：快速路线无 CSSR 信号并停止；某一固定 CSSR 规则在新 pair confirmation 失败并停止；或某一规则值得后续完整验证。不能声称最终未知性能、统计显著性、完整 CSSR 复现或方法创新已经成立。

`RESEARCH_CONTEXT.md` 不修改，所有已有实验产物不覆盖，`final_unknown_test_authorized=false`，`cssr_arpl_combination_authorized=false`。
