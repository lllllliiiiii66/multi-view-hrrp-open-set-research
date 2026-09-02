# ARPL 多视角 HRRP 最小复现实验对接（2026-09-02）

## 0. 权威状态与证据边界

目标仓库：`lllllliiiiii66/multi-view-hrrp-open-set-research`

本任务必须从 commit `ce2015a8f53156549eabecc0c231fefca200dbf2` 或其后继提交开始，并先阅读：

1. `AGENTS.md`
2. `RESEARCH_CONTEXT.md`
3. `docs/amdr_open_set/p0_amdr_closure_preregistration_2026-09-01.md`
4. `docs/amdr_open_set/p0_amdr_closure_results_2026-09-01.md`
5. `configs/amdr/p0_closure_known_only_v1.yaml`
6. 仓库现有 `src/hrrp_osr/models/cnn1d.py`、`models/sets.py`、`training/set_models.py` 与 evaluation 接口

**最新权威结论：AMDR P0 已完成并判定为 `reject`。** Fold 4 中，AMDR Accuracy/Macro-F1 为 72.34%/71.97%，Raw two-view concatenation + KNN 为 78.06%/76.84%，差值为 -5.71 pp/-4.87 pp。AMDR正常收敛且无视角塌缩，因此不再作为主要表示基础，只保留为历史或负面对照。

`AGENTS.md` 与 `RESEARCH_CONTEXT.md` 中仍可能存在“P0进行中、P1只能 AMDR+Thresholded KNN”等旧阶段描述。对于本任务，以上 closure 文档和用户本次明确授权覆盖这些**阶段状态**，但不得静默修改 `AGENTS.md` 或 `RESEARCH_CONTEXT.md`。先在独立 handoff、配置和报告中工作；是否回写主上下文由用户另行确认。

## 1. 本任务回答的研究问题

在严格底层 HRRP 隔离协议下，使用相同的轻量、多视角、置换不变 backbone：

> ARPL/RPL 的 reciprocal-point geometry 与边界约束，是否比普通 Cross-Entropy + Maximum Logit Score 提供更好的已知分类与未知排序折中？

这是一项**最小方法复现和可行性诊断**，不是最终创新方法，也不是一次架构搜索。

## 2. 必须先做的论文—官方代码审计

主论文：Chen et al., “Adversarial Reciprocal Points Learning for Open Set Recognition,” IEEE TPAMI 2021, arXiv:2103.00953。

官方代码：`https://github.com/gary23ai/ARPL`（历史链接 `iCGY96/ARPL` 会重定向）。

开始实现前，创建：

`docs/arpl/arpl_math_code_audit_2026-09-02.md`

至少核对：

- `loss/Dist.py` 中 L2/m 与 dot distance；
- `loss/ARPLoss.py` 中 `logits = dist_l2 - dist_dot`；
- temperature、`weight_pl`、learnable `radius`、`MarginRankingLoss`；
- 训练时和推理时 logits 的含义；
- 官方 `core/test.py` 对 logits 取 row-wise maximum 作为 knownness；
- RPL、ARPL-lite、ARPL+confusing samples 的命名边界；
- 当前 HRRP实现相对官方代码做了哪些必要的 shape/device/API 改动。

不要凭论文示意图自行改写距离符号。必须增加手算单元测试。

## 3. 第一轮唯一允许的模型

### 3.1 输入和数据

- 两视角 HRRP，输入形状 `[batch, 2, 601]`。
- 复用严格底层样本隔离、15°角度帧、跨帧配对、pair manifest、显式随机种子和哈希审计。
- 两视角来自同一模态，模型不使用角度标签或位置编码。
- 视角交换必须不改变输出，增加 permutation audit。
- 暂不扩展到 V=3，不使用 AMDR 投影。

### 3.2 Backbone

复用现有：

- `SharedHRRPEncoder1D`，`feature_dim=128`；
- 两视角共享编码器；
- `h_mean = mean(h_view, dim=views)`；
- 不做注意力、不做额外 MLP 搜索、不做数据增强搜索。

优先复用/小幅扩展 `DeepSetsClassifier`，但需要使其可以显式返回：

- per-view features `[B,2,128]`；
- fused feature `[B,128]`；
- logits。

保持历史调用兼容，不破坏已有 B2/B3 代码。

### 3.3 两个公平对照

只实现：

1. `CE_MLS`：同一 backbone + 普通线性分类头 + CrossEntropy；unknown score 为 `-max_logit`（越大越未知）。
2. `ARPL_LITE`：同一 backbone + reciprocal points + paper/reference-code aligned classification + learnable radius/margin。

第一轮**禁止**：

- GAN / confusing samples / ARPL+CS；
- OpenMax、COSTARR、CSSR；
- attention、Set Transformer、view-specific encoders；
- per-view reciprocal fusion；
- 额外未知损失或真实未知训练数据；
- 大范围超参数搜索。

## 4. 开集开发协议：不得先看最终未知类

最终 7-known/3-unknown 与偶数角 test 不得用于本任务的架构、损失权重、epoch、阈值和分数选择。

只用当前 7 个 known identities 的开发数据建立预注册 surrogate OSR splits：

- 建议固定 3 个 identity-holdout splits；
- 每个 split：5 个 train-known + 2 个 surrogate-unknown；
- 具体类别组合使用确定性规则生成并写入配置，禁止根据结果挑组合；
- train-known 用于训练；known calibration用于早停/阈值；surrogate-unknown只用于报告，不参与训练和最佳 epoch选择；
- threshold只依据 known calibration，默认95% known acceptance；
- 若当前严格 odd-angle pool不足以自然形成这些角色，先输出数据设计审计和最小可行调整提案，不得自动改协议。

首轮执行顺序：

1. CPU/小数据 shape smoke；
2. 一个固定 surrogate split 的端到端 smoke；
3. 资源允许时运行全部 3 个预注册 splits；
4. 本任务结束，不自动运行最终 3 unknown。

## 5. 最小超参数边界

不要做网格搜索。优先从官方实现读取合理默认值，并结合 128维特征进行数值审计。只允许在运行前预注册一组主值和至多一组安全回退值，回退只能处理明确的数值发散，不能根据 surrogate unknown AUROC择优。

共同训练预算、优化器、batch size、early stopping、初始化种子必须对 CE_MLS 与 ARPL_LITE 完全一致。最佳 epoch只能由 known calibration Accuracy（次级 Macro-F1）选择，不能看 surrogate unknown。

至少记录：

- feature norm；
- reciprocal point norm；
- radius轨迹；
- true-class reciprocal distance；
- logits min/max/mean；
- loss各分量；
- 是否出现 NaN/Inf。

## 6. 评价与产物

所有 unknown score统一“越大越未知”。至少报告：

- known-class Accuracy；
- known Macro-F1；
- AUROC；
- OSCR；
- FPR95；
- KCCR、URR及调和平均（若现有接口支持）；
- K+1 Macro-F1。

每个 split/seed 单独报告，再聚合均值；不得只选最好 split。

保存：resolved config、数据和pair manifest及SHA-256、类别角色、随机种子、checkpoint、training log、per-view/fused features（可留本地不提交）、logits、unknown scores、阈值来源、逐样本 predictions、metrics、环境和 git commit。表格必须可由逐样本文件反算。

## 7. 建议文件布局（可按仓库现状小幅调整）

- `src/hrrp_osr/models/arpl.py`
- `src/hrrp_osr/training/arpl_pilot.py`
- `configs/experiments/arpl/arpl_lite_surrogate_osr_v1.yaml`
- `tests/test_arpl_distance.py`
- `tests/test_arpl_loss.py`
- `tests/test_arpl_protocol.py`
- `docs/arpl/arpl_math_code_audit_2026-09-02.md`
- `docs/arpl/arpl_lite_surrogate_preregistration_2026-09-02.md`
- `docs/arpl/arpl_lite_surrogate_results_2026-09-02.md`

不要覆盖旧 B0-B6、AMDR产物或修改其历史语义。

## 8. 必须覆盖的测试

- 官方距离公式的手算小例；
- class/recriprocal index和shape；
- loss可反向传播、参数有梯度；
- CPU device，不得硬编码 `.cuda()`；
- unknown score方向；
- 两视角交换不变；
- CE与ARPL使用同一pair manifest、类别顺序和训练预算；
- surrogate unknown不进入训练、早停、阈值或任何分布拟合；
- 最终3 unknown和偶数角test没有被加载/生成；
- 逐样本结果能够反算聚合指标；
- 历史模型接口兼容。

## 9. 交付和停止

实际运行 `pytest`、目标新增测试、`git diff --check`。若严格数据 bundle可用，运行 smoke及预注册 surrogate splits；若不可用，只完成代码、测试、配置、文档和精确命令，不得编造结果。

最终只汇报：

1. math-vs-code audit结论；
2. 修改文件；
3. 实际运行的检查；
4. CE_MLS 与 ARPL_LITE 的同协议结果；
5. 数值诊断；
6. 是否值得进入完整 ARPL+CS或 per-view reciprocal evidence；
7. 尚未运行的最终 open-set确认。

本任务结束后停止。不得自动启动 CSSR、COSTARR、最终 unknown测试或新的多视角融合方法。
