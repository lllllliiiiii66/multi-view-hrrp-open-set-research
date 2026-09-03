# MV-RPFormer confirmation GPU 运行补充预注册

日期：2026-09-03

状态：在 GPU confirmation 产生任何结果前冻结

关联预注册：`docs/arpl/mv_rpformer_preregistration_2026-09-03.md`

## 1. 变更目的

原预注册将正式运行环境固定为 Merlin、每个任务 4 个 PyTorch intra-op 线程、1 个 inter-op 线程、最多 2 个任务并行。由于 96 个 confirmation 任务在 8 CPU Merlin 上预计需要约 9.7 小时，用户明确授权增加一套独立的 4×RTX 4090 GPU confirmation，并固定为每张卡 4 个独立任务、全机 16 个任务并行。

本补充只改变执行设备与调度并发，不改变研究问题、数据、模型、损失、优化器、epoch、随机种子、阈值、指标或预注册判断规则。

## 2. 冻结 GPU 执行条件

- 代码基线：GPU 兼容提交 `7c90e20adf4a6f19f8f073d87ab12a9252cb28cd`。该提交仅将无参数的 `AdaptiveMaxPool1d(1)` 替换为前向和并列最大值梯度规则均相同的 `torch.max(..., dim=-1, keepdim=True).values`，并增加 CPU 精确等价与 CUDA 确定性测试；模型参数、输出形状和配置不变。
- 配置：`configs/experiments/arpl/mv_rpformer_surrogate_v1.yaml`，配置 SHA-256 固定为 `66fe6c9fa556f2fcba1a5325163d28268570c243018e7a41be99e051a7c7ec23`。
- 数据：继续使用原预注册的处理后 HRRP bundle 及其中冻结的三个哈希。
- development：由于任务源码哈希发生变化，不复用 Merlin 的 CPU development 授权；在同一 GPU 环境从头完整运行 S0–S2 × seed 20260830 × M0–M7 共 24 项，聚合与审计通过后才允许启动 confirmation。
- 设备：同一容器内 4 张 NVIDIA GeForce RTX 4090；每个进程只暴露一张物理 GPU，程序内统一记录为 `cuda`。
- 并发：每张 GPU 固定 4 个独立任务，全机固定 16 个任务并行；不使用 DDP，不把一个任务拆到多张卡。
- CPU 线程：每个任务仍固定 4 个 intra-op 线程和 1 个 inter-op 线程。
- CUDA 确定性：统一设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，继续启用 `torch.use_deterministic_algorithms(True)`。
- 数值方式：不启用 AMP、TF32、`torch.compile`，不改变 batch size 或其他训练配置。
- 运行环境：96 个任务必须使用完全相同的 Python、PyTorch、NumPy、CUDA、GPU 型号和环境变量；最终 phase audit 必须通过运行环境一致性检查。

## 3. 结果隔离与选择边界

- 首次逐卡 CUDA 预检在正式输出目录创建前触发了 PyTorch 对 `AdaptiveMaxPool1d(1)` 确定性反向传播的阻断，没有产生正式 GPU 结果。上述等价实现修复及重新运行 development 的决定只由该运行时错误触发，不依据性能指标。
- GPU confirmation 使用独立输出与日志目录，从第 1 epoch 完整运行 C0–C3 × 3 seeds × M0–M7，共 96 项。
- GPU 的完整且审计通过的 96 项被指定为本补充后的正式 confirmation；不得根据 GPU 与 CPU 的性能择优选择结果。
- Merlin CPU confirmation 继续运行，仅作为独立复核与技术故障保险；不得将两台机器的单元拼接为同一 phase。只有 GPU 发生与性能无关且无法在冻结条件下恢复的技术或完整性失败时，才可由完整 CPU confirmation 替代，并必须记录失败原因。
- GPU 启动、停止、并发数和是否采用其完整结果不得依据任何中间 confirmation 性能指标决定。
- 运行状态检查只读取完成数、进程、资源、错误和完整性状态，不用中间 Accuracy、AUROC、OSCR 或其他指标调整实验。
- GPU 版只有在 96 项全部完成、聚合成功、phase audit 通过并生成最终判定后，才能作为一套完整 confirmation 结果报告。
- 若 GPU 版失败，保留失败证据并继续等待 Merlin；不得通过改变模型或训练参数补救。

## 4. 验收与停止边界

- 启动前检查 4 张 GPU 可见、型号一致、无占用，并验证 CUDA PyTorch、配置、源码、数据和 development 哈希。
- 先运行不进入正式结果目录的 CUDA 最小环境验证；它只用于确认代码可运行和数值有限，不用于改变 16 并发决定。
- 每个正式任务使用独立目录、外部锁、原子 checkpoint 和 `--resume`；失败后只能按同一冻结环境恢复。
- 96 项完成后依次执行 aggregate、audit 和 finalize，保存环境、产物哈希及最终决策。
- 完成本轮 GPU confirmation 后停止，不运行最终 7-known/3-unknown 或偶数角 test，不自动进入其他实验。
