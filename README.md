# 多视角 HRRP 开集识别研究工作区

本仓库保存多视角 HRRP 开集识别研究的可复现上下文、定版协议、文献证据、实验代码、版本化配置和测试。原始数据、checkpoint 与生成的实验结果不进入 Git。

## 新任务入口

1. 完整阅读 [`AGENTS.md`](./AGENTS.md)，遵守阶段顺序、数据隔离和验证要求。
2. 完整阅读 [`RESEARCH_CONTEXT.md`](./RESEARCH_CONTEXT.md)，恢复研究问题、冻结协议与证据边界。
3. 检查已有配置、manifest、checkpoint 和测试结果，避免重复工作。

## 当前阶段

当前主线已经收束为：以组内 Adaptive Manifold Discriminative Regression
（AMDR）闭集多视角融合框架为基础，研究目标身份开集识别。

当前处于 P0：数据协议和两视角构造规则已冻结，并已完成 Python AMDR 的 fold 0 诊断性 smoke。正式预处理和 AMDR `paper_aligned` 数值边界仍待确认。
P0 通过后，第一项方法实验仅为固定 AMDR 表示下的 Thresholded KNN。
旧 B0–B6、OpenMax、CBD、CBD+view 和相关诊断只作历史参考，不是当前默认候选。

当前路线与 Stage 1 边界见：

- [`RESEARCH_CONTEXT.md`](./RESEARCH_CONTEXT.md)；
- [AMDR 论文—代码—数据审查](./docs/sunchenglong_amdr_paper_code_audit_2026-08-29.md)；
- [AMDR 开集研究路线与 Stage 1 协议提案](./docs/amdr_open_set/research_route_and_stage1_protocol_proposal_2026-08-30.md)。
- [P0 Python AMDR 诊断性 smoke 记录](./docs/amdr_open_set/p0_python_amdr_smoke_2026-08-30.md)。
- [P0 AMDR 终止、checkpoint 与 Merlin 执行建议](./docs/amdr_open_set/p0_amdr_checkpoint_and_merlin_execution_2026-08-30.md)。

## Git 边界

版本控制包括：

- 项目规则和研究上下文；
- 文献调研与定版交付物；
- 源码、版本化配置、测试和小型合成 fixture；
- 不包含本地绝对路径或敏感信息的说明文档与 schema。

本地保留、不上传 GitHub：

- 原始和派生 HRRP 数据；
- 生成的完整 manifest、逐样本预测和日志；
- checkpoint、缓存和批量实验结果；
- 密钥、令牌与本地环境文件。

如将来需要跨设备同步大文件，应另行评估受控对象存储或 DVC，不把数据直接加入本仓库。

## 历史数据工具（当前非主线）

以下工具、配置与结果属于旧 B0–B6 路线，保留用于审计和必要时的公平对照；
未经当前 AMDR 协议重新确认，不直接用于新主线实验或数值比较。

版本化主数据配置位于
[`configs/data/hrrp_10class_theta83_hh_v1.yaml`](./configs/data/hrrp_10class_theta83_hh_v1.yaml)。
版本化噪声补齐配置位于
[`configs/data/hrrp_padding_complex_gaussian_v1.yaml`](./configs/data/hrrp_padding_complex_gaussian_v1.yaml)。
原始数据根目录不写入 Git；运行时通过 `--raw-root` 或 `HRRP_RAW_ROOT` 提供。

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/hrrp-p0 build-manifest \
  --config configs/data/hrrp_10class_theta83_hh_v1.yaml \
  --raw-root /path/to/raw-data \
  --output artifacts/manifests/hrrp_10class_theta83_hh_v1
.venv/bin/hrrp-p0 audit-padding \
  --config configs/data/hrrp_10class_theta83_hh_v1.yaml \
  --padding-config configs/data/hrrp_padding_complex_gaussian_v1.yaml \
  --raw-root /path/to/raw-data \
  --output artifacts/diagnostic/padding/hrrp_padding_complex_gaussian_v1_audit.json
.venv/bin/hrrp-p0 materialize-padding \
  --config configs/data/hrrp_10class_theta83_hh_v1.yaml \
  --padding-config configs/data/hrrp_padding_complex_gaussian_v1.yaml \
  --raw-root /path/to/raw-data \
  --output data/processed/hrrp_10class_theta83_hh_padding_v1
.venv/bin/hrrp-p0 run-b0-smoke \
  --config configs/experiments/p0/b0_smoke_v1.yaml \
  --bundle-root data/processed/hrrp_10class_theta83_hh_padding_v1 \
  --output artifacts/results/diagnostic/b0_smoke_v1 \
  --device cpu
.venv/bin/hrrp-p0 run-b0-main \
  --config configs/experiments/p0/b0_main_v1_seed20260810.yaml \
  --bundle-root data/processed/hrrp_10class_theta83_hh_padding_v1 \
  --output artifacts/results/main_v3/b0_main_v1/seed_20260810 \
  --device cuda
.venv/bin/pytest
```

生成目录包含逐样本 `samples.csv`、manifest SHA-256、解析后的完整配置和数据审计报告。
该目录受 `.gitignore` 保护，不会上传原始路径或逐样本数据索引。

`audit-padding` 只重建并审计补齐结果；`materialize-padding` 额外写出
`3600×601` 的 `profiles.npy`、逐样本可追溯 `samples.csv`、配置快照和哈希。
当前数据的长度角色捷径已由用户明确接受为第一轮限制；审计仍保留
`accepted_risk` 标记，不会把它记录为“未检测到”。

## 历史 P0/B0 诊断链路

版本化 smoke 配置位于
[`configs/experiments/p0/b0_smoke_v1.yaml`](./configs/experiments/p0/b0_smoke_v1.yaml)。
它只训练两轮，用于验证统一 1D-CNN、已知类隔离、V=3 跨域集合、B0 固定种子
抽视、MSP/Energy 方向、已知 validation 阈值和完整产物保存；结果只能存入
`diagnostic`，不得进入正式主表。

Merlin CPU 环境已用 Python 3.12.13、PyTorch 2.13.0+cpu 完成相同测试和 smoke run。
正式配置 `b0_main_v1_seed20260810.yaml` 使用 `neural_budget_v1`：最多 100 epochs、
已知 validation accuracy 早停 patience 15、无数据增强，并对 B0 固定执行 30 次单视抽样。
5 个初始化种子已经注册；当前先运行第 1 个种子。A100 worker 与 CPU 环境分离，
正式结果写入 `main_v3`，不会覆盖诊断产物。

五种子正式汇总位于（生成结果，不进入 Git）
`artifacts/results/main_v3/b0_main_v1/multiseed_summary.json`；运行与审计记录见
[`docs/p0_b0_main_v1_2026-08-10.md`](./docs/p0_b0_main_v1_2026-08-10.md)。

## 历史 P1–P3 产物入口

- 版本化配置：`configs/experiments/p1/`、`p2/`、`p3/`。
- 主结果：`artifacts/results/main_v3/`。
- V=5 辅助结果：`artifacts/results/paper_aligned_v5/`。
- 最终反算与哈希审计：
  `artifacts/results/audits/p0_p3_result_audit_v2_2026-08-11.json`。

上述 `artifacts` 均是本地/开发机生成产物，受 `.gitignore` 保护；代码、配置、
测试与本报告可提交 GitHub 用于跨对话恢复上下文。
