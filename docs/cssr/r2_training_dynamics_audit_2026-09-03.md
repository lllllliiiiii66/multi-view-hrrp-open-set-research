# R2 多尺度 CE 训练后期只读审计

> 日期：2026-09-03
>
> 服务阶段：P3 快速迭代实验 `fg_mv_cssr_frozen_r2_v1`
>
> 审计对象：`R2_MS_MEAN_CE`，`angle_fold=0`，`seed=20260830`
>
> 性质：读取上一轮正式产物；没有重新训练、修改 checkpoint 或查看最终 unknown/even-angle test

## 1. 结论

七个目标 identity pair 的 epoch-100 R2 checkpoint、resolved config、pair manifest 和 100 行逐 epoch 训练日志均在原 GPU 正式目录中完整存在，文件哈希与上一轮封口的全树哈希清单一致；七个 checkpoint 均可在正式运行代码 `62e318de82b4221b599e06b1166483673e9c1cd3` 上 `strict=True` 加载。

现有日志可以确认：

- train Accuracy 首次达到 95% 发生在 epoch 7–10，首次达到 99% 发生在 epoch 10–19；
- known calibration Accuracy 首次达到 98% 发生在 epoch 9–22；
- train Accuracy 已经接近饱和后，train loss 仍继续明显下降；
- epoch 100 的 known calibration Accuracy 为 98.08%–99.44%，相对各自全过程最高值仅低 0.24–0.84 个百分点，没有出现一致的末期 Accuracy 崩落。

但日志没有记录 known calibration loss/NLL、平均最大 logit、feature norm 或 CE head weight norm。因此，本轮只能确认“闭集准确率较早饱和后，训练损失仍继续下降”，**不能确认 epoch 100 已进入闭集过度自信阶段，也不能判断后期 logit 或特征尺度是否持续膨胀**。

## 2. 日志能够回答什么

每个 `training_log.jsonl` 均有 100 行，epoch 连续为 1–100。R2 每行实际包含：

- `train_loss`、`train_classification_loss`、`train_margin_loss`；
- `train_accuracy`；
- `known_calibration_accuracy_diagnostic`；
- `known_calibration_macro_f1_diagnostic`；
- learning rate、epoch 耗时、训练样本顺序哈希；
- `checkpoint_selected_for_open_set_performance=false` 和 `pseudo_unknown_generated=false`。

R2 为 CE head，`train_loss` 与 `train_classification_loss` 相同，`train_margin_loss=0`。现有 `_head_diagnostics` 只为 ARPL head 记录 radius/reciprocal-point norm，未为 CE head 记录权重范数。

以下指标没有逐 epoch 记录：

| 缺失指标 | 能否由现有日志补出 | 对结论的影响 |
|---|---|---|
| known calibration loss / NLL | 不能 | 无法判断 calibration 正确率稳定时概率校准是否恶化 |
| 平均最大 logit | 不能 | 无法判断模型置信尺度是否持续增大 |
| feature norm | 不能 | 无法判断表示尺度是否持续增大 |
| CE head weight norm | 不能 | 无法判断分类头范数是否持续增大 |

上一轮 `features_logits_scores.npz` 只保存 epoch 100 的池化后单视角 128 维特征、融合特征和 logits；它不能恢复上述逐 epoch 轨迹，也不包含本轮 CSSR 所需的池化前 `[B,128,L]` 语义特征图。

## 3. Accuracy 轨迹

“Cal 首达 98%”仅表示第一次越过 98% 阈值，不等同于正式定义的“进入平台”。上一轮没有预注册平台判据，而且部分曲线越过阈值后仍有短暂回落，因此本审计不事后指定唯一平台 epoch。

| Pair | 用途 | Train 首达 95% | Train 首达 99% | Cal 首达 98% | Cal@30 | Cal@50 | Cal@70 | Cal@100 | 全程最高 Cal（epoch） | 100−最高 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N1 | pilot | 10 | 15 | 18 | 97.24% | 97.36% | 98.64% | 98.08% | 98.92%（91） | -0.84 pp |
| N4 | pilot | 9 | 12 | 12 | 99.00% | 99.16% | 99.36% | 99.32% | 99.56%（66） | -0.24 pp |
| N2 | pilot | 10 | 19 | 22 | 97.76% | 98.68% | 98.60% | 98.56% | 99.16%（51） | -0.60 pp |
| N0 | conditional confirmation | 7 | 10 | 9 | 98.72% | 96.88% | 99.12% | 98.80% | 99.60%（53） | -0.80 pp |
| N3 | conditional confirmation | 9 | 12 | 12 | 98.80% | 99.24% | 98.80% | 99.04% | 99.32%（53） | -0.28 pp |
| N5 | conditional confirmation | 8 | 14 | 14 | 99.52% | 97.88% | 99.12% | 99.16% | 99.68%（37） | -0.52 pp |
| N6 | conditional confirmation | 8 | 12 | 12 | 99.56% | 99.56% | 99.56% | 99.44% | 100.00%（61） | -0.56 pp |

从 epoch 70 到 epoch 100，七个 pair 的 calibration Accuracy 变化分别为 N1 `-0.56 pp`、N4 `-0.04 pp`、N2 `-0.04 pp`、N0 `-0.32 pp`、N3 `+0.24 pp`、N5 `+0.04 pp`、N6 `-0.12 pp`。这些变化较小且方向不统一，不能据此声称存在一致的末期 calibration Accuracy 退化。

## 4. Train loss 轨迹

| Pair | Loss@30 | Loss@50 | Loss@70 | Loss@100 | 首达 99% 时 loss → epoch 100 loss |
|---|---:|---:|---:|---:|---:|
| N1 | 0.015270 | 0.005668 | 0.002777 | 0.002756 | 0.071675 → 0.002756 |
| N4 | 0.007115 | 0.003214 | 0.001518 | 0.001524 | 0.079217 → 0.001524 |
| N2 | 0.020814 | 0.009732 | 0.006661 | 0.004981 | 0.051867 → 0.004981 |
| N0 | 0.003461 | 0.028261 | 0.001383 | 0.001195 | 0.060110 → 0.001195 |
| N3 | 0.009971 | 0.003819 | 0.002539 | 0.002001 | 0.079815 → 0.002001 |
| N5 | 0.014125 | 0.006859 | 0.002113 | 0.002127 | 0.060506 → 0.002127 |
| N6 | 0.009052 | 0.005613 | 0.001816 | 0.001737 | 0.062887 → 0.001737 |

七个 pair 在第一次达到 99% train Accuracy 后，epoch-100 train loss 都进一步下降。N0/N5 的单点波动说明轨迹不是严格单调，但不改变“准确率饱和后 loss 仍下降”的事实。仅凭交叉熵继续下降不能区分更好拟合、margin 增大、logit 尺度变化或真正的概率过度自信。

## 5. Checkpoint、manifest 与数据可用性

正式 GPU 产物路径模板：

```text
/root/hrrp-runs/ms_mean_head_factorial_surrogate_v1/confirmation_gpu_62e318d/
  {pair_id}/fold_0/seed_20260830/R2_MS_MEAN_CE/
```

每个目录均包含 `_SUCCESS.json`、`checkpoint.pt`、`resolved_config.yaml`、`pair_manifest.csv`、`training_log.jsonl`、normalization、最终特征/logits、预测和审计文件。每个 checkpoint 大小为 586,467 bytes，metadata 均为：

```text
checkpoint_epoch = 100
formal_checkpoint = true
checkpoint_selection = fixed_final_epoch
architecture = ms_mean_head_factorial_v1
head_type = ce
known_class_count = 5
```

所有 checkpoint 内部记录的冻结配置 SHA-256 均为：

`c11daa6e2e5a7d7b72bc36840e60fc871f332c4fc85652636c729aa2eba14c71`

下表哈希均已与上一轮根目录 `artifact_hashes.json` 逐项核对；列顺序为 checkpoint、resolved config 文件、pair manifest、training log。

| Pair | `checkpoint.pt` | `resolved_config.yaml` | `pair_manifest.csv` | `training_log.jsonl` |
|---|---|---|---|---|
| N1 | `a4f6fa3235fbb5cf74b712588a0318f614a05287adec4ee881820424cddbcbaa` | `186be7f2d6d58a19afe94241311f0a5adccf7f0379d47d8fb561d52202a9bd1a` | `0b8a97dcfd744896bbae912c1363379201ced18a55107f80b2d2f3256fb5c5bc` | `eb29d6142cc91c627c266e87e30bf910a792579e49c061620a94d4e004505482` |
| N4 | `169387ad7a87463110ac7a2cd45afd7dac49428538c93c84975162e425d94ff5` | `1519715186eb358d8468106e084930904e109117fcef6f815f6fcb31f99402e1` | `8b0202d1e08ae83eec4bf07fc1dbb6a3f39fef2378ac15e57635709d8872b41a` | `924876758e1c103b6a6595156d9e8a86e048a354e0a7dc6fbda30c9b9036a8a9` |
| N2 | `14e2ac7b686c901112f969fe0bd7f53c29646e7c015bae794d30c39051f9c0b9` | `ebe786dd6fe3e5138bd02bb79de801920cd3e52f71f985b6e669c9c9d54af81d` | `1a7dc0031cf5b32a41131289fb4117a144463c025e93bc7a487e56a3c8c8bd2d` | `120d9f1e1011585d81613119f7b38c7eebc05241f8ff6d53661b7e885484e164` |
| N0 | `142a85b3a090213684126cf695b08fec259724a0bd8399dc1adb40b114aab192` | `5dd55a4cc4cc96a06354cfbcdf1068da7edc90874c9a48572d6ef4757813a77c` | `37dac18016223e08451c6551e279a6136ed494cb9c86edb5f0a938d71a2b115d` | `e14ff5cbacd9e5a07871f81694e56e9112c4a3eae579975d03d5762ca1556879` |
| N3 | `6427a09f3e4a5e67ff652fea6e44c8364b62381acc8338099dccc818ac284bc9` | `9ce91f2254bd2bfc374069a562dc67d139d2cbd6d5e704bcea47def6a51e4f71` | `53fead93617851f8646dc7c76ff3773b6c55a720d3be17feda462535994e7d27` | `16a896ffeb0a29be0ad03b40a6ec98ea625a69100a157e66a687dc6db4accf11` |
| N5 | `74cde2c6b30f1fa96219fe20777dfc632575c8c3c0281706ca016ef2497642df` | `7c66a9df13d11c1dc86b90e53889ae271c569b6f550467f0d54eaef3cba1a241` | `a706c63e47f8522510c2926e70a8072ca8ca183c5ef74957b8451d28d2c47c80` | `b095688b6d0886ae39c17f20892b260777ee70d81617bcea403fe1856fc346a6` |
| N6 | `178dbaa9e461d28825124b688752ed5c1005a8f0265963ef57e5c27a0a65e86e` | `bd0581018961530b1451abd749d374e450a72c0dce311b383a0798d9e9c7debe` | `46b454fc313573121fcf6ad214b91f9e21a2cb996a38d3beaf9c83d8321ce140` | `dec0f7f023fd797b16d07d04d2fbd3a5f635b8c481362a24b166d384bb1db193` |

根哈希清单的本机 summary-only 镜像为：

`/Users/bytedance/Desktop/科研空间/artifacts/results/ms_mean_head_factorial_surrogate_v1/confirmation_gpu_62e318d/artifact_hashes.json`

其 SHA-256 为 `edcf281df07443724d0ade1a0b2d8b20305f85b83099fb74e1c6417ee5d5477c`。

数据 bundle 在本机和 GPU 容器均存在且内容哈希一致：

| 位置 | 路径 |
|---|---|
| 本机 | `/Users/bytedance/Desktop/科研空间/data/processed/hrrp_10class_theta83_hh_padding_v1/` |
| GPU | `/root/hrrp-data/hrrp_10class_theta83_hh_padding_v1/` |

| 对象 | SHA-256 |
|---|---|
| `profiles.npy` | `2dd92282c125f0f677cf1f2dfce828781c8ba4385cf9ae552c4a2c56033c3f5b` |
| `samples.csv` | `748b9f30629c3b3cbe66c6a1dac30863fdab2d81a214e46d8bc3ef7c6022a08a` |
| 声明的 bundle | `79176091b5b745e5df5957f1bb4ade7304c1403a4a32e64622e012db67d5a0c5` |

每个 pair manifest 有 6,000 行：5 个 train-known 类、5 个 known-calibration 类和 2 个 surrogate 类各 500 pairs。每个 train-known 类覆盖 144 个唯一底层样本，总计 720 个；每个 known-calibration 类覆盖 36 个，总计 180 个；每个 surrogate 类覆盖 36 个，总计 72 个。七个 manifest 的全部 84,000 个视角引用均已按 `sample_id`、`processed_row_index`、class 和 angle 对照 `samples.csv`，未发现不一致。

因此，现有 manifest 与 bundle 足以按唯一底层 `sample_id` 提取本轮 CSSR train-known 单视角样本，并执行 known-calibration 的 leave-one-base-sample-out 参考分布；不需要按 pair multiplicity 重复样本。

## 6. 证据边界和后续使用

已经确认：

- N1/N4/N2 三个 pilot pair 以及 N0/N3/N5/N6 四个条件确认 pair 的冻结 R2 产物都完整存在；
- 原 GPU 容器中的七个 checkpoint 可直接用于冻结推理；
- pair manifest 含单视角底层 `sample_id`、row index、角度和 frame 信息；
- 当前产物足以完成本轮所需的唯一底层样本提取。

尚未验证：

- 新增 `forward_feature_map` 后，这些旧 checkpoint 在新代码上的 strict-load 和旧 `forward` 数值一致性；
- epoch 100 是否存在 logit、feature norm 或概率校准意义上的过度自信；
- 池化前语义特征图及其哈希，因为上一轮没有保存该张量。

运行位置边界：本机结果目录及 Pro zip/tar 都是 summary-only，不含上述 checkpoint、日志和 manifest；Merlin 上也未发现本轮 `R2_MS_MEAN_CE` 副本。若本轮继续在原 GPU 容器执行，不存在产物缺失；若改为本机或 Merlin，必须先复制目标 pair 的完整正式目录并重新核对哈希。

本审计不支持重新训练 R2、改变 epoch、选择较早 checkpoint，或依据这些日志调整本轮 CSSR 超参数。正式模型仍固定为 epoch 100。
