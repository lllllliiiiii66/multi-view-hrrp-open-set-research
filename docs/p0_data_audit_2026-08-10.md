# P0 原始 HRRP 数据审计与 manifest 结果

> 审计日期：2026-08-10
> 阶段：P0 数据协议
> 原始数据目录名：`pub-最终要用的10个目标/`（绝对路径仅保存在本地 resolved config）
> 主配置：`configs/data/hrrp_10class_theta83_hh_v1.yaml`

## 结论

原始目录包含 10 个 MATLAB v5 `.mat` 文件，一类一文件，总大小约 1.7 GB。按已确认的 `theta=83°`、`TrcsHH` 和整数方位 `0°–359°` 可为每类唯一选出 360 条底层 HRRP；原始数据的角度覆盖、连续块切分、类别隔离和 split 泄漏检查均通过。

高斯噪声补齐的数值与复现检查已通过。原始 HRRP 长度可以完全区分 known/unknown 角色，用户于 2026-08-10 明确要求暂不处理，作为第一轮已接受风险记录。已生成全量 `3600×601` 派生 HRRP 和逐样本索引；本轮仍未进行归一化、正式模型训练或模型选择。

## 原始结构与粒度

- 每个文件只有一个顶层变量 `merged`，类型为 `1×1` MATLAB struct。
- 每类原始记录数为 7,202：`3,601` 个方位角 × `2` 个仰角。
- `phi` 覆盖 `0°–360°`，步长 `0.1°`；`theta` 为 `83°` 和 `85°`。
- 每个仰角均可唯一选出 `0°–359°` 的 360 个整数方位；`360°` 不进入协议。
- `TrcsHH` 与 `TrcsHV` 有效；`TrcsVV` 与 `TrcsVH` 全为 NaN。
- 当前协议选用 `theta=83°`、`TrcsHH`。

## 类别、角色与维度

未知类采用 `sha256_rank_v1`、seed `20260810` 对 NFC 规范化后的类名做稳定哈希排序并取前三名。该抽取不依赖文件枚举顺序或 Python 随机数实现。

| 类别 | 角色 | HRRP 长度 | RangeX | 源文件 SHA-256 |
|---|---|---:|---:|---|
| CVN77 | known | 601 | -180–180 | `0845ba44e4cd81ec362d25d401564f8165c8d445a13fd9387b0a6887dc3c2145` |
| DDG-1000 | known | 351 | -105–105 | `5dab2666d25c1ef0266ca96d2123ebbb8e82b8410ed2ade1539af9297196fd77` |
| DDG-112 | known | 351 | -105–105 | `d4f55f0c0c2275dc8863684f4e0df724bcc834d0f32491e9cca4062ae0113dc7` |
| LRYYC | unknown | 121 | -36–36 | `a52c174a0e2c11ad609641a825dd41ed7781cef99aac4634f560181e17dc8d79` |
| 汽车运输船9000车级 | unknown | 401 | -120–120 | `5df9f973bcace79e27cc07de8d1a127e35b0a0d01c271cd79cd06c5f5200b217` |
| 油气轮MARVEL CRANE | known | 601 | -180–180 | `6552b44192f681ba1e4797b03cddef17e6df5b1bfd4a23dd15e377aedc36751a` |
| 海洋调查船向阳红10号 | unknown | 201 | -60–60 | `f8d33074113720a987139d0039b9a2ae55851eed1df35e65c34e7c3ef782ba2f` |
| 爱达魔都号 | known | 601 | -180–180 | `ee52823984c6251906758a735961aa8f40ff040e6495f6152f3021ba99ff35b2` |
| 迷你好望角型散货船 | known | 501 | -150–150 | `f679765809ca0c55bf2e85b2ee8928b361e449782e355724ea86de351845dd8e` |
| 集装箱船达飞罗尔多夫级 | known | 601 | -180–180 | `cca8e8c6f905a4853c11944a5749b80e1ca93ac9a93bed568656c9ee84844e63` |

所有 `RangeX` 均以 0 为中心，步长约 0.6；同一类别的 360 条选中记录使用完全相同的 `RangeX`。

## Manifest 与验证结果

本地生成目录：`artifacts/manifests/hrrp_10class_theta83_hh_v1/`（受 `.gitignore` 保护）。

- 行数：3,600 条底层 HRRP。
- Manifest SHA-256：`2ce908e24b68dcd7524e7f1544f97f3761d4955f2d64d8cfe05aafc090d77724`。
- 每类：train 216、validation 72、test 72。
- 每类每域：train 36、validation 12、test 12。
- 已通过：角度完整覆盖、边界唯一、连续块规则、split 互斥、源文件/行身份唯一、profile 哈希跨 split 无重复、7/3 类别数量和未知类隔离。
- 未知类的 train/validation 角度仍在 manifest 中用于完整审计，但统一标记为 `held_out_unknown`，训练与验证资格均为 0；只有未知 test 具有评估资格。
- 第二次独立生成的 `samples.csv` 与正式 manifest 逐字节相同。

## 关于 dB 与噪声补齐

选中 `TrcsHH` 的全体数值范围为约 `-130.16` 到 `+72.36`，其中 99.39% 为负值且没有恰好为 0 的值。这足以确认其为 dB 表示，并证明直接用数值 0 补齐会引入有物理含义的强人工边界。

数据处理程序给出了确定的生成链：

- `rcs_data = Ep'`（HH 通道）；
- 乘 Hamming 窗和幅度系数；
- `range_rcs = ifft(st)`；
- `range_rcs = 4*pi*abs(range_rcs).^2`；
- `range_rcs_dB = fftshift(10*log10(range_rcs))`。

用原始 `Ep` 重算十类 `theta=83°`、整数 `phi=0°–359°` 的全部 `TrcsHH`，采用数据实际使用的幅度系数 `1.8` 时，全局最大绝对误差小于 `6.0e-10 dB`。因此已确认：

- 存储值是功率 dB：`x_db = 10*log10(power)`；
- 还原线性功率应使用 `power = 10**(x_db/10)`；
- `0 dB` 对应相对功率 1，不能作为无信息填充值；
- 程序中的 `1.852` 与数据实际 `1.8` 相差恒定 `0.2473695 dB`，不影响上述 dB 定义。

补齐方案冻结在 `configs/data/hrrp_padding_complex_gaussian_v1.yaml`：

1. 共同网格固定为 `RangeX=-180…180`、步长 0.6、长度 601；原始 profile 按 0 对称居中；
2. 用两个独立零均值高斯分量构造复高斯幅度，取 `|z|^2` 得到严格非负的线性噪声功率；
3. 所有样本使用同一个固定期望功率 `1e-14`（`-140 dB`），不再按原始 profile 能量缩放；
4. `-140 dB` 仅依据 779,112 个已知 train bin 冻结：其最小值为 `-130.1627 dB`，噪声均值约低 10 dB；未知类、validation 和 test 未用于参数选择；
5. 逐样本 seed 由 `algorithm_version + base_seed + sample_id` 经 SHA-256 派生；全局随机状态不参与；
6. 噪声功率用 `10*log10` 转回 dB 后只写入左右补齐区，原始 bin 逐值保持不变；
7. 初始尝试的“补齐总功率/整条信号总功率 = 1e-6”已废弃：强峰会使补齐点最高达到约 `+0.19 dB`，并造成类别相关的噪声标度。

真实数据补齐机械审计结果：

- 3,600 条原始 HRRP 全部输出为 601 点；其中 2,160 条需要补齐，1,440 条原本已为 601 点；
- 共生成 604,800 个补齐点；实际平均功率 `-140.0136 dB`，中位数 `-141.6013 dB`，范围 `-199.1497–-128.3176 dB`；
- 3,600 个派生 seed 全部唯一；所有结果有限；目标网格对齐通过；所有原始 bin 逐值不变；
- 补齐配置绑定原始 manifest SHA-256：`2ce908e24b68dcd7524e7f1544f97f3761d4955f2d64d8cfe05aafc090d77724`。

### 已接受风险：原始长度泄露开集角色

- 已知类长度集合：`{351, 501, 601}`；
- 未知类长度集合：`{121, 201, 401}`；
- 两集合交集为空；
- 补齐区与原始区连接处绝对差的中位数为 `76.78 dB`，最小值仍为 `21.63 dB`，因此边界具有很高的可检测风险；
- 一旦模型从边界推断出原始长度，就能直接推断样本是 known 还是 unknown，这属于数据构成捷径，而不是有效的开集识别能力。

该风险使用 `profile_length_role_shortcut_v1` 作为稳定标识，在补齐配置、审计报告和每一行派生 manifest 中均记录为 `accepted_for_first_round` / `document_and_continue`。审计状态为 `passed_with_accepted_profile_length_role_risk`，不把该风险伪装成检查通过，也不自行增加裁剪、重采样或复杂背景生成协议。

## 派生数据产物

本地生成目录：`data/processed/hrrp_10class_theta83_hh_padding_v1/`（受 `.gitignore` 保护）。

- `profiles.npy`：`3600×601`、`float64`，约 17 MB；SHA-256 `2dd92282c125f0f677cf1f2dfce828781c8ba4385cf9ae552c4a2c56033c3f5b`；
- `samples.csv`：3,600 行，保留类别、角色、split、角度域、真实角度、源行、原始/派生 profile 哈希、左右补齐数、派生 seed 和已接受风险；SHA-256 `748b9f30629c3b3cbe66c6a1dac30863fdab2d81a214e46d8bc3ef7c6022a08a`；
- `resolved_preprocessing.yaml`：完整补齐配置和输入/输出哈希；
- `materialization_report.json`：写出、回读、哈希、风险和审计结果；
- Bundle SHA-256：`79176091b5b745e5df5957f1bb4ade7304c1403a4a32e64622e012db67d5a0c5`。

对同一输出目录连续独立生成两次，`profiles.npy`、`samples.csv` 和 bundle 的 SHA-256 均完全一致。

## 实际测试

- Python 源码编译检查：通过。
- pytest：40 passed。
- 合成 MATLAB fixture 覆盖：协议边界、216/72/72、每域 36/12/12、固定 seed、输入顺序无关、未知类隔离、缺角阻断、跨 split 源行泄漏阻断、manifest 字节级复现、dB/线性功率往返、601 点居中补齐、原 bin 保持、逐样本噪声复现、噪声不编码源 profile 能量、未接受风险时阻断、显式接受风险后继续，以及派生 bundle 重复生成。
- 真实 10 类端到端构建：通过。
- 真实 manifest 独立重复构建与 `cmp`：一致。
- 真实 10 类高斯补齐机械审计：通过；长度角色捷径保留为 `accepted_risk`，命令正常退出 0。
- 真实派生数据写出与全量回读：通过；每行派生 profile 哈希、跨 split 哈希互斥、3,600 个派生 seed 唯一和风险字段追溯均通过。
