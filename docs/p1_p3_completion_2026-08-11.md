# P1–P3 完成与结果审计报告

日期：2026-08-11

## 1. 结论与范围

P1、P2 和 P3 已按 `AGENTS.md` 与 `RESEARCH_CONTEXT.md` 的冻结定义完成，实现范围仅包含 B1–B6，未引入创新方法、伪未知训练、额外重构方法或非冻结数据协议。

- P1 回答 R1/R2：完成 B1 后验平均、B2 Deep Sets 和 B3 Set Transformer。
- P2 服务 R3：完成 B4 逐视 OpenMax 软后验平均与 B5 集合激活 OpenMax。
- P3 服务 R4：完成 B6 主 `V=3` 核心算法复现/适配，以及 B4/B5/B6 独立的 `paper_aligned_v5` 辅助对齐。
- 全部神经方法用 5 个注册初始化种子；B6 为确定性单次运行，不对称地拥有五种子置信区间。

## 2. 数据与公平性不变量

- 主协议仍为 6 个等宽 `60°` 域，每域固定前 `36°` train、中间 `12°` validation、最后 `12°` test。
- 主集合固定 `V=3`，3 条底层 HRRP 来自 3 个不同角度域；输入前随机打乱，不输入角度、域编号或位置信息。
- `V=5` 只用于 B4/B5/B6 辅助对齐，结果保存在 `paper_aligned_v5`，未混入主表。
- 未知类未参与训练、早停、模型选择、OpenMax/GPD 拟合、超参选择或阈值确定。原始 manifest 保留未知类 validation 角度记录，但其 `eligible_for_validation=0`，集合构造器明确排除。
- B0/B1/B4 共用 B0 checkpoint；B2/B5 共用 B2 checkpoint。B4/B5 均不重训练神经模型。
- 主 operating point 仅用已知 validation，按 95% 已知接受率定阈值；所有未知分数均为“越大越未知”。

## 3. 实现与版本化配置

| Baseline | 实现要点 | 主配置 |
|---|---|---|
| B1 | 复用 B0；逐视后验平均，Energy 逐视分数平均 | `configs/experiments/p1/b1_main_v1.yaml` |
| B2 | 共享 1D-CNN + mean pooling + 集合头 | `configs/experiments/p1/b2_main_v1.yaml` |
| B3 | 同构 1D-CNN + 无位置编码 SAB/PMA | `configs/experiments/p1/b3_main_v1.yaml` |
| B4 | 复用 B0；逐视软 `K+1` OpenMax 后验等权平均 | `configs/experiments/p2/b4_main_v2.yaml` |
| B5 | 复用 B2；在集合级激活上独立拟合 OpenMax | `configs/experiments/p2/b5_main_v2.yaml` |
| B6 | 联合动态活动集 group-OMP、联合重构选类、匹配/非匹配双尾 GPD | `configs/experiments/p3/b6_main_v1.yaml` |

OpenMax v2 对 B4/B5 和 `V=3`/`V=5` 使用完全相同的 27 组候选网格：

- tail size：`[10, 20, 40]`；
- alpha rank：`[3, 5, 7]`；
- distance：`[euclidean, cosine, eucos]`；
- 选择准则：已知 validation 的 5 折 95% 接受率误差 + `0.01 ×` IQR 归一化阈值标准差；平局时按候选登记顺序确定。

`b4_main_v1`/`b5_main_v1` 及其 V=5 v1 使用固定原论文风格参数，未满足已冻结的候选网格选择要求。该些产物仅作历史诊断保留，已被 v2 取代，不得进入主比较。

OpenMax 实现依据为 Bendale & Boult 的 [CVPR 2016 论文](https://openaccess.thecvf.com/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html) 和[作者代码](https://github.com/abhijitbendale/OSDN)；B4/B5 是本项目的受控迁移 pipeline，不是原论文数值复现。B6 依据 [JDSR-OSR 原始论文](https://jeit.ac.cn/cn/article/pdf/preview/10.11999/JEIT221284.pdf) 实现核心算法；因原代码与原始数据未公开，只能称“核心算法复现/适配”。

## 4. 主 `V=3` 结果

下表 B0–B3 使用 MSP，B4/B5 使用 OpenMax 未知后验，B6 使用双尾 GPD 分数。B0–B5 为 5 个模型种子均值，B6 为确定性单次运行。

| 方法 | Known Acc. | Known Macro-F1 | AUROC | OSCR | FPR95↓ | K+1 Macro-F1 | Unknown Reject |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 MSP | 0.9162 | 0.9158 | 0.6804 | 0.6632 | 0.6554 | 0.7085 | 0.0681 |
| B1 MSP | 0.9929 | 0.9929 | 0.7433 | 0.7426 | 0.5655 | 0.8239 | 0.3278 |
| B2 MSP | 0.9405 | 0.9405 | 0.6752 | 0.6649 | 0.6262 | 0.7245 | 0.1083 |
| B3 MSP | 0.9381 | 0.9341 | 0.7722 | 0.7535 | 0.6179 | 0.7932 | 0.4111 |
| B4 OpenMax v2 | 0.9929 | 0.9928 | 0.6786 | 0.6737 | 0.8429 | 0.8685 | 0.5056 |
| B5 OpenMax v2 | 0.9167 | 0.9175 | 0.7910 | 0.7402 | 0.6250 | 0.8183 | 0.5333 |
| B6 dual-tail GPD | 0.9286 | 0.9238 | 0.8113 | 0.7875 | 1.0000 | 0.8005 | 0.3333 |

### 按研究问题解读

- R1：B0 到 B1 后，MSP 的 Known Acc. 从 0.9162 升至 0.9929，AUROC 从 0.6804 升至 0.7433，OSCR 从 0.6632 升至 0.7426。这支持“同一单视模型下，3 视简单决策融合对当前数据有效”，不能据此宣称集合学习有效。
- R2：B2 在当前数据上未超过 B1；B3 的 AUROC/OSCR（0.7722/0.7535）高于 B2（0.6752/0.6649），但已知分类相近且种子区间较宽。B2 有 36,391 个参数、约 41.36M FLOPs/V3；B3 有 433,959 个参数、约 42.93M FLOPs/V3。
- R3：需联合观察 B1/B2/B4/B5。OpenMax 加到逐视 B0 pipeline 后（B4）提高当前 operating point 的 K+1 Macro-F1，但 AUROC/OSCR 低于 B1；加到 Deep Sets 集合激活后（B5）的 AUROC 高于 B2，但 Known Acc. 降低。B4↔B5 同时改变训练方式、激活层级与 OpenMax 拟合对象，不解读为只改变 OpenMax 时机的单因素实验。
- R4：主 `V=3` 下，B6 的 AUROC/OSCR 数值高于 B4/B5 均值，但 FPR95=1.0，且 B6 只有一次确定性运行，不能据此宣称稳定超越。主 B6 由已知 validation 选得 `K=1`、`rho=0.5`、非匹配尾权重 `0.0`，因而第一轮结果不能支持“非匹配尾带来改善”。

这些都是当前 7 已知/3 未知划分和当前角度块协议下的第一轮结果；未做额外未知类组合或多数据划分重复，不将差异解读为已证实的因果结论。

## 5. 辅助 `V=5` 结果

| 方法 | Known Acc. | Known Macro-F1 | AUROC | OSCR | FPR95↓ | K+1 Macro-F1 | Unknown Reject |
|---|---:|---:|---:|---:|---:|---:|---:|
| B4 OpenMax v2 | 0.9952 | 0.9952 | 0.5856 | 0.5836 | 0.8167 | 0.8166 | 0.3269 |
| B5 OpenMax v2 | 0.9369 | 0.9374 | 0.8389 | 0.8020 | 0.3726 | 0.8184 | 0.4991 |
| B6 dual-tail GPD | 0.9306 | 0.9262 | 0.6078 | 0.6033 | 1.0000 | 0.7857 | 0.2824 |

B6 V=5 使用论文对齐 `K=2`、`rho=0.7`，非匹配权重根据当前 7/10 类 openness 得到 `0.907485`。主辅助指标仍使用已知 validation 95% 接受率阈值。论文固定 `delta=0.3` 只作诊断：Known Accept=0.5536、Unknown Reject=0.5880、K+1 Macro-F1=0.5874。

## 6. 实际验证

- A100 环境全量测试：`106 passed in 20.48s`。
- 本地静态验证：`compileall` 通过，`git diff --check` 通过。
- 结果树审计：64 个运行目录、126,072 行预测、573 个产物哈希，所有保存指标从逐样本预测反算的最大差为 `0.0`。
- B1/B2/B3/B4/B5 置换不变性审计通过；主 V=3 最大差不超过约 `3.82e-6`。
- B6 主 V=3 置换差约 `2.00e-6`；V=5 第一次以 `1e-5` 容差审计时观察到纯数值浮动 `3.686e-5`，无候选原子选择分歧，单独记录后仅将 V=5 预注册容差调为 `1e-4`，复跑通过；主 V=3 不变。
- B4/B5 v2 每个 seed 均保存 27 候选选择记录、拟合样本 ID、最终参数、checkpoint 引用与哈希、逐样本预测及产物哈希。

最终审计文件：`artifacts/results/audits/p0_p3_result_audit_v2_2026-08-11.json`。

## 7. 项目结构

```text
configs/
  data/                         # P0 数据与补齐协议
  experiments/p0..p3/          # B0–B6 版本化实验配置
src/hrrp_osr/
  data/                         # manifest、processed bundle、V3/V5 集合
  models/                       # 共享 1D-CNN、Deep Sets、Set Transformer
  training/                     # B0/B1、集合模型、OpenMax、B6 runner
  evaluation/                   # 指标、聚合、结果反算审计
  openmax.py                    # OpenMax 拟合与 K+1 后验
  jdsr.py                       # 联合稀疏重构与双尾 GPD
tests/                          # 数据、模型、隔离、置换、指标与数值 fixture
docs/                           # 阶段报告与数值审计
artifacts/results/
  main_v3/                      # 主结果，不进 Git
  paper_aligned_v5/             # V5 辅助结果，不进 Git
  audits/                       # 结果树审计，不进 Git
```

## 8. 仍需确认的事项与风险

1. 长度不一致的 HRRP 已按用户批准的首轮方案补到 601，但补齐位置与类别角色可能形成长度捷径；风险 ID 为 `profile_length_role_shortcut_v1`，状态为 `accepted_for_first_round`。
2. B6 主结果的非匹配尾权重被 validation 选为 0，且 FPR95=1.0，双尾贡献和高接受率 operating point 的失效模式需在后续诊断中单独分析。
3. B0–B5 的五种子置信区间普遍较宽，当前只能作第一轮定量结果，不宜对小差异作确定性排名。
4. 若进入第二轮，应先由用户决定是否处理长度捷径、增加类别划分重复或扩大种子；未经批准不自行加入角度轮换、缓冲带或创新模型。
