import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const markdown = readFileSync(join(root, "A级文献调研报告.md"), "utf8").trim();
const generatedAt = "2026-07-27T12:00:00+08:00";

const parts = markdown.split(/\n(?=## )/);
const blocks = [];

for (let index = 0; index < parts.length; index += 1) {
  const body = parts[index].trim();
  const id = index === 0 ? "title" : `section_${String(index).padStart(2, "0")}`;
  blocks.push({ id, type: "markdown", body });

  if (body.startsWith("## 1. ")) {
    blocks.push({
      id: "coverage_metrics",
      type: "metric-strip",
      cardIds: ["strict_a0", "a0_fulltext", "second_round_new_a0"],
    });
  }

  if (body.startsWith("## 2. ")) {
    blocks.push({
      id: "core_evidence_table",
      type: "table",
      tableId: "core_evidence",
      layout: "full",
    });
  }

  if (body.startsWith("## 4. ")) {
    blocks.push({
      id: "jdsr_performance_chart",
      type: "chart",
      chartId: "jdsr_reported_performance",
      layout: "full",
    });
  }
}

const sources = [
  {
    id: "review_summary",
    label: "A级文献检索覆盖汇总",
    path: "A级文献矩阵.csv",
    query: {
      engine: "portable-sql-values",
      sql: "SELECT * FROM (VALUES (2, 1, 0)) AS t(strict_a0, a0_fulltext, second_round_new_a0);",
      description: "把人工核验后的检索覆盖计数编码为可复核的 SQL VALUES 行；该 SQL 是报告数据打包步骤，不是数据库检索式。",
      executed_at: generatedAt,
      tables_used: [
        "Lundén & Koivunen (2016)",
        "Liu et al. JDSR (2023)",
        "Qu et al. JSR (2022)",
        "Zhang et al. Improved Sparse (2023)",
        "Zhang et al. MtCS (2024)",
        "CN116310504A",
      ],
      filters: [
        "A0 要求多个雷达站或多个单/双基地通道",
        "多个 HRRP 必须共同参与同一次判决",
        "任务必须包含未知类别拒识",
        "必须给出方法和实验结果",
      ],
      metric_definitions: [
        "严格A0文献数：同时满足四项A0条件且经官方摘要或全文核验的独立论文数。",
        "A0已取得全文：可访问并核对完整方法与实验正文的A0论文数。",
        "第二轮新增A0：完成第二轮宽检索后新增的严格A0论文数。",
      ],
    },
  },
  {
    id: "core_matrix",
    label: "A0/A1核心证据审计矩阵",
    path: "A级文献矩阵.csv",
    query: {
      engine: "portable-sql-values",
      sql: [
        "SELECT * FROM (VALUES",
        "('A0', 2016, 'Lundén & Koivunen', '明确多基地、单/双基地通道、同步多角观测', '多基地仿真', '订阅受限（E2）', '严格A0；物理场景最强'),",
        "('A0', 2023, 'Liu et al. JDSR', '论文场景明确多站；实验没有物理站点', 'MSTAR反演HRRP + 方位分组', '官方全文（E1）', '严格A0；须标注伪多站实验'),",
        "('A1', 2022, 'Qu et al. JSR', '摘要明确单站观测', 'MSTAR反演HRRP', '开放论文', '单站多视；JDSR前驱'),",
        "('A1', 2023, 'Zhang et al. Improved Sparse', '摘要无多站或多视联合证据', 'MSTAR反演HRRP', '订阅受限（E2）', '同团队开集稀疏分支'),",
        "('A1', 2024, 'Zhang et al. MtCS', '摘要只说明小角度区间多个观测', 'MSTAR反演HRRP', '订阅受限（E2）', '直接多视开集；待核验站点定义'),",
        "('A1', 2023, 'CN116310504A', '多视流程；无独立物理多站实验证据', '与JDSR场景高度重合', '专利全文', '方法族佐证，不重复计数')",
        ") AS t(tier, year, paper, station_evidence, data, fulltext, decision);",
      ].join("\n"),
      description: "把逐篇人工核验结果编码为可复核的 SQL VALUES 表；原始判定与更完整字段保存在A级文献矩阵.csv。",
      executed_at: generatedAt,
      tables_used: ["A级文献矩阵.csv"],
      filters: [
        "仅展示A0与同一核心方法族A1候选",
        "专利不作为独立论文计数",
      ],
    },
  },
  {
    id: "jdsr_official",
    label: "Liu et al. (2023) JDSR 官方全文与表3",
    href: "https://jeit.ac.cn/cn/article/doi/10.11999/JEIT221284?viewType=HTML",
    query: {
      engine: "portable-sql-values",
      url: "https://jeit.ac.cn/cn/article/doi/10.11999/JEIT221284?viewType=HTML",
      sql: [
        "SELECT * FROM (VALUES",
        "('JDSR-OSR', 'Situation-I', 0.832), ('JDSR-OSR', 'Situation-II', 0.816),",
        "('SR-OSR', 'Situation-I', 0.809), ('SR-OSR', 'Situation-II', 0.732),",
        "('W-SVM', 'Situation-I', 0.786), ('W-SVM', 'Situation-II', 0.687),",
        "('1-vs-set', 'Situation-I', 0.794), ('1-vs-set', 'Situation-II', 0.709),",
        "('KLD-RPA', 'Situation-I', 0.813), ('KLD-RPA', 'Situation-II', 0.752)",
        ") AS t(method, scene, rate);",
      ].join("\n"),
      description: "人工转录论文表3后，以 SQL VALUES 精确表示图表输入；来源数值仍以论文官方表3为准。",
      executed_at: generatedAt,
      tables_used: ["Liu et al. (2023), Table 3"],
      filters: ["仅展示论文内部比较，不进行跨论文排名"],
      metric_definitions: ["平均识别率按论文表3原值展示；0.832在图中格式化为83.2%。"],
    },
  },
];

const artifact = {
  surface: "report",
  manifest: {
    version: 1,
    surface: "report",
    title: "多站多视角 HRRP 开集识别：A级文献调研报告",
    description: "严格区分多站联合观测、单站连续角度、多极化和多模态，并形成A0/A1/A2证据链、技术档案、基线与统一实验协议。",
    generatedAt,
    filters: [],
    cards: [
      {
        id: "strict_a0",
        description: "在本轮可访问文献范围内同时满足多站、多HRRP联合、未知拒识和实验四项条件的论文。",
        dataset: "coverage_summary",
        sourceId: "review_summary",
        metrics: [{ label: "严格 A0 文献", field: "strict_a0", format: "number" }],
      },
      {
        id: "a0_fulltext",
        description: "已能逐项核对完整方法、实验设置和结果的严格A0论文。",
        dataset: "coverage_summary",
        sourceId: "review_summary",
        metrics: [{ label: "A0 已取得全文", field: "a0_fulltext", format: "number" }],
      },
      {
        id: "second_round_new_a0",
        description: "第二轮宽检索在既有种子和引用链之外新增的严格A0论文。",
        dataset: "coverage_summary",
        sourceId: "review_summary",
        metrics: [{ label: "第二轮新增 A0", field: "second_round_new_a0", format: "number" }],
      },
    ],
    charts: [
      {
        id: "jdsr_reported_performance",
        title: "JDSR论文表3的平均识别率",
        subtitle: "仅比较同一论文、同一协议内的两种实验场景；不可用于与2016年论文横向排名。",
        headerMarkdown: "Situation-I 为连续小角度强相关视角；Situation-II 为随机角度顺序。数值来自论文表3。",
        type: "bar",
        dataset: "jdsr_performance",
        sourceId: "jdsr_official",
        encodings: {
          x: { field: "method", type: "nominal", label: "方法" },
          y: { field: "rate", type: "quantitative", label: "平均识别率", format: "percent" },
          color: { field: "scene", type: "nominal", label: "实验场景" },
          tooltip: [
            { field: "method", type: "nominal", label: "方法" },
            { field: "scene", type: "nominal", label: "实验场景" },
            { field: "rate", type: "quantitative", label: "平均识别率", format: "percent" },
          ],
        },
        yAxisTitle: "平均识别率",
        valueFormat: "percent",
        layout: "full",
        comparisonContext: {
          scope: "within-paper",
          warning: "Heterogeneous datasets and unavailable 2016 full metrics prevent cross-paper comparison.",
        },
      },
    ],
    tables: [
      {
        id: "core_evidence",
        title: "A0/A1核心证据审计表",
        subtitle: "“多视”只有在明确来自多个站点或单/双基地通道时，才满足严格A0场景条件。",
        dataset: "core_evidence",
        sourceId: "core_matrix",
        density: "comfortable",
        layout: "full",
        columns: [
          { field: "tier", label: "级别", type: "text" },
          { field: "year", label: "年份", type: "number", align: "right" },
          { field: "paper", label: "文献", type: "text" },
          { field: "station_evidence", label: "多站证据", type: "text" },
          { field: "data", label: "实验数据", type: "text" },
          { field: "fulltext", label: "全文状态", type: "text" },
          { field: "decision", label: "判定", type: "text" },
        ],
      },
    ],
    sources: sources.map(({ query, ...source }) => source),
    blocks,
  },
  snapshot: {
    version: 1,
    generatedAt,
    status: "ready",
    datasets: {
      coverage_summary: [
        { strict_a0: 2, a0_fulltext: 1, second_round_new_a0: 0 },
      ],
      core_evidence: [
        {
          tier: "A0",
          year: 2016,
          paper: "Lundén & Koivunen",
          station_evidence: "明确多基地、单/双基地通道、同步多角观测",
          data: "多基地仿真",
          fulltext: "订阅受限（E2）",
          decision: "严格A0；物理场景最强",
        },
        {
          tier: "A0",
          year: 2023,
          paper: "Liu et al. JDSR",
          station_evidence: "论文场景明确多站；实验没有物理站点",
          data: "MSTAR反演HRRP + 方位分组",
          fulltext: "官方全文（E1）",
          decision: "严格A0；须标注伪多站实验",
        },
        {
          tier: "A1",
          year: 2022,
          paper: "Qu et al. JSR",
          station_evidence: "摘要明确单站观测",
          data: "MSTAR反演HRRP",
          fulltext: "开放论文",
          decision: "单站多视；JDSR前驱",
        },
        {
          tier: "A1",
          year: 2023,
          paper: "Zhang et al. Improved Sparse",
          station_evidence: "摘要无多站或多视联合证据",
          data: "MSTAR反演HRRP",
          fulltext: "订阅受限（E2）",
          decision: "同团队开集稀疏分支",
        },
        {
          tier: "A1",
          year: 2024,
          paper: "Zhang et al. MtCS",
          station_evidence: "摘要只说明小角度区间多个观测",
          data: "MSTAR反演HRRP",
          fulltext: "订阅受限（E2）",
          decision: "直接多视开集；待核验站点定义",
        },
        {
          tier: "A1",
          year: 2023,
          paper: "CN116310504A",
          station_evidence: "多视流程；无独立物理多站实验证据",
          data: "与JDSR场景高度重合",
          fulltext: "专利全文",
          decision: "方法族佐证，不重复计数",
        },
      ],
      jdsr_performance: [
        { method: "JDSR-OSR", scene: "Situation-I", rate: 0.832 },
        { method: "JDSR-OSR", scene: "Situation-II", rate: 0.816 },
        { method: "SR-OSR", scene: "Situation-I", rate: 0.809 },
        { method: "SR-OSR", scene: "Situation-II", rate: 0.732 },
        { method: "W-SVM", scene: "Situation-I", rate: 0.786 },
        { method: "W-SVM", scene: "Situation-II", rate: 0.687 },
        { method: "1-vs-set", scene: "Situation-I", rate: 0.794 },
        { method: "1-vs-set", scene: "Situation-II", rate: 0.709 },
        { method: "KLD-RPA", scene: "Situation-I", rate: 0.813 },
        { method: "KLD-RPA", scene: "Situation-II", rate: 0.752 },
      ],
    },
  },
  sources,
};

writeFileSync(join(root, "artifact.json"), `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
