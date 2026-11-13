---
name: afsim-scenario
description: 把作战规划转化为 AFSIM 想定代码。支持两条入口：自然语言口述规划（nl_to_afsim 工具，由 AFSIM 想定编译器编译强类型 ScenarioIR）和 MissionOrch-LC 生成的结构化 COA Markdown（五步工具链）。触发词：自然语言想定、口述规划、COA、任务规划、AFSIM 想定、scenario generation、想定生成、COA 转 AFSIM。当用户用自然语言描述作战行动、或提供 COA MD 文件时使用。
metadata:
  nanobot:
    requires:
      bins: []
      env: []
    always: false
---

# AFSIM Scenario Generator — 作战规划 → AFSIM 想定

> 本 Skill 教 Agent 如何把作战规划转化为可被 AFSIM Warlock/Wizard 加载运行的
> 想定代码（.txt）。两条入口共用同一套验证器，按输入形态选择。

## 选哪条入口

| 输入形态 | 入口 | 说明 |
| --- | --- | --- |
| 自然语言口述规划（"2架J-16D突入，预警机在后方400km搜索…"） | `nl_to_afsim` | LLM 只填强类型 ScenarioIR，AFSIM 文本由 AFSIM 想定编译器渲染 |
| MissionOrch-LC 输出的结构化 COA Markdown | `afsim_agent_loop` 五步工具链 | 依赖 COA 的实体清单/阶段/矩阵表格结构 |

自然语言输入**不要**走五步工具链：`coa_parse` 需要 `## 六、实体清单` 这类固定表格，
喂自由文本会解析不出实体。

## 入口 A：自然语言 → 想定文件（推荐用于口述规划）

```
自然语言规划
  │  ← LLM 只做语义抽取，输出 ScenarioIR JSON（nl_scenario_plan / nl_to_afsim）
  ▼
ScenarioIR ──► Pydantic 强类型校验 + 能力白名单 + 静态 lint
  │  ← 报错回灌给模型改 IR；再叠一轮 IR 层战术合理性 refine
  ▼
AFSIM 文本 ──► AFSIM 想定编译器的 Jinja2 渲染（同一 IR 必得同一输出，不调用大模型）
  │
  ▼
想定文件 .txt
```

关键设计：**修复发生在 IR 层，不在文本层**。不要用正则去补 `end_platform` ——
文本层补丁能把评分改好看，却治不了根因，还容易改坏本来正确的代码。

三个工具：

- `nl_to_afsim` — 端到端。参数：`request`（必填，自然语言）、`output_path`、
  `center_lat` / `center_lon`（战区中心，默认 27.5 / 122.5）、`duration_s`、
  `scenario_id`、`max_repair_rounds`（默认 3，含需求核对轮）、`audit_requirements`（默认 true）、
  `check_requirements`（默认 true，把需求里的数值约束抽成条款由代码核验，抽取调用不计轮数）。
- `nl_scenario_plan` — 只产出 ScenarioIR，不落地 AFSIM，便于人工审阅 IR。
- `scenario_compile` — AFSIM 想定编译器（AFSIM Scenario Compiler）的工具入口：ScenarioIR → AFSIM 文本，不调用大模型，同一 IR 可反复重放。

API key 走 `DEEPSEEK_API_KEY` 环境变量，别写进工具参数（会进 trace 和反思记忆）。
`DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL` 可切换到任意 OpenAI 兼容服务。

### 输出里必须看的七处

| 字段 | 含义 |
| --- | --- |
| `validator.score` | 四层验证器评分，语法/语义/一致性/战术 |
| `capability_gaps` | **需求里表达不了的能力**，代码扫需求文本得出，不依赖模型自述 |
| `unverified_capabilities` | 用到了放行但未经 `mission.exe` 实跑验证的 AFSIM 类型 |
| `normalizations` | 代码对 IR 做的确定性修正（如删掉无航路的 mover） |
| `requirements_audit` | 需求核对轮结果：`met` 为模型对照代码推算的事实表确认需求数值都已满足，`unconfirmed` 为核对过但最终版没得到确认，空串为未进行 |
| `requirement_checks` / `requirement_notes` | 从需求原文抽出、由代码逐版核验的数值条款；notes 记录被丢弃的条款和抽取失败原因，空列表表示没抽到或未开启 |
| `trace` | 每轮 extract / repair / refine / audit 修了什么 |

`capability_gaps` 非空不代表失败，代表**部分建模 + 缺口已声明**。这是刻意的：
表达不了就如实说，绝不用相近字段假装实现。实测中模型会把几何传感器改名叫
`ESM_SENSOR` 就当侦收做完了，还在 assumptions 里写"使用默认干扰频段"——
而想定里根本没有干扰。这类静默退化比直接报错危险得多。

### 域与 mover 的强约束

平台的 `domain` 决定 `mover_type`，schema 层强制，违反直接打回：

| domain | mover_type | 说明 |
| --- | --- | --- |
| air | `WSF_AIR_MOVER` | 已经过真机验证 |
| sea | `WSF_SURFACE_MOVER` | 高度强制为 0；**未经真机验证** |
| land | `WSF_GROUND_MOVER` | 高度用地形海拔；**未经真机验证** |
| space | 不支持 | `WSF_SPACE_MOVER` 用开普勒根数，与经纬高航点模型不同构 |

固定目标（雷达阵地、发射阵地）不给 `mover_type`；给了 `mover_type` 就必须给 `route`
（至少一个航路点，`speed_mps` 严格大于 0）。想表达平台不动就省略 mover，不要写速度 0 的航路。

### 几何可达性（IR 层，纯计算）

语法检查管不到"想定跑得起来但达不成目标"。这一层用大圆距离 + 3D 斜距校核三件事：

| 检查 | 抓什么 |
| --- | --- |
| 断言可达性 | 观察方与目标全程最近距离 > 传感器 `maximum_range_m` → 断言永远不触发 |
| 航路可完成性 | 航路总长 ÷ 速度 > `duration_s` → 后段航路走不到 |
| 交战探测支撑 | 射手一方所有传感器全程都够不到目标 → 侦察-锁定-打击链断了 |

判定刻意保守，只在"无论如何都不可能"时才报：最近距离取两条轨迹全组合的最小值
（时间同步真实值的下界），再减去采样安全余量才与射程比较。宁可漏报，不能误报——
误报会让修复循环去改一个本来正确的想定。

空间域平台整体跳过：轨道运动未建模，卫星只能摆成一个占位坐标，对它做几何判定
只会产出无法修复的告警，把修复循环带进死胡同。

### 机动性一致性（IR 层，按 type_hint）

`mover_type` 与平台身份要对得上，两个方向都查：

- **固定设施挂了 mover** —— 雷达站/雷达阵地/机场/发射井带着 `mover_type` 和航路，
  语法合法但雷达站会跑。
- **机动装备没有 mover** —— 战斗机/驱逐舰/坦克被建成静止平台，需求里的巡逻、
  拦截、推进全都不会发生。

分类基于 `type_hint` 自由文本，因此认不出来就返回 `unknown` 什么都不报；
`移动式`/`车载`/`自行`/`mobile` 这类修饰词会否决"固定设施"判定（移动式发射阵地是能跑的）。

### 当前已知缺口

电子战干扰、ESM 无源侦收、通信/数据链、导弹飞行与制导、雷达方程级建模、
空间轨道运动、水声探测、行为树自主决策、燃油与空中加油 —— 这些需求会被
`capability_gaps` 显式列出，能建模的部分照常建（平台/航路/探测/定时发射）。

非作战仿真需求（交通客流、经济、社会）在入口就会被拒绝，不会硬编想定。

## 语法权威性顺序（重要）

判断 AFSIM 语法时按这个顺序，不要反过来：

1. **官方 demo 语料**（下文以 `$AFSIM_DEMOS` 表示其 `examples/demos` 目录）
   （4749 文件、40+ 领域）—— 已知能跑的代码
2. AFSIM 官方文档
3. 本目录 `afsim-refs/` 下的二手总结

`afsim-refs/corpus-facts.md` 是从官方语料实测出来的事实表（坐标四种格式、side 任意
标识符、脚本体是另一门语言、块起始的歧义形态…），**并且记录了 `quick-ref.md` 与
`errors-ref.md` 已被证伪的两处说法**：行尾注释其实合法、十进制度坐标其实合法。
引用那两份文档前先看勘误。

## 检查器与语料回归

`scripts/validate_afsim.py` 既能查单文件，也能对整份语料量化误报：

```bash
# 查单个想定
python3 scripts/validate_afsim.py scenario.txt

# 官方 demo 语料回归（4749 文件，约 45 秒；0-error 率 <99% 则退出码非 0）
export AFSIM_DEMOS=/path/to/afsim_verify/examples/demos
python3 scripts/validate_afsim.py --corpus "$AFSIM_DEMOS"

# 定位某条规则的误报根因
python3 scripts/validate_afsim.py --corpus DIR --explain SYN_002
```

当前基线：官方 demo **4728/4749 = 99.6% 零 error**；air_to_air 语料 113/113 零 error。
改任何检查规则后都要重跑这两个数字——检查器的价值取决于它在已知能跑的代码上有多干净，
误报会让自修正循环去改本来正确的代码。

剩余 89 个 error 里 48 个来自 `satellite_demos/.../satcat.txt`（NORAD 目录转储，
平台名真的重复），其余是"块在本文件打开、在另一个文件闭合"这类单文件验证器固有的局限。

## 入口 B：结构化 COA Markdown → 想定（五步工具链）

### WebUI 规划交接与多轮续修

当会话首轮指定 `scenarios/<handoff_id>/coa.md` 时，这是服务固定的结构化 COA 快照，使用本入口的 COA 解析／模板／坐标／生成工具链，不将其改走自然语言入口。先读取该工作区相对路径，核对执行范围、实体清单、地理参考和阶段条件；只执行快照中的已选行动。背景实体不额外执行未选任务；缺少参数或文本涉及未选依赖时，在当前聊天询问用户，不自动补前置行动。

首次生成的想定、解析结果和当前结构数据均保存到同一 `scenarios/<handoff_id>/` 目录，工具的 `output_path`／`output_dir` 显式指向该目录。不要覆盖来源 `coa.md` 或 `source.json`，不要写到其他会话场景目录。模型、工具与修复机制沿用本技能现有配置。

后续用户提出修改时，先读取该会话目录中的当前产物与结构数据，在这些当前数据上修改并更新同目录产物；保留首轮失败、工具错误和用户反馈历史。固定来源是本会话的快照，不读取 `.nanobot/plans/` 最新稿替换来源，不自动另开会话。规划新版本需要用户显式再次交接，产生新来源和新会话。

交接派发中断或生成失败时先查看已有会话／文件并澄清，不自动重复首轮。规划结构报告不等于真实想定 lint 或仿真结果，未执行的验证不得声明通过。

## 核心算法流程

```
输入: COA MD 文件（由 MissionOrch-LC 生成）
  │
  ▼
Step 1: 结构化解析（coa_parse 工具）
  │   提取: entities[], phases[], geo_refs[], matrix[][], effects[], decision_points[]
  │
  ▼
Step 2: 语义映射 + 模板匹配（template_match 工具）
  │   对每个 entity: platform_code → 查 platform-mapping.yaml → AFSIM 参数集
  │   对每种 category: 选择最优模板组合 (platform + sensor + weapon + mover)
  │   特征匹配算法: 按 (category, side, weapons, sensors) 加权余弦相似度排序
  │   去重分组: 相同 platform_code 共享 platform_type 定义
  │
  ▼
Step 3: 坐标解算（coord_resolve 工具）
  │   geo_ref hint（"距目标岛礁 480km"）→ 基于参考点 + 方位 + 距离 → d:m:s 坐标
  │   解算算法: WGS-84 大圆距离 (Haversine/Vincenty), 多约束加权交叉定位
  │
  ▼
Step 4: 分模块代码生成（afsim_generate 工具）
  │   4a: radar_signature / optical_signature 定义
  │   4b: weapon_type 定义（从武器清单映射）
  │   4c: platform_type 定义（按 category 分组，去重）
  │   4d: platform 实例 + route 状态机（COA action 序列 → 航路点）
  │   4e: 全局配置（random_seed, end_time, log_file）
  │   4f: 冲突检测（频谱/空间/资源冲突）
  │
  ▼
Step 5: 自检验证（scenario_validate 工具）
  │   Level 1: 语法 (块闭合/关键字/坐标格式/单位)
  │   Level 2: 语义 (引用一致性/类型匹配/side统一)
  │   Level 3: 一致性 (platform_type↔platform/weapon引用/名称去重)
  │   Level 4: 战术合理性 (速度高度/传感器覆盖/武器射程)
  │   评分: 0-100 分, 按严重度扣分
  │
  ▼
Step 6: 输出到 staging + 归档闭环
```

## Step 1: 结构化解析

使用 `coa_parse` 工具读取 COA MD 文件:

```
coa_parse(file_path="/path/to/coa_output.md")
```

返回结构化 dict:

```json
{
  "name": "方案名称",
  "description": "方案概述",
  "entities": [
    {"unit_id": "B_AWACS", "platform_code": "KJ-500", "quantity": 1, "side": "BLUE", "role": "预警指挥", "weapons": "无", "base": "<BASE:B-Mainland>"}
  ],
  "phases": [
    {"phase_id": "Phase_1", "name": "起飞与编队集结", "trigger": "ON_START"}
  ],
  "geo_refs": [
    {"ref_id": "<BASE:B-Mainland>", "type": "BASE", "side": "BLUE", "hint": "距目标岛礁800km"}
  ],
  "matrix": [
    {"unit_id": "B_AWACS", "phase_id": "Phase_1", "actions": ["TAKEOFF <BASE:B-Mainland>", "MOVE_TO <ORBIT:ORB-AWACS>"]}
  ],
  "effects": [...],
  "decision_points": [...]
}
```

如果 `coa_parse` 工具不可用，Agent 应直接用 `read_file` 读取 MD 文件并手动解析表格。

## Step 2: 语义映射 + 模板匹配

### 2.1 使用 template_match 工具

```
template_match(
  entities_json="<coa_parse返回的entities JSON>",
  mapping_path="/path/to/platform-mapping.yaml"  // 可选，默认自动定位
)
```

返回:
- `matches[]`: 每个 entity 的匹配结果 (matched_mapping, similarity_score, components)
- `platform_groups[]`: 去重后的平台组 (afsim_type_name, category, mover, total_quantity)
- `summary`: 统计 (unique_platform_types, avg_similarity)

### 2.2 平台映射

加载 `platform-mapping.yaml`（与本文件同目录），按 platform_code 查找 AFSIM 参数:

```yaml
# 示例: platform_code = "KJ-500"
KJ-500:
  icon: E-3_AWACS          # AFSIM icon 名
  mover: WSF_AIR_MOVER     # 基类型
  category: awacs          # 功能分类（决定传感器/武器组合）
  speed: 450 kts
  altitude: 35000 ft msl
```

### 2.3 模板选择算法（由 template_match 工具自动执行）

对每个 entity，按以下优先级选择模板:

1. **精确匹配**: platform_code 完全命中 → score=1.0
2. **加权余弦相似度**: 多维特征向量 (category 0.35 + side 0.15 + weapons 0.25 + sensors 0.15 + role 0.10)
3. **回退匹配**: 使用 GENERIC_AIR/GENERIC_GROUND

### 2.4 参考模板位置

| 模板类型 | 路径 |
|----------|------|
| 快速参考 | `afsim-refs/quick-ref.md` |
| 平台模板 | `afsim-templates/platform/` |
| 传感器模板 | `afsim-templates/sensor/` |
| 武器模板 | `afsim-templates/weapon/` |
| 处理器模板 | `afsim-templates/processor/` |
| 脚本模板 | `afsim-templates/script/` |
| 完整语法 | `afsim-refs/complete-ref.md` |
| Demo 索引 | `afsim-refs/demos-index.md` |
| 报错索引 | `afsim-refs/errors-ref.md` |
| COA 映射指南 | `references/coa-to-afsim-guide.md` |

## Step 3: 坐标解算

使用 `coord_resolve` 工具:

```
coord_resolve(
  geo_refs_json="<coa_parse返回的geo_refs JSON>",
  center_lat=27.5,        // 目标区域中心纬度
  center_lon=122.5,       // 目标区域中心经度
  default_side="blue"
)
```

COA 中的地理参考多为**相对描述**（如 "距目标岛礁 480-520km"），该工具自动:

1. **解析绝对坐标**: 如 hint 中包含 d:m:s 或 decimal degrees → 直接解析
2. **提取距离约束**: 正则匹配 "距XXX NNNkm" / ">NNNkm" 等模式
3. **推断方位角**: 从关键词（"西侧"→270°, "前方"→90°）或 ref_type 推断
4. **大圆正算**: Haversine/Vincenty 公式计算目标坐标
5. **多约束加权求解**: 多个距离约束时按精度加权平均

**坐标格式要求**: AFSIM 使用 `d:m:s N/S E/W` 格式，如 `30:00:00n 120:00:00e`。

详细算法见 `references/coa-to-afsim-guide.md`。

## Step 4: 分模块代码生成

使用 `afsim_generate` 工具自动完成代码合成:

```
afsim_generate(
  coa_data_json="<coa_parse返回的完整JSON>",
  coords_json="<coord_resolve返回的JSON>",
  matches_json="<template_match返回的JSON>",
  scenario_name="my_scenario",
  end_time_sec=21600          // 6小时
)
```

返回:
- `scenario_code`: 完整 AFSIM 想定代码 (可直接保存为 .txt)
- `stats`: 统计 (platform_types, platform_instances, sensors, weapons, route_waypoints)
- `conflicts`: 冲突检测结果 (频谱/空间/资源)
- `warnings`: 生成过程中的警告

工具自动生成以下模块（也可由 Agent 手动逐模块生成）:

### 4a: Signature 定义

```text
radar_signature {{NAME}}_SIGNATURE WSF_RADAR_SIGNATURE
   constant {{RCS}} m^2
end_radar_signature
```

### 4b: platform_type 定义

从 Step 2 的映射结果生成。每个唯一的 platform_code 生成一个 platform_type。

### 4c: sensor 定义

按 entity 的 category 分配传感器:
- awacs → 远程预警雷达 sensor
- ew → ESM sensor（被动侦收）
- fighter → 火控雷达 sensor
- sam → 搜索雷达 + 火控雷达

### 4d: weapon_type 定义

从 entity 的 weapons 字段解析武器列表，查武器映射表生成。

### 4e: platform 实例 + route

这是最核心的算法。从 COA matrix 中每个 entity 在每个 phase 的行动序列推导 route:

| COA 行动关键词 | AFSIM route 转换 |
|----------------|------------------|
| TAKEOFF <BASE:X> | 初始 position = BASE 坐标 + speed |
| MOVE_TO <WAYPOINT:X> | route position = WAYPOINT 坐标 + speed |
| PATROL <ORBIT:X> | 两个相邻 position 形成巡逻轨道 |
| ENGAGE | weapon 发射（不生成 route 点，但需 target 指定） |
| LOITER | 同一 position 停留（end_time 控制） |
| EGRESS / RTB | route position 指向撤离点/基地 |
| REFUEL | route position = 加油区坐标 |

### 4f: processor / comm

如有数据链/通信需求，生成 comm_transceiver 和 processor 定义。

### 4g: 全局配置

```text
random_seed 1
end_time {{仿真总时长}} sec
log_file output/{{scenario_name}}.log
thread_count 4
```

## Step 5: 自检验证

使用 `scenario_validate` 工具进行多层级校验:

```
scenario_validate(
  scenario_code="<生成的AFSIM代码>",
  coordinator_json="<coa_parse返回的JSON>"  // 可选, 用于交叉验证
)
```

返回:
- `score`: 0-100 评分 (errors -10, warnings -3, info -1)
- `grade`: A/B/C/D/F 等级
- `issues[]`: 每个问题的详细描述 + 修复建议 + 行号
- `pass`: 是否通过 (errors == 0)

### 四级验证清单

- [ ] 所有 `end_*` 闭合完整（platform_type/end_platform_type, mover/end_mover 等）
- [ ] 基类型拼写正确（`WSF_PLATFORM`, `WSF_AIR_MOVER`, `WSF_GEOMETRIC_SENSOR` 等）
- [ ] 坐标格式 `d:m:s N/S E/W`（如 `30:00:00n 120:00:00e`）
- [ ] 高度单位 `ft msl` 或 `m agl`
- [ ] 速度单位 `kts` 或 `m/s`
- [ ] 时间单位 `sec` 或 `min`
- [ ] weapon/sensor 名称引用一致
- [ ] 无行尾注释（AFSIM 不支持 `#` 注释在行尾）
- [ ] side 字段统一为 `red` / `blue`

## Step 6: 输出与归档

### 输出

将生成的代码写入 `output/staging/{{scenario_name}}.txt`。

告知用户:
- 输出路径
- 包含的 platform_type 数量和列表
- 包含的 sensor / weapon 列表
- 建议下一步: 用 Warlock/Wizard 打开验证

### 归档闭环

用户认可后（或 Warlock 验证通过），移动到 `output/verified/scenario/`。

## 渐进披露

| 层级 | 文件 | 何时加载 |
|------|------|----------|
| 0（始终） | 本文件 | 默认 |
| 1 | `platform-mapping.yaml` | 执行 Step 2 时 |
| 2 | `references/coa-to-afsim-guide.md` | 需要坐标解算或详细映射示例时 |
| 3 | `afsim-refs/quick-ref.md` | 需要 AFSIM 语法参考时 |
| 4 | `afsim-refs/errors-ref.md` | 遇到报错时 |
| 5 | `afsim-refs/complete-ref.md` | 需要完整语法时 |
| 6 | `afsim-refs/demos-index.md` | 需要 Demo 参考时 |

## 算法工具链

| 工具名 | 功能 | 算法核心 |
|--------|------|----------|
| `coa_parse` | COA MD → 结构化 JSON | Markdown 表格解析 + 正则实体抽取 |
| `template_match` | 实体 → AFSIM 模板 | 多维加权余弦相似度 + 约束满足 |
| `coord_resolve` | 地理描述 → d:m:s 坐标 | WGS-84 Haversine/Vincenty + 多约束求解 |
| `afsim_generate` | 结构化数据 → AFSIM 代码 | Action→Route 状态机 + 冲突检测图 |
| `scenario_validate` | AFSIM 代码 → 评分 | 四层校验 (语法/语义/一致性/战术) |

## 示例文件

| 文件 | 说明 |
|------|------|
| `examples/simplified_sead.md` | 简化 SEAD 测试 (6 实体, 3 阶段) |
| `examples/full_sead_coa.md` | 完整 SEAD 方案 (6 编队, 7 阶段, 来自 MissionOrch-LC) |

## 命令入口

| 触发 | 功能 |
|------|------|
| 用户提供 COA MD 文件路径 | 全自动 COA → AFSIM 转化（5 步工具链） |
| "生成 AFSIM 想定" + COA 描述 | 半自动，逐模块确认 |
| "修改想定" + 修改需求 | 基于已有想定增量修改 |
| "验证想定" + AFSIM 代码 | 仅执行 Step 5 验证 |
