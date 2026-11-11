# AFSIM 想定生成 Agent 使用说明

ScenClaw 在 nanobot 之上增加了一套 AFSIM 想定生成工具，把作战规划转换成可交给 `mission.exe` 运行的 AFSIM 2.9 想定项目。它有两条入口：

| 入口 | 输入 | 核心工具 | 是否需要大模型 |
|---|---|---|---|
| 自然语言 | 一段中文或英文的作战描述 | `nl_to_afsim` | 需要（DeepSeek 兼容接口） |
| 结构化 COA | MissionOrch-LC 输出的 COA Markdown | `afsim_agent_loop` | `classic` 模式不需要，`react` 模式需要 |

自然语言入口的流程是：大模型只负责把需求抽取成强类型的 `ScenarioIR`，每个候选 IR 都要经过 Pydantic schema、编译器能力白名单和 AFSIM 静态 lint 检查，报错回灌给模型做有限轮修复（extract → repair → refine）；最后由确定性编译器渲染成 AFSIM 文本。修复发生在 IR 上，不直接修改生成的文本，所以同一份 IR 每次编译的结果逐字节一致。

## 1. 环境准备

```bash
cd ScenClaw
uv venv .venv
uv pip install --python .venv/bin/python -r pyproject.toml
# 国内网络可加: --index-url https://mirrors.aliyun.com/pypi/simple/
```

大模型通过以下环境变量配置，密钥只放在环境变量里，不要写进代码或配置文件：

| 变量 | 作用 | 默认值 |
|---|---|---|
| `DEEPSEEK_API_KEY` | API 密钥（必填） | 无 |
| `DEEPSEEK_BASE_URL` | OpenAI 兼容接口地址 | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | 模型名 | `deepseek-chat` |

```bash
export DEEPSEEK_API_KEY=sk-...
export DEEPSEEK_MODEL=deepseek-flash
```

## 2. 命令行：一句话生成想定

`run_nl_to_afsim.py` 是最小可用入口，不需要完整启动 nanobot。

```bash
.venv/bin/python run_nl_to_afsim.py "8v8超视距空战想定。蓝方8架歼-16分为蓝一编队、蓝二编队,每编队1架长机+3架僚机,自西向东推进,高度9000米,速度250米/秒,编队内间隔约3公里,两编队南北相距约20公里,每架挂4枚中远程空空导弹,雷达探测距离150公里。红方8架同级多用途战斗机分为红一编队、红二编队(各1长机3僚机),自东向西迎头推进,参数与蓝方相同。双方初始相距约240公里。相距约100公里时蓝方开火,蓝一对红一、蓝二对红二一对一分配目标,红方约30秒后还击。断言:各编队长机须探测到对方编队长机。仿真时长1小时。" \
  -o output/air_8v8 --scenario-id air_8v8_bvr --duration 3600 \
  --rounds 4 --max-tokens 32768 --timeout 600 \
  --ir-out output/air_8v8_ir.json
```

| 参数 | 含义 | 默认值 |
|---|---|---|
| `request` | 自然语言作战规划 | 必填 |
| `-o, --output` | 想定项目根目录 | `generated_scenario` |
| `--center-lat` / `--center-lon` | 战区中心经纬度，用于锚定生成的位置 | `27.5` / `122.5` |
| `--duration` | 任务时长（秒） | `3600` |
| `--scenario-id` | 想定标识 | 由模型生成 |
| `--rounds` | IR 最大修复轮数 | `4` |
| `--max-tokens` | 单次调用的输出预算，推理模型的思维链也计入 | `8192` |
| `--timeout` | 单次调用超时（秒） | `120` |
| `--ir-out` | 同时导出 ScenarioIR JSON | 不导出 |

平台数量多时要调大 `--max-tokens`。例如上面的 16 个平台，用 `deepseek-flash` 时 8192 不够，输出会被截断；改成 32768 后 1 轮收敛。

### 输出内容

运行成功后终端会打印项目路径、文件列表、`main.txt` 行数、平台数、修复轮次和验证器评分，以及下面四类提示（有内容时才打印）：

| 提示 | 含义 |
|---|---|
| 表达不了的需求 | 需求里出现了编译器不支持的能力（见第 8 节），这部分不会被仿真 |
| 未经真机验证的能力 | 用到了已放行但没在 AFSIM 上实际跑过的能力 |
| 自动归一化 | 编译器自动修正的字段，例如按机型描述推断 `platform_type`、删掉 schema 未定义且值为空或与同级字段重复的字段 |
| 遗留合理性问题 | 语法合法但战术上可疑的地方 |

生成的项目目录：

```
output/air_8v8/
├── main.txt              # 入口文件, 传给 mission.exe
├── _common.txt           # 聚合 include
├── sensors/<type>.txt    # 每种传感器一个文件
├── weapons/<type>.txt    # 武器效应定义
├── platforms/<type>.txt  # 每种平台类型一个文件
├── event_output.txt      # 事件输出配置
├── scenario_single.txt   # 单文件等价版, 方便 diff
└── output/               # 运行时产物目录 (mission.log 等)
```

在装有 AFSIM 的机器上，进入项目根目录，把 `main.txt` 交给 AFSIM 的 `mission` 可执行程序运行；`mission.log` 和事件输出会写到 `output/` 下。

## 3. 修改 IR 后重新编译

想调整开火时间、航路点这类细节时，不必重新调用大模型：直接改导出的 IR，再用 `scenario_compile` 确定性编译。

```python
import asyncio, json
from nanobot.agent.tools.nl_scenario_planner import ScenarioCompileTool

raw = asyncio.run(ScenarioCompileTool().execute(
    scenario_ir_path="output/air_8v8_ir.json",
    output_path="output/air_8v8",
))
print(json.loads(raw)["validator"])
```

IR 顶层字段为 `scenario_id`、`mission`、`selected_coa`、`sensors`、`platforms`、`assertions`、`engagements`。例如修改 `engagements[*].fire_time_s` 就能调整各机开火时刻。

## 4. 在 nanobot agent 中对话调用

`nanobot/agent/tools/` 下的 AFSIM 工具由 nanobot 自动发现注册，`nanobot/skills/afsim-scenario/` 是对应的内置 skill。nanobot 自身的对话模型按 [`quick-start.md`](./quick-start.md) 配置即可。

注意：AFSIM 工具内部调用大模型时读取的是上面的 `DEEPSEEK_*` 环境变量，不读取 nanobot 的 provider 配置，所以启动前要先导出：

```bash
export DEEPSEEK_API_KEY=sk-...
export DEEPSEEK_MODEL=deepseek-flash
nanobot agent
```

之后在对话里直接描述需求，例如“用 nl_to_afsim 生成一个 4v4 空战想定，输出到 output/air_4v4，max_tokens 设为 32768”，agent 会调用相应工具并返回评分和修复记录。

## 5. Python 调用

```python
import asyncio, json
from nanobot.agent.tools.nl_scenario_planner import NLToAfsimTool

raw = asyncio.run(NLToAfsimTool().execute(
    request="两架歼-20在东海6000米高度巡逻，地面雷达站负责搜索预警",
    output_path="output/patrol",
    duration_s=3600,
    max_repair_rounds=4,
    max_tokens=16384,
    timeout_s=300,
    include_code=False,
))
result = json.loads(raw)
print(result["success"], result.get("validator"))
```

只想拿到 IR、不落地 AFSIM 文本时，用 `NLScenarioPlanTool`（工具名 `nl_scenario_plan`），参数与上面相同，只是没有 `output_path` 和 `include_code`。也可以绕过工具层直接使用 `ScenarioIRPlanner(llm=..., max_repair_rounds=..., max_tokens=..., audit_requirements=..., check_requirements=...)`。

两个工具默认开启需求核对轮（`audit_requirements=true`）：IR 通过全部通用检查后，再把代码推算的事实表（航点到达时刻与坐标、开火时斜距与结算结果、双方初始最近距离、存活数）交给模型对照需求原文里的数值约束，满足则回确认标记，否则输出修正后的 IR，修正版仍按“合理性问题不增加”验收。修正版（或 refine 版）只是带了 schema 小错时会先补一次 repair，再坏才保留上一版。这一轮计入 `max_repair_rounds`，结果见输出的 `requirements_audit`。直接构造 `ScenarioIRPlanner` 时默认关闭。

两个工具还默认开启需求条款核验（`check_requirements=true`）。拿到第一版合法 IR 后，再单独调用一次模型，把需求原文里写明数值的约束抽成固定类型的条款：双方初始距离、编队间隔、开火时段、分阶段开火、开火斜距、进入区域的时刻与存活、高度、速度。模型只抽取、不计算，这次调用不计入 `max_repair_rounds`。需求只按阶段给出时段时（如“0-240 秒展开、240-540 秒争夺……”），提示会让模型把这些阶段用作分阶段开火的时段，以及到达、越线等动作的时段。之后每一版 IR 都由代码按匀速推算逐条核验，不满足的条款以“需求”开头排在合理性问题前面回灌 refine，报文直接给出修法所需的数字，例如应把双方拉开或移近多少 km、开火斜距落在包络内的时段、进区前航程和所需平均速度。按计划打不出的射击（射手或目标已先被击毁）同样核验开火斜距：前面的击毁一旦没生效，它就会在包络外开火。核对轮提示里列出已由代码核验通过的条款，模型不必再推算。抽取出的条款见输出的 `requirement_checks`；条款不合法、选择器匹配不到平台或抽取失败时整条丢弃，并在 `requirement_notes` 里说明，不影响规划本身。extract 轮推理耗尽 `max_tokens` 时重问的那一次同样不计入轮数。

repair、refine 和核对轮的对话里都带着上一版完整 IR，提示要求只输出补丁 `{"__patch__": {...}}`（代码仍接受完整 IR），由代码合并后再走完整校验：对象按键合并，`null` 表示删键；`platforms`、`mission.platforms`、`engagements`、`assertions`、`sensors` 和 `platforms[].weapons` 按各自的 id 合并，元素写 `"__delete__": true` 表示删除，新 id 追加；其余列表整体替换。补丁缺 id 或要删的 id 不存在时整份拒绝，按 repair 重问。原因是推理模型的思维链和正文共用 `max_tokens`：12v12 的完整 IR 有数万字，只改几处却整份重抄，正文常被截断。修订轮被截断时会催一次改用补丁。模型正常收尾（`finish_reason=stop`）却漏写结尾一两个闭合符时由代码补齐；被截断的输出不补，以免把半截补丁当成完整补丁合并。修订版只改了 `platforms[]` 的位置或航路、没动 `mission.platforms` 里的意图时，如果上一版两者一致，代码把新的位置和航路同步到意图并记入 `normalizations`；上一版本就不一致或意图也改了时不动。合理性问题里附带修法所需的数值（开火时刻探测不到目标时给出可探测时段，航路超时时给出已飞完的航点数和所需平均速度），模型不必在思考里重算；一轮修不完可以先修一部分，问题总数不增加就会被接受。输出的 `trace` / `attempts` 里用 `patched` 标出哪些轮是补丁。

## 6. COA Markdown 入口

结构化 COA 走五步工具链：COA 解析 → 坐标解算（WGS-84）→ 模板匹配 → AFSIM 生成 → 四层验证（语法 / 语义 / 一致性 / 战术）。`afsim_agent_loop` 把这五步和自修正循环封装在一起，提供两种互相独立的模式：

| 模式 | 机制 | 依赖 |
|---|---|---|
| `classic`（默认） | 错误依赖图 + 反向 PageRank 定位根因，修复策略库做确定性代码变换，每轮得分单调不降 | 不需要大模型 |
| `react` | 大模型驱动的 Thought → Action → Observation 循环，配合 Reflexion 长短期记忆 | `DEEPSEEK_API_KEY`（未设置时直接报错）；运行中 API 调用失败会降级为固定流水线 |

```python
import asyncio, json
from nanobot.agent.tools.afsim_agent_loop import AFSIMAgentLoopTool

raw = asyncio.run(AFSIMAgentLoopTool().execute(
    coa_file_path="nanobot/skills/afsim-scenario/examples/simplified_sead.md",
    mode="classic",      # 或 "react"
    max_rounds=5,
    target_score=85,
))
result = json.loads(raw)
print(result["final_score"], result["score_trajectory"])
open("sead.txt", "w", encoding="utf-8").write(result["scenario_code"])
```

示例 COA 见 `nanobot/skills/afsim-scenario/examples/`。`react` 模式的长期记忆保存在 `~/.nanobot/afsim_memory.json`。`react` 模式额外返回 `trajectory` 和 `memory_stats`，`classic` 模式返回 `rounds` 和 `score_trajectory`，两者可直接用于对比实验。

## 7. 工具一览

| 工具名 | 作用 |
|---|---|
| `nl_to_afsim` | 自然语言 → AFSIM 想定项目（端到端） |
| `nl_scenario_plan` | 自然语言 → ScenarioIR |
| `scenario_compile` | ScenarioIR → AFSIM 文本（确定性，不调用大模型） |
| `afsim_agent_loop` | COA Markdown → AFSIM，带自修正循环（classic / react） |
| `coa_parse` | 解析 COA Markdown |
| `coord_resolve` | 地理参照解算为 WGS-84 坐标 |
| `template_match` | 实体与平台模板的多维加权匹配 |
| `afsim_generate` | 由 COA、坐标和模板匹配结果生成 AFSIM 代码 |
| `scenario_validate` | 四层想定验证并评分 |

## 8. 能力边界

编译器只放行白名单内的 AFSIM 能力，需求超出白名单时会在“表达不了的需求”里列出，而不是静默生成一个看似正确的想定。

| 类别 | 已放行 |
|---|---|
| 传感器 | `WSF_GEOMETRIC_SENSOR`（按最大探测距离的几何探测） |
| 运动模型 | `WSF_AIR_MOVER`（已验证）；`WSF_SURFACE_MOVER`、`WSF_GROUND_MOVER`（已放行，未经真机验证） |
| 武器 | `WSF_IMPLICIT_WEAPON`，按 `fire_time_s` 定时开火 |
| 断言事件 | `SENSOR_DETECTION_ATTEMPT`、`SENSOR_TRACK_INITIATED`、`WEAPON_FIRED` |

当前不支持、会被标记为能力缺口的内容：电子战/干扰、ESM 无源侦收、通信/数据链、导弹飞行与制导、雷达方程级建模、空间轨道运动、水声探测、自主决策/行为树、燃油与空中加油。

使用时需要注意：

- **武器是即时结算的。** `WSF_IMPLICIT_WEAPON` 配合 `AUTO_ENGAGEMENT_EFFECT`（100 km 球形毁伤，伤害 ≥ 0.99），开火即判定，没有导弹飞行时间。“A 开火后 B 延迟若干秒还击”这类设计中，B 可能在还击前已被判毁。
- **没有编队字段。** 编队只能通过平台命名、初始站位和 COA 任务分配体现，没有长僚机跟随逻辑。
- **开火只看时间。** 不会根据“是否探测到目标”决定开火，需要自己按初始距离和接近速度推算开火时刻。
- **只做作战仿真。** 交通、经济、社会等非作战领域的需求会在入口被直接拒绝。

## 9. 常见问题

| 现象 | 原因与处理 |
|---|---|
| 报“模型输出达到 max_tokens 上限被截断” | 平台多或使用推理模型时输出预算不够，调大 `--max-tokens`（16 个平台建议 32768） |
| 请求超时 | 推理模型单次响应较慢，调大 `--timeout`（如 600） |
| 返回“超出适用范围” | 需求命中非作战领域特征词，改写需求 |
| “未能收敛成合法想定” | 查看打印的各轮错误，通常是缺少兵力构成、地理位置或时间窗，或用到了未放行的能力 |
| 提示“请先设置 DEEPSEEK_API_KEY” | 未导出环境变量 |

## 10. 测试与基准

```bash
# 离线单元/回归测试 (不调用大模型)
.venv/bin/python -m pytest -q tests/tools/test_afsim_*.py tests/tools/test_nl_scenario_*.py

# 端到端 smoke: 收敛性、lint/验证器、重复编译确定性
.venv/bin/python scripts/smoke_nl_to_afsim.py --keep

# 通用性基准: 覆盖空/海/陆/天多域与不同信息完整度, 标记静默退化
.venv/bin/python scripts/bench_nl_generality.py --workers 4 --out bench_result.json
```

`tests/tools/test_afsim_scenario_regression.py` 中依赖真实 AFSIM 语料的用例在语料目录不存在时自动跳过。
