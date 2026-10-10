"""自然语言 → ScenarioIR → AFSIM 想定文件。

这是 nanobot 把"人话规划"变成可执行想定的入口。设计上刻意把不确定性和确定性
切开:

    自然语言规划
        │  ← LLM 只做这一步 (语义抽取), 输出强类型 ScenarioIR JSON
        ▼
    ScenarioIR  ──► Pydantic 校验 + 能力白名单
        │  ← 这一步之后完全确定性, 不再调用大模型
        ▼
    AFSIM 文本  ──► lint + 四层验证
        │
        ▼
    想定文件 (.txt)

为什么不让 LLM 直接写 AFSIM 文本: AFSIM 语法细节多 (块配对、单位、d:m:s 坐标、
定义 vs 引用), 让模型直接生成会产生大量语法级幻觉, 而且同一段自然语言两次生成
结果不一致, 无法复现。改成让模型只填一份 schema 受约束的 IR, 语法正确性由模板
渲染保证 —— 模型可能"想错"(战术不合理), 但不会"写错"(语法不合法)。

修复循环也放在 IR 层而不是文本层: 校验/编译/lint 的报错回灌给模型去改 IR,
而不是用正则去补 ``end_platform``。文本层补丁能把分数改好看, 却治不了根因,
还容易把本来正确的代码改坏。
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from nanobot.afsim import (
    DEFAULT_CAPABILITY_REGISTRY,
    AfsimCompiler,
    CapabilityKind,
    CapabilityRegistry,
    CompileError,
    ScenarioIR,
    ValidationStatus,
    check_geometry,
    lint_afsim,
    write_project,
)
from nanobot.agent.tools.base import Tool


# ═══════════════════════════════════════════════════════
# 结果数据结构
# ═══════════════════════════════════════════════════════


@dataclass
class PlanAttempt:
    """一次 IR 抽取尝试及其失败原因。"""
    round: int
    stage: str            # "extract" | "repair" | "refine"
    ok: bool
    errors: list[str] = field(default_factory=list)
    advisories: list[str] = field(default_factory=list)
    raw_excerpt: str = ""


@dataclass
class PlanResult:
    """自然语言 → ScenarioIR 的结果。"""
    success: bool
    scenario_ir: dict[str, Any] | None = None
    afsim_code: str = ""
    attempts: list[PlanAttempt] = field(default_factory=list)
    lint_diagnostics: list[str] = field(default_factory=list)
    advisories: list[str] = field(default_factory=list)
    capability_gaps: list["CapabilityGap"] = field(default_factory=list)
    unverified_capabilities: list[str] = field(default_factory=list)
    normalizations: list[str] = field(default_factory=list)
    out_of_scope: str = ""

    @property
    def rounds(self) -> int:
        return len(self.attempts)



# ═══════════════════════════════════════════════════════
# Prompt 构建
# ═══════════════════════════════════════════════════════


def _enabled_symbols(registry: CapabilityRegistry, kind: CapabilityKind) -> list[str]:
    """列出某类能力里编译器真正放行的 AFSIM 符号。

    从能力登记表实时推导而不是写死在 prompt 里: 白名单一扩展, 提示词自动跟着变,
    不会出现"提示词允许但编译器拒绝"的错配。
    """
    return sorted(
        capability.afsim_symbol
        for capability in registry.all()
        if capability.kind is kind and capability.compiler_enabled
    )


# ScenarioIR 的紧凑说明。
#
# 这里手写而不是直接塞 model_json_schema(): pydantic 生成的 schema 带大量 $defs
# 和 anyOf, 对模型来说噪声太大。手写版本可能与真实 schema 漂移, 兜底靠的是
# 后面的 pydantic 校验 + 报错回灌修复。
_IR_SHAPE = """{
  "scenario_id": "snake_case_id",
  "mission": {
    "mission_id": "letters_digits_underscore_starting_with_letter",
    "name": "任务名",
    "objective": "一句话任务目标",
    "duration_s": 3600.0,
    "random_seed": 1,
    "platforms": [
      {
        "platform_id": "blue_fighter_1",
        "side": "blue|red|neutral",
        "domain": "air|land|sea|space",
        "type_hint": "自然语言机型/装备描述, 如 J-16 电子战机",
        "initial_position": {"latitude_deg": 27.5, "longitude_deg": 122.5, "altitude_m": 8000.0},
        "route": [{"latitude_deg": 27.8, "longitude_deg": 122.2, "altitude_m": 8000.0}],
        "capabilities": ["sead", "jamming"]
      }
    ],
    "assumptions": ["模型不确定处的显式假设"]
  },
  "selected_coa": {
    "coa_id": "coa_1",
    "mission_id": "必须与 mission.mission_id 完全相同",
    "summary": "行动方案概述",
    "assignments": [{"platform_id": "blue_fighter_1", "task": "压制敌防空雷达", "priority": 80}],
    "rationale": ["为什么这么打"]
  },
  "sensors": [
    {
      "sensor_type_id": "SEARCH_RADAR",
      "base_type": "WSF_GEOMETRIC_SENSOR",
      "mode_name": "search",
      "frame_time_s": 1.0,
      "maximum_range_m": 150000.0,
      "reports_location": true,
      "ignore_same_side": true
    }
  ],
  "platforms": [
    {
      "platform_id": "与 mission.platforms 里的 platform_id 一一对应",
      "platform_type": "WSF_PLATFORM",
      "side": "blue|red|neutral",
      "position": {"latitude_deg": 27.5, "longitude_deg": 122.5, "altitude_m": 8000.0},
      "heading_deg": 90.0,
      "mover_type": "WSF_AIR_MOVER",
      "sensors": [{"instance_name": "radar", "sensor_type_id": "SEARCH_RADAR", "initial_mode": "search", "enabled": true}],
      "weapons": [{"instance_name": "arm", "weapon_type": "WSF_IMPLICIT_WEAPON", "quantity": 4, "update_interval_s": 1.0}],
      "route": [{"position": {"latitude_deg": 27.8, "longitude_deg": 122.2, "altitude_m": 8000.0}, "speed_mps": 250.0, "heading_deg": 90.0}]
    }
  ],
  "assertions": [
    {
      "assertion_id": "detect_sam",
      "event_type": "SENSOR_DETECTION_ATTEMPT|SENSOR_TRACK_INITIATED|WEAPON_FIRED",
      "source_platform": "blue_fighter_1",
      "target_platform": "red_sam_1",
      "sensor_name": "radar",
      "detected_only": false,
      "minimum_count": 1
    }
  ],
  "engagements": [
    {
      "engagement_id": "strike_sam",
      "shooter_platform": "blue_fighter_1",
      "target_platform": "red_sam_1",
      "weapon_name": "arm",
      "fire_time_s": 1200.0
    }
  ]
}"""

_HARD_RULES = """强约束 (违反任何一条都会被校验器打回):
1. 未列出的字段一律不要出现 —— schema 是 extra="forbid", 多写一个字段就整份失败。
2. selected_coa.mission_id 必须等于 mission.mission_id。
3. mission.platforms 与顶层 platforms 的 platform_id 集合必须完全相同 (不多不少)。
4. platform_id / mission_id / sensor_type_id / instance_name / *_id 只能用
   字母开头 + 字母数字下划线连字符, 不要中文、空格、点号。
5. 顶层 platforms[].sensors[].sensor_type_id 必须在 sensors[] 里定义过。
6. assertions / engagements 引用的平台必须存在于 platforms 里;
   engagements.weapon_name 必须是该 shooter 平台 weapons[] 里的 instance_name。
7. engagements.fire_time_s 必须 <= mission.duration_s。
8. 有 route 的平台必须同时给出 mover_type。
9. 纬度 [-90,90], 经度 [-180,180], 高度 [-500,100000] 米, heading_deg [0,360)。
10. sensors 至少 1 个, platforms 至少 1 个, 否则编译器直接拒绝。
11. 数值一律用 SI 单位: 距离米, 速度米/秒, 时间秒。不要写 "150km" 这种带单位的字符串。
12. 中文只允许出现在 name / objective / summary / task / rationale / assumptions /
    type_hint 这些自由文本字段里。"""


def _build_system_prompt(registry: CapabilityRegistry) -> str:
    sensors = _enabled_symbols(registry, CapabilityKind.SENSOR_TYPE)
    movers = _enabled_symbols(registry, CapabilityKind.MOVER_TYPE)
    weapons = _enabled_symbols(registry, CapabilityKind.WEAPON_TYPE)
    events = _enabled_symbols(registry, CapabilityKind.EVENT_TYPE)
    sentinel = _OUT_OF_SCOPE_SENTINEL

    return f"""你是 AFSIM 想定构建器的语义抽取前端。任务: 把用户的自然语言作战规划
转成一份严格符合 ScenarioIR schema 的 JSON。

你只负责语义抽取, 不要输出任何 AFSIM 语法 —— AFSIM 文本由下游确定性编译器渲染。

ScenarioIR 结构 (注释仅供理解, 输出时不要带注释):
{_IR_SHAPE}

{_HARD_RULES}

当前编译器放行的 AFSIM 类型 (只能用这些, 用别的会被能力白名单拒绝):
- sensors[].base_type: {', '.join(sensors) or '(无)'}
- platforms[].mover_type: {', '.join(movers) or '(无)'}
- platforms[].weapons[].weapon_type: {', '.join(weapons) or '(无)'}
- assertions[].event_type: {', '.join(e for e in events if not e.startswith('EVENT_PIPE')) or '(无)'}

**mover 必须与平台所在域一致** (这条最容易错):
- domain="air"  → mover_type="WSF_AIR_MOVER"
- domain="sea"  → mover_type="WSF_SURFACE_MOVER", 且 altitude_m 必须是 0 (船浮在海面)
- domain="land" → mover_type="WSF_GROUND_MOVER", altitude_m 用地形海拔
- domain="space" → **不支持轨道运动**: 不要给 mover_type, 不要给 route,
  建成静止平台, 并在 assumptions 里写明轨道运动未被建模
- 固定目标 (雷达阵地、发射阵地、机场) 一律不给 mover_type 和 route
给舰船套 WSF_AIR_MOVER 会让船"飞起来", 这类错误会被 schema 直接打回。

**给了 mover_type 就必须给 route** (至少一个航路点, 并带 speed_mps)。
只挂 mover 不给航路的平台在仿真里原地不动, 等于白挂 —— 会被 schema 打回。
会动的东西 (战斗机、军舰、坦克) 都要写出它从哪飞/开到哪; 真正不动的
(雷达阵地、发射阵地、机场) 则**连 mover_type 都不要写**。

`speed_mps` 必须严格大于 0。要表达"平台不动"就直接省略 mover_type 和 route,
**不要**写一条 speed_mps 为 0 的航路 —— 那会被 schema 打回。

**航路要能在任务时长内走完**: 航路总长 ÷ 速度 必须小于 duration_s, 否则后段
航路根本走不到。1 个纬度约 111 km, 中纬度 1 个经度约 100 km; 18 节约 9 m/s,
巡航 250 m/s。

**探测几何必须成立**: 写了探测断言, 就要保证观察方与目标在想定期间真的能进入
传感器 maximum_range_m 之内 —— 要么把双方摆近, 要么用航路让它飞进探测圈,
要么把 maximum_range_m 设得足够大。射程 150 km 却让目标停在 400 km 外,
断言永远不会触发, 任务目标也就永远达不成。

建模要求:
- 把规划里出现的每一方兵力都建成平台; 数量词 (如"4架J-16") 展开成 4 个独立平台,
  platform_id 用 blue_j16_1 ... blue_j16_4 这种可读命名。
- 蓝方用 side="blue", 红方/敌方用 side="red", 中立用 "neutral"。
- 同型编队不要放在同一个坐标点, 相隔 0.05-0.2 度。
- 只要规划里提到"探测/发现/预警", 就为对应平台配 sensors 并写一条
  SENSOR_DETECTION_ATTEMPT 或 SENSOR_TRACK_INITIATED 断言。
- 只要提到"打击/发射/摧毁", 就给射手配 weapons 并写一条 engagements, 发射时刻
  要落在任务时长内且晚于双方接近到传感器可探测的时间。
- 传感器 maximum_range_m 要大于它实际需要探测的距离, 否则断言永远不会成立
  (1 个纬度约 111 km, 1 个经度在中纬度约 100 km)。
- 规划没说的参数自己按常识补, 并在 mission.assumptions 里逐条写明补了什么。
- **表达不了的需求点要如实写进 assumptions**, 不要用相近字段假装实现。

如果需求根本不属于 AFSIM 作战/航空航天交战仿真 (例如交通客流、经济、社会仿真),
不要硬编一个想定, 直接输出: {{"{sentinel}": "说明为什么超出范围"}}

只输出 JSON 对象本身, 不要 markdown 代码块, 不要任何解释文字。"""


# ═══════════════════════════════════════════════════════
# 能力缺口侦测 (确定性, 不依赖模型自述)
# ═══════════════════════════════════════════════════════


@dataclass(frozen=True)
class CapabilityGap:
    """需求里提到、但当前 IR/编译器表达不了的能力。"""
    topic: str
    matched: tuple[str, ...]
    why: str
    workaround: str


# 关键词 → 能力缺口。
#
# 为什么要做这层确定性侦测: LLM 面对"实施阻塞干扰"这类超出 IR 表达范围的需求时,
# 倾向于挑一个最接近的字段糊过去 —— 实测中它把几何传感器命名成 ESM_SENSOR 就
# 交差了, 还在 assumptions 里写"使用默认干扰频段", 而想定里根本没有任何干扰。
# 这种"语法全绿 + 评分 100 + 需求没实现"的静默退化, 比直接报错危险得多。
#
# 所以缺口判定不能交给模型自述, 必须由代码扫原始需求文本得出。
_GAP_RULES: tuple[tuple[tuple[str, ...], CapabilityGap], ...] = (
    (
        ("干扰", "压制敌雷达", "阻塞", "jam", "jamming", "电子战", "electronic warfare", "ecm"),
        CapabilityGap(
            topic="电子战/干扰",
            matched=(),
            why="编译器未放行干扰机与电子攻击模型, IR 里没有可表达干扰功率/频段/压制效果的字段",
            workaround="想定只会保留平台与传感器, 干扰效果不会被仿真; 如需干扰须先扩能力白名单",
        ),
    ),
    (
        ("esm", "侦收", "辐射源定位", "测向", "无源探测"),
        CapabilityGap(
            topic="ESM 无源侦收",
            matched=(),
            why="WSF_ESM_SENSOR 未放行; 几何传感器只有距离门限, 无法表达接收频段与辐射源测向",
            workaround="会退化成普通几何探测, 不体现无源/测向语义",
        ),
    ),
    (
        ("数据链", "link-16", "link16", "航迹分发", "共享航迹", "组网", "datalink", "data link"),
        CapabilityGap(
            topic="通信/数据链",
            matched=(),
            why="WSF_COMM_TRANSCEIVER 与航迹分发处理器未放行, IR 无通信链路字段",
            workaround="各平台传感器独立工作, 不会有航迹共享与协同引导",
        ),
    ),
    (
        ("弹道", "飞行轨迹", "制导", "拦截弹", "命中判定", "毁伤评估", "飞出", "trajectory", "guidance"),
        CapabilityGap(
            topic="导弹飞行与制导",
            matched=(),
            why="只放行了 WSF_IMPLICIT_WEAPON (按时刻记一次发射事件), 未放行 WSF_EXPLICIT_WEAPON + WSF_GUIDED_MOVER",
            workaround="只会产生 WEAPON_FIRED 事件, 没有弹体飞行、制导与命中/毁伤过程",
        ),
    ),
    (
        ("波段", "峰值功率", "天线增益", "rcs", "平方米", "雷达方程", "噪声系数", "脉冲"),
        CapabilityGap(
            topic="雷达方程级建模",
            matched=(),
            why="WSF_RADAR_SENSOR 与 radar_signature 未放行; 几何传感器只有 maximum_range, 无功率/增益/RCS",
            workaround="给定的雷达参数会被折算成一个等效最大探测距离, 细节丢失",
        ),
    ),
    (
        ("轨道", "卫星", "过境", "开普勒", "orbit", "satellite"),
        CapabilityGap(
            topic="空间轨道运动",
            matched=(),
            why="WSF_SPACE_MOVER 用开普勒根数, 与本 IR 的经纬高航点模型不同构, 故未放行",
            workaround="空间平台只能建成静止平台; 轨道运动与过境时序不会被仿真",
        ),
    ),
    (
        ("声呐", "声纳", "水声", "sonar", "acoustic"),
        CapabilityGap(
            topic="水声探测",
            matched=(),
            why="WSF_ACOUSTIC_SENSOR 未放行",
            workaround="会退化成几何探测, 不体现水声传播",
        ),
    ),
    (
        ("行为树", "自主决策", "任务处理器", "behavior tree", "自主火控"),
        CapabilityGap(
            topic="自主决策/行为树",
            matched=(),
            why="processor / behavior_tree 未放行; 当前只支持按绝对时刻触发的定时动作",
            workaround="所有动作都是预设时刻执行, 平台不会根据战场态势自主决策",
        ),
    ),
    (
        ("加油", "空中加油", "refuel", "挂载消耗", "油量"),
        CapabilityGap(
            topic="燃油与空中加油",
            matched=(),
            why="fuel 模型未放行, IR 无油量字段",
            workaround="平台不消耗燃油, 加油环节不会被仿真",
        ),
    ),
)

# 明显不属于 AFSIM 作战/航空航天仿真域的需求特征。
#
# 实测里"城市早高峰地铁客流仿真"被硬编成了一个中立方平台 + 传感器的想定 —— 模型
# 过度顺从, 宁可编也不肯说不会。这类需求必须在入口挡掉。
_OUT_OF_SCOPE_HINTS: tuple[str, ...] = (
    "地铁", "客流", "公交", "红绿灯", "物流仓储", "电商", "股票", "股价", "疫情",
    "选课", "考试成绩", "推荐系统", "工厂流水线", "供应链", "打车", "外卖",
)

_OUT_OF_SCOPE_SENTINEL = "__out_of_scope__"


def detect_capability_gaps(request: str) -> list[CapabilityGap]:
    """扫描自然语言需求, 列出当前表达不了的能力点。"""
    lowered = request.lower()
    gaps: list[CapabilityGap] = []
    for keywords, gap in _GAP_RULES:
        hits = tuple(k for k in keywords if k in lowered)
        if hits:
            gaps.append(
                CapabilityGap(
                    topic=gap.topic, matched=hits, why=gap.why, workaround=gap.workaround
                )
            )
    return gaps


def looks_out_of_scope(request: str) -> list[str]:
    """返回命中的非作战仿真域特征词; 空列表表示看起来在范围内。"""
    lowered = request.lower()
    return [hint for hint in _OUT_OF_SCOPE_HINTS if hint in lowered]


# ═══════════════════════════════════════════════════════
# 确定性归一化 (答案唯一的机械修正)
# ═══════════════════════════════════════════════════════


def normalize_ir_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """对答案唯一的机械问题做确定性修正, 返回 (新 payload, 修正说明)。

    这里只处理**真正只有一个正确答案**的问题:

    1. 水面平台高度必须是 0 —— 船浮在海面上。
    2. ``type_hint`` 明确是固定设施 (雷达阵地/机场/发射井) 的平台不该有 mover ——
       平台身份决定了它不动, 没有第二种解法。
    3. ``platform_type`` 填了默认值 ``WSF_PLATFORM`` 时, 从 ``type_hint`` 推机型缩写
       (轰-6K → H6K, 空警-500 → KJ500, 054A → TYPE_054A, 巡洋舰 → CRUISER) ——
       多文件项目里 ``platforms/<type_key>.txt`` 的文件名直接来自这个字段, 全用默认值
       会让 9 个平台挤进同一个 ``WSF_PLATFORM.txt``, 失去类型分家的意义。

    刻意**不**在这里处理"有 mover 却没航路": 那个问题有两个合理答案 (固定阵地该删
    mover, 要拦截的战斗机该补航路), 需要需求语义, 交给模型逐平台决定。我一度让代码
    统一删 mover, 结果把数据链协同、空中对抗想定里所有飞机都变成静止平台。

    也不在这里处理"航路走不完": 缩短航路 / 提高速度 / 延长时长三种解法都合理。

    第 2 条是上一轮补的。第 3 条是这一轮补的: 实测中 LLM 抽 IR 时 9 个平台全填了
    ``WSF_PLATFORM`` (schema 默认值), 导致多文件项目里所有平台挤进同一个类型文件。
    归一化层按 ``type_hint`` 推机型缩写, 比让模型碰运气可靠。

    修正过程必须写进输出 (normalizations), 不能静默改动使用者的 IR。
    """
    result = copy.deepcopy(payload)
    notes: list[str] = []

    platforms = result.get("platforms")
    if not isinstance(platforms, list):
        return result, notes

    domain_by_id: dict[str, str] = {}
    hint_by_id: dict[str, str] = {}
    mission = result.get("mission")
    if isinstance(mission, dict):
        for intent in mission.get("platforms", []) or []:
            if isinstance(intent, dict):
                platform_id = intent.get("platform_id")
                if isinstance(platform_id, str):
                    domain_by_id[platform_id] = intent.get("domain", "")
                    hint_by_id[platform_id] = str(intent.get("type_hint", ""))

    for platform in platforms:
        if not isinstance(platform, dict):
            continue
        platform_id = platform.get("platform_id", "?")

        # 固定设施不该会动: 平台身份唯一决定该删 mover
        if platform.get("mover_type") and classify_mobility(hint_by_id.get(platform_id, "")) == "fixed":
            dropped = platform.pop("mover_type")
            had_route = bool(platform.get("route"))
            platform["route"] = []
            notes.append(
                f"平台 {platform_id} 被描述为固定设施 (\"{hint_by_id.get(platform_id, '')}\"), "
                f"已删除 mover_type={dropped}"
                + ("与航路" if had_route else "")
            )

        # platform_type 填了默认值时, 从 type_hint 推机型缩写
        if platform.get("platform_type") == "WSF_PLATFORM":
            inferred = _infer_platform_type(hint_by_id.get(platform_id, ""))
            if inferred:
                platform["platform_type"] = inferred
                notes.append(
                    f"平台 {platform_id} 的 platform_type 从默认值 WSF_PLATFORM 推断为 "
                    f"{inferred} (基于 type_hint=\"{hint_by_id.get(platform_id, '')}\")"
                )

        if domain_by_id.get(platform_id) == "sea":
            position = platform.get("position")
            if isinstance(position, dict) and position.get("altitude_m", 0) not in (0, 0.0):
                original = position["altitude_m"]
                position["altitude_m"] = 0.0
                notes.append(
                    f"水面平台 {platform_id} 高度 {original}m 已归零 (舰船位于海平面)"
                )
            for point in platform.get("route", []) or []:
                if not isinstance(point, dict):
                    continue
                point_position = point.get("position")
                if isinstance(point_position, dict) and point_position.get("altitude_m", 0) not in (0, 0.0):
                    point_position["altitude_m"] = 0.0

    return result, notes


# ═══════════════════════════════════════════════════════
# 平台机动性分类 (按 type_hint 文本)
# ═══════════════════════════════════════════════════════

# 固定设施: 这类平台不该挂 mover
#
# 实测中"一部地面雷达站负责搜索预警"生成出来的雷达站被赋了 WSF_GROUND_MOVER +
# 一个航路点 —— schema 层完全合法 (有 mover 就有 route), 但雷达站会跑就荒谬了。
# 光靠提示词约束不住, 得在 IR 层按 type_hint 语义查。
_FIXED_INSTALLATION_HINTS: tuple[str, ...] = (
    "雷达站", "雷达阵地", "阵地", "发射阵地", "发射场", "发射井",
    "机场", "空军基地", "地面站", "指挥所", "观测站", "哨所", "炮位",
    "radar site", "radar station", "launch site", "airfield", "airbase",
    "air base", "emplacement", "silo", "bunker", "installation",
)

# 机动平台: 这类平台该挂 mover + route
#
# 刻意用多字词而不是单字"机": 单字会把"机场"也匹配上。
_MOBILE_ASSET_HINTS: tuple[str, ...] = (
    "战斗机", "轰炸机", "预警机", "加油机", "电子战机", "运输机", "侦察机",
    "直升机", "无人机", "歼", "苏-", "su-", "f-16", "f-35",
    "驱逐舰", "巡洋舰", "护卫舰", "航母", "舰艇", "军舰", "潜艇", "快艇",
    "坦克", "装甲车", "步战车", "导弹车", "发射车",
    "fighter", "bomber", "awacs", "tanker", "aircraft", "jet",
    "destroyer", "cruiser", "frigate", "carrier", "submarine",
    "tank", "vehicle", "helicopter", "uav", "drone",
)

# 移动性修饰: 命中后即便看着像固定设施也不判定为固定 (如"移动式红旗-22")
_MOBILITY_OVERRIDE_HINTS: tuple[str, ...] = (
    "移动", "机动", "车载", "自行", "可移动",
    "mobile", "self-propelled", "truck-mounted",
)


def classify_mobility(type_hint: str) -> str:
    """按平台的自然语言描述判断它该不该动: fixed / mobile / unknown。

    判定顺序上固定设施优先, 但被移动性修饰词否决。返回 unknown 表示看不出来,
    这时不做任何主张 —— 这条检查是启发式的, 宁可不报也不要误报。
    """
    text = type_hint.lower()
    if any(hint in text for hint in _MOBILITY_OVERRIDE_HINTS):
        return "mobile" if any(h in text for h in _MOBILE_ASSET_HINTS) else "unknown"
    if any(hint in text for hint in _FIXED_INSTALLATION_HINTS):
        return "fixed"
    if any(hint in text for hint in _MOBILE_ASSET_HINTS):
        return "mobile"
    return "unknown"


# 中文军机/舰船名 → AFSIM platform_type 缩写。
# 只覆盖常用机型, 认不出就返回 None, 让 IR 保留原值 (WSF_PLATFORM)。
# 缩写规则: 尽量用国际通用的北约代号 / 拼音首字母, 与官方 demo 命名习惯对齐。
_PLATFORM_TYPE_ABBREVIATIONS: list[tuple[list[str], str]] = [
    # 轰炸机
    (["轰-6", "轰6", "h6"], "H6K"),
    (["轰-20", "轰20", "h20"], "H20"),
    (["歼轰"], "JH"),
    # 战斗机
    (["歼-20", "歼20", "j-20", "j20"], "J20"),
    (["歼-16", "歼16"], "J16"),
    (["歼-15", "歼15"], "J15"),
    (["歼-10", "歼10"], "J10"),
    (["苏-30", "苏30"], "SU30"),
    (["苏-35", "苏35"], "SU35"),
    # 预警/特种机
    (["空警-500", "空警500", "kj-500", "kj500"], "KJ500"),
    (["空警-2000", "空警2000"], "KJ2000"),
    (["运-20", "运20"], "Y20"),
    (["运-8", "运8"], "Y8"),
    # 直升机
    (["直-20", "直20"], "Z20"),
    (["直-10", "直10"], "Z10"),
    # 无人机
    (["翼龙", "yilong"], "YL"),
    (["彩虹", "ch"], "CH"),
    (["无侦", "wz"], "WZ"),
    # 水面舰艇: 具体型号/舰级在前, 通称在后 ("伯克级驱逐舰" 该匹配 "伯克" 而不是 "驱逐舰")
    (["054a"], "TYPE_054A"),
    (["054b"], "TYPE_054B"),
    (["052d"], "TYPE_052D"),
    (["052c"], "TYPE_052C"),
    (["055"], "TYPE_055"),
    (["航母"], "CVN"),
    (["两栖攻击舰"], "LHD"),
    # 美军平台 (中文语境里常见) —— 必须放在 "驱逐舰"/"巡洋舰" 通称之前
    (["伯克", "阿利伯克"], "DDG_51"),
    (["提康德罗加"], "CG_47"),
    (["尼米兹"], "CVN_68"),
    (["福特"], "CVN_78"),
    (["f-22", "f22"], "F22"),
    (["f-35", "f35"], "F35"),
    (["f/a-18", "fa-18", "大黄蜂"], "FA18"),
    (["b-2", "b2"], "B2"),
    (["b-52", "b52"], "B52"),
    (["b-21"], "B21"),
    # 舰级通称 (中文) —— 放最后, 作为后备
    (["巡洋舰"], "CRUISER"),
    (["驱逐舰"], "DESTROYER"),
    (["护卫舰"], "FRIGATE"),
    (["潜艇"], "SUBMARINE"),
    # 雷达/固定设施 (用于生成更可读的类型名, 虽然它们常走固定设施分支删掉 mover)
    (["雷达阵地", "雷达站"], "RADAR_SITE"),
    (["机场"], "AIRFIELD"),
    (["发射井"], "SILO"),
    (["发射阵地"], "LAUNCH_SITE"),
]


def _infer_platform_type(type_hint: str) -> str | None:
    """从 type_hint 推 platform_type 缩写。

    只处理 LLM 填了默认值 ``WSF_PLATFORM`` 的情况。返回 None 表示认不出,
    调用方保留原值。缩写表按条目顺序匹配, 先中者胜 —— 所以"轰-6K电子战型"
    会匹配到 H6K 而不是被后面的"电子战"抢走。

    为什么用查表而不是让模型碰运气: 同一个 type_hint 两次生成结果不一致,
    无法复现。查表是确定性的, 而且缩写规则与官方 demo 命名习惯对齐
    (H6K / KJ500 / TYPE_054A / DDG_51 都是 AFSIM 语料里常见的平台类型名)。

    移动性修饰词 ("移动式"/"车载"/"自行") 优先于固定设施关键字: "移动式红旗-22
    发射阵地" 是能跑的, 不该匹配到 "发射阵地" -> LAUNCH_SITE。与 ``classify_mobility``
    的 ``_MOBILITY_OVERRIDE_HINTS`` 保持一致的优先级。
    """
    if not type_hint:
        return None
    text = type_hint.lower()
    # 移动性修饰词否决固定设施表: 与 classify_mobility 的优先级一致
    if any(hint in text for hint in _MOBILITY_OVERRIDE_HINTS):
        return None
    for keywords, abbreviation in _PLATFORM_TYPE_ABBREVIATIONS:
        if any(kw in text for kw in keywords):
            return abbreviation
    return None


# ═══════════════════════════════════════════════════════
# IR 层战术合理性检查
# ═══════════════════════════════════════════════════════


def check_ir_plausibility(scenario: ScenarioIR) -> list[str]:
    """IR 层战术合理性 + 几何可达性检查, 返回可读的问题清单。

    这些问题都不违反 schema、也不影响 AFSIM 语法, 因此编译器和 linter 都不会报,
    但它们会让想定"跑得起来却没有意义" —— 典型的是多个平台压在同一个坐标点、
    断言引用了平台上不存在的传感器, 或者传感器射程根本够不到目标。放在 IR 层
    检查的好处是: 此时还知道每个字段的语义, 一旦落成 AFSIM 文本就只剩字符串了。
    """
    issues: list[str] = []

    # 1. 位置完全重合: 多个平台叠在一个点上, 探测/交战几何失去意义
    by_position: dict[tuple[float, float, float], list[str]] = {}
    for platform in scenario.platforms:
        key = (
            round(platform.position.latitude_deg, 6),
            round(platform.position.longitude_deg, 6),
            round(platform.position.altitude_m, 3),
        )
        by_position.setdefault(key, []).append(platform.platform_id)
    for (lat, lon, alt), members in by_position.items():
        if len(members) > 1:
            issues.append(
                f"平台 {', '.join(sorted(members))} 初始位置完全相同 "
                f"({lat}, {lon}, {alt}m); 同型编队请拉开间距 (例如相隔 0.05-0.2 度)"
            )

    # 注: "有 mover 却没航路"已提升为 PlatformIR 的 schema 硬错误 (mover_requires_route),
    # 走不到这里, 因此这里不再重复检查。

    # 2. 断言引用了平台上不存在的传感器: 该断言永远不会被触发
    sensors_by_platform = {
        platform.platform_id: {sensor.instance_name for sensor in platform.sensors}
        for platform in scenario.platforms
    }
    for assertion in scenario.assertions:
        if assertion.sensor_name and assertion.source_platform:
            available = sensors_by_platform.get(assertion.source_platform, set())
            if assertion.sensor_name not in available:
                issues.append(
                    f"断言 {assertion.assertion_id} 引用 {assertion.source_platform} 上的传感器 "
                    f"'{assertion.sensor_name}', 但该平台只有 {sorted(available) or '(无传感器)'}"
                )

    # 4. 同阵营探测断言: 传感器默认 ignore_same_side, 这类断言不可能成立
    side_by_platform = {p.platform_id: p.side for p in scenario.platforms}
    for assertion in scenario.assertions:
        if not assertion.source_platform:
            continue
        source_side = side_by_platform.get(assertion.source_platform)
        target_side = side_by_platform.get(assertion.target_platform)
        if source_side is not None and source_side == target_side:
            issues.append(
                f"断言 {assertion.assertion_id} 的观察方与目标同属 {source_side.value} 阵营; "
                "传感器默认 ignore_same_side, 该断言不会触发"
            )

    # 5. 全场没有传感器: 想定不会产生任何探测事件
    if not any(platform.sensors for platform in scenario.platforms):
        issues.append("没有任何平台装备传感器, 想定不会产生探测事件; 请至少给一方配传感器")

    # 6. 射手一方完全没有传感器: 定时发射虽能执行, 但缺了探测-锁定环节
    for engagement in scenario.engagements:
        shooter_side = side_by_platform.get(engagement.shooter_platform)
        if shooter_side is None:
            continue
        if not any(p.sensors for p in scenario.platforms if p.side == shooter_side):
            issues.append(
                f"交战 {engagement.engagement_id} 的射手方 {shooter_side.value} 全无传感器, "
                "缺少探测-锁定环节; 建议给该方配一个预警/搜索传感器"
            )

    # 6. 几何可达性: 传感器射程能否真的够到目标、航路能否在任务时长内走完
    issues.extend(finding.message for finding in check_geometry(scenario))

    # 7. 机动性与平台身份是否自洽: 雷达站不该会跑, 战斗机不该钉在原地
    intent_by_id = {
        intent.platform_id: intent for intent in scenario.mission.platforms
    }
    for platform in scenario.platforms:
        intent = intent_by_id.get(platform.platform_id)
        if intent is None:
            continue
        expected = classify_mobility(intent.type_hint)
        if expected == "fixed" and platform.mover_type:
            issues.append(
                f"平台 {platform.platform_id} 被描述为固定设施 (\"{intent.type_hint}\"), "
                f"却挂了 mover_type={platform.mover_type} 并带航路; 固定设施应删掉 "
                f"mover_type 和 route"
            )
        elif expected == "mobile" and not platform.mover_type:
            issues.append(
                f"平台 {platform.platform_id} 被描述为机动装备 (\"{intent.type_hint}\"), "
                f"却没有 mover_type, 全程钉在原地; 请补 mover_type 和至少一个航路点"
            )

    return issues


# ═══════════════════════════════════════════════════════
# 规划器
# ═══════════════════════════════════════════════════════


class ScenarioIRPlanner:
    """把自然语言规划抽取成 ScenarioIR, 并在 IR 层做有界自修复。

    修复循环的每一轮都把"上一版 IR + 全部报错"回灌给模型, 让它整份重写。
    只做有界重试 (默认 3 轮): 结构化抽取如果三轮都过不了校验, 通常是需求本身
    缺信息, 再重试只是烧 token。
    """

    def __init__(
        self,
        llm: Any,
        registry: CapabilityRegistry = DEFAULT_CAPABILITY_REGISTRY,
        max_repair_rounds: int = 3,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.max_repair_rounds = max(1, max_repair_rounds)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.compiler = AfsimCompiler(capability_registry=registry)

    # ── 主流程 ────────────────────────────────────────

    def plan(self, request: str, hints: dict[str, Any] | None = None) -> PlanResult:
        # 入口范围闸门: 明显非作战仿真域的需求直接拒绝, 不浪费 API 也不编造想定
        civil_hits = looks_out_of_scope(request)
        if civil_hits:
            return PlanResult(
                success=False,
                out_of_scope=(
                    f"需求命中非作战仿真域特征词 {civil_hits}。本工具只生成 AFSIM "
                    f"作战/航空航天交战想定 (空海陆天平台、传感器探测、武器交战), "
                    f"不能用于交通、经济、社会等领域仿真。"
                ),
            )

        # 能力缺口: 由代码扫需求文本得出, 不依赖模型自述
        gaps = detect_capability_gaps(request)

        system_prompt = _build_system_prompt(self.registry)
        conversation: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user",
             "content": self._build_user_prompt(request, hints or {}, gaps)},
        ]

        attempts: list[PlanAttempt] = []
        best: PlanResult | None = None

        for round_index in range(1, self.max_repair_rounds + 1):
            if round_index == 1:
                stage = "extract"
            elif best is None:
                stage = "repair"   # 还没拿到合法 IR, 修硬错误
            else:
                stage = "refine"   # 已有合法 IR, 只优化战术合理性

            raw = self.llm.chat(
                conversation,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                json_object=True,
            )
            payload, parse_errors = _parse_json_object(raw)
            if payload is None:
                truncated = getattr(self.llm, "last_finish_reason", None) == "length"
                if truncated:
                    parse_errors = [
                        *parse_errors,
                        f"模型输出达到 max_tokens={self.max_tokens} 上限被截断; "
                        f"平台较多时请增大 max_tokens 后重试",
                    ]
                attempts.append(
                    PlanAttempt(round_index, stage, False, parse_errors, raw_excerpt=raw[:400])
                )
                if best is not None:
                    break  # 已有可用结果, 不为了 refine 冒风险继续
                if truncated:
                    break  # 同一预算下重新输出完整 IR 只会再次截断
                conversation += [
                    {"role": "assistant", "content": raw[:2000]},
                    {"role": "user", "content":
                        "上一轮输出不是合法 JSON 对象: " + "; ".join(parse_errors)
                        + "\n请只输出 JSON 对象本身, 重新给出完整的 ScenarioIR。"},
                ]
                continue

            # 模型自己判定超纲时会回 {"__out_of_scope__": "理由"}
            if _OUT_OF_SCOPE_SENTINEL in payload:
                return PlanResult(
                    success=False,
                    attempts=attempts,
                    capability_gaps=gaps,
                    out_of_scope=str(payload[_OUT_OF_SCOPE_SENTINEL]),
                )

            # 先做确定性归一化: 答案唯一的机械问题不必浪费修复轮次
            payload, normalizations = normalize_ir_payload(payload)

            scenario, afsim_code, errors = self._validate_and_compile(payload)
            if scenario is None:
                attempts.append(PlanAttempt(round_index, stage, False, errors))
                if best is not None:
                    break  # refine 把 IR 改坏了, 保留上一版合法结果
                conversation += [
                    {"role": "assistant",
                     "content": json.dumps(payload, ensure_ascii=False)[:6000]},
                    {"role": "user", "content": self._build_repair_prompt(errors)},
                ]
                continue

            advisories = check_ir_plausibility(scenario)
            lint = lint_afsim(afsim_code)
            candidate = PlanResult(
                success=True,
                scenario_ir=payload,
                afsim_code=afsim_code,
                lint_diagnostics=[
                    f"{d.severity.value}:{d.code}:{d.message}" for d in lint.diagnostics
                ],
                advisories=advisories,
                capability_gaps=gaps,
                unverified_capabilities=self._unverified_used(scenario),
                normalizations=normalizations,
            )
            attempts.append(PlanAttempt(round_index, stage, True, advisories=advisories))

            # 单调不退化: 合理性问题不增加就接受新版本。
            #
            # 用 <= 而不是 <: 很多改进不体现在条数上。实测里 refine 把"航路需要
            # 103 分钟"改成"62 分钟"(任务时长 60 分钟), 问题条数仍是 2, 用严格
            # 小于会把这个明显更好的版本丢掉。平手时取较新版本 —— 它多看了一轮
            # 反馈。条数变多才拒绝, 所以仍然不会退化。
            if best is None or len(advisories) <= len(best.advisories):
                best = candidate

            if not best.advisories or round_index == self.max_repair_rounds:
                break

            conversation += [
                {"role": "assistant",
                 "content": json.dumps(payload, ensure_ascii=False)[:6000]},
                {"role": "user", "content": self._build_refine_prompt(best.advisories)},
            ]

        if best is not None:
            best.attempts = attempts
            return best
        return PlanResult(success=False, attempts=attempts, capability_gaps=gaps)

    def _unverified_used(self, scenario: ScenarioIR) -> list[str]:
        """列出该想定用到、但尚未经 mission.exe 实跑验证的 AFSIM 能力。

        release-3 的纪律是"只有真实验证通过才启用编译权限"。本机没有 AFSIM 运行时,
        水面/地面 mover 属于放行但未验证, 必须在输出里如实标注, 不能让使用者以为
        静态检查全绿就等于真机跑得通。
        """
        used: list[tuple[CapabilityKind, str]] = [
            (CapabilityKind.SENSOR_TYPE, sensor.base_type) for sensor in scenario.sensors
        ]
        for platform in scenario.platforms:
            if platform.mover_type:
                used.append((CapabilityKind.MOVER_TYPE, platform.mover_type))
            for weapon in platform.weapons:
                used.append((CapabilityKind.WEAPON_TYPE, weapon.weapon_type))

        unverified: list[str] = []
        for kind, symbol in dict.fromkeys(used):
            capability = self.registry.get(kind, symbol)
            if capability is None:
                continue
            if capability.validation_status is not ValidationStatus.PASSED:
                unverified.append(
                    f"{symbol} (validation_status={capability.validation_status.value}, "
                    f"trust={capability.trust_level.value})"
                )
        return unverified

    # ── Prompt ────────────────────────────────────────

    def _build_user_prompt(
        self, request: str, hints: dict[str, Any], gaps: list[CapabilityGap]
    ) -> str:
        lines = ["以下是作战规划的自然语言描述, 请抽取成 ScenarioIR JSON。", "",
                 "=== 规划正文 ===", request.strip(), ""]
        constraints: list[str] = []
        if hints.get("center_lat") is not None and hints.get("center_lon") is not None:
            constraints.append(
                f"战区中心约在 {hints['center_lat']}°N {hints['center_lon']}°E, "
                "所有平台经纬度请落在该中心 ±3° 范围内"
            )
        if hints.get("duration_s"):
            constraints.append(f"mission.duration_s 用 {hints['duration_s']} 秒")
        if hints.get("scenario_id"):
            constraints.append(f"scenario_id 用 \"{hints['scenario_id']}\"")
        if constraints:
            lines += ["=== 附加约束 ===", *(f"- {c}" for c in constraints), ""]

        if gaps:
            # 提前告知缺口, 避免模型拿最接近的字段去"假装"实现
            lines += [
                "=== 已知无法表达的能力 (编译器白名单未放行) ===",
                "以下需求点当前的 IR 表达不了。**不要**用相近字段假装实现它们 ——",
                "比如不要把几何传感器改个名叫 ESM_SENSOR 就当成侦收做完了。",
                "正确做法: 把能建模的部分 (平台/航路/探测/定时发射) 如实建出来,",
                "然后在 mission.assumptions 里逐条写明哪个需求点未被建模。",
                "",
            ]
            for gap in gaps:
                lines.append(f"- {gap.topic}: {gap.why}")
            lines.append("")

        return "\n".join(lines)

    def _build_repair_prompt(self, errors: list[str]) -> str:
        listed = "\n".join(f"- {e}" for e in errors[:25])
        return (
            "上一版 ScenarioIR 没通过校验, 具体错误如下:\n"
            f"{listed}\n\n"
            "请逐条修掉这些错误, 然后输出**完整的**修正后 ScenarioIR JSON。\n"
            "注意: 只改报错涉及的地方, 其余内容保持不变; 不要输出差异或片段, "
            "要输出完整 JSON 对象。"
        )

    def _build_refine_prompt(self, advisories: list[str]) -> str:
        listed = "\n".join(f"- {a}" for a in advisories[:25])
        return (
            "这一版 ScenarioIR 已经通过 schema 校验并能编译成合法 AFSIM, 但存在"
            "以下战术合理性问题, 会让想定跑得起来却没有意义:\n"
            f"{listed}\n\n"
            "请只针对上述问题做最小修改 (调整坐标、补航路点、改传感器名等), "
            "保持任务意图、平台构成和 platform_id 不变, 然后输出**完整的** ScenarioIR JSON。"
        )

    # ── 校验 + 编译 ───────────────────────────────────

    def _validate_and_compile(
        self, payload: dict[str, Any]
    ) -> tuple[ScenarioIR | None, str, list[str]]:
        """跑完 pydantic 校验 + 能力白名单 + 编译 + lint。

        返回 ``(scenario, afsim_code, errors)``; 任一环节失败则 scenario 为 None,
        errors 汇总全部可回灌给模型的报错。
        """
        try:
            scenario = ScenarioIR.model_validate(payload)
        except ValidationError as exc:
            return None, "", _format_validation_errors(exc)

        try:
            afsim_code = self.compiler.compile(scenario)
        except CompileError as exc:
            return None, "", [f"编译器拒绝: {exc}"]

        lint = lint_afsim(afsim_code)
        if not lint.is_valid:
            return None, "", [
                f"生成的 AFSIM 文本静态检查未通过 (第 {d.line} 行 {d.code}): {d.message}"
                for d in lint.diagnostics
                if d.severity.value == "error"
            ]
        return scenario, afsim_code, []


# ═══════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _parse_json_object(raw: str) -> tuple[dict[str, Any] | None, list[str]]:
    """从模型输出里取出 JSON 对象。

    即使开了服务端 JSON 模式, 模型偶尔还是会包一层 markdown 代码块或在前后带上
    说明文字, 所以这里保留一层容错剥离。
    """
    text = (raw or "").strip()
    if not text:
        return None, ["模型返回空内容"]

    candidates = [text]
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, []
        return None, [f"JSON 顶层应该是对象, 实际是 {type(parsed).__name__}"]
    return None, ["无法从模型输出里解析出 JSON 对象"]


def _format_validation_errors(exc: ValidationError) -> list[str]:
    """把 pydantic 报错压成模型好懂的一行一条。"""
    messages: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(根对象)"
        messages.append(f"字段 {location}: {error['msg']}")
    return messages


def _gaps_payload(gaps: list[CapabilityGap]) -> list[dict[str, Any]]:
    return [
        {
            "topic": gap.topic,
            "matched_keywords": list(gap.matched),
            "why": gap.why,
            "consequence": gap.workaround,
        }
        for gap in gaps
    ]


def _validator_report(afsim_code: str) -> dict[str, Any]:
    """跑 ScenClaw 四层验证器, 给出评分与问题清单。"""
    from nanobot.agent.tools.scenario_validator_tool import ScenarioValidator

    result = ScenarioValidator(afsim_code).validate_all()
    return {
        "score": result.score,
        "errors": result.errors,
        "warnings": result.warnings,
        "issues": [
            {
                "rule": issue.rule_id,
                "severity": issue.severity,
                "line": issue.line_number,
                "description": issue.description,
            }
            for issue in result.issues
        ],
    }


def _write_scenario(output_path: str, afsim_code: str) -> str:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(afsim_code, encoding="utf-8")
    return str(path.resolve())


def _write_project_output(
    output_path: str, scenario: ScenarioIR, afsim_code: str
) -> dict[str, Any]:
    """把想定写成多文件项目, 同时保留单文件版本作为入口快照。

    真实 AFSIM 想定是一个目录 (main.txt + 类型定义 + 平台/传感器/武器子目录),
    不是一个单文件。但为了向后兼容和快速查看, 我们:

    1. 在 ``output_path`` 对应的目录里写多文件项目 (main.txt 是入口)
    2. 同时在该目录下写一份 ``scenario_single.txt``, 内容与旧的单文件编译等价

    返回结构化的项目信息给调用方 (CLI 据此打印目录树)。
    """
    project_root = Path(output_path).expanduser()
    # 如果用户给了 .txt 后缀, 用它当项目目录名的基础 (去掉后缀)
    if project_root.suffix == ".txt":
        project_root = project_root.with_suffix("")
    project_root.mkdir(parents=True, exist_ok=True)

    written = write_project(scenario, project_root)

    # 同时写一份单文件版本 (语义等价, 方便 diff / 快速查看)
    single_path = project_root / "scenario_single.txt"
    single_path.write_text(afsim_code, encoding="utf-8")

    return {
        "project_root": str(project_root.resolve()),
        "entry": str((project_root / "main.txt").resolve()),
        "files": sorted(written.keys()),
        "single_file": str(single_path.resolve()),
    }


def _make_llm(kwargs: dict[str, Any]) -> Any:
    """按工具参数 + 环境变量构造 LLM 客户端。"""
    from nanobot.agent.tools.react_engine import DeepSeekClient

    return DeepSeekClient(
        api_key=kwargs.get("api_key", ""),
        base_url=kwargs.get("base_url", ""),
        model=kwargs.get("model", ""),
        timeout=int(kwargs.get("timeout_s") or 120),
    )


_LLM_PARAMS: dict[str, Any] = {
    "api_key": {
        "type": "string",
        "description": (
            "LLM API key. Omit to read the DEEPSEEK_API_KEY environment variable "
            "(preferred, keeps the key out of tool traces)."
        ),
    },
    "base_url": {
        "type": "string",
        "description": "OpenAI-compatible endpoint. Defaults to DEEPSEEK_BASE_URL or https://api.deepseek.com",
    },
    "model": {
        "type": "string",
        "description": "Model name. Defaults to DEEPSEEK_MODEL or deepseek-chat",
    },
    "max_tokens": {
        "type": "integer",
        "description": (
            "Completion budget per LLM call (reasoning + JSON). Default: 8192; "
            "raise it for scenarios with many platforms."
        ),
        "minimum": 1024,
    },
    "timeout_s": {
        "type": "integer",
        "description": "HTTP timeout per LLM call in seconds. Default: 120",
        "minimum": 10,
    },
}


# ═══════════════════════════════════════════════════════
# Tool: 自然语言 → ScenarioIR
# ═══════════════════════════════════════════════════════


class NLScenarioPlanTool(Tool):
    """自然语言规划 → 强类型 ScenarioIR (不落地 AFSIM 文本)。"""

    _scopes = {"core", "subagent"}

    @property
    def name(self) -> str:
        return "nl_scenario_plan"

    @property
    def description(self) -> str:
        return (
            "Extract a strongly-typed ScenarioIR from a free-form natural-language "
            "operational plan. The LLM only fills a schema-constrained IR; every "
            "candidate is checked against the Pydantic schema, the compiler "
            "capability whitelist and an AFSIM static lint, and validation errors "
            "are fed back for a bounded number of repair rounds. Returns the "
            "ScenarioIR JSON plus the attempt trace."
        )

    @property
    def read_only(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "Natural-language operational plan (Chinese or English).",
                },
                "center_lat": {
                    "type": "number",
                    "description": "Optional theatre centre latitude to anchor generated positions.",
                },
                "center_lon": {
                    "type": "number",
                    "description": "Optional theatre centre longitude to anchor generated positions.",
                },
                "duration_s": {
                    "type": "number",
                    "description": "Optional mission duration in seconds.",
                },
                "scenario_id": {
                    "type": "string",
                    "description": "Optional scenario identifier.",
                },
                "max_repair_rounds": {
                    "type": "integer",
                    "description": "Max IR-level repair rounds. Default: 3.",
                    "minimum": 1,
                    "maximum": 8,
                },
                **_LLM_PARAMS,
            },
            "required": ["request"],
        }

    async def execute(self, **kwargs: Any) -> str:
        try:
            request = (kwargs.get("request") or "").strip()
            if not request:
                return "Error: 'request' must not be empty."

            planner = ScenarioIRPlanner(
                llm=_make_llm(kwargs),
                max_repair_rounds=int(kwargs.get("max_repair_rounds", 3)),
                max_tokens=int(kwargs.get("max_tokens") or 8192),
            )
            result = planner.plan(request, _collect_hints(kwargs))

            return json.dumps(
                {
                    "success": result.success,
                    "out_of_scope": bool(result.out_of_scope),
                    "reason": result.out_of_scope,
                    "rounds": result.rounds,
                    "scenario_ir": result.scenario_ir,
                    "plausibility_advisories": result.advisories,
                    "capability_gaps": _gaps_payload(result.capability_gaps),
                    "unverified_capabilities": result.unverified_capabilities,
                    "normalizations": result.normalizations,
                    "attempts": [
                        {
                            "round": a.round,
                            "stage": a.stage,
                            "ok": a.ok,
                            "errors": a.errors,
                            "advisories": a.advisories,
                        }
                        for a in result.attempts
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        except Exception as exc:
            return f"Error in nl_scenario_plan: {exc}"


# ═══════════════════════════════════════════════════════
# Tool: ScenarioIR → AFSIM 文本
# ═══════════════════════════════════════════════════════


class ScenarioCompileTool(Tool):
    """ScenarioIR → AFSIM 文本 (确定性, 不调用大模型)。"""

    _scopes = {"core", "subagent"}

    @property
    def name(self) -> str:
        return "scenario_compile"

    @property
    def description(self) -> str:
        return (
            "Deterministically compile a ScenarioIR JSON document into AFSIM 2.9 "
            "scenario text. No LLM involved: the same IR always yields byte-identical "
            "output. Runs the capability whitelist, the AFSIM static linter and the "
            "four-level scenario validator, and can write the result to a file."
        )

    @property
    def read_only(self) -> bool:
        return False

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "scenario_ir_json": {
                    "type": "string",
                    "description": "ScenarioIR document as a JSON string.",
                },
                "scenario_ir_path": {
                    "type": "string",
                    "description": "Path to a ScenarioIR JSON file (used when scenario_ir_json is absent).",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional path to write the generated AFSIM scenario text to.",
                },
            },
        }

    async def execute(self, **kwargs: Any) -> str:
        try:
            raw = kwargs.get("scenario_ir_json") or ""
            path = kwargs.get("scenario_ir_path") or ""
            if not raw and not path:
                return "Error: provide either 'scenario_ir_json' or 'scenario_ir_path'."
            if not raw:
                source = Path(path).expanduser()
                if not source.is_file():
                    return f"Error: ScenarioIR file not found: {source}"
                raw = source.read_text(encoding="utf-8")

            try:
                scenario = ScenarioIR.model_validate_json(raw)
            except ValidationError as exc:
                return json.dumps(
                    {"success": False, "stage": "schema",
                     "errors": _format_validation_errors(exc)},
                    ensure_ascii=False, indent=2,
                )

            try:
                afsim_code = AfsimCompiler().compile(scenario)
            except CompileError as exc:
                return json.dumps(
                    {"success": False, "stage": "compile", "errors": [str(exc)]},
                    ensure_ascii=False, indent=2,
                )

            lint = lint_afsim(afsim_code)
            output: dict[str, Any] = {
                "success": lint.is_valid,
                "stage": "compiled",
                "scenario_id": scenario.scenario_id,
                "lines_of_code": len(afsim_code.splitlines()),
                "lint": {
                    "valid": lint.is_valid,
                    "diagnostics": [
                        {"severity": d.severity.value, "code": d.code,
                         "message": d.message, "line": d.line}
                        for d in lint.diagnostics
                    ],
                },
                "validator": _validator_report(afsim_code),
                "afsim_code": afsim_code,
            }
            if kwargs.get("output_path"):
                output["project"] = _write_project_output(
                    kwargs["output_path"], scenario, afsim_code
                )
            return json.dumps(output, ensure_ascii=False, indent=2)
        except Exception as exc:
            return f"Error in scenario_compile: {exc}"


# ═══════════════════════════════════════════════════════
# Tool: 端到端 自然语言 → 想定文件
# ═══════════════════════════════════════════════════════


class NLToAfsimTool(Tool):
    """端到端: 自然语言规划 → 可执行 AFSIM 想定文件。"""

    _scopes = {"core", "subagent"}

    @property
    def name(self) -> str:
        return "nl_to_afsim"

    @property
    def description(self) -> str:
        return (
            "End-to-end pipeline turning a natural-language operational plan into an "
            "AFSIM 2.9 scenario file: LLM semantic extraction into a typed ScenarioIR, "
            "bounded IR-level self-repair driven by schema/capability/lint errors, then "
            "deterministic template rendering into AFSIM text. Self-correction happens "
            "on the IR, never by patching generated text, so the emitted syntax is "
            "correct by construction and reproducible. Returns the scenario code, "
            "validator score and the full repair trace."
        )

    @property
    def read_only(self) -> bool:
        return False

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "Natural-language operational plan (Chinese or English).",
                },
                "output_path": {
                    "type": "string",
                    "description": "Where to write the generated AFSIM scenario (.txt).",
                },
                "center_lat": {
                    "type": "number",
                    "description": "Optional theatre centre latitude. Default: 27.5",
                },
                "center_lon": {
                    "type": "number",
                    "description": "Optional theatre centre longitude. Default: 122.5",
                },
                "duration_s": {
                    "type": "number",
                    "description": "Optional mission duration in seconds. Default: 3600",
                },
                "scenario_id": {
                    "type": "string",
                    "description": "Optional scenario identifier.",
                },
                "max_repair_rounds": {
                    "type": "integer",
                    "description": "Max IR-level repair rounds. Default: 3.",
                    "minimum": 1,
                    "maximum": 8,
                },
                "include_code": {
                    "type": "boolean",
                    "description": "Include the full AFSIM text in the response. Default: true.",
                },
                **_LLM_PARAMS,
            },
            "required": ["request"],
        }

    async def execute(self, **kwargs: Any) -> str:
        try:
            request = (kwargs.get("request") or "").strip()
            if not request:
                return "Error: 'request' must not be empty."

            planner = ScenarioIRPlanner(
                llm=_make_llm(kwargs),
                max_repair_rounds=int(kwargs.get("max_repair_rounds", 3)),
                max_tokens=int(kwargs.get("max_tokens") or 8192),
            )
            result = planner.plan(request, _collect_hints(kwargs))

            trace = [
                {"round": a.round, "stage": a.stage, "ok": a.ok,
                 "errors": a.errors, "advisories": a.advisories}
                for a in result.attempts
            ]
            if result.out_of_scope:
                return json.dumps(
                    {
                        "success": False,
                        "out_of_scope": True,
                        "reason": result.out_of_scope,
                        "trace": trace,
                    },
                    ensure_ascii=False, indent=2,
                )
            if not result.success:
                return json.dumps(
                    {
                        "success": False,
                        "rounds": result.rounds,
                        "message": (
                            "自然语言未能收敛成合法 ScenarioIR。多数情况是规划里缺少"
                            "必要信息 (兵力构成/地理位置/时间窗), 或涉及编译器尚未放行的"
                            "AFSIM 能力。请看 trace 里的报错。"
                        ),
                        "capability_gaps": _gaps_payload(result.capability_gaps),
                        "trace": trace,
                    },
                    ensure_ascii=False, indent=2,
                )

            output: dict[str, Any] = {
                "success": True,
                "rounds": result.rounds,
                "scenario_id": result.scenario_ir.get("scenario_id", ""),
                "lines_of_code": len(result.afsim_code.splitlines()),
                "validator": _validator_report(result.afsim_code),
                "lint_diagnostics": result.lint_diagnostics,
                "plausibility_advisories": result.advisories,
                "capability_gaps": _gaps_payload(result.capability_gaps),
                "unverified_capabilities": result.unverified_capabilities,
                "normalizations": result.normalizations,
                "scenario_ir": result.scenario_ir,
                "trace": trace,
            }
            if kwargs.get("output_path") and result.scenario_ir is not None:
                scenario_obj = ScenarioIR.model_validate(result.scenario_ir)
                output["project"] = _write_project_output(
                    kwargs["output_path"], scenario_obj, result.afsim_code
                )
            if kwargs.get("include_code", True):
                output["afsim_code"] = result.afsim_code
            return json.dumps(output, ensure_ascii=False, indent=2)
        except Exception as exc:
            return f"Error in nl_to_afsim: {exc}"


def _collect_hints(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "center_lat": kwargs.get("center_lat"),
        "center_lon": kwargs.get("center_lon"),
        "duration_s": kwargs.get("duration_s"),
        "scenario_id": kwargs.get("scenario_id"),
    }
