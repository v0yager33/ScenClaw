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
from collections import Counter
from dataclasses import dataclass, field
from itertools import groupby
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
    ScheduledEngagementIR,
    ValidationStatus,
    check_geometry,
    emit_project,
    expand_project,
    ground_distance_m,
    lint_afsim,
    lint_project,
    position_at,
    slant_range_m,
    waypoint_times_s,
    write_project,
)
from nanobot.afsim.requirements import (
    REQUIREMENT_CHECKS_SPEC,
    RequirementCheck,
    describe_requirement_check,
    evaluate_requirement_checks,
    parse_requirement_checks,
)
from nanobot.agent.tools.base import Tool

# ═══════════════════════════════════════════════════════
# 结果数据结构
# ═══════════════════════════════════════════════════════


@dataclass
class PlanAttempt:
    """一次 IR 抽取尝试及其失败原因。"""
    round: int
    stage: str            # "extract" | "repair" | "refine" | "audit"
    ok: bool
    errors: list[str] = field(default_factory=list)
    advisories: list[str] = field(default_factory=list)
    raw_excerpt: str = ""
    # 本轮输出是相对上一版的补丁 (合并后再校验)
    patched: bool = False


@dataclass
class PlanResult:
    """自然语言 → ScenarioIR 的结果。"""
    success: bool
    scenario_ir: dict[str, Any] | None = None
    afsim_code: str = ""
    attempts: list[PlanAttempt] = field(default_factory=list)
    lint_diagnostics: list[str] = field(default_factory=list)
    advisories: list[str] = field(default_factory=list)
    # assumptions 里已写明原因的"打不出"射击, 照样列出, 但不算作合理性问题
    acknowledged: list[str] = field(default_factory=list)
    capability_gaps: list["CapabilityGap"] = field(default_factory=list)
    unverified_capabilities: list[str] = field(default_factory=list)
    normalizations: list[str] = field(default_factory=list)
    out_of_scope: str = ""
    # 需求核对轮: "" 未进行; "met" 模型对照推算事实确认需求都已满足;
    # "unconfirmed" 核对过但最终版本没得到确认 (修正被拒、轮数用完或调用失败)
    requirements_audit: str = ""
    # 从需求原文抽出、由代码逐版核验的数值条款, 以及抽取过程的说明 (丢弃了哪些条款等)
    requirement_checks: list[dict[str, Any]] = field(default_factory=list)
    requirement_notes: list[str] = field(default_factory=list)

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

# 只在白名单放行隐式武器时并入系统提示词, 避免提示词里出现未放行的类型
_IMPLICIT_ENGAGEMENT_RULES = """
**交战按开火时刻结算** (WSF_IMPLICIT_WEAPON): 不模拟导弹飞行, 开火瞬间就摧毁目标,
被毁平台随即从仿真中删除; 命中也不看距离, 射程完全靠你在规划里保证。因此:
- 开火时刻要按航路速度推算双方位置, 距离落在需求给出的交战包络内并留几公里余量
  (需求没给就以我方传感器探测得到目标为准); 时刻、距离都不要卡在边界上,
  否则之后改航路时很容易又滑出去;
- 开火时双方都还要在航路上: 航路飞完后平台怎么飞不由 IR 决定, 开火时刻晚于任一方
  最后一个航点的到达时刻, 距离就没有保证; 后段还要开火或被打的平台, 航路要续到那之后;
- 对射时先开火的一方直接摧毁对方 —— 不要写"A 先射 B、B 稍后回射 A",
  后一条永远不会发生; 要让双方都打出射击, B 必须在被击毁之前先打别的目标;
- 被击毁的平台此后的航点都不会飞到: 需求要求某平台在某时刻之后完成的航段
  (穿越、抵达、撤离等), 它必须活到那时。射击计划决定了谁在何时被毁, 不要为了
  "每发都打得出"把承担这类航段的平台提前打掉;
- 掩护、压制可以写成己方抢先摧毁威胁: 对方针对被掩护平台的射击因射手已被击毁
  而打不出, 这是合理的战术结果;
- 每个目标只安排一次有效射击, 不要再打已被摧毁的平台;
- 同一武器的射击次数不超过它的 quantity;
- 不要靠同一秒互射让双方都开火: 同一时刻的射击也是逐条执行的, 无法保证两发都打得出;
- 有意安排或需求硬性要求、却注定打不出的射击计划, 在 assumptions 里逐条写明它的
  engagement_id 和原因, 校验器据此不再把它算作问题。逐条的时序问题之后会由校验器
  推演并指出, 不必先在思考里逐条推演。
"""

# 推理模型把 max_tokens 大半耗在思考上、正文没写完时, 重问时追加的催促
_THOUGHT_OUT_NUDGE = (
    "注意: 上一次回答的思考过程耗尽了输出预算, 正文没能完整输出。请少做逐条推演, "
    "直接输出完整的 ScenarioIR JSON; 交战时序、几何可达性等问题之后会由校验器逐条指出。"
)

# 需求核对轮: 模型确认需求都已满足时只回这个标记, 不必把整份 IR 再抄一遍
_REQUIREMENTS_MET_SENTINEL = "__requirements_met__"

# 修订轮只输出相对上一版的补丁 (代码仍接受完整 IR): 实测推理模型的思考和正文共用 max_tokens,
# 只改几处却整份重抄 12v12 的 IR, 正文常被截断
_PATCH_KEY = "__patch__"
_DELETE_MARK = "__delete__"
# 这些列表按元素 id 合并, 其余列表整体替换
_PATCH_LIST_IDS: dict[tuple[str, ...], str] = {
    ("platforms",): "platform_id",
    ("mission", "platforms"): "platform_id",
    ("engagements",): "engagement_id",
    ("assertions",): "assertion_id",
    ("sensors",): "sensor_type_id",
    ("platforms", "weapons"): "instance_name",
}

_PATCH_FORMAT = f"""输出格式: 上一版 IR 就在对话里, 只输出改动的补丁 {{"{_PATCH_KEY}": {{...}}}}, 不要整份重抄 (整份重抄常超出输出预算被截断); 代码会把补丁合并到上一版再完整校验。补丁的合并规则:
- 对象按键合并, 只写要改的键; 值写 null 表示删掉这个键;
- platforms、mission.platforms、engagements、assertions、sensors 和 platforms[].weapons 按 id (platform_id、engagement_id、assertion_id、sensor_type_id、instance_name) 合并: 只列要改的元素, 没列出的保持原样; 新 id 追加到末尾; 元素里写 "{_DELETE_MARK}": true 表示删除它;
- 其余列表 (route、platforms[].sensors、assumptions、rationale、assignments 等) 整体替换, 要写出改后的完整列表。
示例: {{"{_PATCH_KEY}": {{"engagements": [{{"engagement_id": "e1", "fire_time_s": 900.0}}, {{"engagement_id": "e2", "{_DELETE_MARK}": true}}], "platforms": [{{"platform_id": "red_f1", "route": [...]}}]}}}}"""

_REVISION_NUDGE = (
    "注意: 上一次回答没能在输出预算内写完 (思考过程或整份重抄占满了 max_tokens)。上一版 IR "
    "就在对话里, 不要整份重抄, 只输出改动的补丁 "
    f'{{"{_PATCH_KEY}": {{...}}}}; 各条问题附带的时段、速度等数值已经算好, 直接采用, '
    "不必重算; 问题很多时先修一部分。"
)

_AUDIT_THOUGHT_OUT_NUDGE = (
    "注意: 上一次回答没能在输出预算内写完。事实表里的数字都已算好, "
    "请直接对照, 不要重算航程; 需求都满足就只输出 "
    f'{{"{_REQUIREMENTS_MET_SENTINEL}": true}}, 否则只输出改动的补丁 '
    f'{{"{_PATCH_KEY}": {{...}}}}, 不要整份重抄。'
)


# 需求条款抽取: 只读需求原文和平台清单, 不看 IR, 也不做计算
_REQUIREMENT_CHECKS_PROMPT = f"""你是作战想定需求的条款抽取器。任务: 从规划正文里找出明确写出数值的约束, 抽成结构化条款, 由代码按想定逐条核验。

只做抽取, 不做计算, 也不评价想定。正文没写明数值的要求 (如"保持编队""适时撤离") 不要抽, 拿不准的宁可不抽; 一句话含多个数值约束时拆成多条。平台选择器只能用平台清单里的 id (可用 * 通配) 或 side:blue、side:red。

正文给出了阶段划分 (如"0-240 秒展开、240-540 秒争夺……") 时, 阶段的起止时刻也是写明的数值, 要用起来: 要求"至少 N 个不同交战时段/波次"却没单独给时段的, 用这些阶段作 fire_phases 的 windows; 某编队的到达、穿越、越线、进入某区域等动作对应某个阶段的, 把该阶段的时段作 reach_area 的 start_s/end_s。

{REQUIREMENT_CHECKS_SPEC}

只输出 JSON 对象 {{"checks": [...]}}, 没有可抽的条款时输出 {{"checks": []}}。
示例: {{"checks": [{{"kind": "fire_window", "shooters": ["blue_reserve_*"], "start_s": 540, "end_s": 900, "quote": "预备队在 540-900 s 内发射"}}, {{"kind": "reach_area", "platforms": ["blue_strike_1", "blue_strike_2"], "lat_min": 27.3, "lat_max": 27.7, "lon_min": 122.8, "start_s": 900, "end_s": 1200, "quote": "突击编队 900-1200 s 越过 122.8°E"}}]}}"""


def _build_system_prompt(registry: CapabilityRegistry) -> str:
    sensors = _enabled_symbols(registry, CapabilityKind.SENSOR_TYPE)
    movers = _enabled_symbols(registry, CapabilityKind.MOVER_TYPE)
    weapons = _enabled_symbols(registry, CapabilityKind.WEAPON_TYPE)
    events = _enabled_symbols(registry, CapabilityKind.EVENT_TYPE)
    sentinel = _OUT_OF_SCOPE_SENTINEL
    engagement_rules = _IMPLICIT_ENGAGEMENT_RULES if "WSF_IMPLICIT_WEAPON" in weapons else ""

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
{engagement_rules}
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
    4. schema 里没有、又不携带信息的字段直接删掉 —— 值为空, 或与同级字段重复。

    刻意**不**在这里处理"有 mover 却没航路": 那个问题有两个合理答案 (固定阵地该删
    mover, 要拦截的战斗机该补航路), 需要需求语义, 交给模型逐平台决定。我一度让代码
    统一删 mover, 结果把数据链协同、空中对抗想定里所有飞机都变成静止平台。

    也不在这里处理"航路走不完": 缩短航路 / 提高速度 / 延长时长三种解法都合理。

    第 2 条是上一轮补的。第 3 条是这一轮补的: 实测中 LLM 抽 IR 时 9 个平台全填了
    ``WSF_PLATFORM`` (schema 默认值), 导致多文件项目里所有平台挤进同一个类型文件。
    归一化层按 ``type_hint`` 推机型缩写, 比让模型碰运气可靠。

    第 4 条来自 12v12 实测: 需求核对轮的修正版把蓝一越线时刻正确地挤进了时间窗,
    却顺手多写了 ``"assumptions_note": null``, 整版被 extra="forbid" 打回; 另一次多写了
    ``"coa_id_note": "coa_1"`` (与 coa_id 重复), 修一轮就要重写 5 万字。携带新内容的多余字段
    仍交给校验器打回, 由模型决定内容该挤进哪个字段。

    修正过程必须写进输出 (normalizations), 不能静默改动使用者的 IR。
    """
    result = copy.deepcopy(payload)
    notes: list[str] = _drop_redundant_extra_fields(result)

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


def _sync_revised_intents(base: dict[str, Any], revised: dict[str, Any]) -> list[str]:
    """修订只改了物理平台的位置或航路时, 原地同步 mission.platforms 里的意图, 返回说明。

    实测: refine 补丁给 platforms[].route 补了航点, 没动 mission.platforms[].route, 交付的
    IR 里意图航路比编译出的航路少一个点。只在上一版两者一致、这一版意图没改时同步;
    上一版本来就不一致, 或这一版意图也改了, 说明模型另有打算, 不替它决定。
    """
    old_platforms = _entries_by_id(base.get("platforms"))
    old_intents = _entries_by_id(_mission_platforms(base))
    intents = _entries_by_id(_mission_platforms(revised))
    synced: list[str] = []
    for platform_id, platform in _entries_by_id(revised.get("platforms")).items():
        old_platform = old_platforms.get(platform_id)
        old_intent, intent = old_intents.get(platform_id), intents.get(platform_id)
        if old_platform is None or old_intent is None or intent is None:
            continue
        before, after = _platform_track(old_platform), _platform_track(platform)
        if before is None or after is None or after == before:
            continue
        if _intent_track(old_intent) != before or _intent_track(intent) != before:
            continue
        intent["initial_position"] = copy.deepcopy(platform["position"])
        intent["route"] = [copy.deepcopy(point["position"]) for point in platform.get("route") or []]
        synced.append(platform_id)
    if not synced:
        return []
    return [f"修订只改了 {'、'.join(synced)} 的物理位置或航路, 已同步到 mission.platforms 里的意图"]


def _mission_platforms(payload: dict[str, Any]) -> Any:
    mission = payload.get("mission")
    return mission.get("platforms") if isinstance(mission, dict) else None


def _entries_by_id(entries: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(entries, list):
        return {}
    return {
        entry["platform_id"]: entry for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("platform_id"), str)
    }


_Point = tuple[float, float, float]
_Track = tuple[_Point, tuple[_Point, ...]]


def _platform_track(platform: dict[str, Any]) -> _Track | None:
    route = platform.get("route") or []
    if not isinstance(route, list):
        return None
    points = [point.get("position") if isinstance(point, dict) else None for point in route]
    return _track(platform.get("position"), points)


def _intent_track(intent: dict[str, Any]) -> _Track | None:
    route = intent.get("route") or []
    if not isinstance(route, list):
        return None
    return _track(intent.get("initial_position"), route)


def _track(start: Any, points: list[Any]) -> _Track | None:
    """起点加航点的可比较形式; 有点缺坐标时返回 None (判断不了两者是否一致)。"""
    keys = [_point_key(point) for point in [start, *points]]
    if any(key is None for key in keys):
        return None
    return keys[0], tuple(keys[1:])


def _point_key(point: Any) -> _Point | None:
    if not isinstance(point, dict):
        return None
    try:
        return (
            float(point["latitude_deg"]),
            float(point["longitude_deg"]),
            float(point.get("altitude_m", 0.0)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _drop_redundant_extra_fields(payload: dict[str, Any]) -> list[str]:
    """原地删掉 schema 里没有、又不携带信息的字段, 返回删除说明。

    不携带信息指: 值为空 (null / 空串 / 空容器), 或与同级某个字段的值完全相同。
    """
    try:
        ScenarioIR.model_validate(payload)
        return []
    except ValidationError as exc:
        locations = [
            tuple(error["loc"]) for error in exc.errors()
            if error["type"] == "extra_forbidden" and error["loc"]
        ]
    notes: list[str] = []
    for location in locations:
        container: Any = payload
        try:
            for key in location[:-1]:
                container = container[key]
        except (KeyError, IndexError, TypeError):
            continue  # 联合类型的报错位置带类名, 对不上原始 JSON, 留给校验器
        name = location[-1]
        if not isinstance(container, dict) or name not in container:
            continue
        value = container[name]
        path = ".".join(str(k) for k in location)
        if value is None or value in ("", [], {}):
            del container[name]
            notes.append(f"删除了 schema 未定义且值为空的字段 {path}")
            continue
        twin = next(
            (k for k, v in container.items()
             if k != name and type(v) is type(value) and v == value),
            None,
        )
        if twin is not None:
            del container[name]
            notes.append(f"删除了 schema 未定义的字段 {path} (值与同级字段 {twin} 重复)")
    return notes


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

    # 8. 射击计划超过弹量: 打光之后的 Fire 调用不会发射
    planned = Counter(
        (engagement.shooter_platform, engagement.weapon_name)
        for engagement in scenario.engagements
    )
    for platform in scenario.platforms:
        for weapon in platform.weapons:
            count = planned.get((platform.platform_id, weapon.instance_name), 0)
            if count > weapon.quantity:
                issues.append(
                    f"平台 {platform.platform_id} 的武器 {weapon.instance_name} 只有 "
                    f"quantity={weapon.quantity} 发, 却安排了 {count} 次射击, 多出的射击不会"
                    f"发生; 请删减射击计划, 需求允许时也可提高 quantity"
                )

    # 9. 交战时序: 射手或目标在开火前已被摧毁
    findings, _ = _simulate_engagements(scenario)
    issues.extend(
        finding.message for finding in findings
        if not _is_acknowledged(finding, scenario)
    )

    return issues


def acknowledged_engagement_issues(scenario: ScenarioIR) -> list[str]:
    """mission.assumptions 里已写明 engagement_id 的"打不出"射击。

    需求可能硬性要求每个平台都有射击计划, 而开火即毁之下最先被击毁的那架注定来不及
    开火; 掩护兵力抢先摧毁威胁, 也会让对方的射击计划落空。把这些也算作必须清零的问题,
    会把模型推向"按时间顺序互相打光"的写法, 需求里的穿越、撤离航段反而没人去飞。
    说明过的照样列出, 只是不再触发 refine。
    """
    findings, _ = _simulate_engagements(scenario)
    return [finding.message for finding in findings if _is_acknowledged(finding, scenario)]


def _requirement_issues(scenario: ScenarioIR, checks: list[RequirementCheck]) -> list[str]:
    """需求数值条款的核验结果; 毁伤按射击计划推演, 被毁平台之后的射击和航段都不算。"""
    if not checks:
        return []
    _, destroyed = _simulate_engagements(scenario)
    return evaluate_requirement_checks(scenario, checks, destroyed)


@dataclass(frozen=True)
class _SequenceFinding:
    engagement_id: str
    message: str
    # 射手先被击毁、同秒互射: 射击打不出, 但可能是有意的掩护安排或需求硬性要求。
    # 打已毁目标、同秒重复射击白白浪费一发, 总能改打别的目标, 不接受说明
    acknowledgeable: bool


def _is_acknowledged(finding: _SequenceFinding, scenario: ScenarioIR) -> bool:
    if not finding.acknowledgeable:
        return False
    # 按整词匹配, 免得 e1 被 e10 的说明顺带覆盖
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_-]){re.escape(finding.engagement_id)}(?![A-Za-z0-9_-])"
    )
    return any(pattern.search(text) for text in scenario.mission.assumptions)


def _simulate_engagements(
    scenario: ScenarioIR,
) -> tuple[list[_SequenceFinding], dict[str, tuple[float, str]]]:
    """按开火时刻推演隐式武器的毁伤, 找出不会发生或白白浪费的射击。

    同时返回推演出的毁伤时刻 ``{platform_id: (时刻, 交战 id)}``。

    WSF_IMPLICIT_WEAPON 开火即摧毁目标, 被毁平台随即从仿真中删除: 它自己的后续
    射击不再执行, 别人再打它也只会被判空跳过。实跑报错的分布还表明 maximum_radius
    不限制射程, 所以这里不看距离, 只看先后。同一秒的射击是各自独立的 execute 事件,
    按事件队列逐条执行, 谁先谁后无法从 IR 预知: 这一秒的目标一律记为被毁, 射手
    自己也在这一秒挨打的射击单独提示 "可能打不出"。
    """
    weapon_type = {
        (platform.platform_id, weapon.instance_name): weapon.weapon_type
        for platform in scenario.platforms
        for weapon in platform.weapons
    }
    ordered = sorted(scenario.engagements, key=lambda engagement: engagement.fire_time_s)

    findings: list[_SequenceFinding] = []
    destroyed: dict[str, tuple[float, str]] = {}
    for fire_time, group in groupby(ordered, key=lambda engagement: engagement.fire_time_s):
        kills: dict[str, str] = {}
        live: list[ScheduledEngagementIR] = []
        for engagement in group:
            key = (engagement.shooter_platform, engagement.weapon_name)
            if weapon_type.get(key) != "WSF_IMPLICIT_WEAPON":
                continue
            shooter = engagement.shooter_platform
            target = engagement.target_platform
            engagement_id = engagement.engagement_id
            if shooter in destroyed:
                killed_at, killer = destroyed[shooter]
                findings.append(_SequenceFinding(engagement_id, (
                    f"交战 {engagement_id}: 射手 {shooter} 已在 {killed_at:g}s "
                    f"被交战 {killer} 摧毁, {fire_time:g}s 的这次射击不会发生。隐式武器"
                    f"开火即摧毁目标, 对射时先开火的一方直接打掉对方; 请把这次射击提前到"
                    f" {killed_at:g}s 之前、改由其他存活平台承担, 或删掉这条计划。若是有意"
                    f"安排 (如己方掩护抢先摧毁威胁) 或需求要求保留, 在 mission.assumptions 里"
                    f"写明 {engagement_id} 和原因"
                ), acknowledgeable=True))
            elif target in destroyed:
                killed_at, killer = destroyed[target]
                findings.append(_SequenceFinding(engagement_id, (
                    f"交战 {engagement_id}: 目标 {target} 已在 {killed_at:g}s "
                    f"被交战 {killer} 摧毁, {fire_time:g}s 的射击会被跳过; "
                    f"请改打其他仍存活的敌方平台"
                ), acknowledgeable=False))
            elif target in kills:
                findings.append(_SequenceFinding(engagement_id, (
                    f"交战 {engagement_id} 与 {kills[target]} 在 {fire_time:g}s "
                    f"同时射击 {target}, 一发即可摧毁, 另一发白白消耗; 请改打其他敌方平台"
                ), acknowledgeable=False))
            else:
                kills[target] = engagement_id
                live.append(engagement)
        for engagement in live:
            attacker = kills.get(engagement.shooter_platform)
            if attacker is not None:
                findings.append(_SequenceFinding(engagement.engagement_id, (
                    f"交战 {engagement.engagement_id}: 射手 {engagement.shooter_platform} 在同一时刻 "
                    f"{fire_time:g}s 也遭到交战 {attacker} 射击。同一时刻的射击也逐条执行, 对方"
                    f"若先执行就会把它摧毁, 谁先执行无法预知, 这次射击可能打不出; 请把它提前到"
                    f" {fire_time:g}s 之前或改由其他平台承担"
                ), acknowledgeable=True))
        for target, engagement_id in kills.items():
            destroyed[target] = (fire_time, engagement_id)
    return findings, destroyed


def _timeline_digest(scenario: ScenarioIR, limit: int = 60) -> list[str]:
    """按当前 IR 推算的航点到达时刻与毁伤时刻, 每个平台一行。

    核对需求里的时间窗本该是确定性计算; 交给推理模型在思考里逐个重算既慢又容易错,
    还会挤占正文的输出预算。
    """
    _, destroyed = _simulate_engagements(scenario)
    lines: list[str] = []
    for platform in scenario.platforms[:limit]:
        times = waypoint_times_s(platform) if platform.route else []
        if times is None:
            parts = ["航路缺速度, 无法推算到达时刻"]
        elif times:
            parts = ["航点预计到达 " + "/".join(f"{t:.0f}" for t in times) + "s"]
        else:
            parts = ["无航路"]
        if platform.platform_id in destroyed:
            killed_at, killer = destroyed[platform.platform_id]
            text = f"{killed_at:g}s 被交战 {killer} 摧毁"
            missed = sum(1 for t in times or [] if t > killed_at)
            if missed:
                text += f", 之后 {missed} 个航点不会飞到"
            parts.append(text)
        lines.append(f"- {platform.platform_id}: " + "; ".join(parts))
    return lines


def _format_point(point: Any) -> str:
    return f"({point.latitude_deg:.3f}, {point.longitude_deg:.3f}, {point.altitude_m:.0f}m)"


def _engagement_outcome(
    engagement: ScheduledEngagementIR,
    destroyed: dict[str, tuple[float, str]],
    implicit: bool,
) -> str:
    """按 _simulate_engagements 的推演结果说明这次射击的结局。"""
    if not implicit:
        return "非隐式武器, 不按开火即毁结算"
    fire_time = engagement.fire_time_s
    shooter_kill = destroyed.get(engagement.shooter_platform)
    if shooter_kill and shooter_kill[0] < fire_time:
        return f"打不出: 射手已在 {shooter_kill[0]:g}s 被交战 {shooter_kill[1]} 摧毁"
    target_kill = destroyed.get(engagement.target_platform)
    if target_kill and target_kill[0] < fire_time:
        return f"被跳过: 目标已在 {target_kill[0]:g}s 被交战 {target_kill[1]} 摧毁"
    if target_kill and target_kill[1] != engagement.engagement_id:
        return f"白白消耗: 同秒已有交战 {target_kill[1]} 射击同一目标"
    outcome = "摧毁目标"
    if shooter_kill:
        outcome += f" (射手同秒遭交战 {shooter_kill[1]} 射击, 可能打不出)"
    return outcome


def _fact_sheet(scenario: ScenarioIR, limit: int = 120) -> str:
    """需求核对轮用的推算事实: 航点坐标与到达时刻、交战结算、双方初始距离、存活数。

    需求里的具体数值 (时间窗、初始距离、开火时段、交战距离) 写不成通用检查, 只能
    交给模型对照原文; 但对照所需的数字由代码算好, 模型只做比对, 不在思考里重算航程。
    """
    _, destroyed = _simulate_engagements(scenario)
    by_id = {platform.platform_id: platform for platform in scenario.platforms}
    implicit = {
        (platform.platform_id, weapon.instance_name)
        for platform in scenario.platforms
        for weapon in platform.weapons
        if weapon.weapon_type == "WSF_IMPLICIT_WEAPON"
    }

    lines = ["平台 (初始位置; 各航点坐标与预计到达时刻; 被毁时刻):"]
    for platform in scenario.platforms[:limit]:
        times = waypoint_times_s(platform) if platform.route else []
        parts = [f"初始 {_format_point(platform.position)}"]
        for index, point in enumerate(platform.route):
            when = f" {times[index]:.0f}s" if times else ""
            parts.append(f"航点{index + 1} {_format_point(point.position)}{when}")
        if times is None:
            parts.append("航路缺速度, 无法推算到达时刻")
        elif not platform.route:
            parts.append("无航路")
        if platform.platform_id in destroyed:
            killed_at, killer = destroyed[platform.platform_id]
            text = f"{killed_at:g}s 被交战 {killer} 摧毁"
            missed = sum(1 for t in times or [] if t > killed_at)
            if missed:
                text += f", 之后 {missed} 个航点不会飞到"
            parts.append(text)
        lines.append(f"- {platform.platform_id} [{platform.side.value}]: " + "; ".join(parts))

    ordered = sorted(scenario.engagements, key=lambda e: (e.fire_time_s, e.engagement_id))
    lines += ["", "交战 (按开火时刻; 斜距按匀速航路推算):"]
    for engagement in ordered[:limit]:
        shooter = by_id.get(engagement.shooter_platform)
        target = by_id.get(engagement.target_platform)
        first = position_at(shooter, engagement.fire_time_s) if shooter else None
        second = position_at(target, engagement.fire_time_s) if target else None
        if first is not None and second is not None:
            distance = f"开火时斜距 {slant_range_m(first, second) / 1000:.1f}km"
        else:
            distance = "开火时位置无法推算 (航路已飞完或缺速度)"
        outcome = _engagement_outcome(
            engagement, destroyed,
            (engagement.shooter_platform, engagement.weapon_name) in implicit,
        )
        lines.append(
            f"- {engagement.fire_time_s:g}s {engagement.engagement_id}: "
            f"{engagement.shooter_platform} → {engagement.target_platform}; {distance}; {outcome}"
        )
    if not ordered:
        lines.append("- (无射击计划)")
    elif len(ordered) > limit:
        lines.append(f"- ... 其余 {len(ordered) - limit} 条未列出")

    sides = sorted({platform.side.value for platform in scenario.platforms})
    lines += ["", "双方初始位置最近的地面距离:"]
    for index, left in enumerate(sides):
        for right in sides[index + 1:]:
            pairs = [
                (ground_distance_m(a.position, b.position), a.platform_id, b.platform_id)
                for a in scenario.platforms if a.side.value == left
                for b in scenario.platforms if b.side.value == right
            ]
            meters, a_id, b_id = min(pairs)
            lines.append(f"- {left}–{right}: {meters / 1000:.1f}km ({a_id} ↔ {b_id})")

    lines += ["", "按射击计划推演的存活数:"]
    for side in sides:
        members = [p.platform_id for p in scenario.platforms if p.side.value == side]
        alive = sum(1 for pid in members if pid not in destroyed)
        lines.append(f"- {side}: {alive}/{len(members)}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════
# 规划器
# ═══════════════════════════════════════════════════════


class ScenarioIRPlanner:
    """把自然语言规划抽取成 ScenarioIR, 并在 IR 层做有界自修复。

    修复循环的每一轮都把"上一版 IR + 全部报错"回灌给模型, 让它整份重写。
    对话里只保留最近一版: 大想定的 IR 动辄几万字符, 逐轮累加很快会撑爆上下文,
    而最新一版已经包含了之前各轮的修改。
    只做有界重试 (默认 3 轮): 结构化抽取如果三轮都过不了校验, 通常是需求本身
    缺信息, 再重试只是烧 token。

    ``audit_requirements`` 打开时, 通用检查清零后再花一轮做需求核对: 通用检查不知道
    需求里的具体数值, 实测里问题清零的版本初始距离、穿越时刻、开火时段都没达标, 自由
    文本里还写错了"谁击毁谁"。核对轮把代码推算的事实表交给模型对照原文, 它也占修复轮数。

    ``check_requirements`` 打开时, 拿到第一版合法 IR 后另调用一次模型, 把需求原文里写明
    数值的约束抽成结构化条款 (不计入修复轮数); 之后每一版都由代码逐条核验, 不满足的条款
    和通用问题一起回灌 refine。数值约束交给代码判定后, 核对轮只需判读剩下的要求。
    """

    def __init__(
        self,
        llm: Any,
        registry: CapabilityRegistry = DEFAULT_CAPABILITY_REGISTRY,
        max_repair_rounds: int = 3,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        audit_requirements: bool = False,
        check_requirements: bool = False,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.max_repair_rounds = max(1, max_repair_rounds)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.audit_requirements = audit_requirements
        self.check_requirements = check_requirements
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
        prompt_head: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user",
             "content": self._build_user_prompt(request, hints or {}, gaps)},
        ]
        conversation = list(prompt_head)

        attempts: list[PlanAttempt] = []
        best: PlanResult | None = None
        best_scenario: ScenarioIR | None = None
        # 下一轮提示所基于的版本, 通常就是 best; 只有需求核对的修正多出了问题时,
        # 先在修正版上 refine 一轮, best 仍保留修正前的版本
        working: PlanResult | None = None
        working_scenario: ScenarioIR | None = None
        best_audited = False
        audit_ran = False
        requirements_met = False
        refine_nudged = False
        # 已有合法结果后, 改坏 schema 的修订版只补救一次
        salvaged = False
        # 本轮输出是否源自需求核对的修正 (含为它补的 repair)
        from_audit = False
        # 修订轮的补丁合并到这一版上; 只有对话里带着上一版完整 IR 时才有
        patch_base: dict[str, Any] | None = None
        stage = "extract"
        # 需求数值条款: 拿到第一版合法 IR 后抽一次; None 表示还没抽
        requirement_checks: list[RequirementCheck] | None = None
        requirement_notes: list[str] = []
        # 抽取阶段推理耗尽后的第一次重问不占修复轮数: 实测它常让 refine 和核对少一轮
        round_budget = self.max_repair_rounds
        extract_retried = False
        round_index = 0

        while round_index < round_budget:
            round_index += 1
            try:
                raw = self.llm.chat(
                    conversation,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    json_object=True,
                )
            except Exception as exc:
                if best is None:
                    raise
                # refine 调用超时或被拒时, 已经拿到的合法结果不能跟着丢掉
                attempts.append(PlanAttempt(
                    round_index, stage, False, [f"模型调用失败: {type(exc).__name__}: {exc}"[:400]],
                ))
                break
            # 只在模型正常收尾时补漏写的闭合符: 截断的半截补丁补齐后会被当成完整补丁合并
            finished = getattr(self.llm, "last_finish_reason", None) == "stop"
            payload, parse_errors = _parse_json_object(raw, close_unbalanced=finished)
            patched = payload is not None and _PATCH_KEY in payload
            patch_errors: list[str] = []
            if patched:
                if patch_base is None:
                    payload, parse_errors = None, [
                        f"这一轮对话里没有上一版 IR 可供合并, {_PATCH_KEY} 补丁无法使用; "
                        "请输出完整的 ScenarioIR"
                    ]
                else:
                    merged, patch_errors = apply_ir_patch(patch_base, payload[_PATCH_KEY])
                    # 补丁合并不了时回灌上一版, 让模型对着它重写补丁
                    payload = merged if merged is not None else copy.deepcopy(patch_base)
            if payload is None:
                truncated = getattr(self.llm, "last_finish_reason", None) == "length"
                # 推理模型的思维链与正文共用 max_tokens: 正文为空或只写了一小截, 说明预算
                # 主要耗在了推演上。这和"IR 本身太长"不同, 推演长度每次不一样, 催它少想、
                # 直接输出就值得再试
                thought_out = truncated and self._reasoning_exhausted(raw)
                if thought_out:
                    parse_errors = [
                        *parse_errors,
                        f"模型的推理过程耗尽了 max_tokens={self.max_tokens}, 正文没能完整输出",
                    ]
                elif truncated:
                    parse_errors = [
                        *parse_errors,
                        f"模型输出达到 max_tokens={self.max_tokens} 上限被截断; "
                        f"平台较多时请增大 max_tokens 后重试",
                    ]
                attempts.append(
                    PlanAttempt(round_index, stage, False, parse_errors, raw_excerpt=raw[:400])
                )
                revising = patch_base is not None
                if stage == "audit":
                    nudge = _AUDIT_THOUGHT_OUT_NUDGE
                elif revising:
                    nudge = _REVISION_NUDGE
                else:
                    nudge = _THOUGHT_OUT_NUDGE
                request_message = conversation[-1]["content"]
                # 修订轮被截断 (哪怕推理没占多少) 多半是整份重抄超了预算, 改输出补丁值得再试一次
                worth_retry = thought_out or (
                    truncated and revising and nudge not in request_message
                )
                # 还没有合法 IR 时每轮都值得再试; 已有合法结果时只催一次, 不为 refine 反复烧预算
                if worth_retry and (best is None or not refine_nudged):
                    refine_nudged = best is not None
                    if best is None and stage == "extract" and not extract_retried:
                        extract_retried = True
                        round_budget += 1
                    if nudge not in request_message:
                        conversation = [
                            *conversation[:-1],
                            {"role": "user",
                             "content": request_message + "\n\n" + nudge},
                        ]
                    continue
                if best is not None:
                    break  # 已有可用结果, 不为了 refine 冒风险继续
                if truncated:
                    break  # 同一预算下重新输出完整 IR 只会再次截断
                # 回灌完整输出: 只给前 2000 字时模型看不到自己写过的内容, 只能整版重新推演,
                # 实测两轮都因此耗尽推理预算。没截断的输出长度本身受 max_tokens 约束
                if patch_base is None:
                    conversation = [
                        *prompt_head,
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content":
                            "上一轮输出无法解析成 ScenarioIR: " + "; ".join(parse_errors)
                            + "\n请只输出 JSON 对象本身, 重新给出完整的 ScenarioIR; "
                            "内容沿用上一轮, 只修正报错处。"},
                    ]
                else:
                    # 修订中途: 保留对话里的上一版 IR 和修改要求, 补丁仍以上一版为底
                    conversation = [
                        *conversation[: len(prompt_head) + 2],
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content":
                            "上一轮输出无法解析: " + "; ".join(parse_errors)
                            + "\n请只输出 JSON 对象本身 (补丁或完整的 ScenarioIR 都可以); "
                            "内容沿用上一轮, 只修正报错处。"},
                    ]
                stage = "repair"   # 还没拿到合法 IR, 修硬错误
                continue

            # 模型自己判定超纲时会回 {"__out_of_scope__": "理由"}
            if _OUT_OF_SCOPE_SENTINEL in payload:
                return PlanResult(
                    success=False,
                    attempts=attempts,
                    capability_gaps=gaps,
                    out_of_scope=str(payload[_OUT_OF_SCOPE_SENTINEL]),
                )

            # 核对轮里模型确认需求都满足: 当前 best 就是最终结果
            if stage == "audit" and payload.get(_REQUIREMENTS_MET_SENTINEL) is True:
                attempts.append(PlanAttempt(round_index, stage, True))
                requirements_met = True
                break

            if patch_errors:
                scenario, afsim_code, errors, normalizations = None, "", patch_errors, []
            else:
                # 修订只改了物理航路时把意图跟上, 否则交付的 IR 里两处航路对不上
                synced = (
                    _sync_revised_intents(patch_base, payload) if patch_base is not None else []
                )
                # 先做确定性归一化: 答案唯一的机械问题不必浪费修复轮次
                payload, normalizations = normalize_ir_payload(payload)
                normalizations = [*synced, *normalizations]
                scenario, afsim_code, errors = self._validate_and_compile(payload)
            if scenario is None:
                attempts.append(PlanAttempt(round_index, stage, False, errors, patched=patched))
                if best is not None:
                    # 修订版常常只是 schema 小错 (多写一个字段、类型不对), 整版丢掉会把
                    # 核对或 refine 做对的修改一起丢掉: 补一次 repair; 再坏就保留上一版合法结果
                    if salvaged:
                        break
                    salvaged = True
                patch_base = payload
                # 回灌完整 IR: 截断后模型看不到自己的上一版, "只改报错处"无从谈起
                conversation = [
                    *prompt_head,
                    {"role": "assistant",
                     "content": json.dumps(payload, ensure_ascii=False)},
                    {"role": "user", "content": self._build_repair_prompt(errors)},
                ]
                stage = "repair"
                continue

            if self.check_requirements and requirement_checks is None:
                requirement_checks, requirement_notes = self._extract_requirement_checks(
                    request, scenario
                )
            # 需求条款排在前面: refine 提示只列前若干条, 需求不满足比通用问题更要紧
            advisories = [
                *_requirement_issues(scenario, requirement_checks or []),
                *check_ir_plausibility(scenario),
            ]
            lint = lint_afsim(afsim_code)
            candidate = PlanResult(
                success=True,
                scenario_ir=payload,
                afsim_code=afsim_code,
                lint_diagnostics=[
                    f"{d.severity.value}:{d.code}:{d.message}" for d in lint.diagnostics
                ],
                advisories=advisories,
                acknowledged=acknowledged_engagement_issues(scenario),
                capability_gaps=gaps,
                unverified_capabilities=self._unverified_used(scenario),
                normalizations=normalizations,
            )
            attempts.append(
                PlanAttempt(round_index, stage, True, advisories=advisories, patched=patched)
            )

            # 单调不退化: 合理性问题不增加就接受新版本。
            #
            # 用 <= 而不是 <: 很多改进不体现在条数上。实测里 refine 把"航路需要
            # 103 分钟"改成"62 分钟"(任务时长 60 分钟), 问题条数仍是 2, 用严格
            # 小于会把这个明显更好的版本丢掉。平手时取较新版本 —— 它多看了一轮
            # 反馈。条数变多才拒绝, 所以仍然不会退化。
            if best is None or len(advisories) <= len(best.advisories):
                best = candidate
                best_scenario = scenario
                best_audited = False
                working, working_scenario = candidate, scenario
            elif from_audit:
                # 按需求修正时引入了新的通用问题: 不替换已验收的版本, 但先在修正版上
                # refine 一轮; 改到问题不多于 best 才接受
                working, working_scenario = candidate, scenario
            else:
                working, working_scenario = best, best_scenario
            from_audit = False

            if round_index >= round_budget:
                break

            # 回灌的 IR 与问题清单必须是同一版: 本轮若因问题变多被拒, 就从 best 重新改
            if working.advisories:
                patch_base = working.scenario_ir
                conversation = [
                    *prompt_head,
                    {"role": "assistant",
                     "content": json.dumps(working.scenario_ir, ensure_ascii=False)},
                    {"role": "user",
                     "content": self._build_refine_prompt(working.advisories, working_scenario)},
                ]
                stage = "refine"
                continue

            # 通用检查已清零 (此时 working 就是 best): 按需另起一轮核对需求里的具体数值
            if not self.audit_requirements or best_audited:
                break
            best_audited = audit_ran = from_audit = True
            patch_base = best.scenario_ir
            conversation = [
                *prompt_head,
                {"role": "assistant",
                 "content": json.dumps(best.scenario_ir, ensure_ascii=False)},
                {"role": "user",
                 "content": self._build_audit_prompt(best_scenario, requirement_checks or [])},
            ]
            stage = "audit"

        if best is not None:
            best.attempts = attempts
            best.requirement_checks = [check.to_dict() for check in requirement_checks or []]
            best.requirement_notes = requirement_notes
            if audit_ran:
                best.requirements_audit = "met" if requirements_met else "unconfirmed"
            return best
        return PlanResult(success=False, attempts=attempts, capability_gaps=gaps)

    def _reasoning_exhausted(self, raw: str) -> bool:
        """截断是因为推演吃掉了预算, 而不是 IR 本身太长。"""
        if not raw.strip():
            return True
        usage = getattr(self.llm, "last_usage", None)
        if not isinstance(usage, dict):
            return False  # 拿不到用量就按 IR 太长处理, 不冒重复截断的风险
        completion = usage.get("completion_tokens") or 0
        details = usage.get("completion_tokens_details") or {}
        reasoning = details.get("reasoning_tokens") or 0
        # 推演占了一半以上: 少想一点, 同样的预算就够写完正文
        return completion > 0 and reasoning * 2 >= completion

    def _extract_requirement_checks(
        self, request: str, scenario: ScenarioIR
    ) -> tuple[list[RequirementCheck], list[str]]:
        """让模型把需求里写明数值的约束抽成条款; 失败时本次不核验条款, 只记一条说明。

        单独一次调用、不带 IR: 模型只读需求原文和平台清单, 不必推演想定。
        """
        hints = {intent.platform_id: intent.type_hint for intent in scenario.mission.platforms}
        roster = "\n".join(
            f"- {p.platform_id} [{p.side.value}] {hints.get(p.platform_id, '')}".rstrip()
            for p in scenario.platforms
        )
        messages = [
            {"role": "system", "content": _REQUIREMENT_CHECKS_PROMPT},
            {"role": "user", "content":
                f"=== 规划正文 ===\n{request.strip()}\n\n=== 平台清单 ===\n{roster}"},
        ]
        try:
            raw = self.llm.chat(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                json_object=True,
            )
        except Exception as exc:
            return [], [f"条款抽取调用失败, 本次不核验需求条款: {type(exc).__name__}: {exc}"[:300]]
        finished = getattr(self.llm, "last_finish_reason", None) == "stop"
        payload, errors = _parse_json_object(raw, close_unbalanced=finished)
        if payload is None:
            return [], [("条款抽取结果无法解析, 本次不核验需求条款: " + "; ".join(errors))[:300]]
        return parse_requirement_checks(payload, scenario)

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
        if len(errors) > 25:
            listed += f"\n- (另有 {len(errors) - 25} 条未列出)"
        return (
            "上一版 ScenarioIR 没通过校验, 具体错误如下:\n"
            f"{listed}\n\n"
            "请逐条修掉这些错误, 只改报错涉及的地方, 其余内容保持不变。\n\n"
            + _PATCH_FORMAT
        )

    def _build_refine_prompt(
        self, advisories: list[str], scenario: ScenarioIR | None = None
    ) -> str:
        limit = 40
        listed = "\n".join(f"- {a}" for a in advisories[:limit])
        if len(advisories) > limit:
            listed += f"\n- (另有 {len(advisories) - limit} 条未列出, 修完上面这些后会再列出)"
        prompt = (
            "这一版 ScenarioIR 已经通过 schema 校验并能编译成合法 AFSIM, 但存在"
            f"以下 {len(advisories)} 条问题 (以\"需求\"开头的是代码按需求原文的数值条款核验出的"
            "不满足项, 其余是战术合理性问题), 会让想定不符合需求或跑得起来却没有意义:\n"
            f"{listed}\n\n"
            "请只针对上述问题做最小修改 (调整坐标、补航路点、改传感器名、调整开火时刻"
            "和目标分配等), 保持任务意图、平台构成和 platform_id 不变。问题里附带的可探测"
            "时段、所需速度等数值已经算好, 直接采用。一轮修不完时可以先修一部分: 问题总数"
            "不增加就会被接受, 没修完的下一轮会再列出。\n"
            "修改时同时核对原始需求里明确给出的数值约束 (时间窗、交战距离/包络、到达时刻、"
            "数量上限), 不要为修上述问题破坏已满足的约束。\n"
            # "每架都要有射击计划"加上开火即毁, 最先被击毁的那架必然打不出;
            # 按条数验收时, 删掉这条计划就能"清零", 却违背了需求
            "有的问题可能与需求的硬性要求冲突而无法消除 (例如需求要求每架都有射击计划, "
            "最先被击毁的那架就来不及开火): 这类问题保留原样并在 mission.assumptions 里"
            "说明, 不要为了消掉问题删减需求要求的平台、航点或射击计划。"
        )
        if scenario is not None:
            prompt += (
                "\n\n按当前 IR 推算的时间线 (匀速、忽略转弯; 未注明摧毁的平台按射击计划"
                "全程存活), 核对时间窗和关键航段时直接用, 不必自己重算:\n"
                + "\n".join(_timeline_digest(scenario))
            )
        return prompt + "\n\n" + _PATCH_FORMAT

    def _build_audit_prompt(
        self, scenario: ScenarioIR, checks: list[RequirementCheck] | None = None
    ) -> str:
        sentinel = f'{{"{_REQUIREMENTS_MET_SENTINEL}": true}}'
        # 走到核对轮时问题已清零, 抽出的条款都已核验通过; 列出来免得模型再逐条重算
        verified = ""
        if checks:
            verified = (
                "\n\n以下需求条款已由代码按当前 IR 核验通过, 不必再核对:\n"
                + "\n".join(f"- {describe_requirement_check(check)}" for check in checks)
            )
        return (
            "这一版 ScenarioIR 已通过 schema、编译和静态检查, 校验器也没有再发现通用的战术"
            "合理性问题。但校验器只查通用规则, 不知道原始规划正文里的具体数值要求。下面是"
            "代码按当前 IR 推算的事实 (匀速、忽略转弯与加减速; 毁伤按射击计划和开火即毁推演), "
            "请直接拿它逐条对照原始规划正文, 不必自己重算:\n"
            "1. 正文明确给出的数值约束 —— 初始距离与间隔、各平台或编队到达/穿越/撤离的"
            "时间窗、开火时段与波次、交战距离、数量与存活要求等 —— 是否都满足; 承担时间窗内"
            "航段的平台必须活到那时;\n"
            "2. mission.assumptions 与 selected_coa.rationale/summary 里关于时刻、存活、"
            "\"谁击毁谁\"的描述是否与下面的事实一致。\n\n"
            f"全部满足且描述一致时, 只输出 {sentinel}, 不要输出别的内容。\n"
            "否则做最小修改 (调整坐标、速度、航点、开火时刻和目标分配, 改正不符的描述), 保持"
            "任务意图、平台构成和 platform_id 不变。修改时不要引入新的交战时序问题; 与需求冲突、"
            "确实无法同时满足的条目, 在 mission.assumptions 里说明。"
            + verified
            + "\n\n" + _PATCH_FORMAT
            + "\n\n=== 按当前 IR 推算的事实 ===\n"
            + _fact_sheet(scenario)
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
        single_report = _validator_report(afsim_code)
        if single_report["errors"]:
            return None, "", [
                f"单文件验证失败: {issue['description']}"
                for issue in single_report["issues"] if issue["severity"] == "error"
            ]
        try:
            _checked_project(scenario)
        except CompileError as exc:
            return None, "", [str(exc)]
        return scenario, afsim_code, []


# ═══════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _parse_json_object(
    raw: str, close_unbalanced: bool = False
) -> tuple[dict[str, Any] | None, list[str]]:
    """从模型输出里取出 JSON 对象。

    即使开了服务端 JSON 模式, 模型偶尔还是会包一层 markdown 代码块或在前后带上
    说明文字, 所以这里保留一层容错剥离。``close_unbalanced`` 为真 (模型正常收尾)
    时, 还会补上漏写的一两个闭合符 (结尾或中途)。
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

    if start == -1:
        return None, ["无法从模型输出里解析出 JSON 对象"]
    # 实测: 5 万字的 IR 完整写完后多了一个 "}", 整版被当成非法 JSON, 重写时又耗尽了推理预算。
    # 只容忍尾部多余的闭合符: 尾部若还有字段, 说明前面提前闭合了, 解出来的对象是残缺的
    try:
        parsed, consumed = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError as exc:
        closed = _close_unbalanced(text[start:]) if close_unbalanced else None
        if closed is not None:
            return closed, []
        # 报出位置: 几万字的 IR 里只说"不是合法 JSON", 模型只能整版重想
        return None, [f"JSON 语法错误 ({exc.msg}, 第 {exc.pos} 个字符): {_json_context(text, exc.pos)}"]
    if not text[consumed:].strip("}] \t\r\n`"):
        return parsed, []
    return None, [
        f"JSON 对象在第 {consumed} 个字符处提前闭合, 后面还有内容: {_json_context(text, consumed)}"
    ]


def _json_context(text: str, position: int, width: int = 60) -> str:
    """取出错位置前后各一小段, 用 ⟦此处⟧ 标出位置。"""
    before = text[max(0, position - width):position]
    after = text[position:position + width]
    return (before + "⟦此处⟧" + after).replace("\n", " ")


_MAX_MISSING_CLOSERS = 2


def _close_unbalanced(text: str) -> dict[str, Any] | None:
    """补上漏写的闭合符; 补不了或补完仍不合法时返回 None。

    实测: 模型正常收尾的补丁漏了最外层的 "}", 整轮被当成非法 JSON 丢掉; 另一次是
    列表中途漏了 "]" 就写了外层的 "}", 修复轮模型又删掉别处的 "}" 去凑配对, 顶层
    字段全嵌进了 mission, 四轮都没修回来。遇到配不上的闭合符时, 在它前面补齐栈里
    更内层的闭合符; 结尾缺的照样补上。只在正文以闭合符结尾、没停在字符串中间、
    一共缺的不超过两个时补 —— 缺得多说明结构本身乱了, 补出来的嵌套未必是模型的本意。
    """
    body = text.rstrip("` \t\r\n")
    if not body.endswith(("}", "]")):
        return None
    pending: list[str] = []
    repaired: list[str] = []
    inserted = 0
    in_string = escaped = False
    for char in body:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            pending.append("}" if char == "{" else "]")
        elif char in "}]":
            if char not in pending:
                return None
            while pending[-1] != char:
                repaired.append(pending.pop())
                inserted += 1
            pending.pop()
        repaired.append(char)
    missing = inserted + len(pending)
    if in_string or not missing or missing > _MAX_MISSING_CLOSERS:
        return None
    try:
        parsed = json.loads("".join(repaired) + "".join(reversed(pending)))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def apply_ir_patch(
    base: dict[str, Any], patch: Any
) -> tuple[dict[str, Any] | None, list[str]]:
    """把修订轮输出的补丁合并到上一版 IR, 返回 ``(合并结果, 报错)``。

    对象按 RFC 7386 合并 (值为 null 即删键); ``_PATCH_LIST_IDS`` 登记的列表按元素 id
    合并, 其余列表整体替换。有一处确定不了要改哪个元素就整份拒绝, 不去猜。
    """
    if not isinstance(patch, dict):
        return None, [f"{_PATCH_KEY} 的值应该是对象, 实际是 {type(patch).__name__}"]
    errors: list[str] = []
    merged = _merge_fields(copy.deepcopy(base), patch, (), errors)
    return (None, errors) if errors else (merged, [])


def _merge_fields(
    base: Any, patch: dict[str, Any], path: tuple[str, ...], errors: list[str]
) -> dict[str, Any]:
    merged = base if isinstance(base, dict) else {}
    for key, value in patch.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = _merge_value(merged.get(key), value, (*path, key), errors)
    return merged


def _merge_value(base: Any, patch: Any, path: tuple[str, ...], errors: list[str]) -> Any:
    id_key = _PATCH_LIST_IDS.get(path)
    if id_key is not None:
        if isinstance(patch, list):
            existing = base if isinstance(base, list) else []
            return _merge_list_by_id(existing, patch, path, id_key, errors)
        errors.append(
            f"补丁里 {'.'.join(path)} 应该是列表, 每个元素用 {id_key} 标出要改的是哪一个; "
            f"实际是 {type(patch).__name__}"
        )
        return base
    if isinstance(patch, dict):
        return _merge_fields(base, patch, path, errors)
    return patch


def _merge_list_by_id(
    base: list[Any], patch: list[Any], path: tuple[str, ...], id_key: str, errors: list[str]
) -> list[Any]:
    """按 id 合并: 已有的就地合并, 新 id 追加, 带删除标记的去掉。"""
    where = ".".join(path)
    merged = list(base)
    index: dict[str, int] = {}
    for position, item in enumerate(merged):
        if isinstance(item, dict) and isinstance(item.get(id_key), str):
            index.setdefault(item[id_key], position)
    deleted: set[int] = set()
    for number, item in enumerate(patch, start=1):
        item_id = item.get(id_key) if isinstance(item, dict) else None
        if not isinstance(item_id, str):
            errors.append(
                f"补丁里 {where} 的第 {number} 个元素缺少 {id_key}, 无法确定要改哪一个; "
                f"这个列表按 {id_key} 合并, 每个元素都要带上它"
            )
            continue
        changes = {key: value for key, value in item.items() if key != _DELETE_MARK}
        if item.get(_DELETE_MARK) is True:
            if item_id in index:
                deleted.add(index[item_id])
            else:
                errors.append(f"补丁要删除的 {where} 元素 {item_id} 在上一版里不存在")
        elif item_id in index:
            position = index[item_id]
            merged[position] = _merge_fields(merged[position], changes, path, errors)
        else:
            index[item_id] = len(merged)
            merged.append(_merge_fields({}, changes, path, errors))
    return [item for position, item in enumerate(merged) if position not in deleted]


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


def _checked_project(scenario: ScenarioIR) -> tuple[dict[str, str], dict[str, Any]]:
    """校验将要写盘的同一份文件字典，错误定位回原 include 文件。"""
    files = emit_project(scenario)
    lint = lint_project(files)
    if not lint.is_valid:
        errors = [
            f"{d.file}:{d.line} {d.code}: {d.message}"
            for d in lint.diagnostics if d.severity.value == "error"
        ]
        raise CompileError("最终项目检查失败: " + "; ".join(errors))
    expanded, sources, _ = expand_project(files)
    report = _validator_report(expanded)
    for issue in report["issues"]:
        line = issue["line"]
        if line is not None and 0 < line <= len(sources):
            issue["expanded_line"] = line
            issue["file"], issue["line"] = sources[line - 1]
    if report["errors"]:
        errors = [
            f"{i.get('file')}:{i['line']} {i['rule']}: {i['description']}"
            for i in report["issues"] if i["severity"] == "error"
        ]
        raise CompileError("最终项目验证失败: " + "; ".join(errors))
    return files, {
        "valid": True,
        "runtime_verified": False,
        "validator": report,
        "diagnostics": [
            {"severity": d.severity.value, "code": d.code,
             "message": d.message, "file": d.file, "line": d.line}
            for d in lint.diagnostics
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
    # 在创建目录或覆盖任何文件前验证单文件和完整 include 项目。
    if not lint_afsim(afsim_code).is_valid or _validator_report(afsim_code)["errors"]:
        raise CompileError("单文件快照验证失败，项目未写入")
    files, validation = _checked_project(scenario)
    single_path = project_root / "scenario_single.txt"
    if not single_path.resolve().is_relative_to(project_root.resolve()):
        raise CompileError("单文件快照路径越界")
    written = write_project(scenario, project_root, files=files)

    # 同时写一份单文件版本 (语义等价, 方便 diff / 快速查看)
    single_path.write_text(afsim_code, encoding="utf-8")

    return {
        "project_root": str(project_root.resolve()),
        "entry": str((project_root / "main.txt").resolve()),
        "files": sorted(written.keys()),
        "single_file": str(single_path.resolve()),
        "validation": validation,
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


_PLANNER_PARAMS: dict[str, Any] = {
    "max_repair_rounds": {
        "type": "integer",
        "description": "Max IR-level repair rounds, the requirements audit included. Default: 3.",
        "minimum": 1,
        "maximum": 8,
    },
    "audit_requirements": {
        "type": "boolean",
        "description": (
            "Once the IR passes every generic check, spend one more round asking the LLM to "
            "compare code-computed facts (waypoint ETAs, fire-time ranges and outcomes, initial "
            "separation, survivors) against the numeric constraints in the request. Default: true."
        ),
    },
    "check_requirements": {
        "type": "boolean",
        "description": (
            "After the first valid IR, make one extra LLM call (not counted as a repair round) "
            "that extracts the numeric constraints stated in the request (side separation, "
            "formation spacing, fire windows and phases, engagement range, area-crossing times, "
            "altitude/speed bands) as structured checks. Code verifies every revision against "
            "them and feeds violations back into refinement. Default: true."
        ),
    },
}


def _make_planner(kwargs: dict[str, Any]) -> "ScenarioIRPlanner":
    return ScenarioIRPlanner(
        llm=_make_llm(kwargs),
        max_repair_rounds=int(kwargs.get("max_repair_rounds", 3)),
        max_tokens=int(kwargs.get("max_tokens") or 8192),
        audit_requirements=bool(kwargs.get("audit_requirements", True)),
        check_requirements=bool(kwargs.get("check_requirements", True)),
    )


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
                **_PLANNER_PARAMS,
                **_LLM_PARAMS,
            },
            "required": ["request"],
        }

    async def execute(self, **kwargs: Any) -> str:
        try:
            request = (kwargs.get("request") or "").strip()
            if not request:
                return "Error: 'request' must not be empty."

            planner = _make_planner(kwargs)
            result = planner.plan(request, _collect_hints(kwargs))

            return json.dumps(
                {
                    "success": result.success,
                    "out_of_scope": bool(result.out_of_scope),
                    "reason": result.out_of_scope,
                    "rounds": result.rounds,
                    "scenario_ir": result.scenario_ir,
                    "plausibility_advisories": result.advisories,
                    "acknowledged_advisories": result.acknowledged,
                    "requirements_audit": result.requirements_audit,
                    "requirement_checks": result.requirement_checks,
                    "requirement_notes": result.requirement_notes,
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
                            "patched": a.patched,
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
            validator = _validator_report(afsim_code)
            _, project_validation = _checked_project(scenario)
            output: dict[str, Any] = {
                "success": lint.is_valid and validator["errors"] == 0,
                "project_validation": project_validation,
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
                "validator": validator,
                "afsim_code": afsim_code,
            }
            if kwargs.get("output_path") and output["success"]:
                output["project"] = _write_project_output(
                    kwargs["output_path"], scenario, afsim_code
                )
            return json.dumps(output, ensure_ascii=False, indent=2)
        except CompileError as exc:
            return json.dumps(
                {"success": False, "stage": "project_validation", "errors": [str(exc)]},
                ensure_ascii=False, indent=2,
            )
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
            "on the IR, never by patching generated text. Both the single file and "
            "the complete include project are statically validated before writing. Returns the scenario code, "
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
                **_PLANNER_PARAMS,
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

            planner = _make_planner(kwargs)
            result = planner.plan(request, _collect_hints(kwargs))

            trace = [
                {"round": a.round, "stage": a.stage, "ok": a.ok,
                 "errors": a.errors, "advisories": a.advisories, "patched": a.patched}
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

            validator = _validator_report(result.afsim_code)
            output: dict[str, Any] = {
                "success": validator["errors"] == 0,
                "runtime_verified": False,
                "rounds": result.rounds,
                "scenario_id": result.scenario_ir.get("scenario_id", ""),
                "lines_of_code": len(result.afsim_code.splitlines()),
                "validator": validator,
                "lint_diagnostics": result.lint_diagnostics,
                "plausibility_advisories": result.advisories,
                "acknowledged_advisories": result.acknowledged,
                "requirements_audit": result.requirements_audit,
                "requirement_checks": result.requirement_checks,
                "requirement_notes": result.requirement_notes,
                "capability_gaps": _gaps_payload(result.capability_gaps),
                "unverified_capabilities": result.unverified_capabilities,
                "normalizations": result.normalizations,
                "scenario_ir": result.scenario_ir,
                "trace": trace,
            }
            if kwargs.get("output_path") and output["success"] and result.scenario_ir is not None:
                scenario_obj = ScenarioIR.model_validate(result.scenario_ir)
                output["project"] = _write_project_output(
                    kwargs["output_path"], scenario_obj, result.afsim_code
                )
            if kwargs.get("include_code", True):
                output["afsim_code"] = result.afsim_code
            return json.dumps(output, ensure_ascii=False, indent=2)
        except CompileError as exc:
            return json.dumps(
                {"success": False, "stage": "project_validation", "errors": [str(exc)]},
                ensure_ascii=False, indent=2,
            )
        except Exception as exc:
            return f"Error in nl_to_afsim: {exc}"


def _collect_hints(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "center_lat": kwargs.get("center_lat"),
        "center_lon": kwargs.get("center_lon"),
        "duration_s": kwargs.get("duration_s"),
        "scenario_id": kwargs.get("scenario_id"),
    }
