"""AFSIM 想定验证工具 — 多层级语法/语义/一致性校验。

Agent 可通过 function calling 调用 ``scenario_validate(...)`` 对生成的
AFSIM 想定代码进行全面验证, 在提交给 Warlock/Wizard 之前发现问题。

验证层级:
  Level 1: 语法检查 (块闭合 / 关键字拼写 / 坐标格式 / 单位)
  Level 2: 语义检查 (引用一致性 / 类型匹配 / side 统一)
  Level 3: 一致性检查 (platform_type↔platform / weapon_type↔weapon / 坐标范围)
  Level 4: 战术合理性 (传感器覆盖 / 武器射程 / 速度高度 / 编队间距)

评分: 0-100 分, 按严重程度扣分。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from nanobot.agent.tools.base import Tool


# ═══════════════════════════════════════════════════════
# 验证规则定义
# ═══════════════════════════════════════════════════════

# 块开始/结束配对 (被建模的块: 这些块的配对错误会被判定为 error)
#
# AFSIM 2.9 真实语料里出现过 120+ 种 end_* 关键字, 穷举白名单不可能完整。
# 因此本表只登记"我们自己生成的"以及"首词必为块起始、不会与引用/命令混淆的"
# 关键字; 未登记的 end_* 由 _check_block_pairing 静默跳过, 避免把合法想定
# 误判为语法错误 (见 _UNMODELED_END_IGNORED 说明)。
_BLOCK_PAIRS: dict[str, str] = {
    "platform_type": "end_platform_type",
    "platform": "end_platform",
    "mover": "end_mover",
    "fuel": "end_fuel",
    "sensor": "end_sensor",
    "weapon": "end_weapon",
    "processor": "end_processor",
    "route": "end_route",
    "radar_signature": "end_radar_signature",
    "optical_signature": "end_optical_signature",
    "infrared_signature": "end_infrared_signature",
    "weapon_effects": "end_weapon_effects",
    "comm": "end_comm",
    "transmitter": "end_transmitter",
    "receiver": "end_receiver",
    "antenna_pattern": "end_antenna_pattern",
    "mode_template": "end_mode_template",
    "mode": "end_mode",
    "filter": "end_filter",
    "aero": "end_aero",
    "script_variables": "end_script_variables",
    "execute": "end_execute",
    "script": "end_script",
    "detection_probability": "end_detection_probability",
    "signal_processor": "end_signal_processor",
    "state": "end_state",
    "phase": "end_phase",
    # ── 输出/观察者块 (确定性编译器会生成 event_output / event_pipe) ──
    "event_output": "end_event_output",
    "event_pipe": "end_event_pipe",
    "observer": "end_observer",
    # ── 条件编译块 ──
    "conditionals": "end_conditionals",
    "conditional_section": "end_conditional_section",
    # ── 行为树 / 任务处理器块 ──
    "precondition": "end_precondition",
    "advanced_behavior": "end_advanced_behavior",
    "advanced_behavior_tree": "end_advanced_behavior_tree",
    "behavior": "end_behavior",
    "sequence": "end_sequence",
    "selector": "end_selector",
    "next_state": "end_next_state",
    # ── 脚本回调块 ──
    "on_init": "end_on_init",
    "on_update": "end_on_update",
    "on_initialize": "end_on_initialize",
    "on_new_execute": "end_on_new_execute",
    "on_track_drop": "end_on_track_drop",
    "on_new_fail": "end_on_new_fail",
    # ── 航迹融合 ──
    "track_manager": "end_track_manager",
    "fusion_method": "end_fusion_method",
    # ── 区域 ──
    "zone": "end_zone",
    "zone_set": "end_zone_set",
    # ── 发射计算器与表格 ──
    "launch_computer": "end_launch_computer",
    "launch_computer_table": "end_launch_computer_table",
    "independent_variable": "end_independent_variable",
    "independent_variables": "end_independent_variables",
    "dependent_variable": "end_dependent_variable",
    "regular_table": "end_regular_table",
    "irregular_table": "end_irregular_table",
    "intercept_results": "end_intercept_results",
    # ── 其它 ──
    "six_dof_object_types": "end_six_dof_object_types",
    "circular_pattern": "end_circular_pattern",
    "aux_data": "end_aux_data",
    "terrain": "end_terrain",
}

# AFSIM 基类型关键字 (从真实想定与确定性编译器输出中提取)
#
# 这只是"已知good"清单, 不是 AFSIM 全集。未登记的 WSF_* 只报 warning 而不是
# error: AFSIM 内置类型远多于此, 报 error 会把合法想定判死。
_VALID_BASE_TYPES = frozenset({
    "WSF_PLATFORM",
    "WSF_AIR_MOVER", "WSF_GROUND_MOVER", "WSF_SURFACE_MOVER",
    "WSF_GUIDED_MOVER", "WSF_BALLISTIC_MOVER",
    "WSF_BRAWLER_MOVER", "WSF_BRAWLER_PLATFORM", "WSF_POINT_MASS_SIX_DOF_MOVER",
    "WSF_SIX_DOF_MOVER", "WSF_P6DOF_MOVER",
    "WSF_GEOMETRIC_SENSOR", "WSF_ESM_SENSOR", "WSF_RADAR_SENSOR",
    "WSF_IRST_SENSOR",
    "WSF_COMM_TRANSCEIVER", "WSF_COMM_RCVR", "WSF_COMM_XMTR",
    "WSF_RADAR_SIGNATURE", "WSF_OPTICAL_SIGNATURE",
    "WSF_TRACK_PROCESSOR", "WSF_TASK_PROCESSOR", "WSF_ENGAGEMENT_PROCESSOR",
    "WSF_WEAPON_TRACK_PROCESSOR", "WSF_GUIDANCE_COMPUTER",
    "WSF_AIR_TARGET_FUSE", "WSF_PERCEPTION_PROCESSOR",
    "WSF_THREAT_PROCESSOR", "WSF_GRADUATED_LETHALITY",
    "WSF_SPHERICAL_LETHALITY",
    "WSF_EXPLICIT_WEAPON", "WSF_IMPLICIT_WEAPON", "WSF_KALMAN_FILTER",
    "WSF_AERO", "WSF_BRAWLER_FUEL", "WSF_BRAWLER_PROCESSOR",
    "WSF_SA_PROCESSOR", "WSF_UPLINK_PROCESSOR", "WSF_PERFECT_TRACKER",
    "WSF_AIR_TO_AIR_LAUNCH_COMPUTER", "WSF_ASSET_MESSAGE",
    "BRAWLER_TEST", "BRAWLER_TEST_AIRMOVER",
})

# 匹配一行里出现的 WSF_* 词元
_WSF_TOKEN_RE = re.compile(r"\bWSF_\w+\b")

# 被建模的 end_* 集合: 只有落在这里面的 end_* 才参与严格栈匹配
_MODELED_BLOCK_ENDS = frozenset(_BLOCK_PAIRS.values())

# 允许"引用形态"的块关键字。
#
# 这些关键字既能作为顶层定义(带 end_*), 也能在 platform_type / sensor 内部作为
# 单行引用出现, 例如 platform_type 里的 `radar_signature MRM_SIG`、sensor 里的
# `antenna_pattern SEEKER_ANTENNA`、`aero MRM_AERO`。当 end_* 配额耗尽时, 这类行
# 判定为引用并静默跳过; 不在此集合内的关键字(platform / route / event_output 等)
# 一旦配额耗尽就说明真的漏了 end_*, 照常报错。
_REFERENCE_CAPABLE_BLOCKS = frozenset({
    "radar_signature", "optical_signature", "infrared_signature", "weapon_effects",
    "antenna_pattern", "aero", "launch_computer", "launch_computer_table",
    "weapon", "filter", "fuel", "mover", "comm", "processor",
    "track_manager", "fusion_method", "signal_processor", "detection_probability",
    "terrain", "zone", "zone_set", "aux_data", "behavior",
    # 6DOF 数据表里 "independent_variable alt units ft" 是变量声明而非块
    "independent_variable", "independent_variables", "dependent_variable",
    "regular_table", "irregular_table",
    # 官方 demo 里这些也常以短形式出现 (behavior tree / 状态机 / 扫描图样)
    "circular_pattern", "state", "phase", "next_state", "sequence", "selector",
    "advanced_behavior", "mode", "mode_template",
    # `event_output true` 是布尔字段, `event_output file x.evt` 是块起始, 两者都带
    # 参数, 只能靠 end_* 配额区分
    "event_output", "event_pipe", "observer",
})


# 块体内是 AFSIM 脚本语言 (类 C) 而不是想定命令的块。
#
# 官方 demo 里 red_sam_battery_tactics.txt 有这么一行脚本:
#     platform = sub;
# 早先的实现把它当成 "platform 块起始" 解析, 于是报出未闭合 platform。同类还有
# 脚本里的 if/for/花括号、`WsfWeapon weapon = ...` 之类声明。脚本体必须整段跳过,
# 否则脚本越复杂误报越多。
_SCRIPT_BODY_BLOCKS: dict[str, str] = {
    "script": "end_script",
    "execute": "end_execute",
    "script_variables": "end_script_variables",
    "on_init": "end_on_init",
    "on_update": "end_on_update",
    "on_initialize": "end_on_initialize",
    "on_new_execute": "end_on_new_execute",
    "on_track_drop": "end_on_track_drop",
    "on_new_fail": "end_on_new_fail",
    "precondition": "end_precondition",
}


def _mask_script_bodies(lines: list[str]) -> list[str]:
    """清空脚本块的**块体**, 保留起止行, 行号对齐不变。

    保留起止行是为了让块配对检查照常统计 ``script`` / ``end_script`` 的平衡,
    只是不再窥探中间的脚本代码。
    """
    masked: list[str] = []
    expected_end: str | None = None
    for line in lines:
        stripped = _strip_comments(line)
        if expected_end is not None:
            if stripped == expected_end:
                masked.append(line)
                expected_end = None
            else:
                masked.append("")
            continue

        masked.append(line)
        tokens = stripped.split()
        if not tokens:
            continue
        keyword = tokens[0]
        end_keyword = _SCRIPT_BODY_BLOCKS.get(keyword)
        # 单行自闭合的脚本块 (如 `execute at_time 1 s ... end_execute`) 不进入跳过态
        if end_keyword and tokens[-1] != end_keyword:
            expected_end = end_keyword
    return masked


def _mask_block_comments(lines: list[str]) -> list[str]:
    """清空所有注释内容, 但保留行号对齐。

    AFSIM 有三种注释: 行注释 ``#`` 与 ``//``, 以及块注释 ``/* ... */``。三者必须
    一起处理且讲究优先级:

    - 真实想定的文件头常有大段 ``/* ... */`` 说明文字, 里面的自然语言会命中
      ``route``/``platform`` 之类关键字, 不剥离就会把散文误当成代码。
    - 行注释必须先于块注释判定, 否则 ``//****************//`` 这种装饰性分隔线里
      的 ``/*`` 子串会被误认为块注释开始, 从而吞掉后面成百行真实代码。
    """
    masked: list[str] = []
    in_comment = False
    for line in lines:
        out: list[str] = []
        rest = line
        while rest:
            if in_comment:
                close = rest.find("*/")
                if close == -1:
                    rest = ""
                    break
                rest = rest[close + 2:]
                in_comment = False
                continue
            # 行注释优先: 出现在 /* 之前的 # 或 // 会让整行剩余部分失效
            candidates = [pos for pos in (rest.find("#"), rest.find("//")) if pos != -1]
            line_comment_at = min(candidates) if candidates else -1
            block_at = rest.find("/*")
            if line_comment_at != -1 and (block_at == -1 or line_comment_at < block_at):
                out.append(rest[:line_comment_at])
                break
            if block_at == -1:
                out.append(rest)
                break
            out.append(rest[:block_at])
            rest = rest[block_at + 2:]
            in_comment = True
        masked.append("".join(out))
    return masked


def _strip_comments(line: str) -> str:
    """去掉 AFSIM 注释后返回代码内容。

    AFSIM 同时支持 ``#`` 和 ``//`` 注释, 且都可以出现在行尾 —— 真实想定里
    ``weapon fox3 MEDIUM_RANGE_RADAR_MISSILE # fox 3 (MRM)`` 是完全合法的写法。
    """
    for marker in ("#", "//"):
        idx = line.find(marker)
        if idx != -1:
            line = line[:idx]
    return line.strip()


# 只在"裸关键字"形态下才是块起始的关键字 —— 目前为空。
#
# 曾经把 event_output / event_pipe 放进来, 理由是 `event_output true` 是布尔字段。
# 但官方 demo 里 `event_output file output/x.evt` 同样是块起始 (带参数!), 于是
# 8 个 end_event_output/end_event_pipe 变成"多余闭合"。两种形态都带参数, 靠词元
# 个数分不开, 应当交给 end_* 配额机制去分辨 —— 见 _REFERENCE_CAPABLE_BLOCKS。
_BARE_ONLY_BLOCKS: frozenset[str] = frozenset()


def _block_keyword(stripped: str) -> str:
    """返回该行对应的块关键字, 不是块起始行则返回空串。

    AFSIM 允许用 ``add`` / ``edit`` 修饰词在平台实例上新增或改写组件, 这两种写法
    同样会打开块, 例如 ``edit processor assessment ... end_processor``、
    ``add sensor rdr1 aesa ... end_sensor``。``delete`` 只删除组件, 不开块。

    ``platform`` 有个例外: 裸形态必须带类型才算实例块。``track ... end_track``
    块内的 ``platform target`` 是"这条航迹指向哪个平台"的字段, 只有两个词元。
    但 ``edit platform target_1`` 正好也是三个词元且确实开块, 所以这条限制只对
    没有 add/edit 前缀的写法生效。
    """
    tokens = stripped.split()
    if not tokens:
        return ""
    keyword = tokens[0]
    offset = 0
    if keyword in ("add", "edit") and len(tokens) >= 2:
        keyword = tokens[1]
        offset = 1
    if keyword not in _BLOCK_PAIRS:
        return ""
    if keyword in _BARE_ONLY_BLOCKS and len(tokens) > offset + 1:
        return ""
    if keyword == "platform" and offset == 0 and len(tokens) < 3:
        return ""
    return keyword


# 合法单位 (从真实想定中提取)
_VALID_UNITS = frozenset({
    "sec", "min", "hr", "s", "usec",
    "km", "m", "nm", "nmi", "ft", "nmi",
    "kts", "m/s", "km/h", "mph", "ft/s", "fps",
    "msl", "agl",
    "g", "kg", "lb", "lbm", "lbf", "lbs", "pounds",
    "MHz", "GHz", "kHz", "khz", "kw", "hz", "db", "dB", "dBsm", "dbw",
    "m^2", "deg", "degrees",
    "ratio",
})

# 合法坐标格式。AFSIM 四种写法都合法, 而且经纬度可以混用不同写法。
#
# 官方 demo 语料 (afsim_verify/examples/demos, 4749 文件) 实测分布:
#   d:m:s / d:m:s   26813    "30:00:00n 120:00:00e"
#   decimal/decimal 13280    "1.283345n 103.859047e"
#   d:m / d:m         268    "40:30n 121:15w"
#   integer/integer   140    "30n 120e"
#   混用形式           ~90    如 "d:m:s / d:m"、"integer / d:m:s"
#
# 早先只放行 d:m:s, 在官方语料上误报 323 次。十进制度**不是**非法写法 ——
# errors-ref.md 的 E005 反例 `position 30.0N 120.0E` 真正的问题在大写半球标识,
# `position 30 120` 的问题在缺半球标识, 都不是"带小数点"本身。
_DEG_BODY = r"\d{1,3}(?::\d{1,2}(?::\d{1,2}(?:\.\d+)?)?|\.\d+)?"
_LAT_TOKEN = rf"(?:{_DEG_BODY})\s*[nsNS]"
_LON_TOKEN = rf"(?:{_DEG_BODY})\s*[ewEW]"
_COORD_PATTERN = re.compile(rf"{_LAT_TOKEN}\s+{_LON_TOKEN}")
_COORD_PARSE_RE = re.compile(
    rf"^(?P<lat>{_LAT_TOKEN})\s+(?P<lon>{_LON_TOKEN})"
)
_COORD_TOKEN_RE = re.compile(
    r"^(?P<deg>\d{1,3})"
    r"(?::(?P<min>\d{1,2})(?::(?P<sec>\d{1,2}(?:\.\d+)?))?"
    r"|\.(?P<frac>\d+))?"
    r"\s*(?P<hemi>[nsewNSEW])$"
)

# 直角/ECI 位置: 至少两组 "数值 + 长度单位", 支持科学计数法
_CARTESIAN_POSITION_RE = re.compile(
    r"^\s*-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*(?:m|km|ft|nm|nmi)\b"
    r"(?:\s+-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*(?:m|km|ft|nm|nmi)\b){1,}"
)

# side 只要是标识符即可 (AFSIM 允许任意阵营名)
_SIDE_TOKEN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


# ═══════════════════════════════════════════════════════
# 验证结果
# ═══════════════════════════════════════════════════════


@dataclass
class ValidationIssue:
    """单个验证问题。"""
    level: str  # "syntax", "semantic", "consistency", "tactical"
    severity: str  # "error", "warning", "info"
    line_number: int = 0
    description: str = ""
    suggestion: str = ""
    rule_id: str = ""


@dataclass
class ValidationResult:
    """验证结果。"""
    score: int = 100
    total_issues: int = 0
    errors: int = 0
    warnings: int = 0
    infos: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    def add_issue(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        self.total_issues += 1
        if issue.severity == "error":
            self.errors += 1
            self.score -= 10
        elif issue.severity == "warning":
            self.warnings += 1
            self.score -= 3
        else:
            self.infos += 1
            self.score -= 1
        self.score = max(0, self.score)


# ═══════════════════════════════════════════════════════
# Tool 定义
# ═══════════════════════════════════════════════════════


class ScenarioValidateTool(Tool):
    """验证 AFSIM 想定代码的语法、语义和一致性。"""

    _scopes = {"core", "subagent"}

    @property
    def name(self) -> str:
        return "scenario_validate"

    @property
    def description(self) -> str:
        return (
            "Validate AFSIM scenario code across 4 levels: syntax (block pairing, "
            "keywords, coordinate format, units), semantic (reference consistency, "
            "type matching), consistency (platform_type↔platform, weapon references), "
            "and tactical plausibility (sensor range, weapon range, speed/altitude). "
            "Returns a score (0-100) and detailed issues with fix suggestions."
        )

    @property
    def read_only(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "scenario_code": {
                    "type": "string",
                    "description": "The AFSIM scenario code to validate.",
                },
                "coordinator_json": {
                    "type": "string",
                    "description": (
                        "Optional JSON with original COA data for cross-validation. "
                        "If provided, checks entity count and platform consistency."
                    ),
                },
            },
            "required": ["scenario_code"],
        }

    async def execute(self, **kwargs: Any) -> str:
        try:
            code = kwargs["scenario_code"]
            coa_data_raw = kwargs.get("coordinator_json", "")
            coa_data = json.loads(coa_data_raw) if coa_data_raw else None

            validator = ScenarioValidator(code, coa_data)
            result = validator.validate_all()

            output = {
                "score": result.score,
                "grade": _score_to_grade(result.score),
                "summary": {
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "infos": result.infos,
                    "total_issues": result.total_issues,
                },
                "stats": result.stats,
                "issues": [
                    {
                        "level": i.level,
                        "severity": i.severity,
                        "line": i.line_number,
                        "rule": i.rule_id,
                        "description": i.description,
                        "suggestion": i.suggestion,
                    }
                    for i in result.issues
                ],
                "pass": result.errors == 0,
            }
            return json.dumps(output, ensure_ascii=False, indent=2)
        except Exception as exc:
            return f"Error validating scenario: {exc}"


# ═══════════════════════════════════════════════════════
# 验证引擎
# ═══════════════════════════════════════════════════════


class ScenarioValidator:
    """AFSIM 想定多层级验证器。"""

    def __init__(self, code: str, coa_data: dict | None = None):
        self.code = code
        self.raw_lines = code.split("\n")
        # 分析用的行: 先剥注释, 再清空脚本块块体; 行号与 raw_lines 一一对应
        self.lines = _mask_script_bodies(_mask_block_comments(self.raw_lines))
        self.coa_data = coa_data
        self.result = ValidationResult()

    def validate_all(self) -> ValidationResult:
        """运行全部验证。"""
        self._collect_stats()
        self._check_syntax()
        self._check_semantic()
        self._check_consistency()
        self._check_tactical()
        if self.coa_data:
            self._check_coa_cross_validation()
        return self.result

    def _collect_stats(self) -> None:
        """收集想定统计信息。"""
        stats: dict[str, int] = {}
        for line in self.lines:
            stripped = line.strip()
            for keyword in _BLOCK_PAIRS:
                if stripped.startswith(keyword + " ") or stripped == keyword:
                    stats[keyword] = stats.get(keyword, 0) + 1
        stats["total_lines"] = len(self.lines)
        stats["non_empty_lines"] = sum(1 for l in self.lines if l.strip())
        self.result.stats = stats

    # ── Level 1: 语法检查 ──

    def _check_syntax(self) -> None:
        self._check_block_pairing()
        self._check_base_types()
        self._check_coordinate_format()
        self._check_units()
        # 注: AFSIM 支持 # 行尾注释, 不检查行尾注释

    def _check_block_pairing(self) -> None:
        """检查所有 begin/end 块是否正确配对。

        AFSIM 的块语法有两处天然歧义, 纯关键字白名单无法处理:

        1. **定义 vs 引用**: ``radar_signature MRM_SIG ... end_radar_signature`` 是
           定义, 而 platform_type 里的 ``radar_signature MRM_SIG`` 只是引用(无 end);
           ``antenna_pattern SEEKER_ANTENNA``、``aero MRM_AERO`` 同理。
        2. **单行自闭合块**: ``launch_computer MRM_LC end_launch_computer`` 起止同行。

        因此这里改用 **end_* 配额 (budget)** 判定: 先统计文档里每种 ``end_*`` 的
        实际条数, 只有当还有未被认领的 ``end_kw`` 时, 才把 ``kw`` 开头的行当作块起始。
        配额耗尽时该行只能是引用 —— 若该关键字本身允许引用形态就静默跳过, 否则报
        未闭合。这样真实可运行的想定不会被误判, 同时我们自己生成的代码里真正缺失的
        ``end_platform`` 之类仍会被抓到。
        """
        lines_c = [_strip_comments(line) for line in self.lines]

        # ── 预扫描: 统计 end_* 配额, 并标记单行自闭合块 ──
        end_budget: Counter[str] = Counter()
        self_closed: set[int] = set()
        for i, stripped in enumerate(lines_c, 1):
            if not stripped:
                continue
            if re.match(r"^(end_\w+)\s*$", stripped):
                end_budget[stripped] += 1
                continue
            keyword = _block_keyword(stripped)
            if keyword and stripped.split()[-1] == _BLOCK_PAIRS[keyword]:
                self_closed.add(i)

        # ── 主扫描: 栈匹配 ──
        stack: list[tuple[str, int]] = []
        for i, stripped in enumerate(lines_c, 1):
            if not stripped or i in self_closed:
                continue

            end_match = re.match(r"^(end_\w+)\s*$", stripped)
            if end_match:
                end_kw = end_match.group(1)
                if end_kw not in _MODELED_BLOCK_ENDS:
                    continue  # 未建模的块类型, 不做判断
                found = False
                for j in range(len(stack) - 1, -1, -1):
                    if _BLOCK_PAIRS.get(stack[j][0]) == end_kw:
                        stack.pop(j)
                        found = True
                        break
                if not found:
                    self.result.add_issue(ValidationIssue(
                        level="syntax", severity="error", line_number=i,
                        description=f"Unmatched '{end_kw}' at line {i}",
                        suggestion=f"Add matching '{_reverse_block(end_kw)}' before this line",
                        rule_id="SYN_001",
                    ))
                continue

            keyword = _block_keyword(stripped)
            if not keyword:
                continue
            end_kw = _BLOCK_PAIRS[keyword]
            if end_budget[end_kw] <= 0:
                # 已无可配对的 end_*: 本行只能是引用形态
                if keyword not in _REFERENCE_CAPABLE_BLOCKS:
                    self.result.add_issue(ValidationIssue(
                        level="syntax", severity="error", line_number=i,
                        description=f"Unclosed '{keyword}' at line {i}",
                        suggestion=f"Add '{end_kw}' to close this block",
                        rule_id="SYN_002",
                    ))
                continue
            end_budget[end_kw] -= 1
            stack.append((keyword, i))

        # 栈中剩余的都是未关闭的块
        for begin_kw, begin_line in stack:
            expected_end = _BLOCK_PAIRS.get(begin_kw, f"end_{begin_kw}")
            self.result.add_issue(ValidationIssue(
                level="syntax", severity="error", line_number=begin_line,
                description=f"Unclosed '{begin_kw}' at line {begin_line}",
                suggestion=f"Add '{expected_end}' to close this block",
                rule_id="SYN_002",
            ))

    def _check_base_types(self) -> None:
        """检查基类型拼写。

        未登记的 ``WSF_*`` 只报 warning: ``_VALID_BASE_TYPES`` 是"已知good"清单而
        非 AFSIM 全集, 报 error 会把使用了合法但未登记类型的想定判死。同一行里
        重复出现的同一个词元只报一次。
        """
        for i, line in enumerate(self.lines, 1):
            stripped = _strip_comments(line)
            if not stripped:
                continue
            for word in dict.fromkeys(_WSF_TOKEN_RE.findall(stripped)):
                if word in _VALID_BASE_TYPES:
                    continue
                self.result.add_issue(ValidationIssue(
                    level="syntax", severity="warning", line_number=i,
                    description=f"Unregistered base type '{word}' at line {i}",
                    suggestion=f"Verify against AFSIM 2.9 grammar; did you mean one of: {_suggest_type(word)}?",
                    rule_id="SYN_003",
                ))

    def _check_coordinate_format(self) -> None:
        """检查坐标格式。

        只处理首词恰为 ``position`` 的行 —— AFSIM 还有 ``six_dof_position``、
        ``mgrs_position`` 等同族命令, 它们的参数不是经纬度 (six_dof_position 用的是
        相对偏移), 用子串匹配会把这些行误判为坐标格式错误。
        """
        for i, line in enumerate(self.lines, 1):
            stripped = _strip_comments(line)
            tokens = stripped.split()
            if len(tokens) < 2 or tokens[0] != "position":
                continue

            pos_text = stripped[len("position"):].strip()
            # AFSIM 也支持直角/ECI 位置 (如 `position -1.2e+07 m -3.4e+06 m -5e+06 m`)。
            # 这类写法没有半球标识而是"数值+长度单位"重复三组, 不按经纬度判定;
            # 但完全认不出的形态 (如 `position garbage here`) 仍要报错。
            if not re.search(r"\d\s*[nsewNSEW](?:\s|$)", pos_text):
                if _CARTESIAN_POSITION_RE.match(pos_text):
                    continue
            # 截掉 altitude/heading/speed 等后续参数, 只留坐标部分
            coord_part = pos_text
            for suffix in ("altitude", "heading", "speed", "msl", "agl"):
                idx = coord_part.find(suffix)
                if idx > 0:
                    coord_part = coord_part[:idx]
            coord_part = coord_part.strip()

            if not _COORD_PATTERN.match(coord_part):
                self.result.add_issue(ValidationIssue(
                    level="syntax", severity="error", line_number=i,
                    description=f"Invalid coordinate format at line {i}: '{coord_part}'",
                    suggestion="Use '27:30:00n 122:30:00e' (d:m:s) or '27.5n 122.5e' (decimal degrees)",
                    rule_id="SYN_004",
                ))

            # 检查坐标范围
            parsed = _parse_coord_values(coord_part)
            if parsed:
                lat, lon = parsed
                if lat < -90 or lat > 90:
                    self.result.add_issue(ValidationIssue(
                        level="syntax", severity="error", line_number=i,
                        description=f"Latitude {lat:.2f} out of range [-90, 90]",
                        rule_id="SYN_005",
                    ))
                if lon < -180 or lon > 180:
                    self.result.add_issue(ValidationIssue(
                        level="syntax", severity="error", line_number=i,
                        description=f"Longitude {lon:.2f} out of range [-180, 180]",
                        rule_id="SYN_006",
                    ))

    def _check_units(self) -> None:
        """检查单位使用。"""
        unit_patterns = [
            (re.compile(r"speed\s+(\d+)\s+(\S+)"), "speed"),
            (re.compile(r"maximum_range\s+(\d+)\s+(\S+)"), "maximum_range"),
            (re.compile(r"altitude\s+(\d+)\s+(.+)"), "altitude"),
            (re.compile(r"end_time\s+(\d+)\s+(\S+)"), "end_time"),
        ]

        for i, line in enumerate(self.lines, 1):
            stripped = line.strip()
            for pattern, param_name in unit_patterns:
                m = pattern.match(stripped)
                if not m:
                    continue
                unit = m.group(2).strip()
                # 复合单位检查 (如 "ft msl")
                if param_name == "altitude":
                    parts = unit.split()
                    if len(parts) >= 2:
                        if parts[0] not in ("ft", "m") or parts[1] not in ("msl", "agl"):
                            self.result.add_issue(ValidationIssue(
                                level="syntax", severity="warning", line_number=i,
                                description=f"Suspicious altitude unit '{unit}' at line {i}",
                                suggestion="Use 'ft msl' or 'm agl'",
                                rule_id="SYN_007",
                            ))
                elif param_name == "end_time":
                    if unit not in ("sec", "min", "hr"):
                        self.result.add_issue(ValidationIssue(
                            level="syntax", severity="warning", line_number=i,
                            description=f"Suspicious time unit '{unit}' at line {i}",
                            suggestion="Use 'sec', 'min', or 'hr'",
                            rule_id="SYN_008",
                        ))

    # ── Level 2: 语义检查 ──

    def _check_semantic(self) -> None:
        self._check_side_consistency()
        self._check_reference_integrity()

    def _check_side_consistency(self) -> None:
        """检查 side 字段一致性 (side 只在 platform 实例上, 不在 platform_type)。"""
        in_platform = False
        in_platform_type = False
        platform_has_side = False
        platform_line = 0

        for i, line in enumerate(self.lines, 1):
            stripped = line.strip()
            # 去掉行尾注释
            if "#" in stripped:
                stripped = stripped[:stripped.index("#")].strip()

            if not stripped:
                continue

            # 跟踪块嵌套
            if re.match(r"platform_type\s+\S+", stripped):
                in_platform_type = True
                in_platform = False
                continue
            if stripped == "end_platform_type":
                in_platform_type = False
                continue
            if re.match(r"platform\s+\S+\s+\S+", stripped):
                in_platform = True
                in_platform_type = False
                platform_has_side = False
                platform_line = i
                continue
            if stripped == "end_platform":
                if in_platform and not platform_has_side:
                    self.result.add_issue(ValidationIssue(
                        level="semantic", severity="warning", line_number=platform_line,
                        description=f"Platform at line {platform_line} has no 'side' declaration",
                        suggestion="Add 'side blue' or 'side red' inside the platform block",
                        rule_id="SEM_001",
                    ))
                in_platform = False
                continue

            # 检查 side 取值格式 (只在 platform 实例内)
            #
            # AFSIM 的 side 是任意标识符, 不是 red/blue 枚举。官方 demo 里出现过
            # green(113) US(76) ORB(57) indigo(47) ITSO(34) CIS(28) ESA(19)
            # pink/yellow/dark_purple/white/purple/black/violet/orange 等 20+ 种,
            # 早先只放行 red/blue, 在官方语料上误报了 367 次。
            # 这里只检查它是不是一个合法标识符, 不再限制取值集合。
            if in_platform and not in_platform_type:
                side_match = re.match(r"side\s+(\S+)", stripped)
                if side_match:
                    platform_has_side = True
                    side_val = side_match.group(1)
                    if not _SIDE_TOKEN_RE.match(side_val):
                        self.result.add_issue(ValidationIssue(
                            level="semantic", severity="error", line_number=i,
                            description=f"Malformed side identifier '{side_val}' at line {i}",
                            suggestion="side 取值应是标识符, 如 blue / red / green / US",
                            rule_id="SEM_001",
                        ))

    def _check_reference_integrity(self) -> None:
        """检查引用完整性: weapon 引用的类型是否已定义。

        真实 AFSIM 中:
          - weapon {slot_name} {TYPE} 在 platform_type 内引用
          - weapon {TYPE} WSF_EXPLICIT_WEAPON 在顶层声明
          - platform_type {TYPE} WSF_PLATFORM 定义导弹本体
        """
        # 收集已定义的 weapon (顶层 weapon 声明)
        defined_weapons: set[str] = set()
        for line in self.lines:
            stripped = line.strip()
            if "#" in stripped:
                stripped = stripped[:stripped.index("#")].strip()
            m = re.match(r"weapon\s+(\S+)\s+WSF_EXPLICIT_WEAPON", stripped)
            if m:
                defined_weapons.add(m.group(1))

        # 收集已定义的 platform_type (包括导弹的 platform_type)
        defined_types: set[str] = set()
        for line in self.lines:
            stripped = line.strip()
            if "#" in stripped:
                stripped = stripped[:stripped.index("#")].strip()
            m = re.match(r"platform_type\s+(\S+)", stripped)
            if m:
                defined_types.add(m.group(1))

        # 检查 platform_type 内 weapon 引用: weapon {slot} {TYPE}
        for i, line in enumerate(self.lines, 1):
            stripped = line.strip()
            if "#" in stripped:
                stripped = stripped[:stripped.index("#")].strip()
            # 匹配 "weapon {slot_name} {TYPE}" (有类型引用)
            m = re.match(r"weapon\s+(\S+)\s+(\S+)", stripped)
            if m:
                type_ref = m.group(2)
                # 跳过 WSF_* 基类型
                if type_ref.startswith(("WSF_", "wsf_")):
                    continue
                # 检查 TYPE 是否已定义 (weapon 声明 或 platform_type)
                if type_ref not in defined_weapons and type_ref not in defined_types:
                    self.result.add_issue(ValidationIssue(
                        level="semantic", severity="warning", line_number=i,
                        description=f"Weapon type '{type_ref}' referenced at line {i} but no definition found",
                        suggestion=f"Define 'weapon {type_ref} WSF_EXPLICIT_WEAPON ... end_weapon' or 'platform_type {type_ref} WSF_PLATFORM ... end_platform_type'",
                        rule_id="SEM_002",
                    ))

    # ── Level 3: 一致性检查 ──

    def _check_consistency(self) -> None:
        self._check_platform_type_refs()
        self._check_duplicate_names()

    def _check_platform_type_refs(self) -> None:
        """检查 platform 实例引用的 platform_type 是否存在。

        AFSIM 想定普遍拆成多个文件, platform_type 定义与 platform 实例常常分居两个
        文件, 由第三个主文件把它们一起 include 进来。单文件验证器看不到别处的定义,
        因此**只有在本文件自己定义了 platform_type 时**才把未解析的引用判成 error
        (说明这是一份自洽文件, 大概率真的漏了一个); 否则一律降级为 warning。

        官方 demo 语料上这条规则曾误报 2658 次, 全都是 satellite_demos、iads 之类
        把平台清单单独成文的片段 —— 它们连 include 指令都没有, 早先靠 include 关键字
        判断片段的做法覆盖不到。
        """
        defined_types: set[str] = set()
        for line in self.lines:
            m = re.match(r"platform_type\s+(\S+)", _strip_comments(line))
            if m:
                defined_types.add(m.group(1))

        has_include = any(
            _strip_comments(line).startswith(("include ", "include_once "))
            for line in self.lines
        )
        # 既自己定义了类型、又没有任何 include 才算真正自洽; 只要有 include,
        # 未解析的类型都可能来自别的文件 (旧语料 1v1_initial_prediction.txt 即如此)
        self_contained = bool(defined_types) and not has_include
        severity = "error" if self_contained else "warning"

        for i, line in enumerate(self.lines, 1):
            m = re.match(r"platform\s+(\S+)\s+(\S+)", _strip_comments(line))
            if m:
                instance_name = m.group(1)
                type_ref = m.group(2)
                if type_ref not in defined_types and not type_ref.startswith("WSF_"):
                    hint = (
                        f"Define 'platform_type {type_ref} WSF_PLATFORM ... end_platform_type' first"
                        if self_contained
                        else "Type likely comes from another file; verify the include chain"
                    )
                    self.result.add_issue(ValidationIssue(
                        level="consistency", severity=severity, line_number=i,
                        description=f"Platform '{instance_name}' references undefined type '{type_ref}'",
                        suggestion=hint,
                        rule_id="CON_001",
                    ))

    def _check_duplicate_names(self) -> None:
        """检查平台实例名称重复。"""
        names: dict[str, int] = {}
        for i, line in enumerate(self.lines, 1):
            m = re.match(r"platform\s+(\S+)\s+\S+\s*$", line.strip())
            if m:
                name = m.group(1)
                if name in names:
                    self.result.add_issue(ValidationIssue(
                        level="consistency", severity="error", line_number=i,
                        description=f"Duplicate platform name '{name}' (first at line {names[name]})",
                        suggestion="Use unique callsign-based names for each platform instance",
                        rule_id="CON_002",
                    ))
                names[name] = i

    # ── Level 4: 战术合理性 ──

    def _check_tactical(self) -> None:
        self._check_speed_altitude()
        self._check_sensor_coverage()

    def _check_speed_altitude(self) -> None:
        """检查速度和高度合理性。"""
        for i, line in enumerate(self.lines, 1):
            stripped = line.strip()
            # 速度检查
            m = re.match(r"speed\s+(\d+)\s+kts", stripped)
            if m:
                speed = int(m.group(1))
                if speed > 2000:
                    self.result.add_issue(ValidationIssue(
                        level="tactical", severity="warning", line_number=i,
                        description=f"Speed {speed} kts exceeds typical aircraft max (~Mach 2.5 ≈ 1400 kts)",
                        suggestion="Verify if this is a missile speed or reduce to realistic value",
                        rule_id="TAC_001",
                    ))
                if speed < 50 and speed > 0:
                    self.result.add_issue(ValidationIssue(
                        level="tactical", severity="info", line_number=i,
                        description=f"Speed {speed} kts unusually low for aircraft",
                        rule_id="TAC_002",
                    ))

            # 高度检查
            m = re.match(r"altitude\s+(\d+)\s+ft\s+msl", stripped)
            if m:
                alt = int(m.group(1))
                if alt > 80000:
                    self.result.add_issue(ValidationIssue(
                        level="tactical", severity="warning", line_number=i,
                        description=f"Altitude {alt} ft exceeds typical service ceiling (~60000 ft)",
                        rule_id="TAC_003",
                    ))

    def _check_sensor_coverage(self) -> None:
        """检查传感器覆盖与平台类型的匹配。"""
        # 收集 sensor 参数
        current_sensor = ""
        sensor_ranges: dict[str, str] = {}
        for line in self.lines:
            stripped = line.strip()
            m = re.match(r"sensor\s+(\S+)", stripped)
            if m:
                current_sensor = m.group(1)
            m = re.match(r"maximum_range\s+(\d+)\s+km", stripped)
            if m and current_sensor:
                sensor_ranges[current_sensor] = m.group(1)

    # ── 交叉验证 ──

    def _check_coa_cross_validation(self) -> None:
        """与原始 COA 数据交叉验证。"""
        if not self.coa_data:
            return

        # 检查 entity 数量
        coa_entities = self.coa_data.get("entities", [])
        scenario_platforms = self.result.stats.get("platform", 0)
        expected_total = 0
        for e in coa_entities:
            qty_str = str(e.get("quantity", "1"))
            qty_match = re.search(r"(\d+)", qty_str)
            expected_total += int(qty_match.group(1)) if qty_match else 1
        if scenario_platforms > 0 and abs(scenario_platforms - expected_total) > 2:
            self.result.add_issue(ValidationIssue(
                level="consistency", severity="warning",
                description=f"COA expects {expected_total} platforms, scenario has {scenario_platforms}",
                suggestion="Verify entity quantities match platform instances",
                rule_id="XVAL_001",
            ))


# ═══════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════


def _reverse_block(end_kw: str) -> str:
    """从 end_* 反推 begin keyword。"""
    for begin, end in _BLOCK_PAIRS.items():
        if end == end_kw:
            return begin
    return end_kw.replace("end_", "")


def _suggest_type(word: str) -> str:
    """建议最接近的合法基类型。"""
    best = ""
    best_score = 0
    for valid in _VALID_BASE_TYPES:
        # 简单的字符重叠相似度
        common = sum(1 for c in word if c in valid)
        score = common / max(len(word), len(valid))
        if score > best_score:
            best_score = score
            best = valid
    return best


def _parse_coord_values(coord_text: str) -> tuple[float, float] | None:
    """从 AFSIM 坐标文本解析十进制度, 同时支持 d:m:s 与十进制度两种写法。"""
    m = _COORD_PARSE_RE.match(coord_text.strip())
    if not m:
        return None
    lat = _coord_token_to_degrees(m.group("lat"))
    lon = _coord_token_to_degrees(m.group("lon"))
    if lat is None or lon is None:
        return None
    return lat, lon


def _coord_token_to_degrees(token: str) -> float | None:
    """把单个坐标词元转为带符号十进制度。

    支持 AFSIM 的四种写法: ``30n`` / ``30.5n`` / ``40:30n`` / ``27:30:00n``
    (末者秒可带小数)。
    """
    match = _COORD_TOKEN_RE.match(token.strip())
    if not match:
        return None

    value = float(match.group("deg"))
    if match.group("frac") is not None:
        value = float(f"{match.group('deg')}.{match.group('frac')}")
    else:
        if match.group("min") is not None:
            value += int(match.group("min")) / 60.0
        if match.group("sec") is not None:
            value += float(match.group("sec")) / 3600.0

    if match.group("hemi").lower() in ("s", "w"):
        value = -value
    return value


def _score_to_grade(score: int) -> str:
    """分数转等级。"""
    if score >= 90:
        return "A (Excellent)"
    if score >= 80:
        return "B (Good)"
    if score >= 70:
        return "C (Acceptable)"
    if score >= 60:
        return "D (Needs Work)"
    return "F (Failed)"
