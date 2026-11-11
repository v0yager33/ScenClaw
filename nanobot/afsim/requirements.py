"""需求原文里写明数值的约束, 由代码逐条核验。

通用合理性检查只管"想定跑得起来是否有意义", 管不了需求里的具体数字: "双方初始相距
200-280 km""预备队在 540-900 s 发射""蓝一编队 900-1200 s 越过 122.8°E" 这类条款,
以前只能交给核对轮的模型判读。实测推理模型逐架推算航点时刻和斜距既慢又容易错,
常常思考到预算耗尽也给不出结论。

这里把这类条款收成几种固定类型: 模型只负责从需求原文里抽出条款 (不做计算), 核验全部
由代码按 IR 匀速推算完成。不满足的条款作为问题回灌修订轮, 消息里直接带上修法所需的数字。
推算口径与 ``geometry`` 一致: 忽略转弯和加减速, 航路飞完之后的位置不计。
例外是交战包络: 开火时刻晚于某一方航路终点时斜距无从保证, 按不满足报。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any

from nanobot.afsim.geometry import (
    ground_distance_m,
    position_at,
    slant_range_m,
    waypoint_times_s,
)
from nanobot.afsim.schemas import GeoPoint, PlatformIR, ScenarioIR, ScheduledEngagementIR

# 抽取提示里的条款说明, 和下面的解析规则放在一起维护
REQUIREMENT_CHECKS_SPEC = """条款类型与字段 (单位: km、s、m、m/s、度; 平台选择器写平台 id, 可用 * 通配如 "blue_g1_*", 或写 "side:blue" 表示一方全部平台):
- side_separation {min_km, max_km}: 双方初始最近一对平台的地面距离
- formation_spacing {groups, min_km, max_km}: 编队内每架与同队最近一架的初始地面距离; groups 每项是一个编队 (选择器或选择器列表), 不给时按阵营分组
- fire_window {shooters, start_s, end_s}: 这些射手的每一次开火都必须落在该时段内
- fire_phases {windows, min_count, shooters}: windows 写 [[起, 止], ...], 要求至少 min_count 个时段内有开火; shooters 可省略 (表示全部射手)
- engagement_range {shooters, min_km, max_km}: 每次开火时射手与目标的斜距; shooters 可省略; 交战包络、射程、毁伤半径是距离上限, 只写 max_km
- reach_area {platforms, lat_min, lat_max, lon_min, lon_max, start_s, end_s, alive}: 每个平台首次进入该经纬度区域的时刻须在时段内; 区域边界和时段都可只给一侧; alive 默认 true, 表示进入时必须仍存活
- altitude_range {platforms, min_m, max_m}: 初始位置和全部航点的高度
- speed_range {platforms, min_mps, max_mps}: 全部航点的速度
- route_points {platforms, min_count}: 每个平台的航点 (route) 至少 min_count 个
有上下限的字段至少给一个; 原文写"约 X"时边界就取 X, 不要自行放宽成区间或补上原文没有的另一侧; 每条可加 quote, 摘一句对应的需求原文 (40 字以内)。"""

_QUOTE_LIMIT = 40
_RANGE_STEP_S = 5.0
# 进区时刻的建议目标离时段边界的距离 (秒); 时段较短时取四分之一宽度
_ENTRY_MARGIN_S = 60.0
_MAX_SPANS = 3
# 实测: 开火时刻贴着包络时段的端点, 下一轮改航路后斜距变成 101 km, 修订轮数用完没修回来
_FIRE_MARGIN_S = 15.0

# 字段类型: num 数值, count 正整数, bool, selectors 选择器列表, groups 编队列表, windows 时段列表
_FIELDS: dict[str, dict[str, str]] = {
    "side_separation": {"min_km": "num", "max_km": "num"},
    "formation_spacing": {"groups": "groups", "min_km": "num", "max_km": "num"},
    "fire_window": {"shooters": "selectors", "start_s": "num", "end_s": "num"},
    "fire_phases": {"windows": "windows", "min_count": "count", "shooters": "selectors"},
    "engagement_range": {"shooters": "selectors", "min_km": "num", "max_km": "num"},
    "reach_area": {
        "platforms": "selectors",
        "lat_min": "num", "lat_max": "num", "lon_min": "num", "lon_max": "num",
        "start_s": "num", "end_s": "num", "alive": "bool",
    },
    "altitude_range": {"platforms": "selectors", "min_m": "num", "max_m": "num"},
    "speed_range": {"platforms": "selectors", "min_mps": "num", "max_mps": "num"},
    "route_points": {"platforms": "selectors", "min_count": "count"},
}
_REQUIRED: dict[str, tuple[str, ...]] = {
    "fire_window": ("shooters",),
    "fire_phases": ("windows", "min_count"),
    "reach_area": ("platforms",),
    "altitude_range": ("platforms",),
    "speed_range": ("platforms",),
    "route_points": ("platforms", "min_count"),
}
_ANY_OF: dict[str, tuple[str, ...]] = {
    "side_separation": ("min_km", "max_km"),
    "formation_spacing": ("min_km", "max_km"),
    "fire_window": ("start_s", "end_s"),
    "engagement_range": ("min_km", "max_km"),
    "reach_area": ("lat_min", "lat_max", "lon_min", "lon_max"),
    "altitude_range": ("min_m", "max_m"),
    "speed_range": ("min_mps", "max_mps"),
}
_BOUND_PAIRS = (
    ("min_km", "max_km"), ("start_s", "end_s"), ("lat_min", "lat_max"),
    ("lon_min", "lon_max"), ("min_m", "max_m"), ("min_mps", "max_mps"),
)


@dataclass(frozen=True)
class RequirementCheck:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    quote: str = ""

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind, **self.params}
        if self.quote:
            data["quote"] = self.quote
        return data


@dataclass(frozen=True)
class _Plan:
    """核验时共享的推演结果。

    ``destroyed`` 是按射击计划推演出的毁伤; ``fire_windows`` 和 ``survive_until`` 来自别的
    条款 (射手限定的开火时段、平台须存活到的进区时刻), 用来在报文里点明两条需求互相牵制。
    实测: 预备队的射击要挪进 540-900 s, 打的却是要在约 907 s 进走廊时仍存活的蓝一; 两条
    报文各说各的, 模型在"挪进时段"和"推迟射击"之间来回改, 两轮都没想到换目标。
    """

    destroyed: dict[str, tuple[float, str]]
    fire_windows: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    survive_until: dict[str, float] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════
# 解析
# ═══════════════════════════════════════════════════════


def parse_requirement_checks(
    raw: Any, scenario: ScenarioIR | None = None
) -> tuple[list[RequirementCheck], list[str]]:
    """把模型抽出的条款整理成 ``RequirementCheck``; 不合法的整条丢弃并记一条说明。

    给了 ``scenario`` 时还会核对选择器: 一个平台都匹配不到的选择器删掉, 删完后
    没有对象可查的条款整条丢弃。
    """
    items = raw.get("checks") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return [], ["条款抽取结果里没有 checks 列表"]
    checks: list[RequirementCheck] = []
    notes: list[str] = []
    for index, item in enumerate(items, 1):
        check, problem = _parse_one(item)
        if check is not None and scenario is not None:
            check, problem = _bind_selectors(check, scenario)
        if check is None:
            notes.append(f"丢弃第 {index} 条条款: {problem}")
        else:
            if problem:
                notes.append(f"第 {index} 条条款: {problem}")
            checks.append(check)
    return checks, notes


def _parse_one(item: Any) -> tuple[RequirementCheck | None, str]:
    if not isinstance(item, dict):
        return None, "不是对象"
    kind = item.get("kind")
    fields = _FIELDS.get(kind) if isinstance(kind, str) else None
    if fields is None:
        return None, f"未知类型 {kind!r}"
    params: dict[str, Any] = {}
    for name, field_type in fields.items():
        value = item.get(name)
        if value is None:
            continue  # 模型常把用不上的字段写成 null
        coerced = _coerce(value, field_type)
        if coerced is None:
            return None, f"{kind}.{name} 取值不合法: {value!r}"
        params[name] = coerced
    missing = [name for name in _REQUIRED.get(kind, ()) if name not in params]
    if missing:
        return None, f"{kind} 缺少字段 {', '.join(missing)}"
    bounds = _ANY_OF.get(kind, ())
    if bounds and not any(name in params for name in bounds):
        return None, f"{kind} 至少要给 {'/'.join(bounds)} 中的一个"
    for low, high in _BOUND_PAIRS:
        if low in params and high in params and params[low] > params[high]:
            return None, f"{kind} 的 {low} 大于 {high}"
    if kind == "fire_phases" and params["min_count"] > len(params["windows"]):
        return None, "fire_phases 的 min_count 超过时段个数"
    quote = item.get("quote")
    quote = quote.strip()[:_QUOTE_LIMIT] if isinstance(quote, str) else ""
    return RequirementCheck(kind=kind, params=params, quote=quote), ""


def _coerce(value: Any, field_type: str) -> Any:
    if field_type == "num":
        return _number(value)
    if field_type == "count":
        number = _number(value)
        if number is None or number < 1 or number != int(number):
            return None
        return int(number)
    if field_type == "bool":
        return value if isinstance(value, bool) else None
    if field_type == "selectors":
        return _selectors(value)
    if field_type == "groups":
        if not isinstance(value, list) or not value:
            return None
        groups = [_selectors(group) for group in value]
        return None if any(group is None for group in groups) else groups
    if field_type == "windows":
        if not isinstance(value, list) or not value:
            return None
        windows = []
        for window in value:
            if not isinstance(window, (list, tuple)) or len(window) != 2:
                return None
            start, end = _number(window[0]), _number(window[1])
            if start is None or end is None or start >= end:
                return None
            windows.append([start, end])
        return sorted(windows)
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _selectors(value: Any) -> list[str] | None:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value:
        return None
    if not all(isinstance(item, str) and item.strip() for item in value):
        return None
    return [item.strip() for item in value]


def _bind_selectors(
    check: RequirementCheck, scenario: ScenarioIR
) -> tuple[RequirementCheck | None, str]:
    params = dict(check.params)
    dropped: list[str] = []

    def keep(selectors: list[str]) -> list[str]:
        kept = [s for s in selectors if _match(s, scenario.platforms)]
        dropped.extend(s for s in selectors if s not in kept)
        return kept

    for name, field_type in _FIELDS[check.kind].items():
        if name not in params:
            continue
        if field_type == "selectors":
            params[name] = keep(params[name])
            if not params[name]:
                return None, f"{check.kind}.{name} 匹配不到任何平台: {', '.join(dropped)}"
        elif field_type == "groups":
            params[name] = [group for group in (keep(g) for g in params[name]) if group]
            if not params[name]:
                return None, f"{check.kind}.{name} 匹配不到任何平台: {', '.join(dropped)}"
    note = f"选择器 {', '.join(dropped)} 匹配不到平台, 已忽略" if dropped else ""
    return RequirementCheck(kind=check.kind, params=params, quote=check.quote), note


def _match(selector: str, platforms: list[PlatformIR]) -> list[PlatformIR]:
    if selector.startswith("side:"):
        side = selector[len("side:"):].strip().lower()
        return [p for p in platforms if p.side.value == side]
    if any(char in selector for char in "*?["):
        return [p for p in platforms if fnmatchcase(p.platform_id, selector)]
    return [p for p in platforms if p.platform_id == selector]


def _resolve(
    selectors: list[str] | None, scenario: ScenarioIR
) -> tuple[list[PlatformIR], list[str]]:
    """选择器 → 平台 (按想定顺序去重); 同时返回匹配不到的选择器。"""
    if selectors is None:
        return list(scenario.platforms), []
    chosen: dict[str, PlatformIR] = {}
    missing: list[str] = []
    for selector in selectors:
        matched = _match(selector, scenario.platforms)
        if not matched:
            missing.append(selector)
        for platform in matched:
            chosen.setdefault(platform.platform_id, platform)
    ordered = [p for p in scenario.platforms if p.platform_id in chosen]
    return ordered, missing


# ═══════════════════════════════════════════════════════
# 核验
# ═══════════════════════════════════════════════════════


def evaluate_requirement_checks(
    scenario: ScenarioIR,
    checks: list[RequirementCheck],
    destroyed: dict[str, tuple[float, str]] | None = None,
) -> list[str]:
    """逐条核验, 返回不满足的条款说明; 全部满足时返回空列表。

    ``destroyed`` 是按射击计划推演出的毁伤 ``{platform_id: (时刻, 交战 id)}``:
    被毁平台之后的射击和航段都不会发生。
    """
    plan = _plan(scenario, checks, destroyed or {})
    issues: list[str] = []
    for check in checks:
        prefix = _prefix(check)
        issues.extend(prefix + message for message in _evaluate(scenario, check, plan))
    return issues


def _plan(
    scenario: ScenarioIR, checks: list[RequirementCheck], destroyed: dict[str, tuple[float, str]]
) -> _Plan:
    fire_windows: dict[str, tuple[float | None, float | None]] = {}
    survive_until: dict[str, float] = {}
    for check in checks:
        params = check.params
        if check.kind == "fire_window":
            for platform in _resolve(params.get("shooters"), scenario)[0]:
                window = (params.get("start_s"), params.get("end_s"))
                fire_windows.setdefault(platform.platform_id, window)
        elif check.kind == "reach_area" and params.get("alive", True):
            for platform in _resolve(params.get("platforms"), scenario)[0]:
                entry = _first_entry(platform, params, scenario.mission.duration_s)
                if entry is not None and entry[0] is not None:
                    pid = platform.platform_id
                    survive_until[pid] = max(survive_until.get(pid, 0.0), entry[0])
    return _Plan(destroyed, fire_windows, survive_until)


def describe_requirement_check(check: RequirementCheck) -> str:
    params = json.dumps(check.params, ensure_ascii=False)
    label = f"「{check.quote}」" if check.quote else check.kind
    return f"{label} → {check.kind} {params}"


def _prefix(check: RequirementCheck) -> str:
    if check.quote:
        return f"需求「{check.quote}」未满足: "
    return f"需求条款 {check.kind} 未满足: "


def _evaluate(scenario: ScenarioIR, check: RequirementCheck, plan: _Plan) -> list[str]:
    params = check.params
    messages: list[str] = []
    selector_field = next(
        (name for name in ("platforms", "shooters") if name in _FIELDS[check.kind]), None
    )
    platforms: list[PlatformIR] = []
    if selector_field is not None:
        platforms, missing = _resolve(params.get(selector_field), scenario)
        if missing:
            messages.append(f"条款涉及的 {', '.join(missing)} 在当前想定里找不到对应平台")
        if not platforms:
            return messages
    evaluator = _EVALUATORS[check.kind]
    messages.extend(evaluator(scenario, params, platforms, plan))
    return messages


def _side_separation(scenario, params, _platforms, _plan) -> list[str]:
    blues = [p for p in scenario.platforms if p.side.value == "blue"]
    reds = [p for p in scenario.platforms if p.side.value == "red"]
    if not blues or not reds:
        return []
    distance, blue, red = min(
        (ground_distance_m(b.position, r.position), b.platform_id, r.platform_id)
        for b in blues for r in reds
    )
    km = distance / 1000
    low, high = params.get("min_km"), params.get("max_km")
    if _within(km, low, high):
        return []
    head = (
        f"双方初始最近的一对 {blue} 与 {red} 地面相距约 {km:.0f} km, "
        f"要求 {_span(low, high, 'km')}"
    )
    if low is not None and km < low:
        move = f"约 {low - km:.0f}-{high - km:.0f} km" if high is not None else f"至少 {low - km:.0f} km"
        return [f"{head}; 需把双方再拉开{move}, 航路随初始位置一起平移"]
    move = f"约 {km - high:.0f}-{km - low:.0f} km" if low is not None else f"至少 {km - high:.0f} km"
    return [f"{head}; 需把双方相向移近{move}, 航路随初始位置一起平移"]


def _formation_spacing(scenario, params, _platforms, _plan) -> list[str]:
    low, high = params.get("min_km"), params.get("max_km")
    groups: list[list[PlatformIR]] = []
    missing: list[str] = []
    if "groups" in params:
        for selectors in params["groups"]:
            members, absent = _resolve(selectors, scenario)
            missing.extend(absent)
            groups.append(members)
    else:
        for side in sorted({p.side.value for p in scenario.platforms}):
            groups.append([p for p in scenario.platforms if p.side.value == side])
    messages = [f"条款涉及的 {', '.join(missing)} 在当前想定里找不到对应平台"] if missing else []
    for members in groups:
        for platform in members:
            others = [p for p in members if p.platform_id != platform.platform_id]
            if not others:
                continue
            distance, nearest = min(
                (ground_distance_m(platform.position, p.position), p.platform_id) for p in others
            )
            km = distance / 1000
            if not _within(km, low, high):
                messages.append(
                    f"{platform.platform_id} 与同队最近的 {nearest} 初始地面相距 {km:.1f} km, "
                    f"不在要求的 {_span(low, high, 'km')} 内; 请调整它的初始位置, 航路随之平移"
                )
    return messages


def _fire_window(scenario, params, platforms, plan) -> list[str]:
    start, end = params.get("start_s"), params.get("end_s")
    shooters = {p.platform_id for p in platforms}
    shots = [e for e in scenario.engagements if e.shooter_platform in shooters]
    window = _span(start, end, "s")
    if not shots:
        return [f"{', '.join(sorted(shooters))} 没有任何射击计划, 需求要求它们在 {window} 开火"]
    return [
        f"交战 {e.engagement_id} ({e.shooter_platform}) 在 {e.fire_time_s:g}s 开火, "
        f"不在要求的 {window} 内; " + _into_window_advice(e, start, end, plan)
        for e in sorted(shots, key=lambda e: e.fire_time_s)
        if not _within(e.fire_time_s, start, end)
    ]


def _into_window_advice(
    engagement: ScheduledEngagementIR, start: float | None, end: float | None, plan: _Plan
) -> str:
    """把射击调进时段的建议; 调进去会提前击毁需求要求存活的目标时, 改为建议换目标。"""
    target = engagement.target_platform
    until = plan.survive_until.get(target)
    if until is None or (start is not None and until <= start):
        return "请把开火时刻调进该时段"
    if end is not None and until >= end:
        return (
            f"但目标 {target} 按需求要在约 {until:.0f}s 进区时仍存活, 时段内开火会提前击毁它;"
            f" 请改打别的目标, 再把开火时刻调进该时段"
        )
    later = f"{until:.0f}s 之后" if end is None else f"{until:.0f}-{end:g}s"
    return (
        f"目标 {target} 按需求要在约 {until:.0f}s 进区时仍存活, 请把开火时刻调进 {later},"
        f" 或改打别的目标"
    )


def _fire_phases(scenario, params, platforms, plan) -> list[str]:
    destroyed = plan.destroyed
    windows: list[list[float]] = params["windows"]
    shooters = {p.platform_id for p in platforms}
    times = sorted(
        e.fire_time_s for e in scenario.engagements
        if e.shooter_platform in shooters and _happens(e, destroyed)
    )
    hit, empty = [], []
    for index, (start, end) in enumerate(windows):
        last = index == len(windows) - 1
        inside = any(start <= t < end or (last and t == end) for t in times)
        (hit if inside else empty).append(f"{start:g}-{end:g}s")
    if len(hit) >= params["min_count"]:
        return []
    actual = f"{times[0]:g}-{times[-1]:g}s" if times else "无"
    return [
        f"要求至少 {params['min_count']} 个时段有开火, 当前只有 {len(hit)} 个"
        f" (有开火: {'、'.join(hit) or '无'}; 没有开火: {'、'.join(empty)});"
        f" 当前实际开火时刻分布在 {actual}, 请把部分射击挪进空的时段 (航路要能支撑)"
    ]


def _engagement_range(scenario, params, platforms, plan) -> list[str]:
    destroyed = plan.destroyed
    low, high = params.get("min_km"), params.get("max_km")
    shooters = {p.platform_id for p in platforms}
    by_id = {p.platform_id: p for p in scenario.platforms}
    duration = scenario.mission.duration_s
    messages: list[str] = []
    for engagement in sorted(scenario.engagements, key=lambda e: e.fire_time_s):
        shooter = by_id.get(engagement.shooter_platform)
        target = by_id.get(engagement.target_platform)
        if shooter is None or target is None or shooter.platform_id not in shooters:
            continue
        moment = engagement.fire_time_s
        km = _slant_km(shooter, target, moment)
        if km is not None and _within(km, low, high):
            continue
        # 实测: 红方航路 624 s 就飞完, 1300 s 的射击推算不出斜距被跳过, 外部按终点
        # 位置量出 136 km。航路之后怎么飞不由 IR 决定, 这种射击的包络同样没有保证
        ended = [
            (platform.platform_id, end)
            for platform in (shooter, target)
            if (end := _route_end_s(platform)) is not None and end < moment
        ]
        if km is None and not ended:
            continue  # 航段缺速度, 由航路检查负责报
        # 按计划打不出的射击照样核验: 前面的击毁一旦没生效, 它会在包络外开火。
        # 实测: 先手方把对方射手提前打掉, 对方那几发计划仍停在 100 km 外
        voided = _voided_by(engagement, destroyed)
        horizon = duration
        if voided is None:
            for platform_id in (shooter.platform_id, target.platform_id):
                killed = destroyed.get(platform_id)
                if killed is not None and killed[1] != engagement.engagement_id:
                    horizon = min(horizon, killed[0])
        spans = _spans(lambda t: _within_or_none(_slant_km(shooter, target, t), low, high), horizon)
        if spans:
            advice = (
                f"按当前航路两者相距在包络内的时段是 {_join_spans(spans)}; 请把开火时刻调进这些时段"
                f" (离端点留 {_FIRE_MARGIN_S:g}s 以上余量, 之后改航路时不至于又滑出包络), 或调整航路"
            )
        else:
            advice = f"按当前航路, {horizon:g}s 之前两者都不在包络内; 请调整航路或改打别的目标"
        if voided is not None:
            pid, killed_at, killer = voided
            advice = (
                f"按计划 {pid} 在 {killed_at:g}s 已被交战 {killer} 摧毁, 这发本打不出; 但前面的击毁一旦没生效它就会开火,"
                f" 计划时刻同样要落在包络内。{advice}"
                + (
                    f"; 挪到 {killed_at:g}s 之前会让这发真的打出去"
                    if any(start < killed_at for start, _ in spans)
                    else ""
                )
            )
        if km is None:
            finished = "、".join(f"{pid} 的航路 {end:.0f}s 飞完" for pid, end in ended)
            messages.append(
                f"交战 {engagement.engagement_id} 在 {moment:g}s 开火时航路已经飞完 ({finished}),"
                f" 之后的位置不由 IR 决定, 斜距不能保证在 {_span(low, high, 'km')} 内;"
                f" 请续上航点让航路覆盖到开火时刻。{advice}"
            )
            continue
        messages.append(
            f"交战 {engagement.engagement_id} 在 {moment:g}s 开火时 {shooter.platform_id} 与 "
            f"{target.platform_id} 斜距约 {km:.0f} km, 不在要求的 {_span(low, high, 'km')} 内。{advice}"
        )
    return messages


def _reach_area(scenario, params, platforms, plan) -> list[str]:
    start, end = params.get("start_s"), params.get("end_s")
    must_survive = params.get("alive", True)
    area = _area_text(params)
    messages: list[str] = []
    for platform in platforms:
        entry = _first_entry(platform, params, scenario.mission.duration_s)
        if entry is None:
            continue  # 航段缺速度, 由航路检查负责报
        entered_at, travelled_m, extent = entry
        pid = platform.platform_id
        if entered_at is None:
            messages.append(
                f"{pid} 按当前航路进不了要求区域 ({area}); 它飞到的范围是 {extent}, 请调整航路"
            )
            continue
        problems: list[str] = []
        killed = plan.destroyed.get(pid)
        if must_survive and killed is not None and killed[0] <= entered_at:
            problems.append(
                f"它在 {killed[0]:g}s 被交战 {killed[1]} 摧毁, 早于进区时刻, 而需求要求进区时仍存活;"
                + _spare_advice(scenario, killed[1], pid, entered_at, plan)
            )
        if not _within(entered_at, start, end):
            problems.append(_entry_timing_advice(platform, travelled_m, entered_at, start, end))
        if problems:
            timing = "" if start is None and end is None else f", 要求 {_span(start, end, 's')}"
            messages.append(
                f"{pid} 在约 {entered_at:.0f}s 首次进入要求区域 ({area}){timing}: "
                + "; ".join(problems)
            )
    return messages


def _spare_advice(scenario, engagement_id: str, pid: str, entered_at: float, plan: _Plan) -> str:
    """让平台活到进区的建议; 射手的开火时段被需求限死在进区之前时, 推迟射击行不通。"""
    shot = next((e for e in scenario.engagements if e.engagement_id == engagement_id), None)
    window = plan.fire_windows.get(shot.shooter_platform) if shot is not None else None
    if window is None or window[1] is None or window[1] >= entered_at:
        return " 请推迟或改掉这次射击, 或让它更早进区"
    return (
        f" 射手 {shot.shooter_platform} 按需求只能在 {_span(*window, 's')} 开火, 推迟到进区之后会出时段;"
        f" 请让它改打别的目标, 或让 {pid} 更早进区"
    )


def _altitude_range(_scenario, params, platforms, _plan) -> list[str]:
    low, high = params.get("min_m"), params.get("max_m")
    messages = []
    for platform in platforms:
        points = [("初始位置", platform.position.altitude_m)]
        points += [(f"航点{i}", rp.position.altitude_m) for i, rp in enumerate(platform.route, 1)]
        bad = [f"{name} {value:g}m" for name, value in points if not _within(value, low, high)]
        if bad:
            messages.append(
                f"{platform.platform_id} 的高度 {'、'.join(bad)} 不在要求的 {_span(low, high, 'm')} 内"
            )
    return messages


def _speed_range(_scenario, params, platforms, _plan) -> list[str]:
    low, high = params.get("min_mps"), params.get("max_mps")
    messages = []
    for platform in platforms:
        bad = [
            f"航点{i} {rp.speed_mps:g}m/s" for i, rp in enumerate(platform.route, 1)
            if rp.speed_mps is not None and not _within(rp.speed_mps, low, high)
        ]
        if bad:
            messages.append(
                f"{platform.platform_id} 的速度 {'、'.join(bad)} 不在要求的 {_span(low, high, 'm/s')} 内"
            )
    return messages


def _route_points(scenario, params, platforms, _plan) -> list[str]:
    need = params["min_count"]
    duration = scenario.mission.duration_s
    messages = []
    for platform in platforms:
        count = len(platform.route)
        if count >= need:
            continue
        times = waypoint_times_s(platform) if platform.route else None
        budget = f"; 当前航路预计 {times[-1]:.0f}s 飞完 (任务 {duration:g}s)" if times else ""
        messages.append(
            f"{platform.platform_id} 只有 {count} 个航点, 要求至少 {need} 个{budget};"
            f" 请拆分长航段或加折线航点, 保持航路在任务时长内飞完"
        )
    return messages


_EVALUATORS = {
    "side_separation": _side_separation,
    "formation_spacing": _formation_spacing,
    "fire_window": _fire_window,
    "fire_phases": _fire_phases,
    "engagement_range": _engagement_range,
    "reach_area": _reach_area,
    "altitude_range": _altitude_range,
    "speed_range": _speed_range,
    "route_points": _route_points,
}


# ═══════════════════════════════════════════════════════
# 推算辅助
# ═══════════════════════════════════════════════════════


def _happens(engagement: ScheduledEngagementIR, destroyed: dict[str, tuple[float, str]]) -> bool:
    """射手和目标在开火时都还在 (同一秒互射谁先谁后无法预知, 按会发生算)。"""
    return _voided_by(engagement, destroyed) is None


def _voided_by(
    engagement: ScheduledEngagementIR, destroyed: dict[str, tuple[float, str]]
) -> tuple[str, float, str] | None:
    """开火前已被摧毁的一方 ``(平台, 时刻, 交战 id)``; 射击会发生时返回 None。"""
    moment = engagement.fire_time_s
    shooter = destroyed.get(engagement.shooter_platform)
    if shooter is not None and shooter[0] < moment:
        return engagement.shooter_platform, *shooter
    target = destroyed.get(engagement.target_platform)
    if target is not None and (
        target[0] < moment or (target[0] == moment and target[1] != engagement.engagement_id)
    ):
        return engagement.target_platform, *target
    return None


def _route_end_s(platform: PlatformIR) -> float | None:
    """航路飞完的时刻; 静止平台或航段缺速度时返回 None。"""
    times = waypoint_times_s(platform) if platform.route else None
    return times[-1] if times else None


def _slant_km(first: PlatformIR, second: PlatformIR, moment: float) -> float | None:
    a, b = position_at(first, moment), position_at(second, moment)
    if a is None or b is None:
        return None
    return slant_range_m(a, b) / 1000


def _within_or_none(value: float | None, low: float | None, high: float | None) -> bool:
    return value is not None and _within(value, low, high)


def _spans(predicate, horizon: float) -> list[tuple[float, float]]:
    """按固定步长扫描 [0, horizon], 返回 predicate 成立的时段。"""
    spans: list[tuple[float, float]] = []
    opened: float | None = None
    last = 0.0
    steps = int(horizon // _RANGE_STEP_S)
    for index in range(steps + 1):
        moment = index * _RANGE_STEP_S
        if predicate(moment):
            if opened is None:
                opened = moment
            last = moment
        elif opened is not None:
            spans.append((opened, last))
            opened = None
    if opened is not None:
        spans.append((opened, last))
    return spans


def _join_spans(spans: list[tuple[float, float]]) -> str:
    text = "、".join(f"{start:.0f}-{end:.0f}s" for start, end in spans[:_MAX_SPANS])
    if len(spans) > _MAX_SPANS:
        text += f" 等 {len(spans)} 段"
    return text


def _inside(point: GeoPoint, params: dict[str, Any]) -> bool:
    return _within(point.latitude_deg, params.get("lat_min"), params.get("lat_max")) and _within(
        point.longitude_deg, params.get("lon_min"), params.get("lon_max")
    )


def _first_entry(
    platform: PlatformIR, params: dict[str, Any], duration_s: float
) -> tuple[float | None, float, str] | None:
    """首次进入区域的时刻、此前的航程 (米) 和飞到的经纬度范围; 航段缺速度时返回 None。

    逐秒采样并补上航点时刻, 航路飞完或任务结束即停。
    """
    if not platform.route:
        position = platform.position
        return (0.0 if _inside(position, params) else None), 0.0, _extent([position])
    times = waypoint_times_s(platform)
    if times is None:
        return None
    horizon = min(duration_s, times[-1])
    moments = sorted({float(t) for t in range(int(horizon) + 1)} | {t for t in times if t <= horizon})
    seen: list[GeoPoint] = []
    travelled = 0.0
    previous: GeoPoint | None = None
    for moment in moments:
        point = position_at(platform, moment)
        if point is None:
            break
        if previous is not None:
            travelled += ground_distance_m(previous, point)
        previous = point
        seen.append(point)
        if _inside(point, params):
            return moment, travelled, _extent(seen)
    return None, travelled, _extent(seen)


def _extent(points: list[GeoPoint]) -> str:
    lats = [p.latitude_deg for p in points]
    lons = [p.longitude_deg for p in points]
    return (
        f"纬度 {min(lats):.2f}-{max(lats):.2f}°N、经度 {min(lons):.2f}-{max(lons):.2f}°E"
    )


def _entry_timing_advice(
    platform: PlatformIR,
    travelled_m: float,
    entered_at: float,
    start: float | None,
    end: float | None,
) -> str:
    times = waypoint_times_s(platform) or []
    schedule = "/".join(f"{t:.0f}" for t in times)
    km = travelled_m / 1000
    fastest = travelled_m / start if start else None
    slowest = travelled_m / end if end else None
    if fastest is not None and slowest is not None:
        speed = f"{slowest:.0f}-{fastest:.0f} m/s"
    elif slowest is not None:
        speed = f"至少 {slowest:.0f} m/s"
    else:
        speed = f"不超过 {fastest:.0f} m/s"
    return (
        f"进区前航程约 {km:.0f} km (航点预计到达 {schedule}s), 要在要求时段进区, 这段的平均速度"
        f"需在 {speed}; 速度受限时改航程 (增删或挪动进区前的航点)"
        + _entry_target(travelled_m, entered_at, start, end)
    )


def _entry_target(
    travelled_m: float, entered_at: float, start: float | None, end: float | None
) -> str:
    """按当前平均速度给出离边界留有余量的进区时刻和对应航程。

    实测: 只给速度区间时模型把进区时刻改到 897-898 s, 要求 900-1200 s, 连续两轮差两三秒;
    速度又被需求限死在 220 m/s 以上, 只能改航程, 却不知道改多少。
    """
    if entered_at <= 0:
        return ""
    margin = _ENTRY_MARGIN_S
    if start is not None and end is not None:
        margin = min(margin, (end - start) / 4)
    if start is not None and entered_at < start:
        target = start + margin
    elif end is not None and entered_at > end:
        target = end - margin
    else:
        return ""
    speed = travelled_m / entered_at
    return (
        f"; 按当前进区前平均速度约 {speed:.0f} m/s, 进区前航程改到约 {speed * target / 1000:.0f} km"
        f" 可在 {target:.0f}s 左右进区, 离时段边界留出余量"
    )


def _area_text(params: dict[str, Any]) -> str:
    parts = []
    if "lat_min" in params or "lat_max" in params:
        parts.append(f"纬度 {_span(params.get('lat_min'), params.get('lat_max'), '°N')}")
    if "lon_min" in params or "lon_max" in params:
        parts.append(f"经度 {_span(params.get('lon_min'), params.get('lon_max'), '°E')}")
    return "、".join(parts)


def _within(value: float, low: float | None, high: float | None) -> bool:
    return (low is None or value >= low) and (high is None or value <= high)


def _span(low: float | None, high: float | None, unit: str) -> str:
    if low is not None and high is not None:
        return f"{low:g}-{high:g}{unit}"
    if low is not None:
        return f"≥{low:g}{unit}"
    return f"≤{high:g}{unit}"
