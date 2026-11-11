"""Deterministic ScenarioIR-to-AFSIM compiler."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .capabilities import (
    DEFAULT_CAPABILITY_REGISTRY,
    CapabilityKind,
    CapabilityRegistry,
)
from .schemas import GeoPoint, PlatformIR, RoutePointIR, ScenarioIR, ScheduledEngagementIR


class CompileError(ValueError):
    """Raised when a valid ScenarioIR cannot be compiled by the MVP backend."""


def _number(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".") or "0"


def _dms(value: float, positive: str, negative: str) -> str:
    """把十进制度转成 AFSIM 的 ``d:m:s`` 坐标词元。

    必须用 d:m:s 而不是十进制度。证据有两条且互相印证:

    - ``afsim-refs/errors-ref.md`` 的 E005 ``Invalid position format`` 把
      ``position 30.0N 120.0E`` 明确列为错误写法, 根因写的是"AFSIM 要求 d:m:s N/S e/w";
    - 真实可运行语料 (afsim_project) 里 189 行 position 全是 d:m:s, 十进制度 0 行。

    早先版本直接输出 ``30n 120e``, 也就是被文档点名的那种错误形式 —— 静态检查全绿,
    但真机大概率在解析阶段就报 E005。
    """
    hemisphere = positive if value >= 0 else negative
    total = abs(value)
    degrees = int(total)
    minutes_full = (total - degrees) * 60.0
    minutes = int(minutes_full)
    seconds = (minutes_full - minutes) * 60.0

    # 浮点误差可能把秒推到 60, 需要向上进位, 否则会写出 30:00:60n 这种非法值
    if round(seconds, 3) >= 60.0:
        seconds = 0.0
        minutes += 1
    if minutes >= 60:
        minutes -= 60
        degrees += 1

    seconds_text = f"{seconds:06.3f}".rstrip("0").rstrip(".")
    return f"{degrees}:{minutes:02d}:{seconds_text}{hemisphere}"


def _latitude(value: float) -> str:
    return _dms(value, "n", "s")


def _longitude(value: float) -> str:
    return _dms(value, "e", "w")


def _position(point: GeoPoint) -> str:
    return (
        f"{_latitude(point.latitude_deg)} {_longitude(point.longitude_deg)} "
        f"altitude {_number(point.altitude_m)} m"
    )


def _route_position(point: RoutePointIR) -> str:
    value = _position(point.position)
    if point.heading_deg is not None:
        value += f" heading {_number(point.heading_deg)} deg"
    if point.speed_mps is not None:
        value += f" speed {_number(point.speed_mps)} m/s"
    return value


def _platform_type_key(platform: PlatformIR) -> str:
    """统一两种输出的类型名；IR 型号是标签，不是已加载的 AFSIM 父类型。"""
    if platform.mover_type:
        suffix = platform.mover_type.removeprefix("WSF_").removesuffix("_MOVER")
        return f"{platform.platform_type}_{suffix}"
    if platform.platform_type.startswith("WSF_"):
        return f"{platform.platform_type}_STATIC"
    return platform.platform_type


def _fire_script(engagement: ScheduledEngagementIR) -> str:
    """定时开火脚本；目标已被删除时跳过，不向 MakeTrack/Fire 传空引用。

    即时效应武器命中后目标平台会被移出仿真。多个射手攻击同一目标时, 后执行的
    ``FindPlatform`` 返回空引用, 直接链式调用 ``MakeTrack()`` 会触发 Script Exception。
    判空写法与真实语料 ``prdata/rules_utils.txt`` 中的
    ``WsfPlatform target_plat = WsfSimulation.FindPlatform(...)`` / ``IsValid()`` 一致。
    """
    return "\n".join([
        f"   execute at_time {_number(engagement.fire_time_s)} sec absolute",
        f'      WsfPlatform target_plat = WsfSimulation.FindPlatform("{engagement.target_platform}");',
        "      if (target_plat.IsValid())",
        "      {",
        f'         Weapon("{engagement.weapon_name}").Fire(target_plat.MakeTrack());',
        "      }",
        "   end_execute",
    ])


def _platform_types(scenario: ScenarioIR) -> dict[str, tuple[str, str | None]]:
    """按型号和 mover 去重，并拒绝不同配置映射到同一个类型名。"""
    types: dict[str, tuple[str, str | None]] = {}
    for platform in scenario.platforms:
        key = _platform_type_key(platform)
        spec = (platform.platform_type, platform.mover_type)
        if key in types and types[key] != spec:
            raise CompileError(f"platform type name collision: {key}")
        types[key] = spec
    return types


class AfsimCompiler:
    """Compile the deliberately small, validated MVP subset of ScenarioIR."""

    def __init__(
        self,
        template_dir: Path | None = None,
        capability_registry: CapabilityRegistry = DEFAULT_CAPABILITY_REGISTRY,
    ) -> None:
        templates = template_dir or Path(__file__).with_name("templates")
        self._environment = Environment(
            loader=FileSystemLoader(templates),
            undefined=StrictUndefined,
            autoescape=False,
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._environment.filters.update(
            number=_number,
            position=_position,
            route_position=_route_position,
            platform_type_key=_platform_type_key,
            fire_script=_fire_script,
        )
        self._capability_registry = capability_registry

    def compile(self, scenario: ScenarioIR) -> str:
        if not scenario.sensors:
            raise CompileError("ScenarioIR.sensors must contain at least one sensor")
        if not scenario.platforms:
            raise CompileError("ScenarioIR.platforms must contain at least one platform")
        unsupported = sorted({
            sensor.base_type
            for sensor in scenario.sensors
            if not self._capability_registry.is_compiler_enabled(
                CapabilityKind.SENSOR_TYPE, sensor.base_type
            )
        })
        if unsupported:
            raise CompileError(f"unsupported MVP sensor base types: {unsupported}")
        unsupported_movers = sorted({
            platform.mover_type
            for platform in scenario.platforms
            if platform.mover_type is not None
            and not self._capability_registry.is_compiler_enabled(
                CapabilityKind.MOVER_TYPE, platform.mover_type
            )
        })
        if unsupported_movers:
            raise CompileError(f"unsupported MVP mover types: {unsupported_movers}")
        unsupported_weapons = sorted({
            weapon.weapon_type
            for platform in scenario.platforms
            for weapon in platform.weapons
            if not self._capability_registry.is_compiler_enabled(
                CapabilityKind.WEAPON_TYPE, weapon.weapon_type
            )
        })
        if unsupported_weapons:
            raise CompileError(f"unsupported MVP weapon types: {unsupported_weapons}")
        required_events = {
            "SENSOR_DETECTION_ATTEMPT",
            "SENSOR_TRACK_INITIATED",
            "EVENT_PIPE_REPLAY",
            "WEAPON_FIRED",
        }
        unsupported_events = sorted(
            event
            for event in required_events
            if not self._capability_registry.is_compiler_enabled(
                CapabilityKind.EVENT_TYPE, event
            )
        )
        if unsupported_events:
            raise CompileError(f"unsupported MVP event types: {unsupported_events}")
        template = self._environment.get_template("minimal_scenario.txt.j2")
        # 预计算唯一的 (platform_type, mover_type) 对, 让模板在实例化之前先声明类型。
        # 否则 linter 会报 "unknown platform type KJ500" (单文件模板之前靠默认值
        # WSF_PLATFORM 蒙混过关, 归一化层把默认值替换成具体类型后这个潜伏的 bug 就露出来了)。
        unique_types = [
            (key, mover) for key, (_, mover) in _platform_types(scenario).items()
        ]
        return template.render(
            scenario=scenario,
            unique_types=unique_types,
            has_weapons=any(platform.weapons for platform in scenario.platforms),
        )


def compile_scenario(scenario: ScenarioIR) -> str:
    return AfsimCompiler().compile(scenario)
