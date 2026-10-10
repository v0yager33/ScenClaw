"""Typed interchange objects used by the scenario-construction pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base type that rejects unknown fields instead of silently ignoring them."""

    model_config = ConfigDict(extra="forbid", frozen=False)


class Side(str, Enum):
    BLUE = "blue"
    RED = "red"
    NEUTRAL = "neutral"


class Domain(str, Enum):
    AIR = "air"
    LAND = "land"
    SEA = "sea"
    SPACE = "space"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


# 运动域 → 该域唯一合法的 mover 类型。
#
# AFSIM 的 mover 按运动域分家: 舰船走水面、车辆走地面、飞机走空中。用
# WSF_AIR_MOVER 去驱动一艘驱逐舰, 语法完全合法、linter 也挑不出毛病, 但运动学
# 是错的 —— 船会"飞"。这类错误只能在 schema 层按域强约束拦住。
#
# space 故意映射为空集: WSF_SPACE_MOVER 用开普勒根数描述轨道, 与本 IR 的
# 经纬高 + 航点模型不同构, 硬套航路会得到轨道语义完全错误的想定。
_MOVER_BY_DOMAIN: dict[str, frozenset[str]] = {
    "air": frozenset({"WSF_AIR_MOVER"}),
    "sea": frozenset({"WSF_SURFACE_MOVER"}),
    "land": frozenset({"WSF_GROUND_MOVER"}),
    "space": frozenset(),
}

# 水面平台的合理高度上限 (米): 船浮在海面上, 不该出现在几千米高度
_SEA_ALTITUDE_CEILING_M = 100.0


class GeoPoint(StrictModel):
    latitude_deg: float = Field(ge=-90.0, le=90.0)
    longitude_deg: float = Field(ge=-180.0, le=180.0)
    altitude_m: float = Field(default=0.0, ge=-500.0, le=100_000.0)


class PlatformIntent(StrictModel):
    platform_id: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    side: Side
    domain: Domain
    type_hint: str = Field(min_length=1)
    initial_position: GeoPoint
    route: list[GeoPoint] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)


class MissionSpec(StrictModel):
    schema_version: str = "0.1.0"
    mission_id: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    name: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    duration_s: float = Field(gt=0.0, le=604_800.0)
    random_seed: int = Field(default=1, ge=0)
    platforms: list[PlatformIntent] = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def platform_ids_are_unique(self) -> "MissionSpec":
        ids = [platform.platform_id for platform in self.platforms]
        if len(ids) != len(set(ids)):
            raise ValueError("platform_id values must be unique")
        return self


class TaskAssignment(StrictModel):
    platform_id: str
    task: str = Field(min_length=1)
    priority: int = Field(default=50, ge=0, le=100)


class COA(StrictModel):
    schema_version: str = "0.1.0"
    coa_id: str = Field(min_length=1)
    mission_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    assignments: list[TaskAssignment] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)


class ComponentReference(StrictModel):
    logical_name: str = Field(min_length=1)
    afsim_type: str = Field(min_length=1)
    source_reference: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class SensorIR(StrictModel):
    """A constrained sensor definition supported by the MVP compiler."""

    sensor_type_id: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    base_type: str = Field(default="WSF_GEOMETRIC_SENSOR", pattern=r"^WSF_[A-Z0-9_]+$")
    mode_name: str = Field(default="search", pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    frame_time_s: float = Field(default=1.0, gt=0.0)
    maximum_range_m: float = Field(gt=0.0)
    reports_location: bool = True
    ignore_same_side: bool = True


class PlatformSensorIR(StrictModel):
    instance_name: str = Field(default="radar", pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    sensor_type_id: str = Field(min_length=1)
    initial_mode: str = Field(default="search", min_length=1)
    enabled: bool = True


class PlatformWeaponIR(StrictModel):
    instance_name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    weapon_type: str = Field(default="WSF_IMPLICIT_WEAPON", pattern=r"^WSF_[A-Z0-9_]+$")
    quantity: int = Field(default=2, ge=1, le=1000)
    update_interval_s: float = Field(default=1.0, gt=0.0, le=60.0)


class RoutePointIR(StrictModel):
    position: GeoPoint
    speed_mps: float | None = Field(default=None, gt=0.0)
    heading_deg: float | None = Field(default=None, ge=0.0, lt=360.0)


class PlatformIR(StrictModel):
    platform_id: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    platform_type: str = Field(default="WSF_PLATFORM", pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    side: Side
    position: GeoPoint
    heading_deg: float = Field(default=0.0, ge=0.0, lt=360.0)
    mover_type: str | None = Field(default=None, pattern=r"^WSF_[A-Z0-9_]+$")
    sensors: list[PlatformSensorIR] = Field(default_factory=list)
    weapons: list[PlatformWeaponIR] = Field(default_factory=list)
    route: list[RoutePointIR] = Field(default_factory=list)

    @model_validator(mode="after")
    def route_requires_mover(self) -> "PlatformIR":
        if self.route and self.mover_type is None:
            raise ValueError("a platform with a route must define mover_type")
        return self

    @model_validator(mode="after")
    def mover_requires_route(self) -> "PlatformIR":
        """挂了 mover 就必须给航路。

        ``add mover`` 却不给 route 的平台在仿真里原地不动 —— 语法合法, 但与
        "突入/巡逻/推进"的意图矛盾。

        报错文案刻意把"补航路"放在首选: 模型既然给了这个平台 mover, 说明它判断该
        平台会动, 那么补航路点才是符合意图的修法。删 mover 只适用于雷达阵地、发射
        阵地这类本就固定的目标 —— 一律删 mover 会把拦截、巡逻这类想定整体变成静止,
        问题从"校验不通过"变成"悄悄降级", 更难发现。
        """
        if self.mover_type is not None and not self.route:
            raise ValueError(
                f"platform {self.platform_id} defines mover_type={self.mover_type} but has "
                f"no route, so it will never move. Preferred fix: add at least one route "
                f"waypoint describing where it goes (with speed_mps). Only if this platform "
                f"is a genuinely fixed installation (radar site, launch site, airfield) "
                f"should you drop mover_type instead"
            )
        return self


class EventAssertion(StrictModel):
    assertion_id: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    event_type: str = Field(
        pattern=r"^(SENSOR_DETECTION_ATTEMPT|SENSOR_TRACK_INITIATED|WEAPON_FIRED)$"
    )
    source_platform: str | None = Field(default=None, min_length=1)
    target_platform: str = Field(min_length=1)
    sensor_name: str | None = Field(default=None, min_length=1)
    detected_only: bool = False
    minimum_count: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def detected_only_is_for_attempts(self) -> "EventAssertion":
        if self.detected_only and self.event_type != "SENSOR_DETECTION_ATTEMPT":
            raise ValueError("detected_only is valid only for SENSOR_DETECTION_ATTEMPT")
        return self


class ScheduledEngagementIR(StrictModel):
    engagement_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    shooter_platform: str = Field(min_length=1)
    target_platform: str = Field(min_length=1)
    weapon_name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    fire_time_s: float = Field(ge=0.0)


class ScenarioIR(StrictModel):
    schema_version: str = "0.1.0"
    scenario_id: str = Field(min_length=1)
    mission: MissionSpec
    selected_coa: COA
    components: list[ComponentReference] = Field(default_factory=list)
    sensors: list[SensorIR] = Field(default_factory=list)
    platforms: list[PlatformIR] = Field(default_factory=list)
    assertions: list[EventAssertion] = Field(default_factory=list)
    engagements: list[ScheduledEngagementIR] = Field(default_factory=list)
    compiler_options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def mission_and_coa_match(self) -> "ScenarioIR":
        if self.selected_coa.mission_id != self.mission.mission_id:
            raise ValueError("selected_coa.mission_id must match mission.mission_id")
        known = {platform.platform_id for platform in self.mission.platforms}
        unknown = {
            item.platform_id for item in self.selected_coa.assignments if item.platform_id not in known
        }
        if unknown:
            raise ValueError(f"COA references unknown platforms: {sorted(unknown)}")
        mission_platforms = {platform.platform_id for platform in self.mission.platforms}
        compiled_platforms = {platform.platform_id for platform in self.platforms}
        if self.platforms and compiled_platforms != mission_platforms:
            missing = sorted(mission_platforms - compiled_platforms)
            extra = sorted(compiled_platforms - mission_platforms)
            raise ValueError(f"compiled platforms differ from mission: missing={missing}, extra={extra}")
        sensor_types = {sensor.sensor_type_id for sensor in self.sensors}
        missing_sensors = sorted(
            {
                sensor.sensor_type_id
                for platform in self.platforms
                for sensor in platform.sensors
                if sensor.sensor_type_id not in sensor_types
            }
        )
        if missing_sensors:
            raise ValueError(f"platforms reference unknown sensor types: {missing_sensors}")
        assertion_platforms = {
            platform_id
            for assertion in self.assertions
            for platform_id in (assertion.source_platform, assertion.target_platform)
            if platform_id is not None
        }
        unknown_assertion_platforms = sorted(assertion_platforms - compiled_platforms)
        if unknown_assertion_platforms:
            raise ValueError(
                f"assertions reference unknown platforms: {unknown_assertion_platforms}"
            )
        platform_by_id = {platform.platform_id: platform for platform in self.platforms}
        for engagement in self.engagements:
            if engagement.shooter_platform not in platform_by_id:
                raise ValueError(
                    f"engagement references unknown shooter: {engagement.shooter_platform}"
                )
            if engagement.target_platform not in platform_by_id:
                raise ValueError(
                    f"engagement references unknown target: {engagement.target_platform}"
                )
            weapon_names = {
                weapon.instance_name
                for weapon in platform_by_id[engagement.shooter_platform].weapons
            }
            if engagement.weapon_name not in weapon_names:
                raise ValueError(
                    f"engagement references unknown weapon {engagement.weapon_name} "
                    f"on {engagement.shooter_platform}"
                )
            if engagement.fire_time_s > self.mission.duration_s:
                raise ValueError("engagement fire time exceeds mission duration")
        return self

    @model_validator(mode="after")
    def movers_match_platform_domain(self) -> "ScenarioIR":
        """按运动域校验 mover 类型与平台高度。

        域信息只存在于 ``mission.platforms[].domain``, 编译后的 AFSIM 文本里就
        丢失了, 所以这条检查必须放在 IR 层。
        """
        domain_by_id = {
            intent.platform_id: intent.domain.value for intent in self.mission.platforms
        }
        problems: list[str] = []
        for platform in self.platforms:
            domain = domain_by_id.get(platform.platform_id)
            if domain is None:
                continue
            allowed = _MOVER_BY_DOMAIN.get(domain)
            if allowed is None:
                continue

            if not allowed:
                if platform.mover_type is not None or platform.route:
                    problems.append(
                        f"platform {platform.platform_id} is in the '{domain}' domain, "
                        f"which the compiler cannot move yet: WSF_SPACE_MOVER uses "
                        f"Keplerian orbital elements, not lat/lon waypoints. Either model "
                        f"it as a stationary platform (drop mover_type and route) or state "
                        f"that orbital motion is out of scope."
                    )
                continue

            if platform.mover_type is not None and platform.mover_type not in allowed:
                problems.append(
                    f"platform {platform.platform_id} is in the '{domain}' domain but uses "
                    f"mover_type={platform.mover_type}; the '{domain}' domain requires "
                    f"{sorted(allowed)[0]}"
                )

            if domain == "sea":
                altitudes = [platform.position.altitude_m] + [
                    point.position.altitude_m for point in platform.route
                ]
                if any(value > _SEA_ALTITUDE_CEILING_M for value in altitudes):
                    problems.append(
                        f"platform {platform.platform_id} is a sea-domain platform but has "
                        f"altitude above {_SEA_ALTITUDE_CEILING_M} m; surface vessels sit at "
                        f"sea level (use altitude_m 0)"
                    )

        if problems:
            raise ValueError("; ".join(problems))
        return self


class Diagnostic(StrictModel):
    code: str
    message: str
    source: str | None = None
    line: int | None = Field(default=None, ge=1)


class SimulationResult(StrictModel):
    schema_version: str = "0.1.0"
    run_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    status: RunStatus
    exit_code: int | None = None
    metrics: dict[str, float | int | str | bool] = Field(default_factory=dict)
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
