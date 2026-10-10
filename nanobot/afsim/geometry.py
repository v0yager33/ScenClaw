"""ScenarioIR 的几何可达性计算。

解决一类"语法全绿但想定没意义"的问题: 需求写着"预警机先探测到敌方雷达阵地",
生成出来的想定里预警机带 150 km 雷达、目标却在 440 km 外 —— 断言永远不会触发,
任务目标永远达不成。编译器和 linter 都发现不了, 因为这不是语法问题。

放在 IR 层算是因为此时还有域语义 (谁带什么传感器、谁沿什么航路走); 一旦落成
AFSIM 文本, 就只剩一堆坐标字符串了。

**判定必须保守**: 只有在"无论如何都不可能"时才报不可达。因此:

- 距离用 3D 斜距 (大圆地面距离与高度差的合成), 而不是只算地面距离 —— 高空平台
  与地面目标之间, 高度差是真实存在的斜距分量。
- 两条航路的最近距离用采样近似。采样只会**高估**真实最近距离, 直接拿它判"超出
  射程"会误报, 所以要减掉一个不小于采样步长的安全余量再比较。
- 忽略时间同步: 取两条轨迹上所有位置两两组合的最小值。这是时间同步下真实最近
  距离的下界 —— 时间同步的位置对只是全组合的子集。用下界判"不可达"才不会冤枉
  一个其实能相遇的想定。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from nanobot.afsim.schemas import GeoPoint, PlatformIR, ScenarioIR

EARTH_RADIUS_M = 6_371_008.8

# 航路采样步长: 每段至少切这么多份, 也不超过这个间距
_SAMPLE_STEP_M = 2_000.0
_MAX_SAMPLES_PER_LEG = 256

# 判定"绝对不可达"时的安全余量, 吸收采样误差与地球模型误差
_SAFETY_MARGIN_M = 2 * _SAMPLE_STEP_M


def ground_distance_m(a: GeoPoint, b: GeoPoint) -> float:
    """两点间大圆地面距离 (米), haversine。"""
    lat1, lat2 = math.radians(a.latitude_deg), math.radians(b.latitude_deg)
    dlat = lat2 - lat1
    dlon = math.radians(b.longitude_deg - a.longitude_deg)
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, h)))


def slant_range_m(a: GeoPoint, b: GeoPoint) -> float:
    """两点间 3D 斜距 (米): 地面距离与高度差的合成。

    传感器 ``maximum_range`` 比的是斜距, 只算地面距离会低估高空平台到地面目标的
    真实距离。
    """
    ground = ground_distance_m(a, b)
    return math.hypot(ground, b.altitude_m - a.altitude_m)


def _interpolate(a: GeoPoint, b: GeoPoint, fraction: float) -> GeoPoint:
    """在 a→b 之间线性插值。

    这里用经纬度线性插值而不是严格大圆插值: 本模块处理的航段通常在几百公里量级,
    两者差异远小于 ``_SAFETY_MARGIN_M``, 而线性插值不会在跨极/跨日界线时产生
    难以调试的异常值。
    """
    return GeoPoint(
        latitude_deg=a.latitude_deg + (b.latitude_deg - a.latitude_deg) * fraction,
        longitude_deg=a.longitude_deg + (b.longitude_deg - a.longitude_deg) * fraction,
        altitude_m=a.altitude_m + (b.altitude_m - a.altitude_m) * fraction,
    )


def waypoints(platform: PlatformIR) -> list[GeoPoint]:
    """平台经过的关键点: 初始位置 + 各航路点。"""
    points = [platform.position]
    points.extend(point.position for point in platform.route)
    return points


def sample_trajectory(platform: PlatformIR) -> list[GeoPoint]:
    """把平台轨迹采样成一串位置点; 静止平台只有一个点。"""
    keys = waypoints(platform)
    if len(keys) == 1:
        return keys

    samples: list[GeoPoint] = [keys[0]]
    for start, end in zip(keys, keys[1:]):
        leg = ground_distance_m(start, end)
        steps = max(1, min(_MAX_SAMPLES_PER_LEG, int(leg / _SAMPLE_STEP_M) + 1))
        for index in range(1, steps + 1):
            samples.append(_interpolate(start, end, index / steps))
    return samples


def route_length_m(platform: PlatformIR) -> float:
    keys = waypoints(platform)
    return sum(
        ground_distance_m(start, end) for start, end in zip(keys, keys[1:])
    )


def route_duration_s(platform: PlatformIR, default_speed_mps: float = 0.0) -> float | None:
    """按各航段速度估算走完航路所需时间 (秒); 无法确定速度时返回 None。

    速度取该航段终点航路点上的 ``speed_mps``; 没给就沿用上一段的速度。
    """
    keys = waypoints(platform)
    if len(keys) < 2:
        return 0.0

    speed = default_speed_mps
    total = 0.0
    for index, (start, end) in enumerate(zip(keys, keys[1:])):
        point_speed = platform.route[index].speed_mps
        if point_speed:
            speed = point_speed
        if speed <= 0:
            return None
        total += ground_distance_m(start, end) / speed
    return total


def min_separation_m(first: PlatformIR, second: PlatformIR) -> float:
    """两个平台在整个想定内可能达到的最近斜距的**下界**。

    忽略时间同步 (取两条轨迹所有位置的两两最小值), 因此结果不大于真实的时间同步
    最近距离。用它判"绝对不可达"才不会误报。
    """
    left = sample_trajectory(first)
    right = sample_trajectory(second)
    best = math.inf
    for a in left:
        for b in right:
            distance = slant_range_m(a, b)
            if distance < best:
                best = distance
    return best


# ═══════════════════════════════════════════════════════
# 可达性检查
# ═══════════════════════════════════════════════════════


@dataclass(frozen=True)
class GeometryFinding:
    """一条几何可达性问题。"""
    kind: str
    message: str


def check_geometry(scenario: ScenarioIR) -> list[GeometryFinding]:
    """检查探测/交战/航路的几何可达性。

    空间域平台会被整体跳过: 编译器不支持轨道运动 (见 capabilities 里
    WSF_SPACE_MOVER 的说明), 卫星只能被摆成一个静止点, 那个点的经纬度本身就是
    没有物理意义的占位值。对它做几何判定只会产出无法修复的告警, 把修复循环带进
    死胡同 —— 轨道建模缺失这件事已经由 capability_gaps 如实声明过了。
    """
    findings: list[GeometryFinding] = []
    by_id = {platform.platform_id: platform for platform in scenario.platforms}
    sensor_range = {
        sensor.sensor_type_id: sensor.maximum_range_m for sensor in scenario.sensors
    }
    unmodelled = {
        intent.platform_id
        for intent in scenario.mission.platforms
        if intent.domain.value == "space"
    }

    findings.extend(_check_assertions(scenario, by_id, sensor_range, unmodelled))
    findings.extend(_check_routes(scenario))
    findings.extend(_check_engagements(scenario, by_id, sensor_range, unmodelled))
    return findings


def _platform_sensor_reach(platform: PlatformIR, sensor_range: dict[str, float]) -> float:
    """平台上所有传感器里最大的探测距离; 无传感器返回 0。"""
    return max(
        (sensor_range.get(sensor.sensor_type_id, 0.0) for sensor in platform.sensors),
        default=0.0,
    )


def _check_assertions(
    scenario: ScenarioIR,
    by_id: dict[str, PlatformIR],
    sensor_range: dict[str, float],
    unmodelled: set[str],
) -> list[GeometryFinding]:
    findings: list[GeometryFinding] = []
    for assertion in scenario.assertions:
        if assertion.event_type == "WEAPON_FIRED":
            continue  # 与传感器几何无关
        source = by_id.get(assertion.source_platform or "")
        target = by_id.get(assertion.target_platform)
        if source is None or target is None:
            continue
        if source.platform_id in unmodelled or target.platform_id in unmodelled:
            continue  # 运动学未建模, 位置无物理意义

        if assertion.sensor_name:
            matched = [
                sensor for sensor in source.sensors
                if sensor.instance_name == assertion.sensor_name
            ]
            reach = max(
                (sensor_range.get(s.sensor_type_id, 0.0) for s in matched), default=0.0
            )
        else:
            reach = _platform_sensor_reach(source, sensor_range)
        if reach <= 0:
            continue  # 传感器缺失由合理性检查负责报

        separation = min_separation_m(source, target)
        if separation - _SAFETY_MARGIN_M > reach:
            findings.append(
                GeometryFinding(
                    kind="assertion_out_of_range",
                    message=(
                        f"断言 {assertion.assertion_id} 不可能成立: "
                        f"{assertion.source_platform} 与 {assertion.target_platform} 全程最近距离约 "
                        f"{separation / 1000:.0f} km, 而传感器最大探测距离只有 "
                        f"{reach / 1000:.0f} km。请把双方拉近、延长航路使其进入探测圈, "
                        f"或把 maximum_range_m 提高到超过 {separation / 1000:.0f} km"
                    ),
                )
            )
    return findings


def _check_routes(scenario: ScenarioIR) -> list[GeometryFinding]:
    findings: list[GeometryFinding] = []
    duration = scenario.mission.duration_s
    for platform in scenario.platforms:
        if not platform.route:
            continue
        needed = route_duration_s(platform)
        if needed is None:
            findings.append(
                GeometryFinding(
                    kind="route_speed_missing",
                    message=(
                        f"平台 {platform.platform_id} 的航路点没有给 speed_mps, "
                        f"无法判断它能否在任务时长内走完航路"
                    ),
                )
            )
            continue
        if needed > duration:
            findings.append(
                GeometryFinding(
                    kind="route_exceeds_duration",
                    message=(
                        f"平台 {platform.platform_id} 航路全长约 "
                        f"{route_length_m(platform) / 1000:.0f} km, 按给定速度需要 "
                        f"{needed / 60:.0f} 分钟, 超过任务时长 {duration / 60:.0f} 分钟, "
                        f"后段航路不会被走到。请缩短航路、提高速度或延长 duration_s"
                    ),
                )
            )
    return findings


def _check_engagements(
    scenario: ScenarioIR,
    by_id: dict[str, PlatformIR],
    sensor_range: dict[str, float],
    unmodelled: set[str],
) -> list[GeometryFinding]:
    """交战必须有探测支撑: 射手一方得有传感器真的够得着目标。"""
    findings: list[GeometryFinding] = []
    for engagement in scenario.engagements:
        shooter = by_id.get(engagement.shooter_platform)
        target = by_id.get(engagement.target_platform)
        if shooter is None or target is None:
            continue
        if shooter.platform_id in unmodelled or target.platform_id in unmodelled:
            continue  # 运动学未建模, 位置无物理意义

        friendly = [
            platform for platform in scenario.platforms
            if platform.side == shooter.side
            and platform.sensors
            and platform.platform_id not in unmodelled
        ]
        if not friendly:
            continue  # 该方全无传感器, 由合理性检查负责报

        best_margin = -math.inf
        best_reach = 0.0
        best_separation = math.inf
        for platform in friendly:
            reach = _platform_sensor_reach(platform, sensor_range)
            separation = min_separation_m(platform, target)
            margin = reach - separation
            if margin > best_margin:
                best_margin, best_reach, best_separation = margin, reach, separation

        if best_margin + _SAFETY_MARGIN_M < 0:
            findings.append(
                GeometryFinding(
                    kind="engagement_without_detection",
                    message=(
                        f"交战 {engagement.engagement_id} 缺少探测支撑: "
                        f"{shooter.side.value} 方所有传感器全程都够不到 "
                        f"{engagement.target_platform} (最好情况最近距离约 "
                        f"{best_separation / 1000:.0f} km, 探测距离 {best_reach / 1000:.0f} km)。"
                        f"发射事件仍会按时刻触发, 但整条侦察-锁定-打击链断了"
                    ),
                )
            )
    return findings
