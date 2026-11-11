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

from nanobot.afsim.schemas import GeoPoint, PlatformIR, ScenarioIR, ScheduledEngagementIR

EARTH_RADIUS_M = 6_371_008.8

# 航路采样步长: 每段至少切这么多份, 也不超过这个间距
_SAMPLE_STEP_M = 2_000.0
_MAX_SAMPLES_PER_LEG = 256

# 判定"绝对不可达"时的安全余量, 吸收采样误差与地球模型误差
_SAFETY_MARGIN_M = 2 * _SAMPLE_STEP_M

# 按时刻推算位置时的余量: 匀速推算忽略转弯半径和加减速, 误差比纯几何采样大
_TIMING_MARGIN_M = 10_000.0


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


def position_at(platform: PlatformIR, time_s: float) -> GeoPoint | None:
    """按各航段速度匀速推算平台在 ``time_s`` 秒时的位置。

    忽略转弯半径和加减速, 只适合开火时刻这类粗粒度判定。静止平台返回初始位置;
    航段缺速度, 或时刻已过航路终点 (之后怎么飞不由 IR 决定) 时返回 None。
    """
    if not platform.route:
        return platform.position
    start = platform.position
    elapsed = 0.0
    speed = 0.0
    for point in platform.route:
        if point.speed_mps:
            speed = point.speed_mps
        if speed <= 0:
            return None
        leg_s = ground_distance_m(start, point.position) / speed
        if time_s <= elapsed + leg_s:
            fraction = (time_s - elapsed) / leg_s if leg_s > 0 else 1.0
            return _interpolate(start, point.position, max(0.0, fraction))
        elapsed += leg_s
        start = point.position
    return None


def waypoint_times_s(platform: PlatformIR) -> list[float] | None:
    """按各航段速度匀速推算到达每个航路点的时刻 (秒); 有航段缺速度时返回 None。"""
    start = platform.position
    elapsed = 0.0
    speed = 0.0
    times: list[float] = []
    for point in platform.route:
        if point.speed_mps:
            speed = point.speed_mps
        if speed <= 0:
            return None
        elapsed += ground_distance_m(start, point.position) / speed
        times.append(elapsed)
        start = point.position
    return times


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
            # 修法所需的数字直接算给模型: 推理模型在思考里逐条重算航程, 实测会耗尽输出预算
            length = route_length_m(platform)
            times = waypoint_times_s(platform) or []
            reached = sum(1 for t in times if t <= duration)
            findings.append(
                GeometryFinding(
                    kind="route_exceeds_duration",
                    message=(
                        f"平台 {platform.platform_id} 航路全长约 "
                        f"{length / 1000:.0f} km, 按给定速度需要 "
                        f"{needed / 60:.0f} 分钟, 超过任务时长 {duration / 60:.0f} 分钟, "
                        f"到 {duration:.0f}s 时只飞完前 {reached}/{len(times)} 个航点, "
                        f"后段航路不会被走到。全程走完平均速度至少要 {length / duration:.0f} m/s; "
                        f"请缩短航路、提高速度或延长 duration_s"
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
    """交战必须有探测支撑: 射手一方得有传感器真的够得着目标。

    先查全程 (不看时间同步) 是否够得着; 够得着的再按开火时刻的推算位置复查一次。
    定时开火不看探测结果, 而 WSF_IMPLICIT_WEAPON 也不按距离判定命中: 开火时目标在探测圈外,
    这一发照样摧毁一个谁也看不见的目标, 想定跑得通, 只是结果失真。
    """
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
            continue

        finding = _check_fire_time_detection(
            engagement, shooter, target, friendly, sensor_range, scenario.mission.duration_s
        )
        if finding is not None:
            findings.append(finding)
    return findings


def _detection_windows(
    target: PlatformIR,
    friendly: list[PlatformIR],
    sensor_range: dict[str, float],
    duration_s: float,
) -> list[tuple[float, float]]:
    """按航路匀速推算, 我方至少一个传感器够得着目标的时段 (秒)。

    航路飞完后位置推算不出, 这些时刻按够不着算。只用来给修复提示, 不参与判定。
    """
    step = max(10.0, duration_s / 360)
    windows: list[tuple[float, float]] = []
    opened: float | None = None
    last = 0.0
    moment = 0.0
    while moment <= duration_s:
        if _covered_at(moment, target, friendly, sensor_range):
            if opened is None:
                opened = moment
            last = moment
        elif opened is not None:
            windows.append((opened, last))
            opened = None
        moment += step
    if opened is not None:
        windows.append((opened, last))
    return windows


def _covered_at(
    moment: float,
    target: PlatformIR,
    friendly: list[PlatformIR],
    sensor_range: dict[str, float],
) -> bool:
    target_at = position_at(target, moment)
    if target_at is None:
        return False
    for platform in friendly:
        observer_at = position_at(platform, moment)
        if observer_at is None:
            continue
        if slant_range_m(observer_at, target_at) <= _platform_sensor_reach(platform, sensor_range):
            return True
    return False


def _check_fire_time_detection(
    engagement: ScheduledEngagementIR,
    shooter: PlatformIR,
    target: PlatformIR,
    friendly: list[PlatformIR],
    sensor_range: dict[str, float],
    duration_s: float,
) -> GeometryFinding | None:
    """开火时刻我方是否至少有一个传感器够得着目标; 位置推算不出来就不报。"""
    moment = engagement.fire_time_s
    target_at = position_at(target, moment)
    if target_at is None:
        return None
    best: tuple[float, str, float, float] | None = None
    for platform in friendly:
        observer_at = position_at(platform, moment)
        if observer_at is None:
            return None  # 有一方位置不确定, 不能断言够不到
        reach = _platform_sensor_reach(platform, sensor_range)
        separation = slant_range_m(observer_at, target_at)
        if best is None or reach - separation > best[0]:
            best = (reach - separation, platform.platform_id, separation, reach)
    if best is None or best[0] + _TIMING_MARGIN_M >= 0:
        return None
    _, observer, separation, reach = best
    # 可探测时段由代码算好: 让模型在思考里按航速逐个推算交会时刻, 实测会耗尽输出预算
    windows = _detection_windows(target, friendly, sensor_range, duration_s)
    if windows:
        spans = "、".join(f"{start:.0f}-{end:.0f}s" for start, end in windows[:3])
        advice = f"按当前航路, 我方能探测到它的时段是 {spans}; 请把开火时刻调进这些时段, 或调整航路"
    else:
        advice = "按当前航路, 任务时长内我方都探测不到它 (航路飞完后的位置不计); 请调整航路"
    return GeometryFinding(
        kind="engagement_out_of_detection_at_fire_time",
        message=(
            f"交战 {engagement.engagement_id} 在 {moment:.0f}s 开火时, 按航路匀速推算 "
            f"{shooter.side.value} 方所有传感器都够不到 {target.platform_id}: 覆盖最好的 {observer} "
            f"相距约 {separation / 1000:.0f} km, 探测距离只有 {reach / 1000:.0f} km。"
            f"定时开火不管目标是否被探测到, 这一发会打向我方看不见的目标。{advice}"
        ),
    )
