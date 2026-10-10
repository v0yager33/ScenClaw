"""几何可达性检查的测试。

这套检查的价值在于抓一类语法检查抓不到的错误: 想定跑得起来, 但传感器根本够不到
目标, 断言永远不触发, 任务目标永远达不成。

同时也要防它反过来添乱 —— 一旦误报"不可达", 修复循环就会去改一个本来正确的想定。
所以既测"该报的报得出", 也测"不该报的不报"。
"""

from __future__ import annotations

from typing import Any

import pytest

from nanobot.afsim import (
    GeoPoint,
    ScenarioIR,
    check_geometry,
    ground_distance_m,
    min_separation_m,
    route_duration_s,
    route_length_m,
    slant_range_m,
)


# ── 距离计算 ────────────────────────────────────────────────────────


def _point(lat: float, lon: float, alt: float = 0.0) -> GeoPoint:
    return GeoPoint(latitude_deg=lat, longitude_deg=lon, altitude_m=alt)


def test_ground_distance_zero_for_same_point() -> None:
    assert ground_distance_m(_point(27.5, 122.5), _point(27.5, 122.5)) == 0.0


def test_ground_distance_one_degree_latitude() -> None:
    """1 个纬度约 111 km, 这是提示词里告诉模型的换算基准。"""
    distance = ground_distance_m(_point(27.0, 122.0), _point(28.0, 122.0))
    assert distance == pytest.approx(111_195, rel=0.01)


def test_ground_distance_longitude_shrinks_with_latitude() -> None:
    """同样 1 个经度, 纬度越高地面距离越短。"""
    low = ground_distance_m(_point(0.0, 120.0), _point(0.0, 121.0))
    mid = ground_distance_m(_point(27.5, 120.0), _point(27.5, 121.0))
    high = ground_distance_m(_point(60.0, 120.0), _point(60.0, 121.0))
    assert low > mid > high


def test_ground_distance_matches_known_east_sea_geometry() -> None:
    """复现实测里那份想定的几何: KJ-500 在 119e, 目标在 123.5e。"""
    distance = ground_distance_m(_point(27.5, 119.0), _point(27.5, 123.5))
    assert distance / 1000 == pytest.approx(443.8, abs=1.0)


def test_slant_range_includes_altitude() -> None:
    """传感器 maximum_range 比的是斜距, 只算地面距离会低估。"""
    flat = _point(27.5, 122.5, 0.0)
    high = _point(27.5, 122.5, 9000.0)
    assert slant_range_m(flat, high) == pytest.approx(9000.0, abs=1.0)


def test_slant_range_is_never_less_than_ground_distance() -> None:
    a = _point(27.5, 119.0, 9000.0)
    b = _point(27.5, 123.5, 0.0)
    assert slant_range_m(a, b) >= ground_distance_m(a, b)


# ── 航路长度与耗时 ──────────────────────────────────────────────────


def _platform(
    platform_id: str,
    lat: float,
    lon: float,
    alt: float = 0.0,
    *,
    mover: str | None = None,
    route: list[dict[str, Any]] | None = None,
    sensors: list[dict[str, Any]] | None = None,
    weapons: list[dict[str, Any]] | None = None,
    side: str = "blue",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "platform_id": platform_id,
        "side": side,
        "position": {"latitude_deg": lat, "longitude_deg": lon, "altitude_m": alt},
    }
    if mover:
        payload["mover_type"] = mover
    if route:
        payload["route"] = route
    if sensors:
        payload["sensors"] = sensors
    if weapons:
        payload["weapons"] = weapons
    return payload


def _waypoint(lat: float, lon: float, alt: float = 0.0, speed: float | None = None) -> dict[str, Any]:
    point: dict[str, Any] = {
        "position": {"latitude_deg": lat, "longitude_deg": lon, "altitude_m": alt}
    }
    if speed is not None:
        point["speed_mps"] = speed
    return point


def _scenario(
    platforms: list[dict[str, Any]],
    *,
    sensors: list[dict[str, Any]] | None = None,
    assertions: list[dict[str, Any]] | None = None,
    engagements: list[dict[str, Any]] | None = None,
    duration_s: float = 3600.0,
    domains: dict[str, str] | None = None,
) -> ScenarioIR:
    domains = domains or {}
    return ScenarioIR.model_validate(
        {
            "scenario_id": "geo_probe",
            "mission": {
                "mission_id": "geo",
                "name": "几何探针",
                "objective": "验证几何可达性",
                "duration_s": duration_s,
                "platforms": [
                    {
                        "platform_id": p["platform_id"],
                        "side": p["side"],
                        "domain": domains.get(p["platform_id"], "air"),
                        "type_hint": "probe",
                        "initial_position": p["position"],
                    }
                    for p in platforms
                ],
            },
            "selected_coa": {
                "coa_id": "coa", "mission_id": "geo", "summary": "probe"
            },
            "sensors": sensors or [],
            "platforms": platforms,
            "assertions": assertions or [],
            "engagements": engagements or [],
        }
    )


def test_route_length_sums_legs() -> None:
    scenario = _scenario(
        [
            _platform(
                "mover1", 27.0, 122.0, 8000.0,
                mover="WSF_AIR_MOVER",
                route=[
                    _waypoint(28.0, 122.0, 8000.0, 250.0),
                    _waypoint(29.0, 122.0, 8000.0, 250.0),
                ],
            )
        ]
    )
    length = route_length_m(scenario.platforms[0])
    assert length / 1000 == pytest.approx(222.4, abs=2.0)


def test_route_duration_uses_leg_speeds() -> None:
    scenario = _scenario(
        [
            _platform(
                "mover1", 27.0, 122.0, 8000.0,
                mover="WSF_AIR_MOVER",
                route=[_waypoint(28.0, 122.0, 8000.0, 250.0)],
            )
        ]
    )
    seconds = route_duration_s(scenario.platforms[0])
    assert seconds == pytest.approx(111_195 / 250, rel=0.02)


def test_route_duration_none_when_speed_missing() -> None:
    scenario = _scenario(
        [
            _platform(
                "mover1", 27.0, 122.0, 8000.0,
                mover="WSF_AIR_MOVER",
                route=[_waypoint(28.0, 122.0, 8000.0)],
            )
        ]
    )
    assert route_duration_s(scenario.platforms[0]) is None


def test_static_platform_has_zero_route_duration() -> None:
    scenario = _scenario([_platform("static1", 27.0, 122.0)])
    assert route_duration_s(scenario.platforms[0]) == 0.0


# ── 最近距离 ────────────────────────────────────────────────────────


def test_min_separation_of_static_platforms() -> None:
    scenario = _scenario(
        [_platform("a", 27.5, 119.0, 9000.0), _platform("b", 27.5, 123.5, 0.0)]
    )
    separation = min_separation_m(scenario.platforms[0], scenario.platforms[1])
    assert separation / 1000 == pytest.approx(443.9, abs=1.0)


def test_min_separation_accounts_for_route_closing_the_gap() -> None:
    """一方沿航路飞向另一方时, 最近距离应当远小于初始距离。"""
    scenario = _scenario(
        [
            _platform(
                "flyer", 27.5, 119.0, 8000.0,
                mover="WSF_AIR_MOVER",
                route=[_waypoint(27.5, 123.4, 8000.0, 250.0)],
            ),
            _platform("target", 27.5, 123.5, 0.0),
        ]
    )
    initial = ground_distance_m(
        scenario.platforms[0].position, scenario.platforms[1].position
    )
    closest = min_separation_m(scenario.platforms[0], scenario.platforms[1])
    assert initial / 1000 == pytest.approx(443.8, abs=1.0)
    assert closest < 20_000


# ── 断言可达性 ──────────────────────────────────────────────────────


_RADAR = [{"sensor_type_id": "SEARCH_RADAR", "maximum_range_m": 150_000.0}]
_RADAR_MOUNT = [{"instance_name": "radar", "sensor_type_id": "SEARCH_RADAR"}]


def _detection_assertion(source: str, target: str, sensor: str | None = "radar") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "assertion_id": "detect",
        "event_type": "SENSOR_DETECTION_ATTEMPT",
        "source_platform": source,
        "target_platform": target,
    }
    if sensor:
        payload["sensor_name"] = sensor
    return payload


def test_assertion_out_of_range_is_reported() -> None:
    """实测里真实发生过的缺陷: 150 km 雷达要探测 440 km 外的目标。"""
    scenario = _scenario(
        [
            _platform("awacs", 27.5, 119.0, 9000.0, sensors=_RADAR_MOUNT),
            _platform("target", 27.5, 123.5, 0.0, side="red"),
        ],
        sensors=_RADAR,
        assertions=[_detection_assertion("awacs", "target")],
    )
    findings = check_geometry(scenario)
    kinds = {f.kind for f in findings}
    assert "assertion_out_of_range" in kinds
    message = next(f.message for f in findings if f.kind == "assertion_out_of_range")
    assert "443" in message or "444" in message
    assert "150" in message


def test_assertion_within_range_is_not_reported() -> None:
    """够得着就不要报 —— 误报会让修复循环去改本来正确的想定。"""
    scenario = _scenario(
        [
            _platform("awacs", 27.5, 122.0, 9000.0, sensors=_RADAR_MOUNT),
            _platform("target", 27.5, 122.5, 0.0, side="red"),
        ],
        sensors=_RADAR,
        assertions=[_detection_assertion("awacs", "target")],
    )
    assert [f for f in check_geometry(scenario) if f.kind == "assertion_out_of_range"] == []


def test_assertion_reachable_via_route_is_not_reported() -> None:
    """初始位置很远但航路会飞进探测圈, 不算不可达。"""
    scenario = _scenario(
        [
            _platform(
                "fighter", 27.5, 119.0, 8000.0,
                mover="WSF_AIR_MOVER",
                sensors=_RADAR_MOUNT,
                route=[_waypoint(27.5, 123.4, 8000.0, 250.0)],
            ),
            _platform("target", 27.5, 123.5, 0.0, side="red"),
        ],
        sensors=_RADAR,
        assertions=[_detection_assertion("fighter", "target")],
        duration_s=7200.0,
    )
    assert [f for f in check_geometry(scenario) if f.kind == "assertion_out_of_range"] == []


def test_assertion_just_inside_range_is_not_reported() -> None:
    """边界附近要偏向不报, 避免采样误差造成误判。"""
    scenario = _scenario(
        [
            _platform("awacs", 27.5, 122.0, 0.0, sensors=_RADAR_MOUNT),
            _platform("target", 28.2, 122.0, 0.0, side="red"),
        ],
        sensors=_RADAR,
        assertions=[_detection_assertion("awacs", "target")],
    )
    distance = ground_distance_m(
        scenario.platforms[0].position, scenario.platforms[1].position
    )
    assert distance < 150_000  # 确认确实在射程内
    assert [f for f in check_geometry(scenario) if f.kind == "assertion_out_of_range"] == []


def test_weapon_fired_assertion_skips_geometry() -> None:
    """WEAPON_FIRED 与传感器几何无关, 不该被判不可达。"""
    scenario = _scenario(
        [
            _platform("shooter", 27.5, 119.0, 8000.0, sensors=_RADAR_MOUNT),
            _platform("target", 27.5, 123.5, 0.0, side="red"),
        ],
        sensors=_RADAR,
        assertions=[
            {
                "assertion_id": "fired",
                "event_type": "WEAPON_FIRED",
                "source_platform": "shooter",
                "target_platform": "target",
            }
        ],
    )
    assert [f for f in check_geometry(scenario) if f.kind == "assertion_out_of_range"] == []


def test_assertion_without_sensor_is_left_to_plausibility_check() -> None:
    """平台没传感器由合理性检查报, 几何检查不重复报。"""
    scenario = _scenario(
        [
            _platform("awacs", 27.5, 119.0, 9000.0),
            _platform("target", 27.5, 123.5, 0.0, side="red"),
        ],
        sensors=_RADAR,
        assertions=[_detection_assertion("awacs", "target", sensor=None)],
    )
    assert [f for f in check_geometry(scenario) if f.kind == "assertion_out_of_range"] == []


# ── 航路与任务时长 ──────────────────────────────────────────────────


def test_route_exceeding_duration_is_reported() -> None:
    scenario = _scenario(
        [
            _platform(
                "slow", 27.0, 122.0, 8000.0,
                mover="WSF_AIR_MOVER",
                route=[_waypoint(31.0, 122.0, 8000.0, 100.0)],
            )
        ],
        duration_s=600.0,   # 10 分钟走 445 km 显然不够
    )
    kinds = {f.kind for f in check_geometry(scenario)}
    assert "route_exceeds_duration" in kinds


def test_route_within_duration_is_not_reported() -> None:
    scenario = _scenario(
        [
            _platform(
                "fast", 27.0, 122.0, 8000.0,
                mover="WSF_AIR_MOVER",
                route=[_waypoint(28.0, 122.0, 8000.0, 250.0)],
            )
        ],
        duration_s=3600.0,
    )
    kinds = {f.kind for f in check_geometry(scenario)}
    assert "route_exceeds_duration" not in kinds


def test_missing_route_speed_is_reported() -> None:
    scenario = _scenario(
        [
            _platform(
                "nospeed", 27.0, 122.0, 8000.0,
                mover="WSF_AIR_MOVER",
                route=[_waypoint(28.0, 122.0, 8000.0)],
            )
        ]
    )
    kinds = {f.kind for f in check_geometry(scenario)}
    assert "route_speed_missing" in kinds


# ── 交战的探测支撑 ──────────────────────────────────────────────────


def test_engagement_without_detection_is_reported() -> None:
    """射手方有传感器但全程够不到目标: 侦察-锁定-打击链断了。"""
    scenario = _scenario(
        [
            _platform(
                "shooter", 27.5, 119.0, 8000.0,
                sensors=_RADAR_MOUNT,
                weapons=[{"instance_name": "arm"}],
            ),
            _platform("target", 27.5, 123.5, 0.0, side="red"),
        ],
        sensors=_RADAR,
        engagements=[
            {
                "engagement_id": "strike",
                "shooter_platform": "shooter",
                "target_platform": "target",
                "weapon_name": "arm",
                "fire_time_s": 600.0,
            }
        ],
    )
    kinds = {f.kind for f in check_geometry(scenario)}
    assert "engagement_without_detection" in kinds


def test_engagement_with_detection_is_not_reported() -> None:
    scenario = _scenario(
        [
            _platform(
                "shooter", 27.5, 122.0, 8000.0,
                sensors=_RADAR_MOUNT,
                weapons=[{"instance_name": "arm"}],
            ),
            _platform("target", 27.5, 122.4, 0.0, side="red"),
        ],
        sensors=_RADAR,
        engagements=[
            {
                "engagement_id": "strike",
                "shooter_platform": "shooter",
                "target_platform": "target",
                "weapon_name": "arm",
                "fire_time_s": 600.0,
            }
        ],
    )
    kinds = {f.kind for f in check_geometry(scenario)}
    assert "engagement_without_detection" not in kinds


def test_space_domain_platform_is_skipped() -> None:
    """空间平台的位置是无物理意义的占位值, 不该产生无法修复的几何告警。

    轨道运动未建模已经由 capability_gaps 声明过; 再报一条"卫星离目标 12000 km"
    只会让修复循环白转 —— 它改不了, 因为根本没有 space mover 可用。
    """
    scenario = _scenario(
        [
            _platform("recon_sat", 0.0, 0.0, 0.0, sensors=_RADAR_MOUNT),
            _platform("ground_target", 27.5, 122.5, 0.0, side="red"),
        ],
        sensors=_RADAR,
        assertions=[_detection_assertion("recon_sat", "ground_target")],
        domains={"recon_sat": "space", "ground_target": "land"},
    )
    assert [f for f in check_geometry(scenario) if f.kind == "assertion_out_of_range"] == []


def test_non_space_platform_at_same_geometry_is_still_reported() -> None:
    """同样的几何, 只要不是空间域就照常报 —— 确认上一条不是无脑放过。"""
    scenario = _scenario(
        [
            _platform("awacs", 0.0, 0.0, 9000.0, sensors=_RADAR_MOUNT),
            _platform("ground_target", 27.5, 122.5, 0.0, side="red"),
        ],
        sensors=_RADAR,
        assertions=[_detection_assertion("awacs", "ground_target")],
        domains={"awacs": "air", "ground_target": "land"},
    )
    kinds = {f.kind for f in check_geometry(scenario)}
    assert "assertion_out_of_range" in kinds


def test_no_findings_for_a_sane_scenario() -> None:
    """一份几何上说得通的想定不该产生任何 finding。"""
    scenario = _scenario(
        [
            _platform(
                "awacs", 27.5, 122.0, 9000.0,
                mover="WSF_AIR_MOVER",
                sensors=_RADAR_MOUNT,
                route=[_waypoint(27.6, 122.1, 9000.0, 200.0)],
            ),
            _platform(
                "intruder", 27.5, 122.6, 8000.0, side="red",
                mover="WSF_AIR_MOVER",
                route=[_waypoint(27.5, 122.2, 8000.0, 250.0)],
            ),
        ],
        sensors=_RADAR,
        assertions=[_detection_assertion("awacs", "intruder")],
        duration_s=3600.0,
    )
    assert check_geometry(scenario) == []
