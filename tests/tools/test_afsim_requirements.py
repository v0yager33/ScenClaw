"""需求数值条款的解析与核验测试。

条款由模型从需求原文里抽出, 核验全部由代码完成。这里既测"不满足的条款报得出、
报文里带着修法所需的数字", 也测"模型抽错的条款被丢弃而不是拿去误报"。
"""

from __future__ import annotations

from typing import Any

import pytest

from nanobot.afsim import ScenarioIR
from nanobot.afsim.requirements import (
    RequirementCheck,
    describe_requirement_check,
    evaluate_requirement_checks,
    parse_requirement_checks,
)


def _plane(
    platform_id: str,
    side: str,
    lat: float,
    lon: float,
    *,
    route: list[tuple[float, float]] | None = None,
    alt: float = 8000.0,
    speed: float = 250.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "platform_id": platform_id,
        "side": side,
        "position": {"latitude_deg": lat, "longitude_deg": lon, "altitude_m": alt},
        "weapons": [{"instance_name": "aam", "quantity": 4}],
    }
    if route:
        payload["mover_type"] = "WSF_AIR_MOVER"
        payload["route"] = [
            {
                "position": {"latitude_deg": p_lat, "longitude_deg": p_lon, "altitude_m": alt},
                "speed_mps": speed,
            }
            for p_lat, p_lon in route
        ]
    return payload


def _shot(engagement_id: str, shooter: str, target: str, time_s: float) -> dict[str, Any]:
    return {
        "engagement_id": engagement_id,
        "shooter_platform": shooter,
        "target_platform": target,
        "weapon_name": "aam",
        "fire_time_s": time_s,
    }


def _scenario(engagements: list[dict[str, Any]] | None = None) -> ScenarioIR:
    """两队各两架, 蓝方向东、红方向西相向飞行; 初始双方相距约 200 km, 同队间隔约 5.6 km。

    蓝机 250 m/s 飞 1 个经度 (约 99 km, 约 396 s 飞完), 红机飞 0.8 个经度 (约 317 s 飞完)。
    """
    platforms = [
        _plane("blue_1", "blue", 27.0, 121.0, route=[(27.0, 122.0)]),
        _plane("blue_2", "blue", 27.05, 121.0, route=[(27.05, 122.0)]),
        _plane("red_1", "red", 27.0, 123.0, route=[(27.0, 122.2)]),
        _plane("red_2", "red", 27.05, 123.0, route=[(27.05, 122.2)]),
    ]
    return ScenarioIR.model_validate({
        "scenario_id": "req_probe",
        "mission": {
            "mission_id": "req",
            "name": "条款探针",
            "objective": "验证需求条款核验",
            "duration_s": 1800.0,
            "platforms": [
                {
                    "platform_id": p["platform_id"],
                    "side": p["side"],
                    "domain": "air",
                    "type_hint": "战斗机",
                    "initial_position": p["position"],
                }
                for p in platforms
            ],
        },
        "selected_coa": {"coa_id": "coa", "mission_id": "req", "summary": "相向接敌"},
        "platforms": platforms,
        "engagements": engagements or [],
    })


def _check(kind: str, quote: str = "", **params: Any) -> RequirementCheck:
    return RequirementCheck(kind=kind, params=params, quote=quote)


def _issues(check: RequirementCheck, scenario: ScenarioIR | None = None, destroyed=None) -> list[str]:
    return evaluate_requirement_checks(scenario or _scenario(), [check], destroyed)


# ── 解析 ────────────────────────────────────────────────────────────


def test_parse_coerces_numbers_skips_nulls_and_trims_quotes() -> None:
    checks, notes = parse_requirement_checks({"checks": [
        {"kind": "side_separation", "min_km": "200", "max_km": 280, "quote": "长" * 60},
        {"kind": "fire_window", "shooters": "blue_*", "start_s": 100, "end_s": None},
    ]})

    assert notes == []
    assert checks[0].params == {"min_km": 200.0, "max_km": 280.0}
    assert len(checks[0].quote) == 40
    # 单个选择器收成列表; 模型写成 null 的字段视为没给
    assert checks[1].params == {"shooters": ["blue_*"], "start_s": 100.0}


def test_parse_accepts_a_bare_list_and_sorts_windows() -> None:
    checks, notes = parse_requirement_checks([
        {"kind": "fire_phases", "windows": [[600, 900], [0, 300]], "min_count": 2},
    ])
    assert notes == []
    assert checks[0].params["windows"] == [[0.0, 300.0], [600.0, 900.0]]


def test_parse_without_a_checks_list_is_noted() -> None:
    assert parse_requirement_checks({"items": []}) == ([], ["条款抽取结果里没有 checks 列表"])


@pytest.mark.parametrize(("item", "message"), [
    ("不是对象", "不是对象"),
    ({"kind": "teleport", "min_km": 1}, "未知类型"),
    ({"kind": "fire_window", "start_s": 100}, "缺少字段 shooters"),
    ({"kind": "side_separation"}, "至少要给"),
    ({"kind": "side_separation", "min_km": 300, "max_km": 200}, "min_km 大于 max_km"),
    ({"kind": "fire_phases", "windows": [[0, 300]], "min_count": 2}, "min_count 超过时段个数"),
    ({"kind": "fire_phases", "windows": [[300, 0]], "min_count": 1}, "取值不合法"),
    ({"kind": "fire_phases", "windows": [[0, 300]], "min_count": 1.5}, "取值不合法"),
    ({"kind": "reach_area", "platforms": ["blue_1"], "lon_min": 122, "alive": "yes"}, "取值不合法"),
    ({"kind": "route_points", "platforms": ["blue_1"]}, "缺少字段 min_count"),
])
def test_invalid_checks_are_dropped_with_a_note(item: Any, message: str) -> None:
    checks, notes = parse_requirement_checks({"checks": [item]})
    assert checks == []
    assert notes[0].startswith("丢弃第 1 条条款") and message in notes[0]


def test_unmatched_selectors_are_ignored_or_drop_the_check() -> None:
    checks, notes = parse_requirement_checks({"checks": [
        {"kind": "altitude_range", "platforms": ["blue_*", "ghost"], "max_m": 9000},
        {"kind": "speed_range", "platforms": ["ghost_*"], "max_mps": 300},
        {"kind": "formation_spacing", "groups": [["side:red"], ["ghost"]], "max_km": 10},
    ]}, _scenario())

    assert [c.kind for c in checks] == ["altitude_range", "formation_spacing"]
    assert checks[0].params["platforms"] == ["blue_*"]
    assert checks[1].params["groups"] == [["side:red"]]
    assert notes == [
        "第 1 条条款: 选择器 ghost 匹配不到平台, 已忽略",
        "丢弃第 2 条条款: speed_range.platforms 匹配不到任何平台: ghost_*",
        "第 3 条条款: 选择器 ghost 匹配不到平台, 已忽略",
    ]


def test_describe_and_serialise_a_check() -> None:
    check = _check("fire_window", "蓝方 900 秒后打击", shooters=["blue_*"], start_s=900.0)
    assert describe_requirement_check(check) == (
        '「蓝方 900 秒后打击」 → fire_window {"shooters": ["blue_*"], "start_s": 900.0}'
    )
    assert check.to_dict() == {
        "kind": "fire_window", "shooters": ["blue_*"], "start_s": 900.0, "quote": "蓝方 900 秒后打击",
    }
    assert "quote" not in _check("side_separation", min_km=1.0).to_dict()


# ── 核验 ────────────────────────────────────────────────────────────


def test_messages_are_prefixed_with_the_quote_or_the_kind() -> None:
    quoted = _issues(_check("altitude_range", "高度不超过 5000 米", platforms=["blue_1"], max_m=5000.0))
    bare = _issues(_check("altitude_range", platforms=["blue_1"], max_m=5000.0))
    assert quoted[0].startswith("需求「高度不超过 5000 米」未满足: blue_1 的高度")
    assert bare[0].startswith("需求条款 altitude_range 未满足: ")


def test_side_separation_suggests_the_direction_to_move() -> None:
    too_far = _issues(_check("side_separation", min_km=100.0, max_km=150.0))
    too_close = _issues(_check("side_separation", min_km=250.0, max_km=300.0))

    assert len(too_far) == 1 and "相向移近约" in too_far[0]
    assert len(too_close) == 1 and "再拉开约" in too_close[0]
    assert "航路随初始位置一起平移" in too_far[0]
    assert _issues(_check("side_separation", min_km=150.0, max_km=250.0)) == []


def test_formation_spacing_reports_each_platform_in_the_group() -> None:
    by_group = _issues(_check("formation_spacing", groups=[["blue_*"]], min_km=8.0, max_km=15.0))
    by_side = _issues(_check("formation_spacing", min_km=8.0, max_km=15.0))

    assert len(by_group) == 2
    assert "blue_1 与同队最近的 blue_2" in by_group[0]
    assert "blue_2 与同队最近的 blue_1" in by_group[1]
    assert len(by_side) == 4  # 不给 groups 时按阵营分组
    assert _issues(_check("formation_spacing", min_km=3.0, max_km=8.0)) == []


def test_fire_window_reports_each_shot_outside_the_window() -> None:
    scenario = _scenario([_shot("early", "blue_1", "red_1", 200.0), _shot("late", "blue_2", "red_2", 500.0)])
    issues = _issues(_check("fire_window", shooters=["blue_*"], start_s=300.0, end_s=600.0), scenario)

    assert len(issues) == 1
    assert "交战 early (blue_1) 在 200s 开火" in issues[0] and "300-600s" in issues[0]


def test_fire_window_without_any_shot_is_reported() -> None:
    issues = _issues(_check("fire_window", shooters=["side:red"], start_s=300.0))
    assert len(issues) == 1 and "没有任何射击计划" in issues[0]


def test_fire_phases_ignore_shots_from_destroyed_shooters() -> None:
    scenario = _scenario([
        _shot("kill_b1", "red_1", "blue_1", 100.0),
        _shot("first", "blue_2", "red_1", 150.0),
        _shot("never", "blue_1", "red_2", 300.0),
    ])
    check = _check("fire_phases", windows=[[0.0, 200.0], [200.0, 400.0]], min_count=2, shooters=["side:blue"])

    assert _issues(check, scenario) == []
    issues = _issues(check, scenario, {"blue_1": (100.0, "kill_b1")})
    assert len(issues) == 1
    assert "当前只有 1 个" in issues[0] and "没有开火: 200-400s" in issues[0]


def test_fire_phases_last_window_includes_its_end() -> None:
    scenario = _scenario([_shot("edge", "blue_1", "red_1", 400.0)])
    check = _check("fire_phases", windows=[[0.0, 200.0], [200.0, 400.0]], min_count=1)
    assert _issues(check, scenario) == []


def test_engagement_range_lists_the_spans_inside_the_envelope() -> None:
    """相向飞行约 200 s 后进入 100 km; 红机约 317 s 飞完航路, 之后的位置不推算。"""
    scenario = _scenario([_shot("long", "blue_1", "red_1", 0.0)])
    issues = _issues(_check("engagement_range", max_km=100.0), scenario)

    assert len(issues) == 1
    assert "交战 long 在 0s 开火时 blue_1 与 red_1 斜距约" in issues[0]
    assert "包络内的时段是 " in issues[0]
    span = issues[0].split("包络内的时段是 ")[1].split("s;")[0]
    start, end = (float(value) for value in span.split("-"))
    assert 190 <= start <= 210 and 305 <= end <= 320
    assert "离端点留 15s 以上余量" in issues[0]


def test_engagement_range_without_any_span_says_so() -> None:
    scenario = _scenario([_shot("long", "blue_1", "red_1", 0.0)])
    issues = _issues(_check("engagement_range", max_km=20.0), scenario)
    assert len(issues) == 1 and "之前两者都不在包络内" in issues[0]


def test_engagement_range_reports_shots_after_the_route_ends() -> None:
    """rerun 实测: 红方航路 624 s 就飞完, 1300 s 的射击被跳过, 外部按终点位置量出 136 km。"""
    scenario = _scenario([_shot("after_route", "blue_1", "red_1", 500.0)])
    issues = _issues(_check("engagement_range", max_km=100.0), scenario)

    assert len(issues) == 1
    assert (
        "交战 after_route 在 500s 开火时航路已经飞完 (blue_1 的航路 396s 飞完、red_1 的航路 317s 飞完)"
        in issues[0]
    )
    assert "请续上航点让航路覆盖到开火时刻" in issues[0]
    assert "包络内的时段是 " in issues[0]


def test_engagement_range_skips_legs_without_speed() -> None:
    """航段缺速度推算不出位置, 由航路检查负责报, 这里不重复。"""
    scenario = _scenario([_shot("no_speed", "blue_1", "red_1", 100.0)])
    for platform in scenario.platforms:
        for point in platform.route:
            point.speed_mps = None
    assert _issues(_check("engagement_range", max_km=100.0), scenario) == []


def test_engagement_range_checks_shots_that_never_happen() -> None:
    """rerun 实测: 对方射手被提前打掉, 那几发计划停在 100 km 外; 击毁没生效就会在包络外开火。"""
    scenario = _scenario([_shot("dead_shooter", "blue_2", "red_2", 50.0)])
    check = _check("engagement_range", max_km=100.0)
    issues = _issues(check, scenario, {"blue_2": (10.0, "other")})

    assert len(issues) == 1
    assert "按计划 blue_2 在 10s 已被交战 other 摧毁" in issues[0]
    assert "包络内的时段是 " in issues[0]  # 按双方都在推算, 不截到 10s
    assert "挪到" not in issues[0]  # 包络时段都在 10s 之后, 挪过去仍打不出


def test_engagement_range_says_when_moving_earlier_makes_the_shot_real() -> None:
    """包络时段落在击毁之前时, 挪进去这发就真的会打出去。"""
    # 两机约 196s 进入 100 km、约 276s 近于 60 km; 290s 已贴得太近
    scenario = _scenario([_shot("too_close", "blue_2", "red_2", 290.0)])
    check = _check("engagement_range", min_km=60.0, max_km=100.0)
    issues = _issues(check, scenario, {"blue_2": (280.0, "other")})

    assert len(issues) == 1
    assert "按计划 blue_2 在 280s 已被交战 other 摧毁" in issues[0]
    assert "挪到 280s 之前会让这发真的打出去" in issues[0]


@pytest.mark.parametrize(("params", "destroyed", "expected"), [
    ({"start_s": 300.0, "end_s": 400.0}, None, ["首次进入要求区域", "平均速度需在"]),
    ({"start_s": 100.0, "end_s": 300.0}, {"blue_1": (100.0, "kill")}, ["被交战 kill 摧毁"]),
    ({"start_s": 100.0, "end_s": 300.0, "alive": False}, {"blue_1": (100.0, "kill")}, []),
    ({"start_s": 100.0, "end_s": 300.0}, None, []),
])
def test_reach_area_checks_entry_time_and_survival(
    params: dict[str, Any], destroyed: dict | None, expected: list[str]
) -> None:
    """blue_1 约 197 s 越过 121.5°E。"""
    issues = _issues(_check("reach_area", platforms=["blue_1"], lon_min=121.5, **params), destroyed=destroyed)
    if not expected:
        assert issues == []
        return
    assert len(issues) == 1
    assert all(text in issues[0] for text in expected)


def test_reach_area_unreachable_reports_the_flown_extent() -> None:
    issues = _issues(_check("reach_area", platforms=["blue_*"], lon_min=122.5))
    assert len(issues) == 2
    assert "进不了要求区域 (经度 ≥122.5°E)" in issues[0] and "飞到的范围是" in issues[0]


def test_reach_area_suggests_an_entry_target_away_from_the_window_edges() -> None:
    """rerun 实测: 只给速度区间时进区时刻在 897/898 s 和要求的 900 s 之间来回拉扯。"""
    early = _issues(_check("reach_area", platforms=["blue_1"], lon_min=121.5, start_s=300.0, end_s=400.0))
    late = _issues(_check("reach_area", platforms=["blue_1"], lon_min=121.5, end_s=150.0))

    assert len(early) == 1 and "进区前航程改到约 81 km 可在 325s 左右进区" in early[0]
    assert len(late) == 1 and "进区前航程改到约 23 km 可在 90s 左右进区" in late[0]


def test_altitude_and_speed_limits_are_checked_per_point() -> None:
    altitude = _issues(_check("altitude_range", platforms=["side:red"], max_m=5000.0))
    speed = _issues(_check("speed_range", platforms=["blue_1"], max_mps=200.0))

    assert len(altitude) == 2 and "red_1 的高度 初始位置 8000m、航点1 8000m" in altitude[0]
    assert len(speed) == 1 and "航点1 250m/s" in speed[0]


def test_route_points_reports_the_shortfall_and_the_time_budget() -> None:
    assert _issues(_check("route_points", platforms=["side:blue"], min_count=1)) == []

    issues = _issues(_check("route_points", platforms=["side:blue"], min_count=4))
    assert len(issues) == 2
    assert "blue_1 只有 1 个航点, 要求至少 4 个; 当前航路预计 396s 飞完 (任务 1800s)" in issues[0]


def test_platforms_missing_from_the_scenario_are_reported() -> None:
    """条款在上一版 IR 上绑定过选择器, 修订后平台被删时要报出来, 而不是悄悄通过。"""
    issues = _issues(_check("altitude_range", platforms=["ghost", "blue_1"], max_m=5000.0))
    assert "条款涉及的 ghost 在当前想定里找不到对应平台" in issues[0]
    assert "blue_1 的高度" in issues[1]
