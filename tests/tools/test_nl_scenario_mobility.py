"""平台机动性一致性检查的测试。

抓的是这类错误: 需求说"一部地面雷达站", 生成出来的雷达站却挂着 WSF_GROUND_MOVER
带着航路 —— schema 合法 (有 mover 就有 route)、语法全绿、评分 100, 但雷达站会跑。
反方向同样要抓: 需求说"两架歼-20巡逻", 结果飞机全是静止平台。

这条检查是基于 type_hint 自由文本的启发式, 所以误报防护和正例一样重要: 分类不出来
就返回 unknown, 什么都不报。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from nanobot.afsim import ScenarioIR
from nanobot.agent.tools.nl_scenario_planner import (
    ScenarioIRPlanner,
    _infer_platform_type,
    check_ir_plausibility,
    classify_mobility,
    normalize_ir_payload,
)
from tools.test_nl_scenario_planner import ScriptedLLM


# ── 分类器本身 ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "type_hint",
    [
        "地面雷达站",
        "远程警戒雷达阵地",
        "S-400 火控雷达阵地",
        "岛礁机场",
        "空军基地",
        "ground radar site",
        "enemy airfield",
        "missile silo",
    ],
)
def test_fixed_installations_are_classified_fixed(type_hint: str) -> None:
    assert classify_mobility(type_hint) == "fixed"


@pytest.mark.parametrize(
    "type_hint",
    [
        "歼-20 战斗机",
        "轰-6K 轰炸机",
        "KJ-500 预警机",
        "J-16D 电子战机",
        "052D 驱逐舰",
        "红方巡洋舰",
        "主战坦克",
        "察打一体无人机",
        "F-16 fighter",
        "Arleigh Burke destroyer",
        "armoured vehicle",
    ],
)
def test_mobile_assets_are_classified_mobile(type_hint: str) -> None:
    assert classify_mobility(type_hint) == "mobile"


@pytest.mark.parametrize(
    "type_hint",
    ["未知目标", "target", "", "蓝方单位", "high value asset"],
)
def test_ambiguous_hints_are_unknown(type_hint: str) -> None:
    """看不出来就别下结论 —— 误报会让修复循环去改本来正确的想定。"""
    assert classify_mobility(type_hint) == "unknown"


def test_mobility_modifier_overrides_fixed_reading() -> None:
    """"移动式红旗-22"是能跑的, 不能因为含"阵地"就判成固定设施。"""
    assert classify_mobility("移动式红旗-22 发射阵地") != "fixed"
    assert classify_mobility("mobile radar site") != "fixed"


def test_airfield_is_not_confused_with_aircraft() -> None:
    """"机场"含"机"字, 不能被机动平台规则误捕。"""
    assert classify_mobility("岛礁机场") == "fixed"


def test_self_propelled_vehicle_is_mobile() -> None:
    assert classify_mobility("自行防空导弹发射车") == "mobile"


# ── 接入合理性检查 ──────────────────────────────────────────────────


def _scenario(entries: list[tuple[str, str, str, bool]]) -> ScenarioIR:
    """entries: (platform_id, type_hint, domain, is_mobile)。"""
    mission_platforms = []
    platforms = []
    for index, (platform_id, type_hint, domain, is_mobile) in enumerate(entries):
        position = {
            "latitude_deg": 27.5 + index * 0.2,
            "longitude_deg": 122.0 + index * 0.2,
            "altitude_m": 0.0 if domain in ("sea", "land") else 8000.0,
        }
        mission_platforms.append(
            {
                "platform_id": platform_id,
                "side": "blue",
                "domain": domain,
                "type_hint": type_hint,
                "initial_position": position,
            }
        )
        platform: dict[str, Any] = {
            "platform_id": platform_id,
            "side": "blue",
            "position": position,
        }
        if is_mobile:
            platform["mover_type"] = {
                "air": "WSF_AIR_MOVER",
                "sea": "WSF_SURFACE_MOVER",
                "land": "WSF_GROUND_MOVER",
            }[domain]
            platform["route"] = [
                {
                    "position": {
                        "latitude_deg": position["latitude_deg"] + 0.1,
                        "longitude_deg": position["longitude_deg"] + 0.1,
                        "altitude_m": position["altitude_m"],
                    },
                    "speed_mps": 200.0,
                }
            ]
        platforms.append(platform)

    return ScenarioIR.model_validate(
        {
            "scenario_id": "mobility_probe",
            "mission": {
                "mission_id": "mobility",
                "name": "机动性探针",
                "objective": "验证机动性一致性",
                "duration_s": 7200.0,
                "platforms": mission_platforms,
            },
            "selected_coa": {
                "coa_id": "coa", "mission_id": "mobility", "summary": "probe"
            },
            "sensors": [
                {"sensor_type_id": "SEARCH_RADAR", "maximum_range_m": 300000.0}
            ],
            "platforms": platforms,
        }
    )


def test_fixed_installation_with_mover_is_reported() -> None:
    """实测里真实发生过: 地面雷达站被赋了 WSF_GROUND_MOVER。"""
    scenario = _scenario([("blue_ground_radar", "地面雷达站", "land", True)])
    issues = check_ir_plausibility(scenario)
    assert any("固定设施" in issue and "blue_ground_radar" in issue for issue in issues)


def test_fixed_installation_without_mover_is_clean() -> None:
    scenario = _scenario([("blue_ground_radar", "地面雷达站", "land", False)])
    assert not [i for i in check_ir_plausibility(scenario) if "固定设施" in i]


def test_mobile_asset_without_mover_is_reported() -> None:
    """需求说巡逻的战斗机, 却建成静止平台 —— 之前 datalink 用例整体变静止的病根。"""
    scenario = _scenario([("blue_j20_1", "歼-20 战斗机", "air", False)])
    issues = check_ir_plausibility(scenario)
    assert any("机动装备" in issue and "blue_j20_1" in issue for issue in issues)


def test_mobile_asset_with_mover_is_clean() -> None:
    scenario = _scenario([("blue_j20_1", "歼-20 战斗机", "air", True)])
    assert not [i for i in check_ir_plausibility(scenario) if "机动装备" in i]


def test_unknown_hint_never_triggers_either_direction() -> None:
    for is_mobile in (True, False):
        scenario = _scenario([("blue_unit", "蓝方单位", "air", is_mobile)])
        issues = check_ir_plausibility(scenario)
        assert not [i for i in issues if "固定设施" in i or "机动装备" in i]


def test_mixed_scenario_flags_only_the_wrong_platform() -> None:
    scenario = _scenario(
        [
            ("blue_j20_1", "歼-20 战斗机", "air", True),          # 正确: 会动
            ("blue_radar", "地面雷达站", "land", True),            # 错误: 站会跑
            ("blue_ddg", "052D 驱逐舰", "sea", True),             # 正确: 会动
        ]
    )
    flagged = [i for i in check_ir_plausibility(scenario) if "固定设施" in i or "机动装备" in i]
    assert len(flagged) == 1
    assert "blue_radar" in flagged[0]


# ── 固定设施带 mover 走确定性归一化 ─────────────────────────────────
#
# 这条问题答案唯一 (平台身份决定它不动), 反复回灌给模型只是浪费轮次 —— 实测中
# "岸基警戒雷达阵地"带着 WSF_GROUND_MOVER 连续两个 refine 轮都没被改掉。


def _payload(entries: list[tuple[str, str, str, bool]]) -> dict[str, Any]:
    """把 _scenario 的入参转成未经校验的原始 payload (归一化跑在校验之前)。"""
    return json.loads(_scenario(entries).model_dump_json())


def test_normalization_drops_mover_from_fixed_installation() -> None:
    payload = _payload([("red_coastal_radar", "岸基警戒雷达阵地", "land", True)])
    assert payload["platforms"][0]["mover_type"] == "WSF_GROUND_MOVER"

    normalized, notes = normalize_ir_payload(payload)
    assert "mover_type" not in normalized["platforms"][0]
    assert normalized["platforms"][0]["route"] == []
    assert any("固定设施" in note and "red_coastal_radar" in note for note in notes)


def test_normalization_keeps_mover_on_mobile_asset() -> None:
    """歼-20 是机动平台, 不该删 mover。但 platform_type 默认值会被推断为 J20。"""
    payload = _payload([("blue_j20_1", "歼-20 战斗机", "air", True)])
    normalized, notes = normalize_ir_payload(payload)
    # mover 保留
    assert normalized["platforms"][0]["mover_type"] == "WSF_AIR_MOVER"
    # platform_type 从默认值 WSF_PLATFORM 推断为 J20 (这是这一轮新加的归一化)
    assert normalized["platforms"][0]["platform_type"] == "J20"
    assert any("J20" in note for note in notes)


def test_normalization_keeps_mover_when_hint_is_ambiguous() -> None:
    """认不出身份就别动 —— 启发式分类不该越权改 IR。"""
    payload = _payload([("blue_unit", "蓝方单位", "air", True)])
    normalized, notes = normalize_ir_payload(payload)
    assert normalized["platforms"][0]["mover_type"] == "WSF_AIR_MOVER"
    assert notes == []


def test_normalization_respects_mobility_modifier() -> None:
    """"移动式发射阵地"是能跑的, 不该被当成固定设施删掉 mover。"""
    payload = _payload([("red_tel_1", "移动式红旗-22发射阵地", "land", True)])
    normalized, notes = normalize_ir_payload(payload)
    assert normalized["platforms"][0]["mover_type"] == "WSF_GROUND_MOVER"
    assert notes == []


# ── platform_type 从 type_hint 推断 ─────────────────────────────────
#
# LLM 抽 IR 时 9 个平台全填了 WSF_PLATFORM (schema 默认值), 导致多文件项目里
# 所有平台挤进同一个 WSF_PLATFORM.txt。归一化层按 type_hint 推机型缩写, 比让
# 模型碰运气可靠。


def test_infer_platform_type_common_chinese_names() -> None:
    assert _infer_platform_type("轰-6K轰炸机") == "H6K"
    assert _infer_platform_type("轰6甲") == "H6K"           # 无连字符也认得出
    assert _infer_platform_type("空警-500 预警机") == "KJ500"
    assert _infer_platform_type("054A护卫舰") == "TYPE_054A"
    assert _infer_platform_type("红方巡洋舰") == "CRUISER"
    assert _infer_platform_type("岸基警戒雷达阵地") == "RADAR_SITE"


def test_infer_platform_type_us_platforms() -> None:
    assert _infer_platform_type("F-35A 战斗机") == "F35"
    assert _infer_platform_type("B-2 隐身轰炸机") == "B2"
    assert _infer_platform_type("阿利·伯克级驱逐舰") == "DDG_51"


def test_infer_platform_type_unknown_returns_none() -> None:
    """认不出就返回 None, 调用方保留原值。"""
    assert _infer_platform_type("") is None
    assert _infer_platform_type("某个不知名的平台") is None
    assert _infer_platform_type("蓝方单位") is None


def test_normalization_infers_platform_type_from_hint() -> None:
    payload = _payload([("blue_h6k_1", "轰-6K轰炸机", "air", True)])
    # 模拟 LLM 填了默认值
    payload["platforms"][0]["platform_type"] = "WSF_PLATFORM"
    normalized, notes = normalize_ir_payload(payload)
    assert normalized["platforms"][0]["platform_type"] == "H6K"
    assert any("H6K" in note and "WSF_PLATFORM" in note for note in notes)


def test_normalization_keeps_explicit_platform_type() -> None:
    """LLM 已经填了具体类型 (非默认值), 归一化不该动它。"""
    payload = _payload([("blue_h6k_1", "轰-6K轰炸机", "air", True)])
    payload["platforms"][0]["platform_type"] = "CUSTOM_H6K_VARIANT"
    normalized, notes = normalize_ir_payload(payload)
    assert normalized["platforms"][0]["platform_type"] == "CUSTOM_H6K_VARIANT"
    assert notes == []


def test_fixed_installation_never_survives_into_advisories() -> None:
    """归一化之后, "固定设施带 mover" 不该再出现在最终 advisories 里。

    这个 fixture 没配传感器, 所以还会有别的合理性告警驱动 refine 轮 —— 这里只断言
    固定设施那一条被彻底消掉了, 而且是靠归一化 (normalizations) 而不是靠模型碰运气。
    """
    broken = _payload([
        ("blue_awacs_1", "空警-500 预警机", "air", True),
        ("red_coastal_radar", "岸基警戒雷达阵地", "land", True),
    ])
    # 给 fixture 补一个传感器, 否则 schema 会报 "platforms reference unknown sensor types"
    broken["sensors"] = [{
        "sensor_type_id": "GENERIC",
        "base_type": "WSF_GEOMETRIC_SENSOR",
        "mode_name": "search",
        "frame_time_s": 1.0,
        "maximum_range_m": 100_000.0,
        "reports_location": True,
        "ignore_same_side": True,
    }]
    for p in broken["platforms"]:
        p["sensors"] = [{
            "instance_name": "radar",
            "sensor_type_id": "GENERIC",
            "initial_mode": "search",
            "enabled": True,
        }]
    payload_json = json.dumps(broken, ensure_ascii=False)
    llm = ScriptedLLM([payload_json] * 3)
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=3).plan("预警机警戒, 岸基雷达阵地固定部署")

    assert result.success
    assert not [a for a in result.advisories if "固定设施" in a]
    assert any("固定设施" in note for note in result.normalizations)
    # 每一轮都该被归一化掉, 不存在"某轮漏掉"
    for attempt in result.attempts:
        assert not [a for a in attempt.advisories if "固定设施" in a]
