"""通用性相关行为的离线测试: 域一致性、能力缺口声明、范围闸门。

这些是"让 agent 对任意自然语言需求都可信"的关键约束, 全部用假 LLM 验证, 不需要
网络。核心主张只有一条: **要么如实建模, 要么如实说做不到, 不许静默降级。**
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from nanobot.afsim import DEFAULT_CAPABILITY_REGISTRY, CapabilityKind, ScenarioIR
from nanobot.agent.tools.nl_scenario_planner import (
    ScenarioIRPlanner,
    detect_capability_gaps,
    looks_out_of_scope,
    normalize_ir_payload,
)
from tools.test_nl_scenario_planner import ScriptedLLM, _dumps, _valid_ir


# ── 域 ↔ mover 一致性 ───────────────────────────────────────────────


def _sea_ir() -> dict[str, Any]:
    """一艘在海上航行的驱逐舰。

    任务时长要给够: 18 节 (约 9 m/s) 走完 106 km 航段需要接近 3.3 小时, 时长给
    1 小时会被几何可达性检查判为"后段航路走不到"。
    """
    return {
        "scenario_id": "sea_probe",
        "mission": {
            "mission_id": "sea_probe",
            "name": "海上巡逻",
            "objective": "驱逐舰巡逻并搜索空中目标",
            "duration_s": 14400.0,
            "platforms": [
                {
                    "platform_id": "blue_ddg",
                    "side": "blue",
                    "domain": "sea",
                    "type_hint": "052D 驱逐舰",
                    "initial_position": {
                        "latitude_deg": 18.0,
                        "longitude_deg": 113.0,
                        "altitude_m": 0.0,
                    },
                }
            ],
        },
        "selected_coa": {
            "coa_id": "coa_1",
            "mission_id": "sea_probe",
            "summary": "驱逐舰由西向东巡逻",
        },
        "sensors": [{"sensor_type_id": "SHIP_RADAR", "maximum_range_m": 200000.0}],
        "platforms": [
            {
                "platform_id": "blue_ddg",
                "side": "blue",
                "position": {
                    "latitude_deg": 18.0,
                    "longitude_deg": 113.0,
                    "altitude_m": 0.0,
                },
                "mover_type": "WSF_SURFACE_MOVER",
                "sensors": [
                    {"instance_name": "radar", "sensor_type_id": "SHIP_RADAR"}
                ],
                "route": [
                    {
                        "position": {
                            "latitude_deg": 18.0,
                            "longitude_deg": 114.0,
                            "altitude_m": 0.0,
                        },
                        "speed_mps": 9.0,
                    }
                ],
            }
        ],
    }


def test_sea_platform_accepts_surface_mover() -> None:
    scenario = ScenarioIR.model_validate(_sea_ir())
    assert scenario.platforms[0].mover_type == "WSF_SURFACE_MOVER"


def test_sea_platform_rejects_air_mover() -> None:
    """给船套空中 mover 会让船"飞", 必须被 schema 拦住。"""
    ir = _sea_ir()
    ir["platforms"][0]["mover_type"] = "WSF_AIR_MOVER"
    with pytest.raises(ValidationError) as excinfo:
        ScenarioIR.model_validate(ir)
    assert "sea" in str(excinfo.value)


def test_sea_platform_rejects_airborne_altitude() -> None:
    ir = _sea_ir()
    ir["platforms"][0]["position"]["altitude_m"] = 8000.0
    ir["platforms"][0]["route"][0]["position"]["altitude_m"] = 8000.0
    with pytest.raises(ValidationError) as excinfo:
        ScenarioIR.model_validate(ir)
    assert "sea level" in str(excinfo.value)


def test_land_platform_requires_ground_mover() -> None:
    ir = _sea_ir()
    ir["mission"]["platforms"][0]["domain"] = "land"
    ir["platforms"][0]["mover_type"] = "WSF_AIR_MOVER"
    with pytest.raises(ValidationError) as excinfo:
        ScenarioIR.model_validate(ir)
    assert "WSF_GROUND_MOVER" in str(excinfo.value)


def test_space_platform_cannot_move() -> None:
    """空间域没有放行的 mover: WSF_SPACE_MOVER 用开普勒根数, 与航点模型不同构。"""
    ir = _sea_ir()
    ir["mission"]["platforms"][0]["domain"] = "space"
    with pytest.raises(ValidationError) as excinfo:
        ScenarioIR.model_validate(ir)
    assert "Keplerian" in str(excinfo.value)


def test_space_platform_allowed_as_stationary() -> None:
    ir = _sea_ir()
    ir["mission"]["platforms"][0]["domain"] = "space"
    ir["platforms"][0].pop("mover_type")
    ir["platforms"][0]["route"] = []
    scenario = ScenarioIR.model_validate(ir)
    assert scenario.platforms[0].mover_type is None


def test_domain_mover_error_is_repaired_through_the_loop() -> None:
    broken = _sea_ir()
    broken["platforms"][0]["mover_type"] = "WSF_AIR_MOVER"

    llm = ScriptedLLM([_dumps(broken), _dumps(_sea_ir())])
    result = ScenarioIRPlanner(llm=llm).plan("驱逐舰海上巡逻")

    assert result.success
    assert any("sea" in e for e in result.attempts[0].errors)
    assert result.scenario_ir["platforms"][0]["mover_type"] == "WSF_SURFACE_MOVER"


# ── 能力白名单确实放行了海陆 mover ──────────────────────────────────


@pytest.mark.parametrize(
    "symbol", ["WSF_AIR_MOVER", "WSF_SURFACE_MOVER", "WSF_GROUND_MOVER"]
)
def test_domain_movers_are_whitelisted(symbol: str) -> None:
    assert DEFAULT_CAPABILITY_REGISTRY.is_compiler_enabled(
        CapabilityKind.MOVER_TYPE, symbol
    )


def test_space_mover_is_not_whitelisted() -> None:
    assert not DEFAULT_CAPABILITY_REGISTRY.is_compiler_enabled(
        CapabilityKind.MOVER_TYPE, "WSF_SPACE_MOVER"
    )


def test_unverified_capabilities_are_reported() -> None:
    """放行但未经真机验证的能力必须在结果里如实标注。"""
    llm = ScriptedLLM([_dumps(_sea_ir())])
    result = ScenarioIRPlanner(llm=llm).plan("驱逐舰海上巡逻")
    assert result.success
    assert any("WSF_SURFACE_MOVER" in item for item in result.unverified_capabilities)


def test_verified_only_scenario_reports_no_unverified() -> None:
    llm = ScriptedLLM([_dumps(_valid_ir())])
    result = ScenarioIRPlanner(llm=llm).plan("预警机搜索红方雷达")
    assert result.success
    assert result.unverified_capabilities == []


# ── 能力缺口侦测 ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "request_text,expected_topic",
    [
        ("2架歼-16D对敌雷达实施阻塞干扰", "电子战/干扰"),
        ("用ESM侦收定位辐射源", "ESM 无源侦收"),
        ("通过Link-16分发航迹给战斗机", "通信/数据链"),
        ("要求能看到拦截弹的飞行轨迹和命中判定", "导弹飞行与制导"),
        ("雷达工作在S波段峰值功率100千瓦天线增益35dB", "雷达方程级建模"),
        ("侦察卫星在500公里轨道上过境", "空间轨道运动"),
        ("潜艇用声呐搜索目标", "水声探测"),
        ("平台按行为树自主决策", "自主决策/行为树"),
        ("运油-20为编队实施空中加油", "燃油与空中加油"),
    ],
)
def test_capability_gap_is_detected(request_text: str, expected_topic: str) -> None:
    topics = {gap.topic for gap in detect_capability_gaps(request_text)}
    assert expected_topic in topics


def test_plain_air_request_reports_no_gap() -> None:
    text = "两架战斗机在8000米高度用机载雷达搜索并拦截一架来袭轰炸机"
    assert detect_capability_gaps(text) == []


def test_gap_detection_is_case_insensitive_for_english() -> None:
    topics = {gap.topic for gap in detect_capability_gaps("apply ECM jamming")}
    assert "电子战/干扰" in topics


def test_gaps_are_injected_into_prompt() -> None:
    """缺口要提前告知模型, 免得它拿相近字段假装实现。"""
    llm = ScriptedLLM([_dumps(_valid_ir())])
    ScenarioIRPlanner(llm=llm).plan("用ESM侦收定位辐射源并实施干扰")
    prompt = llm.calls[0][-1]["content"]
    assert "无法表达的能力" in prompt
    assert "电子战/干扰" in prompt


def test_gaps_are_carried_into_result() -> None:
    llm = ScriptedLLM([_dumps(_valid_ir())])
    result = ScenarioIRPlanner(llm=llm).plan("实施阻塞干扰")
    assert result.success
    assert [gap.topic for gap in result.capability_gaps] == ["电子战/干扰"]


# ── 范围闸门 ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "request_text",
    [
        "做一个城市早高峰地铁客流仿真",
        "模拟电商大促的订单量变化",
        "预测明天的股价走势",
    ],
)
def test_civil_requests_are_refused_without_calling_llm(request_text: str) -> None:
    llm = ScriptedLLM([])   # 空脚本: 一旦调用 LLM 就会抛错
    result = ScenarioIRPlanner(llm=llm).plan(request_text)
    assert result.success is False
    assert result.out_of_scope
    assert llm.calls == []   # 入口就挡住了, 没有浪费 API 调用


def test_military_request_is_not_refused() -> None:
    llm = ScriptedLLM([_dumps(_valid_ir())])
    result = ScenarioIRPlanner(llm=llm).plan("预警机搜索红方雷达阵地")
    assert result.success
    assert result.out_of_scope == ""


def test_model_side_out_of_scope_sentinel_is_honoured() -> None:
    """关键词没覆盖到时, 模型自己判超纲也要被采纳而不是当成脏输出。"""
    llm = ScriptedLLM([json.dumps({"__out_of_scope__": "这是农业灌溉排程问题"})])
    result = ScenarioIRPlanner(llm=llm).plan("给我排一个大棚灌溉计划的仿真")
    assert result.success is False
    assert "农业灌溉" in result.out_of_scope


def test_out_of_scope_keywords_listed() -> None:
    assert looks_out_of_scope("地铁客流统计") == ["地铁", "客流"]
    assert looks_out_of_scope("两架战斗机空战") == []


# ── 确定性归一化 ────────────────────────────────────────────────────


def test_normalization_does_not_touch_mover_without_route() -> None:
    """"有 mover 却没航路"有两个合理答案, 不能由代码替模型决定。

    早先版本让归一化统一删 mover, 结果把数据链协同、空中对抗这类想定里所有飞机
    都变成了静止平台 —— 语法过了、需求没了, 问题从"校验不通过"降级成"悄悄降级"。
    现在交回 schema 报错, 由模型逐平台判断是补航路还是确实该静止。
    """
    ir = _sea_ir()
    ir["platforms"][0]["route"] = []
    normalized, notes = normalize_ir_payload(ir)
    assert normalized["platforms"][0]["mover_type"] == "WSF_SURFACE_MOVER"
    assert notes == []


def test_mover_without_route_error_prefers_adding_a_route() -> None:
    """报错要给出单一首选修法, 否则模型会在两个选项之间反复摇摆。"""
    ir = _sea_ir()
    ir["platforms"][0]["route"] = []
    with pytest.raises(ValidationError) as excinfo:
        ScenarioIR.model_validate(ir)
    message = str(excinfo.value)
    assert "Preferred fix: add at least one route waypoint" in message


def test_mover_without_route_is_repaired_by_adding_route() -> None:
    broken = _sea_ir()
    broken["platforms"][0]["route"] = []

    llm = ScriptedLLM([_dumps(broken), _dumps(_sea_ir())])
    result = ScenarioIRPlanner(llm=llm).plan("驱逐舰海上巡逻")

    assert result.success
    assert result.scenario_ir["platforms"][0]["route"]
    assert any("route waypoint" in e for e in result.attempts[0].errors)


def test_normalization_zeroes_sea_altitude() -> None:
    ir = _sea_ir()
    ir["platforms"][0]["position"]["altitude_m"] = 8000.0
    ir["platforms"][0]["route"][0]["position"]["altitude_m"] = 8000.0
    normalized, notes = normalize_ir_payload(ir)
    assert normalized["platforms"][0]["position"]["altitude_m"] == 0.0
    assert normalized["platforms"][0]["route"][0]["position"]["altitude_m"] == 0.0
    assert any("高度" in note for note in notes)


def test_normalization_leaves_valid_ir_untouched() -> None:
    ir = _sea_ir()
    normalized, notes = normalize_ir_payload(ir)
    assert notes == []
    assert normalized == ir


def test_normalization_does_not_mutate_input() -> None:
    ir = _sea_ir()
    ir["platforms"][0]["route"] = []
    snapshot = json.dumps(ir, sort_keys=True)
    normalize_ir_payload(ir)
    assert json.dumps(ir, sort_keys=True) == snapshot


def test_normalization_is_reported_not_silent() -> None:
    """归一化改动了使用者的 IR, 必须出现在结果里而不是悄悄发生。"""
    airborne_ship = _sea_ir()
    airborne_ship["platforms"][0]["position"]["altitude_m"] = 8000.0
    airborne_ship["platforms"][0]["route"][0]["position"]["altitude_m"] = 8000.0

    llm = ScriptedLLM([_dumps(airborne_ship)])
    result = ScenarioIRPlanner(llm=llm).plan("驱逐舰海上巡逻")

    assert result.success
    assert result.rounds == 1          # 一轮就过, 没有回灌给模型
    assert result.normalizations
    assert any("高度" in note for note in result.normalizations)


def test_normalization_handles_malformed_payload_gracefully() -> None:
    """归一化在 schema 校验之前跑, 必须容忍结构不完整的脏输入。"""
    for payload in ({}, {"platforms": "not-a-list"}, {"platforms": [None]}):
        normalized, notes = normalize_ir_payload(payload)
        assert isinstance(normalized, dict)
        assert notes == []
