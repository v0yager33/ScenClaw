"""自然语言 → ScenarioIR 规划器的离线测试。

用一个按脚本回放的假 LLM 替代真实 API, 因此这些测试不需要网络也不需要 API key。
重点验证控制流而不是模型质量:

- 严格 JSON + pydantic 校验能挡住脏输出;
- 校验/编译报错能回灌并在下一轮被修掉;
- 战术合理性问题会触发 refine 轮, 且 refine 不会让结果退化;
- 修复轮数有上界, 不会无限烧 token。
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from pydantic import ValidationError

from nanobot.afsim import ScenarioIR
from nanobot.agent.tools.nl_scenario_planner import (
    ScenarioIRPlanner,
    _parse_json_object,
    check_ir_plausibility,
)


class ScriptedLLM:
    """按预设脚本依次返回响应的假 LLM, 并记录收到的对话。"""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict], **_: Any) -> str:
        self.calls.append(copy.deepcopy(messages))
        if not self.responses:
            raise AssertionError("假 LLM 被调用的次数超过了脚本长度")
        return self.responses.pop(0)

    @property
    def last_user_message(self) -> str:
        return self.calls[-1][-1]["content"]


# ── 测试用 IR 构造 ──────────────────────────────────────────────────


def _valid_ir() -> dict[str, Any]:
    """一份合法且战术上说得通的最小 IR: 蓝方带雷达, 红方静止目标。"""
    return {
        "scenario_id": "unit_probe",
        "mission": {
            "mission_id": "probe",
            "name": "探测任务",
            "objective": "发现并跟踪红方雷达",
            "duration_s": 1800.0,
            "platforms": [
                {
                    "platform_id": "blue_awacs",
                    "side": "blue",
                    "domain": "air",
                    "type_hint": "预警机",
                    "initial_position": {
                        "latitude_deg": 27.5,
                        "longitude_deg": 121.0,
                        "altitude_m": 9000.0,
                    },
                },
                {
                    "platform_id": "red_radar",
                    "side": "red",
                    "domain": "land",
                    "type_hint": "警戒雷达阵地",
                    "initial_position": {
                        "latitude_deg": 27.5,
                        "longitude_deg": 122.5,
                        "altitude_m": 0.0,
                    },
                },
            ],
        },
        "selected_coa": {
            "coa_id": "coa_1",
            "mission_id": "probe",
            "summary": "预警机搜索红方雷达",
        },
        "sensors": [
            {"sensor_type_id": "SEARCH_RADAR", "maximum_range_m": 200000.0}
        ],
        "platforms": [
            {
                "platform_id": "blue_awacs",
                "side": "blue",
                "position": {
                    "latitude_deg": 27.5,
                    "longitude_deg": 121.0,
                    "altitude_m": 9000.0,
                },
                "mover_type": "WSF_AIR_MOVER",
                "sensors": [
                    {"instance_name": "radar", "sensor_type_id": "SEARCH_RADAR"}
                ],
                "route": [
                    {
                        "position": {
                            "latitude_deg": 27.6,
                            "longitude_deg": 121.2,
                            "altitude_m": 9000.0,
                        },
                        "speed_mps": 200.0,
                    }
                ],
            },
            {
                "platform_id": "red_radar",
                "side": "red",
                "position": {
                    "latitude_deg": 27.5,
                    "longitude_deg": 122.5,
                    "altitude_m": 0.0,
                },
            },
        ],
        "assertions": [
            {
                "assertion_id": "detect_radar",
                "event_type": "SENSOR_DETECTION_ATTEMPT",
                "source_platform": "blue_awacs",
                "target_platform": "red_radar",
                "sensor_name": "radar",
            }
        ],
    }


def _dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


# ── JSON 解析容错 ───────────────────────────────────────────────────


def test_parse_json_object_strips_markdown_fence() -> None:
    payload, errors = _parse_json_object('```json\n{"a": 1}\n```')
    assert payload == {"a": 1}
    assert errors == []


def test_parse_json_object_tolerates_surrounding_prose() -> None:
    payload, _ = _parse_json_object('好的，结果如下：\n{"a": 1}\n希望有帮助')
    assert payload == {"a": 1}


def test_parse_json_object_rejects_non_object() -> None:
    payload, errors = _parse_json_object("[1, 2, 3]")
    assert payload is None
    assert errors


def test_parse_json_object_rejects_empty() -> None:
    payload, errors = _parse_json_object("   ")
    assert payload is None
    assert errors


# ── 单轮成功 ────────────────────────────────────────────────────────


def test_single_round_success_produces_afsim() -> None:
    llm = ScriptedLLM([_dumps(_valid_ir())])
    result = ScenarioIRPlanner(llm=llm).plan("预警机搜索红方雷达")

    assert result.success
    assert result.rounds == 1
    assert result.attempts[0].stage == "extract"
    assert "end_platform" in result.afsim_code
    assert "end_time 1800 sec" in result.afsim_code
    assert result.advisories == []


# ── 报错回灌与修复 ──────────────────────────────────────────────────


def test_schema_error_is_fed_back_and_repaired() -> None:
    """第一轮 mission_id 不匹配 → 第二轮修好。"""
    broken = _valid_ir()
    broken["selected_coa"]["mission_id"] = "mismatched"

    llm = ScriptedLLM([_dumps(broken), _dumps(_valid_ir())])
    result = ScenarioIRPlanner(llm=llm).plan("预警机搜索红方雷达")

    assert result.success
    assert result.rounds == 2
    assert result.attempts[0].ok is False
    assert result.attempts[1].stage == "repair"
    # 报错必须真的回灌给了模型
    assert "mission_id" in llm.calls[1][-1]["content"]


def test_unknown_field_is_rejected_and_reported() -> None:
    """schema 是 extra=forbid, 多余字段必须被挡住并生成可回灌的报错。"""
    broken = _valid_ir()
    broken["totally_unexpected"] = 123

    llm = ScriptedLLM([_dumps(broken), _dumps(_valid_ir())])
    result = ScenarioIRPlanner(llm=llm).plan("预警机搜索红方雷达")

    assert result.success
    assert any("totally_unexpected" in e for e in result.attempts[0].errors)


def test_capability_whitelist_violation_is_reported() -> None:
    """未登记的 AFSIM 类型必须被能力白名单拦下。"""
    broken = _valid_ir()
    broken["sensors"][0]["base_type"] = "WSF_RADAR_SENSOR"  # 白名单未放行

    llm = ScriptedLLM([_dumps(broken), _dumps(_valid_ir())])
    result = ScenarioIRPlanner(llm=llm).plan("预警机搜索红方雷达")

    assert result.success
    assert any("WSF_RADAR_SENSOR" in e for e in result.attempts[0].errors)


def test_invalid_json_triggers_reformat_request() -> None:
    llm = ScriptedLLM(["这不是 JSON", _dumps(_valid_ir())])
    result = ScenarioIRPlanner(llm=llm).plan("预警机搜索红方雷达")

    assert result.success
    assert result.attempts[0].ok is False
    assert "JSON" in llm.calls[1][-1]["content"]


def test_repair_rounds_are_bounded() -> None:
    """一直给坏输出时必须在上界处停下, 而不是无限重试。"""
    broken = _valid_ir()
    broken["selected_coa"]["mission_id"] = "mismatched"

    llm = ScriptedLLM([_dumps(broken)] * 3)
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=3).plan("任务")

    assert result.success is False
    assert result.rounds == 3
    assert llm.responses == []  # 恰好用掉 3 次, 没有多调


# ── 战术合理性检查与 refine 轮 ──────────────────────────────────────


def test_plausibility_flags_colocated_platforms() -> None:
    ir = _valid_ir()
    ir["platforms"][1]["position"] = dict(ir["platforms"][0]["position"])
    issues = check_ir_plausibility(ScenarioIR.model_validate(ir))
    assert any("初始位置完全相同" in issue for issue in issues)


def test_mover_without_route_is_a_schema_error() -> None:
    """挂 mover 不给航路是 schema 硬错误。

    正常路径上 normalize_ir_payload 会先把它改成静止平台 (见
    test_nl_scenario_generality.py), 这里守的是绕过归一化直接构造 IR 的情况。
    """
    ir = _valid_ir()
    ir["platforms"][0]["route"] = []
    with pytest.raises(ValidationError) as excinfo:
        ScenarioIR.model_validate(ir)
    assert "no route" in str(excinfo.value)


def test_plausibility_flags_assertion_on_missing_sensor() -> None:
    ir = _valid_ir()
    ir["assertions"][0]["sensor_name"] = "nonexistent"
    issues = check_ir_plausibility(ScenarioIR.model_validate(ir))
    assert any("nonexistent" in issue for issue in issues)


def test_plausibility_flags_same_side_assertion() -> None:
    ir = _valid_ir()
    ir["assertions"][0]["target_platform"] = "blue_awacs"
    issues = check_ir_plausibility(ScenarioIR.model_validate(ir))
    assert any("同属" in issue for issue in issues)


def test_plausibility_flags_scenario_without_sensors() -> None:
    ir = _valid_ir()
    ir["platforms"][0]["sensors"] = []
    ir["assertions"] = []
    issues = check_ir_plausibility(ScenarioIR.model_validate(ir))
    assert any("没有任何平台装备传感器" in issue for issue in issues)


def test_refine_round_fixes_plausibility_issue() -> None:
    """首轮合法但平台重叠 → refine 轮消除, 最终返回干净结果。"""
    overlapping = _valid_ir()
    overlapping["platforms"][1]["position"] = dict(overlapping["platforms"][0]["position"])

    llm = ScriptedLLM([_dumps(overlapping), _dumps(_valid_ir())])
    result = ScenarioIRPlanner(llm=llm).plan("预警机搜索红方雷达")

    assert result.success
    assert result.rounds == 2
    assert result.attempts[0].advisories  # 第一轮有合理性问题
    assert result.attempts[1].stage == "refine"
    assert result.advisories == []       # 最终版本已清零


def test_refine_never_regresses_to_worse_ir() -> None:
    """refine 轮把 IR 改得更差时, 必须保留上一版较好的结果。"""
    good_but_overlapping = _valid_ir()
    good_but_overlapping["platforms"][1]["position"] = dict(
        good_but_overlapping["platforms"][0]["position"]
    )

    worse = copy.deepcopy(good_but_overlapping)
    worse["platforms"][0]["route"] = []          # 再叠加一个"有 mover 无 route"

    llm = ScriptedLLM([_dumps(good_but_overlapping), _dumps(worse)])
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=2).plan("任务")

    assert result.success
    # 保留的是问题较少的第一版
    assert len(result.advisories) == 1
    assert any("初始位置完全相同" in a for a in result.advisories)


def test_refine_breaking_schema_keeps_previous_valid_result() -> None:
    """refine 轮返回非法 IR 时, 不能把已经拿到的合法结果丢掉。"""
    overlapping = _valid_ir()
    overlapping["platforms"][1]["position"] = dict(overlapping["platforms"][0]["position"])

    broken = _valid_ir()
    broken["selected_coa"]["mission_id"] = "mismatched"

    llm = ScriptedLLM([_dumps(overlapping), _dumps(broken)])
    result = ScenarioIRPlanner(llm=llm).plan("任务")

    assert result.success
    assert result.afsim_code
    assert result.attempts[-1].ok is False


# ── 提示词内容 ──────────────────────────────────────────────────────


def test_hints_are_injected_into_prompt() -> None:
    llm = ScriptedLLM([_dumps(_valid_ir())])
    ScenarioIRPlanner(llm=llm).plan(
        "任务", {"center_lat": 30.0, "center_lon": 125.0, "duration_s": 7200}
    )
    prompt = llm.calls[0][-1]["content"]
    assert "30.0" in prompt and "125.0" in prompt
    assert "7200" in prompt


def test_system_prompt_lists_only_whitelisted_types() -> None:
    llm = ScriptedLLM([_dumps(_valid_ir())])
    ScenarioIRPlanner(llm=llm).plan("任务")
    system_prompt = llm.calls[0][0]["content"]
    assert "WSF_GEOMETRIC_SENSOR" in system_prompt
    assert "WSF_AIR_MOVER" in system_prompt
    assert "WSF_IMPLICIT_WEAPON" in system_prompt
    # 未放行的类型不该出现在提示词里, 否则模型会生成编译器拒绝的 IR
    assert "WSF_RADAR_SENSOR" not in system_prompt


@pytest.mark.parametrize("field_name", ["scenario_id", "mission", "selected_coa", "platforms"])
def test_ir_shape_documents_required_fields(field_name: str) -> None:
    llm = ScriptedLLM([_dumps(_valid_ir())])
    ScenarioIRPlanner(llm=llm).plan("任务")
    assert field_name in llm.calls[0][0]["content"]
