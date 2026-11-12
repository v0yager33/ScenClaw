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
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from nanobot.afsim import AfsimCompiler, CompileError, ScenarioIR, emit_project
from nanobot.agent.tools import nl_scenario_planner as planner_module
from nanobot.agent.tools.nl_scenario_planner import (
    NLToAfsimTool,
    ScenarioCompileTool,
    ScenarioIRPlanner,
    _parse_json_object,
    _write_project_output,
    acknowledged_engagement_issues,
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


def test_parse_json_object_tolerates_trailing_extra_closer() -> None:
    """实测: 完整的 IR 末尾多写了一个 "}", 不该整版作废。"""
    payload, errors = _parse_json_object('{"a": {"b": 1}, "c": [2]}}\n')
    assert payload == {"a": {"b": 1}, "c": [2]}
    assert errors == []


def test_parse_json_object_rejects_premature_close() -> None:
    """闭合符后面还有字段, 说明对象提前闭合了, 解出来的是残缺版。"""
    payload, errors = _parse_json_object('{"a": {"b": 1}}, "c": 2}')
    assert payload is None
    assert "提前闭合" in errors[0]
    assert '"c": 2' in errors[0]


def test_parse_json_object_reports_syntax_error_location() -> None:
    payload, errors = _parse_json_object('{"a": 1, "b": tru}')
    assert payload is None
    assert "第 14 个字符" in errors[0]
    assert '"b": ⟦此处⟧tru}' in errors[0]


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


def test_invalid_json_repair_feeds_back_full_output() -> None:
    """只回灌前 2000 字时, 模型看不到自己写过的 IR, 只能整版重新推演。"""
    broken = _dumps(_valid_ir())[:-1] + "x" * 3000 + "}"
    llm = ScriptedLLM([broken, _dumps(_valid_ir())])
    result = ScenarioIRPlanner(llm=llm).plan("预警机搜索红方雷达")

    assert result.success
    assert llm.calls[1][-2] == {"role": "assistant", "content": broken}
    assert "个字符" in llm.calls[1][-1]["content"]


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


def test_refine_prompt_forbids_deleting_required_content() -> None:
    """按条数验收时, 删掉需求要求的射击计划也能"清零", 必须在提示里堵住。"""
    overlapping = _valid_ir()
    overlapping["platforms"][1]["position"] = dict(overlapping["platforms"][0]["position"])

    llm = ScriptedLLM([_dumps(overlapping), _dumps(_valid_ir())])
    ScenarioIRPlanner(llm=llm).plan("预警机搜索红方雷达")

    refine_prompt = llm.calls[1][-1]["content"]
    assert "核对原始需求" in refine_prompt
    assert "不要为了消掉问题删减需求要求的" in refine_prompt


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
    """refine 轮返回非法 IR 时, 不能把已经拿到的合法结果丢掉; 补救一次仍是坏的就停。"""
    overlapping = _valid_ir()
    overlapping["platforms"][1]["position"] = dict(overlapping["platforms"][0]["position"])

    broken = _valid_ir()
    broken["selected_coa"]["mission_id"] = "mismatched"

    llm = ScriptedLLM([_dumps(overlapping), _dumps(broken), _dumps(broken)])
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=4).plan("任务")

    assert result.success
    assert result.afsim_code
    assert len(llm.calls) == 3
    assert [a.stage for a in result.attempts] == ["extract", "refine", "repair"]
    assert result.attempts[-1].ok is False
    assert any("初始位置完全相同" in a for a in result.advisories)


def test_refine_breaking_schema_gets_one_repair() -> None:
    """refine 修对了问题却带了个 schema 小错: 补一次 repair, 不整版丢掉。"""
    overlapping = _valid_ir()
    overlapping["platforms"][1]["position"] = dict(overlapping["platforms"][0]["position"])

    broken = _valid_ir()
    broken["selected_coa"]["mission_id"] = "mismatched"

    llm = ScriptedLLM([_dumps(overlapping), _dumps(broken), _dumps(_valid_ir())])
    result = ScenarioIRPlanner(llm=llm).plan("任务")

    assert [a.stage for a in result.attempts] == ["extract", "refine", "repair"]
    repair_call = llm.calls[2]
    assert json.loads(repair_call[-2]["content"])["selected_coa"]["mission_id"] == "mismatched"
    assert "没通过校验" in repair_call[-1]["content"]
    assert result.advisories == []


def test_refine_feeds_back_the_complete_previous_ir() -> None:
    """大想定的 IR 远超 6000 字符, 回灌时截断会让模型看不到自己的上一版。"""
    overlapping = _valid_ir()
    overlapping["platforms"][1]["position"] = dict(overlapping["platforms"][0]["position"])
    overlapping["mission"]["assumptions"] = [f"假设 {i}: " + "补全参数" * 40 for i in range(40)]

    llm = ScriptedLLM([_dumps(overlapping), _dumps(_valid_ir())])
    ScenarioIRPlanner(llm=llm).plan("预警机搜索红方雷达")

    fed_back = llm.calls[1][-2]
    assert fed_back["role"] == "assistant"
    assert len(fed_back["content"]) > 6000
    assert json.loads(fed_back["content"])["mission"]["assumptions"] == overlapping["mission"]["assumptions"]


def test_refine_conversation_keeps_only_the_latest_version() -> None:
    """逐轮累加完整 IR 会撑爆上下文; 每轮只带最新一版和它的问题清单。"""
    overlapping = _valid_ir()
    overlapping["platforms"][1]["position"] = dict(overlapping["platforms"][0]["position"])

    llm = ScriptedLLM([_dumps(overlapping)] * 3)
    ScenarioIRPlanner(llm=llm, max_repair_rounds=3).plan("任务")

    assert [len(call) for call in llm.calls] == [2, 4, 4]
    assert llm.calls[2][:2] == llm.calls[0]


def test_regressed_refine_feeds_back_the_kept_version() -> None:
    """问题变多的版本被拒后, 下一轮回灌的 IR 要与列出的问题清单对得上。"""
    kept = _valid_ir()
    kept["platforms"][1]["position"] = dict(kept["platforms"][0]["position"])
    worse = copy.deepcopy(kept)
    worse["assertions"][0]["sensor_name"] = "nonexistent"

    llm = ScriptedLLM([_dumps(kept), _dumps(worse), _dumps(_valid_ir())])
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=3).plan("任务")

    fed_back = json.loads(llm.calls[2][-2]["content"])
    assert fed_back["assertions"][0]["sensor_name"] == "radar"
    assert "nonexistent" not in llm.calls[2][-1]["content"]
    assert result.advisories == []


def test_failed_refine_call_keeps_previous_valid_result() -> None:
    """refine 调用超时不该把首轮已经拿到的合法想定一起丢掉。"""
    overlapping = _valid_ir()
    overlapping["platforms"][1]["position"] = dict(overlapping["platforms"][0]["position"])

    class TimeoutOnRefine(ScriptedLLM):
        def chat(self, messages: list[dict], **kwargs: Any) -> str:
            if self.calls:
                raise TimeoutError("read timeout")
            return super().chat(messages, **kwargs)

    llm = TimeoutOnRefine([_dumps(overlapping)])
    result = ScenarioIRPlanner(llm=llm).plan("任务")

    assert result.success
    assert result.afsim_code
    assert result.attempts[-1].ok is False
    assert "TimeoutError" in result.attempts[-1].errors[0]


def test_failed_first_call_still_raises() -> None:
    """还没有任何合法结果时, 调用失败照常抛给上层工具处理。"""
    class AlwaysTimeout(ScriptedLLM):
        def chat(self, messages: list[dict], **kwargs: Any) -> str:
            raise TimeoutError("read timeout")

    with pytest.raises(TimeoutError):
        ScenarioIRPlanner(llm=AlwaysTimeout([])).plan("任务")


# ── 交战时序与弹量 ──────────────────────────────────────────────────


def _engagement(engagement_id: str, shooter: str, target: str, weapon: str, time_s: float) -> dict[str, Any]:
    return {
        "engagement_id": engagement_id,
        "shooter_platform": shooter,
        "target_platform": target,
        "weapon_name": weapon,
        "fire_time_s": time_s,
    }


def _duel_payload(
    engagements: list[dict[str, Any]], *, quantity: int = 2, assumptions: list[str] | None = None
) -> dict[str, Any]:
    """在最小 IR 上给双方各配一件隐式武器。"""
    ir = _valid_ir()
    ir["platforms"][0]["weapons"] = [{"instance_name": "aam", "quantity": quantity}]
    ir["platforms"][1]["weapons"] = [{"instance_name": "sam", "quantity": quantity}]
    ir["engagements"] = copy.deepcopy(engagements)
    if assumptions is not None:
        ir["mission"]["assumptions"] = assumptions
    return ir


def _duel_ir(
    engagements: list[dict[str, Any]], *, quantity: int = 2, assumptions: list[str] | None = None
) -> ScenarioIR:
    return ScenarioIR.model_validate(
        _duel_payload(engagements, quantity=quantity, assumptions=assumptions)
    )


_RETURN_FIRE = [
    _engagement("strike", "blue_awacs", "red_radar", "aam", 600.0),
    _engagement("reply", "red_radar", "blue_awacs", "sam", 615.0),
]


def test_plausibility_accepts_a_single_supported_shot() -> None:
    scenario = _duel_ir([_engagement("strike", "blue_awacs", "red_radar", "aam", 600.0)])
    assert check_ir_plausibility(scenario) == []


def test_plausibility_flags_return_fire_from_destroyed_shooter() -> None:
    """实跑里的典型写法: A 先射 B, B 晚 15s 回射 A —— 后一发永远打不出来。"""
    scenario = _duel_ir(_RETURN_FIRE)
    issues = check_ir_plausibility(scenario)
    assert any("交战 reply" in i and "射手 red_radar 已在 600s" in i for i in issues)


def test_acknowledged_destroyed_shooter_is_listed_but_not_an_issue() -> None:
    """需求要求每架都有射击计划时, 最先被击毁的那架注定打不出; 写明原因就不再逼模型去消。"""
    scenario = _duel_ir(
        _RETURN_FIRE, assumptions=["reply: 需求要求红方阵地也有射击计划, 它会先被 strike 摧毁"]
    )
    # 红方没配传感器会另报一条, 这里只看时序
    assert not any("射手 red_radar 已在" in i for i in check_ir_plausibility(scenario))
    acknowledged = acknowledged_engagement_issues(scenario)
    assert len(acknowledged) == 1
    assert "射手 red_radar 已在 600s" in acknowledged[0]


def test_acknowledgement_matches_whole_engagement_id() -> None:
    """写了 reply_2 或 xreply 的说明, 不能顺带覆盖 reply。"""
    scenario = _duel_ir(_RETURN_FIRE, assumptions=["reply_2 与 xreply 是有意安排"])
    assert any("射手 red_radar 已在" in i for i in check_ir_plausibility(scenario))
    assert acknowledged_engagement_issues(scenario) == []


def test_wasted_shot_cannot_be_acknowledged() -> None:
    """打已毁目标只是白白浪费一发, 总能改打别的目标, 说明了也照样算问题。"""
    scenario = _duel_ir(
        [
            _engagement("first", "blue_awacs", "red_radar", "aam", 600.0),
            _engagement("second", "blue_awacs", "red_radar", "aam", 700.0),
        ],
        assumptions=["second 是补射"],
    )
    assert any("交战 second" in i for i in check_ir_plausibility(scenario))
    assert acknowledged_engagement_issues(scenario) == []


def test_plan_result_reports_acknowledged_shots_separately() -> None:
    payload = _duel_payload(_RETURN_FIRE, assumptions=["reply 打不出: 红方阵地先被摧毁"])
    result = ScenarioIRPlanner(llm=ScriptedLLM([_dumps(payload)]), max_repair_rounds=1).plan("任务")

    assert result.success
    assert not any("射手 red_radar 已在" in a for a in result.advisories)
    assert any("射手 red_radar 已在" in a for a in result.acknowledged)


def test_timeline_digest_marks_waypoints_lost_to_early_kill() -> None:
    """蓝方 113s 才到航点, 50s 就被击毁: 时间线要直接点出这段航路飞不到。"""
    scenario = _duel_ir([_engagement("ambush", "red_radar", "blue_awacs", "sam", 50.0)])
    digest = "\n".join(planner_module._timeline_digest(scenario))
    assert "- blue_awacs: 航点预计到达 1" in digest
    assert "50s 被交战 ambush 摧毁, 之后 1 个航点不会飞到" in digest
    assert "- red_radar: 无航路" in digest


def test_refine_prompt_carries_computed_timeline() -> None:
    """时间窗核对本该是确定性计算, 不该让推理模型在思考里重算。"""
    llm = ScriptedLLM([_dumps(_duel_payload(_RETURN_FIRE)), _dumps(_valid_ir())])
    ScenarioIRPlanner(llm=llm).plan("任务")

    refine_prompt = llm.calls[1][-1]["content"]
    assert "按当前 IR 推算的时间线" in refine_prompt
    assert "- blue_awacs: 航点预计到达" in refine_prompt
    assert "- red_radar: 无航路; 600s 被交战 strike 摧毁" in refine_prompt


def test_plausibility_flags_shot_at_destroyed_target() -> None:
    scenario = _duel_ir([
        _engagement("first", "blue_awacs", "red_radar", "aam", 600.0),
        _engagement("second", "blue_awacs", "red_radar", "aam", 700.0),
    ])
    issues = check_ir_plausibility(scenario)
    assert any("交战 second" in i and "目标 red_radar 已在 600s" in i for i in issues)


def test_plausibility_flags_simultaneous_shots_on_one_target() -> None:
    scenario = _duel_ir([
        _engagement("first", "blue_awacs", "red_radar", "aam", 600.0),
        _engagement("second", "blue_awacs", "red_radar", "aam", 600.0),
    ])
    issues = check_ir_plausibility(scenario)
    assert any("同时射击 red_radar" in i for i in issues)


def test_plausibility_flags_same_second_exchange_as_uncertain() -> None:
    """同一秒的两条 execute 也是逐条执行的: 不判"射手已毁", 但两发都可能打不出。"""
    scenario = _duel_ir([
        _engagement("strike", "blue_awacs", "red_radar", "aam", 600.0),
        _engagement("reply", "red_radar", "blue_awacs", "sam", 600.0),
    ])
    issues = check_ir_plausibility(scenario)
    assert not any("已在" in i for i in issues)
    assert any("交战 strike" in i and "同一时刻" in i and "交战 reply" in i for i in issues)
    assert any("交战 reply" in i and "同一时刻" in i and "交战 strike" in i for i in issues)


def test_plausibility_accepts_staggered_chain() -> None:
    """被打的一方先打别的目标, 再被击毁: 两发都打得出。"""
    ir = _valid_ir()
    ir["platforms"][0]["weapons"] = [{"instance_name": "aam", "quantity": 2}]
    ir["platforms"][1]["weapons"] = [{"instance_name": "sam", "quantity": 2}]
    wingman = copy.deepcopy(ir["platforms"][0])
    wingman["platform_id"] = "blue_wing"
    wingman["position"]["latitude_deg"] += 0.1
    for point in wingman["route"]:
        point["position"]["latitude_deg"] += 0.1
    ir["platforms"].append(wingman)
    wing_intent = copy.deepcopy(ir["mission"]["platforms"][0])
    wing_intent["platform_id"] = "blue_wing"
    wing_intent["initial_position"] = copy.deepcopy(wingman["position"])
    ir["mission"]["platforms"].append(wing_intent)
    ir["engagements"] = [
        _engagement("red_first", "red_radar", "blue_wing", "sam", 590.0),
        _engagement("blue_kill", "blue_awacs", "red_radar", "aam", 600.0),
    ]
    scenario = ScenarioIR.model_validate(ir)
    issues = check_ir_plausibility(scenario)
    # 红方没配传感器会另报一条, 这里只看时序
    assert not any("已在" in i or "同一时刻" in i or "同时射击" in i for i in issues)


def test_plausibility_flags_shots_beyond_quantity() -> None:
    scenario = _duel_ir(
        [
            _engagement("first", "blue_awacs", "red_radar", "aam", 600.0),
            _engagement("second", "blue_awacs", "red_radar", "aam", 700.0),
        ],
        quantity=1,
    )
    issues = check_ir_plausibility(scenario)
    assert any("quantity=1" in i and "2 次射击" in i for i in issues)


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


def test_system_prompt_explains_instant_kill_semantics() -> None:
    """不告诉模型开火即毁, 它就会写出"先射、再回射"这种打不出来的计划。"""
    llm = ScriptedLLM([_dumps(_valid_ir())])
    ScenarioIRPlanner(llm=llm).plan("任务")
    system_prompt = llm.calls[0][0]["content"]
    assert "开火瞬间就摧毁目标" in system_prompt
    assert "射击次数不超过它的 quantity" in system_prompt
    # 实测只讲"先射再回射打不出"不够, 还要告诉它怎样让双方都开火
    assert "在被击毁之前先打别的目标" in system_prompt
    assert "同一秒互射" in system_prompt
    # 只讲"每发都要打得出", 模型会把承担穿越、撤离航段的平台提前打掉
    assert "此后的航点都不会飞到" in system_prompt
    assert "engagement_id 和原因" in system_prompt


class FinishReasonLLM(ScriptedLLM):
    """按脚本同时给出 finish_reason 的假 LLM。"""

    def __init__(
        self, responses: list[tuple[str, str]], usages: list[dict | None] | None = None
    ) -> None:
        super().__init__([text for text, _ in responses])
        self.reasons = [reason for _, reason in responses]
        self.usages = usages or [None] * len(responses)
        self.last_finish_reason: str | None = None
        self.last_usage: dict | None = None

    def chat(self, messages: list[dict], **kwargs: Any) -> str:
        self.last_finish_reason = self.reasons[len(self.calls)]
        self.last_usage = self.usages[len(self.calls)]
        return super().chat(messages, **kwargs)


# rerun 实测的 refine 轮: 推理吃掉 57424 / 65536, 正文只写出不到一半
_THOUGHT_OUT_USAGE = {"completion_tokens": 65536, "completion_tokens_details": {"reasoning_tokens": 57424}}
_LONG_IR_USAGE = {"completion_tokens": 65536, "completion_tokens_details": {"reasoning_tokens": 4000}}
_PARTIAL_REPLY = '{"scenario_id": "unit_probe", "platforms": ['


def _overlapping_ir() -> dict[str, Any]:
    ir = _valid_ir()
    ir["platforms"][1]["position"] = dict(ir["platforms"][0]["position"])
    return ir


def test_reasoning_exhausted_empty_reply_is_retried() -> None:
    """推理模型把 max_tokens 全耗在思考上时正文为空; 推演长度每次不同, 值得催一次再试。"""
    llm = FinishReasonLLM([("", "length"), (_dumps(_valid_ir()), "stop")])
    result = ScenarioIRPlanner(llm=llm).plan("任务")

    assert result.success
    assert "推理过程耗尽了 max_tokens" in result.attempts[0].errors[-1]
    retry = llm.calls[1]
    assert [m["role"] for m in retry] == ["system", "user"]
    assert retry[1]["content"].startswith(llm.calls[0][1]["content"])
    assert "直接输出完整的 ScenarioIR JSON" in retry[1]["content"]


@pytest.mark.parametrize("usage", [None, _LONG_IR_USAGE])
def test_truncated_partial_reply_is_not_retried(usage: dict | None) -> None:
    """正文写到一半被截断、推理又没占多少, 说明 IR 本身超预算, 同一预算下重试只会再截断。"""
    llm = FinishReasonLLM([(_PARTIAL_REPLY, "length")], usages=[usage])
    result = ScenarioIRPlanner(llm=llm).plan("任务")

    assert not result.success
    assert len(llm.calls) == 1
    assert "上限被截断" in result.attempts[0].errors[-1]


def test_reasoning_heavy_partial_refine_is_nudged_once() -> None:
    """refine 轮推理吃掉大半预算、正文只写出一截: 催它少想, 带着同一份问题清单再试。"""
    llm = FinishReasonLLM(
        [(_dumps(_overlapping_ir()), "stop"), (_PARTIAL_REPLY, "length"), (_dumps(_valid_ir()), "stop")],
        usages=[None, _THOUGHT_OUT_USAGE, None],
    )
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=3).plan("任务")

    assert result.success
    assert result.advisories == []
    assert "推理过程耗尽了 max_tokens" in result.attempts[1].errors[-1]
    retry = llm.calls[2]
    assert retry[-2] == llm.calls[1][-2]  # 回灌的仍是同一版 IR
    assert retry[-1]["content"].startswith(llm.calls[1][-1]["content"])
    # 修订轮的上一版就在对话里: 催它只输出补丁, 不再整份重抄
    assert "只输出改动的补丁" in retry[-1]["content"]
    # rerun 实测: 没有这句时重问照样逐条推演, 第二次又耗尽; extract 的催促有这句, 重问推理几乎为 0
    assert "少做逐条推演" in retry[-1]["content"]


def test_repeated_thought_out_refine_keeps_best_without_more_retries() -> None:
    """已有合法结果时只催一次: 再截断就停, 不为 refine 反复烧预算。"""
    llm = FinishReasonLLM(
        [(_dumps(_overlapping_ir()), "stop"), (_PARTIAL_REPLY, "length"), (_PARTIAL_REPLY, "length")],
        usages=[None, _THOUGHT_OUT_USAGE, _THOUGHT_OUT_USAGE],
    )
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=4).plan("任务")

    assert result.success
    assert len(llm.calls) == 3
    assert any("初始位置完全相同" in a for a in result.advisories)


# ── 需求核对轮 ───────────────────────────────────────────────────


_REQUIREMENTS_MET = _dumps({"__requirements_met__": True})


def _single_strike_payload(fire_time_s: float = 600.0) -> dict[str, Any]:
    return _duel_payload([_engagement("strike", "blue_awacs", "red_radar", "aam", fire_time_s)])


def test_audit_is_off_by_default() -> None:
    llm = ScriptedLLM([_dumps(_valid_ir())])
    result = ScenarioIRPlanner(llm=llm).plan("任务")
    assert result.rounds == 1
    assert result.requirements_audit == ""


def test_audit_confirmation_ends_planning() -> None:
    """通用检查清零后再问一轮; 模型确认需求都满足就保留原版, 不必重抄整份 IR。"""
    llm = ScriptedLLM([_dumps(_single_strike_payload()), _REQUIREMENTS_MET])
    result = ScenarioIRPlanner(llm=llm, audit_requirements=True).plan("蓝方 600 秒打击红方雷达")

    assert result.success
    assert [a.stage for a in result.attempts] == ["extract", "audit"]
    assert result.requirements_audit == "met"
    assert result.scenario_ir["engagements"][0]["fire_time_s"] == 600.0
    audit_call = llm.calls[1]
    assert audit_call[:2] == llm.calls[0]  # 原始需求仍在对话里
    assert json.loads(audit_call[-2]["content"])["engagements"][0]["engagement_id"] == "strike"
    assert "__requirements_met__" in audit_call[-1]["content"]


def test_audit_prompt_carries_fact_sheet() -> None:
    """对照需求所需的数字由代码算好: 航点时刻与坐标、开火斜距与结算、初始距离、存活数。"""
    llm = ScriptedLLM([_dumps(_single_strike_payload()), _REQUIREMENTS_MET])
    ScenarioIRPlanner(llm=llm, audit_requirements=True).plan("任务")

    prompt = llm.calls[1][-1]["content"]
    assert "数值约束" in prompt and "谁击毁谁" in prompt
    assert "- blue_awacs [blue]: 初始 (27.500, 121.000, 9000m); 航点1 (27.600, 121.200, 9000m) 113s" in prompt
    assert "- red_radar [red]: 初始 (27.500, 122.500, 0m); 无航路; 600s 被交战 strike 摧毁" in prompt
    assert "- 600s strike: blue_awacs → red_radar; 开火时位置无法推算" in prompt
    assert "- blue–red: 147.9km (blue_awacs ↔ red_radar)" in prompt
    assert "- blue: 1/1" in prompt and "- red: 0/1" in prompt


def test_fact_sheet_reports_fire_range_and_outcomes() -> None:
    scenario = _duel_ir([
        _engagement("strike", "blue_awacs", "red_radar", "aam", 100.0),
        _engagement("reply", "red_radar", "blue_awacs", "sam", 110.0),
        _engagement("again", "blue_awacs", "red_radar", "aam", 120.0),
    ])
    sheet = planner_module._fact_sheet(scenario)
    assert "- 100s strike: blue_awacs → red_radar; 开火时斜距 " in sheet
    assert "km; 摧毁目标" in sheet
    assert "- 110s reply: red_radar → blue_awacs; 开火时位置无法推算" not in sheet
    assert "reply: red_radar → blue_awacs; 开火时斜距" in sheet
    assert "打不出: 射手已在 100s 被交战 strike 摧毁" in sheet
    assert "again: blue_awacs → red_radar" in sheet and "被跳过: 目标已在 100s" in sheet


def test_fact_sheet_flags_only_same_second_return_fire() -> None:
    """实测: 先开火、一两百秒后才被击毁的射击也被标成"同秒遭射击, 可能打不出"。"""
    engagement = _duel_ir([_engagement("strike", "blue_awacs", "red_radar", "aam", 600.0)]).engagements[0]
    later = {"red_radar": (600.0, "strike"), "blue_awacs": (700.0, "other")}
    same = {"red_radar": (600.0, "strike"), "blue_awacs": (600.0, "other")}
    assert planner_module._engagement_outcome(engagement, later, implicit=True) == "摧毁目标"
    assert "同秒遭交战 other 射击" in planner_module._engagement_outcome(engagement, same, implicit=True)


def test_audit_revision_is_accepted_and_audited_again() -> None:
    """核对轮发现需求没满足就输出修正版; 修正版问题不多于原版就接受, 并再核对一次。"""
    llm = ScriptedLLM([
        _dumps(_single_strike_payload(600.0)),
        _dumps(_single_strike_payload(900.0)),
        _REQUIREMENTS_MET,
    ])
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=3, audit_requirements=True).plan(
        "蓝方 900 秒打击红方雷达"
    )

    assert [a.stage for a in result.attempts] == ["extract", "audit", "audit"]
    assert result.requirements_audit == "met"
    assert result.scenario_ir["engagements"][0]["fire_time_s"] == 900.0
    assert "- 900s strike" in llm.calls[2][-1]["content"]


def test_audit_revision_with_new_issue_gets_one_refine_before_acceptance() -> None:
    """按需求修正时引入了新问题: 先在修正版上 refine, 改干净了再接受, best 不提前被替换。"""
    revised = _single_strike_payload(900.0)
    revised_with_overlap = copy.deepcopy(revised)
    revised_with_overlap["platforms"][1]["position"] = dict(revised["platforms"][0]["position"])
    llm = ScriptedLLM([
        _dumps(_single_strike_payload(600.0)),
        _dumps(revised_with_overlap),
        _dumps(revised),
        _REQUIREMENTS_MET,
    ])
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=4, audit_requirements=True).plan("任务")

    assert [a.stage for a in result.attempts] == ["extract", "audit", "refine", "audit"]
    refine_call = llm.calls[2]
    assert json.loads(refine_call[-2]["content"])["engagements"][0]["fire_time_s"] == 900.0
    assert "初始位置完全相同" in refine_call[-1]["content"]
    assert result.requirements_audit == "met"
    assert result.advisories == []
    assert result.scenario_ir["engagements"][0]["fire_time_s"] == 900.0


def test_audit_revision_that_stays_worse_keeps_the_audited_version() -> None:
    worse = _single_strike_payload(900.0)
    worse["platforms"][1]["position"] = dict(worse["platforms"][0]["position"])
    llm = ScriptedLLM([_dumps(_single_strike_payload(600.0)), _dumps(worse), _dumps(worse)])
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=4, audit_requirements=True).plan("任务")

    assert len(llm.calls) == 3  # 修正版没改好, 原版已核对过, 不再继续
    assert result.advisories == []
    assert result.scenario_ir["engagements"][0]["fire_time_s"] == 600.0
    assert result.requirements_audit == "unconfirmed"


def test_audit_revision_with_schema_slip_is_repaired_then_audited() -> None:
    """实测: 核对轮把时刻改对了, 却顺手多写一个字段; 修掉 schema 错误后照常验收。"""
    slipped = _single_strike_payload(900.0)
    slipped["mission"]["assumptions_note"] = "蓝方改到 900 秒开火"
    llm = ScriptedLLM([
        _dumps(_single_strike_payload(600.0)),
        _dumps(slipped),
        _dumps(_single_strike_payload(900.0)),
        _REQUIREMENTS_MET,
    ])
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=4, audit_requirements=True).plan("任务")

    assert [a.stage for a in result.attempts] == ["extract", "audit", "repair", "audit"]
    assert "assumptions_note" in llm.calls[2][-1]["content"]
    assert result.requirements_audit == "met"
    assert result.scenario_ir["engagements"][0]["fire_time_s"] == 900.0


def test_repaired_audit_revision_with_new_issue_still_gets_refined() -> None:
    slipped = _single_strike_payload(900.0)
    slipped["selected_coa"]["mission_id"] = "mismatched"
    repaired_with_overlap = _single_strike_payload(900.0)
    repaired_with_overlap["platforms"][1]["position"] = dict(
        repaired_with_overlap["platforms"][0]["position"]
    )
    llm = ScriptedLLM([
        _dumps(_single_strike_payload(600.0)),
        _dumps(slipped),
        _dumps(repaired_with_overlap),
        _dumps(_single_strike_payload(900.0)),
        _REQUIREMENTS_MET,
    ])
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=4, audit_requirements=True).plan("任务")

    assert [a.stage for a in result.attempts] == ["extract", "audit", "repair", "refine", "audit"]
    assert json.loads(llm.calls[3][-2]["content"])["engagements"][0]["fire_time_s"] == 900.0
    assert result.scenario_ir["engagements"][0]["fire_time_s"] == 900.0
    assert result.requirements_audit == "met"  # 末轮 refine 才清零, 补一轮核对


def test_null_extra_fields_are_dropped_before_validation() -> None:
    """值为空的多余字段不携带信息, 确定性删掉, 不为它打回整版。"""
    payload = _valid_ir()
    payload["mission"]["assumptions_note"] = None
    payload["platforms"][0]["comment"] = ""
    result = ScenarioIRPlanner(llm=ScriptedLLM([_dumps(payload)])).plan("任务")

    assert result.success and result.rounds == 1
    assert any("mission.assumptions_note" in note for note in result.normalizations)
    assert any("platforms.0.comment" in note for note in result.normalizations)
    assert "assumptions_note" not in result.scenario_ir["mission"]


def test_extra_field_duplicating_a_sibling_is_dropped() -> None:
    """实测: 模型多写了 "coa_id_note": "coa_1", 值与 coa_id 一样, 删掉不丢信息。"""
    payload = _valid_ir()
    payload["selected_coa"]["coa_id_note"] = "coa_1"
    result = ScenarioIRPlanner(llm=ScriptedLLM([_dumps(payload)])).plan("任务")

    assert result.success and result.rounds == 1
    assert any("selected_coa.coa_id_note" in note and "coa_id" in note
               for note in result.normalizations)


def test_extra_field_matching_sibling_only_loosely_is_kept() -> None:
    """True == 1 在 Python 里成立, 但布尔与数字不是同一份信息。"""
    payload = _valid_ir()
    payload["mission"]["duration_s"] = 1
    payload["mission"]["is_drill"] = True
    normalized, _ = planner_module.normalize_ir_payload(payload)
    assert normalized["mission"]["is_drill"] is True


def test_non_null_extra_fields_are_left_for_the_validator() -> None:
    payload = _valid_ir()
    payload["mission"]["assumptions_note"] = "写在了不存在的字段里"
    normalized, notes = planner_module.normalize_ir_payload(payload)

    assert normalized["mission"]["assumptions_note"] == "写在了不存在的字段里"
    assert not any("assumptions_note" in note for note in notes)


def test_audit_without_rounds_left_is_skipped() -> None:
    llm = ScriptedLLM([_dumps(_valid_ir())])
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=1, audit_requirements=True).plan("任务")
    assert result.rounds == 1
    assert result.requirements_audit == ""


def test_refine_clearing_issues_in_the_last_round_gets_one_extra_audit() -> None:
    """实测: refine 用满轮数才清零, 核对被跳过, assumptions 里的战果描述没人对照事实表。"""
    llm = ScriptedLLM([_dumps(_overlapping_ir()), _dumps(_valid_ir()), _REQUIREMENTS_MET])
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=2, audit_requirements=True).plan("任务")
    assert [a.stage for a in result.attempts] == ["extract", "refine", "audit"]
    assert result.requirements_audit == "met"


def test_extra_audit_is_granted_only_once() -> None:
    """实测: 末轮核对改写了击毁时序。改完的版本再核对一次; 还在改就停, 不无限补轮。"""
    llm = ScriptedLLM([
        _dumps(_single_strike_payload(600.0)),
        _dumps(_single_strike_payload(900.0)),
        _dumps(_single_strike_payload(1200.0)),
    ])
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=2, audit_requirements=True).plan("任务")

    assert [a.stage for a in result.attempts] == ["extract", "audit", "audit"]
    assert result.scenario_ir["engagements"][0]["fire_time_s"] == 1200.0
    assert result.requirements_audit == "unconfirmed"


def test_audit_waits_until_generic_issues_are_cleared() -> None:
    llm = ScriptedLLM([_dumps(_overlapping_ir()), _dumps(_valid_ir()), _REQUIREMENTS_MET])
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=3, audit_requirements=True).plan("任务")
    assert [a.stage for a in result.attempts] == ["extract", "refine", "audit"]
    assert result.requirements_audit == "met"


def test_failed_audit_call_keeps_result_unconfirmed() -> None:
    class TimeoutOnAudit(ScriptedLLM):
        def chat(self, messages: list[dict], **kwargs: Any) -> str:
            if self.calls:
                raise TimeoutError("read timeout")
            return super().chat(messages, **kwargs)

    result = ScenarioIRPlanner(
        llm=TimeoutOnAudit([_dumps(_valid_ir())]), audit_requirements=True
    ).plan("任务")
    assert result.success
    assert result.requirements_audit == "unconfirmed"
    assert result.attempts[-1].stage == "audit"


def test_thought_out_audit_is_nudged_with_sentinel_reminder() -> None:
    llm = FinishReasonLLM(
        [(_dumps(_valid_ir()), "stop"), ("", "length"), (_REQUIREMENTS_MET, "stop")],
        usages=[None, _THOUGHT_OUT_USAGE, None],
    )
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=3, audit_requirements=True).plan("任务")

    assert result.requirements_audit == "met"
    retry = llm.calls[2][-1]["content"]
    assert retry.startswith(llm.calls[1][-1]["content"])
    assert "事实表里的数字都已算好" in retry


async def test_tools_enable_audit_by_default(monkeypatch) -> None:
    seen: list[bool] = []
    result = ScenarioIRPlanner(llm=ScriptedLLM([_dumps(_valid_ir())])).plan("任务")

    def fake_plan(self: ScenarioIRPlanner, *_: Any) -> Any:
        seen.append(self.audit_requirements)
        return result

    monkeypatch.setattr(planner_module, "_make_llm", lambda _: None)
    monkeypatch.setattr(ScenarioIRPlanner, "plan", fake_plan)
    output = json.loads(await NLToAfsimTool().execute(request="任务", include_code=False))
    await NLToAfsimTool().execute(request="任务", audit_requirements=False, include_code=False)

    assert seen == [True, False]
    assert output["requirements_audit"] == ""


@pytest.mark.parametrize("field_name", ["scenario_id", "mission", "selected_coa", "platforms"])
def test_ir_shape_documents_required_fields(field_name: str) -> None:
    llm = ScriptedLLM([_dumps(_valid_ir())])
    ScenarioIRPlanner(llm=llm).plan("任务")
    assert field_name in llm.calls[0][0]["content"]


def _broken_project() -> dict[str, str]:
    files = emit_project(ScenarioIR.model_validate(_valid_ir()))
    name = next(path for path in files if path.startswith("platforms/"))
    files[name] = files[name].replace("end_platform_type", "")
    return files


def test_planner_rejects_invalid_project_despite_valid_single_file(monkeypatch) -> None:
    monkeypatch.setattr(planner_module, "emit_project", lambda _: _broken_project())
    llm = ScriptedLLM([_dumps(_valid_ir())])
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=1).plan("预警机搜索红方雷达")
    assert not result.success
    assert any("platforms/" in error for error in result.attempts[0].errors)


async def test_compile_tool_rejects_invalid_project_without_writing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(planner_module, "emit_project", lambda _: _broken_project())
    root = tmp_path / "new_project"
    output = json.loads(await ScenarioCompileTool().execute(
        scenario_ir_json=_dumps(_valid_ir()), output_path=str(root),
    ))
    assert not output["success"]
    assert output["stage"] == "project_validation"
    assert "platforms/" in output["errors"][0]
    assert not root.exists()


async def test_compile_tool_checks_project_even_without_output_path(monkeypatch) -> None:
    monkeypatch.setattr(planner_module, "emit_project", lambda _: _broken_project())
    output = json.loads(await ScenarioCompileTool().execute(scenario_ir_json=_dumps(_valid_ir())))
    assert not output["success"]


def test_project_writer_rejects_invalid_single_snapshot_before_writing(tmp_path: Path) -> None:
    root = tmp_path / "new_project"
    scenario = ScenarioIR.model_validate(_valid_ir())
    text = AfsimCompiler().compile(scenario).replace("end_mover", "")
    with pytest.raises(CompileError, match="快照"):
        _write_project_output(str(root), scenario, text)
    assert not root.exists()


async def test_compile_tool_writes_validated_project_to_selected_directory(tmp_path: Path) -> None:
    original = tmp_path / "air_8v8"
    original.mkdir()
    marker = original / "main.txt"
    marker.write_text("原始对比稿", encoding="utf-8")
    root = tmp_path / "air_8v8_qoder"
    output = json.loads(await ScenarioCompileTool().execute(
        scenario_ir_json=_dumps(_valid_ir()), output_path=str(root),
    ))
    assert output["success"]
    assert output["project"]["validation"]["valid"]
    assert output["project"]["validation"]["runtime_verified"] is False
    assert output["project"]["validation"]["validator"]["errors"] == 0
    assert Path(output["project"]["entry"]) == root / "main.txt"
    assert (root / "scenario_single.txt").is_file()
    assert marker.read_text(encoding="utf-8") == "原始对比稿"


async def test_nl_tool_rejects_invalid_final_project_after_planning(monkeypatch, tmp_path: Path) -> None:
    # 模拟规划成功后最终 emitter 出错，不能沿用规划阶段的 success=True。
    result = ScenarioIRPlanner(llm=ScriptedLLM([_dumps(_valid_ir())])).plan("预警机搜索红方雷达")
    assert result.success
    monkeypatch.setattr(planner_module, "_make_llm", lambda _: None)
    monkeypatch.setattr(ScenarioIRPlanner, "plan", lambda *_: result)
    monkeypatch.setattr(planner_module, "emit_project", lambda _: _broken_project())
    root = tmp_path / "new_project"
    output = json.loads(await NLToAfsimTool().execute(request="预警机搜索红方雷达", output_path=str(root)))
    assert not output["success"]
    assert output["stage"] == "project_validation"
    assert not root.exists()


async def test_compile_tool_does_not_write_when_validator_reports_errors(monkeypatch, tmp_path: Path) -> None:
    real_validator = planner_module._validator_report

    def invalid_single(text: str) -> dict[str, Any]:
        report = real_validator(text)
        # 单文件带生成器头部，include 展开文本以项目头部开始。
        if text.startswith("# Generated"):
            report["errors"] = 1
        return report

    monkeypatch.setattr(planner_module, "_validator_report", invalid_single)
    root = tmp_path / "new_project"
    output = json.loads(await ScenarioCompileTool().execute(
        scenario_ir_json=_dumps(_valid_ir()), output_path=str(root),
    ))
    assert not output["success"]
    assert not root.exists()


# ── 修订轮补丁 ───────────────────────────────────────────────────


def _patch(body: Any) -> str:
    return _dumps({"__patch__": body})


def _restore_red_radar() -> dict[str, Any]:
    """把 _overlapping_ir 里与蓝方重叠的红方雷达挪回原位。"""
    return {"platforms": [
        {"platform_id": "red_radar", "position": _valid_ir()["platforms"][1]["position"]}
    ]}


def test_patch_merges_objects_and_lists_by_id() -> None:
    base = _single_strike_payload(600.0)
    merged, errors = planner_module.apply_ir_patch(base, {
        "mission": {"objective": "压制红方雷达", "assumptions": ["蓝方 900s 开火"]},
        "selected_coa": {"summary": None},
        "engagements": [{"engagement_id": "strike", "fire_time_s": 900.0}],
        "platforms": [
            {"platform_id": "red_radar", "position": {"altitude_m": 50.0}},
            {"platform_id": "blue_awacs", "route": []},
        ],
    })

    assert errors == []
    assert merged["mission"]["objective"] == "压制红方雷达"
    assert merged["mission"]["duration_s"] == 1800.0
    assert merged["mission"]["assumptions"] == ["蓝方 900s 开火"]
    assert "summary" not in merged["selected_coa"]
    assert merged["engagements"] == [{**base["engagements"][0], "fire_time_s": 900.0}]
    assert [p["platform_id"] for p in merged["platforms"]] == ["blue_awacs", "red_radar"]
    assert merged["platforms"][1]["position"] == {
        "latitude_deg": 27.5, "longitude_deg": 122.5, "altitude_m": 50.0,
    }
    assert merged["platforms"][0]["route"] == []  # 未登记的列表整体替换
    assert merged["platforms"][0]["weapons"] == base["platforms"][0]["weapons"]
    assert base["engagements"][0]["fire_time_s"] == 600.0  # 上一版不被改动


def test_patch_appends_new_ids_and_deletes_marked_ones() -> None:
    base = _duel_payload(_RETURN_FIRE)
    late = _engagement("late", "blue_awacs", "red_radar", "aam", 900.0)
    merged, errors = planner_module.apply_ir_patch(base, {"engagements": [
        {"engagement_id": "strike", "__delete__": True},
        late,
    ]})

    assert errors == []
    assert [e["engagement_id"] for e in merged["engagements"]] == ["reply", "late"]
    assert merged["engagements"][1] == late


@pytest.mark.parametrize(("patch", "message"), [
    ({"engagements": [{"fire_time_s": 900.0}]}, "缺少 engagement_id"),
    ({"engagements": [{"engagement_id": "ghost", "__delete__": True}]}, "ghost 在上一版里不存在"),
    ({"platforms": {"platform_id": "red_radar"}}, "platforms 应该是列表"),
    ({"platforms": [{"platform_id": "blue_awacs", "weapons": [{"quantity": 4}]}]},
     "platforms.weapons 的第 1 个元素缺少 instance_name"),
    (["engagements"], "应该是对象"),
])
def test_ambiguous_patch_is_rejected_whole(patch: Any, message: str) -> None:
    merged, errors = planner_module.apply_ir_patch(_single_strike_payload(), patch)
    assert merged is None
    assert any(message in error for error in errors)


@pytest.mark.parametrize("alias", ["mission_platforms", "mission.platforms"])
def test_patch_folds_nested_list_keys_written_at_the_top(alias: str) -> None:
    """rerun 实测: 模型把合并规则里的 mission.platforms 写成补丁顶层键 mission_platforms。"""
    base = _valid_ir()
    merged, errors = planner_module.apply_ir_patch(base, {
        "mission": {"objective": "压制红方雷达"},
        alias: [{"platform_id": "red_radar", "type_hint": "制导雷达阵地"}],
    })

    assert errors == []
    assert alias not in merged
    assert merged["mission"]["objective"] == "压制红方雷达"
    assert merged["mission"]["platforms"][1] == {
        **base["mission"]["platforms"][1], "type_hint": "制导雷达阵地",
    }
    assert merged["mission"]["platforms"][0] == base["mission"]["platforms"][0]


def test_patch_with_both_nested_list_spellings_is_rejected() -> None:
    merged, errors = planner_module.apply_ir_patch(_valid_ir(), {
        "mission": {"platforms": []},
        "mission_platforms": [{"platform_id": "red_radar", "type_hint": "制导雷达阵地"}],
    })

    assert merged is None
    assert any("mission_platforms" in error and "无法确定" in error for error in errors)


def test_refine_patch_is_merged_onto_the_previous_version() -> None:
    llm = ScriptedLLM([_dumps(_overlapping_ir()), _patch(_restore_red_radar())])
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=2).plan("任务")

    assert result.success and result.advisories == []
    assert [a.patched for a in result.attempts] == [False, True]
    assert result.scenario_ir["platforms"][1]["position"]["longitude_deg"] == 122.5
    assert result.scenario_ir["assertions"] == _valid_ir()["assertions"]
    refine_prompt = llm.calls[1][-1]["content"]
    assert "__patch__" in refine_prompt and "按 id" in refine_prompt


def test_patch_hoists_physical_fields_written_into_the_intents() -> None:
    """rerun 实测: 补丁把物理平台的 position、heading_deg 和航点写进了 mission.platforms。"""
    base = _valid_ir()
    moved = {"latitude_deg": 27.4, "longitude_deg": 121.1, "altitude_m": 9000.0}
    route = [{**_EXTRA_LEG, "heading_deg": 30.0}]
    merged, errors = planner_module.apply_ir_patch(base, {"mission": {"platforms": [
        {"platform_id": "blue_awacs", "position": moved, "heading_deg": 80.0, "route": route},
        {"platform_id": "red_radar", "type_hint": "制导雷达阵地"},
    ]}})

    assert errors == []
    platform = merged["platforms"][0]
    assert (platform["position"], platform["heading_deg"], platform["route"]) == (moved, 80.0, route)
    assert merged["mission"]["platforms"][0] == base["mission"]["platforms"][0]
    assert merged["mission"]["platforms"][1]["type_hint"] == "制导雷达阵地"


def test_null_physical_fields_in_the_intents_are_not_hoisted() -> None:
    """rerun 实测: 上一版报意图里多了 position 后, 补丁在 mission.platforms 里写 null 想删掉它。"""
    base = _valid_ir()
    merged, errors = planner_module.apply_ir_patch(base, {"mission": {"platforms": [
        {"platform_id": "blue_awacs", "position": None, "heading_deg": None},
    ]}})

    assert errors == []
    assert merged["platforms"] == base["platforms"]
    assert merged["mission"]["platforms"] == base["mission"]["platforms"]

    # 上一版意图里真写错了 position 时, null 照常把它删掉
    broken = _valid_ir()
    broken["mission"]["platforms"][0]["position"] = broken["platforms"][0]["position"]
    merged, errors = planner_module.apply_ir_patch(broken, {"mission": {"platforms": [
        {"platform_id": "blue_awacs", "position": None},
    ]}})
    assert errors == [] and merged["mission"]["platforms"] == base["mission"]["platforms"]


def test_patch_moves_plan_fields_written_into_the_mission() -> None:
    """rerun 实测: refine 补丁把方案的 summary、assignments、rationale 写进了 mission。"""
    base = _valid_ir()
    merged, errors = planner_module.apply_ir_patch(base, {"mission": {
        "summary": "新方案", "rationale": ["理由"], "assumptions": ["假设"],
    }})

    assert errors == []
    assert (merged["selected_coa"]["summary"], merged["selected_coa"]["rationale"]) == ("新方案", ["理由"])
    assert merged["mission"] == {**base["mission"], "assumptions": ["假设"]}
    merged, errors = planner_module.apply_ir_patch(base, {
        "mission": {"summary": "甲"}, "selected_coa": {"summary": "乙"},
    })
    assert merged is None and any("无法确定" in error for error in errors)


def test_patch_with_a_physical_field_in_both_places_is_rejected() -> None:
    position = _valid_ir()["platforms"][1]["position"]
    merged, errors = planner_module.apply_ir_patch(_valid_ir(), {
        "mission": {"platforms": [{"platform_id": "red_radar", "position": position}]},
        "platforms": [{"platform_id": "red_radar", "position": position}],
    })

    assert merged is None
    assert any("red_radar" in error and "无法确定" in error for error in errors)


def test_physical_platforms_left_inside_the_mission_are_hoisted() -> None:
    """rerun 实测: 写完意图没闭合 mission 就写物理平台, 后一个 "platforms" 键覆盖了前一个。"""
    position = _valid_ir()["platforms"][1]["position"]
    intent = _dumps([{"platform_id": "red_radar", "initial_position": position}])
    physical = _dumps([{"platform_id": "red_radar", "position": position, "heading_deg": 0.0}])
    patch = f'{{"__patch__": {{"mission": {{"platforms": {intent}, "platforms": {physical}}}}}}}'
    llm = ScriptedLLM([_dumps(_overlapping_ir()), patch])
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=2).plan("任务")

    assert result.success and result.advisories == []
    assert result.scenario_ir["platforms"][1]["position"] == position
    assert result.scenario_ir["mission"]["platforms"][1]["initial_position"] == position


def test_patch_without_ids_is_repaired_against_the_previous_version() -> None:
    position = _valid_ir()["platforms"][1]["position"]
    llm = ScriptedLLM([
        _dumps(_overlapping_ir()),
        _patch({"platforms": [{"position": position}]}),
        _patch(_restore_red_radar()),
    ])
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=3).plan("任务")

    assert [a.stage for a in result.attempts] == ["extract", "refine", "repair"]
    assert "缺少 platform_id" in result.attempts[1].errors[0]
    repair_call = llm.calls[2]
    # 补丁合并不了: 回灌的是上一版, 模型对着它重写补丁
    assert json.loads(repair_call[-2]["content"]) == json.loads(llm.calls[1][-2]["content"])
    assert "缺少 platform_id" in repair_call[-1]["content"]
    assert result.advisories == []


def test_patch_without_a_previous_version_is_rejected() -> None:
    llm = ScriptedLLM([_patch({"scenario_id": "x"}), _dumps(_valid_ir())])
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=2).plan("任务")

    assert result.success
    assert "没有上一版 IR" in result.attempts[0].errors[0]
    assert "没有上一版 IR" in llm.calls[1][-1]["content"]


def test_truncated_refine_rewrite_is_nudged_toward_a_patch() -> None:
    """refine 轮推理没占多少却写到上限: 是整份重抄超了预算, 催它改输出补丁。"""
    llm = FinishReasonLLM(
        [(_dumps(_overlapping_ir()), "stop"), (_PARTIAL_REPLY, "length"),
         (_patch(_restore_red_radar()), "stop")],
        usages=[None, _LONG_IR_USAGE, None],
    )
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=3).plan("任务")

    assert result.success and result.advisories == []
    assert "上限被截断" in result.attempts[1].errors[-1]
    retry = llm.calls[2]
    assert retry[-2] == llm.calls[1][-2]
    assert retry[-1]["content"].startswith(llm.calls[1][-1]["content"])
    assert "只输出改动的补丁" in retry[-1]["content"]
    assert result.attempts[2].patched


def test_audit_revision_can_be_a_patch() -> None:
    llm = ScriptedLLM([
        _dumps(_single_strike_payload(600.0)),
        _patch({"engagements": [{"engagement_id": "strike", "fire_time_s": 900.0}]}),
        _REQUIREMENTS_MET,
    ])
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=3, audit_requirements=True).plan(
        "蓝方 900 秒打击红方雷达"
    )

    assert [a.stage for a in result.attempts] == ["extract", "audit", "audit"]
    assert result.requirements_audit == "met"
    assert result.scenario_ir["engagements"][0]["fire_time_s"] == 900.0
    assert result.attempts[1].patched
    assert "__patch__" in llm.calls[1][-1]["content"]


def test_schema_repair_accepts_a_patch() -> None:
    broken = _valid_ir()
    broken["selected_coa"]["mission_id"] = "mismatched"
    llm = ScriptedLLM([_dumps(broken), _patch({"selected_coa": {"mission_id": "probe"}})])
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=2).plan("任务")

    assert result.success and result.attempts[1].patched
    assert "__patch__" in llm.calls[1][-1]["content"]


# ── 需求数值条款 ─────────────────────────────────────────────────


def _checks(*items: dict[str, Any]) -> str:
    return _dumps({"checks": list(items)})


_LATE_STRIKE = {
    "kind": "fire_window", "shooters": ["blue_awacs"], "start_s": 900, "end_s": 1200,
    "quote": "蓝方 900-1200 秒打击",
}


def test_requirement_checks_are_extracted_once_and_fed_into_refine() -> None:
    llm = ScriptedLLM([
        _dumps(_single_strike_payload(600.0)),
        _checks(_LATE_STRIKE),
        _patch({"engagements": [{"engagement_id": "strike", "fire_time_s": 950.0}]}),
    ])
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=3, check_requirements=True).plan(
        "蓝方 900-1200 秒打击红方雷达"
    )

    assert result.success and result.advisories == []
    assert result.scenario_ir["engagements"][0]["fire_time_s"] == 950.0
    # 条款抽取是单独一次调用, 不算修复轮
    assert [a.stage for a in result.attempts] == ["extract", "refine"]
    assert len(llm.calls) == 3
    extract_call = llm.calls[1]
    assert "条款抽取器" in extract_call[0]["content"]
    assert "蓝方 900-1200 秒打击红方雷达" in extract_call[1]["content"]
    assert "- blue_awacs [blue] 预警机" in extract_call[1]["content"]
    first = result.attempts[0].advisories[0]
    assert first.startswith("需求「蓝方 900-1200 秒打击」未满足") and "600s 开火" in first
    assert "以\"需求\"开头" in llm.calls[2][-1]["content"]
    assert result.requirement_checks == [_LATE_STRIKE]


def test_requirement_check_extraction_failure_is_noted_not_fatal() -> None:
    llm = ScriptedLLM([_dumps(_valid_ir()), "抽不出来"])
    result = ScenarioIRPlanner(llm=llm, check_requirements=True).plan("任务")

    assert result.success and result.rounds == 1
    assert result.requirement_checks == []
    assert any("无法解析" in note for note in result.requirement_notes)


def test_invalid_or_unmatched_checks_are_dropped_with_notes() -> None:
    llm = ScriptedLLM([
        _dumps(_valid_ir()),
        _checks({"kind": "fire_window", "shooters": ["ghost_*"], "end_s": 900}, {"kind": "teleport"}),
    ])
    result = ScenarioIRPlanner(llm=llm, check_requirements=True).plan("任务")

    assert result.success and result.requirement_checks == []
    assert len(result.requirement_notes) == 2


def test_audit_prompt_lists_checks_already_verified_by_code() -> None:
    passed = {
        "kind": "fire_window", "shooters": ["blue_awacs"], "start_s": 500, "end_s": 700,
        "quote": "蓝方 600 秒打击",
    }
    llm = ScriptedLLM([_dumps(_single_strike_payload(600.0)), _checks(passed), _REQUIREMENTS_MET])
    result = ScenarioIRPlanner(llm=llm, audit_requirements=True, check_requirements=True).plan(
        "蓝方 600 秒打击红方雷达"
    )

    assert result.requirements_audit == "met"
    audit_prompt = llm.calls[2][-1]["content"]
    assert "已由代码按当前 IR 核验通过" in audit_prompt
    assert "「蓝方 600 秒打击」" in audit_prompt


def test_extract_thought_out_retry_does_not_use_up_a_repair_round() -> None:
    llm = FinishReasonLLM([
        ("", "length"),
        (_dumps(_overlapping_ir()), "stop"),
        (_patch(_restore_red_radar()), "stop"),
    ])
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=2).plan("任务")

    assert result.success and result.advisories == []
    assert [a.stage for a in result.attempts] == ["extract", "extract", "refine"]


def test_revision_thought_out_retry_does_not_use_up_a_repair_round() -> None:
    """rerun 实测: refine 推理耗尽占掉一轮, 42 条问题最后只修掉 3 条。"""
    llm = FinishReasonLLM(
        [(_dumps(_overlapping_ir()), "stop"), (_PARTIAL_REPLY, "length"),
         (_patch(_restore_red_radar()), "stop")],
        usages=[None, _THOUGHT_OUT_USAGE, None],
    )
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=2).plan("任务")

    assert result.success and result.advisories == []
    assert [a.stage for a in result.attempts] == ["extract", "refine", "refine"]


def test_thought_out_refine_retry_lists_fewer_issues(monkeypatch) -> None:
    """rerun 实测: 56 条问题的 refine 带着同一份清单重问, 又一次耗尽推理预算。"""
    issues = [f"问题 {number}" for number in range(30)]
    monkeypatch.setattr(planner_module, "check_ir_plausibility", lambda _scenario: issues)
    llm = FinishReasonLLM(
        [(_dumps(_valid_ir()), "stop"), (_PARTIAL_REPLY, "length"), (_dumps(_valid_ir()), "stop")],
        usages=[None, _THOUGHT_OUT_USAGE, None],
    )
    ScenarioIRPlanner(llm=llm, max_repair_rounds=2).plan("任务")

    assert len(llm.calls) == 3
    first, retry = llm.calls[1][-1]["content"], llm.calls[2][-1]["content"]
    assert "- 问题 29\n" in first
    assert "- 问题 11\n" in retry and "- 问题 12\n" not in retry
    assert "另有 18 条未列出" in retry and "只输出改动的补丁" in retry


def test_schema_slip_in_the_last_revision_is_still_repaired() -> None:
    """末轮补丁多写一个顶层键被 schema 拒掉时仍补一次 repair, 不把这一轮的修改丢掉。"""
    llm = ScriptedLLM([
        _dumps(_overlapping_ir()),
        _patch({**_restore_red_radar(), "notes": "挪回雷达"}),
        _patch({"notes": None}),
    ])
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=2).plan("任务")

    assert result.success and result.advisories == []
    assert [a.stage for a in result.attempts] == ["extract", "refine", "repair"]
    assert "notes" not in result.scenario_ir


async def test_tools_enable_requirement_checks_by_default(monkeypatch) -> None:
    seen: list[bool] = []
    result = ScenarioIRPlanner(llm=ScriptedLLM([_dumps(_valid_ir())])).plan("任务")

    def fake_plan(self: ScenarioIRPlanner, *_: Any) -> Any:
        seen.append(self.check_requirements)
        return result

    monkeypatch.setattr(planner_module, "_make_llm", lambda _: None)
    monkeypatch.setattr(ScenarioIRPlanner, "plan", fake_plan)
    output = json.loads(await NLToAfsimTool().execute(request="任务", include_code=False))
    await NLToAfsimTool().execute(request="任务", check_requirements=False, include_code=False)

    assert seen == [True, False]
    assert output["requirement_checks"] == [] and output["requirement_notes"] == []


def test_extraction_prompt_maps_phase_windows_to_checks() -> None:
    """实测: 需求只按阶段给时段, 模型就漏抽"至少 3 个交战时段"和"穿越落在哪个阶段"。"""
    prompt = planner_module._REQUIREMENT_CHECKS_PROMPT
    assert "fire_phases 的 windows" in prompt and "reach_area 的 start_s/end_s" in prompt


# ── 漏写闭合符 ───────────────────────────────────────────────────


def test_missing_closers_are_added_only_when_safe() -> None:
    assert _parse_json_object('{"a": {"b": [1]}', close_unbalanced=True) == ({"a": {"b": [1]}}, [])
    assert _parse_json_object('{"a": {"b": [1]}')[0] is None  # 默认不补
    for text in (
        '{"a": {"b": "未闭合',            # 停在字符串中间
        '{"a": {"b": [1',                 # 不以闭合符结尾
        '{"a": {"b": {"c": {"d": 1}',     # 缺得太多
        '{"a": [1}}}',                    # 闭合符多了一个
        '{"a": {"b": {"c": [1}',          # 中途补一个、结尾还缺两个, 超过上限
    ):
        assert _parse_json_object(text, close_unbalanced=True)[0] is None, text


def test_closers_missing_mid_stream_are_added_before_the_mismatch() -> None:
    """rerun 实测: 列表中途漏了 "]" 就写了外层的 "}", 修复轮模型删掉别处的 "}" 凑配对。"""
    text = '{"a": {"b": ["x", "y"}, "c": 1}'
    assert _parse_json_object(text, close_unbalanced=True) == ({"a": {"b": ["x", "y"]}, "c": 1}, [])
    assert _parse_json_object(text)[0] is None  # 默认不补

    ir = _dumps(_valid_ir())
    cut = ir.index("]}")
    llm = FinishReasonLLM([(ir[:cut] + ir[cut + 1:], "stop")])
    result = ScenarioIRPlanner(llm=llm).plan("任务")
    assert result.success and result.attempts[0].ok


def test_closer_written_as_the_wrong_kind_is_replaced() -> None:
    """rerun 实测: extract 在列表末尾把 "]" 写成了 "}", 补闭合符会让顶层提前闭合。"""
    assert _parse_json_object('{"a": [1}}', close_unbalanced=True) == ({"a": [1]}, [])
    text = '{"a": {"b": ["x", "y"}}, "c": 1}'
    assert _parse_json_object(text, close_unbalanced=True) == ({"a": {"b": ["x", "y"]}, "c": 1}, [])
    assert _parse_json_object(text)[0] is None  # 默认不修

    ir = _dumps(_valid_ir())
    cut = ir.index("]}")
    llm = FinishReasonLLM([(ir[:cut] + "}" + ir[cut + 1:], "stop")])
    result = ScenarioIRPlanner(llm=llm).plan("任务")
    assert result.success and result.attempts[0].ok
    assert result.scenario_ir == ScenarioIRPlanner(
        llm=ScriptedLLM([ir])
    ).plan("任务").scenario_ir


def test_finished_patch_missing_its_last_brace_is_merged() -> None:
    """rerun 实测: 核对轮补丁正常收尾却漏了最外层 "}", 整轮被丢。"""
    llm = FinishReasonLLM([
        (_dumps(_overlapping_ir()), "stop"),
        (_patch(_restore_red_radar())[:-1], "stop"),
    ])
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=2).plan("任务")

    assert result.success and result.advisories == []
    assert result.attempts[1].ok and result.attempts[1].patched


def test_truncated_reply_is_not_closed() -> None:
    """截断的输出哪怕恰好停在闭合符后也不补: 后面还有没写出来的内容。"""
    llm = FinishReasonLLM([(_dumps(_valid_ir())[:-1], "length")])
    result = ScenarioIRPlanner(llm=llm).plan("任务")
    assert not result.success
    assert "上限被截断" in result.attempts[0].errors[-1]


# ── 修订后同步意图 ───────────────────────────────────────────────


def _synced_ir() -> dict[str, Any]:
    """_valid_ir 里 blue_awacs 的意图没写航路; 这里补齐, 让意图与物理平台一致。"""
    ir = _valid_ir()
    ir["mission"]["platforms"][0]["route"] = [ir["platforms"][0]["route"][0]["position"]]
    return ir


_EXTRA_LEG = {
    "position": {"latitude_deg": 27.7, "longitude_deg": 121.3, "altitude_m": 9000.0},
    "speed_mps": 200.0,
}


def test_revised_route_is_synced_into_the_intent() -> None:
    revised = _synced_ir()
    revised["platforms"][0]["route"].append(_EXTRA_LEG)
    notes = planner_module._sync_intents(revised)

    assert revised["mission"]["platforms"][0]["route"] == [
        point["position"] for point in revised["platforms"][0]["route"]
    ]
    assert len(notes) == 1 and "blue_awacs" in notes[0]


def test_consistent_intents_are_left_alone() -> None:
    ir = _synced_ir()
    expected = copy.deepcopy(ir)

    assert planner_module._sync_intents(ir) == []
    assert ir == expected


def test_intent_is_synced_even_when_it_already_differed() -> None:
    """rerun 实测: repair 把意图航路缩写成一个点, 下一版补全航路时漏改了挪过的初始位置。

    原先只在上一版两者一致时才同步, 这一版就漏了, 交付的 IR 里意图起点和物理平台对不上。
    """
    ir = _valid_ir()  # blue_awacs 的意图没写航路
    platform = ir["platforms"][0]
    platform["route"].append(_EXTRA_LEG)
    platform["position"] = {**platform["position"], "longitude_deg": 120.8}
    notes = planner_module._sync_intents(ir)

    intent = ir["mission"]["platforms"][0]
    assert intent["initial_position"] == platform["position"]
    assert intent["route"] == [point["position"] for point in platform["route"]]
    assert len(notes) == 1 and "blue_awacs" in notes[0]


@pytest.mark.parametrize("move_platform", [True, False])
def test_intent_copy_errors_are_overwritten_by_the_platform(move_platform: bool) -> None:
    """rerun 实测: 补丁把意图和航路一起重写, 意图里的初始经度抄错了 0.01°。"""
    revised = _synced_ir()
    intent = revised["mission"]["platforms"][0]
    if move_platform:
        revised["platforms"][0]["route"].append(_EXTRA_LEG)
        intent["route"] = [*intent["route"], dict(_EXTRA_LEG["position"])]
    start = intent["initial_position"]
    intent["initial_position"] = {**start, "longitude_deg": start["longitude_deg"] + 0.01}
    notes = planner_module._sync_intents(revised)

    platform = revised["platforms"][0]
    assert intent["initial_position"] == platform["position"]
    assert intent["route"] == [point["position"] for point in platform["route"]]
    assert len(notes) == 1 and "blue_awacs" in notes[0]


def test_extracted_intents_follow_the_platforms() -> None:
    """首版 IR 的意图和物理平台对不上时同样以物理平台为准。"""
    ir = _synced_ir()
    intent = ir["mission"]["platforms"][1]
    intent["initial_position"] = {**intent["initial_position"], "latitude_deg": 27.6}
    result = ScenarioIRPlanner(llm=ScriptedLLM([_dumps(ir)])).plan("任务")

    assert result.success and result.rounds == 1
    synced = result.scenario_ir["mission"]["platforms"][1]
    assert synced["initial_position"] == ir["platforms"][1]["position"]
    assert any("red_radar" in note for note in result.normalizations)


def test_omitted_intent_tracks_are_filled_without_a_note() -> None:
    """提示词让意图不写位置和航路, 由代码按物理平台填写; 这不算对不上。"""
    ir = _synced_ir()
    for intent in ir["mission"]["platforms"]:
        del intent["initial_position"]
        intent.pop("route", None)

    assert planner_module._sync_intents(ir) == []
    assert ir["mission"]["platforms"] == _synced_ir()["mission"]["platforms"][:1] + [
        {**_synced_ir()["mission"]["platforms"][1], "route": []}
    ]
    result = ScenarioIRPlanner(llm=ScriptedLLM([_dumps(ir)])).plan("任务")
    assert result.success and result.normalizations == []


def test_system_prompt_writes_tracks_once_and_asks_for_compact_json() -> None:
    llm = ScriptedLLM([_dumps(_valid_ir())])
    ScenarioIRPlanner(llm=llm).plan("任务")
    system_prompt = llm.calls[0][0]["content"]

    assert '"initial_position":' not in system_prompt
    assert "意图里不写 initial_position 和 route" in system_prompt
    assert "不缩进、不换行" in system_prompt


def test_patched_route_keeps_the_intent_in_step() -> None:
    """rerun 实测: refine 补丁只给 platforms[].route 补了航点, 意图航路少一个点。"""
    ir = _synced_ir()
    ir["platforms"][1]["position"] = dict(ir["platforms"][0]["position"])  # 重叠, 触发 refine
    route = [*_synced_ir()["platforms"][0]["route"], _EXTRA_LEG]
    llm = ScriptedLLM([
        _dumps(ir),
        _patch({"platforms": [
            {"platform_id": "blue_awacs", "route": route},
            *_restore_red_radar()["platforms"],
        ]}),
    ])
    result = ScenarioIRPlanner(llm=llm, max_repair_rounds=2).plan("任务")

    assert result.success and result.advisories == []
    intent = result.scenario_ir["mission"]["platforms"][0]
    assert intent["route"] == [point["position"] for point in route]
    assert any("blue_awacs" in note for note in result.normalizations)
