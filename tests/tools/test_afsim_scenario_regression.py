"""AFSIM 想定链路回归测试。

覆盖两个方向的正确性:

1. **验证器不得误报** (soundness): 用 ``afsim_project`` 下真实可运行的 AFSIM 想定
   语料做基准。这些文件是 mission.exe 跑得通的, 验证器若把它们判成 error, 说明规则
   本身有问题 —— 而且危害不止于"分数不好看": Agent Loop 会拿着假错误去"修"正确的
   代码, 越修越坏。
2. **确定性编译器输出必须自洽**: ScenarioIR 编译出来的 AFSIM 文本要同时通过
   ``nanobot.afsim.lint_afsim`` 和 ``ScenarioValidator``。两个检查器口径不一致时,
   自修正循环会陷入互相拆台。

真实语料是外部目录, 不在仓库里, 因此缺失时跳过而不是失败。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from nanobot.afsim import AfsimCompiler, ScenarioIR, lint_afsim
from nanobot.afsim.compiler import _latitude, _longitude
from nanobot.agent.tools.scenario_validator_tool import (
    ScenarioValidator,
    _mask_block_comments,
    _parse_coord_values,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_CORPUS = Path(os.environ.get("AFSIM_REAL_CORPUS", REPO_ROOT.parent / "afsim_project"))
OFFICIAL_DEMOS = Path(
    os.environ.get("AFSIM_OFFICIAL_DEMOS", REPO_ROOT.parent / "afsim_verify/examples/demos")
)

# mission.exe 可直接运行的完整想定 (非 include 片段), 要求零 error 零 warning
COMPLETE_SCENARIOS = [
    "air_to_air/1v1.txt",
    "air_to_air/2v2.txt",
    "air_to_air/8v8_adsc.txt",
    "air_to_air/escort.txt",
    "air_to_air/hvaa_dca.txt",
    "air_to_air/16v16_island_chain.txt",
]

requires_corpus = pytest.mark.skipif(
    not REAL_CORPUS.is_dir(), reason="真实 AFSIM 语料不可用"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# ── 验证器 soundness ────────────────────────────────────────────────


@requires_corpus
@pytest.mark.parametrize("relative", COMPLETE_SCENARIOS)
def test_complete_real_scenario_is_clean(relative: str) -> None:
    path = REAL_CORPUS / relative
    if not path.is_file():
        pytest.skip(f"{relative} 不存在")
    result = ScenarioValidator(_read(path)).validate_all()
    assert result.errors == 0, [
        (i.rule_id, i.line_number, i.description) for i in result.issues
    ]
    assert result.score == 100


@requires_corpus
def test_no_block_pairing_false_positives_across_corpus() -> None:
    """整个真实语料 (含 include 片段) 都不该报块配对错误。

    AFSIM 有 120+ 种块关键字、"定义 vs 引用"歧义和单行自闭合块, 早期实现用固定
    关键字白名单做严格栈匹配, 在这份语料上误报了 300 多个 SYN_001/SYN_002。
    """
    offenders: list[tuple[str, str, int]] = []
    for path in sorted(REAL_CORPUS.glob("**/*.txt")):
        result = ScenarioValidator(_read(path)).validate_all()
        offenders.extend(
            (path.name, i.rule_id, i.line_number)
            for i in result.issues
            if i.rule_id in ("SYN_001", "SYN_002")
        )
    assert offenders == []


@requires_corpus
def test_corpus_has_no_errors_at_all() -> None:
    total = 0
    detail: list[tuple[str, str, int]] = []
    for path in sorted(REAL_CORPUS.glob("**/*.txt")):
        result = ScenarioValidator(_read(path)).validate_all()
        for issue in result.issues:
            if issue.severity == "error":
                total += 1
                detail.append((path.name, issue.rule_id, issue.line_number))
    assert total == 0, detail[:20]


# ── 注释处理 ────────────────────────────────────────────────────────


def test_inline_hash_comment_is_legal() -> None:
    """AFSIM 支持 # 行尾注释, 真实想定里大量使用, 不能报成语法错误。"""
    code = """
platform_type FIGHTER WSF_PLATFORM
   weapon fox3 MEDIUM_RANGE_RADAR_MISSILE # fox 3 (MRM)
      quantity 4
   end_weapon
end_platform_type
end_time 100 sec
"""
    result = ScenarioValidator(code).validate_all()
    assert result.errors == 0
    assert not [i for i in result.issues if i.rule_id == "SYN_009"]


def test_line_comment_wins_over_block_comment_open() -> None:
    """``//****//`` 装饰线里含 ``/*`` 子串, 不能被当成块注释开始。"""
    masked = _mask_block_comments(
        [
            "   //*******************************//",
            "platform_type FIGHTER WSF_PLATFORM",
            "end_platform_type",
        ]
    )
    assert "platform_type FIGHTER WSF_PLATFORM" in masked[1]


def test_block_comment_prose_is_not_parsed_as_code() -> None:
    """``/* ... */`` 里的散文会命中 route/platform 等关键字, 必须先剥离。"""
    code = """
/*
route rules means you go through route_rules
a platform object must exist for each track
*/
end_time 100 sec
"""
    result = ScenarioValidator(code).validate_all()
    assert not [i for i in result.issues if i.rule_id in ("SYN_001", "SYN_002")]


# ── 坐标格式 ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected_lat,expected_lon",
    [
        ("27:30:00n 122:30:00e", 27.5, 122.5),
        ("27.5n 122.5e", 27.5, 122.5),
        ("27:30:00s 122:30:00w", -27.5, -122.5),
        ("00:00:36.035n 120e", 0.01, 120.0),
    ],
)
def test_parse_coord_supports_dms_and_decimal(
    text: str, expected_lat: float, expected_lon: float
) -> None:
    parsed = _parse_coord_values(text)
    assert parsed is not None
    assert parsed[0] == pytest.approx(expected_lat, abs=1e-3)
    assert parsed[1] == pytest.approx(expected_lon, abs=1e-3)


def test_six_dof_position_is_not_a_lat_lon_command() -> None:
    """``six_dof_position 0.0833 -1.167`` 是相对偏移, 不是经纬度。"""
    code = "six_dof_position 0.0833 -1.167\nend_time 100 sec\n"
    result = ScenarioValidator(code).validate_all()
    assert not [i for i in result.issues if i.rule_id == "SYN_004"]


def test_genuine_bad_coordinate_still_reported() -> None:
    code = "platform a WSF_PLATFORM\n   position garbage here\nend_platform\nend_time 100 sec\n"
    result = ScenarioValidator(code).validate_all()
    assert [i for i in result.issues if i.rule_id == "SYN_004"]


# ── 真正的错误仍要被抓到 (避免"放宽到失效") ──────────────────────────


def test_missing_end_platform_is_still_an_error() -> None:
    code = "platform alpha WSF_PLATFORM\n   side blue\nend_time 100 sec\n"
    result = ScenarioValidator(code).validate_all()
    assert [i for i in result.issues if i.rule_id == "SYN_002"]


def test_extra_end_platform_is_still_an_error() -> None:
    code = "platform alpha WSF_PLATFORM\nend_platform\nend_platform\nend_time 100 sec\n"
    result = ScenarioValidator(code).validate_all()
    assert [i for i in result.issues if i.rule_id == "SYN_001"]


def test_unregistered_wsf_type_reported_once_per_line_as_warning() -> None:
    """未登记的 WSF_* 只报 warning, 且一行只报一次。

    早期实现把这段检查嵌在 ``for valid_type in _VALID_BASE_TYPES`` 循环里,
    同一个词元会按白名单条数重复报几十次, 直接把分数打到 0。
    """
    code = "platform_type X WSF_TOTALLY_MADE_UP\nend_platform_type\nend_time 100 sec\n"
    result = ScenarioValidator(code).validate_all()
    hits = [i for i in result.issues if i.rule_id == "SYN_003"]
    assert len(hits) == 1
    assert hits[0].severity == "warning"


# ── 确定性编译器与验证器口径一致 ────────────────────────────────────


def _minimal_ir() -> ScenarioIR:
    return ScenarioIR.model_validate(
        {
            "scenario_id": "regression_probe",
            "mission": {
                "mission_id": "probe",
                "name": "probe",
                "objective": "radar detection probe",
                "duration_s": 300.0,
                "platforms": [
                    {
                        "platform_id": "watcher",
                        "side": "blue",
                        "domain": "land",
                        "type_hint": "radar site",
                        "initial_position": {
                            "latitude_deg": 30.0,
                            "longitude_deg": 120.0,
                            "altitude_m": 20.0,
                        },
                    },
                    {
                        "platform_id": "intruder",
                        "side": "red",
                        "domain": "air",
                        "type_hint": "strike aircraft",
                        "initial_position": {
                            "latitude_deg": 30.3,
                            "longitude_deg": 120.0,
                            "altitude_m": 5000.0,
                        },
                    },
                ],
            },
            "selected_coa": {
                "coa_id": "coa1",
                "mission_id": "probe",
                "summary": "watcher detects intruder",
            },
            "sensors": [
                {
                    "sensor_type_id": "SEARCH_RADAR",
                    "maximum_range_m": 120000.0,
                }
            ],
            "platforms": [
                {
                    "platform_id": "watcher",
                    "side": "blue",
                    "position": {
                        "latitude_deg": 30.0,
                        "longitude_deg": 120.0,
                        "altitude_m": 20.0,
                    },
                    "sensors": [
                        {"instance_name": "radar", "sensor_type_id": "SEARCH_RADAR"}
                    ],
                },
                {
                    "platform_id": "intruder",
                    "side": "red",
                    "position": {
                        "latitude_deg": 30.3,
                        "longitude_deg": 120.0,
                        "altitude_m": 5000.0,
                    },
                    "mover_type": "WSF_AIR_MOVER",
                    "route": [
                        {
                            "position": {
                                "latitude_deg": 30.3,
                                "longitude_deg": 120.0,
                                "altitude_m": 5000.0,
                            },
                            "speed_mps": 200.0,
                        },
                        {
                            "position": {
                                "latitude_deg": 30.0,
                                "longitude_deg": 120.0,
                                "altitude_m": 5000.0,
                            },
                            "speed_mps": 200.0,
                        },
                    ],
                },
            ],
        }
    )


def test_compiler_output_passes_both_checkers() -> None:
    text = AfsimCompiler().compile(_minimal_ir())
    lint = lint_afsim(text)
    assert lint.is_valid, lint.diagnostics
    result = ScenarioValidator(text).validate_all()
    assert result.errors == 0, [
        (i.rule_id, i.line_number, i.description) for i in result.issues
    ]


def test_compiler_is_deterministic() -> None:
    ir = _minimal_ir()
    assert AfsimCompiler().compile(ir) == AfsimCompiler().compile(ir)


# ── 坐标必须是 d:m:s ────────────────────────────────────────────────
#
# 这不是风格偏好。afsim-refs/errors-ref.md 的 E005 "Invalid position format" 把
# `position 30.0N 120.0E` 明确列为错误写法; 真实可运行语料里 189 行 position 全是
# d:m:s, 十进制度 0 行。早先编译器输出的正是被点名的错误形式 —— 静态检查全绿, 但
# 真机大概率在解析阶段就失败。

_DMS_POSITION_RE = re.compile(
    r"^position\s+\d{1,3}:\d{2}:\d{2}(?:\.\d+)?[ns]\s+\d{1,3}:\d{2}:\d{2}(?:\.\d+)?[ew]\b"
)
_DECIMAL_POSITION_RE = re.compile(r"^position\s+\d+(?:\.\d+)?[ns]\s")


def _position_lines(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("position ")
    ]


def test_compiler_emits_dms_coordinates() -> None:
    lines = _position_lines(AfsimCompiler().compile(_minimal_ir()))
    assert lines, "编译结果里应当有 position 行"
    for line in lines:
        assert _DMS_POSITION_RE.match(line), line


def test_compiler_never_emits_decimal_degree_coordinates() -> None:
    lines = _position_lines(AfsimCompiler().compile(_minimal_ir()))
    offenders = [line for line in lines if _DECIMAL_POSITION_RE.match(line)]
    assert offenders == [], offenders


@requires_corpus
def test_generated_coordinate_style_matches_real_corpus() -> None:
    """生成的 position 行要能被同一条真实语料模式匹配。

    扫整个语料而不是只扫顶层想定: 顶层文件用 include 把平台定义引进来, position
    行其实都在 platforms/*.txt 里。
    """
    corpus_lines: list[str] = []
    for path in sorted(REAL_CORPUS.glob("**/*.txt")):
        corpus_lines.extend(_position_lines(_read(path)))
    assert corpus_lines, "语料里应当有 position 行"

    # 真实语料自身全部符合 d:m:s 模式 (0 行十进制度)
    decimal = [line for line in corpus_lines if _DECIMAL_POSITION_RE.match(line)]
    assert decimal == [], decimal[:5]
    assert all(_DMS_POSITION_RE.match(line) for line in corpus_lines)

    # 生成结果也必须符合同一模式
    generated = _position_lines(AfsimCompiler().compile(_minimal_ir()))
    assert all(_DMS_POSITION_RE.match(line) for line in generated)


def test_dms_conversion_round_trips() -> None:
    for latitude, longitude in [
        (27.5, 122.5), (30.0, 120.0), (-27.5, -122.5), (0.01, 120.0), (0.0, 0.0),
    ]:
        text = f"{_latitude(latitude)} {_longitude(longitude)}"
        parsed = _parse_coord_values(text)
        assert parsed is not None, text
        assert parsed[0] == pytest.approx(latitude, abs=1e-4)
        assert parsed[1] == pytest.approx(longitude, abs=1e-4)


def test_dms_carries_seconds_overflow() -> None:
    """浮点误差把秒推到 60 时必须进位, 不能写出 30:00:60n 这种非法值。"""
    for value in (29.999999, 59.9999999, 0.9999999):
        for token in (_latitude(value), _longitude(value)):
            seconds = float(token[:-1].split(":")[2])
            minutes = int(token[:-1].split(":")[1])
            assert seconds < 60.0, token
            assert minutes < 60, token


# ── 官方 demo 语料基准 ──────────────────────────────────────────────
#
# 4749 个文件、40+ 领域, 是目前能拿到的最强"已知能跑"证据。检查器一旦在这上面
# 大量误报, 自修正循环就会去改本来正确的代码, 所以基线必须锁住。
#
# 基线是怎么来的: 最初一版在这份语料上报 3693 个 error, 逐条定位根因后降到 89 ——
# side 不是红蓝枚举、坐标有四种写法、脚本体是另一门语言、跨文件引用无法解析。

requires_demos = pytest.mark.skipif(
    not OFFICIAL_DEMOS.is_dir(), reason="官方 demo 语料不可用"
)

# 剩余 error 允许量。48 个来自 satcat.txt (NORAD 目录转储, 平台名真的重复),
# 其余是块跨文件闭合。收紧这个数字前要先确认不是靠放宽规则换来的。
_DEMO_ERROR_BUDGET = 100


@requires_demos
def test_official_demos_false_positive_rate() -> None:
    total = clean = errors = 0
    for path in sorted(OFFICIAL_DEMOS.glob("**/*.txt")):
        result = ScenarioValidator(_read(path)).validate_all()
        total += 1
        errors += result.errors
        if result.errors == 0:
            clean += 1
    assert total > 4000, f"语料文件数异常: {total}"
    assert clean / total >= 0.99, f"0-error 率 {clean}/{total} 低于 99%"
    assert errors <= _DEMO_ERROR_BUDGET, f"error 总数 {errors} 超出预算"


@requires_demos
def test_official_demos_accept_all_coordinate_formats() -> None:
    """四种坐标写法都不该被判为格式错误。

    早先只放行 d:m:s, 在官方语料上误报 323 次 —— 十进制度用了 13280 次。
    """
    samples = [
        "position 30:00:00n 120:00:00e altitude 10000 ft msl",   # d:m:s
        "position 1.283345n 103.859047e altitude 5000 ft",       # 十进制度
        "position 40:30n 121:15w altitude 100 ft",               # d:m
        "position 30n 120e altitude 0 ft",                       # 整数度
        "position 27:30:00n 121:15w altitude 100 ft",            # 混用
        "position -1.2e+07 m -3.4e+06 m -5.0e+06 m",             # 直角/ECI
    ]
    for line in samples:
        code = f"platform p1 WSF_PLATFORM\n   {line}\nend_platform\nend_time 60 sec\n"
        result = ScenarioValidator(code).validate_all()
        bad = [i for i in result.issues if i.rule_id == "SYN_004"]
        assert bad == [], f"{line} -> {[i.description for i in bad]}"


@requires_demos
def test_arbitrary_side_names_are_accepted() -> None:
    """AFSIM 的 side 是任意标识符, 官方语料里有 green/US/ORB/indigo/ITSO 等 20+ 种。"""
    for side in ("blue", "red", "green", "US", "ORB", "indigo", "ITSO", "dark_purple", "Red"):
        code = f"platform p1 WSF_PLATFORM\n   side {side}\nend_platform\nend_time 60 sec\n"
        result = ScenarioValidator(code).validate_all()
        bad = [i for i in result.issues if i.rule_id == "SEM_001"]
        assert bad == [], f"side {side} -> {[i.description for i in bad]}"


def test_script_body_is_not_parsed_as_scenario_commands() -> None:
    """脚本体是类 C 语言, 不能按想定命令解析。

    官方 demo 的 red_sam_battery_tactics.txt 里有 `platform = sub;` 这行脚本,
    早先被当成 platform 块起始, 报出未闭合。
    """
    code = """
processor tactics WSF_SCRIPT_PROCESSOR
   script void Foo()
      WsfPlatform platform = sub;
      if (x) { platform = other; }
   end_script
end_processor
end_time 60 sec
"""
    result = ScenarioValidator(code).validate_all()
    bad = [i for i in result.issues if i.rule_id in ("SYN_001", "SYN_002")]
    assert bad == [], [i.description for i in bad]


def test_platform_field_inside_track_block_is_not_a_block() -> None:
    """``track`` 块内的 ``platform target`` 是字段, 只有两个词元。"""
    code = """
platform shooter WSF_PLATFORM
   track
      platform target
   end_track
end_platform
end_time 60 sec
"""
    result = ScenarioValidator(code).validate_all()
    bad = [i for i in result.issues if i.rule_id in ("SYN_001", "SYN_002")]
    assert bad == [], [i.description for i in bad]


def test_edit_platform_is_still_a_block() -> None:
    """``edit platform NAME`` 也是三个词元, 但确实开块, 不能被上一条规则误杀。"""
    code = "edit platform alpha\n   side blue\nend_platform\nend_time 60 sec\n"
    result = ScenarioValidator(code).validate_all()
    assert [i for i in result.issues if i.rule_id == "SYN_001"] == []


def test_event_output_boolean_field_versus_block() -> None:
    """``event_output file x.evt`` 是块起始; ``event_output true`` 是布尔字段。"""
    as_block = "event_output file out.evt\n   enable WEAPON_FIRED\nend_event_output\nend_time 60 sec\n"
    assert ScenarioValidator(as_block).validate_all().errors == 0

    as_field = "engage_output\n   file fort.6\n   event_output true\nend_engage_output\nend_time 60 sec\n"
    bad = [
        i for i in ScenarioValidator(as_field).validate_all().issues
        if i.rule_id in ("SYN_001", "SYN_002")
    ]
    assert bad == [], [i.description for i in bad]
