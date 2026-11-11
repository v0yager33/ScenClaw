"""Project emitter tests: ScenarioIR → 多文件 AFSIM 项目。

官方 demo 语料 (4749 个 txt, 1121 个目录) 里每个想定都是一个目录: 主入口 + 类型
定义 + 平台/传感器/武器子目录。之前 `AfsimCompiler.compile()` 只产单文件, 不符合
真实工程组织。`emit_project()` / `write_project()` 修这个。

测试重点:

1. **结构**: 必须产出 main.txt / _common.txt / event_output.txt, 以及
   sensors/ weapons/ platforms/ 子目录里每种类型一个文件。
2. **类型 vs 实例分离**: 4 架轰-6K 共享同一个 `platforms/H6K.txt`, 各自在
   main.txt 里是 `platform blue_h6k_1 H6K` / `blue_h6k_2 H6K` 等实例。
3. **include 闭合**: main.txt include 的所有文件都得真实存在 (返回的字典里)。
4. **类型无重复**: 多个平台共享同一 sensor_type_id 时, 类型文件只生成一份。
5. **与单文件编译等价**: 把 project 所有 txt 按 include 顺序拼接, 应该和
   `compile()` 输出语义等价 (允许空白/注释差异)。这是回归测试的判据。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.afsim import (
    AfsimCompiler,
    CompileError,
    ScenarioIR,
    emit_project,
    expand_project,
    lint_afsim,
    lint_project,
    write_project,
)
from nanobot.afsim.schemas import (
    COA,
    GeoPoint,
    MissionSpec,
    PlatformIntent,
    PlatformIR,
    PlatformSensorIR,
    PlatformWeaponIR,
    RoutePointIR,
    ScheduledEngagementIR,
    SensorIR,
    Side,
)


def _minimal_scenario() -> ScenarioIR:
    """构造一个最小合法 ScenarioIR: 1 个平台 + 1 个传感器 + 1 个武器。"""
    mission = MissionSpec(
        mission_id="m1",
        name="test",
        objective="test",
        duration_s=3600.0,
        platforms=[
            PlatformIntent(
                platform_id="blue_j20_1",
                side=Side.BLUE,
                domain="air",
                type_hint="歼-20 战斗机",
                initial_position=GeoPoint(latitude_deg=25.0, longitude_deg=122.0),
            )
        ],
    )
    coa = COA(coa_id="c1", mission_id="m1", summary="test")
    sensor = SensorIR(
        sensor_type_id="J20_RADAR",
        maximum_range_m=200_000.0,
    )
    platform = PlatformIR(
        platform_id="blue_j20_1",
        platform_type="J20",
        side=Side.BLUE,
        position=GeoPoint(latitude_deg=25.0, longitude_deg=122.0, altitude_m=8000.0),
        mover_type="WSF_AIR_MOVER",
        sensors=[PlatformSensorIR(sensor_type_id="J20_RADAR")],
        weapons=[PlatformWeaponIR(instance_name="missile", weapon_type="WSF_IMPLICIT_WEAPON")],
        route=[
            RoutePointIR(
                position=GeoPoint(latitude_deg=26.0, longitude_deg=123.0, altitude_m=8000.0),
                speed_mps=250.0,
            )
        ],
    )
    return ScenarioIR(
        scenario_id="s1",
        mission=mission,
        selected_coa=coa,
        sensors=[sensor],
        platforms=[platform],
    )


def _multi_platform_scenario() -> ScenarioIR:
    """4 架轰-6K 共享同一 platform_type + 同一 sensor_type。用来测类型去重。"""
    platforms = []
    for i in range(1, 5):
        platforms.append(
            PlatformIR(
                platform_id=f"blue_h6k_{i}",
                platform_type="H6K",
                side=Side.BLUE,
                position=GeoPoint(latitude_deg=20.0 + i * 0.1, longitude_deg=115.0),
                mover_type="WSF_AIR_MOVER",
                sensors=[PlatformSensorIR(sensor_type_id="H6K_RADAR")],
                weapons=[PlatformWeaponIR(instance_name="missile", weapon_type="WSF_IMPLICIT_WEAPON")],
                route=[
                    RoutePointIR(
                        position=GeoPoint(latitude_deg=21.0 + i * 0.1, longitude_deg=116.0),
                        speed_mps=200.0,
                    )
                ],
            )
        )
    return ScenarioIR(
        scenario_id="s2",
        mission=MissionSpec(
            mission_id="m2",
            name="multi",
            objective="multi",
            duration_s=3600.0,
            platforms=[
                PlatformIntent(
                    platform_id=p.platform_id,
                    side=Side.BLUE,
                    domain="air",
                    type_hint="轰-6K",
                    initial_position=p.position,
                )
                for p in platforms
            ],
        ),
        selected_coa=COA(coa_id="c2", mission_id="m2", summary="multi"),
        sensors=[SensorIR(sensor_type_id="H6K_RADAR", maximum_range_m=150_000.0)],
        platforms=platforms,
    )


# ── 结构 ─────────────────────────────────────────────────────────────


def test_emit_project_has_required_files() -> None:
    files = emit_project(_minimal_scenario())
    required = {"main.txt", "_common.txt", "event_output.txt", "output/.gitkeep"}
    assert required.issubset(files.keys()), f"missing: {required - files.keys()}"
    # 子目录里必须有类型文件
    assert any(k.startswith("sensors/") for k in files)
    assert any(k.startswith("weapons/") for k in files)
    assert any(k.startswith("platforms/") for k in files)


def test_emit_project_sensor_type_file_content() -> None:
    files = emit_project(_minimal_scenario())
    sensor_file = files["sensors/J20_RADAR.txt"]
    # 必须是合法的 AFSIM sensor 定义
    assert "sensor J20_RADAR WSF_GEOMETRIC_SENSOR" in sensor_file
    assert "mode_template" in sensor_file
    assert "end_sensor" in sensor_file
    assert "maximum_range 200000 m" in sensor_file


def test_emit_project_platform_type_file_content() -> None:
    files = emit_project(_minimal_scenario())
    # 类型文件名现在按 mover 分家: J20 + WSF_AIR_MOVER -> J20_AIR
    platform_file = files["platforms/J20_AIR.txt"]
    # 必须是合法的 platform_type 定义
    assert "platform_type J20_AIR WSF_PLATFORM" in platform_file
    assert "mover WSF_AIR_MOVER\n   end_mover" in platform_file
    assert "end_platform_type" in platform_file
    assert "constant" not in platform_file
    assert "end_radar_signature" not in platform_file
    # 类型定义里不该有实例层面的东西 (位置/航路)
    assert "position" not in platform_file
    assert "route" not in platform_file


def test_emit_project_main_has_instances_not_types() -> None:
    files = emit_project(_minimal_scenario())
    main = files["main.txt"]
    # main.txt 里应该有平台实例 (platform ID TYPE_KEY, 其中 TYPE_KEY 按 mover 分家)
    assert "platform blue_j20_1 J20_AIR" in main
    # 实例层面的属性: side / position / sensor / weapon / route
    assert "side blue" in main
    assert "position" in main
    assert "add sensor radar J20_RADAR" in main
    assert "add weapon missile WSF_IMPLICIT_WEAPON" in main
    assert "route" in main
    # main.txt 里不该重复定义类型 (那是 platforms/ 文件的事)
    assert "platform_type J20_AIR" not in main
    assert "sensor J20_RADAR WSF_GEOMETRIC_SENSOR" not in main


def test_emit_project_common_aggregates_includes() -> None:
    files = emit_project(_minimal_scenario())
    common = files["_common.txt"]
    assert "include_once sensors/J20_RADAR.txt" in common
    assert "include_once weapons/WSF_IMPLICIT_WEAPON.txt" in common
    assert "include_once platforms/J20_AIR.txt" in common


def test_emit_project_main_includes_common() -> None:
    files = emit_project(_minimal_scenario())
    main = files["main.txt"]
    assert "file_path ." in main
    assert "include_once _common.txt" in main
    assert "include_once event_output.txt" in main
    assert "end_time 3600 sec" in main


# ── 类型去重 ─────────────────────────────────────────────────────────


def test_shared_sensor_type_written_once() -> None:
    """4 架轰-6K 共享 H6K_RADAR, 类型文件只该有一份。"""
    files = emit_project(_multi_platform_scenario())
    sensor_files = [k for k in files if k.startswith("sensors/")]
    assert sensor_files == ["sensors/H6K_RADAR.txt"]


def test_shared_platform_type_written_once() -> None:
    """4 架轰-6K 共享 H6K 基型 + 同 mover, platforms/ 下只该有一个文件。"""
    files = emit_project(_multi_platform_scenario())
    platform_files = [k for k in files if k.startswith("platforms/")]
    # H6K + WSF_AIR_MOVER -> H6K_AIR
    assert platform_files == ["platforms/H6K_AIR.txt"]


def test_all_instances_appear_in_main() -> None:
    """4 个实例都该在 main.txt 里各自出现, 引用正确的类型键。"""
    files = emit_project(_multi_platform_scenario())
    main = files["main.txt"]
    for i in range(1, 5):
        assert f"platform blue_h6k_{i} H6K_AIR" in main


def test_different_movers_split_into_separate_files() -> None:
    """同一基型 + 不同 mover 必须拆成多个类型文件, 否则所有平台会共享首个 mover。

    这是上一版 project emitter 的 bug: 4 架轰-6K (WSF_AIR_MOVER) 排第一,
    红方巡洋舰 (WSF_SURFACE_MOVER) 和岸基雷达 (无 mover) 都被错误地写进
    WSF_PLATFORM_AIR.txt。修复后应该按 (platform_type, mover_type) 分家。
    """
    from nanobot.afsim.schemas import (
        COA,
        GeoPoint,
        MissionSpec,
        PlatformIntent,
        PlatformIR,
        PlatformSensorIR,
        RoutePointIR,
        SensorIR,
        Side,
    )

    # 构造 3 个平台: 同一基型 WSF_PLATFORM, 不同 mover
    platforms = [
        PlatformIR(
            platform_id="blue_h6k_1",
            platform_type="WSF_PLATFORM",
            side=Side.BLUE,
            position=GeoPoint(latitude_deg=20.0, longitude_deg=115.0, altitude_m=8000.0),
            mover_type="WSF_AIR_MOVER",
            sensors=[PlatformSensorIR(sensor_type_id="R1")],
            route=[RoutePointIR(
                position=GeoPoint(latitude_deg=21.0, longitude_deg=116.0, altitude_m=8000.0),
                speed_mps=200.0,
            )],
        ),
        PlatformIR(
            platform_id="red_cg_1",
            platform_type="WSF_PLATFORM",
            side=Side.RED,
            position=GeoPoint(latitude_deg=18.0, longitude_deg=113.0),
            mover_type="WSF_SURFACE_MOVER",
            sensors=[PlatformSensorIR(sensor_type_id="R1")],
            route=[RoutePointIR(
                position=GeoPoint(latitude_deg=19.0, longitude_deg=114.0),
                speed_mps=10.0,
            )],
        ),
        PlatformIR(
            platform_id="red_radar_1",
            platform_type="WSF_PLATFORM",
            side=Side.RED,
            position=GeoPoint(latitude_deg=16.0, longitude_deg=112.0, altitude_m=50.0),
            mover_type=None,   # 固定设施
            sensors=[PlatformSensorIR(sensor_type_id="R1")],
        ),
    ]
    scenario = ScenarioIR(
        scenario_id="s3",
        mission=MissionSpec(
            mission_id="m3", name="split", objective="split", duration_s=3600.0,
            platforms=[
                PlatformIntent(
                    platform_id=p.platform_id,
                    side=p.side,
                    # 每个平台的域必须与其 mover 一致, 否则 schema 会拒绝
                    domain=(
                        "air" if p.mover_type == "WSF_AIR_MOVER"
                        else "sea" if p.mover_type == "WSF_SURFACE_MOVER"
                        else "land"
                    ),
                    type_hint="x",
                    initial_position=p.position,
                )
                for p in platforms
            ],
        ),
        selected_coa=COA(coa_id="c3", mission_id="m3", summary="split"),
        sensors=[SensorIR(sensor_type_id="R1", maximum_range_m=100_000.0)],
        platforms=platforms,
    )

    files = emit_project(scenario)
    platform_files = sorted(k for k in files if k.startswith("platforms/"))
    # 必须拆成 3 个文件: AIR / SURFACE / 无 mover
    assert platform_files == [
        "platforms/WSF_PLATFORM_AIR.txt",      # 空中
        "platforms/WSF_PLATFORM_STATIC.txt",   # 无 mover (固定)
        "platforms/WSF_PLATFORM_SURFACE.txt",  # 水面
    ]

    # 每个文件里的 mover 必须正确
    assert "mover WSF_AIR_MOVER" in files["platforms/WSF_PLATFORM_AIR.txt"]
    assert "mover WSF_SURFACE_MOVER" in files["platforms/WSF_PLATFORM_SURFACE.txt"]
    # 固定设施文件里不该有 mover 块
    assert "mover" not in files["platforms/WSF_PLATFORM_STATIC.txt"]
    assert "platform_type WSF_PLATFORM_STATIC WSF_PLATFORM" in files["platforms/WSF_PLATFORM_STATIC.txt"]

    # main.txt 里每个实例引用正确的类型键
    main = files["main.txt"]
    assert "platform blue_h6k_1 WSF_PLATFORM_AIR" in main
    assert "platform red_cg_1 WSF_PLATFORM_SURFACE" in main
    assert "platform red_radar_1 WSF_PLATFORM_STATIC" in main
    single = AfsimCompiler().compile(scenario)
    assert lint_project(files).is_valid
    assert lint_afsim(single).is_valid
    for platform_id, key in (
        ("blue_h6k_1", "WSF_PLATFORM_AIR"),
        ("red_cg_1", "WSF_PLATFORM_SURFACE"),
        ("red_radar_1", "WSF_PLATFORM_STATIC"),
    ):
        assert f"platform {platform_id} {key}" in single


# ── include 闭合 ─────────────────────────────────────────────────────


def test_all_included_files_exist() -> None:
    """main.txt / _common.txt include 的每个文件都必须在返回的字典里。"""
    files = emit_project(_multi_platform_scenario())
    main = files["main.txt"]
    common = files["_common.txt"]
    for line in (main + "\n" + common).splitlines():
        line = line.strip()
        if line.startswith("include_once "):
            included = line.split()[1]
            assert included in files, f"included file {included!r} not in project"


# ── 写盘 ─────────────────────────────────────────────────────────────


def test_write_project_creates_directory_tree(tmp_path: Path) -> None:
    root = tmp_path / "my_scenario"
    written = write_project(_minimal_scenario(), root)
    assert root.is_dir()
    assert (root / "main.txt").is_file()
    assert (root / "_common.txt").is_file()
    assert (root / "sensors" / "J20_RADAR.txt").is_file()
    assert (root / "weapons" / "WSF_IMPLICIT_WEAPON.txt").is_file()
    # 类型文件名现在按 mover 分家: J20 + WSF_AIR_MOVER -> J20_AIR
    assert (root / "platforms" / "J20_AIR.txt").is_file()
    assert (root / "event_output.txt").is_file()
    assert (root / "output" / ".gitkeep").is_file()
    # 返回的字典里每个路径都得真实存在
    for rel, full in written.items():
        assert full.is_file(), f"{rel} -> {full} missing"


def test_write_project_overwrites_existing(tmp_path: Path) -> None:
    """重新生成想定就该覆盖旧文件 (这是预期行为)。"""
    root = tmp_path / "my_scenario"
    root.mkdir()
    (root / "main.txt").write_text("stale content", encoding="utf-8")
    write_project(_minimal_scenario(), root)
    assert "stale content" not in (root / "main.txt").read_text(encoding="utf-8")


# ── 与单文件编译的语义等价 ───────────────────────────────────────────


def test_project_and_single_file_compile_same_scenario(tmp_path: Path) -> None:
    """project 里所有 txt 按 include 顺序拼接, 应该和单文件编译语义等价。

    这里不要求字节级相等 (空白/注释允许差异), 只要求关键构造都出现:
    - 每个 platform_id 都出现
    - 每个 sensor_type_id 都出现
    - weapon_effects 定义出现
    - end_time 值一致
    """
    scenario = _minimal_scenario()
    single = AfsimCompiler().compile(scenario)
    files = emit_project(scenario)
    concatenated, _, errors = expand_project(files)
    assert not errors
    assert lint_project(files).is_valid
    assert lint_afsim(single).is_valid

    # 关键构造都得在。注意单文件模板把 weapon_effects 命名为 AUTO_ENGAGEMENT_EFFECT,
    # 项目也沿用这个名字 (与武器类型名 WSF_IMPLICIT_WEAPON 不同概念)。
    # 项目里类型文件名按 mover 分家: J20 + WSF_AIR_MOVER -> J20_AIR
    for token in (
        "platform blue_j20_1 J20_AIR",
        "sensor J20_RADAR WSF_GEOMETRIC_SENSOR",
        "end_time 3600 sec",
        "side blue",
        "mover WSF_AIR_MOVER",
        "maximum_range 200000 m",
    ):
        assert token in concatenated, f"project missing {token!r}"
        assert token in single, f"single file missing {token!r}"

    # 有武器但没有 engagements 时，两种输出也必须定义效应。
    effect = "weapon_effects AUTO_ENGAGEMENT_EFFECT WSF_SPHERICAL_LETHALITY"
    assert single.count(effect) == concatenated.count(effect) == 1
    assert "add mover" not in single

    def commands(text: str) -> list[str]:
        return [
            line.strip().replace("file output/", "file ")
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith(("#", "//", "file_path", "log_file"))
        ]

    assert commands(single) == commands(concatenated)

    # 单文件里出现 1 次的, project 里也得出现 1 次 (类型定义不重复)
    assert single.count("sensor J20_RADAR WSF_GEOMETRIC_SENSOR") == 1
    assert concatenated.count("sensor J20_RADAR WSF_GEOMETRIC_SENSOR") == 1


def test_project_main_passes_validator_on_its_own() -> None:
    """main.txt 单独过验证器不该报"类型未定义"的错 —— 因为 validator 对
    带 include 的文件降级为 warning (CON_001)。这是预期行为, 与旧语料一致。
    """
    from nanobot.agent.tools.scenario_validator_tool import ScenarioValidator

    files = emit_project(_minimal_scenario())
    validator = ScenarioValidator(files["main.txt"])
    result = validator.validate_all()
    errors = [i for i in result.issues if i.severity == "error"]
    # 不该有"类型未定义"的硬错 (CON_001 在带 include 时降级为 warning)
    assert not any("undefined type" in i.description.lower() for i in errors)


def test_common_defines_shared_signature_before_includes() -> None:
    files = emit_project(_minimal_scenario())
    common = files["_common.txt"]
    definition = "radar_signature 10dB_FuzzBall WSF_RADAR_SIGNATURE"
    assert common.count(definition) == 1
    assert common.index("end_radar_signature") < common.index("include_once")
    assert AfsimCompiler().compile(_minimal_scenario()).count(definition) == 1


@pytest.mark.parametrize("end", ["end_platform_type", "end_mover"])
def test_project_detects_unclosed_included_type_with_source(end: str) -> None:
    files = emit_project(_minimal_scenario())
    name = "platforms/J20_AIR.txt"
    files[name] = files[name].replace(end, "")
    result = lint_project(files)
    assert not result.is_valid
    assert any(d.file == name and d.line for d in result.diagnostics)


@pytest.mark.parametrize("missing", ["sensors/J20_RADAR.txt", "platforms/J20_AIR.txt"])
def test_project_rejects_missing_transitive_include(missing: str) -> None:
    files = emit_project(_minimal_scenario())
    del files[missing]
    result = lint_project(files)
    assert not result.is_valid
    assert any(d.code == "include.missing" and d.file == "_common.txt" for d in result.diagnostics)


def test_project_expands_include_once_only_once() -> None:
    files = emit_project(_minimal_scenario())
    files["main.txt"] += "\ninclude_once _common.txt\n"
    assert lint_project(files).is_valid
    text, _, errors = expand_project(files)
    assert not errors
    assert text.count("platform_type J20_AIR WSF_PLATFORM") == 1


def test_project_rejects_include_cycles_and_unsafe_paths() -> None:
    files = emit_project(_minimal_scenario())
    files["_common.txt"] += "\ninclude main.txt\n"
    assert any(d.code == "include.cycle" for d in lint_project(files).diagnostics)
    files["_common.txt"] = "include_once ../other.txt\n"
    assert any(d.code == "include.invalid_path" for d in lint_project(files).diagnostics)


@pytest.mark.parametrize("mutation,code", [
    (lambda text: text.replace(" WSF_PLATFORM", ""), "platform_type.missing_parent"),
    (lambda text: text.replace("10dB_FuzzBall", "UNKNOWN_SIG"), "radar_signature.undefined"),
    (lambda text: text + text, "platform_type.duplicate"),
])
def test_project_rejects_bad_type_declarations(mutation, code: str) -> None:
    files = emit_project(_minimal_scenario())
    name = "platforms/J20_AIR.txt"
    files[name] = mutation(files[name])
    assert any(d.code == code for d in lint_project(files).diagnostics)


def test_project_rejects_signature_definition_reference_mix() -> None:
    files = emit_project(_minimal_scenario())
    name = "platforms/J20_AIR.txt"
    files[name] = files[name].replace(
        "   radar_signature 10dB_FuzzBall",
        "   radar_signature 10dB_FuzzBall\n      constant 10 dBsm\n   end_radar_signature",
    )
    result = lint_project(files)
    assert not result.is_valid
    assert any(d.code == "radar_signature.invalid_constant" for d in result.diagnostics)


def test_project_rejects_missing_weapon_effects_definition() -> None:
    files = emit_project(_minimal_scenario())
    files["weapons/WSF_IMPLICIT_WEAPON.txt"] = ""
    result = lint_project(files)
    assert any(d.code == "weapon_effects.undefined" for d in result.diagnostics)


def test_project_rejects_undefined_wsf_prefixed_custom_type() -> None:
    files = emit_project(_minimal_scenario())
    files["main.txt"] = files["main.txt"].replace(" J20_AIR", " WSF_PLATFORM_MISSING")
    assert any(d.code == "platform.unknown_type" for d in lint_project(files).diagnostics)


def test_invalid_project_does_not_overwrite_existing_files(tmp_path: Path) -> None:
    root = tmp_path / "scenario"
    root.mkdir()
    original = root / "main.txt"
    original.write_text("original", encoding="utf-8")
    files = emit_project(_minimal_scenario())
    files["platforms/J20_AIR.txt"] = "platform_type J20_AIR\n"
    with pytest.raises(CompileError):
        write_project(_minimal_scenario(), root, files=files)
    assert original.read_text(encoding="utf-8") == "original"
    assert list(root.iterdir()) == [original]


def test_project_write_stays_in_selected_directory(tmp_path: Path) -> None:
    original = tmp_path / "air_8v8"
    original.mkdir()
    marker = original / "main.txt"
    marker.write_text("original", encoding="utf-8")
    write_project(_minimal_scenario(), tmp_path / "air_8v8_qoder")
    assert marker.read_text(encoding="utf-8") == "original"
    assert list(original.iterdir()) == [marker]


def test_project_rejects_symlink_escape_before_writing(tmp_path: Path) -> None:
    root = tmp_path / "scenario"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "platforms").symlink_to(outside, target_is_directory=True)
    with pytest.raises(CompileError, match="越界"):
        write_project(_minimal_scenario(), root)
    assert list(outside.iterdir()) == []
    assert not (root / "main.txt").exists()


@pytest.mark.parametrize("path", ["../outside.txt", "/tmp/outside.txt", "C:/outside.txt"])
def test_project_rejects_output_path_traversal(path: str) -> None:
    files = emit_project(_minimal_scenario())
    files[path] = ""
    assert any(d.code == "project.invalid_path" for d in lint_project(files).diagnostics)


def test_type_key_collision_is_rejected() -> None:
    scenario = _multi_platform_scenario()
    scenario.platforms[1].platform_type = "H6K_AIR"
    scenario.platforms[1].mover_type = None
    scenario.platforms[1].route = []
    with pytest.raises(CompileError, match="collision"):
        AfsimCompiler().compile(scenario)
    with pytest.raises(CompileError, match="collision"):
        emit_project(scenario)


def test_disabled_sensor_is_explicitly_off_in_both_outputs() -> None:
    scenario = _minimal_scenario()
    scenario.platforms[0].sensors[0].enabled = False
    assert "      off" in AfsimCompiler().compile(scenario)
    assert "      off" in emit_project(scenario)["main.txt"]


def test_air8v8_has_complete_types_routes_and_original_fire_times() -> None:
    scenario = _minimal_scenario()
    prototype = scenario.platforms[0]
    intent = scenario.mission.platforms[0]
    platforms = []
    intents = []
    engagements = []
    for side, fire_time in ((Side.BLUE, 300), (Side.RED, 330)):
        enemy = "red" if side == Side.BLUE else "blue"
        for group in (1, 2):
            for slot in (1, 2, 3, 4):
                name = f"{side.value}_j16_{group}_{slot}"
                platform = prototype.model_copy(deep=True, update={
                    "platform_id": name, "side": side,
                    "platform_type": "J16" if side == Side.BLUE else "WSF_PLATFORM",
                })
                platform.weapons[0].instance_name = "mrm"
                platform.weapons[0].quantity = 4
                platforms.append(platform)
                intents.append(intent.model_copy(deep=True, update={"platform_id": name, "side": side}))
                engagements.append(ScheduledEngagementIR(
                    engagement_id=f"fire_{name}", shooter_platform=name,
                    target_platform=f"{enemy}_j16_{group}_{slot}",
                    weapon_name="mrm", fire_time_s=fire_time,
                ))
    scenario.platforms = platforms
    scenario.mission.platforms = intents
    scenario.engagements = engagements
    scenario = ScenarioIR.model_validate(scenario.model_dump())
    files = emit_project(scenario)
    single = AfsimCompiler().compile(scenario)
    assert lint_project(files).is_valid
    assert lint_afsim(single).is_valid
    for text in (files["main.txt"], single):
        assert sum(line.startswith("platform ") for line in text.splitlines()) == 16
        assert text.count("   route\n") == text.count("   end_route") == 16
        assert text.count("execute at_time 300 sec absolute") == 8
        assert text.count("execute at_time 330 sec absolute") == 8
        assert text.count("quantity 4") == 16
    assert set(path for path in files if path.startswith("platforms/")) == {
        "platforms/J16_AIR.txt", "platforms/WSF_PLATFORM_AIR.txt",
    }
    for name in ("J16_AIR", "WSF_PLATFORM_AIR"):
        assert f"platform_type {name} WSF_PLATFORM" in single
        assert files[f"platforms/{name}.txt"].count("end_platform_type") == 1
        assert files[f"platforms/{name}.txt"].count("end_mover") == 1


@pytest.mark.parametrize("end", ["end_weapon", "end_weapon_effects", "end_execute"])
def test_linter_detects_missing_weapon_and_script_closures(end: str) -> None:
    scenario = _multi_platform_scenario()
    scenario.engagements = [ScheduledEngagementIR(
        engagement_id="fire_1", shooter_platform="blue_h6k_1", target_platform="blue_h6k_2",
        weapon_name="missile", fire_time_s=300,
    )]
    text = AfsimCompiler().compile(scenario)
    assert lint_afsim(text).is_valid
    result = lint_afsim(text.replace(end, "", 1))
    assert not result.is_valid


def test_shared_effect_is_defined_once_for_different_weapon_types() -> None:
    scenario = _multi_platform_scenario()
    scenario.platforms[1].weapons[0].weapon_type = "WSF_EXPLICIT_WEAPON"
    files = emit_project(scenario)
    text, _, errors = expand_project(files)
    assert not errors
    assert text.count("weapon_effects AUTO_ENGAGEMENT_EFFECT WSF_SPHERICAL_LETHALITY") == 1
    assert lint_project(files).is_valid


def test_fire_script_skips_target_already_removed_from_simulation() -> None:
    # 即时毁伤会删除目标; 两个射手攻击同一目标时, 后一发不能对空引用 MakeTrack。
    scenario = _multi_platform_scenario()
    scenario.engagements = [
        ScheduledEngagementIR(
            engagement_id=f"fire_{shooter}", shooter_platform=shooter,
            target_platform="blue_h6k_2", weapon_name="missile", fire_time_s=fire_time,
        )
        for shooter, fire_time in (("blue_h6k_1", 300), ("blue_h6k_3", 330))
    ]
    files = emit_project(scenario)
    single = AfsimCompiler().compile(scenario)
    assert lint_project(files).is_valid
    assert lint_afsim(single).is_valid
    guarded = (
        '      WsfPlatform target_plat = WsfSimulation.FindPlatform("blue_h6k_2");\n'
        "      if (target_plat.IsValid())\n"
        "      {\n"
        '         Weapon("missile").Fire(target_plat.MakeTrack());\n'
        "      }\n"
        "   end_execute"
    )
    for text in (files["main.txt"], single):
        assert "FindPlatform(\"blue_h6k_2\").MakeTrack()" not in text
        assert text.count(guarded) == 2
