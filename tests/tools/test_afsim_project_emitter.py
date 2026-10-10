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


from nanobot.afsim import (
    AfsimCompiler,
    ScenarioIR,
    emit_project,
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
    assert "platform_type J20_AIR" in platform_file
    # mover 在 platform_type 里是单行引用 (与官方 demo 一致), 不带 end_mover
    assert "mover WSF_AIR_MOVER" in platform_file
    assert "end_mover" not in platform_file
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
        COA, GeoPoint, MissionSpec, PlatformIntent, PlatformIR,
        PlatformSensorIR, RoutePointIR, SensorIR, Side,
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
        "platforms/WSF_PLATFORM.txt",          # 无 mover (固定)
        "platforms/WSF_PLATFORM_AIR.txt",      # 空中
        "platforms/WSF_PLATFORM_SURFACE.txt",  # 水面
    ]

    # 每个文件里的 mover 必须正确
    assert "mover WSF_AIR_MOVER" in files["platforms/WSF_PLATFORM_AIR.txt"]
    assert "mover WSF_SURFACE_MOVER" in files["platforms/WSF_PLATFORM_SURFACE.txt"]
    # 固定设施文件里不该有 mover 块
    assert "mover" not in files["platforms/WSF_PLATFORM.txt"]

    # main.txt 里每个实例引用正确的类型键
    main = files["main.txt"]
    assert "platform blue_h6k_1 WSF_PLATFORM_AIR" in main
    assert "platform red_cg_1 WSF_PLATFORM_SURFACE" in main
    assert "platform red_radar_1 WSF_PLATFORM" in main


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
    concatenated = "\n".join(files[k] for k in sorted(files))

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

    # weapon_effects: 两边都引用 AUTO_ENGAGEMENT_EFFECT; 项目里还必须有定义块
    # (单文件模板有个潜伏的 bug: 只引用不定义, 真机跑会报 undefined。这里不修它,
    # 只断言 project 比单文件更完整)
    assert "weapon_effects AUTO_ENGAGEMENT_EFFECT" in single
    assert "weapon_effects AUTO_ENGAGEMENT_EFFECT WSF_SPHERICAL_LETHALITY" in concatenated

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
