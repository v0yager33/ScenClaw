"""Evidence-backed registry for the AFSIM subset the compiler may emit."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .provenance import SourceKind, TrustLevel, ValidationStatus


class CapabilityKind(str, Enum):
    SENSOR_TYPE = "sensor_type"
    MOVER_TYPE = "mover_type"
    PROCESSOR_TYPE = "processor_type"
    WEAPON_TYPE = "weapon_type"
    SCRIPT_API = "script_api"
    EVENT_TYPE = "event_type"


@dataclass(frozen=True)
class Evidence:
    source_kind: SourceKind
    relative_path: str
    line: int | None = None
    note: str | None = None


@dataclass(frozen=True)
class Capability:
    capability_id: str
    kind: CapabilityKind
    afsim_symbol: str
    afsim_version: str
    trust_level: TrustLevel
    validation_status: ValidationStatus
    compiler_enabled: bool
    evidence: tuple[Evidence, ...]


class CapabilityRegistry:
    """Read-only lookup used as a hard boundary around compiler output."""

    def __init__(self, capabilities: tuple[Capability, ...]) -> None:
        keys = [(item.kind, item.afsim_symbol) for item in capabilities]
        if len(keys) != len(set(keys)):
            raise ValueError("capability kind/symbol pairs must be unique")
        self._capabilities = capabilities
        self._by_key = {(item.kind, item.afsim_symbol): item for item in capabilities}

    def all(self) -> tuple[Capability, ...]:
        return self._capabilities

    def get(self, kind: CapabilityKind, afsim_symbol: str) -> Capability | None:
        return self._by_key.get((kind, afsim_symbol))

    def is_compiler_enabled(self, kind: CapabilityKind, afsim_symbol: str) -> bool:
        capability = self.get(kind, afsim_symbol)
        return capability is not None and capability.compiler_enabled


DEFAULT_CAPABILITY_REGISTRY = CapabilityRegistry(
    (
        Capability(
            capability_id="sensor.geometric.v1",
            kind=CapabilityKind.SENSOR_TYPE,
            afsim_symbol="WSF_GEOMETRIC_SENSOR",
            afsim_version="2.9.0",
            trust_level=TrustLevel.VERIFIED_RUNTIME,
            validation_status=ValidationStatus.PASSED,
            compiler_enabled=True,
            evidence=(
                Evidence(
                    source_kind=SourceKind.AFSIM_GRAMMAR,
                    relative_path="src/core/wsf/grammar/wsf.ag",
                    line=3610,
                    note="WSF_GEOMETRIC_SENSOR grammar definition",
                ),
                Evidence(
                    source_kind=SourceKind.AFSIM_TEST_SCENARIO,
                    relative_path=(
                        "src/core/wsf_l16/test_mission/"
                        "test_reporting_responsibility.txt"
                    ),
                    line=20,
                    note="AFSIM source-tree usage example",
                ),
            ),
        ),
        Capability(
            capability_id="mover.air.v1",
            kind=CapabilityKind.MOVER_TYPE,
            afsim_symbol="WSF_AIR_MOVER",
            afsim_version="2.9.0",
            trust_level=TrustLevel.VERIFIED_RUNTIME,
            validation_status=ValidationStatus.PASSED,
            compiler_enabled=True,
            evidence=(
                Evidence(
                    source_kind=SourceKind.AFSIM_GRAMMAR,
                    relative_path="src/core/wsf/grammar/wsf.ag",
                    line=3317,
                    note="WSF_AIR_MOVER grammar definition",
                ),
                Evidence(
                    source_kind=SourceKind.AFSIM_TEST_SCENARIO,
                    relative_path=(
                        "src/core/wsf_parser/legacy_test/proxy_test/"
                        "test_add_many_waypoints.txt"
                    ),
                    line=12,
                    note="AFSIM source-tree mover and route example",
                ),
            ),
        ),
        # ── 水面与地面 mover ────────────────────────────────────────
        #
        # 这两个 mover 的想定写法与已验证的 WSF_AIR_MOVER 完全同构:
        # `add mover WSF_X ... end_mover` + 顶层 route 航点, 因此编译风险低。
        # 但**尚未用 mission.exe 实跑验证**, 所以 trust_level 标为
        # UNVERIFIED_SECONDARY、validation_status 标为 NOT_RUN。
        # 依赖它们的想定会被工具输出显式标注为"未经真机验证"。
        #
        # 不放行它们的代价更大: 需求里的舰船/坦克会被静默套上 WSF_AIR_MOVER,
        # 变成"语法全绿但军事上错误"的想定, 比直接报不支持更难发现。
        Capability(
            capability_id="mover.surface.v1",
            kind=CapabilityKind.MOVER_TYPE,
            afsim_symbol="WSF_SURFACE_MOVER",
            afsim_version="2.9.0",
            trust_level=TrustLevel.UNVERIFIED_SECONDARY,
            validation_status=ValidationStatus.NOT_RUN,
            compiler_enabled=True,
            evidence=(
                Evidence(
                    source_kind=SourceKind.THIRD_PARTY_SKILL,
                    relative_path="nanobot/skills/afsim-scenario/afsim-refs/quick-ref.md",
                    line=109,
                    note="WSF_SURFACE_MOVER 列为舰船 mover",
                ),
                Evidence(
                    source_kind=SourceKind.THIRD_PARTY_SKILL,
                    relative_path="nanobot/skills/afsim-scenario/afsim-refs/demos-index.md",
                    line=15,
                    note="舰船类想定 demo 索引指向 WSF_SURFACE_MOVER",
                ),
            ),
        ),
        Capability(
            capability_id="mover.ground.v1",
            kind=CapabilityKind.MOVER_TYPE,
            afsim_symbol="WSF_GROUND_MOVER",
            afsim_version="2.9.0",
            trust_level=TrustLevel.UNVERIFIED_SECONDARY,
            validation_status=ValidationStatus.NOT_RUN,
            compiler_enabled=True,
            evidence=(
                Evidence(
                    source_kind=SourceKind.THIRD_PARTY_SKILL,
                    relative_path="nanobot/skills/afsim-scenario/afsim-refs/quick-ref.md",
                    line=110,
                    note="WSF_GROUND_MOVER 列为地面 mover",
                ),
                Evidence(
                    source_kind=SourceKind.THIRD_PARTY_SKILL,
                    relative_path="nanobot/skills/afsim-scenario/afsim-refs/demos-index.md",
                    line=16,
                    note="地面/发射车类想定 demo 索引指向 WSF_GROUND_MOVER",
                ),
            ),
        ),
        # WSF_SPACE_MOVER 刻意不放行: 它用开普勒根数描述轨道, 与本 IR 的
        # 经纬高 + 航点模型不同构 (见 quick-ref.md:111)。硬套航路会得到
        # 一个轨道语义完全错误的想定, 因此宁可如实报"不支持"。

        Capability(
            capability_id="event.sensor_detection_attempt.v1",
            kind=CapabilityKind.EVENT_TYPE,
            afsim_symbol="SENSOR_DETECTION_ATTEMPT",
            afsim_version="2.9.0",
            trust_level=TrustLevel.VERIFIED_RUNTIME,
            validation_status=ValidationStatus.PASSED,
            compiler_enabled=True,
            evidence=(
                Evidence(
                    source_kind=SourceKind.AFSIM_DOCUMENTATION,
                    relative_path="src/core/wsf/doc/event/sensor_detection_attempt.txt",
                    line=10,
                    note="Official event_output signature",
                ),
                Evidence(
                    source_kind=SourceKind.AFSIM_TEST_SCENARIO,
                    relative_path="src/core/wsf_nx/test_mission/test_stc_signal_processor.txt",
                    line=179,
                    note="Official event_output usage",
                ),
            ),
        ),
        Capability(
            capability_id="event.sensor_track_initiated.v1",
            kind=CapabilityKind.EVENT_TYPE,
            afsim_symbol="SENSOR_TRACK_INITIATED",
            afsim_version="2.9.0",
            trust_level=TrustLevel.VERIFIED_RUNTIME,
            validation_status=ValidationStatus.PASSED,
            compiler_enabled=True,
            evidence=(
                Evidence(
                    source_kind=SourceKind.AFSIM_DOCUMENTATION,
                    relative_path="src/core/wsf/doc/event/sensor_track_initiated.txt",
                    line=10,
                    note="Official event_output signature",
                ),
            ),
        ),
        Capability(
            capability_id="weapon.implicit.v1",
            kind=CapabilityKind.WEAPON_TYPE,
            afsim_symbol="WSF_IMPLICIT_WEAPON",
            afsim_version="2.9.0",
            trust_level=TrustLevel.VERIFIED_RUNTIME,
            validation_status=ValidationStatus.PASSED,
            compiler_enabled=True,
            evidence=(
                Evidence(
                    source_kind=SourceKind.AFSIM_DOCUMENTATION,
                    relative_path="src/core/wsf_mil/doc/wsf_implicit_weapon.rst",
                    line=10,
                    note="Official implicit weapon model documentation",
                ),
                Evidence(
                    source_kind=SourceKind.VERIFIED_LOCAL_SCENARIO,
                    relative_path="examples/validation/implicit_weapon_probe.txt",
                    note="Local mission.exe WEAPON_FIRED validation",
                ),
            ),
        ),
        Capability(
            capability_id="event.weapon_fired.v1",
            kind=CapabilityKind.EVENT_TYPE,
            afsim_symbol="WEAPON_FIRED",
            afsim_version="2.9.0",
            trust_level=TrustLevel.VERIFIED_RUNTIME,
            validation_status=ValidationStatus.PASSED,
            compiler_enabled=True,
            evidence=(
                Evidence(
                    source_kind=SourceKind.AFSIM_DOCUMENTATION,
                    relative_path="src/core/wsf_mil/doc/event/weapon_fired.rst",
                    line=10,
                    note="Official WEAPON_FIRED event format",
                ),
                Evidence(
                    source_kind=SourceKind.VERIFIED_LOCAL_SCENARIO,
                    relative_path="examples/validation/implicit_weapon_probe.txt",
                    note="Local mission.exe WEAPON_FIRED validation",
                ),
            ),
        ),
        Capability(
            capability_id="output.event_pipe.v1",
            kind=CapabilityKind.EVENT_TYPE,
            afsim_symbol="EVENT_PIPE_REPLAY",
            afsim_version="2.9.0",
            trust_level=TrustLevel.VERIFIED_RUNTIME,
            validation_status=ValidationStatus.PASSED,
            compiler_enabled=True,
            evidence=(
                Evidence(
                    source_kind=SourceKind.AFSIM_DOCUMENTATION,
                    relative_path="src/mission/doc/mission.rst",
                    line=32,
                    note="Binary event report for Mystic replay",
                ),
                Evidence(
                    source_kind=SourceKind.AFSIM_TEST_SCENARIO,
                    relative_path="src/core/wsf_nx/test_mission/test_chaff.txt",
                    line=103,
                    note="Official event_pipe file example",
                ),
                Evidence(
                    source_kind=SourceKind.VERIFIED_LOCAL_SCENARIO,
                    relative_path=(
                        "runtime/afsim-scenario-builder/runs/"
                        "82f31d6753644be695c3a3529e3a9816/replay.aer"
                    ),
                    note="mission.exe generated replay opened successfully in Mystic",
                ),
            ),
        ),
    )
)
