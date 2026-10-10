"""确定性 AFSIM 编译后端 — ScenarioIR → AFSIM 2.9 文本。

本包是 ScenClaw 里唯一被允许直接产出 AFSIM 语法的组件。它刻意做得很小:

  ScenarioIR JSON
    → Pydantic 强类型校验 (schemas)
    → 能力白名单检查 (capabilities)
    → Jinja2 模板确定性渲染 (compiler)
    → AFSIM 文本
    → 静态检查 (linter)

关键性质: 同一个 ScenarioIR 输入必然得到同一份 AFSIM 输出, 编译过程不调用大模型。
这样 LLM 的不确定性被限制在"自然语言 → ScenarioIR"这一步, 只要 IR 通过校验,
落到 AFSIM 文本的映射就是可复现、可审计的。

上游 (LLM 负责): 自然语言 → MissionSpec / COA / ScenarioIR
本包 (确定性):   ScenarioIR → AFSIM 文本
"""

from __future__ import annotations

from nanobot.afsim.capabilities import (
    DEFAULT_CAPABILITY_REGISTRY,
    Capability,
    CapabilityKind,
    CapabilityRegistry,
    Evidence,
)
from nanobot.afsim.compiler import AfsimCompiler, CompileError, compile_scenario
from nanobot.afsim.project_emitter import emit_project, write_project
from nanobot.afsim.geometry import (
    GeometryFinding,
    check_geometry,
    ground_distance_m,
    min_separation_m,
    route_duration_s,
    route_length_m,
    slant_range_m,
)
from nanobot.afsim.linter import LintDiagnostic, LintResult, Severity, lint_afsim
from nanobot.afsim.provenance import SourceKind, TrustLevel, ValidationStatus
from nanobot.afsim.schemas import (
    COA,
    Diagnostic,
    Domain,
    EventAssertion,
    GeoPoint,
    MissionSpec,
    PlatformIntent,
    PlatformIR,
    PlatformSensorIR,
    PlatformWeaponIR,
    RoutePointIR,
    RunStatus,
    ScenarioIR,
    ScheduledEngagementIR,
    SensorIR,
    Side,
    SimulationResult,
    TaskAssignment,
)

__all__ = [
    "COA",
    "AfsimCompiler",
    "Capability",
    "CapabilityKind",
    "CapabilityRegistry",
    "CompileError",
    "DEFAULT_CAPABILITY_REGISTRY",
    "Diagnostic",
    "Domain",
    "EventAssertion",
    "Evidence",
    "GeoPoint",
    "GeometryFinding",
    "LintDiagnostic",
    "LintResult",
    "MissionSpec",
    "PlatformIR",
    "PlatformIntent",
    "PlatformSensorIR",
    "PlatformWeaponIR",
    "RoutePointIR",
    "RunStatus",
    "ScenarioIR",
    "ScheduledEngagementIR",
    "SensorIR",
    "Severity",
    "Side",
    "SimulationResult",
    "SourceKind",
    "TaskAssignment",
    "TrustLevel",
    "ValidationStatus",
    "check_geometry",
    "compile_scenario",
    "emit_project",
    "ground_distance_m",
    "lint_afsim",
    "min_separation_m",
    "route_duration_s",
    "route_length_m",
    "slant_range_m",
    "write_project",
]
