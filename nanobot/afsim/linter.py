"""Static checks for the generated, constrained AFSIM text subset."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import PurePosixPath


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class LintDiagnostic:
    severity: Severity
    code: str
    message: str
    line: int | None = None
    file: str | None = None


@dataclass(frozen=True)
class LintResult:
    diagnostics: tuple[LintDiagnostic, ...]

    @property
    def is_valid(self) -> bool:
        return not any(item.severity is Severity.ERROR for item in self.diagnostics)


_BLOCK_STARTS = {
    "sensor": "end_sensor",
    "platform": "end_platform",
    "platform_type": "end_platform_type",
    "mode": "end_mode",
    "mode_template": "end_mode_template",
    "route": "end_route",
    "event_output": "end_event_output",
    "event_pipe": "end_event_pipe",
    "mover": "end_mover",
    "weapon": "end_weapon",
    "weapon_effects": "end_weapon_effects",
    "radar_signature": "end_radar_signature",
    "execute": "end_execute",
}
_BLOCK_ENDS = set(_BLOCK_STARTS.values())
_POSITION_RE = re.compile(
    r"\bposition\s+(?P<lat>\d+(?:\.\d+)?)(?P<ns>[ns])\s+"
    r"(?P<lon>\d+(?:\.\d+)?)(?P<ew>[ew])\b",
    re.IGNORECASE,
)


def _content(line: str) -> str:
    return line.split("#", 1)[0].split("//", 1)[0].strip()


def lint_afsim(text: str) -> LintResult:
    diagnostics: list[LintDiagnostic] = []
    stack: list[tuple[str, int]] = []
    sensor_types: dict[str, int] = {}
    platform_types: dict[str, int] = {}
    platform_ids: dict[str, int] = {}
    sensor_references: list[tuple[str, int]] = []
    platform_references: list[tuple[str, int]] = []
    signatures: dict[str, int] = {}
    effects: dict[str, int] = {}
    signature_references: list[tuple[str, int]] = []
    effect_references: list[tuple[str, int]] = []
    has_end_time = False

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = _content(raw_line)
        if not line:
            continue
        tokens = line.split()
        keyword = tokens[0].lower()
        # execute 块体是脚本语言，不按想定命令扫描。
        if stack and stack[-1][0] == "end_execute" and keyword != "end_execute":
            continue

        if keyword == "end_time":
            has_end_time = True

        match = _POSITION_RE.search(line)
        if match:
            latitude = float(match.group("lat"))
            longitude = float(match.group("lon"))
            if latitude > 90:
                diagnostics.append(
                    LintDiagnostic(Severity.ERROR, "coordinate.latitude", "latitude exceeds 90 degrees", line_number)
                )
            if longitude > 180:
                diagnostics.append(
                    LintDiagnostic(
                        Severity.ERROR, "coordinate.longitude", "longitude exceeds 180 degrees", line_number
                    )
                )

        if keyword in _BLOCK_ENDS:
            if not stack:
                diagnostics.append(
                    LintDiagnostic(Severity.ERROR, "block.unexpected_end", f"unexpected {keyword}", line_number)
                )
                continue
            expected, start_line = stack.pop()
            if keyword != expected:
                diagnostics.append(
                    LintDiagnostic(
                        Severity.ERROR,
                        "block.mismatch",
                        f"expected {expected} for block opened at line {start_line}, got {keyword}",
                        line_number,
                    )
                )
            continue

        if keyword == "sensor" and len(tokens) >= 3:
            name = tokens[1]
            if name in sensor_types:
                diagnostics.append(
                    LintDiagnostic(Severity.ERROR, "sensor.duplicate", f"duplicate sensor type {name}", line_number)
                )
            sensor_types[name] = line_number
            stack.append(("end_sensor", line_number))
        elif keyword == "platform_type" and len(tokens) >= 2:
            name = tokens[1]
            if name in platform_types or name == "WSF_PLATFORM":
                diagnostics.append(LintDiagnostic(
                    Severity.ERROR, "platform_type.duplicate",
                    f"duplicate or built-in platform type {name}", line_number,
                ))
            platform_types[name] = line_number
            if len(tokens) < 3 or tokens[2] == "end_platform_type":
                diagnostics.append(LintDiagnostic(
                    Severity.ERROR, "platform_type.missing_parent",
                    f"platform type {name} must declare a parent", line_number,
                ))
            else:
                platform_references.append((tokens[2], line_number))
            stack.append(("end_platform_type", line_number))
        elif keyword == "platform" and len(tokens) >= 3:
            name, platform_type = tokens[1], tokens[2]
            if name in platform_ids:
                diagnostics.append(
                    LintDiagnostic(Severity.ERROR, "platform.duplicate", f"duplicate platform {name}", line_number)
                )
            platform_ids[name] = line_number
            platform_references.append((platform_type, line_number))
            stack.append(("end_platform", line_number))
        elif keyword == "add" and len(tokens) >= 3 and tokens[1].lower() in _BLOCK_STARTS:
            component = tokens[1].lower()
            if component == "sensor" and len(tokens) >= 4:
                sensor_references.append((tokens[3], line_number))
            stack.append((_BLOCK_STARTS[component], line_number))
        elif keyword in ("radar_signature", "weapon_effects") and len(tokens) >= 2:
            definitions = signatures if keyword == "radar_signature" else effects
            references = signature_references if keyword == "radar_signature" else effect_references
            if len(tokens) >= 3 and tokens[2] != _BLOCK_STARTS[keyword]:
                if stack:
                    diagnostics.append(LintDiagnostic(
                        Severity.ERROR, f"{keyword}.definition_scope",
                        f"{keyword} definitions must be top-level", line_number,
                    ))
                if tokens[1] in definitions:
                    diagnostics.append(LintDiagnostic(
                        Severity.ERROR, f"{keyword}.duplicate",
                        f"duplicate {keyword} {tokens[1]}", line_number,
                    ))
                definitions[tokens[1]] = line_number
                stack.append((_BLOCK_STARTS[keyword], line_number))
            else:
                references.append((tokens[1], line_number))
                if not stack:
                    diagnostics.append(LintDiagnostic(
                        Severity.ERROR, f"{keyword}.missing_base",
                        f"top-level {keyword} must declare a base type", line_number,
                    ))
        elif keyword == "constant" and (not stack or stack[-1][0] != "end_radar_signature"):
            diagnostics.append(LintDiagnostic(
                Severity.ERROR, "radar_signature.invalid_constant",
                "constant must be inside a radar_signature definition, not a reference", line_number,
            ))
        elif keyword in _BLOCK_STARTS:
            stack.append((_BLOCK_STARTS[keyword], line_number))

        # 支持同一行起止的空块，但仍检查其声明和引用。
        if stack and stack[-1][1] == line_number and tokens[-1] == stack[-1][0]:
            stack.pop()

    for expected, start_line in reversed(stack):
        diagnostics.append(
            LintDiagnostic(
                Severity.ERROR,
                "block.unclosed",
                f"block opened at line {start_line} is missing {expected}",
                start_line,
            )
        )

    for sensor_type, line_number in sensor_references:
        if sensor_types.get(sensor_type, line_number) >= line_number and sensor_type != "WSF_GEOMETRIC_SENSOR":
            diagnostics.append(
                LintDiagnostic(
                    Severity.ERROR,
                    "sensor.unknown_type",
                    f"unknown sensor type {sensor_type}",
                    line_number,
                )
            )
    for platform_type, line_number in platform_references:
        if platform_types.get(platform_type, line_number) >= line_number and platform_type != "WSF_PLATFORM":
            diagnostics.append(
                LintDiagnostic(
                    Severity.ERROR,
                    "platform.unknown_type",
                    f"unknown platform type {platform_type}",
                    line_number,
                )
            )
    for kind, definitions, references in (
        ("radar_signature", signatures, signature_references),
        ("weapon_effects", effects, effect_references),
    ):
        for name, line_number in references:
            if definitions.get(name, line_number) >= line_number:
                diagnostics.append(LintDiagnostic(
                    Severity.ERROR, f"{kind}.undefined",
                    f"{kind} {name} is not defined before use", line_number,
                ))
    if not has_end_time:
        diagnostics.append(
            LintDiagnostic(Severity.ERROR, "simulation.missing_end_time", "missing end_time command")
        )
    if not platform_ids:
        diagnostics.append(LintDiagnostic(Severity.WARNING, "platform.none", "scenario defines no platforms"))

    return LintResult(tuple(diagnostics))


def _project_path(value: str) -> str | None:
    """只接受项目根内的相对路径，不访问外部文件系统。"""
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or ":" in value or not path.parts:
        return None
    return path.as_posix()


def expand_project(
    files: Mapping[str, str], entry: str = "main.txt"
) -> tuple[str, list[tuple[str, int]], list[LintDiagnostic]]:
    """按生成项目的 file_path . 约定展开 include，保留原文件和行号。"""
    lines: list[str] = []
    sources: list[tuple[str, int]] = []
    diagnostics: list[LintDiagnostic] = []
    seen: set[str] = set()
    active: list[str] = []
    for name in files:
        if _project_path(name) != name:
            diagnostics.append(LintDiagnostic(
                Severity.ERROR, "project.invalid_path", "invalid project-relative path", file=name,
            ))
    if diagnostics:
        return "", sources, diagnostics

    def visit(name: str, once: bool, source: str, line_number: int) -> None:
        target = _project_path(name)
        if target is None:
            diagnostics.append(LintDiagnostic(
                Severity.ERROR, "include.invalid_path", f"unsafe include path {name}",
                line_number, source,
            ))
            return
        if target not in files:
            diagnostics.append(LintDiagnostic(
                Severity.ERROR, "include.missing", f"included file not found: {target}",
                line_number, source,
            ))
            return
        if target in active:
            diagnostics.append(LintDiagnostic(
                Severity.ERROR, "include.cycle", f"include cycle: {' -> '.join(active + [target])}",
                line_number, source,
            ))
            return
        if once and target in seen:
            return
        seen.add(target)
        active.append(target)
        for number, raw in enumerate(files[target].splitlines(), 1):
            content = _content(raw)
            tokens = content.split(maxsplit=1)
            if tokens and tokens[0] in ("include", "include_once"):
                included = tokens[1].strip().strip('"') if len(tokens) == 2 else ""
                visit(included, tokens[0] == "include_once", target, number)
            else:
                lines.append(raw)
                sources.append((target, number))
        active.pop()

    visit(entry, False, entry, 1)
    return "\n".join(lines), sources, diagnostics


def lint_project(files: Mapping[str, str], entry: str = "main.txt") -> LintResult:
    """校验真正的递归 include 项目，而不是仅校验入口或另一份单文件。"""
    text, sources, diagnostics = expand_project(files, entry)
    if not diagnostics:
        for diagnostic in lint_afsim(text).diagnostics:
            if diagnostic.line is not None and diagnostic.line <= len(sources):
                file, line = sources[diagnostic.line - 1]
                diagnostic = replace(diagnostic, file=file, line=line)
            diagnostics.append(diagnostic)
    return LintResult(tuple(diagnostics))
