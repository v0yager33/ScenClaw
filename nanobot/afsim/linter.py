"""Static checks for the generated, constrained AFSIM text subset."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class LintDiagnostic:
    severity: Severity
    code: str
    message: str
    line: int | None = None


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
}
_BLOCK_ENDS = set(_BLOCK_STARTS.values())
_POSITION_RE = re.compile(
    r"\bposition\s+(?P<lat>\d+(?:\.\d+)?)(?P<ns>[ns])\s+"
    r"(?P<lon>\d+(?:\.\d+)?)(?P<ew>[ew])\b",
    re.IGNORECASE,
)


def _content(line: str) -> str:
    return line.split("#", 1)[0].strip()


def lint_afsim(text: str) -> LintResult:
    diagnostics: list[LintDiagnostic] = []
    stack: list[tuple[str, int]] = []
    sensor_types: dict[str, int] = {}
    platform_types: dict[str, int] = {}
    platform_ids: dict[str, int] = {}
    sensor_references: list[tuple[str, int]] = []
    platform_references: list[tuple[str, int]] = []
    has_end_time = False

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = _content(raw_line)
        if not line:
            continue
        tokens = line.split()
        keyword = tokens[0].lower()

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
            platform_types[name] = line_number
            if tokens[-1].lower() != "end_platform_type":
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
        elif keyword == "add" and len(tokens) >= 4 and tokens[1].lower() == "sensor":
            sensor_references.append((tokens[3], line_number))
            stack.append(("end_sensor", line_number))
        elif keyword == "add" and len(tokens) >= 3 and tokens[1].lower() == "mover":
            stack.append(("end_mover", line_number))
        elif keyword == "end_mover":
            if not stack:
                diagnostics.append(
                    LintDiagnostic(
                        Severity.ERROR,
                        "block.unexpected_end",
                        "unexpected end_mover",
                        line_number,
                    )
                )
            else:
                expected, start_line = stack.pop()
                if expected != "end_mover":
                    diagnostics.append(
                        LintDiagnostic(
                            Severity.ERROR,
                            "block.mismatch",
                            f"expected {expected} for block opened at line {start_line}, got end_mover",
                            line_number,
                        )
                    )
        elif keyword == "sensor" and len(tokens) == 2:
            stack.append(("end_sensor", line_number))
        elif keyword in _BLOCK_STARTS:
            stack.append((_BLOCK_STARTS[keyword], line_number))

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
        if sensor_type not in sensor_types and not sensor_type.startswith("WSF_"):
            diagnostics.append(
                LintDiagnostic(
                    Severity.ERROR,
                    "sensor.unknown_type",
                    f"unknown sensor type {sensor_type}",
                    line_number,
                )
            )
    for platform_type, line_number in platform_references:
        if platform_type not in platform_types and not platform_type.startswith("WSF_"):
            diagnostics.append(
                LintDiagnostic(
                    Severity.ERROR,
                    "platform.unknown_type",
                    f"unknown platform type {platform_type}",
                    line_number,
                )
            )
    if not has_end_time:
        diagnostics.append(
            LintDiagnostic(Severity.ERROR, "simulation.missing_end_time", "missing end_time command")
        )
    if not platform_ids:
        diagnostics.append(LintDiagnostic(Severity.WARNING, "platform.none", "scenario defines no platforms"))

    return LintResult(tuple(diagnostics))
