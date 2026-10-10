"""COA Markdown 解析工具 — 将 MissionOrch-LC 输出的 COA MD 解析为结构化 dict。

Agent 可通过 function calling 调用 ``coa_parse(file_path=...)`` 获取:
- entities[]: 作战实体清单
- phases[]: 阶段定义
- geo_refs[]: 地理参考
- matrix[]: 单元×阶段行动矩阵
- effects[]: 效果链
- decision_points[]: 决策点
- risks[]: 关键风险
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


class COAParseTool(Tool):
    """解析 MissionOrch-LC 输出的 COA Markdown 文件为结构化数据。"""

    _scopes = {"core", "subagent"}

    @property
    def name(self) -> str:
        return "coa_parse"

    @property
    def description(self) -> str:
        return (
            "Parse a MissionOrch-LC COA Markdown file into structured data. "
            "Returns JSON with entities, phases, geo_refs, matrix, effects, "
            "decision_points, and risks. Use this before generating AFSIM "
            "scenario code from a COA plan."
        )

    @property
    def read_only(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the COA Markdown file (.md)",
                    "minLength": 1,
                },
            },
            "required": ["file_path"],
        }

    async def execute(self, file_path: str, **kwargs: Any) -> str:
        try:
            path = Path(file_path).expanduser()
            if not path.exists():
                return f"Error: File not found: {file_path}"
            if not path.is_file():
                return f"Error: Not a file: {file_path}"

            content = path.read_text(encoding="utf-8")
            if len(content.strip()) < 100:
                return "Error: COA file is empty or too short"

            result = _parse_coa_markdown(content)
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as exc:
            return f"Error parsing COA: {exc}"


# ── 解析核心实现 ──


def _parse_coa_markdown(text: str) -> dict[str, Any]:
    """将 COA Markdown 文本解析为结构化字典。"""
    sections = _split_sections(text)

    result: dict[str, Any] = {
        "name": _parse_title(sections),
        "description": _find_section(sections, "方案概述", "概述", "overview") or "",
        "entities": _parse_entity_manifest(sections),
        "phases": [],
        "geo_refs": _parse_geo_refs(sections),
        "matrix": [],
        "effects": _parse_effects(sections),
        "decision_points": _parse_decision_points(sections),
        "risks": _parse_risks(sections),
    }

    # COA 矩阵同时提供 phases 和 matrix
    phases, matrix = _parse_coa_matrix(sections)
    result["phases"] = phases
    result["matrix"] = matrix

    return result


def _split_sections(text: str) -> dict[str, str]:
    """按 ## 标题切分 Markdown 为段落字典。"""
    sections: dict[str, str] = {}
    current_key = "header"
    current_lines: list[str] = []

    for line in text.split("\n"):
        heading_match = re.match(r"^##\s+(.+)", line)
        if heading_match:
            if current_lines:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key = heading_match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections[current_key] = "\n".join(current_lines).strip()
    return sections


def _find_section(sections: dict[str, str], *keywords: str) -> str | None:
    """按关键字模糊匹配段落。"""
    for key, value in sections.items():
        key_lower = key.lower()
        for kw in keywords:
            if kw.lower() in key_lower:
                return value
    return None


def _parse_title(sections: dict[str, str]) -> str:
    """从 header 段提取方案名称。"""
    header = sections.get("header", "")
    match = re.search(r"#\s+COA方案[：:]\s*(.+)", header)
    if match:
        return match.group(1).strip()
    # 也检查 # 开头的一级标题
    match = re.search(r"#\s+(.+)", header)
    return match.group(1).strip() if match else "Generated COA"


# ── Entity Manifest 解析 ──


def _parse_entity_manifest(sections: dict[str, str]) -> list[dict[str, Any]]:
    """解析 Entity Manifest / 实体清单 表格。"""
    content = _find_section(sections, "实体清单", "entity manifest", "Entity")
    if not content:
        return []

    rows = _parse_markdown_table(content)
    entities: list[dict[str, Any]] = []

    for row in rows:
        unit_id = _get_cell(row, "unit_id", "单元ID", "编队ID", "ID")
        if not unit_id:
            continue

        # 提取 platform_code（可能有 [PLATFORM:X] 格式）
        platform_raw = _get_cell(row, "platform_code", "平台型号", "PLATFORM")
        platform_match = re.search(r"\[?PLATFORM[:：]?\s*(\S+?)\]?", platform_raw or "")
        platform_code = platform_match.group(1) if platform_match else (platform_raw or "")

        quantity_str = _get_cell(row, "quantity", "数量", "x") or "1"
        quantity_match = re.search(r"(\d+)", quantity_str)
        quantity = int(quantity_match.group(1)) if quantity_match else 1

        entities.append({
            "unit_id": unit_id,
            "platform_code": platform_code.strip(),
            "quantity": quantity,
            "side": (_get_cell(row, "side", "阵营") or "BLUE").upper(),
            "role": _get_cell(row, "role", "角色", "职能") or "",
            "callsign": _get_cell(row, "callsign", "呼号") or "",
            "base": _get_cell(row, "base", "基地") or "",
            "weapons": _get_cell(row, "weapons", "武器") or "",
            "notes": _get_cell(row, "notes", "备注") or "",
        })

    return entities


# ── Geo Reference 解析 ──


def _parse_geo_refs(sections: dict[str, str]) -> list[dict[str, Any]]:
    """解析 Geo Reference / 地理参考 表格。"""
    content = _find_section(sections, "地理参考", "geo reference", "Geo")
    if not content:
        return []

    rows = _parse_markdown_table(content)
    refs: list[dict[str, Any]] = []

    for row in rows:
        ref_id = _get_cell(row, "ref_id", "参考ID", "ID")
        if not ref_id:
            continue

        refs.append({
            "ref_id": ref_id,
            "type": (_get_cell(row, "type", "类型") or "").upper(),
            "name": _get_cell(row, "name", "名称") or "",
            "side": (_get_cell(row, "side", "阵营") or "").upper(),
            "hint": _get_cell(row, "hint", "提示", "位置描述") or "",
            "notes": _get_cell(row, "notes", "备注") or "",
        })

    return refs


# ── COA Matrix 解析 ──


def _parse_coa_matrix(sections: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """解析 COA 矩阵表格，返回 (phases, matrix)。"""
    content = _find_section(sections, "COA矩阵", "coa matrix", "矩阵", "COA Matrix")
    if not content:
        return [], []

    table_lines = [
        line.strip() for line in content.split("\n")
        if line.strip() and "|" in line
    ]
    if len(table_lines) < 3:
        return [], []

    # 解析阶段表头
    header_cells = _split_table_row(table_lines[0])
    phases = _parse_phase_headers(header_cells[1:])

    # 解析数据行
    data_lines = [
        line for line in table_lines[1:]
        if not _is_separator_row(_split_table_row(line))
    ]

    matrix: list[dict[str, Any]] = []
    for line in data_lines:
        cells = _split_table_row(line)
        if not cells:
            continue

        # 提取 unit_id
        unit_cell = cells[0]
        unit_id_match = re.search(r"[（(]\s*([A-Za-z0-9_\-]+)\s*[）)]", unit_cell)
        unit_id = unit_id_match.group(1) if unit_id_match else ""

        if not unit_id:
            continue

        action_cells = cells[1:]
        for idx, phase in enumerate(phases):
            cell_text = action_cells[idx].strip() if idx < len(action_cells) else ""
            actions = _parse_action_cell(cell_text)
            if actions:
                matrix.append({
                    "unit_id": unit_id,
                    "phase_id": phase["phase_id"],
                    "actions": actions,
                })

    return phases, matrix


def _parse_phase_headers(header_cells: list[str]) -> list[dict[str, str]]:
    """解析阶段表头单元格。"""
    phases: list[dict[str, str]] = []
    for idx, cell in enumerate(header_cells):
        cell_clean = cell.replace("<br>", "\n").replace("<BR>", "\n")

        # 提取阶段名
        name_match = re.search(
            r"阶段\s*\d+\s*[:：]\s*(.+?)(?:\s*\(|$|\n|\*)", cell_clean
        )
        name = name_match.group(1).strip() if name_match else cell_clean.split("\n")[0].strip()

        # 提取触发条件
        trigger = ""
        trigger_match = re.search(r"触发[→]?\s*[→]?\s*(.+?)(?:\s*\*\s*|$)", cell_clean)
        if trigger_match:
            trigger = trigger_match.group(1).strip()

        phases.append({
            "phase_id": f"Phase_{idx + 1}",
            "name": name,
            "trigger": trigger,
        })

    return phases


def _parse_action_cell(cell_text: str) -> list[str]:
    """解析矩阵单元格中的行动列表。"""
    if not cell_text or cell_text.strip() in ("-", "—", "N/A", ""):
        return []

    text = cell_text.replace("<br>", "\n").replace("<BR>", "\n")

    # 按编号拆分
    numbered = re.split(r"\n?\s*\d+\.\s+", text)
    numbered = [item.strip() for item in numbered if item.strip()]
    if numbered:
        return numbered

    # 按分号拆分
    if ";" in text or "；" in text:
        return [item.strip() for item in re.split(r"[;；]+", text) if item.strip()]

    if "\n" in text:
        return [item.strip() for item in text.split("\n") if item.strip()]

    return [text.strip()] if text.strip() else []


# ── 效果链 / 决策点 / 风险 ──


def _parse_effects(sections: dict[str, str]) -> list[dict[str, Any]]:
    """解析效果链表格。"""
    content = _find_section(sections, "效果链", "effects")
    if not content:
        return []

    effects: list[dict[str, Any]] = []
    for row in _parse_markdown_table(content):
        eid = _get_cell(row, "效果ID", "effect_id", "ID")
        if not eid:
            continue
        effects.append({
            "effect_id": eid,
            "description": _get_cell(row, "效果描述", "description") or "",
            "measures": _split_list(_get_cell(row, "衡量指标", "measures") or ""),
            "achieved_by": _split_list(_get_cell(row, "达成实体", "达成单元", "achieved_by") or ""),
        })
    return effects


def _parse_decision_points(sections: dict[str, str]) -> list[dict[str, Any]]:
    """解析决策点表格。"""
    content = _find_section(sections, "决策点", "decision")
    if not content:
        return []

    dps: list[dict[str, Any]] = []
    for row in _parse_markdown_table(content):
        dp_id = _get_cell(row, "决策点ID", "dp_id", "ID")
        if not dp_id:
            continue
        dps.append({
            "dp_id": dp_id,
            "phase_id": _get_cell(row, "所属阶段", "phase") or "",
            "condition": _get_cell(row, "触发条件", "condition") or "",
            "options_text": _get_cell(row, "选项", "options") or "",
        })
    return dps


def _parse_risks(sections: dict[str, str]) -> list[dict[str, str]]:
    """解析关键风险表格。"""
    content = _find_section(sections, "关键风险", "risk", "风险")
    if not content:
        return []

    risks: list[dict[str, str]] = []
    for row in _parse_markdown_table(content):
        desc = _get_cell(row, "风险描述", "description")
        if not desc:
            continue
        risks.append({
            "category": _get_cell(row, "风险类别", "category") or "GENERAL",
            "description": desc,
            "mitigation": _get_cell(row, "缓解措施", "mitigation") or "",
        })
    return risks


# ── Markdown 表格工具函数 ──


def _parse_markdown_table(text: str) -> list[dict[str, str]]:
    """解析 Markdown 表格为字典列表。"""
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    table_lines = [line for line in lines if "|" in line]
    if len(table_lines) < 3:
        return []

    headers = _split_table_row(table_lines[0])
    data_lines = [
        line for line in table_lines[1:]
        if not _is_separator_row(_split_table_row(line))
    ]

    rows: list[dict[str, str]] = []
    for line in data_lines:
        cells = _split_table_row(line)
        row = {}
        for idx, header in enumerate(headers):
            row[header] = cells[idx].strip() if idx < len(cells) else ""
        rows.append(row)
    return rows


def _split_table_row(line: str) -> list[str]:
    """拆分 Markdown 表格行。"""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    """判断是否为分隔线。"""
    sep = re.compile(r"^[-:\s]*$")
    return all(sep.match(cell) for cell in cells)


def _get_cell(row: dict[str, str], *candidate_keys: str) -> str | None:
    """从行字典中按候选键获取值。"""
    for key in candidate_keys:
        if key in row:
            val = row[key].strip()
            if val and val not in ("-", "—", "N/A"):
                return val
        for rk, rv in row.items():
            if key.lower() in rk.lower():
                val = rv.strip()
                if val and val not in ("-", "—", "N/A"):
                    return val
    return None


def _split_list(value: str) -> list[str]:
    """按分隔符拆分列表。"""
    if not value:
        return []
    return [item.strip() for item in re.split(r"[;；,，]+", value) if item.strip()]
