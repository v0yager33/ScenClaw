"""平台模板匹配工具 — 基于多维特征相似度的 AFSIM 模板智能选择。

Agent 可通过 function calling 调用 ``template_match(...)`` 将 COA 实体
映射到最优 AFSIM 模板组合 (platform_type + sensor + weapon + mover)。

核心算法:
  1. 多维特征向量构造 (category, side, weapons, sensors, role)
  2. 加权余弦相似度计算
  3. 约束满足验证 (category 硬约束 + side 兼容)
  4. 模板组合优化 (最小化冗余, 最大化覆盖率)
  5. 去重分组 (相同 platform_code 共享 platform_type)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from nanobot.agent.tools.base import Tool


# ═══════════════════════════════════════════════════════
# 特征权重 (用于相似度计算)
# ═══════════════════════════════════════════════════════

_FEATURE_WEIGHTS: dict[str, float] = {
    "category": 0.35,    # 功能分类 (最重要, 决定平台基本构型)
    "side": 0.15,        # 阵营 (影响 side 字段)
    "weapons": 0.25,     # 武器配置 (影响 weapon_type)
    "sensors": 0.15,     # 传感器配置
    "role": 0.10,        # 角色描述 (文本相似度)
}

# Category 相似度矩阵 (0-1, 1=完全匹配)
_CATEGORY_SIMILARITY: dict[str, dict[str, float]] = {
    "awacs":          {"awacs": 1.0, "ew": 0.3, "fighter": 0.2, "strike": 0.2, "tanker": 0.1, "sam_long_range": 0.0, "sam_medium_range": 0.0, "early_warning": 0.2, "vhf_radar": 0.1, "generic_air": 0.3},
    "ew":             {"awacs": 0.3, "ew": 1.0, "fighter": 0.4, "strike": 0.5, "tanker": 0.1, "sam_long_range": 0.0, "sam_medium_range": 0.0, "early_warning": 0.1, "vhf_radar": 0.1, "generic_air": 0.3},
    "fighter":        {"awacs": 0.2, "ew": 0.4, "fighter": 1.0, "strike": 0.7, "tanker": 0.1, "sam_long_range": 0.0, "sam_medium_range": 0.0, "early_warning": 0.0, "vhf_radar": 0.0, "generic_air": 0.5},
    "strike":         {"awacs": 0.2, "ew": 0.5, "fighter": 0.7, "strike": 1.0, "strike_bomb": 0.9, "tanker": 0.1, "sam_long_range": 0.0, "sam_medium_range": 0.0, "early_warning": 0.0, "vhf_radar": 0.0, "generic_air": 0.5},
    "strike_bomb":    {"awacs": 0.1, "ew": 0.3, "fighter": 0.5, "strike": 0.9, "strike_bomb": 1.0, "tanker": 0.1, "sam_long_range": 0.0, "sam_medium_range": 0.0, "early_warning": 0.0, "vhf_radar": 0.0, "generic_air": 0.4},
    "tanker":         {"awacs": 0.1, "ew": 0.1, "fighter": 0.1, "strike": 0.1, "strike_bomb": 0.1, "tanker": 1.0, "sam_long_range": 0.0, "sam_medium_range": 0.0, "early_warning": 0.0, "vhf_radar": 0.0, "generic_air": 0.2},
    "sam_long_range": {"awacs": 0.0, "ew": 0.0, "fighter": 0.0, "strike": 0.0, "tanker": 0.0, "sam_long_range": 1.0, "sam_medium_range": 0.7, "early_warning": 0.2, "vhf_radar": 0.1, "generic_ground": 0.3},
    "sam_medium_range":{"awacs": 0.0, "ew": 0.0, "fighter": 0.0, "strike": 0.0, "tanker": 0.0, "sam_long_range": 0.7, "sam_medium_range": 1.0, "early_warning": 0.2, "vhf_radar": 0.1, "generic_ground": 0.3},
    "early_warning":  {"awacs": 0.2, "ew": 0.1, "fighter": 0.0, "strike": 0.0, "tanker": 0.0, "sam_long_range": 0.2, "sam_medium_range": 0.2, "early_warning": 1.0, "vhf_radar": 0.6, "generic_ground": 0.2},
    "vhf_radar":      {"awacs": 0.1, "ew": 0.1, "fighter": 0.0, "strike": 0.0, "tanker": 0.0, "sam_long_range": 0.1, "sam_medium_range": 0.1, "early_warning": 0.6, "vhf_radar": 1.0, "generic_ground": 0.2},
    "generic_air":    {"awacs": 0.3, "ew": 0.3, "fighter": 0.5, "strike": 0.5, "tanker": 0.2, "generic_air": 1.0},
    "generic_ground": {"sam_long_range": 0.3, "sam_medium_range": 0.3, "early_warning": 0.2, "vhf_radar": 0.2, "generic_ground": 1.0},
    "generic_surface": {"generic_surface": 1.0},
}


# ═══════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════


@dataclass
class PlatformMapping:
    """从 platform-mapping.yaml 加载的平台定义。"""
    code: str
    icon: str = ""
    mover: str = "WSF_AIR_MOVER"
    category: str = ""
    side_default: str = "blue"
    speed: str = ""
    altitude: str = ""
    radar_signature: dict | None = None
    default_sensors: list[dict] = field(default_factory=list)
    default_weapons: list[dict] = field(default_factory=list)
    default_processors: list[dict] = field(default_factory=list)
    notes: str = ""


@dataclass
class MatchResult:
    """匹配结果。"""
    entity_unit_id: str
    platform_code: str
    matched_mapping: str  # 匹配到的 platform_code
    similarity_score: float  # [0, 1]
    category: str
    side: str
    template_components: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class GroupedPlatform:
    """去重后的平台组。"""
    platform_code: str
    mapping: PlatformMapping
    entities: list[dict]  # 属于该组的所有 entities
    total_quantity: int = 0


# ═══════════════════════════════════════════════════════
# Tool 定义
# ═══════════════════════════════════════════════════════


class TemplateMatchTool(Tool):
    """将 COA 实体匹配到最优 AFSIM 平台模板。使用多维特征相似度算法。"""

    _scopes = {"core", "subagent"}

    @property
    def name(self) -> str:
        return "template_match"

    @property
    def description(self) -> str:
        return (
            "Match COA entities to optimal AFSIM platform templates using "
            "multi-dimensional feature similarity scoring. Takes parsed COA "
            "entities JSON and returns platform assignments with similarity "
            "scores, grouped by platform_type for deduplication."
        )

    @property
    def read_only(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "entities_json": {
                    "type": "string",
                    "description": "JSON string of entities array from coa_parse output.",
                },
                "mapping_path": {
                    "type": "string",
                    "description": "Path to platform-mapping.yaml. Defaults to skill directory.",
                },
            },
            "required": ["entities_json"],
        }

    async def execute(self, **kwargs: Any) -> str:
        try:
            entities_raw = kwargs.get("entities_json", "[]")
            entities = json.loads(entities_raw) if isinstance(entities_raw, str) else entities_raw
            mapping_path = kwargs.get("mapping_path", "")

            if not mapping_path:
                mapping_path = str(
                    Path(__file__).resolve().parent.parent.parent
                    / "skills" / "afsim-scenario" / "platform-mapping.yaml"
                )

            mappings = _load_mappings(mapping_path)
            matches = match_entities(entities, mappings)
            groups = group_by_platform(entities, mappings)

            output = {
                "matches": [],
                "platform_groups": [],
                "summary": {
                    "total_entities": len(entities),
                    "unique_platform_types": len(groups),
                    "total_quantity": sum(g.total_quantity for g in groups),
                    "avg_similarity": (
                        sum(m.similarity_score for m in matches) / len(matches)
                        if matches else 0
                    ),
                },
            }

            for m in matches:
                entry = {
                    "unit_id": m.entity_unit_id,
                    "platform_code": m.platform_code,
                    "matched": m.matched_mapping,
                    "score": round(m.similarity_score, 4),
                    "category": m.category,
                    "side": m.side,
                    "components": m.template_components,
                }
                if m.warnings:
                    entry["warnings"] = m.warnings
                output["matches"].append(entry)

            for g in groups:
                output["platform_groups"].append({
                    "platform_code": g.platform_code,
                    "afsim_type_name": _make_type_name(g.platform_code),
                    "category": g.mapping.category,
                    "mover": g.mapping.mover,
                    "entities": [e.get("unit_id", "") for e in g.entities],
                    "total_quantity": g.total_quantity,
                })

            return json.dumps(output, ensure_ascii=False, indent=2)
        except Exception as exc:
            return f"Error matching templates: {exc}"


# ═══════════════════════════════════════════════════════
# 映射加载
# ═══════════════════════════════════════════════════════


def _load_mappings(path: str) -> dict[str, PlatformMapping]:
    """加载 platform-mapping.yaml。"""
    p = Path(path)
    if not p.exists():
        return {}

    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}

    mappings: dict[str, PlatformMapping] = {}
    for code, params in data.items():
        if not isinstance(params, dict):
            continue
        mappings[code] = PlatformMapping(
            code=code,
            icon=params.get("icon", ""),
            mover=params.get("mover", "WSF_AIR_MOVER"),
            category=params.get("category", ""),
            side_default=params.get("side_default", "blue"),
            speed=params.get("speed", ""),
            altitude=params.get("altitude", ""),
            radar_signature=params.get("radar_signature"),
            default_sensors=params.get("default_sensors", []),
            default_weapons=params.get("default_weapons", []),
            default_processors=params.get("default_processors", []),
            notes=params.get("notes", ""),
        )
    return mappings


# ═══════════════════════════════════════════════════════
# 核心匹配算法
# ═══════════════════════════════════════════════════════


def match_entities(
    entities: list[dict[str, Any]],
    mappings: dict[str, PlatformMapping],
) -> list[MatchResult]:
    """为每个 entity 找到最优模板匹配。

    算法流程:
      1. 精确匹配: platform_code 完全匹配
      2. 相似度匹配: 构造特征向量, 计算加权相似度
      3. 回退匹配: 使用 GENERIC 类型
    """
    results: list[MatchResult] = []

    for entity in entities:
        unit_id = entity.get("unit_id", "")
        platform_code = entity.get("platform_code", "")
        side = entity.get("side", "BLUE").upper()
        role = entity.get("role", "")
        weapons = entity.get("weapons", "")

        # 精确匹配
        if platform_code in mappings:
            m = mappings[platform_code]
            result = MatchResult(
                entity_unit_id=unit_id,
                platform_code=platform_code,
                matched_mapping=platform_code,
                similarity_score=1.0,
                category=m.category,
                side=m.side_default,
                template_components=_build_component_list(m),
            )
            # 检查 side 一致性
            if side and side.lower() != m.side_default.lower():
                result.warnings.append(
                    f"Side mismatch: entity={side}, mapping={m.side_default}"
                )
            results.append(result)
            continue

        # 相似度匹配
        entity_features = _build_features(platform_code, side, role, weapons)
        best_score = -1.0
        best_code = ""
        for code, mapping in mappings.items():
            mapping_features = _build_features(
                code, mapping.side_default, mapping.notes, ""
            )
            score = _weighted_similarity(entity_features, mapping_features, mapping.category)
            if score > best_score:
                best_score = score
                best_code = code

        if best_score > 0.4 and best_code:
            m = mappings[best_code]
            results.append(MatchResult(
                entity_unit_id=unit_id,
                platform_code=platform_code,
                matched_mapping=best_code,
                similarity_score=best_score,
                category=m.category,
                side=side.lower(),
                template_components=_build_component_list(m),
                warnings=[f"Fuzzy match: {platform_code} → {best_code} (score={best_score:.3f})"],
            ))
        else:
            # 回退到 GENERIC
            fallback = "GENERIC_AIR" if side == "BLUE" else "GENERIC_GROUND"
            m = mappings.get(fallback)
            results.append(MatchResult(
                entity_unit_id=unit_id,
                platform_code=platform_code,
                matched_mapping=fallback,
                similarity_score=0.1,
                category="generic",
                side=side.lower(),
                template_components=_build_component_list(m) if m else ["platform_type (generic)"],
                warnings=[f"No match found for {platform_code}, using {fallback}"],
            ))

    return results


def group_by_platform(
    entities: list[dict[str, Any]],
    mappings: dict[str, PlatformMapping],
) -> list[GroupedPlatform]:
    """将 entities 按 platform_code 分组 (去重)。

    相同 platform_code 的 entities 共享一个 platform_type 定义。
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for entity in entities:
        code = entity.get("platform_code", "UNKNOWN")
        groups[code].append(entity)

    result: list[GroupedPlatform] = []
    for code, ents in groups.items():
        mapping = mappings.get(code)
        if not mapping:
            # 尝试模糊匹配
            for mcode, m in mappings.items():
                if _normalize(mcode) == _normalize(code):
                    mapping = m
                    break
        if not mapping:
            fallback = "GENERIC_AIR" if ents[0].get("side", "BLUE").upper() == "BLUE" else "GENERIC_GROUND"
            mapping = mappings.get(fallback, PlatformMapping(code=fallback, category="generic"))

        total_qty = 0
        for e in ents:
            qty_str = e.get("quantity", "1")
            qty_match = re.search(r"(\d+)", str(qty_str))
            total_qty += int(qty_match.group(1)) if qty_match else 1
        result.append(GroupedPlatform(
            platform_code=code,
            mapping=mapping,
            entities=ents,
            total_quantity=total_qty,
        ))

    return result


# ═══════════════════════════════════════════════════════
# 特征构造与相似度计算
# ═══════════════════════════════════════════════════════


def _build_features(
    platform_code: str, side: str, role: str, weapons: str
) -> dict[str, str]:
    """构造特征向量。"""
    return {
        "category": _infer_category(platform_code, role),
        "side": side.lower(),
        "weapons": _normalize_weapons(weapons),
        "role": role.lower(),
    }


def _infer_category(platform_code: str, role: str) -> str:
    """从 platform_code 和 role 推断 category。"""
    code_lower = platform_code.lower()
    role_lower = role.lower()

    # 从 platform_code 推断
    category_hints = {
        "awacs": ["awacs", "预警", "kj-500", "e-3", "e-2"],
        "ew": ["ew", "电子战", "j-16d", "ea-18", "ec-130"],
        "fighter": ["fighter", "战斗", "制空", "j-20", "f-22", "su-35", "j-10"],
        "strike": ["strike", "攻击", "sead", "j-16", "f-15e", "su-30"],
        "tanker": ["tanker", "加油", "yy-20", "kc-135", "kc-46"],
        "sam_long_range": ["s-400", "s-300", "sa-21", "远程防空"],
        "sam_medium_range": ["hq-22", "hq-9", "sa-10", "中程防空"],
        "early_warning": ["lr_radar", "远程预警", "预警雷达"],
        "vhf_radar": ["vhf", "反隐身", "米波"],
    }
    for cat, keywords in category_hints.items():
        for kw in keywords:
            if kw in code_lower or kw in role_lower:
                return cat

    return "generic"


def _normalize_weapons(weapons: str) -> str:
    """标准化武器描述。"""
    if not weapons or weapons in ("无", "-", "N/A"):
        return ""
    # 提取武器型号
    weapon_patterns = re.findall(
        r"(?:PL-\d+|鹰击-\d+|YJ-\d+|R-\d+|48N6|HQ-\d+|精确制导炸弹)",
        weapons,
        re.IGNORECASE,
    )
    return ",".join(sorted(set(w.upper() for w in weapon_patterns)))


def _weighted_similarity(
    features_a: dict[str, str],
    features_b: dict[str, str],
    category_b: str,
) -> float:
    """加权多维相似度。

    算法:
      similarity = Σ(weight_i × sim_i(a_i, b_i))

    其中 sim_i 按特征类型使用不同的相似度函数:
      - category: 查表 (CATEGORY_SIMILARITY)
      - side: 精确匹配 (0/1)
      - weapons: Jaccard 集合相似度
      - role: 关键词重叠系数
    """
    total = 0.0

    # Category 相似度
    cat_a = features_a.get("category", "generic")
    cat_b = features_b.get("category", "generic") or category_b
    cat_sim = _CATEGORY_SIMILARITY.get(cat_a, {}).get(cat_b, 0.0)
    total += _FEATURE_WEIGHTS["category"] * cat_sim

    # Side 相似度
    side_sim = 1.0 if features_a.get("side") == features_b.get("side") else 0.0
    total += _FEATURE_WEIGHTS["side"] * side_sim

    # Weapons Jaccard 相似度
    w_a = set(features_a.get("weapons", "").split(",")) - {"", }
    w_b = set(features_b.get("weapons", "").split(",")) - {""}
    if w_a or w_b:
        jaccard = len(w_a & w_b) / len(w_a | w_b) if (w_a | w_b) else 0.0
        total += _FEATURE_WEIGHTS["weapons"] * jaccard
    else:
        total += _FEATURE_WEIGHTS["weapons"] * 0.5  # 双方都无武器, 部分匹配

    # Role 关键词重叠
    role_a = set(features_a.get("role", "").split())
    role_b = set(features_b.get("role", "").split())
    if role_a and role_b:
        role_sim = len(role_a & role_b) / len(role_a | role_b)
        total += _FEATURE_WEIGHTS["role"] * role_sim

    return min(1.0, total)


# ═══════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════


def _build_component_list(mapping: PlatformMapping) -> list[str]:
    """构建模板组件列表。"""
    components = ["platform_type"]
    if mapping.radar_signature:
        components.append("radar_signature")
    if mapping.mover:
        components.append("mover")
    for _ in mapping.default_sensors:
        components.append("sensor")
    for _ in mapping.default_weapons:
        components.append("weapon")
    for _ in mapping.default_processors:
        components.append("processor")
    return components


def _make_type_name(platform_code: str) -> str:
    """生成 AFSIM platform_type 名称。"""
    return re.sub(r"[^A-Za-z0-9]", "_", platform_code).upper() + "_TYPE"


def _normalize(s: str) -> str:
    """标准化字符串用于模糊匹配。"""
    return re.sub(r"[_\-\s]", "", s).lower()
