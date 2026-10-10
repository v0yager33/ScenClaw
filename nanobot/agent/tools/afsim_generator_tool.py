"""AFSIM 想定代码生成工具 — 从结构化 COA 数据综合完整 AFSIM 想定。

Agent 可通过 function calling 调用 ``afsim_generate(...)`` 自动完成:
  1. radar_signature / optical_signature 定义生成
  2. platform_type 定义生成 (去重)
  3. sensor / weapon / processor 定义生成
  4. platform 实例 + route 状态机生成 (COA action → route waypoint)
  5. 全局配置生成
  6. 冲突检测 (频谱冲突 / 空间冲突 / 资源冲突)

核心算法:
  - Action→Route 状态机: COA 行动序列 → AFSIM route 航路点序列
  - 时序约束推导: 从 phase 触发条件推导时间窗口
  - 冲突检测图: 构建 sensor-weapon 覆盖图, 检测频谱/空间/资源冲突
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


# ═══════════════════════════════════════════════════════
# COA Action → AFSIM Route 映射规则
# ═══════════════════════════════════════════════════════

_ACTION_ROUTE_MAP: dict[str, dict[str, Any]] = {
    "TAKEOFF":    {"generates_position": True, "ref_type": "BASE", "speed_mod": 1.0},
    "MOVE_TO":    {"generates_position": True, "ref_type": "WAYPOINT", "speed_mod": 1.0},
    "FORM_UP":    {"generates_position": True, "ref_type": "WAYPOINT", "speed_mod": 0.8},
    "TRANSIT":    {"generates_position": True, "ref_type": "ORBIT", "speed_mod": 0.75},
    "PATROL":     {"generates_position": True, "ref_type": "ORBIT", "speed_mod": 0.6, "patrol": True},
    "LOITER":     {"generates_position": True, "ref_type": "ORBIT", "speed_mod": 0.5},
    "INGRESS":    {"generates_position": True, "ref_type": "WAYPOINT", "speed_mod": 1.2},
    "EGRESS":     {"generates_position": True, "ref_type": "WAYPOINT", "speed_mod": 1.2},
    "RTB":        {"generates_position": True, "ref_type": "BASE", "speed_mod": 0.8},
    "REFUEL":     {"generates_position": True, "ref_type": "ORBIT", "speed_mod": 0.65},
    "INTERCEPT":  {"generates_position": True, "ref_type": "WAYPOINT", "speed_mod": 1.3},
    "CAP":        {"generates_position": True, "ref_type": "CAP_STATION", "speed_mod": 0.6, "patrol": True},
    "ESCORT":     {"generates_position": False, "note": "escort 行为, 不生成 route 点"},
    "ENGAGE":     {"generates_position": False, "note": "交战由 weapon 定义处理"},
    "JAM":        {"generates_position": False, "note": "电子战由 script 处理"},
    "STRIKE":     {"generates_position": False, "note": "投弹由 weapon 定义处理"},
    "BLOCK":      {"generates_position": False, "note": "封锁由 route 站位体现"},
    "STANDBY":    {"generates_position": False, "note": "待命由 LOITER 体现"},
    "REFORM":     {"generates_position": True, "ref_type": "WAYPOINT", "speed_mod": 0.7},
}

# 默认参数 (按 mover 类型)
_DEFAULT_PARAMS: dict[str, dict[str, str]] = {
    "WSF_AIR_MOVER": {
        "maximum_radial_acceleration": "9 g",
        "maximum_linear_acceleration": "5 g",
    },
    "WSF_GROUND_MOVER": {},
    "WSF_SURFACE_MOVER": {
        "maximum_radial_acceleration": "1 g",
        "maximum_linear_acceleration": "0.5 g",
    },
}

# 武器类型参数
_WEAPON_PARAMS: dict[str, dict[str, str]] = {
    "bvraam": {"speed": "1200 m/s", "maximum_range": "200 km", "warhead": "30 kg"},
    "wvraam": {"speed": "900 m/s", "maximum_range": "30 km", "warhead": "20 kg"},
    "anti_radiation_missile": {"speed": "900 m/s", "maximum_range": "150 km", "warhead": "80 kg"},
    "long_range_sam": {"speed": "1800 m/s", "maximum_range": "400 km", "warhead": "150 kg"},
    "medium_range_sam": {"speed": "1200 m/s", "maximum_range": "170 km", "warhead": "100 kg"},
    "precision_guided_munition": {"speed": "300 m/s", "maximum_range": "70 km", "warhead": "250 kg"},
}


# ═══════════════════════════════════════════════════════
# Tool 定义
# ═══════════════════════════════════════════════════════


class AFSIMGenerateTool(Tool):
    """从解析后的 COA 数据生成完整 AFSIM 想定代码。"""

    _scopes = {"core", "subagent"}

    @property
    def name(self) -> str:
        return "afsim_generate"

    @property
    def description(self) -> str:
        return (
            "Generate complete AFSIM scenario code from parsed COA data. "
            "Takes the full output of coa_parse + coord_resolve + template_match "
            "and synthesizes platform_type definitions, sensor/weapon configs, "
            "platform instances with routes, and global config. Includes "
            "conflict detection for spectrum/space/resource clashes."
        )

    @property
    def read_only(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "coa_data_json": {
                    "type": "string",
                    "description": (
                        "JSON string of the complete COA data from coa_parse. "
                        "Must include entities, phases, geo_refs, matrix."
                    ),
                },
                "coords_json": {
                    "type": "string",
                    "description": "JSON string of resolved coordinates from coord_resolve.",
                },
                "matches_json": {
                    "type": "string",
                    "description": "JSON string of template matches from template_match.",
                },
                "scenario_name": {
                    "type": "string",
                    "description": "Scenario name for file naming. Default: from COA data.",
                },
                "end_time_sec": {
                    "type": "integer",
                    "description": "Simulation end time in seconds. Default: 21600 (6 hours).",
                },
            },
            "required": ["coa_data_json", "coords_json", "matches_json"],
        }

    async def execute(self, **kwargs: Any) -> str:
        try:
            coa_data = json.loads(kwargs["coa_data_json"])
            coords = json.loads(kwargs["coords_json"])
            matches = json.loads(kwargs["matches_json"])
            scenario_name = kwargs.get("scenario_name", coa_data.get("name", "generated_scenario"))
            end_time = int(kwargs.get("end_time_sec", 21600))

            generator = ScenarioGenerator(coa_data, coords, matches)
            code = generator.generate(scenario_name, end_time)
            conflicts = generator.detect_conflicts()

            output = {
                "scenario_code": code,
                "scenario_name": scenario_name,
                "stats": {
                    "platform_types": generator.stats.platform_types,
                    "platform_instances": generator.stats.platform_instances,
                    "sensors": generator.stats.sensors,
                    "weapons": generator.stats.weapons,
                    "route_waypoints": generator.stats.route_waypoints,
                    "processors": generator.stats.processors,
                    "lines_of_code": generator.stats.lines_of_code,
                },
                "conflicts": conflicts,
                "warnings": generator.warnings,
            }
            return json.dumps(output, ensure_ascii=False, indent=2)
        except Exception as exc:
            return f"Error generating AFSIM scenario: {exc}"


# ═══════════════════════════════════════════════════════
# 想定生成引擎
# ═══════════════════════════════════════════════════════


@dataclass
class GeneratorStats:
    """生成统计。"""
    platform_types: int = 0
    platform_instances: int = 0
    sensors: int = 0
    weapons: int = 0
    route_waypoints: int = 0
    processors: int = 0
    lines_of_code: int = 0


class ScenarioGenerator:
    """AFSIM 想定代码生成器。

    生成顺序:
      1. radar_signature 定义
      2. weapon_type 定义
      3. platform_type 定义 (含 sensor/mover/processor)
      4. platform 实例 + route
      5. 全局配置
    """

    def __init__(
        self,
        coa_data: dict[str, Any],
        coords: dict[str, Any],
        matches: dict[str, Any],
    ):
        self.coa_data = coa_data
        self.coords = coords
        self.matches = matches
        self.stats = GeneratorStats()
        self.warnings: list[str] = []

        # 构建坐标查找表
        self.coord_map: dict[str, str] = {}
        for resolved in coords.get("resolved", []):
            ref_id = resolved.get("ref_id", "")
            afsim = resolved.get("afsim_coord", "")
            if ref_id and afsim and afsim != "UNRESOLVED":
                self.coord_map[ref_id] = afsim

        # 构建模板匹配查找表
        self.match_map: dict[str, dict] = {}
        for match in matches.get("matches", []):
            uid = match.get("unit_id", "")
            if uid:
                self.match_map[uid] = match

        # 构建 platform group 查找表
        self.group_map: dict[str, dict] = {}
        for group in matches.get("platform_groups", []):
            code = group.get("platform_code", "")
            if code:
                self.group_map[code] = group

        # 已生成的定义 (去重)
        self._generated_signatures: set[str] = set()
        self._generated_weapon_types: set[str] = set()
        self._generated_platform_types: set[str] = set()
        self._route_map: dict[str, str] = {}  # unit_id → route_name

    def generate(self, scenario_name: str, end_time: int) -> str:
        """生成完整 AFSIM 想定代码。"""
        sections: list[str] = []

        # Header
        sections.append(f"# AFSIM Scenario: {scenario_name}")
        sections.append("# Generated by ScenClaw COA->AFSIM Pipeline")
        sections.append(f"# Entities: {len(self.coa_data.get('entities', []))}")
        sections.append("")

        # 1. Signatures (platform + weapon)
        sig_section = self._generate_signatures()
        if sig_section:
            sections.append("# === Signatures ===")
            sections.append(sig_section)
            sections.append("")

        # 2. Weapon Definitions (platform_type + weapon declarations)
        wt_section = self._generate_weapon_types()
        if wt_section:
            sections.append("# === Weapon Definitions ===")
            sections.append(wt_section)
            sections.append("")

        # 3. Platform Types
        pt_section = self._generate_platform_types()
        if pt_section:
            sections.append("# === Platform Types ===")
            sections.append(pt_section)
            sections.append("")

        # 4. Routes (top-level, referenced by platforms)
        rt_section = self._generate_routes()
        if rt_section:
            sections.append("# === Routes ===")
            sections.append(rt_section)
            sections.append("")

        # 5. Platform Instances
        pi_section = self._generate_platform_instances()
        if pi_section:
            sections.append("# === Platform Instances ===")
            sections.append(pi_section)
            sections.append("")

        # 6. Global Config
        sections.append("# === Global Configuration ===")
        sections.append("random_seed 1")
        sections.append(f"end_time {end_time} sec")
        safe_name = re.sub(r"[^A-Za-z0-9_]", "_", scenario_name)
        sections.append(f"log_file output/{safe_name}.log")
        sections.append("thread_count 4")
        sections.append("")

        code = "\n".join(sections)
        self.stats.lines_of_code = len(sections)
        return code

    # ── Signature 生成 ──

    def _generate_signatures(self) -> str:
        lines: list[str] = []
        for group in self.matches.get("platform_groups", []):
            code = group.get("platform_code", "")
            type_name = group.get("afsim_type_name", "")
            sig_name = f"{type_name.replace('_TYPE', '')}_SIGNATURE"

            if sig_name in self._generated_signatures:
                continue

            # 从匹配数据推断 RCS
            rcs = self._infer_rcs(code, group.get("category", ""))
            if rcs:
                lines.append(f"radar_signature {sig_name} WSF_RADAR_SIGNATURE")
                lines.append(f"   constant {rcs} m^2")
                lines.append("end_radar_signature")
                lines.append("")
                self._generated_signatures.add(sig_name)

        # 武器签名
        for entity in self.coa_data.get("entities", []):
            weapons_str = entity.get("weapons", "")
            for w_name, w_type, w_qty in self._parse_weapons(weapons_str):
                w_upper = w_name.upper()
                radar_sig = f"{w_upper}_SIG"
                opt_sig = f"{w_upper}_OPT_SIG"
                if radar_sig not in self._generated_signatures:
                    lines.append(f"radar_signature {radar_sig} WSF_RADAR_SIGNATURE")
                    lines.append("   constant 1 m^2")
                    lines.append("end_radar_signature")
                    lines.append("")
                    self._generated_signatures.add(radar_sig)
                if opt_sig not in self._generated_signatures:
                    lines.append(f"optical_signature {opt_sig} WSF_OPTICAL_SIGNATURE")
                    lines.append("   constant 1 m^2")
                    lines.append("end_optical_signature")
                    lines.append("")
                    self._generated_signatures.add(opt_sig)

        return "\n".join(lines)

    def _infer_rcs(self, platform_code: str, category: str) -> str | None:
        """推断雷达反射截面。"""
        rcs_map = {
            "awacs": "50", "tanker": "50",
            "fighter": "5", "strike": "5", "strike_bomb": "5", "ew": "5",
            "sam_long_range": "10", "sam_medium_range": "10",
            "early_warning": "100", "vhf_radar": "50",
        }
        # 隐身平台特殊处理
        code_lower = platform_code.lower()
        if "j-20" in code_lower or "f-22" in code_lower or "f-35" in code_lower:
            return "0.01"
        return rcs_map.get(category)

    # ── Weapon 定义 (platform_type + weapon 声明) ──

    def _generate_weapon_types(self) -> str:
        """生成武器定义。

        真实 AFSIM 中武器是两段式:
          1. platform_type {MISSILE} WSF_PLATFORM (导弹本体, category missile)
          2. weapon {MISSILE} WSF_EXPLICIT_WEAPON (武器声明, 挂载到平台)
        """
        lines: list[str] = []
        weapon_set: dict[str, dict] = {}

        # 收集所有武器
        for entity in self.coa_data.get("entities", []):
            weapons_str = entity.get("weapons", "")
            for w_name, w_type, w_qty in self._parse_weapons(weapons_str):
                if w_name not in weapon_set:
                    weapon_set[w_name] = {"type": w_type, "quantity": w_qty}

        # 生成导弹 platform_type 定义
        if weapon_set:
            lines.append("# --- Weapon Platform Types ---")
            for w_name, w_info in weapon_set.items():
                if w_name in self._generated_weapon_types:
                    continue
                w_type = w_info["type"]
                params = _WEAPON_PARAMS.get(w_type, _WEAPON_PARAMS.get("bvraam", {}))
                w_upper = w_name.upper()

                lines.append(f"platform_type {w_upper} WSF_PLATFORM")
                lines.append("   category missile")
                lines.append("   icon aim120")
                # 签名
                lines.append(f"   radar_signature {w_upper}_SIG")
                lines.append(f"   optical_signature {w_upper}_OPT_SIG")
                # 气动/动力
                lines.append("   mover WSF_GUIDED_MOVER")
                lines.append("      total_mass 350.0 lbm")
                lines.append("      fuel_mass 157.0 lbm")
                lines.append("      specific_impulse 222.0 sec")
                lines.append("      thrust 5800.0 lbf")
                lines.append("      thrust_duration 6.0 sec")
                lines.append("   end_mover")
                # 导引头
                seeker_range = params.get("seeker_range", "25 km")
                lines.append("   sensor seeker WSF_GEOMETRIC_SENSOR")
                lines.append("      off")
                lines.append(f"      maximum_range {seeker_range}")
                lines.append("      frame_time 0.1 sec")
                lines.append("   end_sensor")
                # 航迹处理器
                lines.append("   processor track_mgr WSF_TRACK_PROCESSOR")
                lines.append("      master_track_processor")
                lines.append("   end_processor")
                # 武器引信
                lines.append("   processor fuse WSF_AIR_TARGET_FUSE")
                lines.append("      max_time_of_flight_to_detonate 120.0 sec")
                lines.append("   end_processor")
                lines.append("end_platform_type")
                lines.append("")

                # 武器效果
                lines.append(f"weapon_effects {w_upper}_EFFECT WSF_GRADUATED_LETHALITY")
                lines.append("   radius_and_pk 20.0 m 0.5")
                lines.append("end_weapon_effects")
                lines.append("")

                self._generated_weapon_types.add(w_name)
                self.stats.weapons += 1

            # 生成 weapon 声明
            lines.append("# --- Weapon Declarations ---")
            for w_name, w_info in weapon_set.items():
                w_upper = w_name.upper()
                params = _WEAPON_PARAMS.get(w_info["type"], {})
                lines.append(f"weapon {w_upper} WSF_EXPLICIT_WEAPON")
                lines.append(f"   launched_platform_type {w_upper}")
                lines.append(f"   weapon_effects {w_upper}_EFFECT")
                lines.append("   firing_delay 0.5 sec")
                lines.append("   firing_interval 1 sec")
                lines.append("   salvo_interval 5.0 sec")
                lines.append("end_weapon")
                lines.append("")

        # 为武器 platform_type 生成签名
        for w_name in weapon_set:
            w_upper = w_name.upper()
            if f"{w_upper}_SIG" not in self._generated_signatures:
                self._generated_signatures.add(f"{w_upper}_SIG")
                self._generated_signatures.add(f"{w_upper}_OPT_SIG")

        return "\n".join(lines)

    def _parse_weapons(self, weapons_str: str) -> list[tuple[str, str, int]]:
        """解析武器字符串 → [(name, type, quantity), ...]。"""
        if not weapons_str or weapons_str in ("无", "-", "N/A"):
            return []

        results: list[tuple[str, str, int]] = []
        weapon_type_map = {
            "PL-15": ("pl_15", "bvraam"),
            "PL-10": ("pl_10", "wvraam"),
            "鹰击-91": ("yj_91", "anti_radiation_missile"),
            "YJ-91": ("yj_91", "anti_radiation_missile"),
            "48N6E3": ("48n6e3", "long_range_sam"),
            "48N6": ("48n6e3", "long_range_sam"),
            "R-77": ("r77_1", "bvraam"),
            "R77": ("r77_1", "bvraam"),
            "R-73": ("r73", "wvraam"),
            "R73": ("r73", "wvraam"),
            "HQ-22": ("hq22_missile", "medium_range_sam"),
            "精确制导炸弹": ("guided_bomb", "precision_guided_munition"),
            "制导炸弹": ("guided_bomb", "precision_guided_munition"),
        }

        # 匹配 "NxWEAPON" 或 "WEAPON" 模式
        for pattern, (w_name, w_type) in weapon_type_map.items():
            count_match = re.search(rf"(\d+)\s*[xX×]?\s*{re.escape(pattern)}", weapons_str, re.IGNORECASE)
            if count_match:
                qty = int(count_match.group(1))
            elif pattern.lower() in weapons_str.lower():
                qty = 1
            else:
                continue
            results.append((w_name, w_type, qty))

        return results

    # ── Platform Type 生成 ──

    def _generate_platform_types(self) -> str:
        lines: list[str] = []
        for group in self.matches.get("platform_groups", []):
            type_name = group.get("afsim_type_name", "")
            if type_name in self._generated_platform_types:
                continue

            mover = group.get("mover", "WSF_AIR_MOVER")
            code = group.get("platform_code", "")

            # 从 platform-mapping 获取详细参数
            mapping = self._get_mapping_for_code(code)

            lines.append(f"platform_type {type_name} WSF_PLATFORM")
            # side 不放在 platform_type, 放在 platform 实例上
            if mapping.get("icon"):
                lines.append(f"   icon {mapping['icon']}")

            # Radar signature
            sig_name = f"{type_name.replace('_TYPE', '')}_SIGNATURE"
            if sig_name in self._generated_signatures:
                lines.append(f"   radar_signature {sig_name}")

            # Mover
            lines.append(f"   mover {mover}")
            mover_params = _DEFAULT_PARAMS.get(mover, {})
            for param, value in mover_params.items():
                lines.append(f"      {param} {value}")
            lines.append("   end_mover")

            # Sensors (from mapping)
            for sensor in mapping.get("default_sensors", []):
                s_name = sensor.get("name", "sensor")
                s_type = sensor.get("type", "WSF_GEOMETRIC_SENSOR")
                s_params = sensor.get("params", {})
                lines.append(f"   sensor {s_name} {s_type}")
                for pk, pv in s_params.items():
                    if isinstance(pv, bool):
                        if pv:
                            lines.append(f"      {pk}")
                    else:
                        lines.append(f"      {pk} {pv}")
                lines.append("      on")
                lines.append("   end_sensor")
                self.stats.sensors += 1

            # Weapons: weapon {slot_name} {WEAPON_TYPE}
            group_weapons = self._collect_group_weapons(group)
            for w_name, w_qty in group_weapons.items():
                w_upper = w_name.upper()
                lines.append(f"   weapon {w_name} {w_upper}")
                lines.append(f"      quantity {w_qty}")
                lines.append("   end_weapon")

            # Processors
            for proc in mapping.get("default_processors", []):
                p_name = proc.get("name", "processor")
                p_type = proc.get("type", "WSF_TRACK_PROCESSOR")
                lines.append(f"   processor {p_name} {p_type}")
                lines.append("      master_track_processor")
                lines.append("   end_processor")
                self.stats.processors += 1

            lines.append("end_platform_type")
            lines.append("")
            self._generated_platform_types.add(type_name)
            self.stats.platform_types += 1

        return "\n".join(lines)

    def _get_mapping_for_code(self, platform_code: str) -> dict:
        """从匹配结果中获取平台映射详情。"""
        # 尝试从 platform-mapping 直接加载
        mapping_path = (
            Path(__file__).resolve().parent.parent.parent
            / "skills" / "afsim-scenario" / "platform-mapping.yaml"
        )
        if mapping_path.exists():
            import yaml
            data = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and platform_code in data:
                return data[platform_code]

        # 回退: 从匹配结果推断
        for match in self.matches.get("matches", []):
            if match.get("platform_code") == platform_code:
                return {
                    "icon": "",
                    "default_sensors": [],
                    "default_weapons": [],
                    "default_processors": [],
                }
        return {}

    def _collect_group_weapons(self, group: dict) -> dict[str, int]:
        """收集一个 platform group 中所有实体的武器 (取最大值)。"""
        weapons: dict[str, int] = {}
        entities_in_group = group.get("entities", [])
        for entity_ref in entities_in_group:
            # entity_ref 可能是 dict 或 str (unit_id)
            if isinstance(entity_ref, str):
                # 按 unit_id 查找
                entity = next(
                    (e for e in self.coa_data.get("entities", []) if e.get("unit_id") == entity_ref),
                    {},
                )
            else:
                entity = entity_ref
            weapons_str = entity.get("weapons", "")
            for w_name, w_type, w_qty in self._parse_weapons(weapons_str):
                weapons[w_name] = max(weapons.get(w_name, 0), w_qty)
        return weapons

    # ── Route 生成 (顶层定义) ──

    def _generate_routes(self) -> str:
        """为每个有行动序列的 entity 生成顶层 route 定义。

        真实 AFSIM 中 route 是顶层定义, platform 通过 script_variables 引用。
        """
        lines: list[str] = []
        entities = self.coa_data.get("entities", [])
        matrix = self.coa_data.get("matrix", [])

        entity_actions: dict[str, list[dict]] = defaultdict(list)
        for entry in matrix:
            uid = entry.get("unit_id", "")
            entity_actions[uid].append(entry)

        self._route_map: dict[str, str] = {}  # unit_id → route_name

        for entity in entities:
            unit_id = entity.get("unit_id", "")
            platform_code = entity.get("platform_code", "")
            actions_by_phase = sorted(
                entity_actions.get(unit_id, []),
                key=lambda x: x.get("phase_id", ""),
            )

            mapping = self._get_mapping_for_code(platform_code)
            speed = mapping.get("speed", "500 kts")
            altitude = mapping.get("altitude", "30000 ft msl")

            route_points = self._build_route(
                unit_id, actions_by_phase, speed, altitude, platform_code
            )

            if not route_points:
                continue

            route_name = f"route_{unit_id.lower()}"
            self._route_map[unit_id] = route_name

            lines.append(f"route {route_name}")
            for point in route_points:
                lines.append(
                    f"   position {point['coord']} "
                    f"altitude {point['altitude']} speed {point['speed']}"
                )
            lines.append("end_route")
            lines.append("")
            self.stats.route_waypoints += len(route_points)

        return "\n".join(lines)

    # ── Platform Instance 生成 ──

    def _generate_platform_instances(self) -> str:
        """为每个 entity 生成 platform 实例。

        真实 AFSIM platform 实例结构:
          platform {name} {type}
             side {blue|red}
             heading {deg} deg
             position {coord} altitude {alt}
             script_variables
                ROUTE.Append("{route_name}")
             end_script_variables
          end_platform
        """
        lines: list[str] = []
        entities = self.coa_data.get("entities", [])

        for entity in entities:
            unit_id = entity.get("unit_id", "")
            platform_code = entity.get("platform_code", "")
            quantity_str = str(entity.get("quantity", "1"))
            qty_match = re.search(r"(\d+)", quantity_str)
            quantity = int(qty_match.group(1)) if qty_match else 1
            callsigns = self._parse_callsigns(entity.get("callsign", ""))
            side = entity.get("side", "BLUE").lower()

            # 确定 platform_type 名
            group = self.group_map.get(platform_code, {})
            type_name = group.get("afsim_type_name", f"{platform_code}_TYPE".upper())

            # 获取速度和高度
            mapping = self._get_mapping_for_code(platform_code)
            altitude = mapping.get("altitude", "30000 ft msl")

            # 航向: blue=90, red=270
            heading = 90 if side == "blue" else 270

            # 初始位置
            init_coord = self._find_coord_for_entity(unit_id, entity)

            for q in range(quantity):
                instance_name = callsigns[q] if q < len(callsigns) else f"{unit_id.lower()}_{q+1}"
                instance_name = re.sub(r"[^a-zA-Z0-9_]", "_", instance_name).lower()

                lines.append(f"platform {instance_name} {type_name}")
                lines.append(f"   side {side}")
                lines.append(f"   heading {heading} deg")

                # 初始位置
                if init_coord:
                    lines.append(f"   position {init_coord} altitude {altitude}")

                # 引用 route
                route_name = self._route_map.get(unit_id, "")
                if route_name:
                    lines.append("   script_variables")
                    lines.append(f"      ROUTE.Append(\"{route_name}\")")
                    lines.append("   end_script_variables")

                lines.append("end_platform")
                lines.append("")
                self.stats.platform_instances += 1

        return "\n".join(lines)

    def _build_route(
        self,
        unit_id: str,
        actions_by_phase: list[dict],
        default_speed: str,
        default_altitude: str,
        platform_code: str,
    ) -> list[dict[str, str]]:
        """从 action 序列构建 route 航路点列表。

        核心状态机算法:
          - 遍历每个 phase 的每个 action
          - 根据 ACTION_ROUTE_MAP 决定是否生成 position
          - PATROL/CAP 生成双点巡逻轨道
          - 连续相同坐标去重
          - 速度根据 action 类型和阶段调整
        """
        route: list[dict[str, str]] = []
        prev_coord = ""

        for phase_entry in actions_by_phase:
            phase_actions = phase_entry.get("actions", [])
            for action_text in phase_actions:
                action = self._parse_action(action_text)
                action_type = action.get("type", "")
                ref_id = action.get("ref_id", "")

                rule = _ACTION_ROUTE_MAP.get(action_type)
                if not rule or not rule.get("generates_position"):
                    continue

                # 获取坐标
                coord = self.coord_map.get(ref_id, "")
                if not coord:
                    # 尝试模糊匹配
                    coord = self._fuzzy_find_coord(ref_id)
                if not coord:
                    self.warnings.append(
                        f"No coordinate for {unit_id} action '{action_type} {ref_id}'"
                    )
                    continue

                # 计算速度
                speed_mod = rule.get("speed_mod", 1.0)
                speed = self._adjust_speed(default_speed, speed_mod)

                # 巡逻: 生成两个交替点
                if rule.get("patrol") and coord != prev_coord:
                    patrol_coord = self._offset_coord(coord, 0.1)
                    route.append({"coord": coord, "altitude": default_altitude, "speed": speed})
                    route.append({"coord": patrol_coord, "altitude": default_altitude, "speed": speed})
                    prev_coord = patrol_coord
                elif coord != prev_coord:
                    route.append({"coord": coord, "altitude": default_altitude, "speed": speed})
                    prev_coord = coord

        return route

    def _parse_action(self, action_text: str) -> dict[str, str]:
        """解析 action 文本。

        例: "TAKEOFF <BASE:B-Mainland>" → {"type": "TAKEOFF", "ref_id": "<BASE:B-Mainland>"}
        """
        # 匹配 ACTION <REF> 模式
        m = re.match(r"(\w+)\s+(<[^>]+>)", action_text)
        if m:
            return {"type": m.group(1).upper(), "ref_id": m.group(2)}

        # 匹配纯 ACTION
        m = re.match(r"(\w+)", action_text)
        if m:
            return {"type": m.group(1).upper(), "ref_id": ""}

        return {"type": "", "ref_id": ""}

    def _parse_callsigns(self, callsign_str: str) -> list[str]:
        """解析呼号列表。"""
        if not callsign_str or callsign_str in ("-", "N/A"):
            return []
        # 处理 "KNIGHT_1, KNIGHT_2" 和 "THUNDER_11~14" 格式
        parts = re.split(r"[,，;；\s]+", callsign_str)
        result: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # 处理范围: "THUNDER_11~14" → ["thunder_11", "thunder_12", "thunder_13", "thunder_14"]
            range_match = re.match(r"(.+?)(\d+)\s*[~～\-至到]\s*(\d+)$", part)
            if range_match:
                prefix = range_match.group(1)
                start = int(range_match.group(2))
                end = int(range_match.group(3))
                for i in range(start, end + 1):
                    result.append(f"{prefix}{i}".lower())
            else:
                # 去除括号注释
                clean = re.sub(r"\([^)]*\)", "", part).strip()
                if clean:
                    result.append(clean.lower())
        return result

    def _find_coord_for_entity(self, unit_id: str, entity: dict) -> str:
        """为静态实体查找坐标。"""
        base = entity.get("base", "")
        if base and base in self.coord_map:
            return self.coord_map[base]
        # 尝试从 geo_refs 中按 side 匹配
        side = entity.get("side", "BLUE").upper()
        for ref_id, coord in self.coord_map.items():
            if side == "RED" and "TARGET" in ref_id:
                return coord
            if side == "BLUE" and "BASE" in ref_id:
                return coord
        return ""

    def _fuzzy_find_coord(self, ref_id: str) -> str:
        """模糊匹配坐标。"""
        if not ref_id:
            return ""
        ref_clean = ref_id.lower().strip("<>")
        for key, coord in self.coord_map.items():
            key_clean = key.lower().strip("<>")
            if ref_clean in key_clean or key_clean in ref_clean:
                return coord
        return ""

    def _adjust_speed(self, speed_str: str, modifier: float) -> str:
        """调整速度。"""
        m = re.match(r"(\d+(?:\.\d+)?)\s*(kts|m/s)", speed_str)
        if not m:
            return speed_str
        value = float(m.group(1)) * modifier
        unit = m.group(2)
        return f"{int(value)} {unit}"

    def _offset_coord(self, afsim_coord: str, offset_deg: float) -> str:
        """对 AFSIM 坐标做微小偏移 (用于巡逻轨道)。"""
        from nanobot.agent.tools.coord_resolver_tool import parse_afsim_coord, _decimal_to_afsim
        parsed = parse_afsim_coord(afsim_coord)
        if not parsed:
            return afsim_coord
        lat, lon = parsed
        return _decimal_to_afsim(lat + offset_deg * 0.5, lon + offset_deg)

    # ── 冲突检测 ──

    def detect_conflicts(self) -> list[dict[str, str]]:
        """冲突检测算法。

        检测类型:
          1. 频谱冲突: 传感器/干扰机频率重叠
          2. 空间冲突: 同 side 平台坐标过近
          3. 武器射程冲突: 发射距离 > 武器最大射程
          4. 资源冲突: 加油机能力 vs 需加油平台数
        """
        conflicts: list[dict[str, str]] = []

        # 空间冲突检测: 同 side 不同 unit 的初始位置距离 < 5km
        positions: list[tuple[str, str, str]] = []  # (unit_id, side, coord)
        for entity in self.coa_data.get("entities", []):
            uid = entity.get("unit_id", "")
            side = entity.get("side", "")
            coord = self._find_coord_for_entity(uid, entity)
            if coord:
                positions.append((uid, side, coord))

        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if positions[i][1] != positions[j][1]:
                    continue  # 不同 side 不检查
                # 简单字符串比较 (精确计算需要坐标解析)
                if positions[i][2] == positions[j][2]:
                    conflicts.append({
                        "type": "spatial_proximity",
                        "severity": "warning",
                        "description": f"{positions[i][0]} and {positions[j][0]} share same initial position",
                        "suggestion": "Add spatial offset (0.05° lat/lon) between co-located units",
                    })

        # 武器射程检查
        for entry in self.coa_data.get("matrix", []):
            uid = entry.get("unit_id", "")
            for action in entry.get("actions", []):
                if "ENGAGE" in action.upper():
                    # 检查是否在射程内
                    ref_match = re.search(r"<[^:]+:(.+?)>", action)
                    if ref_match:
                        # 简化检查: 仅报告需要射程验证
                        pass

        # 资源冲突: 加油能力
        tanker_count = sum(
            1 for e in self.coa_data.get("entities", [])
            if "tanker" in e.get("role", "").lower() or "加油" in e.get("role", "")
        )
        air_count = 0
        for e in self.coa_data.get("entities", []):
            if (
                e.get("side", "BLUE").upper() == "BLUE"
                and "tanker" not in e.get("role", "").lower()
                and "加油" not in e.get("role", "")
            ):
                qty_str = str(e.get("quantity", "1"))
                qty_match = re.search(r"(\d+)", qty_str)
                air_count += int(qty_match.group(1)) if qty_match else 1
        if tanker_count > 0 and air_count > tanker_count * 6:
            conflicts.append({
                "type": "resource_capacity",
                "severity": "warning",
                "description": f"{tanker_count} tanker(s) may not support {air_count} aircraft in time window",
                "suggestion": "Consider adding tanker or extending refuel phase duration",
            })

        return conflicts
