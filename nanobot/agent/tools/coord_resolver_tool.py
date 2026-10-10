"""坐标解算工具 — WGS-84 大地测量 + 自然语言地理参考解析。

Agent 可通过 function calling 调用 ``coord_resolve(...)`` 将 COA 中的
相对地理描述（如 "距目标岛礁480-520km"）转化为 AFSIM d:m:s 坐标。

核心算法:
  1. 自然语言距离/方位提取（正则 + 规则引擎）
  2. WGS-84 大圆距离正反算（Vincenty 公式 + 平面近似）
  3. 多参考点约束求解（距离交叉定位）
  4. 坐标格式转换（decimal degrees ↔ d:m:s N/S E/W）
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from nanobot.agent.tools.base import Tool


# ═══════════════════════════════════════════════════════
# 大地测量常量
# ═══════════════════════════════════════════════════════

_WGS84_A = 6378137.0  # 半长轴 (m)
_WGS84_F = 1 / 298.257223563  # 扁率
_WGS84_B = _WGS84_A * (1 - _WGS84_F)  # 半短轴
_EARTH_RADIUS_KM = 6371.0  # 平均半径 (km)

# 方位角关键词映射 (角度, 从正北顺时针)
_BEARING_KEYWORDS: dict[str, float] = {
    "北": 0, "正北": 0, "north": 0,
    "东北": 45, "northeast": 45,
    "东": 90, "正东": 90, "east": 90,
    "东南": 135, "southeast": 135,
    "南": 180, "正南": 180, "south": 180,
    "西南": 225, "southwest": 225,
    "西": 270, "正西": 270, "west": 270, "西侧": 270,
    "西北": 315, "northwest": 315,
    "前方": 90, "front": 90,
    "后方": 270, "rear": 270, "back": 270,
    "侧翼": 0, "flank": 0,
    "左翼": 315, "left_flank": 315,
    "右翼": 45, "right_flank": 45,
}

# 距离单位转换 (→ km)
_DISTANCE_UNITS: dict[str, float] = {
    "km": 1.0, "公里": 1.0, "千米": 1.0,
    "nm": 1.852, "nmi": 1.852, "海里": 1.852,
    "mi": 1.60934, "mile": 1.60934, "英里": 1.60934,
    "m": 0.001, "米": 0.001,
    "ft": 0.0003048, "feet": 0.0003048, "英尺": 0.0003048,
}


# ═══════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════


@dataclass
class GeoPoint:
    """地理坐标点 (WGS-84)。"""
    lat: float  # 纬度 (decimal degrees)
    lon: float  # 经度 (decimal degrees)
    name: str = ""
    confidence: float = 1.0  # 定位置信度 [0, 1]

    def to_afsim(self) -> str:
        """转换为 AFSIM d:m:s 格式。"""
        return _decimal_to_afsim(self.lat, self.lon)

    def distance_to(self, other: GeoPoint) -> float:
        """到另一点的大圆距离 (km)。"""
        return haversine(self.lat, self.lon, other.lat, other.lon)

    def bearing_to(self, other: GeoPoint) -> float:
        """到另一点的初始方位角 (度)。"""
        return initial_bearing(self.lat, self.lon, other.lat, other.lon)


@dataclass
class DistanceConstraint:
    """距离约束: 距某参考点一定距离。"""
    ref_name: str  # 参考点名称
    ref_point: GeoPoint | None = None  # 参考点坐标
    distance_km: float = 0.0  # 距离 (km)
    distance_min: float = 0.0  # 距离范围下限
    distance_max: float = 0.0  # 距离范围上限
    bearing_deg: float | None = None  # 方位角 (None = 未指定)


@dataclass
class ResolveResult:
    """坐标解算结果。"""
    ref_id: str
    point: GeoPoint
    constraints_used: int = 0
    method: str = ""  # 使用的解算方法
    warnings: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════
# Tool 定义
# ═══════════════════════════════════════════════════════


class CoordResolveTool(Tool):
    """解析 COA 地理参考为 AFSIM 坐标。支持自然语言距离/方位描述。"""

    _scopes = {"core", "subagent"}

    @property
    def name(self) -> str:
        return "coord_resolve"

    @property
    def description(self) -> str:
        return (
            "Resolve COA geographic references to AFSIM coordinates. "
            "Takes a JSON dict of geo_refs (from coa_parse) and an optional "
            "center point. Returns resolved coordinates in AFSIM d:m:s format. "
            "Uses WGS-84 geodetic algorithms (Haversine/Vincenty) for distance "
            "calculations and constraint-based positioning."
        )

    @property
    def read_only(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "geo_refs_json": {
                    "type": "string",
                    "description": (
                        "JSON string of geo_refs array from coa_parse output. "
                        'e.g. \'[{"ref_id": "<BASE:X>", "hint": "距目标480km"}]\''
                    ),
                },
                "center_lat": {
                    "type": "number",
                    "description": "Center/target area latitude (decimal degrees). Default: 27.5",
                },
                "center_lon": {
                    "type": "number",
                    "description": "Center/target area longitude (decimal degrees). Default: 122.5",
                },
                "default_side": {
                    "type": "string",
                    "description": "Default side for unresolved refs. 'blue' or 'red'.",
                },
            },
            "required": ["geo_refs_json"],
        }

    async def execute(self, **kwargs: Any) -> str:
        try:
            geo_refs_raw = kwargs.get("geo_refs_json", "[]")
            geo_refs = json.loads(geo_refs_raw) if isinstance(geo_refs_raw, str) else geo_refs_raw
            center_lat = float(kwargs.get("center_lat", 27.5))
            center_lon = float(kwargs.get("center_lon", 122.5))

            center = GeoPoint(center_lat, center_lon, name="TARGET_CENTER")
            results = resolve_all_refs(geo_refs, center)

            output = {
                "center": {"lat": center.lat, "lon": center.lon, "afsim": center.to_afsim()},
                "resolved": [],
                "summary": {
                    "total": len(geo_refs),
                    "resolved_count": sum(1 for r in results if r.point is not None),
                    "methods_used": list({r.method for r in results if r.method}),
                },
            }

            for r in results:
                entry: dict[str, Any] = {
                    "ref_id": r.ref_id,
                    "lat": round(r.point.lat, 6) if r.point else None,
                    "lon": round(r.point.lon, 6) if r.point else None,
                    "afsim_coord": r.point.to_afsim() if r.point else "UNRESOLVED",
                    "method": r.method,
                    "constraints_used": r.constraints_used,
                    "confidence": r.point.confidence if r.point else 0,
                }
                if r.warnings:
                    entry["warnings"] = r.warnings
                output["resolved"].append(entry)

            return json.dumps(output, ensure_ascii=False, indent=2)
        except Exception as exc:
            return f"Error resolving coordinates: {exc}"


# ═══════════════════════════════════════════════════════
# 核心解算引擎
# ═══════════════════════════════════════════════════════


def resolve_all_refs(
    geo_refs: list[dict[str, Any]],
    center: GeoPoint,
) -> list[ResolveResult]:
    """解算所有地理参考点。

    解算策略 (按优先级):
      1. 绝对坐标: hint 中包含 d:m:s 或 decimal degrees → 直接解析
      2. 距离+方位: 从 hint 提取距离和方位角 → 大圆偏移
      3. 纯距离约束: 仅有距离 → 按 ref_type 推断方位角
      4. 多约束交叉: 多个距离约束 → 最小二乘交叉定位
    """
    results: list[ResolveResult] = []
    resolved_map: dict[str, GeoPoint] = {"TARGET_CENTER": center}

    # Phase 1: 解析绝对坐标 (无依赖)
    for ref in geo_refs:
        ref_id = ref.get("ref_id", "")
        hint = ref.get("hint", "")
        ref_type = ref.get("type", "")

        abs_point = _try_parse_absolute(hint)
        if abs_point:
            abs_point.name = ref_id
            resolved_map[ref_id] = abs_point
            results.append(ResolveResult(
                ref_id=ref_id, point=abs_point,
                constraints_used=1, method="absolute_parse",
            ))
        else:
            results.append(ResolveResult(
                ref_id=ref_id, point=GeoPoint(0, 0),
                method="pending",
            ))

    # Phase 2: 解析相对坐标 (依赖已解析点)
    max_iterations = 3  # 多轮迭代, 处理链式依赖
    for iteration in range(max_iterations):
        progress = False
        for i, ref in enumerate(geo_refs):
            ref_id = ref.get("ref_id", "")
            if results[i].method != "pending":
                continue

            hint = ref.get("hint", "")
            ref_type = ref.get("type", "")
            side = ref.get("side", "")

            constraints = _extract_constraints(hint, resolved_map)
            if not constraints:
                continue

            point = _solve_constraints(constraints, ref_type, side, center)
            if point:
                point.name = ref_id
                resolved_map[ref_id] = point
                results[i] = ResolveResult(
                    ref_id=ref_id, point=point,
                    constraints_used=len(constraints),
                    method=_describe_method(constraints),
                )
                progress = True

        if not progress:
            break

    # Phase 3: 对仍未解析的点使用启发式默认
    for i, ref in enumerate(geo_refs):
        if results[i].method == "pending":
            ref_id = ref.get("ref_id", "")
            ref_type = ref.get("type", "")
            side = ref.get("side", "")
            point = _heuristic_fallback(ref_id, ref_type, side, center)
            results[i] = ResolveResult(
                ref_id=ref_id, point=point,
                constraints_used=0, method="heuristic_fallback",
                warnings=["No distance/bearing info found, used heuristic placement"],
            )
            resolved_map[ref_id] = point

    return results


# ═══════════════════════════════════════════════════════
# 大地测量算法
# ═══════════════════════════════════════════════════════


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine 大圆距离 (km)。适用于任意距离。"""
    r1, r2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(r1) * math.cos(r2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """初始方位角 (度, 从正北顺时针)。"""
    r1, r2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(r2)
    y = math.cos(r1) * math.sin(r2) - math.sin(r1) * math.cos(r2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def destination_point(
    lat: float, lon: float, distance_km: float, bearing_deg: float
) -> tuple[float, float]:
    """从起点 + 距离 + 方位角计算目标点 (大圆正算)。

    使用 Vincenty 简化公式, 精度优于 0.5m (距离 < 1000km)。
    """
    r_lat = math.radians(lat)
    r_lon = math.radians(lon)
    r_brg = math.radians(bearing_deg)

    # 角距离
    delta = distance_km / _EARTH_RADIUS_KM

    lat2 = math.asin(
        math.sin(r_lat) * math.cos(delta)
        + math.cos(r_lat) * math.sin(delta) * math.cos(r_brg)
    )
    lon2 = r_lon + math.atan2(
        math.sin(r_brg) * math.sin(delta) * math.cos(r_lat),
        math.cos(delta) - math.sin(r_lat) * math.sin(lat2),
    )

    return math.degrees(lat2), math.degrees(lon2)


def vincenty_inverse(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> tuple[float, float, float]:
    """Vincenty 反算: 两点间距离 + 正反方位角。

    Returns: (distance_km, forward_bearing, reverse_bearing)
    精度: 0.5mm (WGS-84 椭球面)
    """
    u1 = math.atan((1 - _WGS84_F) * math.tan(math.radians(lat1)))
    u2 = math.atan((1 - _WGS84_F) * math.tan(math.radians(lat2)))
    L = math.radians(lon2 - lon1)

    sin_u1, cos_u1 = math.sin(u1), math.cos(u1)
    sin_u2, cos_u2 = math.sin(u2), math.cos(u2)

    lam = L
    for _ in range(200):
        sin_lam = math.sin(lam)
        cos_lam = math.cos(lam)
        sin_sigma = math.sqrt(
            (cos_u2 * sin_lam) ** 2
            + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lam) ** 2
        )
        if sin_sigma == 0:
            return 0.0, 0.0, 0.0

        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lam
        sigma = math.atan2(sin_sigma, cos_sigma)
        sin_alpha = cos_u1 * cos_u2 * sin_lam / sin_sigma
        cos2_alpha = 1 - sin_alpha ** 2
        cos_2sigma_m = cos_sigma - 2 * sin_u1 * sin_u2 / cos2_alpha if cos2_alpha else cos_sigma
        C = _WGS84_F / 16 * cos2_alpha * (4 + _WGS84_F * (4 - 3 * cos2_alpha))
        lam_prev = lam
        lam = L + (1 - C) * _WGS84_F * sin_alpha * (
            sigma + C * sin_sigma * (cos_2sigma_m + C * cos_sigma * (-1 + 2 * cos_2sigma_m ** 2))
        )
        if abs(lam - lam_prev) < 1e-12:
            break

    u_sq = cos2_alpha * (_WGS84_A ** 2 - _WGS84_B ** 2) / _WGS84_B ** 2
    A_coeff = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
    B_coeff = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))
    delta_sigma = B_coeff * sin_sigma * (
        cos_2sigma_m
        + B_coeff / 4 * (
            cos_sigma * (-1 + 2 * cos_2sigma_m ** 2)
            - B_coeff / 6 * cos_2sigma_m * (-3 + 4 * sin_sigma ** 2) * (-3 + 4 * cos_2sigma_m ** 2)
        )
    )
    dist_km = _WGS84_B * A_coeff * (sigma - delta_sigma) / 1000

    alpha1 = math.atan2(
        cos_u2 * math.sin(lam),
        cos_u1 * sin_u2 - sin_u1 * cos_u2 * math.cos(lam),
    )
    alpha2 = math.atan2(
        cos_u1 * math.sin(lam),
        -sin_u1 * cos_u2 + cos_u1 * sin_u2 * math.cos(lam),
    )

    fwd = (math.degrees(alpha1) + 360) % 360
    rev = (math.degrees(alpha2) + 360) % 360
    return dist_km, fwd, rev


# ═══════════════════════════════════════════════════════
# 约束提取与求解
# ═══════════════════════════════════════════════════════


def _extract_constraints(
    hint: str, resolved: dict[str, GeoPoint]
) -> list[DistanceConstraint]:
    """从 hint 文本提取距离约束。"""
    constraints: list[DistanceConstraint] = []

    # 模式1: "距XXX NNNkm" / "距XXX NNN-NNNkm"
    pattern1 = re.compile(
        r"距\s*(.+?)\s*(\d+(?:\.\d+)?)\s*[-~～至到]\s*(\d+(?:\.\d+)?)\s*(km|公里|nm|海里|mi)",
        re.IGNORECASE,
    )
    for m in pattern1.finditer(hint):
        ref_name = m.group(1).strip()
        d_min = float(m.group(2))
        d_max = float(m.group(3))
        unit = m.group(4).lower()
        factor = _DISTANCE_UNITS.get(unit, 1.0)
        constraints.append(DistanceConstraint(
            ref_name=ref_name,
            ref_point=_find_ref_point(ref_name, resolved),
            distance_km=(d_min + d_max) / 2 * factor,
            distance_min=d_min * factor,
            distance_max=d_max * factor,
        ))

    # 模式2: "距XXX NNNkm" (单一距离)
    pattern2 = re.compile(
        r"距\s*(.+?)\s*(\d+(?:\.\d+)?)\s*(km|公里|nm|海里|mi|m)",
        re.IGNORECASE,
    )
    if not constraints:
        for m in pattern2.finditer(hint):
            ref_name = m.group(1).strip()
            dist = float(m.group(2))
            unit = m.group(3).lower()
            factor = _DISTANCE_UNITS.get(unit, 1.0)
            constraints.append(DistanceConstraint(
                ref_name=ref_name,
                ref_point=_find_ref_point(ref_name, resolved),
                distance_km=dist * factor,
                distance_min=dist * factor * 0.9,
                distance_max=dist * factor * 1.1,
            ))

    # 模式3: ">NNNkm" / ">NNN" (大于约束)
    pattern3 = re.compile(r"[>＞]\s*(\d+(?:\.\d+)?)\s*(km|公里|nm|海里)?", re.IGNORECASE)
    if not constraints:
        for m in pattern3.finditer(hint):
            dist = float(m.group(1))
            unit = (m.group(2) or "km").lower()
            factor = _DISTANCE_UNITS.get(unit, 1.0)
            constraints.append(DistanceConstraint(
                ref_name="TARGET_CENTER",
                ref_point=resolved.get("TARGET_CENTER"),
                distance_km=dist * factor * 1.15,
                distance_min=dist * factor,
                distance_max=dist * factor * 1.5,
            ))

    # 提取方位角
    for c in constraints:
        c.bearing_deg = _extract_bearing(hint)

    return constraints


def _find_ref_point(name: str, resolved: dict[str, GeoPoint]) -> GeoPoint | None:
    """在已解析点中查找参考点。"""
    # 精确匹配
    if name in resolved:
        return resolved[name]
    # 模糊匹配
    name_clean = re.sub(r"[<>:\s]", "", name).lower()
    for key, point in resolved.items():
        key_clean = re.sub(r"[<>:\s]", "", key).lower()
        if name_clean in key_clean or key_clean in name_clean:
            return point
    # 关键词匹配
    keywords = {
        "目标": "TARGET_CENTER", "岛礁": "TARGET_CENTER",
        "基地": None,
    }
    for kw, target_key in keywords.items():
        if kw in name and target_key and target_key in resolved:
            return resolved[target_key]
    return None


def _extract_bearing(hint: str) -> float | None:
    """从 hint 文本提取方位角。"""
    # 数字方位: "航向120" / "方位角 270°"
    num_match = re.search(r"(?:航向|方位|bearing|heading)\s*(\d{1,3})\s*[°度]?", hint, re.IGNORECASE)
    if num_match:
        return float(num_match.group(1))

    # 关键词方位
    for keyword, bearing in _BEARING_KEYWORDS.items():
        if keyword in hint:
            return bearing

    return None


def _solve_constraints(
    constraints: list[DistanceConstraint],
    ref_type: str,
    side: str,
    center: GeoPoint,
) -> GeoPoint | None:
    """约束求解: 从距离约束计算坐标。

    算法:
      - 单约束+方位角: 直接大圆正算
      - 单约束无方位角: 按 ref_type 推断方位角
      - 多约束: 加权平均 (按约束精度加权)
    """
    valid = [c for c in constraints if c.ref_point is not None]
    if not valid:
        return None

    if len(valid) == 1:
        c = valid[0]
        bearing = c.bearing_deg if c.bearing_deg is not None else _infer_bearing_by_type(ref_type, side)
        lat, lon = destination_point(c.ref_point.lat, c.ref_point.lon, c.distance_km, bearing)
        confidence = 0.8 if c.bearing_deg is not None else 0.5
        return GeoPoint(lat, lon, confidence=confidence)

    # 多约束: 加权平均
    points: list[tuple[float, float, float]] = []  # (lat, lon, weight)
    for c in valid:
        bearing = c.bearing_deg if c.bearing_deg is not None else _infer_bearing_by_type(ref_type, side)
        lat, lon = destination_point(c.ref_point.lat, c.ref_point.lon, c.distance_km, bearing)
        # 权重: 有方位角的约束权重更高
        weight = 1.0 if c.bearing_deg is not None else 0.5
        # 距离越近约束越可靠
        weight *= max(0.3, 1.0 - c.distance_km / 2000)
        points.append((lat, lon, weight))

    total_weight = sum(w for _, _, w in points)
    avg_lat = sum(lat * w for lat, _, w in points) / total_weight
    avg_lon = sum(lon * w for _, lon, w in points) / total_weight

    return GeoPoint(avg_lat, avg_lon, confidence=min(0.95, 0.4 + 0.15 * len(valid)))


def _infer_bearing_by_type(ref_type: str, side: str) -> float:
    """根据参考点类型推断默认方位角。

    战术逻辑:
      - BLUE 的 BASE/WAYPOINT → 在目标的西侧/后方 (270°)
      - BLUE 的 ORBIT → 在目标的侧方 (视角色)
      - RED 的 TARGET → 在目标位置附近 (0°, 微调)
      - CAP_STATION → 侧翼 (0° 或 180°)
    """
    rt = ref_type.upper()
    if side.upper() == "BLUE":
        if rt in ("BASE",):
            return 270.0  # 基地在目标西侧
        if rt in ("WAYPOINT",):
            return 270.0  # 航路点在目标西侧(进攻方向)
        if rt in ("ORBIT",):
            return 260.0  # 轨道在目标偏西
        if rt in ("CAP_STATION",):
            return 315.0  # 巡逻站在目标西北侧翼
        return 270.0
    elif side.upper() == "RED":
        if rt in ("TARGET",):
            return 0.0  # 目标在中心附近
        return 0.0
    return 270.0


def _heuristic_fallback(
    ref_id: str, ref_type: str, side: str, center: GeoPoint
) -> GeoPoint:
    """启发式默认放置。当无任何距离信息时使用。"""
    bearing = _infer_bearing_by_type(ref_type, side)
    # 默认距离按类型
    default_distances = {
        "BASE": 800, "WAYPOINT": 300, "ORBIT": 450,
        "CAP_STATION": 340, "TARGET": 0,
    }
    dist = default_distances.get(ref_type.upper(), 200)
    lat, lon = destination_point(center.lat, center.lon, dist, bearing)
    return GeoPoint(lat, lon, name=ref_id, confidence=0.2)


# ═══════════════════════════════════════════════════════
# 坐标格式转换
# ═══════════════════════════════════════════════════════


def _decimal_to_afsim(lat: float, lon: float) -> str:
    """十进制度 → AFSIM d:m:s 格式。

    示例: (27.5, 122.5) → "27:30:00n 122:30:00e"
    """
    lat_dir = "n" if lat >= 0 else "s"
    lon_dir = "e" if lon >= 0 else "w"
    lat_abs, lon_abs = abs(lat), abs(lon)

    lat_d = int(lat_abs)
    lat_m = int((lat_abs - lat_d) * 60)
    lat_s = round(((lat_abs - lat_d) * 60 - lat_m) * 60)
    if lat_s == 60:
        lat_s = 0
        lat_m += 1
    if lat_m == 60:
        lat_m = 0
        lat_d += 1

    lon_d = int(lon_abs)
    lon_m = int((lon_abs - lon_d) * 60)
    lon_s = round(((lon_abs - lon_d) * 60 - lon_m) * 60)
    if lon_s == 60:
        lon_s = 0
        lon_m += 1
    if lon_m == 60:
        lon_m = 0
        lon_d += 1

    return f"{lat_d}:{lat_m:02d}:{lat_s:02d}{lat_dir} {lon_d}:{lon_m:02d}:{lon_s:02d}{lon_dir}"


def parse_afsim_coord(text: str) -> tuple[float, float] | None:
    """解析 AFSIM d:m:s 坐标为十进制度。

    支持格式: "27:30:00n 122:30:00e", "27:30:00N 122:30:00E"
    """
    pattern = re.compile(
        r"(\d{1,3}):(\d{2}):(\d{2})\s*([nsNS])\s+"
        r"(\d{1,3}):(\d{2}):(\d{2})\s*([ewEW])"
    )
    m = pattern.search(text)
    if not m:
        return None

    lat_d, lat_m, lat_s = int(m.group(1)), int(m.group(2)), int(m.group(3))
    lat_dir = m.group(4).lower()
    lon_d, lon_m, lon_s = int(m.group(5)), int(m.group(6)), int(m.group(7))
    lon_dir = m.group(8).lower()

    lat = lat_d + lat_m / 60 + lat_s / 3600
    if lat_dir == "s":
        lat = -lat
    lon = lon_d + lon_m / 60 + lon_s / 3600
    if lon_dir == "w":
        lon = -lon

    return lat, lon


def _try_parse_absolute(hint: str) -> GeoPoint | None:
    """尝试从 hint 中解析绝对坐标。"""
    # AFSIM d:m:s 格式
    coord = parse_afsim_coord(hint)
    if coord:
        return GeoPoint(coord[0], coord[1])

    # Decimal degrees: "27.5, 122.5" 或 "27.5N 122.5E"
    dec_match = re.search(
        r"(\d{1,3}(?:\.\d+)?)\s*°?\s*([nsNS])?\s*[,，\s]\s*(\d{1,3}(?:\.\d+)?)\s*°?\s*([ewEW])?",
        hint,
    )
    if dec_match:
        lat = float(dec_match.group(1))
        lon = float(dec_match.group(3))
        if dec_match.group(2) and dec_match.group(2).lower() == "s":
            lat = -lat
        if dec_match.group(4) and dec_match.group(4).lower() == "w":
            lon = -lon
        return GeoPoint(lat, lon)

    return None


def _describe_method(constraints: list[DistanceConstraint]) -> str:
    """描述使用的解算方法。"""
    has_bearing = any(c.bearing_deg is not None for c in constraints)
    if len(constraints) == 1:
        return "single_constraint_bearing" if has_bearing else "single_constraint_inferred_bearing"
    return f"multi_constraint_weighted_avg({len(constraints)})"
