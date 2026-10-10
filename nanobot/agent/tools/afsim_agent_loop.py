"""AFSIM Agent Loop — 错误依赖图驱动的多轮自修正代码生成引擎。

本模块实现 nanobot 的核心算法创新: 将代码生成从"一次性流水线"升级为
"自主决策循环"，类似于 Claude Code 的 Agent Loop 架构。

核心算法:
  1. ErrorDependencyGraph: 构建验证错误之间的因果关系图，
     用 PageRank 算法找到"修复杠杆最大的根因错误"。
  2. FixStrategyLibrary: 为每种错误类型匹配修复策略，
     每个策略是一个确定性的代码变换函数。
  3. AgentLoop: 自主循环 — 生成→验证→分析→修复→验证→...
     保证得分单调非递减，当达到目标分数或无改善时终止。

架构对比:
  传统流水线: parse → match → resolve → generate → validate (跑一遍结束)
  Agent Loop: generate → validate → ┌─ analyze_errors
                                     ├─ find_root_causes (PageRank)
                                     ├─ select_fixes (策略匹配)
                                     ├─ apply_fixes (代码变换)
                                     ├─ re_validate
                                     └─ check_convergence ──→ 回到 analyze
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from nanobot.agent.tools.base import Tool


# ═══════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════


@dataclass
class LoopSnapshot:
    """每轮循环的快照。"""
    round: int
    code: str
    score: int
    errors: int
    warnings: int
    actions_taken: list[str] = field(default_factory=list)
    root_causes_fixed: list[str] = field(default_factory=list)


@dataclass
class FixAction:
    """一个修复动作。"""
    rule_id: str
    action_type: str  # insert, delete, replace, rename
    target_line: int = 0
    old_text: str = ""
    new_text: str = ""
    description: str = ""


@dataclass
class LoopResult:
    """Agent Loop 最终结果。"""
    final_code: str
    final_score: int
    final_errors: int
    rounds: int
    converged: bool
    history: list[LoopSnapshot] = field(default_factory=list)
    total_fixes_applied: int = 0
    score_trajectory: list[int] = field(default_factory=list)


# ═══════════════════════════════════════════════════════
# 错误依赖图
# ═══════════════════════════════════════════════════════


# 错误因果关系规则: (原因 rule_id, 结果 rule_id) → 依赖描述
# 含义: 如果存在 rule_id_A 的错误，它会导致 rule_id_B 的错误
_CAUSAL_RULES: dict[tuple[str, str], str] = {
    # 块未闭合 → 内部所有引用检查失败
    ("SYN_002", "SEM_002"): "unclosed_block_masks_weapon_refs",
    ("SYN_002", "CON_001"): "unclosed_block_masks_platform_refs",
    ("SYN_002", "CON_002"): "unclosed_block_causes_parse_errors",
    # weapon_type 未定义 → platform 内 weapon 引用报错
    ("SEM_002", "CON_001"): "missing_definition_causes_ref_error",
    # 坐标格式错误 → position 语句解析失败
    ("SYN_004", "SYN_002"): "bad_coord_breaks_block_structure",
    # 基类型拼写错误 → 块结构混乱
    ("SYN_003", "SYN_002"): "bad_basetype_breaks_block",
    # side 值错误 → 一致性检查连锁失败
    ("SEM_001", "CON_001"): "bad_side_causes_consistency_fail",
}

# 同一行的错误倾向于有因果关系
_SAME_LINE_CAUSALITY = True


class ErrorDependencyGraph:
    """错误依赖图: 分析验证错误之间的因果关系。

    算法:
      1. 将每个 issue 作为图节点
      2. 根据 _CAUSAL_RULES 建立有向边 (cause → effect)
      3. 根据行号邻近性补充隐含依赖
      4. 用 PageRank 计算每个节点的"修复杠杆"
      5. 返回 top-K 根因节点 (修复后能连锁消除最多下游错误)
    """

    def __init__(self):
        self.nodes: list[dict] = []  # issue dicts
        self.edges: list[tuple[int, int, float]] = []  # (from, to, weight)
        self._adj: dict[int, list[tuple[int, float]]] = {}

    def build(self, issues: list[dict]) -> None:
        """从验证问题列表构建依赖图。"""
        self.nodes = issues
        self.edges = []
        self._adj = {}

        n = len(issues)
        for i in range(n):
            self._adj[i] = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                weight = self._compute_edge_weight(issues[i], issues[j])
                if weight > 0:
                    self.edges.append((i, j, weight))
                    self._adj[i].append((j, weight))

    def _compute_edge_weight(self, cause: dict, effect: dict) -> float:
        """计算两个 issue 之间的因果边权重。"""
        weight = 0.0

        # 规则匹配
        cause_rule = cause.get("rule", "")
        effect_rule = effect.get("rule", "")
        key = (cause_rule, effect_rule)
        if key in _CAUSAL_RULES:
            weight += 0.6

        # 同行/邻近行 → 强因果关系
        if _SAME_LINE_CAUSALITY:
            line_diff = abs(cause.get("line", 0) - effect.get("line", 0))
            if line_diff == 0 and cause.get("line", 0) > 0:
                weight += 0.3
            elif line_diff <= 3:
                weight += 0.15

        # error → warning 的因果 (修 error 可能消除 warning)
        if cause.get("severity") == "error" and effect.get("severity") == "warning":
            weight += 0.1

        # 语法错误 → 语义错误的因果 (语法修复后语义问题可能消失)
        if cause.get("level") == "syntax" and effect.get("level") in ("semantic", "consistency"):
            weight += 0.2

        return min(weight, 1.0)

    def pagerank(self, damping: float = 0.85, iterations: int = 50) -> list[float]:
        """PageRank: 计算每个错误节点的"修复杠杆"。

        高 PageRank = 修复此节点能影响最多下游节点 = 根因。

        与标准 PageRank 的区别:
          - 边方向: cause → effect (修 cause 能消除 effect)
          - 目标: 找到"影响最大化"的节点，即入度+出度综合最高的
          - 我们用反向 PageRank: effect 指向 cause (谁的"被依赖"最多)
        """
        n = len(self.nodes)
        if n == 0:
            return []
        if n == 1:
            return [1.0]

        # 构建反向邻接表 (effect → cause)
        # 这样 PageRank 高的节点 = 被最多 effect 依赖的 cause = 根因
        reverse_adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n)}
        for i in range(n):
            for j, w in self._adj.get(i, []):
                reverse_adj[j].append((i, w))

        # 初始化
        scores = [1.0 / n] * n

        for _ in range(iterations):
            new_scores = [0.0] * n
            for i in range(n):
                rank = (1 - damping) / n
                for src, weight in reverse_adj.get(i, []):
                    # src 指向 i 的权重
                    out_weight_sum = sum(w for _, w in reverse_adj.get(src, []))
                    if out_weight_sum > 0:
                        rank += damping * scores[src] * (weight / out_weight_sum)
                    else:
                        rank += damping * scores[src] / n
                new_scores[i] = rank

            # 归一化
            total = sum(new_scores)
            if total > 0:
                new_scores = [s / total for s in new_scores]
            scores = new_scores

        return scores

    def find_root_causes(self, top_k: int = 5) -> list[tuple[int, float]]:
        """找到 top-K 根因错误。

        返回: [(node_index, pagerank_score), ...]
        筛选: 只返回 severity=error 的节点 (修 warning 不如修 error 划算)
        """
        scores = self.pagerank()
        if not scores:
            return []

        # 按分数排序，只保留 error 级别
        indexed = [(i, s) for i, s in enumerate(scores)
                   if self.nodes[i].get("severity") == "error"]
        indexed.sort(key=lambda x: x[1], reverse=True)

        return indexed[:top_k]

    def get_downstream(self, node_idx: int) -> list[int]:
        """获取某节点的所有下游节点（修复此节点后可能消失的错误）。"""
        visited: set[int] = set()
        queue = [node_idx]
        while queue:
            current = queue.pop(0)
            for neighbor, _ in self._adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return list(visited)


# ═══════════════════════════════════════════════════════
# 修复策略库
# ═══════════════════════════════════════════════════════


class FixStrategyLibrary:
    """修复策略库: 为每种错误类型匹配修复动作。

    每个策略是一个代码变换函数:
      输入: AFSIM 代码 (list[str]) + issue dict
      输出: (修改后的代码, FixAction 描述)
    """

    def __init__(self):
        self._strategies: dict[str, Callable] = {
            "SYN_001": self._fix_unmatched_end,
            "SYN_002": self._fix_unclosed_block,
            "SYN_003": self._fix_bad_basetype,
            "SYN_004": self._fix_bad_coord,
            "SYN_007": self._fix_bad_unit,
            "SYN_008": self._fix_bad_time_unit,
            "SEM_001": self._fix_bad_side,
            "SEM_002": self._fix_missing_weapon_def,
            "CON_001": self._fix_missing_platform_type,
            "CON_002": self._fix_duplicate_name,
            "TAC_001": self._fix_bad_speed,
            "TAC_003": self._fix_bad_altitude,
        }

    def select(self, issue: dict) -> Callable | None:
        """根据 rule_id 选择修复策略。"""
        rule_id = issue.get("rule", "")
        return self._strategies.get(rule_id)

    def apply_fix(self, code: str, issue: dict) -> tuple[str, FixAction | None]:
        """应用修复策略。"""
        strategy = self.select(issue)
        if strategy is None:
            return code, None
        return strategy(code, issue)

    # ── 语法修复 ──

    def _fix_unclosed_block(self, code: str, issue: dict) -> tuple[str, FixAction | None]:
        """SYN_002: 块未闭合 → 在合适位置插入 end_* 行。"""
        lines = code.split("\n")
        line_num = issue.get("line", 0)
        if line_num < 1 or line_num > len(lines):
            return code, None

        # 从 issue 描述中提取需要闭合的块类型
        desc = issue.get("description", "")
        block_match = re.search(r"Unclosed '(\w+)'", desc)
        if not block_match:
            return code, None

        block_kw = block_match.group(1)
        end_kw = f"end_{block_kw}"

        # 找到下一个同级块或文件末尾，在其前面插入 end_*
        insert_pos = line_num  # 0-based → 在该行之后
        for i in range(line_num, len(lines)):
            stripped = lines[i].strip()
            # 遇到下一个同级块开始 或 更高层级块 → 在此之前插入
            if stripped.startswith(block_kw + " ") or stripped.startswith("platform_type ") or stripped.startswith("platform "):
                if i > line_num:
                    insert_pos = i
                    break
        else:
            insert_pos = len(lines)

        lines.insert(insert_pos, end_kw)
        new_code = "\n".join(lines)
        return new_code, FixAction(
            rule_id="SYN_002", action_type="insert",
            target_line=insert_pos + 1, new_text=end_kw,
            description=f"Inserted '{end_kw}' to close unclosed block",
        )

    def _fix_unmatched_end(self, code: str, issue: dict) -> tuple[str, FixAction | None]:
        """SYN_001: 多余的 end_* → 删除该行。"""
        lines = code.split("\n")
        line_num = issue.get("line", 0)
        if line_num < 1 or line_num > len(lines):
            return code, None

        removed = lines.pop(line_num - 1)
        new_code = "\n".join(lines)
        return new_code, FixAction(
            rule_id="SYN_001", action_type="delete",
            target_line=line_num, old_text=removed,
            description=f"Removed unmatched '{removed.strip()}'",
        )

    def _fix_bad_basetype(self, code: str, issue: dict) -> tuple[str, FixAction | None]:
        """SYN_003: 基类型拼写错误 → 用建议替换。"""
        suggestion = issue.get("suggestion", "")
        desc = issue.get("description", "")
        # 提取错误词和建议词
        bad_match = re.search(r"Unknown base type '(\w+)'", desc)
        suggest_match = re.search(r"Did you mean one of: (\w+)", suggestion)

        if not bad_match or not suggest_match:
            return code, None

        bad_word = bad_match.group(1)
        good_word = suggest_match.group(1)

        lines = code.split("\n")
        line_num = issue.get("line", 0)
        if 1 <= line_num <= len(lines):
            lines[line_num - 1] = lines[line_num - 1].replace(bad_word, good_word)
            new_code = "\n".join(lines)
            return new_code, FixAction(
                rule_id="SYN_003", action_type="replace",
                target_line=line_num, old_text=bad_word, new_text=good_word,
                description=f"Replaced '{bad_word}' with '{good_word}'",
            )
        return code, None

    def _fix_bad_coord(self, code: str, issue: dict) -> tuple[str, FixAction | None]:
        """SYN_004: 坐标格式错误 → 尝试修复为 d:m:s 格式。"""
        lines = code.split("\n")
        line_num = issue.get("line", 0)
        if line_num < 1 or line_num > len(lines):
            return code, None

        line = lines[line_num - 1]
        # 尝试提取十进制度并转换
        dec_match = re.search(r"(-?\d+\.?\d*)\s*[,\s]\s*(-?\d+\.?\d*)", line)
        if dec_match:
            lat = float(dec_match.group(1))
            lon = float(dec_match.group(2))
            dms = _decimal_to_dms(lat, lon)
            new_line = line.replace(dec_match.group(0), dms)
            lines[line_num - 1] = new_line
            new_code = "\n".join(lines)
            return new_code, FixAction(
                rule_id="SYN_004", action_type="replace",
                target_line=line_num, old_text=dec_match.group(0), new_text=dms,
                description="Converted decimal degrees to d:m:s format",
            )
        return code, None

    def _fix_bad_unit(self, code: str, issue: dict) -> tuple[str, FixAction | None]:
        """SYN_007/008: 单位错误 → 替换为合法单位。"""
        lines = code.split("\n")
        line_num = issue.get("line", 0)
        if line_num < 1 or line_num > len(lines):
            return code, None

        line = lines[line_num - 1]
        # 修复高度单位
        if "altitude" in line:
            line = re.sub(r"(\d+)\s+(?!ft\s+msl|m\s+agl)(\w+)", r"\1 ft msl", line)
        # 修复时间单位
        if "end_time" in line:
            line = re.sub(r"(\d+)\s+(?!sec|min|hr)(\w+)", r"\1 sec", line)

        if line != lines[line_num - 1]:
            lines[line_num - 1] = line
            new_code = "\n".join(lines)
            return new_code, FixAction(
                rule_id=issue.get("rule", "SYN_007"), action_type="replace",
                target_line=line_num, new_text=line.strip(),
                description="Fixed unit to valid AFSIM format",
            )
        return code, None

    def _fix_bad_time_unit(self, code: str, issue: dict) -> tuple[str, FixAction | None]:
        """SYN_008: 时间单位错误。"""
        return self._fix_bad_unit(code, issue)

    def _fix_inline_comment(self, code: str, issue: dict) -> tuple[str, FixAction | None]:
        """SYN_009: 行尾注释 → 移除 # 后内容。"""
        lines = code.split("\n")
        line_num = issue.get("line", 0)
        if line_num < 1 or line_num > len(lines):
            return code, None

        line = lines[line_num - 1]
        if "#" in line and not line.strip().startswith("#"):
            # 移除行尾注释
            idx = line.index("#")
            lines[line_num - 1] = line[:idx].rstrip()
            new_code = "\n".join(lines)
            return new_code, FixAction(
                rule_id="SYN_009", action_type="replace",
                target_line=line_num,
                description="Removed inline comment (AFSIM does not support)",
            )
        return code, None

    # ── 语义修复 ──

    def _fix_bad_side(self, code: str, issue: dict) -> tuple[str, FixAction | None]:
        """SEM_001: side 值错误 → 替换为 red/blue。"""
        lines = code.split("\n")
        line_num = issue.get("line", 0)
        if line_num < 1 or line_num > len(lines):
            return code, None

        line = lines[line_num - 1]
        desc = issue.get("description", "")
        bad_match = re.search(r"Invalid side value '(\w+)'", desc)
        if bad_match:
            bad_val = bad_match.group(1)
            # 推断正确值: 如果包含 "r" 用 red, 否则 blue
            good_val = "red" if "r" in bad_val.lower() else "blue"
            lines[line_num - 1] = line.replace(bad_val, good_val)
            new_code = "\n".join(lines)
            return new_code, FixAction(
                rule_id="SEM_001", action_type="replace",
                target_line=line_num, old_text=bad_val, new_text=good_val,
                description=f"Fixed side value: {bad_val} → {good_val}",
            )
        return code, None

    def _fix_missing_weapon_def(self, code: str, issue: dict) -> tuple[str, FixAction | None]:
        """SEM_002: weapon 类型未定义 → 插入 platform_type + weapon 声明。"""
        desc = issue.get("description", "")
        name_match = re.search(r"Weapon type '(\w+)' referenced", desc)
        if not name_match:
            name_match = re.search(r"Weapon '(\w+)'", desc)
        if not name_match:
            return code, None

        w_name = name_match.group(1)
        w_upper = w_name.upper()

        # 检查是否已有 weapon 声明
        if re.search(rf"weapon\s+{re.escape(w_upper)}\s+WSF_EXPLICIT_WEAPON", code):
            return code, None

        # 插入 platform_type + weapon 声明
        weapon_template = f"""platform_type {w_upper} WSF_PLATFORM
   category missile
   icon aim120
   mover WSF_GUIDED_MOVER
      total_mass 350.0 lbm
      fuel_mass 157.0 lbm
      specific_impulse 222.0 sec
      thrust 5800.0 lbf
      thrust_duration 6.0 sec
   end_mover
   sensor seeker WSF_GEOMETRIC_SENSOR
      off
      maximum_range 25 km
      frame_time 0.1 sec
   end_sensor
   processor track_mgr WSF_TRACK_PROCESSOR
      master_track_processor
   end_processor
end_platform_type

weapon_effects {w_upper}_EFFECT WSF_GRADUATED_LETHALITY
   radius_and_pk 20.0 m 0.5
end_weapon_effects

weapon {w_upper} WSF_EXPLICIT_WEAPON
   launched_platform_type {w_upper}
   weapon_effects {w_upper}_EFFECT
   firing_delay 0.5 sec
   firing_interval 1 sec
   salvo_interval 5.0 sec
end_weapon
"""

        lines = code.split("\n")
        # 在第一个 platform_type (非武器的) 之前插入
        insert_pos = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("platform_type ") and "missile" not in lines[i:i+5]:
                insert_pos = i
                break

        if insert_pos == 0:
            insert_pos = len(lines)

        lines.insert(insert_pos, weapon_template.rstrip())
        lines.insert(insert_pos + 1, "")
        new_code = "\n".join(lines)

        return new_code, FixAction(
            rule_id="SEM_002", action_type="insert",
            target_line=insert_pos + 1, new_text=f"platform_type {w_upper} + weapon {w_upper}",
            description=f"Added missing weapon definition for '{w_name}'",
        )

    # ── 一致性修复 ──

    def _fix_missing_platform_type(self, code: str, issue: dict) -> tuple[str, FixAction | None]:
        """CON_001: platform_type 未定义 → 插入最小定义。"""
        desc = issue.get("description", "")
        # 提取 type 名
        type_match = re.search(r"references undefined type '(\w+)'", desc)
        if not type_match:
            return code, None

        type_name = type_match.group(1)

        # 检查是否已有定义
        if re.search(rf"platform_type\s+{re.escape(type_name)}\s", code):
            return code, None

        # 推断 side: 从上下文查找
        side = "blue"
        lines = code.split("\n")
        for line in lines:
            if type_name in line and "side" in line:
                side_match = re.match(r"\s*side\s+(\w+)", line.strip())
                if side_match:
                    side = side_match.group(1)
                    break

        pt_template = f"""platform_type {type_name} WSF_PLATFORM
   side {side}
   mover WSF_AIR_MOVER
      maximum_radial_acceleration 5 g
      maximum_linear_acceleration 3 g
   end_mover
end_platform_type
"""

        # 在第一个 platform 实例之前插入
        insert_pos = 0
        for i, line in enumerate(lines):
            if re.match(r"platform\s+\w+", line.strip()):
                insert_pos = i
                break

        if insert_pos == 0:
            # 在 Global Config 之前
            for i, line in enumerate(lines):
                if "random_seed" in line or "end_time" in line:
                    insert_pos = i
                    break

        if insert_pos == 0:
            insert_pos = len(lines)

        lines.insert(insert_pos, pt_template.rstrip())
        lines.insert(insert_pos + 1, "")
        new_code = "\n".join(lines)

        return new_code, FixAction(
            rule_id="CON_001", action_type="insert",
            target_line=insert_pos + 1, new_text=f"platform_type {type_name} ...",
            description=f"Added missing platform_type definition for '{type_name}'",
        )

    def _fix_duplicate_name(self, code: str, issue: dict) -> tuple[str, FixAction | None]:
        """CON_002: 平台名称重复 → 追加 _2, _3 后缀。"""
        desc = issue.get("description", "")
        name_match = re.search(r"Duplicate platform name '(\w+)'", desc)
        if not name_match:
            return code, None

        name = name_match.group(1)
        line_num = issue.get("line", 0)
        lines = code.split("\n")
        if line_num < 1 or line_num > len(lines):
            return code, None

        new_name = f"{name}_2"
        # 检查 _2 是否也重复
        suffix = 2
        while new_name in code:
            suffix += 1
            new_name = f"{name}_{suffix}"

        lines[line_num - 1] = lines[line_num - 1].replace(name, new_name, 1)
        new_code = "\n".join(lines)

        return new_code, FixAction(
            rule_id="CON_002", action_type="rename",
            target_line=line_num, old_text=name, new_text=new_name,
            description=f"Renamed duplicate platform: {name} → {new_name}",
        )

    # ── 战术修复 ──

    def _fix_bad_speed(self, code: str, issue: dict) -> tuple[str, FixAction | None]:
        """TAC_001: 速度过大 → 限制到 1400 kts。"""
        lines = code.split("\n")
        line_num = issue.get("line", 0)
        if line_num < 1 or line_num > len(lines):
            return code, None

        line = lines[line_num - 1]
        # 限制速度
        m = re.match(r"(\s*speed\s+)(\d+)(\s+kts)", line)
        if m:
            speed = int(m.group(2))
            if speed > 1400:
                lines[line_num - 1] = f"{m.group(1)}1400{m.group(3)}"
                new_code = "\n".join(lines)
                return new_code, FixAction(
                    rule_id="TAC_001", action_type="replace",
                    target_line=line_num,
                    description=f"Capped speed {speed} → 1400 kts",
                )
        return code, None

    def _fix_bad_altitude(self, code: str, issue: dict) -> tuple[str, FixAction | None]:
        """TAC_003: 高度过高 → 限制到 60000 ft。"""
        lines = code.split("\n")
        line_num = issue.get("line", 0)
        if line_num < 1 or line_num > len(lines):
            return code, None

        line = lines[line_num - 1]
        m = re.match(r"(\s*altitude\s+)(\d+)(\s+ft\s+msl)", line)
        if m:
            alt = int(m.group(2))
            if alt > 60000:
                lines[line_num - 1] = f"{m.group(1)}60000{m.group(3)}"
                new_code = "\n".join(lines)
                return new_code, FixAction(
                    rule_id="TAC_003", action_type="replace",
                    target_line=line_num,
                    description=f"Capped altitude {alt} → 60000 ft",
                )
        return code, None


# ═══════════════════════════════════════════════════════
# Agent Loop 主引擎
# ═══════════════════════════════════════════════════════


class AgentLoop:
    """Agent Loop: 多轮自修正代码生成引擎。

    核心循环:
      1. validate(code) → issues[]
      2. build ErrorDependencyGraph(issues)
      3. find_root_causes (PageRank top-K)
      4. for each root cause: select_fix → apply_fix
      5. re_validate(new_code)
      6. if score improved: accept; else: revert
      7. if score >= target or no improvement: stop

    收敛保证:
      - 每轮只接受 score 不降低的修复 (单调非递减)
      - 连续 2 轮无改善则终止
      - 最大轮次限制
    """

    def __init__(
        self,
        generate_fn: Callable,
        validate_fn: Callable,
        max_rounds: int = 5,
        target_score: int = 85,
        top_k_fixes: int = 5,
    ):
        self.generate_fn = generate_fn
        self.validate_fn = validate_fn
        self.max_rounds = max_rounds
        self.target_score = target_score
        self.top_k_fixes = top_k_fixes
        self.fix_library = FixStrategyLibrary()
        self.history: list[LoopSnapshot] = []

    def run(
        self,
        coa_data: dict,
        coords: dict,
        matches: dict,
        scenario_name: str = "generated",
        end_time: int = 21600,
    ) -> LoopResult:
        """执行完整的 Agent Loop。"""

        # Round 0: 初始生成
        gen_output = self.generate_fn(coa_data, coords, matches, scenario_name, end_time)
        if isinstance(gen_output, dict):
            code = gen_output.get("scenario_code", "")
        else:
            code = str(gen_output)

        # 初始验证
        val_output = self.validate_fn(code)
        if isinstance(val_output, dict):
            current_score = val_output.get("score", 0)
            current_issues = val_output.get("issues", [])
            current_errors = val_output.get("summary", {}).get("errors", 0)
            current_warnings = val_output.get("summary", {}).get("warnings", 0)
        else:
            val_data = json.loads(val_output)
            current_score = val_data.get("score", 0)
            current_issues = val_data.get("issues", [])
            current_errors = val_data.get("summary", {}).get("errors", 0)
            current_warnings = val_data.get("summary", {}).get("warnings", 0)

        self.history.append(LoopSnapshot(
            round=0, code=code, score=current_score,
            errors=current_errors, warnings=current_warnings,
            actions_taken=["initial_generation"],
        ))

        score_trajectory = [current_score]
        total_fixes = 0
        no_improvement_count = 0

        # Agent Loop
        for round_num in range(1, self.max_rounds + 1):
            # 终止条件 1: 达到目标分数
            if current_score >= self.target_score:
                break

            # 终止条件 2: 没有错误了
            if current_errors == 0 and current_warnings <= 2:
                break

            # 终止条件 3: 连续无改善
            if no_improvement_count >= 2:
                break

            # Step 1: 构建错误依赖图
            graph = ErrorDependencyGraph()
            graph.build(current_issues)

            # Step 2: 找根因
            root_causes = graph.find_root_causes(top_k=self.top_k_fixes)

            if not root_causes:
                break  # 没有 error 可修

            # Step 3: 逐个尝试修复
            new_code = code
            actions_this_round: list[str] = []
            fixes_this_round = 0
            root_causes_fixed: list[str] = []

            for node_idx, pr_score in root_causes:
                issue = current_issues[node_idx]
                fixed_code, action = self.fix_library.apply_fix(new_code, issue)

                if action is not None and fixed_code != new_code:
                    new_code = fixed_code
                    actions_this_round.append(action.description)
                    fixes_this_round += 1
                    root_causes_fixed.append(issue.get("rule", "unknown"))
                    # 记录下游影响
                    downstream = graph.get_downstream(node_idx)
                    if downstream:
                        actions_this_round.append(
                            f"  └─ cascading fix: may resolve {len(downstream)} downstream issues"
                        )

            if fixes_this_round == 0:
                no_improvement_count += 1
                continue

            # Step 4: 重新验证
            new_val_output = self.validate_fn(new_code)
            if isinstance(new_val_output, dict):
                new_val_data = new_val_output
            else:
                new_val_data = json.loads(new_val_output)

            new_score = new_val_data.get("score", 0)
            new_errors = new_val_data.get("summary", {}).get("errors", 0)
            new_warnings = new_val_data.get("summary", {}).get("warnings", 0)
            new_issues = new_val_data.get("issues", [])

            # Step 5: 单调收敛检查
            if new_score >= current_score:
                # 接受修复
                code = new_code
                current_score = new_score
                current_errors = new_errors
                current_warnings = new_warnings
                current_issues = new_issues
                total_fixes += fixes_this_round
                no_improvement_count = 0
            else:
                # 回退: 修复导致分数下降
                no_improvement_count += 1

            # 记录快照
            self.history.append(LoopSnapshot(
                round=round_num, code=code, score=current_score,
                errors=current_errors, warnings=current_warnings,
                actions_taken=actions_this_round,
                root_causes_fixed=root_causes_fixed,
            ))
            score_trajectory.append(current_score)

        converged = current_score >= self.target_score or current_errors == 0

        return LoopResult(
            final_code=code,
            final_score=current_score,
            final_errors=current_errors,
            rounds=len(self.history) - 1,
            converged=converged,
            history=self.history,
            total_fixes_applied=total_fixes,
            score_trajectory=score_trajectory,
        )


# ═══════════════════════════════════════════════════════
# Nanobot Tool 接口
# ═══════════════════════════════════════════════════════


class AFSIMAgentLoopTool(Tool):
    """Agent Loop: COA→AFSIM 全自动多轮自修正代码生成。

    这是一个 meta-tool，内部编排 coa_parse → template_match → coord_resolve
    → afsim_generate → scenario_validate 的完整循环，并加入错误依赖图
    驱动的多轮自修正。

    支持两种模式:
      1. classic  (默认): 固定流水线 Agent Loop (ErrorDependencyGraph + PageRank)
      2. react:           ReAct 引擎 (LLM 自主决策 + Reflexion 记忆)
    """

    _scopes = {"core", "subagent"}

    @property
    def name(self) -> str:
        return "afsim_agent_loop"

    @property
    def description(self) -> str:
        return (
            "Full COA→AFSIM pipeline with multi-round self-correction. "
            "Parses COA, matches templates, resolves coordinates, generates "
            "AFSIM code, then iteratively fixes errors using an error "
            "dependency graph (PageRank root cause analysis) until convergence. "
            "Returns final code, score trajectory, and fix history."
        )

    @property
    def read_only(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "coa_file_path": {
                    "type": "string",
                    "description": "Path to the COA Markdown file.",
                },
                "center_lat": {
                    "type": "number",
                    "description": "Target area center latitude. Default: 27.5",
                },
                "center_lon": {
                    "type": "number",
                    "description": "Target area center longitude. Default: 122.5",
                },
                "scenario_name": {
                    "type": "string",
                    "description": "Scenario name. Default: from COA.",
                },
                "max_rounds": {
                    "type": "integer",
                    "description": "Max correction rounds. Default: 5.",
                },
                "target_score": {
                    "type": "integer",
                    "description": "Target validation score (0-100). Default: 85.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["classic", "react"],
                    "description": "Agent mode: 'classic' (fixed pipeline) or 'react' (LLM-driven ReAct + Reflexion memory). Default: 'classic'.",
                },
                "api_key": {
                    "type": "string",
                    "description": "DeepSeek API key (required for react mode).",
                },
            },
            "required": ["coa_file_path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        try:
            coa_path = kwargs["coa_file_path"]
            center_lat = float(kwargs.get("center_lat", 27.5))
            center_lon = float(kwargs.get("center_lon", 122.5))
            scenario_name = kwargs.get("scenario_name", "generated_scenario")
            max_rounds = int(kwargs.get("max_rounds", 5))
            target_score = int(kwargs.get("target_score", 85))
            mode = kwargs.get("mode", "classic")

            # 延迟导入工具 (避免循环依赖)
            from nanobot.agent.tools.coa_parser_tool import COAParseTool
            from nanobot.agent.tools.coord_resolver_tool import CoordResolveTool
            from nanobot.agent.tools.template_matcher_tool import TemplateMatchTool

            parse_tool = COAParseTool()
            match_tool = TemplateMatchTool()
            coord_tool = CoordResolveTool()

            # Step 1: 解析 COA
            parse_result = await parse_tool.execute(file_path=coa_path)
            if parse_result.startswith("Error"):
                return f"COA parse failed: {parse_result}"
            coa_data = json.loads(parse_result)

            if not scenario_name or scenario_name == "generated_scenario":
                scenario_name = coa_data.get("name", "generated_scenario")

            # Step 2: 模板匹配
            match_result = await match_tool.execute(
                entities_json=json.dumps(coa_data.get("entities", []))
            )
            matches = json.loads(match_result)

            # Step 3: 坐标解算
            coord_result = await coord_tool.execute(
                geo_refs_json=json.dumps(coa_data.get("geo_refs", [])),
                center_lat=center_lat, center_lon=center_lon,
            )
            coords = json.loads(coord_result)

            # 定义 generate 和 validate 函数 (供 AgentLoop / ReActEngine 调用)
            # 使用同步类避免 async event loop 冲突
            from nanobot.agent.tools.afsim_generator_tool import ScenarioGenerator
            from nanobot.agent.tools.scenario_validator_tool import ScenarioValidator

            def generate_fn(coa, crds, mtchs, name, end_time):
                try:
                    generator = ScenarioGenerator(coa, crds, mtchs)
                    code = generator.generate(name, end_time)
                    return {
                        "scenario_code": code,
                        "stats": {
                            "platform_types": generator.stats.platform_types,
                            "platform_instances": generator.stats.platform_instances,
                            "lines_of_code": generator.stats.lines_of_code,
                        },
                    }
                except Exception as e:
                    return {"scenario_code": "", "error": str(e)}

            def validate_fn(code):
                try:
                    validator = ScenarioValidator(code)
                    result = validator.validate_all()
                    return {
                        "score": result.score,
                        "summary": {
                            "errors": result.errors,
                            "warnings": result.warnings,
                            "infos": result.infos,
                            "total_issues": result.total_issues,
                        },
                        "issues": [
                            {
                                "level": i.level,
                                "severity": i.severity,
                                "line": i.line_number,
                                "rule": i.rule_id,
                                "description": i.description,
                                "suggestion": i.suggestion,
                            }
                            for i in result.issues
                        ],
                    }
                except Exception:
                    return {"score": 0, "issues": [], "summary": {"errors": 1, "warnings": 0}}

            # ── 模式选择 ──
            if mode == "react":
                return await self._run_react(
                    kwargs, coa_data, coords, matches, scenario_name,
                    target_score, generate_fn, validate_fn,
                )

            # Step 4: 运行经典 Agent Loop
            agent_loop = AgentLoop(
                generate_fn=generate_fn,
                validate_fn=validate_fn,
                max_rounds=max_rounds,
                target_score=target_score,
            )

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: agent_loop.run(coa_data, coords, matches, scenario_name)
            )

            # 构造输出
            output = {
                "mode": "classic",
                "scenario_code": result.final_code,
                "scenario_name": scenario_name,
                "final_score": result.final_score,
                "final_errors": result.final_errors,
                "converged": result.converged,
                "rounds": result.rounds,
                "total_fixes": result.total_fixes_applied,
                "score_trajectory": result.score_trajectory,
                "history": [
                    {
                        "round": s.round,
                        "score": s.score,
                        "errors": s.errors,
                        "warnings": s.warnings,
                        "actions": s.actions_taken,
                        "root_causes_fixed": s.root_causes_fixed,
                    }
                    for s in result.history
                ],
            }

            return json.dumps(output, ensure_ascii=False, indent=2)

        except Exception as exc:
            return f"Error in agent loop: {exc}"

    # ── ReAct 模式 ─────────────────────────────────────────

    async def _run_react(
        self,
        kwargs: dict,
        coa_data: dict,
        coords: dict,
        matches: dict,
        scenario_name: str,
        target_score: int,
        generate_fn: Callable,
        validate_fn: Callable,
    ) -> str:
        """ReAct 模式: LLM 驱动的自主决策 + Reflexion 记忆。"""
        from pathlib import Path as _Path
        from nanobot.agent.tools.memory_store import MemoryStore
        from nanobot.agent.tools.react_engine import DeepSeekClient, ReActEngine

        api_key = kwargs.get("api_key", "")

        # 初始化记忆存储
        memory_path = _Path.home() / ".nanobot" / "afsim_memory.json"
        memory = MemoryStore(persist_path=memory_path)

        # 初始化 LLM 客户端 (未显式传 api_key 时回落到 DEEPSEEK_API_KEY)
        try:
            llm = DeepSeekClient(api_key=api_key)
        except ValueError as exc:
            return f"Error: {exc}"

        # 修复策略库 (供 fix 工具使用)
        fix_library = FixStrategyLibrary()

        # 定义 analyze 函数 (错误依赖图 + PageRank)
        def analyze_fn(issues: list[dict]) -> dict:
            graph = ErrorDependencyGraph()
            graph.build(issues)
            root_causes = graph.find_root_causes(top_k=5)
            return {"root_causes": root_causes}

        # 定义 fix 函数
        def fix_fn(code: str, issue: dict) -> tuple[str, Any]:
            return fix_library.apply_fix(code, issue)

        # 构建工具注册表
        tools = {
            "generate": generate_fn,
            "validate": validate_fn,
            "analyze": analyze_fn,
            "fix": fix_fn,
        }

        # 初始化 ReAct 引擎
        engine = ReActEngine(
            llm_client=llm,
            tools=tools,
            memory=memory,
            max_steps=kwargs.get("max_rounds", 15),
            target_score=target_score,
        )

        # 运行 ReAct 循环
        task = f"Generate AFSIM scenario code from COA: {scenario_name}"
        context = {
            "coa_data": coa_data,
            "coords": coords,
            "matches": matches,
            "scenario_name": scenario_name,
            "end_time": kwargs.get("end_time", 21600),
        }

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: engine.run(task, context)
        )

        # 构造输出
        output = {
            "mode": "react",
            "scenario_code": result.final_code,
            "scenario_name": scenario_name,
            "final_score": result.final_score,
            "final_errors": result.final_errors,
            "success": result.success,
            "total_steps": result.total_steps,
            "trajectory": [
                {
                    "step": s.step,
                    "thought": s.thought,
                    "action": s.action,
                    "observation": s.observation[:200],
                }
                for s in result.trajectory
            ],
            "reflections": result.reflections,
            "memory_stats": memory.get_stats(),
        }

        return json.dumps(output, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════


def _decimal_to_dms(lat: float, lon: float) -> str:
    """十进制度 → AFSIM d:m:s 格式。"""
    lat_dir = "n" if lat >= 0 else "s"
    lon_dir = "e" if lon >= 0 else "w"
    lat_abs, lon_abs = abs(lat), abs(lon)

    lat_d = int(lat_abs)
    lat_m = int((lat_abs - lat_d) * 60)
    lat_s = round(((lat_abs - lat_d) * 60 - lat_m) * 60)
    if lat_s == 60:
        lat_s = 0; lat_m += 1

    lon_d = int(lon_abs)
    lon_m = int((lon_abs - lon_d) * 60)
    lon_s = round(((lon_abs - lon_d) * 60 - lon_m) * 60)
    if lon_s == 60:
        lon_s = 0; lon_m += 1

    return f"{lat_d}:{lat_m:02d}:{lat_s:02d}{lat_dir} {lon_d}:{lon_m:02d}:{lon_s:02d}{lon_dir}"
