"""Reflexion-style 记忆存储 — 结构化长短期记忆 + 按需检索。

设计原则 (回应"直接注入太激进"的顾虑):
  1. 按需检索: 只在 agent 主动调用 recall 时返回经验，不主动注入 prompt
  2. 结构化模式: 存储 (error_pattern → fix → outcome) 三元组，不是原始反思文本
  3. 蒸馏压缩: 多次相似经验自动合并为一条规则，避免冗余
  4. 置信度门控: 只返回 success_rate >= 阈值的规则，低置信度的不干扰决策

记忆层次:
  ShortTermMemory (单次运行，进程内):
    - 已尝试的修复及其结果 (防止重试)
    - 本轮错误模式快照
    - 实时反思草稿

  LongTermMemory (跨会话持久化，JSON 文件):
    - rules: 蒸馏后的结构化规则列表
    - stats: 全局统计 (总运行数、平均收敛轮次、常见错误类型)
    - 每条规则: {pattern, fix_hint, success_rate, occurrence_count, last_seen}
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════


@dataclass
class MemoryRule:
    """一条蒸馏后的长期记忆规则。"""
    rule_id: str                          # 关联的验证规则 ID, 如 "SEM_002"
    pattern_signature: str                # 错误模式签名, 如 "weapon_undefined"
    fix_hint: str                         # 修复提示 (简洁自然语言)
    best_fix_action: str                  # 最佳修复动作名
    success_rate: float = 0.0             # 成功率
    occurrence_count: int = 0             # 出现次数
    last_seen: str = ""                   # 最后出现时间 ISO 格式
    source_scenarios: list[str] = field(default_factory=list)  # 来源场景名

    def confidence(self) -> float:
        """置信度 = 成功率 × log(出现次数 + 1)。"""
        import math
        return self.success_rate * math.log(self.occurrence_count + 1)


@dataclass
class ShortTermEntry:
    """短期记忆条目 — 单次运行内的修复记录。"""
    round: int
    rule_id: str
    fix_action: str
    success: bool
    score_before: int
    score_after: int
    signature: str  # 去重用的签名


@dataclass
class ReflectionEntry:
    """一次运行结束后的反思。"""
    scenario_type: str
    total_rounds: int
    final_score: int
    converged: bool
    key_lessons: list[str]  # 关键教训 (简洁)
    common_errors: list[str]  # 常见错误类型
    timestamp: str = ""


# ═══════════════════════════════════════════════════════
# MemoryStore
# ═══════════════════════════════════════════════════════


class MemoryStore:
    """Reflexion-style 记忆存储。

    用法:
      memory = MemoryStore(persist_path="~/.nanobot/afsim_memory.json")

      # 运行中: 记录修复尝试
      memory.record_short_term(round=1, rule_id="SEM_002", fix_action="add_weapon",
                               success=True, score_before=70, score_after=100)

      # 遇到错误时: 按需查询 (只返回最相关的 1 条)
      hint = memory.recall(rule_id="SEM_002", context="weapon undefined")
      # → {"fix_hint": "Use platform_type + weapon WSF_EXPLICIT_WEAPON", "confidence": 0.85}

      # 运行结束后: 反思并蒸馏
      memory.reflect_and_store(scenario_type="SEAD", rounds=2, final_score=100, ...)
    """

    # 置信度阈值: 低于此值不返回 (避免低质量记忆干扰决策)
    MIN_CONFIDENCE = 0.3
    # 最大返回条数: 按需检索只返回 top-1
    MAX_RECALL_RESULTS = 1
    # 蒸馏合并阈值: pattern_signature 相同的规则合并
    DISTILL_SIMILARITY_THRESHOLD = 0.8

    def __init__(self, persist_path: str | Path | None = None):
        self.persist_path = Path(persist_path) if persist_path else None

        # 短期记忆 (每次 run() 前清空)
        self._short_term: list[ShortTermEntry] = []
        self._attempted_signatures: set[str] = set()

        # 长期记忆 (从磁盘加载)
        self._rules: list[MemoryRule] = []
        self._reflections: list[ReflectionEntry] = []
        self._stats: dict[str, Any] = {
            "total_runs": 0,
            "total_converged": 0,
            "avg_rounds": 0.0,
            "error_frequency": {},  # rule_id → count
        }

        self._load()

    # ── 短期记忆 ──────────────────────────────────────────

    def reset_short_term(self) -> None:
        """每次运行开始前清空短期记忆。"""
        self._short_term.clear()
        self._attempted_signatures.clear()

    def record_short_term(
        self,
        round: int,
        rule_id: str,
        fix_action: str,
        success: bool,
        score_before: int,
        score_after: int,
    ) -> None:
        """记录一次修复尝试到短期记忆。"""
        signature = f"{rule_id}:{fix_action}:{round}"
        entry = ShortTermEntry(
            round=round,
            rule_id=rule_id,
            fix_action=fix_action,
            success=success,
            score_before=score_before,
            score_after=score_after,
            signature=signature,
        )
        self._short_term.append(entry)
        self._attempted_signatures.add(signature)

    def already_tried(self, rule_id: str, fix_action: str) -> bool:
        """检查某修复是否已在本次运行中尝试过 (防止重试)。"""
        for entry in self._short_term:
            if entry.rule_id == rule_id and entry.fix_action == fix_action:
                return True
        return False

    def get_short_term_summary(self) -> str:
        """返回短期记忆的可读摘要 (供 ReAct prompt 使用)。"""
        if not self._short_term:
            return "No fixes attempted yet."
        lines = []
        for entry in self._short_term[-5:]:  # 最近 5 条
            status = "✓" if entry.success else "✗"
            delta = entry.score_after - entry.score_before
            lines.append(
                f"  Round {entry.round}: {status} {entry.rule_id} → "
                f"{entry.fix_action} (score {entry.score_before}→{entry.score_after}, Δ{delta:+d})"
            )
        return "\n".join(lines)

    # ── 长期记忆: 按需检索 ────────────────────────────────

    def recall(self, rule_id: str = "", context: str = "") -> dict | None:
        """按需检索长期记忆 — 只返回最相关的 1 条规则。

        这是"不激进"的核心:
        - 不在 prompt 初始时注入所有记忆
        - 只在 agent 主动调用时返回 1 条最相关的
        - 只返回置信度 >= MIN_CONFIDENCE 的规则

        Args:
            rule_id: 验证规则 ID (如 "SEM_002")
            context: 错误上下文文本 (用于关键词匹配)

        Returns:
            {"fix_hint": str, "best_fix_action": str, "confidence": float}
            或 None (无匹配)
        """
        candidates = []

        for rule in self._rules:
            score = self._match_score(rule, rule_id, context)
            if score > 0 and rule.confidence() >= self.MIN_CONFIDENCE:
                candidates.append((score, rule))

        if not candidates:
            return None

        # 取最高分
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_rule = candidates[0][1]

        return {
            "fix_hint": best_rule.fix_hint,
            "best_fix_action": best_rule.best_fix_action,
            "confidence": round(best_rule.confidence(), 2),
            "occurrence_count": best_rule.occurrence_count,
            "rule_id": best_rule.rule_id,
        }

    def _match_score(self, rule: MemoryRule, rule_id: str, context: str) -> float:
        """计算规则与当前查询的匹配分数。"""
        score = 0.0

        # rule_id 精确匹配 → 高分
        if rule_id and rule.rule_id == rule_id:
            score += 1.0

        # 上下文关键词匹配
        if context:
            context_lower = context.lower()
            pattern_lower = rule.pattern_signature.lower()
            hint_lower = rule.fix_hint.lower()

            # pattern_signature 出现在 context 中
            if pattern_lower in context_lower:
                score += 0.5

            # fix_hint 中的关键词出现在 context 中
            hint_words = set(re.findall(r"\w+", hint_lower))
            context_words = set(re.findall(r"\w+", context_lower))
            overlap = hint_words & context_words
            if overlap:
                score += 0.1 * len(overlap)

        return score

    # ── 长期记忆: 反思与蒸馏 ──────────────────────────────

    def reflect_and_store(
        self,
        scenario_type: str,
        total_rounds: int,
        final_score: int,
        converged: bool,
        key_lessons: list[str] | None = None,
        common_errors: list[str] | None = None,
    ) -> ReflectionEntry:
        """运行结束后: 反思并存储到长期记忆。

        流程:
          1. 从短期记忆中提取 (rule_id, fix_action, success) 模式
          2. 蒸馏: 合并到已有的长期规则中 (更新 success_rate, count)
          3. 存储反思条目
          4. 更新全局统计
          5. 持久化到磁盘
        """
        if key_lessons is None:
            key_lessons = self._extract_lessons()
        if common_errors is None:
            common_errors = list({e.rule_id for e in self._short_term})

        reflection = ReflectionEntry(
            scenario_type=scenario_type,
            total_rounds=total_rounds,
            final_score=final_score,
            converged=converged,
            key_lessons=key_lessons,
            common_errors=common_errors,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        self._reflections.append(reflection)

        # 蒸馏: 从短期记忆提取模式，更新长期规则
        self._distill_from_short_term()

        # 更新统计
        self._stats["total_runs"] += 1
        if converged:
            self._stats["total_converged"] += 1
        # 滑动平均
        prev_avg = self._stats["avg_rounds"]
        n = self._stats["total_runs"]
        self._stats["avg_rounds"] = round((prev_avg * (n - 1) + total_rounds) / n, 2)

        for error_type in common_errors:
            self._stats["error_frequency"][error_type] = \
                self._stats["error_frequency"].get(error_type, 0) + 1

        self._save()
        return reflection

    def _extract_lessons(self) -> list[str]:
        """从短期记忆中提取关键教训。"""
        lessons = []
        for entry in self._short_term:
            if entry.success and entry.score_after > entry.score_before:
                lessons.append(
                    f"{entry.rule_id}: {entry.fix_action} improved score "
                    f"by {entry.score_after - entry.score_before}"
                )
            elif not entry.success:
                lessons.append(
                    f"{entry.rule_id}: {entry.fix_action} did NOT help, avoid retrying"
                )
        return lessons[:5]  # 最多 5 条

    def _distill_from_short_term(self) -> None:
        """蒸馏: 从短期记忆提取模式，合并到长期规则中。"""
        # 按 (rule_id, fix_action) 分组
        groups: dict[tuple[str, str], list[ShortTermEntry]] = {}
        for entry in self._short_term:
            key = (entry.rule_id, entry.fix_action)
            groups.setdefault(key, []).append(entry)

        for (rule_id, fix_action), entries in groups.items():
            successes = sum(1 for e in entries if e.success)
            total = len(entries)
            success_rate = successes / total if total > 0 else 0.0

            # 查找已有规则
            existing = None
            for rule in self._rules:
                if rule.rule_id == rule_id and rule.best_fix_action == fix_action:
                    existing = rule
                    break

            if existing:
                # 合并: 更新成功率和计数
                old_count = existing.occurrence_count
                old_success = existing.success_rate * old_count
                new_total = old_count + total
                existing.success_rate = round((old_success + successes) / new_total, 3)
                existing.occurrence_count = new_total
                existing.last_seen = time.strftime("%Y-%m-%dT%H:%M:%S")
            else:
                # 新建规则
                # 从 fix_action 推断 pattern_signature 和 fix_hint
                pattern_sig = self._infer_pattern_signature(rule_id, fix_action)
                fix_hint = self._infer_fix_hint(rule_id, fix_action)
                self._rules.append(MemoryRule(
                    rule_id=rule_id,
                    pattern_signature=pattern_sig,
                    fix_hint=fix_hint,
                    best_fix_action=fix_action,
                    success_rate=round(success_rate, 3),
                    occurrence_count=total,
                    last_seen=time.strftime("%Y-%m-%dT%H:%M:%S"),
                ))

    def _infer_pattern_signature(self, rule_id: str, fix_action: str) -> str:
        """从 rule_id 和 fix_action 推断错误模式签名。"""
        # 基于已知规则 ID 的模式映射
        pattern_map = {
            "SEM_002": "weapon_undefined",
            "SEM_001": "invalid_side",
            "SYN_001": "unmatched_end",
            "SYN_002": "unclosed_block",
            "SYN_003": "bad_basetype",
            "SYN_004": "bad_coord",
            "SYN_007": "bad_unit",
            "CON_001": "missing_platform_type",
            "CON_002": "orphan_reference",
            "TAC_001": "unreachable_altitude",
            "TAC_003": "altitude_too_high",
        }
        return pattern_map.get(rule_id, f"{rule_id}_pattern")

    def _infer_fix_hint(self, rule_id: str, fix_action: str) -> str:
        """从 rule_id 和 fix_action 推断修复提示。"""
        hint_map = {
            "SEM_002": "Use platform_type + weapon WSF_EXPLICIT_WEAPON two-stage declaration",
            "SEM_001": "Set side to 'red' or 'blue' on platform instance only",
            "SYN_001": "Remove unmatched end_* keyword",
            "SYN_002": "Add missing end_* keyword to close block",
            "SYN_003": "Replace with valid WSF base type",
            "SYN_004": "Convert coordinates to d:m:s format",
            "SYN_007": "Use valid AFSIM units (ft, sec, km, deg, etc.)",
            "CON_001": "Add missing platform_type definition",
            "CON_002": "Fix orphaned reference or add missing definition",
            "TAC_001": "Adjust altitude to reachable range",
            "TAC_003": "Cap altitude to 60000 ft",
        }
        return hint_map.get(rule_id, f"Apply fix: {fix_action}")

    # ── 持久化 ────────────────────────────────────────────

    def _load(self) -> None:
        """从磁盘加载长期记忆。"""
        if not self.persist_path or not self.persist_path.exists():
            return

        try:
            with open(self.persist_path, encoding="utf-8") as f:
                data = json.load(f)

            self._rules = [
                MemoryRule(**r) for r in data.get("rules", [])
            ]
            self._reflections = [
                ReflectionEntry(**r) for r in data.get("reflections", [])
            ]
            self._stats = data.get("stats", self._stats)
        except (json.JSONDecodeError, TypeError, KeyError):
            # 文件损坏 → 从空开始
            pass

    def _save(self) -> None:
        """持久化到磁盘。"""
        if not self.persist_path:
            return

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "rules": [asdict(r) for r in self._rules],
            "reflections": [asdict(r) for r in self._reflections[-50:]],  # 保留最近 50 条
            "stats": self._stats,
        }

        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ── 查询辅助 ──────────────────────────────────────────

    def get_stats(self) -> dict:
        """返回全局统计信息。"""
        return dict(self._stats)

    def get_rules(self) -> list[MemoryRule]:
        """返回所有长期规则 (调试用)。"""
        return list(self._rules)

    def get_reflections(self) -> list[ReflectionEntry]:
        """返回所有反思 (调试用)。"""
        return list(self._reflections)
