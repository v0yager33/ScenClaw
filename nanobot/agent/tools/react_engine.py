"""ReAct Engine — Reasoning + Acting 自主规划引擎。

基于 ReAct 论文: 交替执行推理 (Thought) 和行动 (Action)，
通过观察 (Observation) 反馈来驱动多步决策。

与传统固定流水线的区别:
  固定流水线: generate → validate → fix → validate → stop (步骤固定)
  ReAct:      Thought → Action → Observation → Thought → ... (步骤动态)

核心设计:
  1. LLM 在每一步生成 Thought (推理) + Action (工具选择)
  2. 引擎执行 Action，返回 Observation
  3. LLM 根据 Observation 决定下一步
  4. 循环直到 LLM 选择 "stop" 或达到最大步数

记忆集成 (Reflexion):
  - 短期记忆: 注入到每一步的 prompt 中 (已尝试的修复摘要)
  - 长期记忆: 不主动注入，只在 LLM 调用 recall action 时按需检索
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from nanobot.agent.tools.memory_store import MemoryStore


# ═══════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════


@dataclass
class ReActStep:
    """ReAct 循环的一步: Thought → Action → Observation。"""
    step: int
    thought: str
    action: str
    action_input: dict
    observation: str


@dataclass
class ReActResult:
    """ReAct 循环的最终结果。"""
    success: bool
    final_code: str
    final_score: int
    final_errors: int
    total_steps: int
    trajectory: list[ReActStep] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════
# LLM Client (httpx 直调 DeepSeek API)
# ═══════════════════════════════════════════════════════


class DeepSeekClient:
    """轻量级 DeepSeek API 客户端 (httpx 直调，不依赖 openai SDK)。

    只依赖 OpenAI 兼容的 ``/v1/chat/completions``，因此把 ``base_url`` 换成任何
    兼容服务 (vLLM、Azure、通义千问兼容模式等) 都能直接用。
    """

    ENV_API_KEY = "DEEPSEEK_API_KEY"
    ENV_BASE_URL = "DEEPSEEK_BASE_URL"
    ENV_MODEL = "DEEPSEEK_MODEL"

    DEFAULT_BASE_URL = "https://api.deepseek.com"
    DEFAULT_MODEL = "deepseek-chat"

    # 值得重试的 HTTP 状态: 限流与网关/服务端抖动
    RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

    def __init__(self, api_key: str = "", base_url: str = "",
                 model: str = "", timeout: int = 120,
                 max_retries: int = 3, retry_backoff_s: float = 2.0):
        self.api_key = api_key or os.environ.get(self.ENV_API_KEY, "")
        if not self.api_key:
            raise ValueError(
                f"缺少 API key: 显式传入 api_key 或设置环境变量 {self.ENV_API_KEY}"
            )
        base = base_url or os.environ.get(self.ENV_BASE_URL) or self.DEFAULT_BASE_URL
        self.base_url = base.rstrip("/")
        self.model = model or os.environ.get(self.ENV_MODEL) or self.DEFAULT_MODEL
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.retry_backoff_s = retry_backoff_s
        # 最近一次调用的 finish_reason; "length" 表示输出被 max_tokens 截断。
        # 推理模型的思维链与正文共用 max_tokens, 大想定最容易在这里被截断。
        self.last_finish_reason: str | None = None
        # 最近一次调用的 usage; 推理模型在 completion_tokens_details 里给出思维链占了多少
        self.last_usage: dict[str, Any] | None = None

    @classmethod
    def from_env(cls, **overrides: Any) -> "DeepSeekClient":
        """从环境变量构造客户端; 显式参数优先。

        让 API key 走环境变量而不是工具参数, 避免密钥被写进 trace / 反思记忆 /
        日志里。
        """
        return cls(**overrides)

    def chat(self, messages: list[dict], temperature: float = 0.3,
             max_tokens: int = 2048, json_object: bool = False) -> str:
        """调用 chat completions API，返回助手回复文本。

        Args:
            json_object: 打开服务端 JSON 模式, 强制返回可解析的 JSON 对象。
                用于结构化抽取 (自然语言 → ScenarioIR), 免得再去正则抠代码块。
        """
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_object:
            payload["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, headers=headers, json=payload)
                if resp.status_code in self.RETRYABLE_STATUS:
                    raise httpx.HTTPStatusError(
                        f"retryable status {resp.status_code}: {resp.text[:200]}",
                        request=resp.request, response=resp,
                    )
                resp.raise_for_status()
                body = resp.json()
                choice = body["choices"][0]
                self.last_finish_reason = choice.get("finish_reason")
                self.last_usage = body.get("usage")
                return choice["message"]["content"] or ""
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status is not None and status not in self.RETRYABLE_STATUS:
                    raise  # 4xx 配置/鉴权问题, 重试无意义
                last_error = exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_backoff_s * (2 ** attempt))

        raise RuntimeError(
            f"DeepSeek API 在 {self.max_retries} 次尝试后仍失败: {last_error}"
        ) from last_error


# ═══════════════════════════════════════════════════════
# ReAct Engine
# ═══════════════════════════════════════════════════════


# 可用动作列表 (供 LLM 选择)
AVAILABLE_ACTIONS = [
    "generate",    # 生成 AFSIM 代码
    "validate",    # 验证当前代码
    "analyze",     # 分析错误 (构建依赖图 + PageRank)
    "fix",         # 应用修复
    "recall",      # 查询长期记忆 (按需)
    "reflect",     # 反思并记录经验
    "stop",        # 终止循环
]

ACTION_DESCRIPTIONS = """Available actions:
1. generate - Generate AFSIM scenario code from COA data. Input: {"coa_data": ..., "coords": ..., "matches": ..., "scenario_name": ..., "end_time": ...}
2. validate - Validate current AFSIM code. Input: {"code": "current code"} (omit to use last generated code)
3. analyze - Analyze validation errors using error dependency graph + PageRank. Input: {"issues": [...]} (omit to use last validation issues)
4. fix - Apply a specific fix to the code. Input: {"rule_id": "SEM_002", "issue": {...}} (use issues from validate)
5. recall - Query long-term memory for past experience with similar errors. Input: {"rule_id": "SEM_002", "context": "error description text"}
6. reflect - Generate reflection and store to memory. Input: {"key_lessons": ["lesson1", ...]}
7. stop - Stop the loop when satisfied with the result. Input: {}"""


class ReActEngine:
    """ReAct-style 自主规划引擎。

    用法:
      engine = ReActEngine(llm_client, tools, memory)
      result = engine.run(task="Generate AFSIM from SEAD COA", context={...})
    """

    SYSTEM_PROMPT = """You are an AFSIM code generation agent that uses reasoning and acting to generate valid AFSIM scenario code.

You work in a Thought → Action → Observation loop:
- Thought: reason about the current state and decide what to do next
- Action: choose one action from the available actions
- Observation: see the result of your action

Key principles:
1. Start by generating code, then validate, then fix errors iteratively
2. When you encounter an unfamiliar error, use "recall" to check past experience
3. Use "analyze" to find root cause errors (fixing root causes cascades to fix downstream errors)
4. After fixing, always re-validate to check if the fix worked
5. Use "reflect" before stopping to record lessons for future runs
6. Stop when score >= 85 and errors == 0

{action_descriptions}

Respond in EXACTLY this format (no markdown, no code blocks):
Thought: <your reasoning>
Action: <action name>
Action Input: <JSON object>"""

    def __init__(
        self,
        llm_client: DeepSeekClient,
        tools: dict[str, Callable],
        memory: MemoryStore,
        max_steps: int = 15,
        target_score: int = 85,
    ):
        self.llm = llm_client
        self.tools = tools
        self.memory = memory
        self.max_steps = max_steps
        self.target_score = target_score
        self.trajectory: list[ReActStep] = []

        # 运行时状态
        self._current_code: str = ""
        self._current_score: int = 0
        self._current_errors: int = 0
        self._current_issues: list[dict] = []
        self._context: dict = {}

    def run(self, task: str, context: dict) -> ReActResult:
        """执行 ReAct 循环。

        Args:
            task: 任务描述 (如 "Generate AFSIM from SEAD COA")
            context: 运行上下文 (coa_data, coords, matches, scenario_name, end_time)

        Returns:
            ReActResult with final code, score, trajectory
        """
        self._context = context
        self.trajectory = []
        self.memory.reset_short_term()

        system_prompt = self.SYSTEM_PROMPT.format(
            action_descriptions=ACTION_DESCRIPTIONS
        )

        conversation: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._build_initial_prompt(task, context)},
        ]

        for step in range(1, self.max_steps + 1):
            # LLM 生成 Thought + Action
            try:
                response = self.llm.chat(
                    conversation,
                    temperature=0.3,
                    max_tokens=1024,
                )
            except Exception as e:
                # API 调用失败 → 用降级策略
                return self._fallback_run(task, context, f"LLM API error: {e}")

            # 解析 Thought + Action + Action Input
            thought, action, action_input = self._parse_response(response)

            if not action:
                # 解析失败 → 提示 LLM 重新格式化
                conversation.append({"role": "assistant", "content": response})
                conversation.append({
                    "role": "user",
                    "content": "Invalid format. Please respond with:\nThought: ...\nAction: ...\nAction Input: {...}",
                })
                continue

            # 执行 Action
            observation = self._execute_action(action, action_input, step)

            # 记录轨迹
            react_step = ReActStep(
                step=step,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
            )
            self.trajectory.append(react_step)

            # 更新对话
            conversation.append({"role": "assistant", "content": response})
            conversation.append({"role": "user", "content": f"Observation: {observation}"})

            # 检查终止条件
            if action == "stop":
                break

            # 检查自动终止: 分数达标
            if self._current_score >= self.target_score and self._current_errors == 0:
                # 自动反思并终止
                self._auto_reflect(task)
                break

        # 如果达到最大步数还没 stop，自动反思
        if self.trajectory and self.trajectory[-1].action != "stop":
            self._auto_reflect(task)

        success = self._current_score >= self.target_score and self._current_errors == 0

        return ReActResult(
            success=success,
            final_code=self._current_code,
            final_score=self._current_score,
            final_errors=self._current_errors,
            total_steps=len(self.trajectory),
            trajectory=self.trajectory,
            reflections=[s.thought for s in self.trajectory if "lesson" in s.thought.lower()],
        )

    # ── Prompt 构建 ──────────────────────────────────────

    def _build_initial_prompt(self, task: str, context: dict) -> str:
        """构建初始用户 prompt。"""
        coa_summary = self._summarize_coa(context.get("coa_data", {}))
        short_term = self.memory.get_short_term_summary()

        prompt = f"""Task: {task}

Scenario: {context.get('scenario_name', 'unknown')}
COA Summary: {coa_summary}
End Time: {context.get('end_time', 21600)} seconds

Short-term memory (fixes tried so far):
{short_term}

Start by generating the AFSIM code."""

        return prompt

    def _summarize_coa(self, coa_data: dict) -> str:
        """生成 COA 数据的简洁摘要。"""
        if not coa_data:
            return "No COA data provided."

        entities = coa_data.get("entities", [])
        phases = coa_data.get("phases", [])
        sides = set(e.get("side", "unknown") for e in entities)

        return (
            f"{len(entities)} entities ({len(sides)} sides: {', '.join(sides)}), "
            f"{len(phases)} phases"
        )

    # ── 响应解析 ──────────────────────────────────────────

    def _parse_response(self, response: str) -> tuple[str, str, dict]:
        """解析 LLM 的响应为 (thought, action, action_input)。

        期望格式:
          Thought: <reasoning>
          Action: <action_name>
          Action Input: <JSON>
        """
        thought = ""
        action = ""
        action_input = {}

        # 提取 Thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|$)", response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # 提取 Action
        action_match = re.search(r"Action:\s*(\w+)", response)
        if action_match:
            action = action_match.group(1).strip().lower()

        # 提取 Action Input (JSON)
        input_match = re.search(r"Action Input:\s*(\{.*?\})", response, re.DOTALL)
        if input_match:
            try:
                action_input = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                # 尝试更宽松的解析
                try:
                    # 移除可能的尾随逗号
                    raw = input_match.group(1).rstrip()
                    raw = re.sub(r",\s*}", "}", raw)
                    action_input = json.loads(raw)
                except json.JSONDecodeError:
                    action_input = {}

        # 验证 action 是否合法
        if action and action not in AVAILABLE_ACTIONS:
            action = ""

        return thought, action, action_input

    # ── 动作执行 ──────────────────────────────────────────

    def _execute_action(self, action: str, action_input: dict, step: int) -> str:
        """执行动作，返回观察结果。"""
        try:
            if action == "generate":
                return self._do_generate(action_input)
            elif action == "validate":
                return self._do_validate(action_input)
            elif action == "analyze":
                return self._do_analyze(action_input)
            elif action == "fix":
                return self._do_fix(action_input, step)
            elif action == "recall":
                return self._do_recall(action_input)
            elif action == "reflect":
                return self._do_reflect(action_input)
            elif action == "stop":
                return "Loop stopped."
            else:
                return f"Unknown action: {action}"
        except Exception as e:
            return f"Action '{action}' failed: {e}"

    def _do_generate(self, action_input: dict) -> str:
        """执行代码生成。"""
        ctx = self._context
        coa_data = action_input.get("coa_data", ctx.get("coa_data", {}))
        coords = action_input.get("coords", ctx.get("coords", {}))
        matches = action_input.get("matches", ctx.get("matches", {}))
        scenario_name = action_input.get("scenario_name", ctx.get("scenario_name", "generated"))
        end_time = action_input.get("end_time", ctx.get("end_time", 21600))

        result = self.tools["generate"](coa_data, coords, matches, scenario_name, end_time)

        if isinstance(result, dict):
            self._current_code = result.get("scenario_code", "")
        else:
            self._current_code = str(result)

        line_count = len(self._current_code.split("\n"))
        return f"Generated {line_count} lines of AFSIM code. Use 'validate' to check."

    def _do_validate(self, action_input: dict) -> str:
        """执行验证。"""
        code = action_input.get("code", self._current_code)
        if not code:
            return "No code to validate. Generate first."

        result = self.tools["validate"](code)

        if isinstance(result, dict):
            val_data = result
        else:
            try:
                val_data = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                return f"Validation returned invalid result: {str(result)[:200]}"

        self._current_score = val_data.get("score", 0)
        self._current_errors = val_data.get("summary", {}).get("errors", 0)
        self._current_issues = val_data.get("issues", [])

        # 格式化观察结果
        issues_str = self._format_issues(self._current_issues)
        return (
            f"Score: {self._current_score}/100, "
            f"Errors: {self._current_errors}, "
            f"Warnings: {val_data.get('summary', {}).get('warnings', 0)}\n"
            f"Issues:\n{issues_str}"
        )

    def _do_analyze(self, action_input: dict) -> str:
        """执行错误分析 (依赖图 + PageRank)。"""
        issues = action_input.get("issues", self._current_issues)
        if not issues:
            return "No issues to analyze."

        result = self.tools["analyze"](issues)

        if isinstance(result, dict):
            root_causes = result.get("root_causes", [])
        else:
            try:
                result_data = json.loads(result)
                root_causes = result_data.get("root_causes", [])
            except (json.JSONDecodeError, TypeError):
                return f"Analysis failed: {str(result)[:200]}"

        if not root_causes:
            return "No root causes found. All issues may be independent."

        lines = ["Root causes (by PageRank, fix these first):"]
        for i, (idx, pr) in enumerate(root_causes[:5]):
            issue = issues[idx] if idx < len(issues) else {}
            lines.append(
                f"  {i+1}. [{issue.get('rule', '?')}] PageRank={pr:.3f} - "
                f"{issue.get('description', '')[:80]}"
            )
        return "\n".join(lines)

    def _do_fix(self, action_input: dict, step: int) -> str:
        """执行修复。"""
        rule_id = action_input.get("rule_id", "")
        issue = action_input.get("issue", {})

        if not issue and self._current_issues:
            # 从 rule_id 查找对应的 issue
            for i in self._current_issues:
                if i.get("rule") == rule_id:
                    issue = i
                    break

        if not issue:
            return f"No issue found for rule_id '{rule_id}'."

        # 检查是否已尝试过
        fix_action = f"fix_{rule_id}"
        if self.memory.already_tried(rule_id, fix_action):
            return f"Already tried fixing {rule_id} this run. Try a different approach."

        score_before = self._current_score
        code_before = self._current_code

        result = self.tools["fix"](self._current_code, issue)

        if isinstance(result, tuple):
            new_code = result[0]
        elif isinstance(result, dict):
            new_code = result.get("code", self._current_code)
        else:
            new_code = self._current_code

        if new_code == code_before:
            self.memory.record_short_term(
                step, rule_id, fix_action, False, score_before, self._current_score
            )
            return f"Fix for {rule_id} produced no change. Try a different approach."

        self._current_code = new_code

        # 自动重新验证
        val_result = self.tools["validate"](new_code)
        if isinstance(val_result, dict):
            val_data = val_result
        else:
            try:
                val_data = json.loads(val_result)
            except (json.JSONDecodeError, TypeError):
                val_data = {}

        new_score = val_data.get("score", 0)
        new_errors = val_data.get("summary", {}).get("errors", 0)
        new_issues = val_data.get("issues", [])

        success = new_score >= score_before
        self.memory.record_short_term(
            step, rule_id, fix_action, success, score_before, new_score
        )

        self._current_score = new_score
        self._current_errors = new_errors
        self._current_issues = new_issues

        delta = new_score - score_before
        status = "improved" if delta > 0 else ("same" if delta == 0 else "degraded")
        return (
            f"Applied fix for {rule_id}. Score: {score_before}→{new_score} ({status}, Δ{delta:+d}). "
            f"Errors: {new_errors}."
        )

    def _do_recall(self, action_input: dict) -> str:
        """按需查询长期记忆。"""
        rule_id = action_input.get("rule_id", "")
        context = action_input.get("context", "")

        hint = self.memory.recall(rule_id=rule_id, context=context)

        if hint is None:
            return "No relevant past experience found. Proceed with standard fix strategy."

        return (
            f"Past experience (confidence: {hint['confidence']}, "
            f"seen {hint['occurrence_count']} times):\n"
            f"  Rule: {hint['rule_id']}\n"
            f"  Hint: {hint['fix_hint']}\n"
            f"  Best fix: {hint['best_fix_action']}"
        )

    def _do_reflect(self, action_input: dict) -> str:
        """反思并记录。"""
        key_lessons = action_input.get("key_lessons", [])
        if not key_lessons:
            key_lessons = self.memory._extract_lessons()

        scenario_type = self._context.get("scenario_name", "unknown")
        total_rounds = len([s for s in self.trajectory if s.action == "fix"])
        converged = self._current_score >= self.target_score and self._current_errors == 0

        self.memory.reflect_and_store(
            scenario_type=scenario_type,
            total_rounds=total_rounds,
            final_score=self._current_score,
            converged=converged,
            key_lessons=key_lessons,
        )

        return f"Reflection stored. {len(key_lessons)} lessons learned."

    def _auto_reflect(self, task: str) -> None:
        """自动反思 (当达到目标或最大步数时)。"""
        scenario_type = self._context.get("scenario_name", "unknown")
        total_rounds = len([s for s in self.trajectory if s.action == "fix"])
        converged = self._current_score >= self.target_score and self._current_errors == 0

        self.memory.reflect_and_store(
            scenario_type=scenario_type,
            total_rounds=total_rounds,
            final_score=self._current_score,
            converged=converged,
        )

    # ── 辅助 ──────────────────────────────────────────────

    def _format_issues(self, issues: list[dict]) -> str:
        """格式化问题列表为可读字符串。"""
        if not issues:
            return "  (no issues)"
        lines = []
        for i, issue in enumerate(issues[:10]):  # 最多 10 条
            severity = issue.get("severity", "?")
            rule = issue.get("rule", "?")
            desc = issue.get("description", "")[:80]
            line = issue.get("line", "?")
            lines.append(f"  [{severity}] {rule} (line {line}): {desc}")
        if len(issues) > 10:
            lines.append(f"  ... and {len(issues) - 10} more")
        return "\n".join(lines)

    def _fallback_run(self, task: str, context: dict, error_msg: str) -> ReActResult:
        """LLM API 不可用时的降级策略: 用原有固定流水线。"""
        # 直接生成 + 验证
        try:
            ctx = context
            gen_result = self.tools["generate"](
                ctx.get("coa_data", {}), ctx.get("coords", {}),
                ctx.get("matches", {}), ctx.get("scenario_name", "generated"),
                ctx.get("end_time", 21600),
            )
            if isinstance(gen_result, dict):
                code = gen_result.get("scenario_code", "")
            else:
                code = str(gen_result)

            val_result = self.tools["validate"](code)
            if isinstance(val_result, dict):
                val_data = val_result
            else:
                val_data = json.loads(val_result)

            self._current_code = code
            self._current_score = val_data.get("score", 0)
            self._current_errors = val_data.get("summary", {}).get("errors", 0)
            self._current_issues = val_data.get("issues", [])
        except Exception:
            pass

        return ReActResult(
            success=False,
            final_code=self._current_code,
            final_score=self._current_score,
            final_errors=self._current_errors,
            total_steps=0,
            trajectory=[],
            reflections=[f"Fallback mode: {error_msg}"],
        )
