"""In-run context compaction for long agent runs.

The Consolidator keeps multi-turn sessions in budget by archiving whole user
turns, and ``AgentRunner._snip_history`` trims the prompt as a last resort.
Neither copes with one long turn: a single user message followed by hundreds
of tool calls has no user boundary to cut at, so snipping keeps the whole turn
and the prompt grows without bound.  ``ContextCompactor`` covers that case.

When the prompt nears the context budget, the messages between the system
prompt and a recent tail are folded into a structured summary written by the
model (requests, technical concepts, files, errors, open work, next step).
The folded messages are written verbatim to a JSONL transcript in the
workspace and the summary points at it, so the agent can grep or read details
back when the summary is not enough.  User messages and the files touched by
tool calls are listed next to the summary without going through the model, so
the task statement cannot be lost to a bad summary.  Later compactions merge
the previous summary with the newly folded messages.

Compaction only changes the view sent to the model.  The run's message list,
which callers persist, is never modified.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import tiktoken
from loguru import logger

from nanobot.providers.base import LLMProvider
from nanobot.trajectory.instrumentation import context_boundary
from nanobot.utils.helpers import (
    ensure_dir,
    estimate_message_tokens,
    safe_filename,
    strip_think,
    truncate_text,
)
from nanobot.utils.prompt_templates import render_template

_TRANSCRIPTS_DIR = ".nanobot/transcripts"
# Fold at least this many messages and this share of the budget; anything
# smaller cannot bring the prompt back under the threshold.
_MIN_FOLD_MESSAGES = 2
_MIN_FOLD_SHARE = 0.1
_PROMPT_OVERHEAD_TOKENS = 512
# (tool result, tool arguments, assistant text) clip limits for the summarizer
# input: loose first, tight when the segment does not fit.
_CLIP_LIMITS = ((1500, 500, 3000), (300, 150, 800))
_USER_MESSAGE_CHARS = 4000
_PINNED_USER_CHARS = 1500
_MAX_PINNED_USER_MESSAGES = 8
_MAX_LISTED_FILES = 20
_FILE_ARG_KEYS = ("path", "file_path", "filepath", "filename")
_SUMMARY_RE = re.compile(r"<summary>(.*?)(?:</summary>|\Z)", re.S)
_ANALYSIS_RE = re.compile(r"<analysis>.*?(?:</analysis>|\Z)", re.S)


@lru_cache(maxsize=1)
def _encoding() -> Any:
    return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    try:
        return len(_encoding().encode(text))
    except Exception:
        return len(text) // 4


def _clip_tokens(text: str, max_tokens: int) -> str:
    try:
        tokens = _encoding().encode(text)
        if len(tokens) <= max_tokens:
            return text
        return _encoding().decode(tokens[:max_tokens]) + "\n... (truncated)"
    except Exception:
        return truncate_text(text, max_tokens * 4)


def _one_line(text: str, max_chars: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= max_chars else flat[:max_chars] + "..."


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, dict):
                parts.append(f"[{block.get('type') or 'block'}]")
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part)
    return json.dumps(content, ensure_ascii=False, default=str)


def _tool_call_parts(tool_call: Any) -> tuple[str, str]:
    if not isinstance(tool_call, dict):
        return "tool", ""
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return "tool", ""
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, default=str) if arguments else ""
    return str(function.get("name") or "tool"), arguments


def _leading_system_count(messages: list[dict[str, Any]]) -> int:
    count = 0
    for message in messages:
        if message.get("role") != "system":
            break
        count += 1
    return count


def _format_message(
    message: dict[str, Any],
    *,
    result_chars: int,
    args_chars: int,
    text_chars: int,
) -> str | None:
    role = message.get("role")
    text = _content_text(message.get("content"))
    if role == "user":
        return f"[user]\n{truncate_text(text, _USER_MESSAGE_CHARS)}"
    if role == "assistant":
        lines = [f"[assistant]\n{truncate_text(text, text_chars)}" if text else "[assistant]"]
        for tool_call in message.get("tool_calls") or []:
            name, arguments = _tool_call_parts(tool_call)
            lines.append(f"-> {name}({truncate_text(arguments, args_chars)})")
        return "\n".join(lines)
    if role == "tool":
        name = message.get("name") or "tool"
        return f"[tool result: {name}]\n{truncate_text(text, result_chars)}"
    if text:
        return f"[{role}]\n{truncate_text(text, text_chars)}"
    return None


def _extract_summary(content: str | None) -> str:
    text = strip_think(content or "").strip()
    match = _SUMMARY_RE.search(text)
    if match:
        return match.group(1).strip()
    return _ANALYSIS_RE.sub("", text).strip()


class ContextCompactor:
    """Fold the older part of one run into a summary when the prompt nears its budget.

    One instance lives for one ``AgentRunner.run`` call.  ``upto`` indexes the
    run's message list: messages before it (after the leading system prompt)
    are represented by ``summary``.
    """

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        *,
        budget: int,
        trigger_ratio: float = 0.8,
        keep_ratio: float = 0.3,
        summary_ratio: float = 0.15,
        workspace: Path | None = None,
        session_key: str | None = None,
        retry_mode: str = "standard",
    ) -> None:
        self.provider = provider
        self.model = model
        self.budget = budget
        self.threshold = int(budget * trigger_ratio)
        self.keep_tokens = int(budget * keep_ratio)
        self.summary_tokens = max(256, min(int(budget * summary_ratio), 6000))
        self.workspace = workspace
        self.session_key = session_key
        self.retry_mode = retry_mode
        self.summary: str | None = None
        self.upto = 0
        self.folded = 0
        self.compactions = 0
        self.transcript_path: Path | None = None
        self._run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        self._user_messages: list[str] = []
        self._files: list[str] = []
        self._notice: dict[str, Any] | None = None
        self._usage: dict[str, int] = {}

    def view(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return the messages the model should see, with the folded part replaced."""
        if self._notice is None:
            return messages
        head = _leading_system_count(messages)
        return [*messages[:head], dict(self._notice), *messages[self.upto:]]

    def take_usage(self) -> dict[str, int]:
        """Return and reset token usage spent on summarization."""
        usage, self._usage = self._usage, {}
        return usage

    @context_boundary
    async def maybe_compact(
        self,
        messages: list[dict[str, Any]],
        measure: Callable[[list[dict[str, Any]]], int],
    ) -> list[dict[str, Any]]:
        """Compact when ``measure(view)`` exceeds the threshold; return the view to send."""
        view = self.view(messages)
        before = measure(view)
        if before <= self.threshold:
            return view
        start = max(self.upto, _leading_system_count(messages))
        cut = self._pick_cut(messages, start)
        if cut is None:
            return view
        segment = messages[start:cut]
        segment_tokens = sum(estimate_message_tokens(message) for message in segment)
        if len(segment) < _MIN_FOLD_MESSAGES or segment_tokens < self.budget * _MIN_FOLD_SHARE:
            return view

        self.transcript_path = self._write_transcript(segment) or self.transcript_path
        summary = await self._summarize(segment)
        self._remember(segment)
        self.summary = summary
        self.upto = cut
        self.folded += len(segment)
        self.compactions += 1
        self._notice = {"role": "user", "content": self._render_notice()}
        compacted = self.view(messages)
        logger.info(
            "Context compaction #{} for {}: folded {} messages, prompt ~{} -> ~{} tokens (budget {}), transcript {}",
            self.compactions,
            self.session_key or "default",
            len(segment),
            before,
            measure(compacted),
            self.budget,
            self.transcript_path or "not saved",
        )
        return compacted

    def _pick_cut(self, messages: list[dict[str, Any]], start: int) -> int | None:
        """Pick where the kept tail starts: recent messages within ``keep_tokens``."""
        count = len(messages)
        cut = count
        kept = 0
        for idx in range(count - 1, start - 1, -1):
            tokens = estimate_message_tokens(messages[idx])
            if cut < count and kept + tokens > self.keep_tokens:
                break
            kept += tokens
            cut = idx
        # Never start the tail with tool results: keep the call that produced them.
        while cut > start and cut < count and messages[cut].get("role") == "tool":
            cut -= 1
        if cut <= start or cut >= count:
            return None
        return cut

    def _write_transcript(self, segment: list[dict[str, Any]]) -> Path | None:
        if self.workspace is None:
            return None
        try:
            root = ensure_dir(
                Path(self.workspace) / _TRANSCRIPTS_DIR / safe_filename(self.session_key or "default")
            )
            path = root / f"{self._run_id}.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                for message in segment:
                    handle.write(json.dumps(message, ensure_ascii=False, default=str) + "\n")
            return path
        except Exception:
            logger.exception("Failed to write compaction transcript for {}", self.session_key or "default")
            return None

    async def _summarize(self, segment: list[dict[str, Any]]) -> str:
        max_words = max(150, self.summary_tokens * 3 // 4)
        system = render_template("agent/context_compact.md", strip=True, max_words=max_words)
        prompt = self._render_input(segment, self.budget - _count_tokens(system) - _PROMPT_OVERHEAD_TOKENS)
        try:
            response = await self.provider.chat_with_retry(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                tools=None,
                tool_choice=None,
                retry_mode=self.retry_mode,
            )
            if response.finish_reason == "error":
                raise RuntimeError(f"LLM returned error: {response.content}")
            for key, value in (response.usage or {}).items():
                if isinstance(value, int):
                    self._usage[key] = self._usage.get(key, 0) + value
            summary = _extract_summary(response.content)
            if not summary:
                raise RuntimeError("LLM returned an empty summary")
        except Exception:
            logger.exception(
                "Context compaction summary failed for {}; using a mechanical digest",
                self.session_key or "default",
            )
            summary = self._fallback_summary(segment)
        return _clip_tokens(summary, self.summary_tokens * 2)

    def _render_input(self, segment: list[dict[str, Any]], max_tokens: int) -> str:
        header = f"<previous_summary>\n{self.summary}\n</previous_summary>\n\n" if self.summary else ""
        lines: list[str] = []
        for result_chars, args_chars, text_chars in _CLIP_LIMITS:
            lines = [
                line
                for message in segment
                if (line := _format_message(
                    message,
                    result_chars=result_chars,
                    args_chars=args_chars,
                    text_chars=text_chars,
                ))
            ]
            text = header + "<messages>\n" + "\n\n".join(lines) + "\n</messages>"
            if _count_tokens(text) <= max_tokens:
                return text
        # Still too long: keep the newest messages that fit; the rest is in the transcript.
        available = max_tokens - _count_tokens(header) - 64
        kept: list[str] = []
        for line in reversed(lines):
            tokens = _count_tokens(line) + 2
            if kept and tokens > available:
                break
            kept.append(line)
            available -= tokens
        kept.reverse()
        omitted = len(lines) - len(kept)
        note = f"[{omitted} older messages omitted for length; they are in the transcript]\n\n" if omitted else ""
        return header + "<messages>\n" + note + "\n\n".join(kept) + "\n</messages>"

    def _fallback_summary(self, segment: list[dict[str, Any]]) -> str:
        lines = ["(No model summary was available; this is a mechanical digest of the removed messages.)"]
        if self.summary:
            lines += ["", "Earlier summary:", self.summary]
        notes = [
            _one_line(_content_text(message.get("content")), 300)
            for message in segment
            if message.get("role") == "assistant" and _content_text(message.get("content")).strip()
        ]
        if notes:
            lines += ["", "Assistant notes (latest last):", *(f"- {note}" for note in notes[-10:])]
        results = {
            str(message.get("tool_call_id")): _content_text(message.get("content"))
            for message in segment
            if message.get("role") == "tool"
        }
        calls: list[str] = []
        for message in segment:
            if message.get("role") != "assistant":
                continue
            for tool_call in message.get("tool_calls") or []:
                name, arguments = _tool_call_parts(tool_call)
                call_id = str(tool_call.get("id")) if isinstance(tool_call, dict) else ""
                result = _one_line(results.get(call_id, "(no result)"), 160)
                calls.append(f"- {name}({_one_line(arguments, 120)}) -> {result}")
        if calls:
            lines += ["", f"Tool calls ({len(calls)}, latest last):", *calls[-40:]]
        return "\n".join(lines)

    def _remember(self, segment: list[dict[str, Any]]) -> None:
        for message in segment:
            role = message.get("role")
            if role == "user":
                text = _content_text(message.get("content")).strip()
                if text:
                    self._user_messages.append(truncate_text(text, _PINNED_USER_CHARS))
            elif role == "assistant":
                for tool_call in message.get("tool_calls") or []:
                    _, arguments = _tool_call_parts(tool_call)
                    try:
                        parsed = json.loads(arguments) if arguments else {}
                    except ValueError:
                        continue
                    if not isinstance(parsed, dict):
                        continue
                    for key in _FILE_ARG_KEYS:
                        value = parsed.get(key)
                        if isinstance(value, str) and value.strip():
                            if value in self._files:
                                self._files.remove(value)
                            self._files.append(value)

    def _render_notice(self) -> str:
        users = self._user_messages
        if len(users) > _MAX_PINNED_USER_MESSAGES:
            # Keep the opening request and the latest follow-ups.
            users = [users[0], *users[-(_MAX_PINNED_USER_MESSAGES - 1):]]
        return render_template(
            "agent/context_compact_notice.md",
            strip=True,
            folded=self.folded,
            transcript=str(self.transcript_path) if self.transcript_path else "",
            summary=self.summary or "",
            files=self._files[-_MAX_LISTED_FILES:],
            user_messages=users,
        )
