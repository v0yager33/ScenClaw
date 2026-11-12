"""Tests for in-run context compaction (ContextCompactor and its AgentRunner wiring)."""

from __future__ import annotations

import json
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.context_compactor import ContextCompactor
from nanobot.config.schema import AgentDefaults
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.utils.helpers import estimate_message_tokens, estimate_prompt_tokens_chain

_TASK = "Build the scenario and keep every platform ID stable."


def _measure(view):
    return sum(estimate_message_tokens(message) for message in view)


def _step(i: int, *, words: int = 120) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": f"step {i}",
            "tool_calls": [{
                "id": f"c{i}",
                "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": f"src/m{i}.py"})},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": f"c{i}",
            "name": "read_file",
            "content": f"result {i} " + " ".join(f"tok{i}_{k}" for k in range(words)),
        },
    ]


def _long_run(steps: int, start: int = 0) -> list[dict]:
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": _TASK}]
    for i in range(start, start + steps):
        messages.extend(_step(i))
    return messages


def _summary_provider(*summaries: str) -> tuple[MagicMock, list[list[dict]]]:
    provider = MagicMock()
    prompts: list[list[dict]] = []
    replies = iter(summaries)

    async def chat_with_retry(*, messages, **kwargs):
        prompts.append(messages)
        return LLMResponse(
            content=f"<analysis>notes</analysis>\n<summary>{next(replies)}</summary>",
            tool_calls=[],
            usage={"prompt_tokens": 7, "completion_tokens": 3},
        )

    provider.chat_with_retry = chat_with_retry
    return provider, prompts


@pytest.mark.asyncio
async def test_compactor_leaves_prompt_alone_below_threshold():
    messages = _long_run(4)
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock()
    compactor = ContextCompactor(provider, "m", budget=_measure(messages) * 2)

    view = await compactor.maybe_compact(messages, _measure)

    assert view is messages
    provider.chat_with_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_compactor_folds_old_messages_into_summary(tmp_path):
    messages = _long_run(12)
    original = deepcopy(messages)
    provider, prompts = _summary_provider("S1 read twelve modules")
    compactor = ContextCompactor(
        provider, "m", budget=int(_measure(messages) * 0.9), workspace=tmp_path, session_key="cli:direct",
    )

    view = await compactor.maybe_compact(messages, _measure)

    assert messages == original
    assert len(prompts) == 1
    prompt = prompts[0][1]["content"]
    assert "<previous_summary>" not in prompt
    assert f"[user]\n{_TASK}" in prompt
    assert "-> read_file(" in prompt

    assert view[0] == messages[0]
    notice = view[1]
    assert notice["role"] == "user"
    assert notice["content"].startswith("[Context compacted]")
    assert "S1 read twelve modules" in notice["content"]
    assert "notes" not in notice["content"]
    assert _TASK in notice["content"]
    assert "- src/m0.py" in notice["content"]
    assert view[2:] == messages[compactor.upto:]
    assert view[2]["role"] == "assistant"
    assert _measure(view) < _measure(messages)

    transcript = compactor.transcript_path
    assert transcript is not None and transcript.parent == tmp_path / ".nanobot" / "transcripts" / "cli_direct"
    assert str(transcript) in notice["content"]
    lines = transcript.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == messages[1:compactor.upto]
    assert compactor.take_usage() == {"prompt_tokens": 7, "completion_tokens": 3}
    assert compactor.take_usage() == {}


@pytest.mark.asyncio
async def test_compactor_merges_previous_summary_on_next_compaction(tmp_path):
    messages = _long_run(12)
    provider, prompts = _summary_provider("S1 first part", "S2 merged")
    budget = int(_measure(messages) * 0.9)
    compactor = ContextCompactor(provider, "m", budget=budget, workspace=tmp_path)
    await compactor.maybe_compact(messages, _measure)
    first_upto = compactor.upto

    for i in range(12, 30):
        messages.extend(_step(i))
    view = await compactor.maybe_compact(messages, _measure)

    assert compactor.compactions == 2
    assert len(prompts) == 2
    assert "<previous_summary>\nS1 first part\n</previous_summary>" in prompts[1][1]["content"]
    assert compactor.upto > first_upto
    assert "S2 merged" in view[1]["content"]
    assert "S1 first part" not in view[1]["content"]
    assert view[1]["content"].count(_TASK) == 1
    assert view[2:] == messages[compactor.upto:]
    lines = compactor.transcript_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == compactor.upto - 1


@pytest.mark.asyncio
async def test_compactor_falls_back_to_mechanical_digest_when_llm_fails():
    messages = _long_run(12)
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=RuntimeError("provider down"))
    compactor = ContextCompactor(provider, "m", budget=int(_measure(messages) * 0.9))

    view = await compactor.maybe_compact(messages, _measure)

    notice = view[1]["content"]
    assert compactor.compactions == 1
    assert "mechanical digest" in notice
    assert 'read_file({"path": "src/m0.py"}) -> result 0' in notice
    assert _TASK in notice
    assert compactor.transcript_path is None


def test_compactor_tail_never_starts_with_tool_results():
    messages = _long_run(3)
    calls = [
        {"id": f"p{k}", "type": "function", "function": {"name": "exec", "arguments": "{}"}}
        for k in range(3)
    ]
    messages.append({"role": "assistant", "content": "", "tool_calls": calls})
    for k in range(3):
        messages.append({
            "role": "tool", "tool_call_id": f"p{k}", "name": "exec",
            "content": " ".join(f"out{k}_{n}" for n in range(200)),
        })
    group_start = len(messages) - 4
    compactor = ContextCompactor(MagicMock(), "m", budget=1000, keep_ratio=0.3)

    cut = compactor._pick_cut(messages, 1)

    assert cut == group_start
    assert messages[cut]["role"] == "assistant"


def test_context_compaction_config_defaults():
    defaults = AgentDefaults()
    assert defaults.context_compaction is True
    assert defaults.context_compaction_ratio == 0.8
    assert AgentDefaults.model_validate({"contextCompaction": False}).context_compaction is False


@pytest.mark.asyncio
async def test_runner_compacts_long_single_turn_within_budget(tmp_path):
    from nanobot.agent.runner import AgentRunner, AgentRunSpec

    provider = MagicMock(spec=LLMProvider)
    main_requests: list[list[dict]] = []
    summary_requests: list[list[dict]] = []
    steps = 8

    async def chat_with_retry(*, messages, **kwargs):
        if messages[0]["content"].startswith("You are compacting"):
            summary_requests.append(messages)
            return LLMResponse(content="<summary>probed several files</summary>", tool_calls=[], usage={})
        main_requests.append(messages)
        n = len(main_requests) - 1
        if n < steps:
            return LLMResponse(
                content=f"probing {n}",
                tool_calls=[ToolCallRequest(id=f"call_{n}", name="probe", arguments={"path": f"file_{n}.txt"})],
                usage={},
            )
        return LLMResponse(content="done", tool_calls=[], usage={})

    provider.chat_with_retry = chat_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(side_effect=lambda *a, **k: " ".join(f"w{n}" for n in range(300)))
    spec = AgentRunSpec(
        initial_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "do the long task"},
        ],
        tools=tools,
        model="test-model",
        max_iterations=steps + 2,
        max_tool_result_chars=AgentDefaults().max_tool_result_chars,
        max_tokens=500,
        context_window_tokens=3000,
        workspace=tmp_path,
        session_key="cli:direct",
        context_compaction=True,
    )
    budget = 3000 - 500 - 1024

    result = await AgentRunner(provider).run(spec)

    assert result.final_content == "done"
    assert summary_requests
    compacted = [
        request for request in main_requests
        if any(str(m.get("content", "")).startswith("[Context compacted]") for m in request)
    ]
    assert compacted
    notice = next(m for m in compacted[-1] if str(m.get("content", "")).startswith("[Context compacted]"))
    assert "probed several files" in notice["content"]
    assert "do the long task" in notice["content"]
    assert "file_0.txt" in notice["content"]
    for request in main_requests:
        declared: set[str] = set()
        for message in request:
            for call in message.get("tool_calls") or []:
                declared.add(call["id"])
            if message.get("role") == "tool":
                assert message["tool_call_id"] in declared
    last_estimate, _ = estimate_prompt_tokens_chain(provider, "test-model", main_requests[-1], [])
    assert last_estimate <= budget

    # The persisted run keeps every message and none of the synthetic notice.
    assert sum(1 for m in result.messages if m.get("role") == "tool") == steps
    assert not any(str(m.get("content", "")).startswith("[Context compacted]") for m in result.messages)
    transcripts = list((tmp_path / ".nanobot" / "transcripts" / "cli_direct").glob("*.jsonl"))
    assert len(transcripts) == 1


@pytest.mark.asyncio
async def test_runner_without_compaction_keeps_single_turn_oversized():
    """Documents the gap compaction fills: snipping cannot cut a single long turn."""
    from nanobot.agent.runner import AgentRunner, AgentRunSpec

    provider = MagicMock(spec=LLMProvider)
    main_requests: list[list[dict]] = []

    async def chat_with_retry(*, messages, **kwargs):
        main_requests.append(messages)
        n = len(main_requests) - 1
        if n < 8:
            return LLMResponse(
                content="", tool_calls=[ToolCallRequest(id=f"call_{n}", name="probe", arguments={})], usage={},
            )
        return LLMResponse(content="done", tool_calls=[], usage={})

    provider.chat_with_retry = chat_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(side_effect=lambda *a, **k: " ".join(f"w{n}" for n in range(300)))
    await AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{"role": "system", "content": "system"}, {"role": "user", "content": "task"}],
        tools=tools,
        model="test-model",
        max_iterations=10,
        max_tool_result_chars=AgentDefaults().max_tool_result_chars,
        max_tokens=500,
        context_window_tokens=3000,
    ))

    last_estimate, _ = estimate_prompt_tokens_chain(provider, "test-model", main_requests[-1], [])
    assert last_estimate > 3000 - 500 - 1024
