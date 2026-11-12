"""供应商、runner 与 ReAct 的离线集成，不连接真实服务。"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.trajectory import core, http
from nanobot.trajectory.processing import load_rows, process, read_events
from nanobot.trajectory.provider import RecordingProvider


@pytest.fixture
def recorder(tmp_path):
    old = core._RECORDER
    value = core.configure("scenclaw", tmp_path / "capture")
    yield value
    value.flush()
    core._RECORDER = old


def rows(recorder):
    recorder.flush()
    return read_events([recorder.directory])[0]


class Provider(LLMProvider):
    supports_progress_deltas = True

    def __init__(self, responses):
        super().__init__("unit-test-key", "https://test.invalid")
        self.responses = iter(responses)

    def get_default_model(self):
        return "mock"

    async def chat(self, messages, **kwargs):
        return next(self.responses)

    async def chat_stream(self, messages, **kwargs):
        for text in ("first", "second"):
            if kwargs.get("on_content_delta"):
                await kwargs["on_content_delta"](text)
        return next(self.responses)


@pytest.mark.asyncio
async def test_provider_stream_and_capture_failure_keep_callbacks(recorder, monkeypatch):
    provider = RecordingProvider(Provider([LLMResponse(content="firstsecond")]), "mock")
    def fail(**kwargs):
        raise ValueError("collector failed")
    monkeypatch.setattr(core.Call, "delta", lambda self, **kwargs: fail(**kwargs))
    callback = AsyncMock()
    assert provider.supports_progress_deltas
    result = await provider.chat_stream(messages=[{"role": "user", "content": "task"}], on_content_delta=callback)
    assert result.content == "firstsecond"
    assert [c.args[0] for c in callback.await_args_list] == ["first", "second"]
    assert len([r for r in rows(recorder) if r["event"] == "llm_start"]) == 1
    assert recorder.errors > 0


@pytest.mark.asyncio
async def test_fallback_attempts_share_context(recorder):
    from nanobot.providers.fallback_provider import FallbackProvider
    error = LLMResponse(content="server error", finish_reason="error", error_kind="server_error", error_status_code=500)
    primary = RecordingProvider(Provider([error]), "primary")
    backup = RecordingProvider(Provider([LLMResponse(content="ok")]), "backup")
    preset = SimpleNamespace(model="backup", max_tokens=100, temperature=0.2, reasoning_effort=None)
    provider = FallbackProvider(primary, [preset], lambda _: backup)
    result = await provider.chat_with_retry(messages=[{"role": "user", "content": "task"}])
    assert result.content == "ok"
    starts = [r for r in rows(recorder) if r["event"] == "llm_start"]
    assert len(starts) == 2
    assert len({r["trace_id"] for r in starts}) == 1
    assert len({r["call_id"] for r in starts}) == 2
    assert len({r["attempt_id"] for r in starts}) == 2


@pytest.mark.asyncio
async def test_runner_records_raw_and_visible_tool_results(recorder, tmp_path):
    from nanobot.agent.runner import AgentRunner, AgentRunSpec
    provider = RecordingProvider(Provider([
        LLMResponse(content="", tool_calls=[ToolCallRequest(id="real-tool", name="lookup", arguments={})]),
        LLMResponse(content="done"),
    ]), "mock")
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="result " * 2000)
    tools.get.return_value = None
    runner = AgentRunner(provider)
    result = await runner.run( AgentRunSpec( initial_messages=[ {
        "role": "user",
        "content": "task"}],
        tools=tools,
        model="mock",
        max_iterations=3, max_tool_result_chars=200))
    assert result.final_content == "done"
    captured = rows(recorder)
    raw = next(r for r in captured if r["event"] == "tool_result")
    visible = next(r for r in captured if r["event"] == "tool_visible")
    assert raw["data"]["result"] == "result " * 2000
    assert len(visible["data"]["message"]["content"]) < len(raw["data"]["result"])
    assert raw["tool_call_id"] == visible["tool_call_id"] == "real-tool"
    starts = [r for r in captured if r["event"] == "llm_start"]
    assert raw["parent_call_id"] == starts[0]["call_id"]
    assert starts[1]["data"]["request"]["messages"][-1]["content"] == visible["data"]["message"]["content"]
    process([recorder.directory], tmp_path / "work")
    assert any(r["capture_relation"] == "paired" for r in load_rows(tmp_path / "work/tool-links.jsonl"))


@pytest.mark.asyncio
@pytest.mark.parametrize("local", [False, True])
async def test_enabled_http_client_preserves_timeout_pool_and_reasoning(recorder, monkeypatch, local):
    import nanobot.providers.openai_compat_provider as module
    from nanobot.providers.registry import ProviderSpec
    from openai import AsyncOpenAI
    requested = []
    def answer(request):
        requested.append(request)
        return httpx.Response( 200, json={
            "id": "c",
            "object": "chat.completion",
            "model": "mock",
            "choices": [
            {
            "index": 0,
            "message": {
            "role": "assistant",
            "content": "ok",
            "reasoning_content": "native thinking"},
            "finish_reason": "stop"}]})
    original = http.AsyncClient
    created = []
    def client(**kwargs):
        created.append(kwargs)
        return original(transport=httpx.MockTransport(answer), **kwargs)
    monkeypatch.setattr(http, "AsyncClient", client)
    monkeypatch.setattr(module, "AsyncOpenAI", AsyncOpenAI)
    spec = ProviderSpec(name="deepseek", keywords=(), env_key="", is_local=local)
    concrete = module.OpenAICompatProvider(api_key="unit-test-key", api_base="https://test.invalid/v1", spec=spec)
    provider = RecordingProvider(concrete, "mock")
    assert (await provider.chat(messages=[{"role": "user", "content": "task"}], model="deepseek-chat")).content == "ok"
    assert len(created) == 1
    assert created[0]["timeout"] == 120.0
    if local:
        assert created[0]["limits"].keepalive_expiry == 0
    else:
        assert created[0]["limits"].max_connections == 1000
        assert created[0]["limits"].max_keepalive_connections == 100
        assert created[0]["follow_redirects"] is True
    wire = next(r for r in rows(recorder) if r["event"] == "http_end")
    assert wire["data"]["response"]["reasoning_content"] == "native thinking"
    await concrete._client.close()


def test_react_keeps_fix_and_validation_segments(recorder):
    from nanobot.trajectory.react import action_scope, record_tools
    tools = record_tools({"fix": lambda code, issue: "fixed", "validate": lambda code: {"summary": {"errors": 0}}})
    class Engine:
        _current_code = "broken"
        @action_scope
        def action(self, action, action_input, step):
            self._current_code = tools["fix"](self._current_code, "error")
            tools["validate"](self._current_code)
            return "validated"
    with core.scope(trace_id="react"):
        assert Engine().action("fix", {}, 1) == "validated"
    captured = rows(recorder)
    before = next(r for r in captured if r["event"] == "react_action")
    after = next(r for r in captured if r["event"] == "react_observation")
    assert before["data"]["code_before"] == "broken"
    assert after["data"]["code_after"] == "fixed"
    assert next( r for r in captured if r[
        "event"] == "validation_observation")[
        "data"][
        "validation_scope"] == "scenario_static_rules"
    assert not any(r["event"] == "verification" for r in captured)
