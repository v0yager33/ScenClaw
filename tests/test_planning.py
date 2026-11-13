"""规划联动离线回归：固定远端、版本边界、WS 交接及同会话续修。"""
from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest

from nanobot.bus.queue import MessageBus
from nanobot.channels.websocket import WebSocketChannel, WebSocketConfig
from nanobot.config.schema import Config, PlanningConfig
from nanobot.security.network import configure_ssrf_whitelist
from nanobot.security.workspace_access import current_workspace_scope, default_workspace_scope
from nanobot.session.manager import SessionManager
from nanobot.webui.gateway_services import build_gateway_services
from nanobot.webui.planning_client import PlanningClient, PlanningClientError
from nanobot.webui.planning_store import PlanningStore, digest
from nanobot.webui.planning_ws import PlanningGateway
from nanobot.webui.transcript import read_transcript_lines

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


MARKDOWN = """# COA方案：离线
## COA矩阵
| 单元 | 阶段1: 出动 | 阶段2: 返回 |
| -- | -- | -- |
| 单元一 (U1) | 1. 起飞<br>2. 侦察 | 1. 返航 |
## 实体清单
| unit_id | role |
| -- | -- |
| U1 | 执行 |
## 地理参考
保留原点与边界。
"""
TASKS = [
    {"id": f"{phase}:U1:{index}", "phase_id": phase, "phase_name": name,
     "unit_id": "U1", "unit_name": "单元一", "action_index": index, "text": text}
    for phase, name, index, text in [
        ("Phase_1", "出动", 0, "起飞"), ("Phase_1", "出动", 1, "侦察"),
        ("Phase_2", "返回", 0, "返航"),
    ]
]


def prepared(markdown=MARKDOWN, selected=None):
    snapshot = markdown if selected is None else "仅执行所选行动\n" + markdown.replace("1. 起飞<br>", "")
    return {"markdown": markdown, "content_hash": digest(markdown), "coa": {},
            "tasks": copy.deepcopy(TASKS), "validation": {"passed": False, "issues": ["规模较小"]},
            "snapshot_markdown": snapshot, "snapshot_hash": digest(snapshot),
            "selected_task_ids": [task["id"] for task in TASKS] if selected is None else selected,
            "selection_note": "未选依赖请在原会话确认"}


def job(index, status="completed"):
    return {"job_id": index.get("job_id") or str(uuid4()), "request_id": index["request_id"],
            "mission": index["mission"], "input_hash": index["input_hash"], "status": status,
            "logs": [], "result": {"markdown": MARKDOWN, "prepared": prepared(),
                                     "generation": {"final_score": 91}}}


def completed(store, root):
    index = store.create(root, "离线任务", str(uuid4()))
    return store.update_job(root, index["plan_id"], job(index))


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr("nanobot.config.paths.get_data_dir", lambda: tmp_path / "data")
    monkeypatch.setattr("nanobot.webui.workspaces.get_webui_dir", lambda: tmp_path / "webui")
    configure_ssrf_whitelist([])
    yield
    configure_ssrf_whitelist([])


def test_config_and_production_injection(tmp_path, monkeypatch):
    from nanobot.channels.manager import ChannelManager
    config = Config()
    assert not config.planning.enabled
    config.planning = PlanningConfig(enabled=True, apiToken="test-only")
    config.agents.defaults.workspace = str(tmp_path)
    config.channels.websocket = {"enabled": True, "allowFrom": ["*"]}
    monkeypatch.setattr("nanobot.channels.registry.discover_channel_names", lambda: ["websocket"])
    monkeypatch.setattr("nanobot.channels.registry.discover_enabled", lambda *a, **kw: {"websocket": WebSocketChannel})
    manager = ChannelManager(config, MessageBus())
    assert manager.channels["websocket"].gateway.planning.config is config.planning
    assert "test-only" not in repr(config.planning)


async def test_http_client_security_and_auth():
    config = PlanningConfig(enabled=True, api_token="test-only")
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=prepared())

    client = PlanningClient(config, transport=httpx.MockTransport(handler))
    with pytest.raises(PlanningClientError, match="SSRF"):
        await client.prepare(MARKDOWN)
    assert not calls
    configure_ssrf_whitelist(["127.0.0.1/32"])
    result = await client.prepare(MARKDOWN)
    assert result["markdown"] == MARKDOWN
    assert calls[0].headers["Authorization"] == "Bearer test-only"
    assert calls[0].url.query == b""
    assert json.loads(calls[0].content)["markdown"] == MARKDOWN
    for address in ["file:///etc/passwd", "http://user:secret@localhost", "http://localhost?secret=1"]:
        with pytest.raises(PlanningClientError, match="HTTP"):
            await PlanningClient(PlanningConfig(enabled=True, api_token="x", api_base=address)).prepare(MARKDOWN)


@pytest.mark.parametrize("response,code", [
    (httpx.Response(302, headers={"Location": "http://other.invalid"}), "redirect_rejected"),
    (httpx.Response(401, json={"error": {"code": "unauthorized", "message": "secret/internal/path"}}), "unauthorized"),
    (httpx.Response(200, content="invalid"), "invalid_response"),
])
async def test_http_failures_are_specific_and_redacted(response, code):
    configure_ssrf_whitelist(["127.0.0.1/32"])
    client = PlanningClient(PlanningConfig(enabled=True, api_token="x"), transport=httpx.MockTransport(lambda _: response))
    with pytest.raises(PlanningClientError) as caught:
        await client.prepare(MARKDOWN)
    assert caught.value.code == code
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize("exception,code", [(httpx.ReadTimeout, "service_timeout"), (httpx.ConnectError, "service_unavailable")])
async def test_transport_errors(exception, code):
    configure_ssrf_whitelist(["127.0.0.1/32"])

    def fail(request):
        raise exception("secret", request=request)

    with pytest.raises(PlanningClientError) as caught:
        await PlanningClient(PlanningConfig(enabled=True, api_token="x"), transport=httpx.MockTransport(fail)).prepare(MARKDOWN)
    assert caught.value.code == code


def test_versions_conflict_boundaries_and_immutability(tmp_path):
    store = PlanningStore()
    original = completed(store, tmp_path)
    plan_id = original["plan"]["plan_id"]
    changed = MARKDOWN + "\n人工修改\n"
    saved = store.save(tmp_path, plan_id, 1, changed, prepared(changed))
    assert saved["document"]["revision"] == 2
    assert "generation" not in saved["document"]
    assert store.get(tmp_path, plan_id, 1)["document"] == original["document"]
    with pytest.raises(PlanningClientError, match="草稿"):
        store.save(tmp_path, plan_id, 1, changed, prepared(changed))
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(PlanningClientError):
        store.get(other, plan_id)
    with pytest.raises(PlanningClientError):
        store.get(tmp_path, "../other")
    with pytest.raises(PlanningClientError):
        store.path(tmp_path, "../outside")
    (other / ".nanobot").symlink_to(tmp_path / ".nanobot", target_is_directory=True)
    with pytest.raises(PlanningClientError, match="路径"):
        store.get(other, plan_id)


@pytest.mark.parametrize("fault", ["duplicate", "wrong_id", "snapshot"])
def test_invalid_prepared_is_rejected(tmp_path, fault):
    store = PlanningStore()
    bad = prepared()
    if fault == "duplicate":
        bad["tasks"].append(copy.deepcopy(bad["tasks"][0]))
    elif fault == "wrong_id":
        bad["tasks"][0]["id"] = "client-supplied-id"
    else:
        bad["snapshot_hash"] = "invalid"
    with pytest.raises(PlanningClientError):
        store._check_prepared(MARKDOWN, bad)


def test_crash_between_revision_and_index_recovers(tmp_path, monkeypatch):
    store = PlanningStore()
    original = completed(store, tmp_path)
    plan_id = original["plan"]["plan_id"]
    write = store._write

    def fail(root, relative, data, **kw):
        if relative.endswith("plan.json"):
            raise OSError("模拟进程中断")
        return write(root, relative, data, **kw)

    monkeypatch.setattr(store, "_write", fail)
    changed = MARKDOWN + "\n已提交版本\n"
    with pytest.raises(OSError):
        store.save(tmp_path, plan_id, 1, changed, prepared(changed))
    recovered = PlanningStore().get(tmp_path, plan_id)
    assert recovered["document"]["markdown"] == changed
    assert recovered["plan"]["revision"] == 2


def test_job_binding_and_stale_poll_cannot_undo_cancel(tmp_path):
    store = PlanningStore()
    index = store.create(tmp_path, "任务", str(uuid4()))
    running = job(index, "running")
    store.update_job(tmp_path, index["plan_id"], running)
    cancelled = {**running, "status": "cancelled"}
    store.update_job(tmp_path, index["plan_id"], cancelled)
    assert store.update_job(tmp_path, index["plan_id"], running)["plan"]["status"] == "cancelled"
    with pytest.raises(PlanningClientError, match="来源"):
        store.update_job(tmp_path, index["plan_id"], {**running, "job_id": str(uuid4())})


@pytest.fixture
def channel(tmp_path):
    bus = MessageBus()
    sessions = SessionManager(tmp_path)
    config = WebSocketConfig(allow_from=["*"], websocket_requires_token=False)
    gateway = build_gateway_services(config=config, bus=bus, session_manager=sessions, static_dist_path=None,
                                     workspace_path=tmp_path, default_restrict_to_workspace=True,
                                     runtime_model_name=None, runtime_surface="browser", runtime_capabilities_overrides=None)
    client = SimpleNamespace(prepare=AsyncMock(side_effect=prepared), request=AsyncMock())
    planning = PlanningGateway(PlanningConfig(enabled=True), client=client)
    ch = WebSocketChannel(config, bus, gateway=replace(gateway, planning=planning))
    connection = MagicMock()
    connection.send = AsyncMock()
    return ch, connection, planning, default_workspace_scope(tmp_path, restrict_to_workspace=True)


def handoff_request(detail, **overrides):
    return {"action": "handoff", "plan_id": detail["plan"]["plan_id"], "revision": 1,
            "content_hash": detail["document"]["content_hash"], "handoff_id": str(uuid4()),
            "selected_task_ids": [TASKS[1]["id"]], **overrides}


async def test_handoff_preview_idempotency_source_and_second_message(channel):
    ch, connection, planning, scope = channel
    detail = completed(planning.store, scope.project_path)
    request = handoff_request(detail)
    preview = await planning.handle(ch, connection, "u1", scope, {**request, "preview": True})
    assert "仅执行" in preview["snapshot_markdown"]
    assert not (scope.project_path / "scenarios").exists()
    source = await planning.handle(ch, connection, "u1", scope, request)
    assert source["status"] == "dispatched"
    first = await asyncio.wait_for(ch.bus.consume_inbound(), 1)
    assert first.chat_id == source["chat_id"]
    assert first.metadata["workspace_scope"]["project_path"] == str(scope.project_path)
    assert source["coa_path"] in first.content
    session = ch.gateway.session_manager.get_or_create(first.session_key)
    assert session.metadata["planning_source"]["revision"] == 1
    assert "api_token" not in json.dumps(session.metadata)
    assert (scope.project_path / source["coa_path"]).read_text(encoding="utf-8") == preview["snapshot_markdown"]
    planning.client.prepare.side_effect = PlanningClientError("service_unavailable", "离线")
    assert await planning.handle(ch, connection, "u1", scope, request) == source
    assert ch.bus.inbound_size == 0
    await ch._dispatch_envelope(connection, "u1", {"type": "message", "chat_id": first.chat_id,
                               "content": "修改速度并保留错误反馈", "webui": True, "turn_id": "second"})
    second = await ch.bus.consume_inbound()
    assert second.session_key == first.session_key
    assert second.metadata["workspace_scope"] == first.metadata["workspace_scope"]
    transcript = read_transcript_lines(first.session_key)
    assert any(source["coa_path"] in row.get("text", "") for row in transcript)
    assert any("修改速度" in row.get("text", "") for row in transcript)
    assert not any(json.loads(args.args[0])["event"] == "attached" for args in connection.send.await_args_list)


async def test_interrupted_handoff_never_replays(channel, monkeypatch):
    ch, connection, planning, scope = channel
    detail = completed(planning.store, scope.project_path)
    request = handoff_request(detail)
    monkeypatch.setattr(ch, "_dispatch_envelope", AsyncMock(side_effect=RuntimeError("模拟中断")))
    first = await planning.handle(ch, connection, "u1", scope, request)
    assert first["status"] == "interrupted" and first["chat_id"]
    source = await planning.handle(ch, connection, "u1", scope, request)
    assert source == first
    assert source["status"] == "interrupted"
    assert ch._dispatch_envelope.await_count == 1
    with pytest.raises(PlanningClientError, match="绑定"):
        await planning.handle(ch, connection, "u1", scope, {**request, "selected_task_ids": None})


async def test_ws_requests_are_bounded_correlated_and_chat_is_not_blocked(channel):
    ch, connection, planning, scope = channel
    gate = asyncio.Event()
    planning.client.request.side_effect = lambda *a, **kw: None

    async def blocked(*a, **kw):
        await gate.wait()
        raise PlanningClientError("not_found", "不存在", 404)

    planning.client.request.side_effect = blocked
    for index in range(9):
        await ch._dispatch_envelope(connection, "u1", {"type": "planning", "action": "status", "request_id": str(index)})
    await ch._dispatch_envelope(connection, "u1", {"type": "message", "chat_id": str(uuid4()), "content": "普通聊天"})
    assert (await asyncio.wait_for(ch.bus.consume_inbound(), 1)).content == "普通聊天"
    gate.set()
    await asyncio.gather(*list(planning.tasks))
    events = [json.loads(args.args[0]) for args in connection.send.await_args_list]
    responses = [event for event in events if event["event"] == "planning_response"]
    assert {row["request_id"] for row in responses} == {str(i) for i in range(9)}
    assert sum(not row["ok"] for row in responses) == 1
    assert responses[0]["error"]["code"] == "busy"
    await planning.close()


async def test_mock_provider_and_scenario_tool_keep_second_turn_history(channel, monkeypatch):
    from nanobot.agent.loop import AgentLoop
    from nanobot.agent.tools.base import Tool
    from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

    ch, connection, planning, scope = channel
    detail = completed(planning.store, scope.project_path)
    source = await planning.handle(ch, connection, "u1", scope, handoff_request(detail, selected_task_ids=None))
    first = await ch.bus.consume_inbound()
    provider = MagicMock(spec=LLMProvider)
    provider.get_default_model.return_value = "offline-test"
    provider.generation = SimpleNamespace(max_tokens=4096, temperature=0.1, reasoning_effort=None)
    path = f'{source["output_dir"]}/scenario.txt'
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(content=None, tool_calls=[ToolCallRequest(id="t1", name="mock_scenario", arguments={"path": path})]),
        LLMResponse(content="生成失败：请反馈修改意见"),
        LLMResponse(content=None, tool_calls=[ToolCallRequest(id="t2", name="mock_scenario", arguments={"path": path})]),
        LLMResponse(content="已依据反馈更新同一产物"),
    ])
    monkeypatch.setattr(AgentLoop, "_register_default_tools", lambda self: None)
    loop = AgentLoop(bus=ch.bus, provider=provider, workspace=scope.project_path,
                     session_manager=ch.gateway.session_manager, context_compaction=False)
    seen = []

    async def mock_tool(name, arguments, **kw):
        active = current_workspace_scope()
        seen.append((active.project_path, arguments["path"]))
        if len(seen) == 1:
            return "Error: 离线模拟语法错误，未运行真实 lint"
        artifact = active.project_path / arguments["path"]
        artifact.write_text("离线修复产物", encoding="utf-8")
        return "已修复"

    class MockScenarioTool(Tool):
        name = "mock_scenario"
        description = "离线模拟想定工具，不执行真实 lint 或仿真"
        parameters = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

        async def execute(self, **kwargs):
            return await mock_tool(self.name, kwargs)

    loop.tools.register(MockScenarioTool())
    first.metadata.pop("_wants_stream", None)
    await loop._process_message(first)
    await ch._dispatch_envelope(connection, "u1", {"type": "message", "chat_id": first.chat_id,
                               "content": "请依据上轮错误修改", "webui": True, "turn_id": "second"})
    second = await ch.bus.consume_inbound()
    second.metadata.pop("_wants_stream", None)
    await loop._process_message(second)
    assert seen == [(scope.project_path, path), (scope.project_path, path)]
    assert (scope.project_path / path).read_text(encoding="utf-8") == "离线修复产物"
    history = loop.sessions.get_or_create(first.session_key).messages
    text = json.dumps(history, ensure_ascii=False)
    assert "离线模拟语法错误" in text and "请依据上轮错误修改" in text
    assert loop.sessions.get_or_create(first.session_key).metadata["planning_source"]["coa_path"] == source["coa_path"]
    assert "离线模拟语法错误" in json.dumps(provider.chat_with_retry.await_args_list[2].kwargs["messages"], ensure_ascii=False)
