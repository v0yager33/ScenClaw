"""规划 RPC 位于通道边缘，交接通过现有普通 message 链路。"""
from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from nanobot.config.schema import PlanningConfig
from nanobot.security.workspace_access import WorkspaceScopeError
from nanobot.webui.planning_client import PlanningClient, PlanningClientError
from nanobot.webui.planning_store import (
    PlanningStore,
    digest,
    identifier,
    revision_number,
    timestamp,
)


class PlanningGateway:
    def __init__(self, config: PlanningConfig, *, client=None, store=None):
        self.config = config
        self.client = client or PlanningClient(config)
        self.store = store or PlanningStore(config.max_payload_bytes)
        self.tasks: set[asyncio.Task] = set()
        self.mutations = asyncio.Lock()

    async def submit(self, channel, connection, client_id: str, envelope: dict):
        request_id = envelope.get("request_id")
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            await channel._send_event(connection, "planning_response", request_id=request_id,
                                      ok=False, error={"code": "invalid_request", "message": "request_id 无效"})
            return
        if len(self.tasks) >= 8:
            await channel._send_event(connection, "planning_response", request_id=request_id,
                                      ok=False, error={"code": "busy", "message": "规划请求繁忙，请稍后重试"})
            return

        async def run():
            try:
                if not channel.is_allowed(client_id):
                    raise PlanningClientError("forbidden", "当前用户没有访问权限", 403)
                if len(json.dumps(envelope, ensure_ascii=False).encode()) > self.config.max_payload_bytes + 256 * 1024:
                    raise PlanningClientError("payload_too_large", "规划请求超过内容上限", 413)
                scope = channel._workspaces.scope_for_new_chat(
                    envelope, controls_available=channel._workspace_controls_available(connection))
                data = await self.handle(channel, connection, client_id, scope, envelope)
                await channel._send_event(connection, "planning_response", request_id=request_id, ok=True, data=data)
            except (PlanningClientError, WorkspaceScopeError) as exc:
                await channel._send_event(connection, "planning_response", request_id=request_id, ok=False,
                                          error={"code": getattr(exc, "code", "workspace_rejected"),
                                                 "message": str(exc), "status": exc.status})
            except asyncio.CancelledError:
                raise
            except Exception:
                await channel._send_event(connection, "planning_response", request_id=request_id, ok=False,
                                          error={"code": "internal_error", "message": "规划操作失败，请检查服务日志"})
        task = asyncio.create_task(run())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def handle(self, channel, connection, client_id, scope, request: dict) -> dict:
        action, root = request.get("action"), scope.project_path
        if action == "status":
            try:
                # /health 不需要鉴权，使用 prepare 之外的作业接口验证 Bearer 配置。
                await self.client.request("GET", "/v1/plans/00000000-0000-0000-0000-000000000000")
            except PlanningClientError as exc:
                if exc.status != 404:
                    return {"enabled": self.config.enabled, "available": False,
                            "poll_interval_s": self.config.poll_interval_s,
                            "error": {"code": exc.code, "message": str(exc)}}
            return {"enabled": True, "available": True, "poll_interval_s": self.config.poll_interval_s}
        if action == "list":
            plans = await asyncio.to_thread(self.store.list, root)
            return {"plans": [{key: row[key] for key in ("plan_id", "revision", "status", "created_at")}
                              | {"mission": row["mission"][:200]} for row in plans]}
        if action == "get":
            plan_id = identifier(request.get("plan_id"))
            current = await asyncio.to_thread(self.store.get, root, plan_id, request.get("revision"))
            index = current["plan"]
            if index["job_id"] and index["status"] in {"queued", "running"}:
                try:
                    job = await self.client.request("GET", f'/v1/plans/{identifier(index["job_id"])}')
                    async with self.mutations:
                        await asyncio.to_thread(self.store.update_job, root, plan_id, job)
                    current = await asyncio.to_thread(self.store.get, root, plan_id, request.get("revision"))
                except PlanningClientError as exc:
                    current["service_error"] = {"code": exc.code, "message": str(exc)}
            return current
        if action == "create":
            async with self.mutations:
                index = await asyncio.to_thread(self.store.create, root, request.get("mission"),
                                                request.get("generation_request_id"))
                if not index["job_id"]:
                    job = await self.client.request("POST", "/v1/plans", {
                        "request_id": index["request_id"], "mission": index["mission"],
                    })
                    return await asyncio.to_thread(self.store.update_job, root, index["plan_id"], job)
                return await asyncio.to_thread(self.store.get, root, index["plan_id"])
        if action == "save":
            async with self.mutations:
                plan_id = identifier(request.get("plan_id"))
                await asyncio.to_thread(self.store.get, root, plan_id)
                markdown = request.get("markdown")
                if not isinstance(markdown, str):
                    raise PlanningClientError("invalid_request", "完整 Markdown 必须为文本")
                prepared = await self.client.prepare(markdown)
                return await asyncio.to_thread(self.store.save, root, plan_id,
                                                request.get("base_revision"), markdown, prepared)
        if action == "cancel":
            async with self.mutations:
                plan_id = identifier(request.get("plan_id"))
                index = (await asyncio.to_thread(self.store.get, root, plan_id))["plan"]
                if index["job_id"]:
                    job = await self.client.request("POST", f'/v1/plans/{identifier(index["job_id"])}/cancel', {})
                    return await asyncio.to_thread(self.store.update_job, root, plan_id, job)
                return await asyncio.to_thread(self.store.get, root, plan_id)
        if action == "handoff":
            async with self.mutations:
                return await self.handoff(channel, connection, client_id, scope, request)
        raise PlanningClientError("invalid_action", "不支持的规划操作")

    async def handoff(self, channel, connection, client_id, scope, request):
        root = scope.project_path
        plan_id = identifier(request.get("plan_id"))
        revision = revision_number(request.get("revision"))
        current = await asyncio.to_thread(self.store.get, root, plan_id, revision)
        document = current["document"]
        if request.get("content_hash") != document["content_hash"]:
            raise PlanningClientError("version_conflict", "交接版本内容已变化，请重新选择", 409)
        selection = request.get("selected_task_ids")
        if selection is not None and (not isinstance(selection, list) or any(not isinstance(item, str) for item in selection)):
            raise PlanningClientError("invalid_selection", "选择必须为行动 ID 数组")
        ids = [task["id"] for task in document["tasks"]]
        if selection is not None and (not selection or len(set(selection)) != len(selection) or not set(selection).issubset(ids)):
            raise PlanningClientError("invalid_selection", "请选择有效且不重复的行动")
        selected = None if selection is None else [item for item in ids if item in selection]
        binding = {"plan_id": plan_id, "revision": revision, "content_hash": document["content_hash"],
                   "selected_task_ids": selected}
        handoff_id = None if request.get("preview") is True else identifier(request.get("handoff_id"))
        if handoff_id:
            existing = await asyncio.to_thread(self.store.handoff_existing, root, handoff_id)
            if existing:
                if any(existing.get(key) != value for key, value in binding.items()):
                    raise PlanningClientError("idempotency_conflict", "交接 ID 已绑定其他选择", 409)
                snapshot_path = self.store.path(root, existing["coa_path"])
                if not snapshot_path.is_file() or digest(await asyncio.to_thread(snapshot_path.read_text, encoding="utf-8")) != existing["snapshot_hash"]:
                    raise PlanningClientError("content_mismatch", "交接快照已变化，无法复用", 409)
                self.ensure_session(channel, existing, scope)
                if existing["status"] != "dispatched":
                    existing = await asyncio.to_thread(self.store.mark_handoff, root, existing, "interrupted")
                return existing
        prepared = await self.client.prepare(document["markdown"], selected)
        self.store._check_prepared(document["markdown"], prepared)
        snapshot = prepared["snapshot_markdown"]
        expected = ids if selected is None else selected
        if prepared.get("selected_task_ids") != expected or prepared["tasks"] != document["tasks"]:
            raise PlanningClientError("content_mismatch", "远端选择与当前版本行动索引不一致", 502)
        if digest(snapshot) != prepared["snapshot_hash"]:
            raise PlanningClientError("content_mismatch", "选择快照校验失败", 502)
        if not handoff_id:
            return {"snapshot_markdown": snapshot, "snapshot_hash": prepared["snapshot_hash"],
                    "selection_note": prepared["selection_note"], "selected_task_ids": prepared["selected_task_ids"]}
        sessions = channel.gateway.session_manager
        if sessions is None:
            raise PlanningClientError("session_unavailable", "会话存储不可用，无法交接", 503)
        chat_id = str(uuid4())
        folder = f"scenarios/{handoff_id}"
        source = {**binding, "handoff_id": handoff_id, "chat_id": chat_id,
                  "snapshot_hash": prepared["snapshot_hash"], "coa_path": f"{folder}/coa.md",
                  "output_dir": folder, "created_at": timestamp(), "status": "interrupted"}
        await asyncio.to_thread(self.store.create_handoff, root, source, snapshot)
        self.ensure_session(channel, source, scope)
        content = (
            f"请依据工作区相对路径 {folder}/coa.md 中的固定 COA 快照生成 AFSIM 想定。\n"
            "这是结构化 COA；使用 afsim-scenario 的 COA 解析／模板／坐标／生成工具链，"
            "严格遵守快照执行范围，不恢复未选行动。缺少必要参数请在本会话询问。\n"
            f"将想定与当前结构数据保存到 {folder}/ 内。后续修改先读取该目录的当前产物，"
            "在同一目录更新并保留错误／反馈历史，不覆盖 coa.md 或 source.json，"
            "不读取规划区最新稿，不另开会话。结构检查不代表已通过真实仿真。"
        )
        source = await asyncio.to_thread(self.store.mark_handoff, root, source, "dispatching")
        try:
            published = await channel._dispatch_envelope(connection, client_id, {
                "type": "message", "chat_id": chat_id, "content": content,
                "webui": True, "turn_id": handoff_id, "workspace_scope": scope.metadata(),
            })
            if published is not True:
                raise PlanningClientError("dispatch_interrupted", "交接发布未确认，请打开已有会话检查", 503)
            source = await asyncio.to_thread(self.store.mark_handoff, root, source, "dispatched")
        except asyncio.CancelledError:
            await asyncio.to_thread(self.store.mark_handoff, root, source, "interrupted")
            raise
        except Exception:
            # 已创建来源与会话，返回引用供用户检查；不自动重放可能已发布的首轮。
            source = await asyncio.to_thread(self.store.mark_handoff, root, source, "interrupted")
        await channel._send_event(connection, "session_updated", chat_id=chat_id, scope="messages")
        return source

    def ensure_session(self, channel, source, scope):
        """恢复派发前中断的空会话；从不重放首轮。"""
        sessions = channel.gateway.session_manager
        if sessions is None:
            raise PlanningClientError("session_unavailable", "会话存储不可用，无法交接", 503)
        chat_id = identifier(source["chat_id"])
        channel._workspaces.persist_scope(chat_id, scope)
        session = sessions.get_or_create(f"websocket:{chat_id}")
        session.metadata["planning_source"] = {key: source[key] for key in (
            "plan_id", "revision", "handoff_id", "content_hash", "snapshot_hash", "coa_path", "output_dir")}
        sessions.save(session)

    async def close(self):
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
