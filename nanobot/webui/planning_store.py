"""工作区内的规划版本和交接快照；路径只由服务生成。"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from uuid import UUID, uuid4

from nanobot.webui.planning_client import PlanningClientError


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def identifier(value) -> str:
    try:
        return str(UUID(str(value)))
    except ValueError as exc:
        raise PlanningClientError("invalid_id", "规划或交接 ID 无效") from exc


def revision_number(value) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise PlanningClientError("invalid_revision", "版本号必须为正整数")
    return value


class PlanningStore:
    def __init__(self, max_payload_bytes: int = 2 * 1024 * 1024):
        self.max_payload_bytes = max_payload_bytes
        self.lock = threading.RLock()

    def path(self, root: Path, relative: str) -> Path:
        """等价 allowed_dir 检查；即使 Full Access 也不能越出规划工作区。"""
        root = root.resolve(strict=True)
        path = root / relative
        if not path.resolve(strict=False).is_relative_to(root):
            raise PlanningClientError("path_rejected", "规划路径越出工作区", 403)
        cursor = path
        while cursor != root:
            if cursor.is_symlink():
                raise PlanningClientError("path_rejected", "规划路径不允许符号链接", 403)
            cursor = cursor.parent
        return path

    def _read(self, root: Path, relative: str) -> dict:
        path = self.path(root, relative)
        if not path.is_file():
            raise PlanningClientError("not_found", "当前工作区中不存在该规划或版本", 404)
        if path.stat().st_size > self.max_payload_bytes * 12:
            raise PlanningClientError("payload_too_large", "规划文件超过上限", 413)
        return json.loads(path.read_text(encoding="utf-8"))

    def _write(self, root: Path, relative: str, data: dict | str, *, immutable=False):
        path = self.path(root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path(root, relative)
        if immutable and path.exists():
            raise PlanningClientError("version_conflict", "不可变版本已经存在", 409)
        encoded = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, allow_nan=False)
        fd, name = tempfile.mkstemp(dir=path.parent, prefix=".planning-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            self.path(root, relative)
            os.replace(name, path)
            try:
                directory_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass  # Windows 不支持目录 fsync，文件仍使用原子替换。
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def list(self, root: Path) -> list[dict]:
        with self.lock:
            directory = self.path(root, ".nanobot/plans")
            if not directory.exists():
                return []
            plans = []
            for folder in directory.iterdir():
                if folder.name.startswith("."):
                    continue
                plan_id = identifier(folder.name)
                plans.append(self._read(root, f".nanobot/plans/{plan_id}/plan.json"))
            return sorted(plans, key=lambda plan: plan["created_at"], reverse=True)

    def get(self, root: Path, plan_id: str, revision: int | None = None) -> dict:
        with self.lock:
            plan_id = identifier(plan_id)
            index = self._read(root, f".nanobot/plans/{plan_id}/plan.json")
            # 不可变版本已落盘而索引写入中断时，恢复已提交的连续版本。
            recovered = False
            while index["revision"]:
                next_revision = index["revision"] + 1
                relative = f".nanobot/plans/{plan_id}/revisions/{next_revision}.json"
                if not self.path(root, relative).exists():
                    break
                orphan = self._read(root, relative)
                if (orphan.get("plan_id") != plan_id or orphan.get("revision") != next_revision
                        or orphan.get("base_revision") != index["revision"]
                        or digest(orphan["markdown"]) != orphan.get("content_hash")):
                    raise PlanningClientError("content_mismatch", "未完成版本恢复校验失败", 409)
                index.update(revision=next_revision, updated_at=timestamp())
                recovered = True
            if recovered:
                self._write(root, f".nanobot/plans/{plan_id}/plan.json", index)
            selected = index["revision"] if revision is None else revision_number(revision)
            document = None
            if selected:
                if selected > index["revision"]:
                    raise PlanningClientError("not_found", "规划版本不存在", 404)
                document = self._read(root, f".nanobot/plans/{plan_id}/revisions/{selected}.json")
                if (document.get("plan_id") != plan_id or document.get("revision") != selected
                        or digest(document["markdown"]) != document["content_hash"]):
                    raise PlanningClientError("content_mismatch", "规划版本内容校验失败", 409)
            return {"plan": index, "document": document}

    def create(self, root: Path, mission: str, request_id: str) -> dict:
        with self.lock:
            request_id = identifier(request_id)
            if not isinstance(mission, str) or not mission.strip() or len(mission) > 65536:
                raise PlanningClientError("invalid_mission", "请输入有效的任务描述")
            for index in self.list(root):
                if index["request_id"] == request_id:
                    if index["input_hash"] != digest(mission):
                        raise PlanningClientError("idempotency_conflict", "请求 ID 已用于不同任务", 409)
                    return index
            plan_id = str(uuid4())
            index = {"plan_id": plan_id, "request_id": request_id, "mission": mission,
                     "input_hash": digest(mission), "revision": 0, "job_id": None,
                     "status": "interrupted", "created_at": timestamp(), "updated_at": timestamp(), "logs": []}
            self._write(root, f".nanobot/plans/{plan_id}/plan.json", index)
            return index

    def update_job(self, root: Path, plan_id: str, job: dict) -> dict:
        with self.lock:
            index = self.get(root, plan_id)["plan"]
            job_id = identifier(job["job_id"])
            if (job.get("request_id") != index["request_id"]
                    or job.get("input_hash") != index["input_hash"]
                    or job.get("mission") != index["mission"]
                    or (index["job_id"] and index["job_id"] != job_id)):
                raise PlanningClientError("content_mismatch", "远端作业与当前规划来源不一致", 502)
            statuses = {"queued", "running", "completed", "failed", "cancelled", "timed_out", "interrupted"}
            if job.get("status") not in statuses:
                raise PlanningClientError("invalid_response", "远端作业状态无效", 502)
            if index["job_id"] and (index["status"] not in {"queued", "running"}
                    or (index["status"] == "running" and job["status"] == "queued")):
                return self.get(root, plan_id)
            index.update(job_id=job_id, status=job["status"],
                         logs=job.get("logs", [])[-100:], error=job.get("error"), updated_at=timestamp())
            if job["status"] == "completed" and not index["revision"]:
                result = job["result"]
                prepared = result["prepared"]
                self._check_prepared(result["markdown"], prepared)
                document = {"plan_id": plan_id, "revision": 1, "mission": index["mission"],
                            "job_id": index["job_id"], "markdown": result["markdown"],
                            "content_hash": prepared["content_hash"], "coa": prepared["coa"],
                            "tasks": prepared["tasks"], "validation": prepared["validation"],
                            "generation": result.get("generation", {}), "created_at": timestamp()}
                # 源生成结果只保存一份，所有后续版本使用 revision 1 的来源。
                document["original_result"] = result
                relative = f".nanobot/plans/{plan_id}/revisions/1.json"
                path = self.path(root, relative)
                if path.exists():
                    existing = self._read(root, relative)
                    if existing["content_hash"] != document["content_hash"]:
                        raise PlanningClientError("version_conflict", "生成来源版本冲突", 409)
                else:
                    self._write(root, relative, document, immutable=True)
                index["revision"] = 1
            self._write(root, f".nanobot/plans/{plan_id}/plan.json", index)
            return self.get(root, plan_id)

    def _check_prepared(self, markdown: str, prepared: dict):
        if len(markdown.encode("utf-8")) > self.max_payload_bytes:
            raise PlanningClientError("payload_too_large", "规划正文超过内容上限", 413)
        if not isinstance(prepared, dict) or prepared.get("content_hash") != digest(markdown) or prepared.get("markdown") != markdown:
            raise PlanningClientError("content_mismatch", "解析结果与完整原稿不一致", 502)
        tasks = prepared.get("tasks")
        if not isinstance(tasks, list) or not tasks or not isinstance(prepared.get("coa"), dict) or not isinstance(prepared.get("validation"), dict):
            raise PlanningClientError("invalid_response", "规划解析结构无效", 502)
        ids = set()
        for task in tasks:
            if not isinstance(task, dict) or any(not isinstance(task.get(key), str) or not task[key] for key in (
                    "id", "phase_id", "phase_name", "unit_id", "unit_name", "text")):
                raise PlanningClientError("invalid_response", "规划行动字段无效", 502)
            index = task.get("action_index")
            if (not isinstance(index, int) or isinstance(index, bool) or index < 0
                    or task["id"] != f'{quote(task["phase_id"], safe="")}:{quote(task["unit_id"], safe="")}:{index}' or task["id"] in ids):
                raise PlanningClientError("invalid_response", "规划行动 ID 无效或重复", 502)
            ids.add(task["id"])
        snapshot = prepared.get("snapshot_markdown")
        selection = prepared.get("selected_task_ids")
        if (not isinstance(snapshot, str) or not snapshot.strip()
                or len(snapshot.encode("utf-8")) > self.max_payload_bytes
                or digest(snapshot) != prepared.get("snapshot_hash")
                or not isinstance(selection, list) or not selection
                or any(not isinstance(item, str) for item in selection)
                or len(set(selection)) != len(selection) or not set(selection).issubset(ids)):
            raise PlanningClientError("invalid_response", "规划选择快照无效", 502)

    def save(self, root: Path, plan_id: str, base_revision: int, markdown: str, prepared: dict) -> dict:
        with self.lock:
            base_revision = revision_number(base_revision)
            current = self.get(root, plan_id)
            if base_revision != current["plan"]["revision"]:
                raise PlanningClientError("version_conflict", "规划已被更新，请保留草稿并重新加载版本", 409)
            self._check_prepared(markdown, prepared)
            revision = base_revision + 1
            document = {"plan_id": plan_id, "revision": revision, "mission": current["plan"]["mission"],
                        "job_id": current["plan"]["job_id"], "markdown": markdown,
                        "content_hash": prepared["content_hash"], "coa": prepared["coa"],
                        "tasks": prepared["tasks"], "validation": prepared["validation"],
                        "source_revision": 1, "base_revision": base_revision, "created_at": timestamp()}
            self._write(root, f".nanobot/plans/{plan_id}/revisions/{revision}.json", document, immutable=True)
            index = current["plan"]
            index.update(revision=revision, updated_at=timestamp())
            self._write(root, f".nanobot/plans/{plan_id}/plan.json", index)
            return self.get(root, plan_id)

    def handoff_existing(self, root: Path, handoff_id: str) -> dict | None:
        with self.lock:
            relative = f"scenarios/{identifier(handoff_id)}/source.json"
            if not self.path(root, relative).exists():
                return None
            return self._read(root, relative)

    def create_handoff(self, root: Path, source: dict, markdown: str) -> dict:
        with self.lock:
            folder = f'scenarios/{identifier(source["handoff_id"])}'
            # 快照先落盘；来源存在即表示快照完整。孤立快照只允许同内容恢复。
            snapshot_path = self.path(root, f"{folder}/coa.md")
            if snapshot_path.exists():
                if digest(snapshot_path.read_text(encoding="utf-8")) != source["snapshot_hash"]:
                    raise PlanningClientError("content_mismatch", "已有交接快照内容不一致", 409)
            else:
                self._write(root, f"{folder}/coa.md", markdown, immutable=True)
            self._write(root, f"{folder}/source.json", source, immutable=True)
            return source

    def mark_handoff(self, root: Path, source: dict, status: str):
        with self.lock:
            source = {**source, "status": status}
            self._write(root, f'scenarios/{identifier(source["handoff_id"])}/source.json', source)
            return source
