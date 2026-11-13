"""固定配置地址的异步规划客户端，不接受浏览器 URL 或重定向。"""
from __future__ import annotations

import asyncio
import json
from urllib.parse import urlsplit

import httpx

from nanobot.config.schema import PlanningConfig
from nanobot.security.network import validate_url_target


class PlanningClientError(ValueError):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.status = code, status


class PlanningClient:
    def __init__(self, config: PlanningConfig, *, transport=None):
        self.config = config
        self.transport = transport

    async def request(self, method: str, path: str, data: dict | None = None) -> dict:
        config = self.config
        if not config.enabled:
            raise PlanningClientError("not_configured", "任务规划服务未启用", 503)
        if not config.api_token.strip():
            raise PlanningClientError("not_configured", "任务规划服务缺少令牌", 503)
        base = urlsplit(config.api_base)
        if base.scheme not in {"http", "https"} or not base.hostname or base.username or base.password or base.query or base.fragment:
            raise PlanningClientError("invalid_config", "规划服务地址必须是无凭据的 HTTP(S) 地址")
        url = config.api_base.rstrip("/") + path
        safe, _ = await asyncio.to_thread(validate_url_target, url)
        if not safe:
            raise PlanningClientError("target_blocked", "规划服务地址被 SSRF 策略拒绝或无法解析；请检查显式白名单", 403)
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8") if data is not None else None
        if data is not None and "markdown" in data and len(data["markdown"].encode("utf-8")) > config.max_payload_bytes:
            raise PlanningClientError("payload_too_large", "规划正文超过内容上限", 413)
        if payload and len(payload) > config.max_payload_bytes + 256 * 1024:
            raise PlanningClientError("payload_too_large", "规划请求超过内容上限", 413)
        try:
            async with httpx.AsyncClient(transport=self.transport, timeout=config.request_timeout_s,
                                         follow_redirects=False, trust_env=False) as client:
                async with client.stream(method, url, content=payload, headers={
                    "Authorization": f"Bearer {config.api_token}", "Content-Type": "application/json",
                }) as response:
                    if 300 <= response.status_code < 400:
                        raise PlanningClientError("redirect_rejected", "规划服务返回重定向，已拒绝", 502)
                    chunks = bytearray()
                    async for chunk in response.aiter_bytes():
                        chunks.extend(chunk)
                        if len(chunks) > config.max_payload_bytes * 12:
                            raise PlanningClientError("payload_too_large", "规划响应超过内容上限", 502)
                    result = json.loads(chunks)
                    if not isinstance(result, dict):
                        raise ValueError("响应不是对象")
                    if response.status_code >= 400:
                        error = result.get("error") or {}
                        # 服务错误只暴露约定的有限错误码，不回显远端任意正文。
                        code = error.get("code", "service_error")
                        messages = {"unauthorized": "规划服务鉴权失败", "queue_full": "规划队列已满",
                                    "invalid_selection": "请选择有效且不重复的行动",
                                    "invalid_references": "规划存在重复 ID 或悬空引用",
                                    "unparseable_plan": "无法解析规划矩阵", "idempotency_conflict": "幂等请求内容冲突"}
                        raise PlanningClientError(code if code in messages else "service_error",
                                                  messages.get(code, "规划服务拒绝请求"), response.status_code)
                    return result
        except httpx.TimeoutException as exc:
            raise PlanningClientError("service_timeout", "规划服务请求超时", 504) from exc
        except httpx.HTTPError as exc:
            raise PlanningClientError("service_unavailable", "规划服务不可达", 502) from exc
        except (ValueError, UnicodeDecodeError) as exc:
            if isinstance(exc, PlanningClientError):
                raise
            raise PlanningClientError("invalid_response", "规划服务响应格式无效", 502) from exc

    async def prepare(self, markdown: str, selected_task_ids: list[str] | None = None) -> dict:
        return await self.request("POST", "/v1/coa/prepare", {
            "markdown": markdown, "selected_task_ids": selected_task_ids,
        })
