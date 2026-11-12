"""具体供应商的显式记录适配；重试/fallback 仍由原有基类负责。"""
from __future__ import annotations

from nanobot.providers.base import LLMProvider

from .core import emit, trace_call
from .instrumentation import run_scope
from .stream import SafeFrames


class RecordingProvider(LLMProvider):
    def __init__(self, provider, name):
        self.inner = provider
        self.name = name
        self.api_key = provider.api_key
        self.api_base = provider.api_base

    def __getattr__(self, name):
        return getattr(self.inner, name)

    @property
    def generation(self):
        return self.inner.generation

    @generation.setter
    def generation(self, value):
        self.inner.generation = value

    @property
    def supports_progress_deltas(self):
        return self.inner.supports_progress_deltas

    def get_default_model(self):
        return self.inner.get_default_model()

    async def chat(self, messages, **kwargs):
        with trace_call( self.name, kwargs.get(
            "model")
            or self.get_default_model( ), {
            "messages": messages,
            **kwargs}) as call:
            response = await self.inner.chat(messages=messages, **kwargs)
            call.complete(response)
            if response.finish_reason == "error":
                emit("provider_error", {"response": response})
            return response

    async def chat_stream(self, messages, **kwargs):
        request = {"messages": messages, **{k: v for k, v in kwargs.items() if not k.startswith("on_")}}
        with trace_call(self.name, kwargs.get("model") or self.get_default_model(), request) as call:
            safe = SafeFrames(lambda frame: call.delta(**frame))
            for name, field in ( (
                "on_content_delta",
                "content"),
                (
                "on_thinking_delta",
                "thinking"),
                (
                "on_tool_call_delta",
                "tool_calls")):
                original = kwargs.get(name)

                def build(callback, key):
                    async def tracked(value):
                        safe.push({key: value})
                        if callback:
                            await callback(value)
                    return tracked
                kwargs[name] = build(original, field)
            try:
                response = await self.inner.chat_stream(messages=messages, **kwargs)
            finally:
                safe.drain(True)
            call.complete(response)
            if response.finish_reason == "error":
                emit("provider_error", {"response": response})
            return response

    @run_scope("ProviderRetry")
    async def chat_with_retry(self, *args, **kwargs):
        return await super().chat_with_retry(*args, **kwargs)

    @run_scope("ProviderRetry")
    async def chat_stream_with_retry(self, *args, **kwargs):
        return await super().chat_stream_with_retry(*args, **kwargs)

    async def aclose(self):
        close = getattr(self.inner, "aclose", None)
        if close:
            await close()
