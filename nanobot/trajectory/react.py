"""ReAct 的真实执行结果与模型可见 observation 分别持久化。"""
from functools import wraps

from .core import capture_failure, digest, emit, scope, uid


def record(event, build):
    try:
        emit(event, build())
    except Exception:
        capture_failure()


def record_tools(tools):
    result = {}
    for name, function in tools.items():
        def build(tool_name, execute):
            @wraps(execute)
            def wrapped(*args, **kwargs):
                with scope(tool_call_id=uid(), tool_name=tool_name):
                    emit("tool_start", {"name": tool_name, "args": args, "arguments": kwargs})
                    try:
                        value = execute(*args, **kwargs)
                    except BaseException as exc:
                        emit("tool_end", {"name": tool_name, "error": str(exc), "status": "error"})
                        raise
                    record(
                        "tool_end",
                        lambda: {
                        "name": tool_name,
                        "result": value,
                        "status": "success",
                        "input_artifact_hash": digest(
                        args[ 0]) if args else None})
                    if tool_name == "validate":
                        # 静态规则检查不是任务正确性证明，保留其真实范围而不自动标 verified。
                        record(
                            "validation_observation",
                            lambda: {
                            "name": tool_name,
                            "result": value,
                            "artifact_hash": digest(
                            args[ 0]) if args else None,
                            "source": "real_execution",
                            "validation_scope": "scenario_static_rules"})
                    return value
            return wrapped
        result[name] = build(name, function)
    return result


def action_scope(function):
    @wraps(function)
    def wrapped(self, action, action_input, step):
        with scope(action_id=uid(), stage=action, iteration=step):
            before = self._current_code
            record(
                "react_action",
                lambda: {
                "name": action,
                "arguments": action_input,
                "code_before": before,
                "artifact_hash": digest(
                before)})
            result = function(self, action, action_input, step)
            record(
                "react_observation",
                lambda: {
                "name": action,
                "observation": result,
                "code_after": self._current_code,
                "artifact_hash": digest(
                self._current_code)})
            return result
    return wrapped
