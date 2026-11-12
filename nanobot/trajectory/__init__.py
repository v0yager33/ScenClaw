"""本地轨迹采集；默认开启，无外部上传。"""
from .core import configure, emit, get_recorder, scope, trace_call


def configure_default(config=None):
    if config is None:
        from nanobot.config.loader import load_config
        config = load_config()
    directory = config.trajectory.dir.expanduser() if config.trajectory.dir else config.workspace_path / "trajectories"
    if not directory.is_absolute():
        directory = config.workspace_path / directory
    return configure("scenclaw", directory, config.trajectory.enabled)


__all__ = ["configure", "emit", "get_recorder", "scope", "trace_call"]
