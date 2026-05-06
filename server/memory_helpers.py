"""
Memory 的 one-shot 使用助手

所有 server 层需要短暂读写 Memory 的地方（列 session、查历史、校验归属、删数据）
都走这里的 context manager，避免重复构造 + 确保 close。
"""

from contextlib import contextmanager

from src.config import Config
from src.memory import Memory, MemoryConfig

from .settings import CONFIG_PATH


@contextmanager
def open_memory():
    """用完即关的 Memory 上下文管理器。

    用法:
        with open_memory() as mem:
            owner = mem.session_owner(sid)
    """
    cfg = Config(CONFIG_PATH)
    mem = Memory(MemoryConfig(
        db_path=cfg.get("memory_db", "memory.db"),
        dialect=cfg.dialect,
    ))
    try:
        yield mem
    finally:
        mem.close()
