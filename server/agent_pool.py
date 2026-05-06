"""
Agent 池 —— 进程内共享

key = (username, session_id)：一个用户可以同时挂载多个会话 Agent，
支持客户端侧栏并发切换。

线程 / 协程安全：
  - _pool_lock 保护"创建 Agent"这一步的并发竞态
  - _agent_locks[key] 保证同一 (user, session) 的 chat / reset / clear 串行
"""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Optional

from src.config import Config

from .settings import CONFIG_PATH, IDLE_TIMEOUT


# ─── 全局状态 ──────────────────────────────────────────────
_agent_pool: dict = {}          # (username, session_id) -> Agent
_agent_locks: dict = {}         # (username, session_id) -> asyncio.Lock
_last_active: dict = {}         # (username, session_id) -> timestamp
_pool_lock: Optional[asyncio.Lock] = None  # 创建并发锁；lifespan 里注入


def init_pool_lock() -> None:
    """由 lifespan 调用，创建主 loop 上的 pool lock"""
    global _pool_lock
    _pool_lock = asyncio.Lock()


def get_pool() -> dict:
    return _agent_pool


def get_locks() -> dict:
    return _agent_locks


def touch(key: tuple) -> None:
    _last_active[key] = time.time()


# ─── 创建 / 取用 ──────────────────────────────────────────
def _create_agent_sync(username: str, session_id: str):
    """同步创建 Agent（阻塞）。每次调用重读 config.json，保证热更新生效。

    session_id 为空时，Agent.__init__ 会自动生成新 id 并落 sessions 行。
    """
    config = Config(CONFIG_PATH)
    return config.create_agent_from_config(
        resume=False,
        session_id=session_id,
        user_id=username,
    )


async def aget_or_create(username: str, session_id: str):
    """取/造 (username, session_id) 对应的 Agent。双重检查 + pool lock"""
    key = (username, session_id)
    touch(key)
    if key in _agent_pool:
        return _agent_pool[key]

    assert _pool_lock is not None, "Pool lock 未初始化（应由 lifespan 注入）"
    async with _pool_lock:
        if key in _agent_pool:
            return _agent_pool[key]
        agent = await asyncio.to_thread(_create_agent_sync, username, session_id)
        _agent_pool[key] = agent
        _agent_locks[key] = asyncio.Lock()
        return agent


async def create_new_session(username: str):
    """为 username 开一个全新 session 的 Agent，返回 agent。

    旧 session 如在池内应由调用方先 save_state；此函数只负责新建。
    """
    new_agent = await asyncio.to_thread(_create_agent_sync, username, "")
    new_key = (username, new_agent.session_id)

    assert _pool_lock is not None
    async with _pool_lock:
        _agent_pool[new_key] = new_agent
        _agent_locks[new_key] = asyncio.Lock()
        _last_active[new_key] = time.time()
    return new_agent


# ─── 回收 ────────────────────────────────────────────────
def _remove_agent(key: tuple) -> None:
    """无锁驱逐（调用方需自行持有锁，或确认没有并发）"""
    agent = _agent_pool.pop(key, None)
    if agent is not None:
        try:
            agent.save_state()
        except Exception as e:
            print(f"[warn] save_state failed for {key}: {e}", file=sys.stderr)
    _agent_locks.pop(key, None)
    _last_active.pop(key, None)


async def remove_agent_locked(key: tuple) -> None:
    """等 chat lock 空闲后再驱逐，确保正在进行的流跑完"""
    lock = _agent_locks.get(key)
    if lock is not None:
        async with lock:
            _remove_agent(key)
    else:
        _remove_agent(key)


def remove_agent_no_save(key: tuple) -> None:
    """不触发 save_state 直接驱逐（用于 session/delete —— 数据要删掉了）"""
    lock = _agent_locks.get(key)
    if lock is not None:
        # 调用方应已进入异步上下文，这里只是同步清 dict
        pass
    _agent_pool.pop(key, None)
    _agent_locks.pop(key, None)
    _last_active.pop(key, None)


# ─── 后台清理 ────────────────────────────────────────────
async def cleanup_idle_agents() -> None:
    """后台任务：定期驱逐空闲 Agent。在 lifespan 里 create_task"""
    while True:
        await asyncio.sleep(300)
        now = time.time()
        idle = [k for k, t in list(_last_active.items()) if now - t > IDLE_TIMEOUT]
        for key in idle:
            await remove_agent_locked(key)


def remove_all_for_shutdown() -> None:
    """进程退出时调用：同步落盘所有 agent 状态"""
    for key in list(_agent_pool.keys()):
        _remove_agent(key)
