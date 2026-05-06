"""
鉴权 & 用户/Token 数据层

- users / tokens 表初始化
- token 生成、校验
- FastAPI Depends: get_current_user
- session 归属校验: check_session_ownership
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
import uuid

from fastapi import Depends, HTTPException, Query

from .agent_pool import get_pool
from .memory_helpers import open_memory
from .settings import USER_DB_PATH


# ─── users.db 初始化 ──────────────────────────────────────
def init_user_db() -> None:
    os.makedirs(os.path.dirname(USER_DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(USER_DB_PATH)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username   TEXT PRIMARY KEY,
                created_at REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tokens (
                token      TEXT PRIMARY KEY,
                username   TEXT NOT NULL,
                created_at REAL,
                FOREIGN KEY (username) REFERENCES users(username)
            )
        """)
        conn.commit()
    finally:
        conn.close()


def user_db() -> sqlite3.Connection:
    """打开一个 users.db 连接。调用方负责 close。"""
    return sqlite3.connect(USER_DB_PATH)


# ─── Token ────────────────────────────────────────────────
def generate_token(username: str) -> str:
    raw = f"{username}:{uuid.uuid4().hex}:{time.time()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def verify_token(token: str) -> str:
    conn = user_db()
    try:
        row = conn.execute(
            "SELECT username FROM tokens WHERE token = ?", (token,)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise HTTPException(status_code=401, detail="无效或过期的 token")
    return row[0]


async def get_current_user(token: str = Query(..., description="登录返回的 token")) -> str:
    """FastAPI Depends：把 ?token= 解析成 username"""
    return verify_token(token)


# ─── session 归属校验 ─────────────────────────────────────
def check_session_ownership(username: str, session_id: str) -> None:
    """严格校验 session_id 存在且归属当前用户。

    - 空 session_id → 400
    - session 不存在 → 404
    - 跨用户 → 403
    - 已在 Agent 池里视为归属 OK（省一次 DB 往返）
    """
    sid = (session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id 不能为空")
    if (username, sid) in get_pool():
        return

    with open_memory() as mem:
        owner = mem.session_owner(sid)
    if owner is None:
        raise HTTPException(status_code=404, detail="session 不存在")
    if owner != username:
        raise HTTPException(status_code=403, detail="无权访问该 session")
