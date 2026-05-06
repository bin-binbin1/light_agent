"""
认证相关路由: /agent/login, /agent/logout
"""

import time

from fastapi import APIRouter, Depends, HTTPException

from ..agent_pool import aget_or_create, get_pool, remove_agent_locked
from ..memory_helpers import open_memory
from ..response import ok
from ..schemas import LoginRequest
from ..security import generate_token, get_current_user, user_db
from ..settings import TOKEN_TTL_SECONDS

router = APIRouter()


@router.post("/agent/login", summary="用户登录（仅用户名）")
async def login(req: LoginRequest):
    username = req.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="用户名不能为空")

    conn = user_db()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO users (username, created_at) VALUES (?, ?)",
            (username, time.time()),
        )
        conn.execute(
            "DELETE FROM tokens WHERE username = ? AND created_at < ?",
            (username, time.time() - TOKEN_TTL_SECONDS),
        )
        token = generate_token(username)
        conn.execute(
            "INSERT INTO tokens (token, username, created_at) VALUES (?, ?, ?)",
            (token, username, time.time()),
        )
        conn.commit()
    finally:
        conn.close()

    # 初始 session：优先恢复用户最近活跃的会话；没有就新建一个
    with open_memory() as mem:
        latest_sid = mem.get_latest_session(username)

    if latest_sid:
        session_id = latest_sid
    else:
        # 没有历史会话：立刻创建一个 Agent 以保证 sessions 行存在
        agent = await aget_or_create(username, "")
        session_id = agent.session_id

    return ok({
        "token": token,
        "username": username,
        "session_id": session_id,
    })


@router.post("/agent/logout", summary="退出登录")
async def logout(username: str = Depends(get_current_user)):
    # 驱逐该用户名下所有 (username, *) 的 Agent
    pool = get_pool()
    victims = [k for k in list(pool.keys()) if k[0] == username]
    for k in victims:
        await remove_agent_locked(k)

    conn = user_db()
    try:
        conn.execute("DELETE FROM tokens WHERE username = ?", (username,))
        conn.commit()
    finally:
        conn.close()
    return ok(msg="已退出")
