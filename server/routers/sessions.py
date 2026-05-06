"""
会话管理路由
  POST /agent/reset            开启新会话
  POST /agent/clear            清除指定 session 历史
  GET  /agent/session          当前会话信息（不传 session_id 返回最近）
  GET  /agent/sessions         列出所有 session
  POST /agent/session/delete   删除 session
  GET  /agent/history          查指定 session 历史
"""

import asyncio
import sys

from fastapi import APIRouter, Depends, Query

from ..agent_pool import (
    create_new_session,
    get_locks,
    get_pool,
    remove_agent_no_save,
)
from ..memory_helpers import open_memory
from ..response import ok
from ..schemas import SessionActionRequest
from ..security import check_session_ownership, get_current_user

router = APIRouter()


@router.post("/agent/reset", summary="开启新会话（旧会话数据保留）")
async def reset_session(req: SessionActionRequest, username: str = Depends(get_current_user)):
    check_session_ownership(username, req.session_id)

    pool = get_pool()
    locks = get_locks()

    # 若旧 Agent 在池里，先把它的内存上下文落盘，确保旧 session 历史完整
    old_key = (username, req.session_id)
    if old_key in pool:
        async with locks[old_key]:
            try:
                pool[old_key].save_state()
            except Exception as e:
                print(f"[warn] save_state on reset failed: {e}", file=sys.stderr)

    # 新建独立 Agent（session_id="" → Agent.__init__ 自动生成新 id）
    # 不复用旧 Agent，避免并发 chat 持有旧引用时 session_id 被换出造成写错会话
    new_agent = await create_new_session(username)
    return ok({"session_id": new_agent.session_id}, msg="已开启新会话")


@router.post("/agent/clear", summary="清除指定 session 的历史记录（session_id 保留可用）")
async def clear_history(req: SessionActionRequest, username: str = Depends(get_current_user)):
    check_session_ownership(username, req.session_id)

    pool = get_pool()
    locks = get_locks()
    key = (username, req.session_id)

    if key in pool:
        async with locks[key]:
            await asyncio.to_thread(pool[key].clear_history)
    else:
        # 不在池里就直接操作 DB，避免为了清历史而多创建一个 Agent
        with open_memory() as mem:
            mem.clear_session_messages(req.session_id)
            mem.delete_context_snapshot(req.session_id)

    return ok({"session_id": req.session_id}, msg="历史已清空")


@router.get("/agent/session", summary="当前会话信息")
async def get_session(
    username: str = Depends(get_current_user),
    session_id: str = Query("", description="不传则返回最近会话"),
):
    sid = (session_id or "").strip()
    if sid:
        check_session_ownership(username, sid)
        return ok({"username": username, "session_id": sid})

    with open_memory() as mem:
        latest_sid = mem.get_latest_session(username) or ""
    return ok({"username": username, "session_id": latest_sid})


@router.get("/agent/sessions", summary="列出当前用户所有 session")
async def list_sessions(username: str = Depends(get_current_user)):
    with open_memory() as mem:
        sessions = mem.list_sessions_by_user(username)
    return ok({"sessions": sessions})


@router.post("/agent/session/delete", summary="删除指定 session（连同所有历史）")
async def delete_session(req: SessionActionRequest, username: str = Depends(get_current_user)):
    check_session_ownership(username, req.session_id)

    pool = get_pool()
    locks = get_locks()
    key = (username, req.session_id)

    # 若该 session 在池里，先驱逐 Agent（不触发 save_state——反正数据要删）
    if key in pool:
        lock = locks.get(key)
        if lock is not None:
            async with lock:
                remove_agent_no_save(key)
        else:
            remove_agent_no_save(key)

    # 彻底删除 session 及其所有数据
    with open_memory() as mem:
        mem.clear_session(req.session_id)
        mem.delete_context_snapshot(req.session_id)

    return ok({"session_id": req.session_id}, msg="会话已删除")


@router.get("/agent/history", summary="指定会话的历史")
async def get_history(
    username: str = Depends(get_current_user),
    session_id: str = Query(..., description="要查询的 session_id"),
):
    check_session_ownership(username, session_id)

    pool = get_pool()
    key = (username, session_id)
    if key in pool:
        return ok({"messages": pool[key].get_history()})

    # 不在池里就走 one-shot Memory，不为只读请求扩池
    with open_memory() as mem:
        messages = mem.get_all_messages(session_id)
    return ok({"messages": messages})
