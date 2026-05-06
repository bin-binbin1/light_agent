"""
运维/系统路由
  POST /agent/reload   清空 agent 池（下次请求按最新 config 重建）
  GET  /health         健康检查（不在 /agent/* 组，代理按需白名单）
"""

from fastapi import APIRouter

from ..agent_pool import get_pool, remove_agent_locked
from ..response import ok

router = APIRouter()


@router.post("/agent/reload", summary="[开发] 清空 agent pool，下次请求按最新 config 重建")
async def reload_pool():
    pool = get_pool()
    count = 0
    for key in list(pool.keys()):
        await remove_agent_locked(key)
        count += 1
    return ok({"cleared": count}, msg=f"已清空 {count} 个 agent，下次请求按最新 config 重建")


@router.get("/health", summary="健康检查")
async def health():
    return ok({"status": "ok", "users_online": len(get_pool())})
