"""
Light Agent - FastAPI 服务端（装配层）

这一层只负责：
  - sys.path / dotenv 启动准备
  - FastAPI 应用实例 + 中间件 + lifespan
  - 全局异常处理（统一包装 {code,msg,data}）
  - 挂载 routers 与静态资源

具体业务逻辑位于：
  - server/agent_pool.py   Agent 池 & 生命周期
  - server/security.py     鉴权 & 归属校验
  - server/memory_helpers  Memory 一次性操作
  - server/routers/*       各路由分组（auth / chat / sessions / system / web）

接口返回约定：
  所有 JSON 端点统一返回 {"code": int, "msg": str, "data": Any}
  - code=0  成功
  - code!=0 失败（通常等于 HTTP 状态码）
例外：
  - /agent/chat   SSE 流式事件，不做外层包装
  - /agent/index  HTML，/agent/static/*  静态资源，不包装
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager

# —— sys.path 兜底：允许 `from src.xxx` ——
_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_here)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# —— .env 就近加载 ——
try:
    from dotenv import load_dotenv
    for _p in (os.path.join(_project_root, ".env"),
               os.path.join(os.path.dirname(_project_root), ".env")):
        if os.path.exists(_p):
            load_dotenv(_p)
            break
except ImportError:
    pass

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .agent_pool import (
    cleanup_idle_agents,
    init_pool_lock,
    remove_all_for_shutdown,
)
from .response import err
from .routers import auth, chat, sessions, system, web
from .security import init_user_db
from .settings import WEB_DIR


# ─── Lifespan ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_user_db()
    init_pool_lock()

    cleanup_task = asyncio.create_task(cleanup_idle_agents())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        remove_all_for_shutdown()


# ─── FastAPI App ─────────────────────────────────────────
app = FastAPI(
    title="Light Agent",
    description="轻量 Agent 框架 · FastAPI 服务端",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# —— 静态资源 ——
if os.path.isdir(WEB_DIR):
    app.mount("/agent/static", StaticFiles(directory=WEB_DIR), name="static")


# ─── 全局异常处理（统一包装 {code,msg,data}） ───────────
@app.exception_handler(StarletteHTTPException)
async def _http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=err(exc.status_code, str(exc.detail)),
    )


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=err(422, "参数校验失败", {"errors": exc.errors()}),
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    print(f"[unhandled] {type(exc).__name__}: {exc}", file=sys.stderr)
    return JSONResponse(
        status_code=500,
        content=err(500, f"服务器内部错误: {type(exc).__name__}"),
    )


# ─── 挂载路由 ─────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(sessions.router)
app.include_router(system.router)
app.include_router(web.router)
