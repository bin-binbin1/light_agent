"""
Light Agent - FastAPI 服务端
支持用户名登录、流式对话（SSE）、会话管理

接口返回约定：
  所有 JSON 端点统一返回 {"code": int, "msg": str, "data": Any}
  - code=0    表示成功，msg="ok"，data 为业务数据
  - code!=0   表示失败，通常等于 HTTP 状态码，msg 为错误描述，data 为 null 或补充信息

例外：
  - /agent/chat 是 SSE 流式事件（event:/data: 协议），不做外层包装
  - /agent/index 返回 HTML，/agent/static/* 返回静态资源，均不包装
"""

from __future__ import annotations

import os
import sys
import uuid
import time
import sqlite3
import asyncio
import hashlib
import dataclasses
import json as _json
from typing import Any
from datetime import datetime
from contextlib import asynccontextmanager

# 确保项目根目录在 sys.path，允许 `from src.xxx` 导入
_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_here)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from dotenv import load_dotenv
    for _p in (os.path.join(_project_root, ".env"),
               os.path.join(os.path.dirname(_project_root), ".env")):
        if os.path.exists(_p):
            load_dotenv(_p)
            break
except ImportError:
    pass

from fastapi import FastAPI, Depends, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.config import Config
from src.events import AgentEvent


# ─── 常量 ───

CONFIG_PATH = os.getenv("LIGHT_AGENT_CONFIG", "config/config.json")
USER_DB_PATH = os.getenv("USER_DB_PATH", "data/users.db")
WEB_DIR = os.path.join(_project_root, "web")
IDLE_TIMEOUT = int(os.getenv("LIGHT_AGENT_IDLE_TIMEOUT", "1800"))          # 30 分钟
TOKEN_TTL_SECONDS = int(os.getenv("LIGHT_AGENT_TOKEN_TTL", str(7 * 86400)))  # 7 天


# ─── 统一返回格式 ───

def ok(data: Any = None, msg: str = "ok") -> dict:
    return {"code": 0, "msg": msg, "data": data}


def err(code: int, msg: str, data: Any = None) -> dict:
    return {"code": code, "msg": msg, "data": data}


# ─── 全局状态（进程内共享） ───

_agent_pool: dict = {}          # username -> Agent
_agent_locks: dict = {}         # username -> asyncio.Lock（同一用户的 chat/reset/logout 串行化）
_last_active: dict = {}         # username -> timestamp
_pool_lock: asyncio.Lock | None = None   # agent 创建并发保护，lifespan 里懒初始化


# ─── 用户 / Token DB ───

def _init_user_db() -> None:
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


def _user_db() -> sqlite3.Connection:
    return sqlite3.connect(USER_DB_PATH)


def _generate_token(username: str) -> str:
    raw = f"{username}:{uuid.uuid4().hex}:{time.time()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _verify_token(token: str) -> str:
    conn = _user_db()
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
    return _verify_token(token)


# ─── Agent 池 ───

def _create_agent_sync(username: str):
    """同步创建 Agent（阻塞）。每次调用重读 config.json，保证热更新生效"""
    config = Config(CONFIG_PATH)
    return config.create_agent_from_config(
        resume=True,
        session_id="",
        user_id=username,
    )


async def _aget_or_create_agent(username: str):
    """取/造该用户的 Agent。首次创建走线程池避免阻塞 event loop，并加 pool lock 防并发重建"""
    _last_active[username] = time.time()
    if username in _agent_pool:
        return _agent_pool[username]

    assert _pool_lock is not None, "Pool lock 未初始化（应由 lifespan 注入）"
    async with _pool_lock:
        if username in _agent_pool:   # double-check
            return _agent_pool[username]
        agent = await asyncio.to_thread(_create_agent_sync, username)
        _agent_pool[username] = agent
        _agent_locks[username] = asyncio.Lock()
        return agent


def _remove_agent(username: str) -> None:
    agent = _agent_pool.pop(username, None)
    if agent is not None:
        try:
            agent.save_state()
        except Exception as e:
            print(f"[warn] save_state failed for {username}: {e}", file=sys.stderr)
    _agent_locks.pop(username, None)
    _last_active.pop(username, None)


async def _remove_agent_locked(username: str) -> None:
    """拿到该用户 chat lock 后再释放 Agent，保证正在进行的 chat 流跑完"""
    lock = _agent_locks.get(username)
    if lock is not None:
        async with lock:
            _remove_agent(username)
    else:
        _remove_agent(username)


async def _cleanup_idle_agents() -> None:
    while True:
        await asyncio.sleep(300)
        now = time.time()
        idle = [u for u, t in list(_last_active.items()) if now - t > IDLE_TIMEOUT]
        for username in idle:
            await _remove_agent_locked(username)


# ─── Lifespan ───

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool_lock
    _init_user_db()
    _pool_lock = asyncio.Lock()

    cleanup_task = asyncio.create_task(_cleanup_idle_agents())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        for username in list(_agent_pool.keys()):
            _remove_agent(username)


# ─── FastAPI App ───

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

if os.path.isdir(WEB_DIR):
    app.mount("/agent/static", StaticFiles(directory=WEB_DIR), name="static")


# ─── 全局异常处理（统一包装成 {code,msg,data}） ───

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


# ─── 请求模型 ───

class LoginRequest(BaseModel):
    username: str


class ChatRequest(BaseModel):
    message: str


# ─── 接口 ───

@app.post("/agent/login", summary="用户登录（仅用户名）")
async def login(req: LoginRequest):
    username = req.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="用户名不能为空")

    conn = _user_db()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO users (username, created_at) VALUES (?, ?)",
            (username, time.time()),
        )
        conn.execute(
            "DELETE FROM tokens WHERE username = ? AND created_at < ?",
            (username, time.time() - TOKEN_TTL_SECONDS),
        )
        token = _generate_token(username)
        conn.execute(
            "INSERT INTO tokens (token, username, created_at) VALUES (?, ?, ?)",
            (token, username, time.time()),
        )
        conn.commit()
    finally:
        conn.close()

    agent = await _aget_or_create_agent(username)
    return ok({
        "token": token,
        "username": username,
        "session_id": agent.session_id,
    })


@app.post("/agent/logout", summary="退出登录")
async def logout(username: str = Depends(get_current_user)):
    await _remove_agent_locked(username)

    conn = _user_db()
    try:
        conn.execute("DELETE FROM tokens WHERE username = ?", (username,))
        conn.commit()
    finally:
        conn.close()
    return ok(msg="已退出")


def _sse(event_name: str, data_obj) -> str:
    return f"event: {event_name}\ndata: {_json.dumps(data_obj, ensure_ascii=False)}\n\n"


@app.post("/agent/chat", summary="对话（SSE 流式，不做 code/msg/data 包装）")
async def chat(req: ChatRequest, username: str = Depends(get_current_user)):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    send_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message_with_time = f"[发送时间: {send_time}]\n{message}"

    agent = await _aget_or_create_agent(username)
    lock = _agent_locks[username]

    async def event_stream():
        async with lock:
            try:
                async for chunk in agent.achat_stream(message_with_time):
                    if isinstance(chunk, AgentEvent):
                        yield _sse(chunk.type, dataclasses.asdict(chunk))
                    else:
                        yield _sse("chunk", {"text": chunk})
                yield _sse("done", {})
            except Exception as e:
                print(f"[chat stream error] {type(e).__name__}: {e}", file=sys.stderr)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/agent/reset", summary="重置对话")
async def reset_session(username: str = Depends(get_current_user)):
    agent = await _aget_or_create_agent(username)
    lock = _agent_locks[username]
    async with lock:
        agent.reset()
    return ok({"session_id": agent.session_id}, msg="对话已重置")


@app.get("/agent/session", summary="当前会话信息")
async def get_session(username: str = Depends(get_current_user)):
    agent = await _aget_or_create_agent(username)
    return ok({"username": username, "session_id": agent.session_id})


@app.get("/agent/history", summary="当前会话历史")
async def get_history(username: str = Depends(get_current_user)):
    agent = await _aget_or_create_agent(username)
    return ok({"messages": agent.get_history()})


@app.post("/agent/reload", summary="[开发] 清空 agent pool，下次请求按最新 config 重建")
async def reload_pool():
    assert _pool_lock is not None
    count = 0
    async with _pool_lock:
        for username in list(_agent_pool.keys()):
            await _remove_agent_locked(username)
            count += 1
    return ok({"cleared": count}, msg=f"已清空 {count} 个 agent，下次请求按最新 config 重建")


@app.get("/agent/index", response_class=HTMLResponse, summary="Web 聊天页面（HTML，不包装）")
async def index_page():
    html_path = os.path.join(WEB_DIR, "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="web/index.html 不存在")
    return FileResponse(html_path, media_type="text/html")


@app.get("/health", summary="健康检查")
async def health():
    return ok({"status": "ok", "users_online": len(_agent_pool)})
