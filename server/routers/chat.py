"""
对话路由: /agent/chat (SSE 流式)

注意：SSE 流不走统一 {code,msg,data} 包装，走 event:/data: 协议。
"""

import dataclasses
import json as _json
import sys
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from src.events import AgentEvent

from ..agent_pool import aget_or_create, get_locks
from ..schemas import ChatRequest
from ..security import check_session_ownership, get_current_user

router = APIRouter()


def _sse(event_name: str, data_obj) -> str:
    return f"event: {event_name}\ndata: {_json.dumps(data_obj, ensure_ascii=False)}\n\n"


@router.post("/agent/chat", summary="对话（SSE 流式，不做 code/msg/data 包装）")
async def chat(req: ChatRequest, username: str = Depends(get_current_user)):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")
    check_session_ownership(username, req.session_id)

    send_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message_with_time = f"[发送时间: {send_time}]\n{message}"

    agent = await aget_or_create(username, req.session_id)
    lock = get_locks()[(username, req.session_id)]

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
