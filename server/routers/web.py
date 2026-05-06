"""
Web 静态页面路由
  GET /agent/index     返回 web/index.html（HTML 本身不做统一包装）
"""

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from ..settings import WEB_DIR

router = APIRouter()


@router.get("/agent/index", response_class=HTMLResponse, summary="Web 聊天页面（HTML，不包装）")
async def index_page():
    html_path = os.path.join(WEB_DIR, "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="web/index.html 不存在")
    return FileResponse(html_path, media_type="text/html")
