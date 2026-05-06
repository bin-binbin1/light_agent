"""
server 层请求体 pydantic 模型
"""

from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str


class ChatRequest(BaseModel):
    message: str
    session_id: str


class SessionActionRequest(BaseModel):
    """reset / clear / session/delete 等只需要 session_id 的接口共用"""
    session_id: str
