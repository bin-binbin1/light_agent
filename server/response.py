"""
统一 JSON 响应格式: {"code": int, "msg": str, "data": Any}
"""

from typing import Any


def ok(data: Any = None, msg: str = "ok") -> dict:
    return {"code": 0, "msg": msg, "data": data}


def err(code: int, msg: str, data: Any = None) -> dict:
    return {"code": code, "msg": msg, "data": data}
