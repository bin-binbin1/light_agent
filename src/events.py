"""
Agent 事件 - 供 achat_stream 流式输出状态
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class AgentEvent:
    """基础事件，子类通过 type 区分"""
    type: str = ""


@dataclass
class ThinkingEvent(AgentEvent):
    type: str = "thinking"


@dataclass
class ToolCallEvent(AgentEvent):
    type: str = "tool_call"
    name: str = ""          # 内部工具名，仅日志/调试用
    display: str = ""       # 对用户展示的文案


@dataclass
class ToolResultEvent(AgentEvent):
    type: str = "tool_result"
    name: str = ""          # 内部工具名
    duration_ms: int = 0
    success: bool = True
    display: str = ""       # 对用户展示的文案


@dataclass
class RetryEvent(AgentEvent):
    type: str = "retry"
    reason: str = ""   # "429" 等
    attempt: int = 0
    max_attempts: int = 0
    wait_seconds: float = 0.0


@dataclass
class ErrorEvent(AgentEvent):
    type: str = "error"
    message: str = ""
