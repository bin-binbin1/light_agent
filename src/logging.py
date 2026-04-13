"""
Logging 模块 - 格式化日志输出
输出 AI 的思考过程、操作细节
"""

import sys
import time
from enum import Enum
from typing import Optional
from dataclasses import dataclass


class LogLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARN = 2
    ERROR = 3


class LogType(Enum):
    THINKING = "💭 THINKING"
    TOOL_CALL = "🔧 TOOL_CALL"
    TOOL_RESULT = "📦 TOOL_RESULT"
    TOOL_CALL_REASON = "📝 TOOL_CALL_REASON"
    RESPONSE = "💬 RESPONSE"
    SYSTEM = "⚙️ SYSTEM"
    ERROR = "❌ ERROR"
    COMPRESS = "🗜️ COMPRESS"


@dataclass
class LogConfig:
    level: LogLevel = LogLevel.DEBUG
    show_timestamp: bool = True
    show_type_prefix: bool = True
    colorize: bool = True


class Logger:
    """格式化日志器"""

    COLORS = {
        LogLevel.DEBUG: "\033[90m",     # 灰色
        LogLevel.INFO: "\033[92m",      # 绿色
        LogLevel.WARN: "\033[93m",      # 黄色
        LogLevel.ERROR: "\033[91m",     # 红色
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
    }

    def __init__(self, config: Optional[LogConfig] = None):
        self.config = config or LogConfig()
        self.disabled_types: set = set()  # 被禁用的 LogType 集合

    def _format(self, log_type: LogType, message: str, level: LogLevel = LogLevel.INFO) -> str:
        parts = []

        # 时间戳
        if self.config.show_timestamp:
            ts = time.strftime("%H:%M:%S")
            parts.append(f"{self.COLORS['dim']}{ts}{self.COLORS['reset']}")

        # 类型前缀
        if self.config.show_type_prefix:
            parts.append(f"{log_type.value}")

        # 消息
        if self.config.colorize:
            color = self.COLORS.get(level, "")
            parts.append(f"{color}{message}{self.COLORS['reset']}")
        else:
            parts.append(message)

        return " | ".join(parts)

    def disable(self, *log_types: LogType):
        """禁用指定类型的日志输出"""
        self.disabled_types.update(log_types)

    def enable(self, *log_types: LogType):
        """重新启用指定类型的日志输出"""
        self.disabled_types.difference_update(log_types)

    def log(self, log_type: LogType, message: str, level: LogLevel = LogLevel.INFO):
        if log_type in self.disabled_types:
            return
        if level.value >= self.config.level.value:
            print(self._format(log_type, message, level), file=sys.stderr)

    def thinking(self, content: str):
        self.log(LogType.THINKING, content)

    def tool_call(self, name: str, arguments: str):
        self.log(LogType.TOOL_CALL, f"{name}({arguments})")

    def tool_result(self, name: str, result: str):
        preview = result[:200] + "..." if len(result) > 200 else result
        self.log(LogType.TOOL_RESULT, f"{name} → {preview}")

    def response(self, content: str):
        self.log(LogType.RESPONSE, content)

    def system(self, message: str):
        self.log(LogType.SYSTEM, message)

    def error(self, message: str):
        self.log(LogType.ERROR, message, LogLevel.ERROR)

    def compress(self, message: str):
        self.log(LogType.COMPRESS, message)


# 全局默认 logger
default_logger = Logger()
