"""
Logging 模块 - 格式化日志输出
输出 AI 的思考过程、操作细节
"""

import glob
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from enum import Enum
from typing import Optional, TextIO
from dataclasses import dataclass


# ─── trace_id 上下文 ────────────────────────────────────
# 用 ContextVar 在协程间隐式传递 trace_id：
#   - HTTP 请求入口 set 一次
#   - agent / framework / access 各层 log 时自动从这里取
#   - asyncio.Task / await 都正确隔离（每协程独立）
# 不需要改 agent 业务代码任何 logger 调用签名。
_TRACE_LEN = 8  # uuid4 前 N 位；日志里短一点好读，单服务百万 QPS 也几乎不冲突
_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")

# 显示开关：light_agent 包默认 False = 日志里不带 [trace=xxx]，
# 这样独立使用 light_agent 的用户输出格式不变。
# 由调用方（如 safety_score_agent server 启动时）显式 set_trace_display(True) 开启。
# ContextVar 的 set/get 本身仍照常工作（几乎零开销），开关只控制是否在日志行里渲染。
_TRACE_DISPLAY: bool = False


def set_trace_display(enabled: bool) -> None:
    """运行期切换 trace 显示开关（不重启进程，立即生效）。"""
    global _TRACE_DISPLAY
    _TRACE_DISPLAY = bool(enabled)


def is_trace_display_enabled() -> bool:
    return _TRACE_DISPLAY


def new_trace_id() -> str:
    """生成新的 trace_id（uuid4 前 N 位）"""
    return uuid.uuid4().hex[:_TRACE_LEN]


def set_trace_id(tid: str) -> None:
    """设置当前协程的 trace_id（HTTP 请求入口/中间件调用）"""
    _trace_id_var.set(tid or "")


def get_trace_id() -> str:
    """读当前协程的 trace_id；无则返回空串"""
    return _trace_id_var.get()


class LogLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARN = 2
    ERROR = 3


class LogType(Enum):
    # 纯 ASCII 标签：跨系统（k8s/ELK/grep/纯文本终端）兼容性好。
    # 文字部分统一对齐到 11 字符（最长 TOOL_REASON），方便扫读。
    THINKING         = "THINK      "
    TOOL_CALL        = "TOOL_CALL  "
    TOOL_RESULT      = "TOOL_RET   "
    TOOL_CALL_REASON = "TOOL_REASON"
    RESPONSE         = "REPLY      "
    SYSTEM           = "SYS        "
    ERROR            = "ERR        "
    COMPRESS         = "PACK       "
    DEBUG            = "DBG        "


@dataclass
class LogConfig:
    level: LogLevel = LogLevel.DEBUG
    show_timestamp: bool = True
    show_type_prefix: bool = True
    colorize: bool = True
    # 文件输出（None 表示走 stderr）。配 set_log_file() 使用。
    log_file: Optional[str] = None
    # 按天轮转，保留多少份历史
    log_backup_count: int = 30


class DailyFileHandler(logging.Handler):
    """按天直接写新文件名 base_path.YYYY-MM-DD，不重命名任何旧文件。

    跟 TimedRotatingFileHandler 的区别：
      标准 handler 是"主文件 -> 改名归档 + 新建主文件"，会让按文件名追踪进度的
      日志收集器把新主文件当成新文件重抓一遍。本 handler 永远不重命名——
      每个文件从生到死只用一个名字，收集器对每个文件的进度持久有效。

    base_path 用户传入是"逻辑路径"（如 logs/agent.log），实际写入的是带日期后缀的
    logs/agent.log.YYYY-MM-DD。日志平台用 logs/agent.log.* 通配匹配收集即可。

    backup_count: 保留几天历史。每天切换时检查一次，超过的本进程删除。0 = 不删。
    """

    def __init__(self, base_path: str, backup_count: int = 0, encoding: str = "utf-8"):
        super().__init__()
        self.base_path = base_path
        self.backup_count = max(0, int(backup_count))
        self.encoding = encoding
        self._current_date: Optional[str] = None
        self._stream: Optional[TextIO] = None

    @staticmethod
    def _today() -> str:
        return time.strftime("%Y-%m-%d")

    def _ensure_stream(self) -> None:
        today = self._today()
        if today == self._current_date and self._stream is not None:
            return
        # 跨天 / 首次：关旧、开今天的文件
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        path = f"{self.base_path}.{today}"
        log_dir = os.path.dirname(os.path.abspath(path))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        # 用 mode="a" 而不是 "w"——同一天进程重启不能覆盖之前内容
        self._stream = open(path, "a", encoding=self.encoding)
        self._current_date = today
        self._cleanup_old()

    def _cleanup_old(self) -> None:
        """保留最近 backup_count 个 base_path.YYYY-MM-DD 文件，删旧的。"""
        if self.backup_count <= 0:
            return
        # base_path.YYYY-MM-DD 文件列表（按文件名排序 = 按日期升序）
        candidates = sorted(glob.glob(f"{self.base_path}.20[0-9][0-9]-[0-1][0-9]-[0-3][0-9]"))
        # 多于 backup_count 的最早的几个删掉
        excess = len(candidates) - self.backup_count
        for path in candidates[:max(0, excess)]:
            try:
                os.remove(path)
            except Exception:
                # 删不掉不影响主流程，下次再试
                pass

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._ensure_stream()
            line = self.format(record)
            self._stream.write(line + "\n")
            self._stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        try:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
        finally:
            super().close()


class Logger:
    """格式化日志器"""

    # 通用 ANSI 控制符
    COLORS = {
        LogLevel.DEBUG: "\033[90m",     # 灰色
        LogLevel.INFO: "\033[92m",      # 绿色
        LogLevel.WARN: "\033[93m",      # 黄色
        LogLevel.ERROR: "\033[91m",     # 红色
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
    }

    # 不同 LogType 用独立颜色，方便扫读时一眼区分
    # 颜色不能渲染的终端（旧 cmd / 文件重定向）下，emoji + 文字缩写仍生效
    TYPE_COLORS = {
        LogType.THINKING:         "\033[96m",   # 青
        LogType.TOOL_CALL:        "\033[93m",   # 黄
        LogType.TOOL_RESULT:      "\033[95m",   # 品红
        LogType.TOOL_CALL_REASON: "\033[33m",   # 暗黄
        LogType.RESPONSE:         "\033[92m",   # 绿
        LogType.SYSTEM:           "\033[94m",   # 蓝
        LogType.ERROR:            "\033[91m",   # 红
        LogType.COMPRESS:         "\033[35m",   # 紫
        LogType.DEBUG:            "\033[90m",   # 灰
    }

    def __init__(self, config: Optional[LogConfig] = None):
        self.config = config or LogConfig()
        self.disabled_types: set = set()  # 被禁用的 LogType 集合
        self._file_handler: Optional[DailyFileHandler] = None
        # 配置里写了 log_file 就立即接管
        if self.config.log_file:
            self.set_log_file(self.config.log_file, self.config.log_backup_count)

    @staticmethod
    def _single_line(text: str) -> str:
        """把多行文本压成单行：换行 -> \\n 字面，回车 -> 丢弃，制表符保留。
        这样一条日志永远占一行，grep/awk/日志收集系统好处理。"""
        if text is None:
            return ""
        # 先把 CRLF 折成 LF，再把 LF 转成可见 \n 序列
        return text.replace("\r\n", "\n").replace("\r", "").replace("\n", "\\n")

    def set_log_file(self, path: str, backup_count: int = 30) -> None:
        """切换到文件输出（按天直接写新文件，保留 backup_count 份历史）。
        调用后 stderr 不再输出，颜色码不写入文件。

        实际写入的文件是 path.YYYY-MM-DD（带日期后缀）。
        path 本身不会被创建，仅作为命名前缀。日志收集器盯 path.* 通配即可。
        每个文件从生到死只用一个名字，跨天直接开新文件，不会触发文件名变更。

        Args:
            path: 日志文件命名前缀（如 logs/agent.log）。目录会自动创建。
            backup_count: 保留几份历史日志（按天，0 = 不删）。
        """
        # 旧 handler 先关掉，避免重复挂
        if self._file_handler is not None:
            try:
                self._file_handler.close()
            except Exception:
                pass

        handler = DailyFileHandler(path, backup_count=backup_count, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._file_handler = handler

        self.config.log_file = path
        self.config.colorize = False  # 文件里不写 ANSI 颜色码

    def _format(self, log_type: LogType, message: str, level: LogLevel = LogLevel.INFO) -> str:
        parts = []

        # 消息先单行化（无论终端还是文件，都是一行）
        message = self._single_line(message)

        # 时间戳：完整日期+时间，跨天/跨日志归档时不会丢上下文
        if self.config.show_timestamp:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            if self.config.colorize:
                parts.append(f"{self.COLORS['dim']}{ts}{self.COLORS['reset']}")
            else:
                parts.append(ts)

        # 类型前缀：按 LogType 着色（emoji + 等宽文字）
        if self.config.show_type_prefix:
            if self.config.colorize:
                tcolor = self.TYPE_COLORS.get(log_type, "")
                parts.append(f"{tcolor}{log_type.value}{self.COLORS['reset']}")
            else:
                parts.append(log_type.value)

        # trace_id 前缀：开关开 + 当前协程有 trace 时才注入
        # 默认关闭 -> 日志输出和加 trace 之前完全一致（向后兼容）
        # 开关开 -> grep "[trace=xxxxxxxx]" 拉全链
        if _TRACE_DISPLAY:
            tid = get_trace_id()
            if tid:
                message = f"[trace={tid}] {message}"

        # 消息：按 level 着色（错误红、debug 灰、其余跟随终端默认）
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
        if level.value < self.config.level.value:
            return

        line = self._format(log_type, message, level)
        if self._file_handler is not None:
            # 走文件：用 LogRecord 包一下，享受 TimedRotatingFileHandler 的轮转
            record = logging.LogRecord(
                name="light_agent", level=logging.INFO, pathname="",
                lineno=0, msg=line, args=None, exc_info=None,
            )
            self._file_handler.emit(record)
        else:
            print(line, file=sys.stderr)

    def thinking(self, content: str):
        self.log(LogType.THINKING, content)

    def tool_call(self, name: str, arguments: str):
        self.log(LogType.TOOL_CALL, f"{name}({arguments})")

    def tool_result(self, name: str, result: str, max_length: int = 200):
        preview = result[:max_length] + "..." if len(result) > max_length else result
        self.log(LogType.TOOL_RESULT, f"{name} -> {preview}")

    def response(self, content: str):
        self.log(LogType.RESPONSE, content)

    def system(self, message: str):
        self.log(LogType.SYSTEM, message)

    def error(self, message: str):
        self.log(LogType.ERROR, message, LogLevel.ERROR)

    def compress(self, message: str):
        self.log(LogType.COMPRESS, message)

    def debug(self, message: str):
        self.log(LogType.DEBUG, message, LogLevel.DEBUG)

    def set_debug(self, enabled: bool):
        """快捷开关：启用/禁用 DEBUG 级别日志"""
        if enabled:
            self.enable(LogType.DEBUG)
        else:
            self.disable(LogType.DEBUG)


# 全局默认 logger
default_logger = Logger()
