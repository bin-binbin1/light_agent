"""
Agent 模块 - 对话管理核心
封装 Agent 的创建、记忆、工具调用
"""

import json
import re
import asyncio
import uuid
from typing import List, Dict, Optional, Any, AsyncGenerator, Union
from dataclasses import dataclass, field


# 仅匹配开头的 <think>...</think>（允许前置空白；DOTALL 让 . 匹配换行）
_LEADING_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_leading_think(text: str) -> str:
    """剥离消息开头的 <think>...</think> 块；中间出现的不处理。

    用于把存进 _context 的 assistant 内容裁掉思考块，下一轮 LLM 历史更干净；
    日志输出仍用原始文本，避免丢失调试信息。
    """
    if not text:
        return text
    return _LEADING_THINK_RE.sub("", text, count=1)

from .llm import BaseLLM, Message, LLMResponse
from .memory import Memory, MemoryConfig
from .tools import ToolRegistry, create_default_tools
from .prompt import PromptManager
from .agent_logging import Logger, default_logger, LogType
from .events import (
    AgentEvent, ThinkingEvent, ToolCallEvent, ToolResultEvent,
    RetryEvent, ErrorEvent,
)


@dataclass
class AgentConfig:
    """Agent 配置"""
    name: str = "agent"
    system_prompt: str = ""
    context_window: int = 128000
    temperature: float = 0.7
    max_tokens: int = 4096
    memory_config: MemoryConfig = field(default_factory=MemoryConfig)
    debug: bool = False
    user_id: str = "default_user"
    session_id: str = ""
    drop_leading_think: bool = False

    # 由 from_config 自动从 Config 透传的字段名 → (默认值, 类型转换)
    # 新增"运行期可调"字段时只在这里加一行即可，不必再去 config.py / xiaoai_agent.py 各加一遍
    _CONFIG_FIELDS = (
        # (字段名,            默认值,               转换器)
        ("name",               "assistant",         str),
        ("system_prompt",      "你是一个有用的 AI 助手。", str),
        ("context_window",     128000,              int),
        ("temperature",        0.7,                 float),
        ("max_tokens",         4096,                int),
        ("debug",              False,               bool),
        ("drop_leading_think", False,               bool),
    )

    @classmethod
    def from_config(cls, cfg, *,
                    user_id: str = "",
                    session_id: str = "",
                    memory_config: Optional[MemoryConfig] = None) -> "AgentConfig":
        """从一个 Config（或任何带 .get(key, default) 的对象）装配 AgentConfig。

        新增运行期字段时，只需要：
          1. 在 AgentConfig 上加 dataclass 字段（带默认值）
          2. 在 _CONFIG_FIELDS 元组里加一行
        无需再去 create_agent_from_config / xiaoai_agent.py 等装配点改透传。

        Args:
            cfg: Config 实例或任何鸭子类型（需有 get(key, default) 方法）
            user_id: 覆盖 cfg 的 user_id；空串则回退 cfg.get("user_id")
            session_id: 显式指定 session_id
            memory_config: 由调用方组装好的 MemoryConfig（None 则用 dataclass 默认）
        """
        kwargs = {
            name: conv(cfg.get(name, default))
            for name, default, conv in cls._CONFIG_FIELDS
        }
        kwargs["user_id"] = user_id or cfg.get("user_id", "default_user")
        kwargs["session_id"] = session_id
        if memory_config is not None:
            kwargs["memory_config"] = memory_config
        return cls(**kwargs)


class Agent:
    """对话 Agent"""

    def __init__(self, llm: BaseLLM, config: Optional[AgentConfig] = None,
                 tools: Optional[ToolRegistry] = None,
                 logger: Optional[Logger] = None):
        self.llm = llm
        self.config = config or AgentConfig()
        self.tools = tools or create_default_tools()
        self.logger = logger or default_logger
        self.prompt_mgr = PromptManager(self.config.system_prompt or PromptManager().system_prompt)
        self.memory = Memory(self.config.memory_config, llm=llm)

        # 使用配置中的 session_id，如果没有则基于 user_id 生成
        if self.config.session_id:
            self.session_id = self.config.session_id
        else:
            self.session_id = f"{self.config.user_id}_{str(uuid.uuid4())[:8]}"

        # 初始化会话
        self.memory.create_session(self.session_id, self.config.context_window, self.config.user_id)

        # 工具注册
        self.tools = tools or create_default_tools(memory=self.memory)
        self.tools.set_context(user_id=self.config.user_id)

        # 内存上下文（Write-Back Cache）
        self._context: List[Dict] = self.memory.load_context(self.session_id)

        # 后台压缩任务
        self._compress_task: Optional[asyncio.Task] = None

        self.logger.system(f"Agent '{self.config.name}' 已创建, session={self.session_id}")

    # ─── 上下文操作 ───

    def _append_context(self, role: str, content: str,
                        tool_calls: Optional[List[Dict]] = None,
                        tool_call_id: Optional[str] = None):
        """追加消息到内存上下文"""
        msg = {"role": role, "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        self._context.append(msg)

    def _build_messages(self) -> List[Message]:
        """从内存上下文构建 LLM 消息列表"""
        tools_desc = self.prompt_mgr.format_tool_descriptions(
            self.tools.to_openai_format()
        )
        system_msg = self.prompt_mgr.build_system_message(tools_desc)
        messages = [Message(role="system", content=system_msg)]

        for msg in self._context:
            messages.append(Message(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                tool_calls=msg.get("tool_calls"),
                tool_call_id=msg.get("tool_call_id")
            ))
        return messages

    # ─── 后台压缩 ───

    def _maybe_start_compress(self):
        """检测是否需要压缩，需要则启动后台任务（对用户无感）"""
        if self._compress_task and not self._compress_task.done():
            return  # 已有压缩任务在跑
        need = self.memory.should_compress(self.session_id, self._context)
        if not need:
            need = self.memory.should_compress_idle(self.session_id)
        if need:
            # 快照当前上下文交给后台压缩
            snapshot = list(self._context)
            self._compress_task = asyncio.create_task(self._do_compress(snapshot))

    async def _do_compress(self, snapshot: List[Dict]):
        """后台压缩：用快照生成摘要，完成后替换 _context"""
        try:
            self.logger.compress("后台压缩启动...")
            new_context = await self.memory.acompress(self.session_id, snapshot)
            # 压缩期间用户可能追加了新消息，把增量追加上去
            extra = self._context[len(snapshot):]
            self._context = new_context + extra
            self.logger.compress("后台压缩完成")
        except Exception as e:
            self.logger.error(f"后台压缩失败: {e}")

    # ─── 状态持久化 ───

    def save_state(self):
        """将内存上下文快照存入 SQLite（释放前调用）"""
        self.memory.save_context(self.session_id, self._context)

    def load_state(self):
        """从 SQLite 加载上下文到内存"""
        self._context = self.memory.load_context(self.session_id)

    def inject_memory(self, messages: List[Dict], mode: str = "append"):
        """将外部消息注入到 agent 上下文（写入 DB + 更新内存）。

        Args:
            messages: 消息列表，每条需有 role/content，可选 tool_calls / tool_call_id
            mode: "append" 追加, "replace" 清空后替换
        """
        self._context = self.memory.inject_messages(self.session_id, messages, mode)
        self.logger.system(f"外部 memory 已注入: mode={mode}, 共 {len(messages)} 条")

    async def ainject_memory(self, messages: List[Dict], mode: str = "append"):
        """异步版本的外部 memory 注入"""
        self._context = await self.memory.ainject_messages(self.session_id, messages, mode)
        self.logger.system(f"外部 memory 已注入: mode={mode}, 共 {len(messages)} 条")

    # ─── 同步对话 ───

    def chat(self, user_input: str) -> str:
        """单轮对话"""
        self.memory.add_message(self.session_id, "user", user_input)
        self._append_context("user", user_input)
        self.memory.touch_session(self.session_id)
        self.logger.system(f"收到用户消息: {user_input[:200]}")

        # 同步模式下仍然同步压缩
        if self.memory.should_compress(self.session_id, self._context):
            self.logger.compress("上下文即将超限，启动记忆压缩...")
            self._context = self.memory.compress(self.session_id, self._context)
            self.logger.compress("压缩完成")

        messages = self._build_messages()
        tools = self.tools.to_openai_format() or None

        response = self.llm.chat(
            messages=messages,
            tools=tools,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        if response.tool_calls:
            return self._handle_tool_calls(response)

        self.memory.add_message(self.session_id, "assistant", response.content)
        ctx_content = (
            strip_leading_think(response.content)
            if self.config.drop_leading_think else response.content
        )
        self._append_context("assistant", ctx_content)
        self.logger.response(response.content)
        # drop_leading_think 时返回值也剥离，调用方拿到干净文本
        return ctx_content

    def _handle_tool_calls(self, response: LLMResponse) -> str:
        """处理工具调用循环"""
        tc_dicts = [
            {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
            for tc in response.tool_calls
        ]
        self.memory.add_message(self.session_id, "assistant", response.content or "", tool_calls=tc_dicts)
        ctx_reason = (
            strip_leading_think(response.content or "")
            if self.config.drop_leading_think else (response.content or "")
        )
        self._append_context("assistant", ctx_reason, tool_calls=tc_dicts)

        if self.config.debug:
            self.logger.log(LogType.TOOL_CALL_REASON, f"tool_call_reason: {response.content}")

        for tc in response.tool_calls:
            self.logger.tool_call(tc.name, str(tc.arguments))
            result = self.tools.execute(tc.name, tc.arguments)
            self.logger.tool_result(tc.name, result)
            self.memory.add_message(self.session_id, "tool", result, tool_call_id=tc.id)
            self._append_context("tool", result, tool_call_id=tc.id)

        messages = self._build_messages()
        tools = self.tools.to_openai_format() or None

        final_response = self.llm.chat(
            messages=messages,
            tools=tools,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        if final_response.raw and final_response.raw.get("error"):
            error_msg = final_response.raw.get("error_message", "未知错误")
            self.logger.error(f"工具调用后 LLM 返回错误: {error_msg}")
            if self.config.debug:
                error_detail = final_response.raw.get("error_detail", "")
                if error_detail:
                    self.logger.error(f"详细错误: {error_detail}")

        if final_response.tool_calls:
            return self._handle_tool_calls(final_response)

        self.memory.add_message(self.session_id, "assistant", final_response.content)
        ctx_final = (
            strip_leading_think(final_response.content)
            if self.config.drop_leading_think else final_response.content
        )
        self._append_context("assistant", ctx_final)
        self.logger.response(final_response.content)
        return ctx_final

    # ─── 会话管理 ───

    def reset(self) -> str:
        """切换到新 session；旧 session 的所有数据保留在数据库，随时可继续对话。

        返回新的 session_id。
        """
        # 把当前内存上下文快照落到旧 session，避免 _context 里的未落盘消息丢失
        if self._context:
            try:
                self.memory.save_context(self.session_id, self._context)
            except Exception as e:
                self.logger.error(f"save_context on reset failed: {e}")
        if self._compress_task and not self._compress_task.done():
            self._compress_task.cancel()
        new_sid = f"{self.config.user_id}_{str(uuid.uuid4())[:8]}"
        self.memory.create_session(new_sid, self.config.context_window, self.config.user_id)
        self.session_id = new_sid
        self._context = []
        self._compress_task = None
        self.logger.system(f"新建 session={new_sid}（旧会话数据保留）")
        return new_sid

    def clear_history(self) -> None:
        """清空当前 session 的消息/摘要/索引/快照，session_id 本身保留可继续使用。"""
        if self._compress_task and not self._compress_task.done():
            self._compress_task.cancel()
            self._compress_task = None
        self.memory.clear_session_messages(self.session_id)
        self.memory.delete_context_snapshot(self.session_id)
        self._context = []
        self.logger.system(f"会话历史已清空: session={self.session_id}")

    def get_history(self) -> List[Dict]:
        """获取对话历史（全量，从数据库）"""
        return self.memory.get_all_messages(self.session_id)

    # ─── 异步流式对话 ───

    async def achat_stream(self, user_input: str) -> AsyncGenerator[Union[str, AgentEvent], None]:
        """异步流式对话，yield 文本片段或 AgentEvent 状态事件"""
        await self.memory.aadd_message(self.session_id, "user", user_input)
        self._append_context("user", user_input)
        await self.memory.atouch_session(self.session_id)
        self.logger.system(f"收到用户消息: {user_input[:100]}")

        # 后台异步压缩，不阻塞用户
        self._maybe_start_compress()

        # 先发思考状态
        yield ThinkingEvent()

        messages = self._build_messages()
        tools = self.tools.to_openai_format() or None

        try:
            async for chunk in self._astream_with_tool_handling(messages, tools):
                yield chunk
        except Exception as e:
            yield ErrorEvent(message=str(e))
            raise

    async def _astream_with_tool_handling(self, messages: List, tools) -> AsyncGenerator[Union[str, AgentEvent], None]:
        """流式调用 LLM，检测 tool calls 并处理"""
        import time as _time

        accumulated_content = ""
        tool_call_response = None

        async for chunk in self.llm.achat_stream(
            messages=messages, tools=tools,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        ):
            if isinstance(chunk, LLMResponse):
                tool_call_response = chunk
            elif isinstance(chunk, AgentEvent):
                # LLM 层冒泡上来的事件（如 RetryEvent）
                yield chunk
            else:
                accumulated_content += chunk
                yield chunk

        # 工具调用后的 LLM 错误检测（对齐同步版）
        if tool_call_response and tool_call_response.raw and tool_call_response.raw.get("error"):
            error_msg = tool_call_response.raw.get("error_message", "未知错误")
            self.logger.error(f"工具调用后 LLM 返回错误: {error_msg}")
            if self.config.debug:
                error_detail = tool_call_response.raw.get("error_detail", "")
                if error_detail:
                    self.logger.error(f"详细错误: {error_detail}")

        if tool_call_response and tool_call_response.tool_calls:
            async for chunk in self._ahandle_tool_calls_stream(tool_call_response, tools):
                yield chunk
        else:
            await self.memory.aadd_message(self.session_id, "assistant", accumulated_content)
            self._append_context("assistant", accumulated_content)
            self.logger.response(accumulated_content)

    async def _ahandle_tool_calls_stream(self, response, tools) -> AsyncGenerator[Union[str, AgentEvent], None]:
        """执行工具调用，然后流式获取后续 LLM 回复"""
        import time as _time

        tc_dicts = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
            for tc in response.tool_calls
        ]
        await self.memory.aadd_message(
            self.session_id, "assistant", response.content or "", tool_calls=tc_dicts
        )
        self._append_context("assistant", response.content or "", tool_calls=tc_dicts)

        if self.config.debug:
            self.logger.log(LogType.TOOL_CALL_REASON, f"tool_call_reason: {response.content}")

        for tc in response.tool_calls:
            self.logger.tool_call(tc.name, str(tc.arguments))
            # 发工具调用开始事件（仅 display 文案，不透出原始参数）
            tool_def = self.tools.get(tc.name)
            display_calling = tool_def.display_calling if tool_def else f"调用 {tc.name}..."
            yield ToolCallEvent(name=tc.name, display=display_calling)

            t0 = _time.time()
            success = True
            try:
                result = await self.tools.aexecute(tc.name, tc.arguments)
            except Exception as e:
                result = f"工具执行错误: {e}"
                success = False
                self.logger.error(f"工具 {tc.name} 执行异常: {e}")
            duration_ms = int((_time.time() - t0) * 1000)

            self.logger.tool_result(tc.name, result)
            # 发工具结果事件（仅 display 文案，不透出原始结果）
            if tool_def:
                display_result = tool_def.display_done if success else tool_def.display_failed
            else:
                display_result = "调用成功" if success else "调用失败"
            yield ToolResultEvent(
                name=tc.name,
                duration_ms=duration_ms,
                success=success,
                display=display_result,
            )

            await self.memory.aadd_message(self.session_id, "tool", result, tool_call_id=tc.id)
            self._append_context("tool", result, tool_call_id=tc.id)

        # 工具执行完，下一轮 LLM 调用前再发一次 thinking
        yield ThinkingEvent()
        messages = self._build_messages()

        async for chunk in self._astream_with_tool_handling(messages, tools):
            yield chunk
