"""
EphemeralAgent - 纯内存临时 Agent

与 Agent 功能等价，但不依赖 Memory / SQLite：
  - 上下文仅存内存，进程结束即丢失
  - 无压缩、无持久化、无 RAG 检索
  - 通过 inject_memory() 注入外部上下文
  - 适合短生命周期场景（每次请求新建、无状态服务）

用法:
    agent = EphemeralAgent(llm=llm, config=config, tools=tools)
    agent.inject_memory(history, mode="replace")
    async for chunk in agent.achat_stream("你好"):
        print(chunk)
"""

import json
import uuid
from typing import List, Dict, Optional, AsyncGenerator, Union

from .llm import BaseLLM, Message, LLMResponse
from .tools import ToolRegistry, create_default_tools
from .prompt import PromptManager
from .agent_logging import Logger, default_logger, LogType
from .events import (
    AgentEvent, ThinkingEvent, ToolCallEvent, ToolResultEvent,
    ErrorEvent,
)
from .agent import AgentConfig, strip_leading_think


class EphemeralAgent:
    """纯内存临时 Agent，无 DB 依赖"""

    def __init__(self, llm: BaseLLM, config: Optional[AgentConfig] = None,
                 tools: Optional[ToolRegistry] = None,
                 logger: Optional[Logger] = None):
        self.llm = llm
        self.config = config or AgentConfig()
        self.tools = tools or create_default_tools()
        self.logger = logger or default_logger
        self.prompt_mgr = PromptManager(self.config.system_prompt or PromptManager().system_prompt)

        # session_id（仅用于日志标识，不写 DB）
        if self.config.session_id:
            self.session_id = self.config.session_id
        else:
            self.session_id = f"{self.config.user_id}_{str(uuid.uuid4())[:8]}"

        # 工具注册
        self.tools.set_context(user_id=self.config.user_id)

        # 纯内存上下文
        self._context: List[Dict] = []

        self.logger.system(f"EphemeralAgent '{self.config.name}' 已创建, session={self.session_id}")

    # ─── 上下文注入 ───

    def inject_memory(self, messages: List[Dict], mode: str = "append"):
        """注入外部消息到上下文（纯内存，不写 DB）。

        Args:
            messages: 消息列表，每条需有 role/content，可选 tool_calls / tool_call_id
            mode: "append" 追加, "replace" 清空后替换
        """
        normalized = []
        for m in messages:
            msg = {"role": m.get("role", "user"), "content": m.get("content", "")}
            if m.get("tool_calls"):
                msg["tool_calls"] = m["tool_calls"]
            if m.get("tool_call_id"):
                msg["tool_call_id"] = m["tool_call_id"]
            normalized.append(msg)

        if mode == "replace":
            self._context = normalized
        else:
            self._context.extend(normalized)
        self.logger.system(f"外部 memory 已注入: mode={mode}, 共 {len(messages)} 条")

    # ─── 上下文操作 ───

    def _append_context(self, role: str, content: str,
                        tool_calls: Optional[List[Dict]] = None,
                        tool_call_id: Optional[str] = None):
        msg = {"role": role, "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        self._context.append(msg)

    def _build_messages(self) -> List[Message]:
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
                tool_call_id=msg.get("tool_call_id"),
            ))
        return messages

    # ─── 状态管理（no-op） ───

    def save_state(self):
        pass

    def load_state(self):
        pass

    def reset(self) -> str:
        self._context = []
        new_sid = f"{self.config.user_id}_{str(uuid.uuid4())[:8]}"
        self.session_id = new_sid
        self.logger.system(f"EphemeralAgent reset, session={new_sid}")
        return new_sid

    def clear_history(self) -> None:
        self._context = []

    def get_history(self) -> List[Dict]:
        return list(self._context)

    # ─── 同步对话 ───

    def chat(self, user_input: str) -> str:
        import time as _time

        self._append_context("user", user_input)
        self.logger.system(f"收到用户消息: {user_input[:200]}")

        messages = self._build_messages()
        tools = self.tools.to_openai_format() or None

        t0 = _time.time()
        response = self.llm.chat(
            messages=messages, tools=tools,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        round1_ms = int((_time.time() - t0) * 1000)
        n_tools = len(response.tool_calls or [])
        self.logger.system(f"[timing] Round1 LLM = {round1_ms}ms, tool_calls={n_tools}")

        if response.tool_calls:
            return self._handle_tool_calls(response)

        ctx_content = (
            strip_leading_think(response.content)
            if self.config.drop_leading_think else response.content
        )
        self._append_context("assistant", ctx_content)
        self.logger.response(response.content)
        # drop_leading_think 时返回值也剥离，调用方拿到干净文本
        return ctx_content

    def _handle_tool_calls(self, response: LLMResponse, _round: int = 1) -> str:
        import time as _time

        tc_dicts = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
            for tc in response.tool_calls
        ]
        ctx_reason = (
            strip_leading_think(response.content or "")
            if self.config.drop_leading_think else (response.content or "")
        )
        self._append_context("assistant", ctx_reason, tool_calls=tc_dicts)

        if self.config.debug:
            self.logger.log(LogType.TOOL_CALL_REASON, f"tool_call_reason: {response.content}")

        for tc in response.tool_calls:
            self.logger.tool_call(tc.name, str(tc.arguments))
            t_tool = _time.time()
            try:
                result = self.tools.execute(tc.name, tc.arguments)
            except Exception as e:
                result = f"工具执行错误: {e}"
                self.logger.error(f"工具 {tc.name} 执行异常: {e}")
            tool_ms = int((_time.time() - t_tool) * 1000)
            self.logger.tool_result(tc.name, result)
            self.logger.system(f"[timing] tool {tc.name} = {tool_ms}ms")
            self._append_context("tool", result, tool_call_id=tc.id)

        messages = self._build_messages()
        tools = self.tools.to_openai_format() or None

        t0 = _time.time()
        final_response = self.llm.chat(
            messages=messages, tools=tools,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        round_ms = int((_time.time() - t0) * 1000)
        n_tools = len(final_response.tool_calls or [])
        self.logger.system(
            f"[timing] Round{_round + 1} LLM = {round_ms}ms, tool_calls={n_tools}"
        )

        if final_response.tool_calls:
            return self._handle_tool_calls(final_response, _round=_round + 1)

        ctx_final = (
            strip_leading_think(final_response.content)
            if self.config.drop_leading_think else final_response.content
        )
        self._append_context("assistant", ctx_final)
        self.logger.response(final_response.content)
        return ctx_final

    # ─── 异步对话（完整返回） ───

    async def achat(self, user_input: str) -> str:
        """异步非流式对话，等 LLM 全部生成完后返回完整文本。"""
        import asyncio
        return await asyncio.to_thread(self.chat, user_input)

    # ─── 异步流式对话（仅文本） ───

    async def achat_stream_text(self, user_input: str) -> AsyncGenerator[str, None]:
        """异步流式对话，仅 yield 文本片段（过滤掉 AgentEvent 状态事件）。

        与 achat_stream 的区别：
          - achat_stream: yield Union[str, AgentEvent]，事件透出给上层 SSE 用
          - achat_stream_text: 只 yield str，方便客户端直接拼接文本
        """
        async for chunk in self.achat_stream(user_input):
            if isinstance(chunk, str):
                yield chunk
            # AgentEvent 全部丢弃

    # ─── 异步流式对话 ───

    async def achat_stream(self, user_input: str) -> AsyncGenerator[Union[str, AgentEvent], None]:
        self._append_context("user", user_input)
        self.logger.system(f"收到用户消息: {user_input[:100]}")

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
                yield chunk
            else:
                accumulated_content += chunk
                yield chunk

        if tool_call_response and tool_call_response.tool_calls:
            async for chunk in self._ahandle_tool_calls_stream(tool_call_response, tools):
                yield chunk
        else:
            self._append_context("assistant", accumulated_content)
            self.logger.response(accumulated_content)

    async def _ahandle_tool_calls_stream(self, response, tools) -> AsyncGenerator[Union[str, AgentEvent], None]:
        import time as _time

        tc_dicts = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
            for tc in response.tool_calls
        ]
        self._append_context("assistant", response.content or "", tool_calls=tc_dicts)

        if self.config.debug:
            self.logger.log(LogType.TOOL_CALL_REASON, f"tool_call_reason: {response.content}")

        for tc in response.tool_calls:
            self.logger.tool_call(tc.name, str(tc.arguments))
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
            duration_ms = int((_time.time() - t0) * 1000)

            self.logger.tool_result(tc.name, result)
            if tool_def:
                display_result = tool_def.display_done if success else tool_def.display_failed
            else:
                display_result = "调用成功" if success else "调用失败"
            yield ToolResultEvent(
                name=tc.name, duration_ms=duration_ms,
                success=success, display=display_result,
            )

            self._append_context("tool", result, tool_call_id=tc.id)

        yield ThinkingEvent()
        messages = self._build_messages()

        async for chunk in self._astream_with_tool_handling(messages, tools):
            yield chunk
