"""
Agent 模块 - 对话管理核心
封装 Agent 的创建、记忆、工具调用
"""

import json
import asyncio
import uuid
from typing import List, Dict, Optional, Any, AsyncGenerator, Union
from dataclasses import dataclass, field

from .llm import BaseLLM, Message, LLMResponse
from .memory import Memory, MemoryConfig
from .tools import ToolRegistry, create_default_tools
from .prompt import PromptManager
from .logging import Logger, default_logger, LogType


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

    # ─── 同步对话 ───

    def chat(self, user_input: str) -> str:
        """单轮对话"""
        self.memory.add_message(self.session_id, "user", user_input)
        self._append_context("user", user_input)
        self.memory.touch_session(self.session_id)
        self.logger.system(f"收到用户消息: {user_input[:100]}")

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
        self._append_context("assistant", response.content)
        self.logger.response(response.content)
        return response.content

    def _handle_tool_calls(self, response: LLMResponse) -> str:
        """处理工具调用循环"""
        tc_dicts = [
            {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
            for tc in response.tool_calls
        ]
        self.memory.add_message(self.session_id, "assistant", response.content or "", tool_calls=tc_dicts)
        self._append_context("assistant", response.content or "", tool_calls=tc_dicts)

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
        self._append_context("assistant", final_response.content)
        self.logger.response(final_response.content)
        return final_response.content

    # ─── 会话管理 ───

    def reset(self):
        """重置对话：清空上下文，新建 session"""
        if self._compress_task and not self._compress_task.done():
            self._compress_task.cancel()
        self._context = []
        self.memory.delete_context_snapshot(self.session_id)
        self.memory.clear_session(self.session_id)
        self.session_id = f"{self.config.user_id}_{str(uuid.uuid4())[:8]}"
        self.memory.create_session(self.session_id, self.config.context_window, self.config.user_id)
        self.logger.system(f"对话已重置, 新 session={self.session_id}")

    def get_history(self) -> List[Dict]:
        """获取对话历史（全量，从数据库）"""
        return self.memory.get_all_messages(self.session_id)

    # ─── 异步流式对话 ───

    async def achat_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        """异步流式对话，yield 文本片段"""
        await self.memory.aadd_message(self.session_id, "user", user_input)
        self._append_context("user", user_input)
        await self.memory.atouch_session(self.session_id)
        self.logger.system(f"收到用户消息: {user_input[:100]}")

        # 后台异步压缩，不阻塞用户
        self._maybe_start_compress()

        messages = self._build_messages()
        tools = self.tools.to_openai_format() or None

        async for chunk in self._astream_with_tool_handling(messages, tools):
            yield chunk

    async def _astream_with_tool_handling(self, messages: List, tools) -> AsyncGenerator[str, None]:
        """流式调用 LLM，检测 tool calls 并处理"""
        accumulated_content = ""
        tool_call_response = None

        async for chunk in self.llm.achat_stream(
            messages=messages, tools=tools,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        ):
            if isinstance(chunk, LLMResponse):
                tool_call_response = chunk
            else:
                accumulated_content += chunk
                yield chunk

        if tool_call_response and tool_call_response.tool_calls:
            async for chunk in self._ahandle_tool_calls_stream(tool_call_response, tools):
                yield chunk
        else:
            await self.memory.aadd_message(self.session_id, "assistant", accumulated_content)
            self._append_context("assistant", accumulated_content)
            self.logger.response(accumulated_content)

    async def _ahandle_tool_calls_stream(self, response, tools) -> AsyncGenerator[str, None]:
        """执行工具调用，然后流式获取后续 LLM 回复"""
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
            result = await self.tools.aexecute(tc.name, tc.arguments)
            self.logger.tool_result(tc.name, result)
            await self.memory.aadd_message(self.session_id, "tool", result, tool_call_id=tc.id)
            self._append_context("tool", result, tool_call_id=tc.id)

        messages = self._build_messages()

        async for chunk in self._astream_with_tool_handling(messages, tools):
            yield chunk
