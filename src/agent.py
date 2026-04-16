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
            # 为每个用户生成独立的会话ID
            self.session_id = f"{self.config.user_id}_{str(uuid.uuid4())[:8]}"

        # 初始化会话
        self.memory.create_session(self.session_id, self.config.context_window, self.config.user_id)

        # 工具注册（传入 memory 以支持记忆检索）
        self.tools = tools or create_default_tools(memory=self.memory)
        # 注入 agent 上下文，工具函数声明了对应参数则自动收到
        self.tools.set_context(user_id=self.config.user_id)
        self.logger.system(f"Agent '{self.config.name}' 已创建, session={self.session_id}")

    def chat(self, user_input: str) -> str:
        """单轮对话"""
        # 添加用户消息
        self.memory.add_message(self.session_id, "user", user_input)
        self.memory.touch_session(self.session_id)
        self.logger.system(f"收到用户消息: {user_input[:100]}")

        # 检查是否需要压缩（上下文超限 或 闲置太久且未压缩过）
        if self.memory.should_compress(self.session_id):
            self.logger.compress("上下文即将超限，启动记忆压缩...")
            self.memory.compress(self.session_id)
            self.logger.compress("压缩完成")
        elif self.memory.should_compress_idle(self.session_id):
            self.logger.compress(f"闲置超过{self.config.memory_config.idle_compress_hours}h且未压缩过，启动记忆压缩...")
            self.memory.compress(self.session_id)
            self.logger.compress("压缩完成")

        # 构建消息
        messages = self._build_messages()
        tools = self.tools.to_openai_format() or None

        # 调用 LLM
        response = self.llm.chat(
            messages=messages,
            tools=tools,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        # 处理工具调用
        if response.tool_calls:
            return self._handle_tool_calls(response)

        # 保存回复
        self.memory.add_message(self.session_id, "assistant", response.content)
        self.logger.response(response.content)

        return response.content

    def _build_messages(self) -> List[Message]:
        """构建发送给 LLM 的消息列表"""
        from .llm import Message

        # 系统提示词（含工具描述）
        tools_desc = self.prompt_mgr.format_tool_descriptions(
            self.tools.to_openai_format()
        )
        system_msg = self.prompt_mgr.build_system_message(tools_desc)

        messages = [Message(role="system", content=system_msg)]

        # 添加记忆上下文
        context_messages = self.memory.get_context_for_llm(self.session_id)
        for msg in context_messages:
            messages.append(Message(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                tool_calls=msg.get("tool_calls"),
                tool_call_id=msg.get("tool_call_id")
            ))

        return messages

    def _handle_tool_calls(self, response: LLMResponse) -> str:
        """处理工具调用循环"""
        # 保存 assistant 的 tool_calls 消息
        tc_dicts = [
            {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
            for tc in response.tool_calls
        ]
        self.memory.add_message(self.session_id, "assistant", response.content or "", tool_calls=tc_dicts)

        if self.config.debug:
            self.logger.log(LogType.TOOL_CALL_REASON, f"tool_call_reason: {response.content}")
        # 执行每个工具
        for tc in response.tool_calls:
            self.logger.tool_call(tc.name, str(tc.arguments))
            result = self.tools.execute(tc.name, tc.arguments)
            self.logger.tool_result(tc.name, result)
            # 添加工具结果到记忆
            self.memory.add_message(self.session_id, "tool", result, tool_call_id=tc.id)

        # 重新调用 LLM 获取最终回复
        messages = self._build_messages()
        tools = self.tools.to_openai_format() or None

        final_response = self.llm.chat(
            messages=messages,
            tools=tools,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        # 检查是否有错误
        if final_response.raw and final_response.raw.get("error"):
            error_msg = final_response.raw.get("error_message", "未知错误")
            self.logger.error(f"工具调用后 LLM 返回错误: {error_msg}")

            # 在 debug 模式下输出详细错误信息
            if self.config.debug:
                error_detail = final_response.raw.get("error_detail", "")
                if error_detail:
                    self.logger.error(f"详细错误: {error_detail}")

        # 如果还有工具调用，递归处理
        if final_response.tool_calls:
            return self._handle_tool_calls(final_response)

        self.memory.add_message(self.session_id, "assistant", final_response.content)
        self.logger.response(final_response.content)

        return final_response.content

    def reset(self):
        """重置对话"""
        self.memory.clear_session(self.session_id)
        # 为每个用户生成独立的会话ID
        self.session_id = f"{self.config.user_id}_{str(uuid.uuid4())[:8]}"
        self.memory.create_session(self.session_id, self.config.context_window, self.config.user_id)
        self.logger.system(f"对话已重置, 新 session={self.session_id}")

    def get_history(self) -> List[Dict]:
        """获取对话历史"""
        return self.memory.get_messages(self.session_id)

    # ─── 异步流式对话 ───

    async def achat_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        """异步流式对话，yield 文本片段"""
        await self.memory.aadd_message(self.session_id, "user", user_input)
        await self.memory.atouch_session(self.session_id)
        self.logger.system(f"收到用户消息: {user_input[:100]}")

        if await self.memory.ashould_compress(self.session_id):
            self.logger.compress("上下文即将超限，启动记忆压缩...")
            await self.memory.acompress(self.session_id)
            self.logger.compress("压缩完成")
        elif await self.memory.ashould_compress_idle(self.session_id):
            self.logger.compress(f"闲置超过{self.config.memory_config.idle_compress_hours}h且未压缩过，启动记忆压缩...")
            await self.memory.acompress(self.session_id)
            self.logger.compress("压缩完成")

        messages = await asyncio.to_thread(self._build_messages)
        tools = self.tools.to_openai_format() or None

        async for chunk in self._astream_with_tool_handling(messages, tools):
            yield chunk

    async def _astream_with_tool_handling(self, messages: List, tools) -> AsyncGenerator[str, None]:
        """流式调用 LLM，检测 tool calls 并处理"""
        from .llm import LLMResponse

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

        if self.config.debug:
            self.logger.log(LogType.TOOL_CALL_REASON, f"tool_call_reason: {response.content}")

        for tc in response.tool_calls:
            self.logger.tool_call(tc.name, str(tc.arguments))
            result = await self.tools.aexecute(tc.name, tc.arguments)
            self.logger.tool_result(tc.name, result)
            await self.memory.aadd_message(self.session_id, "tool", result, tool_call_id=tc.id)

        messages = await asyncio.to_thread(self._build_messages)

        async for chunk in self._astream_with_tool_handling(messages, tools):
            yield chunk
