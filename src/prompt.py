"""
Prompt 模块 - 提示词管理
支持系统提示词、精简压缩
"""


class PromptTemplate:
    """提示词模板"""

    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)


# 内置提示词
DEFAULT_SYSTEM_PROMPT = """你是一个有用的 AI 助手。你可以：
1. 回答用户问题
2. 调用工具完成任务
3. 保持对话连贯性

当需要调用工具时，请直接调用，不要询问用户。
如果工具返回错误，请尝试其他方法或告知用户。"""

TOOL_CALLING_PROMPT = """你有以下工具可用：
{tool_descriptions}

使用工具时，请按照以下格式调用：
- 直接使用 function calling，不要手动编写 JSON
- 工具参数必须符合定义的 schema

工具执行结果会自动返回给你。"""

COMPRESS_PROMPT = """请将以下对话提炼为简洁的摘要。
要求：
1. 保留关键信息、决策和结论
2. 忽略闲聊和重复内容
3. 使用中文
4. 摘要长度不超过原文的 30%

对话内容：
{conversation}"""


class PromptManager:
    """提示词管理器"""

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.system_prompt = system_prompt

    def build_system_message(self, tools_description: str = "") -> str:
        """构建完整系统提示词"""
        prompt = self.system_prompt
        if tools_description:
            prompt += "\n\n" + TOOL_CALLING_PROMPT.format(tool_descriptions=tools_description)
        return prompt

    def format_tool_descriptions(self, tools) -> str:
        """格式化工具描述"""
        if not tools:
            return ""
        lines = []
        for tool in tools:
            func = tool["function"]
            lines.append(f"- {func['name']}: {func['description']}")
        return "\n".join(lines)

    def compress_prompt(self, conversation: str) -> str:
        """生成压缩提示词"""
        return COMPRESS_PROMPT.format(conversation=conversation)

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt
