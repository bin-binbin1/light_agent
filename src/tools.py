"""
Tools 模块 - 工具注册与管理
OpenAI Function Calling 风格
"""

import json
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema 格式
    function: Callable
    required: List[str] = field(default_factory=list)

    def to_openai_format(self) -> Dict:
        """转为 OpenAI function calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required
                }
            }
        }

    def execute(self, arguments: Dict[str, Any]) -> str:
        """执行工具"""
        try:
            result = self.function(**arguments)
            if isinstance(result, (dict, list)):
                return json.dumps(result, ensure_ascii=False)
            return str(result)
        except Exception as e:
            return f"工具执行错误: {str(e)}"


class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, name: str, description: str, parameters: Dict[str, Any],
                 function: Callable, required: Optional[List[str]] = None):
        """注册工具"""
        tool = Tool(
            name=name,
            description=description,
            parameters=parameters,
            function=function,
            required=required or list(parameters.keys())
        )
        self._tools[name] = tool
        return tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def to_openai_format(self) -> List[Dict]:
        """获取 OpenAI 格式的工具列表"""
        return [tool.to_openai_format() for tool in self._tools.values()]

    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        """执行指定工具"""
        tool = self.get(name)
        if not tool:
            return f"工具 '{name}' 不存在"
        return tool.execute(arguments)


def create_default_tools(memory=None) -> ToolRegistry:
    """创建默认工具集"""
    registry = ToolRegistry()

    # 计算器
    def calculator(expression: str) -> str:
        """安全的数学计算"""
        try:
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return "包含不允许的字符"
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"计算错误: {e}"

    registry.register(
        name="calculator",
        description="执行数学计算，输入数学表达式",
        parameters={"expression": {"type": "string", "description": "数学表达式，如 2+3*4"}},
        function=calculator
    )

    # 当前时间
    def get_time() -> str:
        import datetime
        return datetime.datetime.now().isoformat()

    registry.register(
        name="get_time",
        description="获取当前时间",
        parameters={},
        function=get_time,
        required=[]
    )

    # 记忆检索
    if memory:
        def search_memory(query: str, top_k: int = 5) -> str:
            """搜索历史对话记忆。当用户问到久远的具体细节时使用。"""
            results = memory.search_messages(memory.get_current_session() or "", query, top_k=top_k)
            if not results:
                return "未找到相关历史记录"

            lines = [f"找到 {len(results)} 条相关记录："]
            for r in results:
                time_str = r.get("time_str", "未知时间")
                role = r["role"]
                content = r["content"][:300]
                lines.append(f"\n[{time_str}] {role}: {content}")

            return "\n".join(lines)

        registry.register(
            name="search_memory",
            description="搜索历史对话记忆。当用户询问过去聊过的具体细节、结论、数据时使用此工具。",
            parameters={
                "query": {"type": "string", "description": "搜索关键词"},
                "top_k": {"type": "integer", "description": "返回条数，默认5"}
            },
            function=search_memory,
            required=["query"]
        )

    return registry
