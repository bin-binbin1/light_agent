"""
Tools 模块 - 工具注册与管理
OpenAI Function Calling 风格
支持装饰器+自动推断注册
"""

import json
import asyncio
import inspect
import importlib
import sys
import os
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from functools import wraps


def parse_docstring(func) -> tuple[str, Dict[str, str]]:
    """解析函数的docstring，提取描述和参数说明"""
    docstring = func.__doc__ or ""
    lines = [line.strip() for line in docstring.strip().split('\n')]

    # 提取函数描述（第一行或第一段）
    description = lines[0] if lines else ""

    # 提取参数描述
    param_descriptions = {}
    in_args_section = False

    for line in lines:
        if line.lower().startswith('args:'):
            in_args_section = True
            continue
        elif in_args_section and line and ':' in line:
            param_name, param_desc = line.split(':', 1)
            param_descriptions[param_name.strip()] = param_desc.strip()
        elif in_args_section and not line:
            # 遇到空行，退出参数解析
            break

    return description, param_descriptions


def infer_parameter_type(param) -> Dict[str, str]:
    """推断参数类型"""
    if param.annotation != inspect.Parameter.empty:
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object"
        }
        return {"type": type_map.get(param.annotation, "string")}
    return {"type": "string"}  # 默认类型


def auto_infer_tool_info(func) -> Dict[str, Any]:
    """自动推断工具信息"""

    # 1. 获取函数名
    name = func.__name__

    # 2. 解析docstring
    description, param_descriptions = parse_docstring(func)

    # 3. 分析函数签名
    signature = inspect.signature(func)
    parameters = {}
    required = []

    for param_name, param in signature.parameters.items():
        # 跳过 self 参数（如果是方法）
        if param_name == 'self':
            continue

        # 推断参数类型
        param_info = infer_parameter_type(param)

        # 从docstring获取参数描述
        param_desc = param_descriptions.get(param_name, "")
        param_info["description"] = param_desc

        parameters[param_name] = param_info

        # 检查是否必需参数
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "name": name,
        "description": description,
        "parameters": parameters,
        "required": required
    }


def tool(func=None, *, name: Optional[str] = None, description: Optional[str] = None,
         parameters: Optional[Dict[str, Any]] = None, required: Optional[List[str]] = None):
    """
    工具注册装饰器

    Args:
        func: 被装饰的函数
        name: 工具名称（可选，默认使用函数名）
        description: 工具描述（可选，默认使用docstring第一行）
        parameters: 参数定义（可选，默认自动推断）
        required: 必需参数列表（可选，默认自动推断）
    """
    def decorator(f):
        # 存储工具元数据
        f._tool_metadata = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "required": required
        }
        return f

    # 如果直接调用装饰器（无参数）
    if func is None:
        return decorator

    # 如果直接装饰函数
    return decorator(func)


def discover_tools_from_modules(module_names: List[str]) -> List[Callable]:
    """
    从指定模块中自动发现被 @tool 装饰的函数

    Args:
        module_names: 模块名称列表，如 ['src.tools', 'custom.tools']

    Returns:
        List[Callable]: 发现的工具函数列表
    """
    discovered_tools = []

    for module_name in module_names:
        try:
            # 导入模块
            module = importlib.import_module(module_name)

            # 遍历模块中的所有对象
            for attr_name in dir(module):
                # 跳过私有属性
                if attr_name.startswith('_'):
                    continue

                attr = getattr(module, attr_name)

                # 检查是否是函数且具有 _tool_metadata 属性
                if callable(attr) and hasattr(attr, '_tool_metadata'):
                    discovered_tools.append(attr)

        except ImportError as e:
            print(f"警告: 无法导入模块 {module_name}: {e}")
        except Exception as e:
            print(f"警告: 处理模块 {module_name} 时出错: {e}")

    return discovered_tools


def discover_tools_from_package(package_path: str) -> List[Callable]:
    """
    从指定包路径中递归发现被 @tool 装饰的函数

    Args:
        package_path: 包路径，如 'src' 或 'custom'

    Returns:
        List[Callable]: 发现的工具函数列表
    """
    discovered_tools = []

    # 将包路径添加到 sys.path（如果不在其中）
    if package_path not in sys.path:
        sys.path.insert(0, package_path)

    # 递归遍历包目录
    for root, dirs, files in os.walk(package_path):
        # 跳过 __pycache__ 等目录
        dirs[:] = [d for d in dirs if not d.startswith('__')]

        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                # 构建模块路径
                rel_path = os.path.relpath(root, package_path)
                if rel_path == '.':
                    module_name = file[:-3]  # 移除 .py
                else:
                    module_name = rel_path.replace(os.sep, '.') + '.' + file[:-3]

                # 构建完整模块名
                full_module_name = f"{package_path}.{module_name}"

                try:
                    # 导入模块
                    module = importlib.import_module(full_module_name)

                    # 发现工具函数
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if callable(attr) and hasattr(attr, '_tool_metadata'):
                            discovered_tools.append(attr)

                except ImportError as e:
                    print(f"警告: 无法导入模块 {full_module_name}: {e}")
                except Exception as e:
                    print(f"警告: 处理模块 {full_module_name} 时出错: {e}")

    return discovered_tools


@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema 格式
    function: Callable
    required: List[str] = field(default_factory=list)
    context_keys: List[str] = field(default_factory=list)  # 由 context 注入的参数名，不暴露给 LLM

    def to_openai_format(self) -> Dict:
        """转为 OpenAI function calling 格式（排除 context 注入的参数）"""
        properties = {k: v for k, v in self.parameters.items() if k not in self.context_keys}
        required = [r for r in self.required if r not in self.context_keys]
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

    def execute(self, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        """执行工具，自动注入 context 中匹配的参数"""
        try:
            if context:
                sig = inspect.signature(self.function)
                inject = {k: v for k, v in context.items() if k in sig.parameters}
                arguments = {**arguments, **inject}
            result = self.function(**arguments)
            if isinstance(result, (dict, list)):
                return json.dumps(result, ensure_ascii=False)
            return str(result)
        except Exception as e:
            return f"工具执行错误: {str(e)}"

    async def aexecute(self, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        """异步执行工具，支持 async 函数"""
        try:
            if context:
                sig = inspect.signature(self.function)
                inject = {k: v for k, v in context.items() if k in sig.parameters}
                arguments = {**arguments, **inject}
            if inspect.iscoroutinefunction(self.function):
                result = await self.function(**arguments)
            else:
                result = await asyncio.to_thread(self.function, **arguments)
            if isinstance(result, (dict, list)):
                return json.dumps(result, ensure_ascii=False)
            return str(result)
        except Exception as e:
            return f"工具执行错误: {str(e)}"


class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._context: Dict[str, Any] = {}

    def set_context(self, **kwargs) -> None:
        """设置注入上下文（如 user_id），工具执行时自动传入匹配签名的参数"""
        self._context.update(kwargs)
        # 更新已注册工具的 context_keys
        for t in self._tools.values():
            sig = inspect.signature(t.function)
            t.context_keys = [k for k in self._context if k in sig.parameters]

    def register(self, name: str, description: str, parameters: Dict[str, Any],
                 function: Callable, required: Optional[List[str]] = None):
        """注册工具（传统方式）"""
        tool = Tool(
            name=name,
            description=description,
            parameters=parameters,
            function=function,
            required=required or list(parameters.keys())
        )
        self._tools[name] = tool
        return tool

    def register_function(self, func: Callable) -> Tool:
        """
        注册函数（装饰器方式）

        Args:
            func: 被装饰的函数，包含 _tool_metadata 属性
        """
        # 获取工具元数据
        metadata = getattr(func, '_tool_metadata', {})

        # 自动推断工具信息
        inferred_info = auto_infer_tool_info(func)

        # 合并元数据（手动配置优先）
        name = metadata.get('name') or inferred_info['name']
        description = metadata.get('description') or inferred_info['description']
        parameters = metadata.get('parameters') or inferred_info['parameters']
        required = metadata.get('required') or inferred_info['required']

        # 注册工具
        tool = Tool(
            name=name,
            description=description,
            parameters=parameters,
            function=func,
            required=required,
            context_keys=[k for k in self._context if k in inspect.signature(func).parameters]
        )
        self._tools[name] = tool
        return tool

    def register_decorated(self, *functions: Callable) -> 'ToolRegistry':
        """
        批量注册装饰过的函数

        Args:
            functions: 一个或多个被装饰的函数
        """
        for func in functions:
            self.register_function(func)
        return self

    def register_from_modules(self, module_names: List[str]) -> 'ToolRegistry':
        """
        从指定模块中自动注册工具

        Args:
            module_names: 模块名称列表
        """
        tools = discover_tools_from_modules(module_names)
        for tool_func in tools:
            self.register_function(tool_func)
        return self

    def register_from_package(self, package_path: str) -> 'ToolRegistry':
        """
        从指定包路径中自动注册工具

        Args:
            package_path: 包路径
        """
        tools = discover_tools_from_package(package_path)
        for tool_func in tools:
            self.register_function(tool_func)
        return self

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
        return tool.execute(arguments, context=self._context)

    async def aexecute(self, name: str, arguments: Dict[str, Any]) -> str:
        """异步执行指定工具"""
        tool = self.get(name)
        if not tool:
            return f"工具 '{name}' 不存在"
        return await tool.aexecute(arguments, context=self._context)


def create_default_tools(memory=None) -> ToolRegistry:
    """创建默认工具集"""
    registry = ToolRegistry()

    # 计算器 - 使用装饰器方式
    @tool
    def calculator(expression: str) -> str:
        """安全的数学计算

        Args:
            expression: 数学表达式，如 2+3*4
        """
        try:
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return "包含不允许的字符"
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"计算错误: {e}"

    # 当前时间 - 使用装饰器方式
    @tool
    def get_time() -> str:
        """获取当前时间"""
        import datetime
        return datetime.datetime.now().isoformat()

    # 记忆检索 - 使用装饰器方式
    if memory:
        @tool
        def search_memory(query: str, top_k: int = 5) -> str:
            """搜索历史对话记忆

            Args:
                query: 搜索关键词
                top_k: 返回条数，默认5
            """
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

        # 批量注册装饰过的函数
        registry.register_decorated(calculator, get_time, search_memory)
    else:
        registry.register_decorated(calculator, get_time)

    return registry


if __name__ == "__main__":
    # 简单测试装饰器功能
    registry = ToolRegistry()

    @tool
    def test_function(param1: str) -> str:
        """测试函数"""
        return param1

    registry.register_decorated(test_function)
    print("装饰器功能测试通过")
