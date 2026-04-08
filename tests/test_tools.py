"""Tools 模块测试"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools import ToolRegistry, create_default_tools


def test_tool_registry():
    registry = ToolRegistry()

    def add(a: int, b: int) -> int:
        return a + b

    registry.register(
        name="add",
        description="两数相加",
        parameters={
            "a": {"type": "integer", "description": "第一个数"},
            "b": {"type": "integer", "description": "第二个数"}
        },
        function=add
    )

    assert "add" in registry.list_tools()

    result = registry.execute("add", {"a": 3, "b": 5})
    assert result == "8"

    # OpenAI 格式
    fmt = registry.to_openai_format()
    assert len(fmt) == 1
    assert fmt[0]["function"]["name"] == "add"

    print("✅ test_tool_registry passed")


def test_default_tools():
    registry = create_default_tools()

    assert "calculator" in registry.list_tools()
    assert "get_time" in registry.list_tools()

    result = registry.execute("calculator", {"expression": "2 + 3 * 4"})
    assert result == "14"

    print("✅ test_default_tools passed")


if __name__ == "__main__":
    test_tool_registry()
    test_default_tools()
    print("\n全部测试通过 ✅")
