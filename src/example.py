"""
示例：自定义工具 + 多 Agent 路由
"""

from src.llm import LLMFactory
from src.agent import Agent, AgentConfig
from src.tools import create_default_tools
from src.router import Router, keyword_rule


def demo():
    # 创建 LLM
    llm = LLMFactory.create("openai", "your-api-key")

    # 注册自定义工具
    tools = create_default_tools()

    def search_web(query: str) -> str:
        """模拟搜索"""
        return f"搜索 '{query}' 的结果：这是一个示例结果"

    tools.register(
        name="search_web",
        description="搜索网页",
        parameters={"query": {"type": "string", "description": "搜索关键词"}},
        function=search_web
    )

    # 创建 Agent
    agent = Agent(llm, AgentConfig(name="main"), tools=tools)

    # 使用
    response = agent.chat("你好")


def demo_router():
    """多 Agent 路由示例"""
    llm = LLMFactory.create("openai", "your-api-key")

    router = Router()

    # 注册不同 Agent
    router.register("default", Agent(llm, AgentConfig(name="通用助手")))
    router.register("code", Agent(llm, AgentConfig(name="代码助手", system_prompt="你是编程专家。")))
    router.register("translate", Agent(llm, AgentConfig(name="翻译助手", system_prompt="你是翻译专家。")))

    # 添加路由规则
    router.add_rule(keyword_rule({
        "代码": "code",
        "编程": "code",
        "翻译": "translate",
    }))

    # 使用
    response = router.chat("帮我写一段 Python 代码")  # -> code agent


if __name__ == "__main__":
    demo()
