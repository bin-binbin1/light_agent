"""
Main - 命令行对话接口
可被外部调用，作为对话入口
"""

import sys
import os
import json
import readline  # 支持上下键历史
from typing import Optional

from src.llm import LLMFactory, BaseLLM
from src.agent import Agent, AgentConfig
from src.memory import MemoryConfig
from src.tools import ToolRegistry, create_default_tools
from src.logging import Logger, LogConfig, LogLevel
from src.config import Config


def create_agent_from_config(config: Config) -> Agent:
    """从配置创建 Agent"""
    api_key = config.api_key
    if not api_key:
        print("❌ 请设置 api_key:")
        print("  python3 -m src.config set api_key <your-key>")
        print("  或设置环境变量: export OPENAI_API_KEY=xxx")
        sys.exit(1)

    llm = LLMFactory.create(config.provider, api_key, config.model or None)

    agent_config = AgentConfig(
        name=config.get("name", "assistant"),
        system_prompt=config.get("system_prompt", ""),
        context_window=config.context_window,
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens", 4096),
        memory_config=MemoryConfig(
            db_path=config.get("memory_db", "memory.db"),
            compress_threshold=config.compress_threshold,
            keep_ratio=config.keep_ratio,
            idle_compress_hours=config.idle_compress_hours,
        )
    )

    log_config = LogConfig(
        level=LogLevel.DEBUG if config.get("debug", False) else LogLevel.INFO,
        colorize=config.get("colorize", True),
    )
    logger = Logger(log_config)

    return Agent(llm, agent_config, logger=logger)


def interactive_chat(agent: Agent):
    """交互式对话循环"""
    print(f"\n🤖 Light Agent [{agent.config.name}] 已启动")
    print("   输入消息开始对话")
    print("   /quit  退出")
    print("   /reset 重置对话")
    print("   /history 查看历史")
    print("-" * 50)

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("👋 再见")
            break

        if user_input == "/reset":
            agent.reset()
            print("🔄 对话已重置")
            continue

        if user_input == "/history":
            history = agent.get_history()
            for msg in history[-10:]:
                role = msg["role"]
                content = msg.get("content", "")[:80]
                print(f"  [{role}]: {content}")
            continue

        try:
            response = agent.chat(user_input)
            print(f"\n🤖 Agent: {response}")
        except Exception as e:
            print(f"\n❌ 错误: {e}")


def main():
    """主入口"""
    config_path = "config/config.json"
    if len(sys.argv) > 1 and not sys.argv[1].startswith("/"):
        config_path = sys.argv[1]

    config = Config(config_path)
    agent = create_agent_from_config(config)
    interactive_chat(agent)


if __name__ == "__main__":
    main()
