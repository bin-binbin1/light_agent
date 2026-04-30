"""
Light Agent - 命令行对话入口
--------------------------------
配置:
  1. 编辑 config/config.json,填写 provider / model / 可选 api_key
  2. api_key 建议用环境变量 (如 OPENAI_API_KEY / MIFY_KEY),也可直接写在 config 里
  3. 缺少 model 或 api_key 时会打印清楚的错误并退出

使用:
  python main.py                       # 默认 config/config.json
  python main.py my_config.json        # 指定配置

对话中的命令:
  /help    查看命令
  /reset   重置对话
  /history 最近历史
  /config  查看当前配置
  /quit    退出
"""

import sys
import os
import json
import asyncio
from typing import Optional

# 让 `from src.xxx` 能找到 light_agent/src
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

# 加载 .env（就近找 light_agent/.env，其次项目根 .env）
try:
    from dotenv import load_dotenv
    for _p in (os.path.join(_here, ".env"),
               os.path.join(os.path.dirname(_here), ".env")):
        if os.path.exists(_p):
            load_dotenv(_p)
            break
except ImportError:
    pass  # python-dotenv 是可选依赖

from src.agent import Agent
from src.agent_logging import LogType
from src.config import Config
from src.events import AgentEvent, ThinkingEvent, ToolCallEvent, ToolResultEvent, RetryEvent, ErrorEvent


# ─── 终端颜色（ANSI,Windows 现代终端都支持）──
C_DIM = "\033[2m"
C_BOLD = "\033[1m"
C_CYAN = "\033[36m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_RED = "\033[31m"
C_RESET = "\033[0m"


def _color(s: str, c: str) -> str:
    return f"{c}{s}{C_RESET}"




# ─── 异步流式对话循环 ────────────────────────────────────
async def _chat_loop(agent: Agent, config: Config):
    print()
    print(_color("━" * 60, C_DIM))
    print(_color(f"🤖 Light Agent [{agent.config.name}]  已启动", C_BOLD + C_CYAN))
    print(f"   provider = {_color(config.provider, C_GREEN)}   "
          f"model = {_color(config.model or '(default)', C_GREEN)}")
    print(f"   session  = {_color(agent.session_id, C_DIM)}")
    print(_color("   输入 /help 查看命令,/quit 退出", C_DIM))
    print(_color("━" * 60, C_DIM))

    while True:
        try:
            user_input = await asyncio.to_thread(input, _color("\n👤 你: ", C_BOLD))
            user_input = user_input.strip()
        except (EOFError, KeyboardInterrupt):
            print(_color("\n👋 再见", C_DIM))
            break

        if not user_input:
            continue

        # 命令处理
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit"):
                print(_color("👋 再见", C_DIM))
                break
            if cmd == "/help":
                print(_color(
                    "  /reset   重置对话\n"
                    "  /history 最近历史\n"
                    "  /config  查看当前配置\n"
                    "  /quit    退出",
                    C_DIM,
                ))
                continue
            if cmd == "/reset":
                agent.reset()
                print(_color(f"🔄 对话已重置  new session = {agent.session_id}", C_GREEN))
                continue
            if cmd == "/history":
                history = agent.get_history()
                if not history:
                    print(_color("  (无历史)", C_DIM))
                for msg in history[-10:]:
                    role = msg.get("role", "")
                    content = (msg.get("content") or "")[:80]
                    print(f"  [{role}] {content}")
                continue
            if cmd == "/config":
                for k, v in config.data.items():
                    if k == "api_key" and v:
                        v = f"{str(v)[:6]}...{str(v)[-4:]}"
                    print(f"  {k}: {v}")
                continue
            print(_color(f"未知命令: {user_input}   输入 /help", C_YELLOW))
            continue

        # 流式对话
        print(_color("\n🤖 ", C_CYAN), end="", flush=True)
        try:
            async for chunk in agent.achat_stream(user_input):
                if isinstance(chunk, ThinkingEvent):
                    # 可选: 打印一个小提示
                    pass
                elif isinstance(chunk, ToolCallEvent):
                    print(_color(f"\n🔧 {chunk.display or chunk.name}", C_YELLOW),
                          end="", flush=True)
                elif isinstance(chunk, ToolResultEvent):
                    ok = "✓" if chunk.success else "✗"
                    sec = chunk.duration_ms / 1000
                    print(_color(f"  {ok} {chunk.display or ''} ({sec:.2f}s)", C_DIM))
                    print(_color("🤖 ", C_CYAN), end="", flush=True)
                elif isinstance(chunk, RetryEvent):
                    print(_color(
                        f"\n⚠ {chunk.reason},{chunk.wait_seconds:.1f}s 后重试 "
                        f"({chunk.attempt}/{chunk.max_attempts})",
                        C_YELLOW,
                    ))
                elif isinstance(chunk, ErrorEvent):
                    print(_color(f"\n❌ {chunk.message}", C_RED))
                elif isinstance(chunk, AgentEvent):
                    # 未知事件,忽略
                    pass
                else:
                    # 文本片段
                    print(chunk, end="", flush=True)
            print()
        except Exception as e:
            print(_color(f"\n❌ 错误: {e}", C_RED))
            if config.get("debug", False):
                import traceback
                traceback.print_exc()


# ─── 入口 ────────────────────────────────────────────────
def _parse_args(argv) -> str:
    """返回 config_path"""
    config_path = "config/config.json"
    for arg in argv[1:]:
        if arg in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        elif not arg.startswith("-"):
            config_path = arg
    return config_path


def main():
    config_path = _parse_args(sys.argv)

    # Config 会在文件不存在时自动写入 DEFAULT_CONFIG
    config = Config(config_path)

    # 创建 Agent（缺 model / api_key 时会抛 RuntimeError）
    try:
        agent = config.create_agent_from_config()
    except RuntimeError as e:
        print(_color(f"❌ {e}", C_RED))
        sys.exit(1)

    # 不打印 response 日志（由终端自行流式渲染）
    agent.logger.disable(LogType.RESPONSE)

    try:
        asyncio.run(_chat_loop(agent, config))
    finally:
        agent.save_state()


if __name__ == "__main__":
    main()
