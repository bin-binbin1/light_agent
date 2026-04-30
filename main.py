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

from src.llm import LLMFactory, OpenAICompatibleLLM, LLMType
from src.agent import Agent, AgentConfig
from src.memory import Memory, MemoryConfig
from src.tools import create_default_tools
from src.agent_logging import Logger, LogConfig, LogLevel, LogType
from src.config import Config
from src.events import AgentEvent, ThinkingEvent, ToolCallEvent, ToolResultEvent, RetryEvent, ErrorEvent


# ─── 环境变量候选列表（按 provider 查） ──
_ENV_KEY_NAMES = {
    "xiaomi": ["MIFY_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "kimi": ["KIMI_API_KEY", "MOONSHOT_API_KEY"],
    "minimax": ["MINIMAX_API_KEY"],
    "grok": ["GROK_API_KEY", "XAI_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
}


def _get_env_key(provider: str) -> str:
    """按 provider 查环境变量里是否有 key"""
    for name in _ENV_KEY_NAMES.get(provider, [f"{provider.upper()}_API_KEY"]):
        v = os.environ.get(name)
        if v:
            return v
    return ""


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


def _resolve_api_key(config: Config) -> str:
    """key 优先级: 环境变量 > config.api_key"""
    return _get_env_key(config.provider) or config.get("api_key", "")


def _check_config(config: Config) -> bool:
    """启动前检查 config 必填项，有问题就打印清楚的错误并返回 False"""
    ok = True

    # model 必填
    if not config.model:
        print(_color(
            f"❌ 缺少 model。请在 {config.config_path} 里填写 model 字段。",
            C_RED,
        ))
        ok = False

    # api_key: 环境变量 or config 二选一
    if not _resolve_api_key(config):
        env_names = _ENV_KEY_NAMES.get(
            config.provider, [f"{config.provider.upper()}_API_KEY"]
        )
        print(_color(
            f"❌ 缺少 api_key。请设置环境变量 {'/'.join(env_names)}，"
            f"或在 {config.config_path} 里填写 api_key 字段。",
            C_RED,
        ))
        ok = False

    return ok


# ─── 创建 Agent ──────────────────────────────────────────
def create_agent(config: Config,
                 resume: bool = True,
                 session_id: str = "",
                 user_id: str = "") -> Agent:
    """根据 config 创建 Agent（仿照 src/my_agent.py 的 create_my_agent）

    Args:
        config: 已加载的 Config 对象
        resume: True 时自动恢复该用户最近的 session
        session_id: 指定 session_id（优先级最高）
        user_id: 覆盖 config 的 user_id
    """
    # key: 环境变量 > config
    api_key = _resolve_api_key(config)
    if not api_key:
        raise RuntimeError(
            f"api_key 为空。请设置环境变量 "
            f"{'/'.join(_ENV_KEY_NAMES.get(config.provider, [config.provider.upper() + '_API_KEY']))} "
            f"或在 {config.config_path} 里填 api_key。"
        )

    # LLM：如果 config 里给了 base_url，就直接走 OpenAICompatibleLLM；否则用 LLMFactory
    base_url = config.get("base_url", "")
    if base_url or config.provider not in LLMFactory.PROVIDERS:
        llm = OpenAICompatibleLLM(
            api_key=api_key,
            model=config.model,
            base_url=base_url or "https://api.openai.com/v1",
            vision_model=config.get("vision_model", ""),
            capabilities=[LLMType.TEXT, LLMType.VISION] if config.get("vision_model") else [LLMType.TEXT],
        )
    else:
        llm = LLMFactory.create(config.provider, api_key, config.model or None)

    # user_id / session_id 解析
    if not user_id:
        user_id = config.get("user_id", "default_user")
    db_path = config.get("memory_db", "memory.db")

    if not session_id and resume:
        mem = Memory(MemoryConfig(db_path=db_path, dialect=config.get("dialect", "sqlite")))
        session_id = mem.get_latest_session(user_id) or ""
        mem.close()
        if session_id:
            print(_color(f"[RESUME] 恢复历史会话: {session_id}", C_DIM))

    agent_config = AgentConfig(
        name=config.get("name", "assistant"),
        system_prompt=config.get("system_prompt", "你是一个有用的 AI 助手。"),
        context_window=config.context_window,
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens", 4096),
        memory_config=MemoryConfig(
            db_path=db_path,
            compress_threshold=config.compress_threshold,
            keep_ratio=config.keep_ratio,
            idle_compress_hours=config.idle_compress_hours,
            dialect=config.get("dialect", "sqlite"),
        ),
        debug=config.get("debug", False),
        user_id=user_id,
        session_id=session_id,
    )

    logger = Logger(LogConfig(
        level=LogLevel.DEBUG if config.get("debug", False) else LogLevel.INFO,
        colorize=config.get("colorize", True),
    ))
    # 不打印 response 日志（终端里我们自己流式渲染）
    logger.disable(LogType.RESPONSE)

    tools = create_default_tools()
    return Agent(llm=llm, config=agent_config, tools=tools, logger=logger)


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

    # 必填项校验：缺 model 或 api_key 就直接退出并提示
    if not _check_config(config):
        sys.exit(1)

    agent = create_agent(config)
    try:
        asyncio.run(_chat_loop(agent, config))
    finally:
        agent.save_state()


if __name__ == "__main__":
    main()
