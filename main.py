"""
Light Agent - 命令行对话入口
--------------------------------
开箱即用:
  1. 首次启动若 api_key 缺失,会引导你填写 provider / api_key / model
  2. 之后的启动直接读 config/config.json
  3. 支持环境变量覆盖: {PROVIDER}_API_KEY (如 OPENAI_API_KEY / DEEPSEEK_API_KEY)

使用:
  python main.py                       # 默认 config/config.json
  python main.py my_config.json        # 指定配置
  python main.py --reconfig            # 强制重新走一遍引导

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

from src.llm import LLMFactory
from src.agent import Agent, AgentConfig
from src.memory import MemoryConfig
from src.tools import create_default_tools
from src.agent_logging import Logger, LogConfig, LogLevel, LogType
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


# ─── 首次使用引导 ────────────────────────────────────────
def _run_setup(config: Config):
    """交互式让用户填 provider / api_key / model,写回 config"""
    print(_color("\n━━━ 首次使用向导 ━━━", C_CYAN + C_BOLD))
    print("先填几项基本信息，之后会自动写入 config 文件。\n")

    providers = list(LLMFactory.PROVIDERS.keys())
    print(_color("可用 provider:", C_BOLD))
    for i, p in enumerate(providers, 1):
        info = LLMFactory.PROVIDERS[p]
        print(f"  {i}. {_color(p, C_GREEN)}  (默认模型: {info.get('default_model', '-')})")

    # 选 provider
    cur_provider = config.get("provider") or "openai"
    while True:
        raw = input(f"\n选择 provider (数字或名字，默认 {cur_provider}): ").strip()
        if not raw:
            provider = cur_provider
            break
        if raw.isdigit() and 1 <= int(raw) <= len(providers):
            provider = providers[int(raw) - 1]
            break
        if raw in providers:
            provider = raw
            break
        print(_color(f"  不认识 '{raw}',请重新输入。", C_YELLOW))

    config.set("provider", provider)
    default_model = LLMFactory.PROVIDERS[provider].get("default_model", "")

    # 填 api_key（环境变量优先提示）
    env_key_name = f"{provider.upper()}_API_KEY"
    env_key_val = os.environ.get(env_key_name, "")
    cur_key = config.get("api_key", "")

    print(f"\n{_color('API Key', C_BOLD)} (provider={provider})")
    if env_key_val:
        print(f"  检测到环境变量 {env_key_name},将优先使用;直接回车保持当前 config 值。")
    elif cur_key:
        print(f"  当前 config 中已配置 (长度 {len(cur_key)}),直接回车保留。")
    else:
        print(f"  可直接粘贴,或在终端设置环境变量: export {env_key_name}=xxx 再重启。")

    raw = input(f"  输入 api_key (回车跳过): ").strip()
    if raw:
        config.set("api_key", raw)

    # 填 model
    cur_model = config.get("model") or default_model
    print(f"\n{_color('Model', C_BOLD)} (默认 {default_model})")
    raw = input(f"  输入 model (回车用默认 '{cur_model}'): ").strip()
    config.set("model", raw or cur_model)

    # system_prompt 可选
    cur_prompt = config.get("system_prompt") or "你是一个有用的 AI 助手。"
    print(f"\n{_color('系统提示词 (system_prompt)', C_BOLD)}")
    print(f"  当前: {cur_prompt[:60]}{'...' if len(cur_prompt) > 60 else ''}")
    raw = input(f"  输入新的提示词 (回车保持不变): ").strip()
    if raw:
        config.set("system_prompt", raw)

    config.save()
    print(_color(f"\n✓ 配置已保存到 {config.config_path}\n", C_GREEN))


def _ensure_api_key(config: Config, interactive: bool = True) -> bool:
    """确保 api_key 可用;缺失且可交互就启动引导。返回是否可继续。"""
    # Config.api_key 会自动读环境变量,所以这里用它判断
    if config.api_key:
        return True

    if not interactive:
        print(_color(
            f"❌ 缺少 api_key。请设置环境变量 "
            f"{config.provider.upper()}_API_KEY,或在 {config.config_path} 里填写 api_key 字段。",
            C_RED,
        ))
        return False

    print(_color(f"⚠ 未检测到 {config.provider.upper()}_API_KEY 环境变量,"
                 f"且 config 里 api_key 为空。", C_YELLOW))
    _run_setup(config)
    return bool(config.api_key)


# ─── 创建 Agent ──────────────────────────────────────────
def _create_agent(config: Config) -> Agent:
    llm = LLMFactory.create(config.provider, config.api_key, config.model or None)

    agent_config = AgentConfig(
        name=config.get("name", "assistant"),
        system_prompt=config.get("system_prompt", "你是一个有用的 AI 助手。"),
        context_window=config.context_window,
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens", 4096),
        memory_config=MemoryConfig(
            db_path=config.get("memory_db", "memory.db"),
            compress_threshold=config.compress_threshold,
            keep_ratio=config.keep_ratio,
            idle_compress_hours=config.idle_compress_hours,
            dialect=config.get("dialect", "sqlite"),
        ),
        debug=config.get("debug", False),
        user_id=config.get("user_id", "default_user"),
        session_id=config.get("session_id", ""),
    )

    logger = Logger(LogConfig(
        level=LogLevel.DEBUG if config.get("debug", False) else LogLevel.INFO,
        colorize=config.get("colorize", True),
    ))
    # 不打印 response 日志（我们自己在终端渲染）
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
def _parse_args(argv):
    """返回 (config_path, reconfig_flag)"""
    config_path = "config/config.json"
    reconfig = False
    for arg in argv[1:]:
        if arg in ("--reconfig", "-r"):
            reconfig = True
        elif arg in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        elif not arg.startswith("-"):
            config_path = arg
    return config_path, reconfig


def main():
    config_path, reconfig = _parse_args(sys.argv)

    # Config 会在文件不存在时自动写入 DEFAULT_CONFIG
    config = Config(config_path)

    # 强制重新配置
    if reconfig:
        _run_setup(config)

    # 检查并引导 api_key
    if not _ensure_api_key(config):
        sys.exit(1)

    agent = _create_agent(config)
    try:
        asyncio.run(_chat_loop(agent, config))
    finally:
        agent.save_state()


if __name__ == "__main__":
    main()
