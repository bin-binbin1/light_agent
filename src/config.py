"""
Config 模块 - 配置管理
支持读取、修改、保存；并提供 create_agent_from_config() 一步创建 Agent
"""

import json
import os
from typing import Any, Optional, List
from pathlib import Path


DEFAULT_CONFIG = {
    # —— 基本身份 ——
    "name": "assistant",
    "provider": "openai",
    "api_key": "",
    "model": "",
    "base_url": "",              # OpenAI-兼容端点；自建/厂商专用时必填
    "vision_model": "",          # 多模态模型名（可选）
    "tts_model": "",             # TTS 模型名（可选）
    "tts_voice": "",             # TTS 默认音色（可选）

    # —— 对话行为 ——
    "system_prompt": "你是一个有用的 AI 助手。",
    "context_window": 128000,
    "temperature": 0.7,
    "max_tokens": 4096,

    # —— 记忆 ——
    "memory_db": "memory.db",
    "dialect": "sqlite",         # 记忆存储方言：sqlite / mysql
    "compress_threshold": 0.5,
    "keep_ratio": 0.3,
    "idle_compress_hours": 6,

    # —— 日志 / 会话 ——
    "debug": False,
    "colorize": True,
    "user_id": "default_user",
    "session_id": "",
}


# ─── 每个 provider 对应的环境变量名列表（优先级从前往后） ──
ENV_KEY_NAMES = {
    "xiaomi": ["MIFY_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "kimi": ["KIMI_API_KEY", "MOONSHOT_API_KEY"],
    "minimax": ["MINIMAX_API_KEY"],
    "grok": ["GROK_API_KEY", "XAI_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
}


def _get_env_key(provider: str) -> str:
    for name in ENV_KEY_NAMES.get(provider, [f"{provider.upper()}_API_KEY"]):
        v = os.environ.get(name)
        if v:
            return v
    return ""


class Config:
    """配置管理"""

    def __init__(self, config_path: str = "config/config.json"):
        self.config_path = config_path
        self._data: dict = {}
        self.load()

    def load(self):
        """加载配置"""
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            self._data = DEFAULT_CONFIG.copy()
            self.save()

    def save(self):
        """保存配置"""
        os.makedirs(os.path.dirname(self.config_path) or ".", exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=4, ensure_ascii=False)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any):
        self._data[key] = value

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self._data[k] = v

    @property
    def data(self) -> dict:
        return self._data.copy()

    # 快捷属性

    @property
    def provider(self) -> str:
        return self.get("provider", "openai")

    @property
    def api_key(self) -> str:
        """优先环境变量 (按 provider 查 ENV_KEY_NAMES)，其次 config 文件"""
        return _get_env_key(self.provider) or self.get("api_key", "")

    @api_key.setter
    def api_key(self, value: str):
        self.set("api_key", value)

    @property
    def env_key_names(self) -> List[str]:
        """当前 provider 支持的环境变量名列表，供错误提示使用"""
        return ENV_KEY_NAMES.get(
            self.provider, [f"{self.provider.upper()}_API_KEY"]
        )

    @property
    def model(self) -> str:
        return self.get("model", "")

    @model.setter
    def model(self, value: str):
        self.set("model", value)

    @property
    def base_url(self) -> str:
        return self.get("base_url", "")

    @base_url.setter
    def base_url(self, value: str):
        self.set("base_url", value)

    @property
    def vision_model(self) -> str:
        return self.get("vision_model", "")

    @property
    def dialect(self) -> str:
        return self.get("dialect", "sqlite")

    @property
    def debug(self) -> bool:
        return bool(self.get("debug", False))

    @property
    def context_window(self) -> int:
        return self.get("context_window", 128000)

    @context_window.setter
    def context_window(self, value: int):
        self.set("context_window", value)

    @property
    def compress_threshold(self) -> float:
        return self.get("compress_threshold", 0.5)

    @compress_threshold.setter
    def compress_threshold(self, value: float):
        self.set("compress_threshold", value)

    @property
    def keep_ratio(self) -> float:
        return self.get("keep_ratio", 0.3)

    @keep_ratio.setter
    def keep_ratio(self, value: float):
        self.set("keep_ratio", value)

    @property
    def idle_compress_hours(self) -> float:
        return self.get("idle_compress_hours", 6)

    @idle_compress_hours.setter
    def idle_compress_hours(self, value: float):
        self.set("idle_compress_hours", value)

    def __repr__(self):
        safe = {k: ("***" if k == "api_key" and v else v) for k, v in self._data.items()}
        return f"Config({safe})"

    # ─── 一键创建 Agent ───
    def create_agent_from_config(self,
                                 resume: bool = True,
                                 session_id: str = "",
                                 user_id: str = "",
                                 tools=None,
                                 logger=None):
        """按当前配置创建一个 Agent。

        Args:
            resume: True 时自动恢复该 user 最近的 session
            session_id: 指定 session_id（优先级最高）
            user_id: 覆盖 config 的 user_id
            tools: 自定义 ToolRegistry（为空则 create_default_tools()）
            logger: 自定义 Logger（为空则按 config.debug/colorize 创建）

        Raises:
            RuntimeError: model 或 api_key 缺失
        """
        # 延迟导入，避免循环依赖 & 让 Config 在不需要 Agent 时也能用
        from .llm import LLMFactory, OpenAICompatibleLLM, LLMType
        from .agent import Agent, AgentConfig
        from .memory import Memory, MemoryConfig
        from .tools import create_default_tools
        from .agent_logging import Logger, LogConfig, LogLevel

        if not self.model:
            raise RuntimeError(
                f"缺少 model，请在 {self.config_path} 里填写 model 字段。"
            )
        api_key = self.api_key
        if not api_key:
            raise RuntimeError(
                f"缺少 api_key，请设置环境变量 {'/'.join(self.env_key_names)} "
                f"或在 {self.config_path} 里填写 api_key 字段。"
            )

        # LLM：自建 base_url 或未知 provider 走 OpenAICompatibleLLM，否则走 LLMFactory
        if self.base_url or self.provider not in LLMFactory.PROVIDERS:
            caps = [LLMType.TEXT]
            if self.vision_model:
                caps.append(LLMType.VISION)
            llm = OpenAICompatibleLLM(
                api_key=api_key,
                model=self.model,
                base_url=self.base_url or "https://api.openai.com/v1",
                vision_model=self.vision_model,
                capabilities=caps,
            )
        else:
            llm = LLMFactory.create(self.provider, api_key, self.model or None)

        # user_id / session_id 解析
        if not user_id:
            user_id = self.get("user_id", "default_user")
        db_path = self.get("memory_db", "memory.db")

        if not session_id and resume:
            mem = Memory(MemoryConfig(db_path=db_path, dialect=self.dialect))
            session_id = mem.get_latest_session(user_id) or ""
            mem.close()

        agent_config = AgentConfig(
            name=self.get("name", "assistant"),
            system_prompt=self.get("system_prompt", "你是一个有用的 AI 助手。"),
            context_window=self.context_window,
            temperature=self.get("temperature", 0.7),
            max_tokens=self.get("max_tokens", 4096),
            memory_config=MemoryConfig(
                db_path=db_path,
                compress_threshold=self.compress_threshold,
                keep_ratio=self.keep_ratio,
                idle_compress_hours=self.idle_compress_hours,
                dialect=self.dialect,
            ),
            debug=self.debug,
            user_id=user_id,
            session_id=session_id,
        )

        if logger is None:
            logger = Logger(LogConfig(
                level=LogLevel.DEBUG if self.debug else LogLevel.INFO,
                colorize=self.get("colorize", True),
            ))

        if tools is None:
            tools = create_default_tools()

        return Agent(llm=llm, config=agent_config, tools=tools, logger=logger)


def config_cli():
    """命令行配置工具"""
    import sys

    config = Config()

    if len(sys.argv) < 2:
        print("用法:")
        print("  python3 config_cli.py show              # 显示配置")
        print("  python3 config_cli.py get <key>         # 获取值")
        print("  python3 config_cli.py set <key> <value> # 设置值")
        print("  python3 config_cli.py provider <name>   # 切换厂商")
        print("  python3 config_cli.py api_key <key>     # 设置 API Key")
        print()
        print("当前配置:")
        print(config)
        return

    cmd = sys.argv[1]

    if cmd == "show":
        for k, v in config.data.items():
            if k == "api_key" and v:
                v = v[:8] + "..." + v[-4:]
            print(f"  {k}: {v}")

    elif cmd == "get" and len(sys.argv) >= 3:
        key = sys.argv[2]
        print(f"{key}: {config.get(key)}")

    elif cmd == "set" and len(sys.argv) >= 4:
        key = sys.argv[2]
        value = sys.argv[3]

        # 类型转换
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif "." in value:
            try:
                value = float(value)
            except ValueError:
                pass
        else:
            try:
                value = int(value)
            except ValueError:
                pass

        config.set(key, value)
        config.save()
        print(f"✅ {key} = {value}")

    elif cmd == "provider" and len(sys.argv) >= 3:
        config.set("provider", sys.argv[2])
        config.save()
        print(f"✅ provider = {sys.argv[2]}")

    elif cmd == "api_key" and len(sys.argv) >= 3:
        config.api_key = sys.argv[2]
        config.save()
        print("✅ api_key 已更新")

    else:
        print(f"未知命令: {cmd}")


if __name__ == "__main__":
    config_cli()
