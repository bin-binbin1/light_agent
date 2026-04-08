"""
Config 模块 - 配置管理
支持读取、修改、保存配置
"""

import json
import os
from typing import Any, Optional
from pathlib import Path


DEFAULT_CONFIG = {
    "name": "assistant",
    "provider": "openai",
    "api_key": "",
    "model": "",
    "system_prompt": "你是一个有用的 AI 助手。",
    "context_window": 128000,
    "temperature": 0.7,
    "max_tokens": 4096,
    "memory_db": "memory.db",
    "compress_threshold": 0.5,
    "keep_ratio": 0.3,
    "idle_compress_hours": 6,
    "debug": False,
    "colorize": True,
    "user_id": "default_user",
    "session_id": "",
}


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
        # 优先环境变量
        env_key = f"{self.provider.upper()}_API_KEY"
        return os.environ.get(env_key) or self.get("api_key", "")

    @api_key.setter
    def api_key(self, value: str):
        self.set("api_key", value)

    @property
    def model(self) -> str:
        return self.get("model", "")

    @model.setter
    def model(self, value: str):
        self.set("model", value)

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
