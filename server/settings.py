"""
server 层常量与环境变量读取
"""

import os

# 项目根目录 (light_agent/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 配置文件路径
CONFIG_PATH = os.getenv("LIGHT_AGENT_CONFIG", "config/config.json")

# 用户 / token 数据库路径
USER_DB_PATH = os.getenv("USER_DB_PATH", "data/users.db")

# Web 静态资源目录
WEB_DIR = os.path.join(PROJECT_ROOT, "web")

# Agent 空闲释放阈值（秒）
IDLE_TIMEOUT = int(os.getenv("LIGHT_AGENT_IDLE_TIMEOUT", "1800"))

# Token 有效期（秒）
TOKEN_TTL_SECONDS = int(os.getenv("LIGHT_AGENT_TOKEN_TTL", str(7 * 86400)))
