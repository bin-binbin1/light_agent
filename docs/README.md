# Light Agent - 轻量化 Agent 框架

面向超长文本对话优化的 Python Agent 框架。

## 架构

```
light_agent/
├── main.py           # 命令行入口，可被外部调用
├── src/
│   ├── __init__.py
│   ├── llm.py        # LLM 统一接口（DeepSeek/Kimi/OpenAI/MiniMax/Grok）
│   ├── agent.py      # Agent 对话管理核心
│   ├── router.py     # 多 Agent 路由
│   ├── memory.py     # 记忆管理（SQLite，自动精简）
│   ├── tools.py      # 工具注册与调用（OpenAI Function Calling 风格）
│   ├── prompt.py     # 提示词管理
│   ├── logging.py    # 格式化日志（思考/工具调用/回复）
│   └── example.py    # 使用示例
├── tests/
│   ├── test_memory.py
│   └── test_tools.py
├── config/
│   └── config.json   # 配置文件模板
├── docs/
└── README.md
```

## 快速开始

### 1. 配置

编辑 `config/config.json`：

```json
{
    "provider": "openai",
    "api_key": "your-api-key",
    "model": "gpt-4o",
    "system_prompt": "你是一个有用的 AI 助手。",
    "context_window": 128000,
    "compress_threshold": 0.5,
    "keep_ratio": 0.3
}
```

### 2. 命令行运行

```bash
python3 main.py
```

### 3. 作为库调用

```python
from src.llm import LLMFactory
from src.agent import Agent, AgentConfig

llm = LLMFactory.create("deepseek", "your-api-key")
agent = Agent(llm, AgentConfig(name="my-agent"))

response = agent.chat("你好")
print(response)
```

## 核心模块

### LLM 统一接口 (llm.py)

支持 6 家厂商，统一 OpenAI 兼容格式：
- OpenAI
- DeepSeek
- Kimi (Moonshot)
- MiniMax
- Grok (xAI)
- OpenRouter

```python
from src.llm import LLMFactory

llm = LLMFactory.create("deepseek", "api-key", model="deepseek-chat")
response = llm.chat([{"role": "user", "content": "你好"}])
```

### Agent (agent.py)

对话管理核心，集成记忆 + 工具 + LLM。

```python
agent = Agent(llm, AgentConfig(name="bot", context_window=128000))
response = agent.chat("今天天气怎么样？")
```

### 记忆 (memory.py)

- SQLite 存储
- 上下文使用率 > 50% 时自动触发精简
- 精简策略：保留最近 30% 原文，其余提炼为摘要
- 支持多会话管理

### 工具 (tools.py)

OpenAI Function Calling 风格：

```python
from src.tools import ToolRegistry

registry = ToolRegistry()

def get_weather(city: str) -> str:
    return f"{city}: 晴, 25°C"

registry.register(
    name="get_weather",
    description="获取天气",
    parameters={"city": {"type": "string", "description": "城市名"}},
    function=get_weather
)

agent = Agent(llm, tools=registry)
```

### Router (router.py)

多 Agent 路由，支持关键词规则和意图识别：

```python
from src.router import Router, keyword_rule

router = Router()
router.register("default", general_agent)
router.register("code", code_agent)

router.add_rule(keyword_rule({"代码": "code", "编程": "code"}))

response = router.chat("帮我写段代码")  # -> code_agent
```

### 日志 (logging.py)

格式化输出 AI 的思考和操作过程：

```
14:30:01 | 💭 THINKING | 用户问了天气问题
14:30:02 | 🔧 TOOL_CALL | get_weather({"city": "北京"})
14:30:03 | 📦 TOOL_RESULT | get_weather → 北京: 晴, 25°C
14:30:04 | 💬 RESPONSE | 北京今天天气晴朗，25°C
```

## 配置项说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| provider | LLM 厂商 | openai |
| api_key | API 密钥 | - |
| model | 模型名称 | 厂商默认 |
| context_window | 上下文窗口大小 | 128000 |
| compress_threshold | 压缩触发阈值 | 0.5 |
| keep_ratio | 压缩后保留比例 | 0.3 |
| temperature | 生成温度 | 0.7 |
| max_tokens | 最大输出长度 | 4096 |
| debug | 调试模式 | false |
