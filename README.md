# Light Agent

**轻量化 Python Agent 框架，专为超长对话场景设计。**

零依赖向量数据库，零依赖重型框架。仅凭 `requests` + `flask` + Python 标准库，实现完整的 AI Agent 能力栈：多模型接入、工具调用、持久记忆、自动压缩、多用户会话隔离，以及开箱即用的 Web 聊天界面。

---

## 为什么选择 Light Agent

| 痛点 | Light Agent 的解法 |
|------|-------------------|
| 长对话上下文爆炸 | SQLite 全量存储 + LLM 自动摘要压缩 + 关键词 RAG 检索，无需向量数据库 |
| 多模型切换繁琐 | 统一 OpenAI 兼容接口，6 家厂商一行切换 |
| 工具注册样板代码多 | 装饰器 / 自动推断 / 模块扫描三种注册方式，函数签名即文档 |
| 框架太重不可控 | 整个框架约 2000 行代码，每个模块可独立使用 |

---

## 功能一览

- **6 家 LLM 厂商** — OpenAI / DeepSeek / Kimi / MiniMax / Grok / OpenRouter，统一接口
- **多模态能力** — 文本对话、图像理解 (Vision)、文字转语音 (TTS)、语音转文字 (STT)、图像生成 (DALL-E 3)
- **智能记忆系统** — SQLite 持久化，上下文超限自动压缩，闲置 6 小时自动摘要，关键词 RAG 检索历史
- **工具调用** — OpenAI Function Calling 风格，支持装饰器、自动推断、模块发现、上下文注入
- **多 Agent 路由** — 关键词规则 / LLM 意图识别 / 能力自动路由
- **多用户会话隔离** — 用户级数据隔离，支持多会话管理
- **三种运行方式** — 命令行 REPL / Flask HTTP API / 作为库导入
- **Web 聊天界面** — 深色主题单页应用，开箱即用

---

## 快速开始

### 安装依赖

```bash
pip install requests flask
```

> Python 3.9+，`flask` 仅在启动 HTTP 服务时需要。

### 配置

编辑 `config/config.json`（首次运行自动生成默认配置）：

```json
{
    "provider": "deepseek",
    "api_key": "your-api-key",
    "model": "deepseek-chat"
}
```

也可通过环境变量设置 API Key：

```bash
export DEEPSEEK_API_KEY=your-api-key
```

### 命令行对话

```bash
python3 main.py
```

支持 `/quit` 退出、`/reset` 重置对话、`/history` 查看历史。

### 启动 HTTP 服务 + Web 界面

```bash
python3 -m src.server
```

打开浏览器访问 `http://localhost:8000`，即可使用 Web 聊天界面。

### 作为库使用

```python
from src.llm import LLMFactory
from src.agent import Agent, AgentConfig

llm = LLMFactory.create("deepseek", "your-api-key")
agent = Agent(llm, AgentConfig(name="my-agent"))

response = agent.chat("你好")
print(response)
```

---

## 项目结构

```
light_agent/
├── main.py                 # 命令行入口
├── config/
│   └── config.json         # 运行配置
├── src/
│   ├── agent.py            # Agent 核心（对话管理、工具调用循环、记忆压缩）
│   ├── llm.py              # LLM 统一接口（6 家厂商 + TTS/STT/图像生成）
│   ├── memory.py           # 记忆系统（SQLite 持久化 + RAG 检索 + 自动压缩）
│   ├── tools.py            # 工具注册与执行（装饰器 / 自动推断 / 上下文注入）
│   ├── router.py           # 多 Agent 路由 + 能力路由
│   ├── session.py          # 多用户多会话管理
│   ├── server.py           # Flask HTTP API 服务
│   ├── config.py           # 配置管理（JSON + 环境变量 + CLI 工具）
│   ├── prompt.py           # 系统提示词模板
│   ├── logging.py          # 结构化日志（思考 / 工具调用 / 回复）
│   ├── utils.py            # 工具函数
│   └── example.py          # 使用示例
├── tests/                  # 单元测试
├── web/                    # Web 聊天界面（HTML + JS + CSS）
└── docs/                   # 详细文档
```

---

## 核心设计

### 记忆系统：无向量数据库的 RAG

传统方案依赖 Embedding + 向量数据库做语义检索，Light Agent 采用更轻量的方案：

1. **全量存储** — 每条消息（用户/助手/工具）存入 SQLite
2. **关键词索引** — 对消息内容进行中英文分词（正则 `[\u4e00-\u9fff]+|[a-zA-Z]+|\d+`），建立倒排索引
3. **自动压缩** — 当上下文预估 token 超过窗口的 50% 时，用 LLM 将旧消息摘要化，仅保留最近 30% 原文 + 摘要
4. **闲置压缩** — 会话闲置超过 6 小时且未压缩过，自动触发摘要
5. **RAG 检索** — 用户消息中的关键词匹配历史索引，按相关度排序返回 Top-K 结果

这套方案在单机场景下足够高效，且完全消除了向量数据库的运维成本。

### 工具系统：上下文注入

工具函数可以声明需要运行时上下文（如 `user_id`），框架会自动注入这些参数并从 LLM 可见的 schema 中隐藏：

```python
from src.tools import tool

@tool(description="搜索用户的历史对话")
def search_memory(query: str, user_id: str = "") -> str:
    """query: 搜索关键词"""
    # user_id 由框架自动注入，LLM 不会看到这个参数
    results = memory.search_messages(user_id, query)
    return str(results)
```

### 多模型路由

```python
from src.router import Router, keyword_rule, intent_rule

router = Router()
router.register("default", general_agent)
router.register("code", code_agent)
router.register("creative", creative_agent)

# 关键词路由
router.add_rule(keyword_rule({"代码": "code", "编程": "code"}))

# LLM 意图路由
router.add_rule(intent_rule(llm, {"code": "编程相关", "creative": "创意写作"}))

response = router.chat("帮我写段 Python 代码")  # -> code_agent
```

---

## LLM 厂商支持

| 厂商 | Provider ID | 默认模型 | 能力 |
|------|------------|----------|------|
| OpenAI | `openai` | `gpt-4o` | TEXT, VISION |
| DeepSeek | `deepseek` | `deepseek-chat` | TEXT |
| Kimi (月之暗面) | `kimi` | `moonshot-v1-8k` | TEXT, VISION |
| MiniMax | `minimax` | `MiniMax-M2.5` | TEXT |
| Grok (xAI) | `grok` | `grok-2-latest` | TEXT |
| OpenRouter | `openrouter` | `openai/gpt-4o` | TEXT, VISION |

所有厂商均通过 OpenAI 兼容格式接入，切换只需修改 `provider` 和 `api_key`。

扩展能力（基于 OpenAI API）：

| 能力 | 类名 | 说明 |
|------|------|------|
| TTS | `OpenAITTS` | 文字转语音 |
| STT | `OpenAISTT` | 语音转文字 (Whisper) |
| 图像生成 | `OpenAIImageGen` | DALL-E 3 |

---

## HTTP API

启动服务后可用的接口：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| POST | `/api/users` | 创建用户 |
| GET | `/api/users` | 用户列表 |
| POST | `/api/sessions` | 创建会话 |
| GET | `/api/users/<user_id>/sessions` | 获取用户会话列表 |
| POST | `/api/chat` | 发送消息 |
| POST | `/api/chat/stream` | 流式对话 |
| POST | `/api/search` | 记忆检索 |
| GET | `/api/stats` | 统计信息 |

```bash
# 创建用户
curl -X POST http://localhost:8000/api/users \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user1", "display_name": "张三"}'

# 创建会话
curl -X POST http://localhost:8000/api/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user1", "title": "测试对话"}'

# 对话
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user1", "session_id": "SESSION_ID", "message": "你好"}'
```

---

## 配置项

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `provider` | LLM 厂商 | `openai` |
| `api_key` | API 密钥（可用环境变量 `{PROVIDER}_API_KEY` 覆盖） | `""` |
| `model` | 模型名称（留空则使用厂商默认） | `""` |
| `system_prompt` | 系统提示词 | `你是一个有用的 AI 助手。` |
| `context_window` | 上下文窗口大小 (tokens) | `128000` |
| `temperature` | 生成温度 | `0.7` |
| `max_tokens` | 最大输出长度 | `4096` |
| `compress_threshold` | 压缩触发阈值（上下文使用率） | `0.5` |
| `keep_ratio` | 压缩后保留最近消息的比例 | `0.3` |
| `idle_compress_hours` | 闲置多少小时后自动压缩 | `6` |
| `debug` | 调试模式 | `false` |

也可通过 CLI 工具管理配置：

```bash
python3 -m src.config show           # 查看所有配置
python3 -m src.config set provider deepseek
python3 -m src.config api_key your-key
```

---

## 扩展指南

### 添加自定义工具

**方式一：装饰器**

```python
from src.tools import tool

@tool(description="获取天气信息")
def get_weather(city: str) -> str:
    """city: 城市名称"""
    return f"{city}: 晴天, 25°C"
```

**方式二：手动注册**

```python
registry = ToolRegistry()
registry.register(
    name="get_weather",
    description="获取天气信息",
    parameters={"city": {"type": "string", "description": "城市名"}},
    function=get_weather
)
```

**方式三：模块自动发现**

```python
from src.tools import ToolRegistry

registry = ToolRegistry()
registry.discover_tools_from_package("my_tools")  # 扫描包内所有 @tool 装饰的函数
```

### 添加新 LLM 厂商

在 `llm.py` 的 `LLMFactory.PROVIDERS` 字典中添加：

```python
"new_provider": {
    "base_url": "https://api.newprovider.com/v1",
    "default_model": "model-name",
    "capabilities": [LLMType.TEXT, LLMType.VISION]
}
```

---

## 测试

```bash
python3 -m pytest tests/
```

覆盖记忆模块、工具系统、会话管理及用户隔离验证。

---

## 依赖

| 包 | 用途 | 是否必须 |
|----|------|----------|
| `requests` | LLM API 调用 | 是 |
| `flask` | HTTP 服务 + Web 界面 | 仅服务端需要 |

其余均为 Python 标准库（`sqlite3`, `json`, `dataclasses`, `inspect`, `uuid` 等）。

---

## License

MIT
