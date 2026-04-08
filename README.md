# Light Agent

轻量级 Python Agent 框架

## 项目概述

一个模块化的 AI Agent 开发框架，支持多模型接入、工具调用、长期记忆、会话管理。

> **注意**：此 `light_agent` 目录是 `safety_score_agent` 项目的一个子模块/依赖项，为其提供核心的 Agent 框架功能。

## 目录结构

```
light_agent/
├── src/                    # 核心代码
│   ├── agent.py            # Agent 核心（对话管理、工具调用）
│   ├── llm.py              # LLM 接口（多模型支持）
│   ├── memory.py           # 记忆管理（SQLite + RAG）
│   ├── tools.py            # 工具注册与执行
│   ├── server.py           # HTTP API 服务
│   ├── router.py           # 多 Agent 路由
│   ├── config.py           # 配置管理
│   ├── session.py          # 会话管理
│   ├── prompt.py           # 提示词管理
│   ├── logging.py          # 日志系统
│   └── example.py          # 示例代码
├── tests/                  # 单元测试
│   ├── test_agent.py
│   ├── test_memory.py
│   ├── test_tools.py
│   └── test_session.py
├── config/                 # 配置文件
│   └── config.json
├── docs/                   # 文档
│   └── README.md
├── main.py                 # 入口文件
└── README.md               # 项目说明
```

---

## 核心模块

### 1. agent.py - Agent 核心

**职责**：对话管理、工具调用循环、记忆压缩

**主要类**：
- `AgentConfig` - Agent 配置（名称、上下文窗口、温度、token 限制）
- `Agent` - 主类

**核心方法**：
```python
agent = Agent(llm, config, tools, logger)
response = agent.chat(user_input)  # 单轮对话
agent.reset()  # 重置会话
history = agent.get_history()  # 获取历史
```

**功能特性**：
- 自动记忆压缩（上下文超限 / 闲置触发）
- 工具调用循环（支持多轮工具调用）
- 会话隔离（UUID 区分）

---

### 2. llm.py - LLM 接口

**职责**：统一封装多厂家 API（DeepSeek、Kimi、OpenAI、MiniMax、Grok、OpenRouter）

**主要类**：
- `BaseLLM` - 抽象基类
- `OpenAICompatibleLLM` - OpenAI 兼容接口
- `LLMFactory` - 工厂类，创建各厂商实例
- `OpenAITTS` / `OpenAISTT` / `OpenAIImageGen` - 扩展能力

**支持的能力**：
| 能力 | 说明 |
|------|------|
| TEXT | 纯文本对话 |
| VISION | 图像理解 |
| TTS | 文字转语音 |
| STT | 语音转文字 |
| IMAGE_GEN | 图像生成 |

**使用示例**：
```python
llm = LLMFactory.create("deepseek", api_key, "deepseek-chat")
response = llm.chat(messages, tools=tools, temperature=0.7)
```

---

### 3. memory.py - 记忆管理

**职责**：SQLite 持久化、关键词检索、自动压缩

**主要类**：
- `MemoryConfig` - 配置（压缩阈值、保留比例、闲置时间）
- `Memory` - 主类

**核心功能**：
- **全量存储**：所有消息存入 SQLite
- **自动压缩**：上下文超限时触发，生成摘要 + 保留最近消息
- **RAG 检索**：关键词搜索历史对话
- **按时间检索**：时间范围查询

**数据库表**：
- `sessions` - 会话信息
- `messages` - 消息记录
- `summaries` - 压缩摘要
- `message_index` - 关键词索引

**核心方法**：
```python
memory.add_message(session_id, "user", content)
memory.compress(session_id)  # 压缩
results = memory.search_messages(session_id, query, top_k=5)  # 搜索
context = memory.get_context_for_llm(session_id)  # 获取上下文
```

---

### 4. tools.py - 工具系统

**职责**：OpenAI Function Calling 风格工具注册与执行

**主要类**：
- `Tool` - 工具定义
- `ToolRegistry` - 工具注册表

**内置工具**：
- `calculator` - 数学计算
- `get_time` - 获取当前时间
- `search_memory` - 历史记忆检索

**使用示例**：
```python
registry = ToolRegistry()
registry.register(
    name="my_tool",
    description="工具描述",
    parameters={"arg1": {"type": "string", "description": "参数1"}},
    function=my_func
)
tools_desc = registry.to_openai_format()  # 转为 OpenAI 格式
result = registry.execute("my_tool", {"arg1": "value"})
```

---

### 5. server.py - HTTP 服务

**职责**：Flask REST API，提供对话、用户、会话管理接口

**API 端点**：
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| POST | `/api/users` | 创建用户 |
| GET | `/api/users` | 用户列表 |
| POST | `/api/sessions` | 创建会话 |
| GET | `/api/users/<user_id>/sessions` | 会话列表 |
| POST | `/api/chat` | 对话 |
| POST | `/api/chat/stream` | 流式对话 |
| POST | `/api/search` | 记忆搜索 |
| GET | `/api/stats` | 统计信息 |

**启动服务**：
```python
run_server(host="0.0.0.0", port=8000, config_path="config/config.json")
```

---

### 6. router.py - 路由系统

**职责**：多 Agent 路由、能力路由

**主要类**：
- `ModelPool` - 模型池，按能力管理 LLM
- `Router` - 多 Agent 路由器
- `CapabilityRouter` - 能力路由器（自动选择合适模型）

**功能**：
- 按关键词路由到指定 Agent
- 按 LLM 判断意图路由
- 按能力（TEXT/VISION/TTS/IMAGE_GEN）自动选择模型

---

### 7. config.py - 配置管理

**职责**：读取/保存 JSON 配置，支持环境变量覆盖

**默认配置**：
```json
{
  "provider": "openai",
  "api_key": "",
  "model": "",
  "system_prompt": "你是一个有用的 AI 助手。",
  "context_window": 128000,
  "temperature": 0.7,
  "max_tokens": 4096,
  "compress_threshold": 0.5,
  "keep_ratio": 0.3,
  "idle_compress_hours": 6
}
```

**使用示例**：
```python
config = Config("config/config.json")
config.set("provider", "deepseek")
config.save()
```

---

### 8. session.py - 会话管理

**职责**：用户与会话的映射关系

**主要功能**：
- 创建/删除会话
- 用户管理
- 会话统计

---

## 快速开始

### 1. 配置

编辑 `config/config.json`：
```json
{
  "provider": "deepseek",
  "api_key": "your-api-key",
  "model": "deepseek-chat"
}
```

### 2. 运行服务

```bash
cd light_agent
python3 main.py
```

或直接运行 server：
```bash
python3 -m src.server
```

### 3. API 调用

```bash
# 创建会话
curl -X POST http://localhost:8000/api/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user1", "title": "测试会话"}'

# 对话
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user1", "session_id": "xxx", "message": "你好"}'
```

---

## 依赖

```
flask
requests
```

---

## 扩展

### 添加自定义工具

```python
def my_tool(arg1: str, arg2: int) -> str:
    return f"{arg1} x {arg2}"

registry.register(
    name="my_tool",
    description="自定义工具",
    parameters={
        "arg1": {"type": "string", "description": "参数1"},
        "arg2": {"type": "integer", "description": "参数2"}
    },
    function=my_tool,
    required=["arg1"]
)
```

### 添加新模型厂商

在 `llm.py` 的 `LLMFactory.PROVIDERS` 中添加：
```python
"new_provider": {
    "base_url": "https://api.newprovider.com/v1",
    "default_model": "model-name",
    "capabilities": [LLMType.TEXT, LLMType.VISION]
}
```