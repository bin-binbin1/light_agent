# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

Light Agent 是一个轻量级 Python Agent 框架，面向超长对话场景。**核心原则：零向量数据库、零重型框架**——依赖仅 `requests` / `httpx` / `fastapi` / `uvicorn` / `python-dotenv`，其余都走标准库。修改时应优先保持这种轻量性，新增依赖前需谨慎评估。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 命令行交互对话（异步流式 + 自动恢复最近 session）
python main.py [config/config.json]

# 启动 HTTP 服务 + Web UI（FastAPI + uvicorn）
python server/run.py --host 0.0.0.0 --port 8000
python server/run.py --reload                    # 开发模式：代码热重载（与 --workers 互斥）
python server/run.py --workers 4                 # 每个 worker 独立 agent_pool → 反向代理需 sticky session

# 生产部署（gunicorn + UvicornWorker，Linux 推荐；无需 wsgi.py）
gunicorn -w 4 -k uvicorn.workers.UvicornWorker server.app:app --bind 0.0.0.0:8000

# 配置 CLI（会写回 config.json）
python -m src.config show
python -m src.config set <key> <value>
python -m src.config provider <openai|deepseek|kimi|minimax|grok|openrouter|xiaomi>
python -m src.config api_key <key>

# 测试
python -m pytest tests/
python -m pytest tests/test_memory.py::test_basic -v   # 单测
python tests/test_memory.py                            # 测试脚本内部有 sys.path 注入，可直跑
```

## API Key 配置链路（三种方式，优先级从高到低）

1. **`.env` 文件**（推荐）— `main.py:33-41` 启动时先找 `light_agent/.env`，再找 `../.env`，用 `python-dotenv` 写入 `os.environ`。`server/app.py:36-44` 同样逻辑
2. **系统环境变量** — 按 `src/config.py:45-53` 的 `ENV_KEY_NAMES` 查，每个 provider 有多候选名（如 kimi 认 `KIMI_API_KEY` 也认 `MOONSHOT_API_KEY`，小米内部 provider `xiaomi` 用 `MIFY_KEY`）
3. **`config.json` 的 `api_key` 字段** — 最后兜底

绝不要把 key 写进 `config.json` 后提交 git（`.gitignore` 已排除 `.env`，但 `config/config.json` **未排除**）。

## 架构要点（需跨文件理解的部分）

### 1. `Config.create_agent_from_config()` 是唯一规范的 Agent 创建入口

`src/config.py:188-276` 实现，`main.py:182` 和 `server/app.py:141` 都调用它。以前散落在 main.py 的 80 多行装配逻辑（check_config / resolve_api_key / create_agent）都已收敛到这里。新写入口点（server、CLI、库使用）**都应走这个方法**，不要重新散装 LLM + Memory + AgentConfig。

内部路由规则：
- `config.base_url` 非空 **或** `provider` 不在 `LLMFactory.PROVIDERS` 里 → 直接构造 `OpenAICompatibleLLM`（自建代理、中转站、Ollama、vLLM 走这条）
- 否则 → `LLMFactory.create(provider, api_key, model)`（走内置厂商配置）

### 2. 记忆系统：三层存储 + Write-Back Cache

一条消息的生命周期涉及三个存储层：

- **`Agent._context`**（内存）— 实际发给 LLM 的消息列表，压缩后的"有效上下文"
- **`messages` 表**（全量）— 每条消息都写入，从不删除；用于 RAG 和历史回溯
- **`context_snapshots` 表**（Write-Back）— `_context` 的持久化快照，通过 `save_state()` / `load_state()` 同步

关键不变量：`chat()` 每条消息既 `memory.add_message()`（入库）又 `_append_context()`（进内存）。压缩只操作 `_context`，不影响 `messages` 全量记录。启动时 `Agent.__init__` 调 `memory.load_context()`：有快照则用快照，否则 fallback 到 `get_context_for_llm`（最近摘要 + 最近消息）。

### 3. 压缩路径：同步 vs 后台异步

- **同步 `chat()`**：命中阈值时在 LLM 调用前直接 `memory.compress()`，阻塞用户
- **异步 `achat_stream()`**：`_maybe_start_compress()` 启动后台 `asyncio.Task`，用**快照**压缩。关键细节：压缩期间用户可能追加新消息，完成后必须 `new_context + self._context[len(snapshot):]` 把增量拼回，否则丢消息（见 `agent.py:116-122`）

触发条件：`should_compress`（token 估算 `char_count * 1.5` 超过 `context_window * compress_threshold`），或 `should_compress_idle`（闲置超 `idle_compress_hours` 且**从未压缩过**）。

### 4. 工具系统：上下文注入 + LLM 隐藏

工具函数在签名中声明运行时参数（如 `user_id: str = ""`），框架从 `ToolRegistry._context` 自动注入，并**从 LLM 可见的 schema 中剔除**（`Tool.to_openai_format()` 用 `context_keys` 过滤）。

- `ToolRegistry.set_context(user_id=...)` 会扫描所有已注册工具的签名更新 `context_keys`
- 但 `register_function` 只基于当前 `_context` 计算一次 → **先 `set_context` 再注册**更可靠
- 三种注册方式：装饰器 `@tool(...)` + `register_decorated`、传统 `register(name, ...)`、包扫描 `discover_tools_from_package`

### 5. 多用户隔离

数据层双层隔离：所有 SQL 查询都带 `(session_id, user_id)`。

- `SessionManager`（`src/session.py`）管理 `users` + `sessions` 表
- `Memory` 的所有 SQL 都同时过滤 `session_id` AND `user_id`，**即便 session_id 全局唯一也不简化**——防御性设计

服务端的 Agent 池（`server/app.py:79-191`）是另一层：

- `_agent_pool: dict[username, Agent]` 按用户名一一对应，每个用户一个常驻 Agent（session 由 `create_agent_from_config(resume=True)` 自动恢复最近一条）
- `_agent_locks[username]` 确保同一用户的 chat / reset / logout 串行执行
- `_cleanup_idle_agents()` 后台每 300s 扫一次，闲置超 `LIGHT_AGENT_IDLE_TIMEOUT`（默认 1800s）的 Agent 被驱逐并 `save_state()`
- `--workers >1` 时每个 worker 独立维护 `_agent_pool`，反向代理必须按 token/username 做 sticky routing，否则用户会在 worker 间跳跃丢 Agent 状态

### 6. SQL 方言抽象（sqlite / mysql）

`src/dialect.py` 定义 `DIALECTS` 字典（placeholder、autoincrement_pk、insert_or_ignore、insert_or_replace 等），`Memory` 和 `SessionManager` 启动时读 `config.dialect` 取方言对象。**不要用硬编码 `?` 或 `INSERT OR IGNORE`**——用 `self._ph` 和 `self._d["insert_or_ignore"]`。

**目前虽然还是 `sqlite3.connect()`**，但建表语句、字段类型（`VARCHAR(64)`）、PK 定义已经方言化。切换 MySQL 时只需替换 connection 层。

### 7. `keyword_index` 表迁移（RAG 检索）

`memory.py` 有新旧两套索引表：

- **旧**：`message_index`（每消息一行，`keywords` 列用 `|` 拼接）
- **新**：`keyword_index`（独立倒排表，一词一行，`(user_id, session_id, keyword)` 组合索引）

`Memory.__init__` 调 `_maybe_migrate_keyword_index()`：首启时若新表空而旧表非空，一次性回填。搜索走新表 `GROUP BY + COUNT` 聚合匹配分，性能远优于旧的 `LIKE %word%`。**改动索引相关代码时注意两张表都在**，迁移完旧表不再写入但保留。

### 8. 服务端（FastAPI + SSE）

`server/app.py` 是单一 FastAPI 应用，路由组仅 `/agent/*`：

- **认证**：`POST /agent/login` 接收 `{username}`，写入 `data/users.db` 的 `users` + `tokens` 表，返回 sha256 token（TTL 默认 7 天，可用 `LIGHT_AGENT_TOKEN_TTL` 覆盖）。后续接口通过 `?token=...` query param + `get_current_user` Depends 校验
- **对话**：`POST /agent/chat` 返回 SSE 流（`text/event-stream`），`AgentEvent` 子类序列化成 `event: <type>\ndata: <json>`，文本 chunk 包成 `event: chunk`，结束发 `event: done`。**SSE 不走统一 `{code,msg,data}` 包装**，其他 JSON 端点都走
- **其他**：`/agent/reset` 新 session、`/agent/session` 当前会话 id、`/agent/history` 历史、`/agent/reload` 开发用（清空 pool 强制重建以读取 config 热更新）
- **静态**：`/agent/index` 返回 `web/index.html`，`/agent/static/*` 挂 `web/` 目录
- **全局异常处理器**把 `HTTPException` / `RequestValidationError` / 未捕获 `Exception` 统一包成 `{code,msg,data}` JSON

`server/run.py` 是 uvicorn 启动器（argparse 包装），读取 `server.app:app` 入口。

### 9. LLM 抽象

6 家厂商都走 `OpenAICompatibleLLM`（`src/llm.py`），区别仅在 `LLMFactory.PROVIDERS` 字典。**扩展新厂商只加字典条目，不派生新类**。自建端点走 `config.base_url` + `OpenAICompatibleLLM` 直连，不需要进 `PROVIDERS`。

- 图像理解：检测消息含 `image_url` 自动切 `vision_model`
- 429 重试：最多 5 次指数退避 + 抖动，通过 `RetryEvent` 冒泡给流式消费者
- TTS/STT/ImageGen 是独立类（`OpenAITTS` / `OpenAISTT` / `OpenAIImageGen`），不是 `BaseLLM` 子类

### 10. 事件流

`src/events.py` 的 `AgentEvent` 子类通过 `achat_stream` yield 给消费者：`ThinkingEvent` / `ToolCallEvent` / `ToolResultEvent` / `RetryEvent` / `ErrorEvent`。

工具的 `display_calling` / `display_done` / `display_failed` 文案**和内部 `name` 分离**——日志打印 `name`（调试用），用户看 `display`（产品语言）。`main.py:124-152` 的循环示范了事件消费模式，`server/app.py:337-348` 演示 SSE 打包。

## 常见坑

- **`server/run.py` 必须从项目根执行**：脚本开头把父目录注入 `sys.path`，uvicorn 入口字符串是 `"server.app:app"`（带包前缀）。不能 `cd server && python run.py`，会找不到 `src.*`
- **`--workers >1` 的 Agent 池是进程隔离的**：每个 worker 一份 `_agent_pool`，用户请求不走 sticky 就会命中不同 worker 触发重复 Agent 创建。生产部署通过反向代理按 token/username hash 即可
- **SQLite 并发**：`main.py` 读 `config.memory_db`（默认 `memory.db`），server 按 config 读（当前 `data/server_memory.db`）。两个入口默认不是同一库，但若改配置让它们共享同一文件需注意文件锁
- **`sync_light_agent.sh`** 是维护者专用（硬编码 Windows 路径 `C:\code\...`），把本仓库同步到 `safety_score_agent/light_agent/`，不是构建脚本
- **`README.md` 和 `docs/README.md` 都已过时**：README 里仍写 `python -m src.server`（实际 `server/run.py`）、`flask` 依赖（实际 `fastapi`）、`/api/*` 路由（实际 `/agent/*`）。以当前 `server/app.py` 源码为准
- **Windows 下 `readline`** 靠 `requirements.txt` 里的 `pyreadline3`（已用 `sys_platform == "win32"` 标记）。纯净解释器不装直接跑 `main.py` 会 ImportError
- **`config/config.json` 未被 `.gitignore` 排除**：如果主动往里写 `api_key` 会被提交 git。推荐走 `.env` 路径
- **`tests/conftest.py` 把 `/tmp/xxx` 重写到系统 tempdir**：所以测试代码里写 `"/tmp/test_memory.db"` 在 Windows 也能跑；**这是测试兼容层，生产路径里别出现 `/tmp/`**
