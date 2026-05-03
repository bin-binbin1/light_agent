# Flutter Chat 客户端技术方案

> 本文件是 `flutter_chat` 项目的技术方案归档，记录设计决策、目录结构、实现顺序和验证方法。服务端接口以 `../server/app.py` 源码为准，本文档如与源码不一致以源码为准并及时更新。

## 背景

Light Agent 提供 FastAPI 服务端（`../server/app.py`），已暴露用户登录 / SSE 流式对话 / 会话管理 / 历史查询等接口。当前已有一个空壳 Android 原生项目（`../android_chat/`）和一个 Web 参考实现（`../web/app.js`）。

改用 Flutter 实现客户端的理由：

- 跨平台（Android / Windows / Web 同时跑）
- 状态驱动 UI，流式 chunk 更新更丝滑
- 借机统一三端交互逻辑，避免 Web / 安卓双写

目标是复刻 Web 端全部功能，并改进：

- 用 `flutter_markdown` 替代 Web 端自实现的极简 Markdown（Web 端只支持代码块）
- 加 `CancelToken` 支持切换会话时中断正在进行的 SSE 流（Web 端没有）
- 用 Riverpod 做状态驱动，流式 chunk 更新不卡顿
- 滚动跟随加防抖 + 判断用户是否手动上翻

## 技术选型

| 维度 | 选择 | 说明 |
|---|---|---|
| 状态管理 | Riverpod (`flutter_riverpod` 2.x) | 类型安全、编译期校验、社区主流 |
| HTTP / SSE | `dio` + `ResponseType.stream` | 可控、无额外依赖，SSE 按帧自解析 |
| 本地存储 | `shared_preferences` | token / username / 当前 sessionId |
| Markdown 渲染 | `flutter_markdown` + `flutter_highlight` | 完整 Markdown + 代码高亮 |
| 路由 | 原生 `Navigator.pushReplacement` | 登录页 + 主页两个路由，不上 go_router |
| 目标平台 | Android + Windows + Web | 暂不含 iOS / macOS / Linux |

## 项目初始化命令

```powershell
cd D:\projects\light_agent
flutter create --platforms=android,windows,web --project-name=flutter_chat flutter_chat
```

## pubspec.yaml 依赖清单

```yaml
dependencies:
  flutter:
    sdk: flutter
  flutter_riverpod: ^2.5.1
  dio: ^5.7.0
  shared_preferences: ^2.3.2
  flutter_markdown: ^0.7.4
  flutter_highlight: ^0.7.0
  intl: ^0.19.0          # 时间格式化
```

## 目录结构

```
flutter_chat/
└── lib/
    ├── main.dart                              # 入口：ProviderScope + MaterialApp
    ├── app.dart                               # 顶层 App，根据登录态切 LoginPage / ChatPage
    ├── core/
    │   ├── constants.dart                     # baseUrl（可配置注释）、Keys
    │   ├── storage/
    │   │   └── auth_storage.dart              # SharedPreferences 封装
    │   ├── api/
    │   │   ├── api_client.dart                # Dio 单例 + {code,msg,data} 拆包 + 401 处理
    │   │   ├── api_exception.dart             # ApiException(code/msg)
    │   │   └── sse_client.dart                # SSE 流式解析：dio stream → Stream<AgentEvent>
    │   └── models/
    │       ├── api_response.dart              # ApiResponse<T>
    │       ├── agent_event.dart               # sealed class AgentEvent + 子类
    │       ├── message.dart                   # Message + ToolCall + ToolCallFunction
    │       └── session_info.dart              # SessionInfo
    └── features/
        ├── auth/
        │   ├── auth_repository.dart           # /agent/login, /agent/logout, /agent/session
        │   ├── auth_provider.dart             # authStateProvider + loginController
        │   └── login_page.dart                # 用户名输入框 + 登录按钮
        ├── sessions/
        │   ├── session_repository.dart        # /agent/sessions, reset, clear, delete
        │   ├── sessions_provider.dart
        │   └── session_drawer.dart            # 左侧抽屉
        └── chat/
            ├── chat_repository.dart           # /agent/chat (SSE), /agent/history
            ├── chat_provider.dart             # StateNotifier<ChatState>
            ├── chat_page.dart                 # Scaffold + Drawer + 消息列表 + 输入栏
            └── widgets/
                ├── message_bubble.dart        # user / assistant 气泡
                ├── tool_bubble.dart           # 工具调用状态条
                ├── thinking_dots.dart         # 三点跳动
                └── info_bubble.dart           # retry / 系统消息
```

## 关键设计要点

### 1. AgentEvent 用 sealed class（`core/models/agent_event.dart`）

Dart 3 的 sealed class 让 SSE 事件分发可以写成穷尽式 switch。服务端事件 dataclass 定义在 `../src/events.py`：

```dart
sealed class AgentEvent {}
class ChunkEvent extends AgentEvent { final String text; }
class ThinkingEvent extends AgentEvent {}
class ToolCallEvent extends AgentEvent {
  final String name;       // 内部名，日志/调试用
  final String display;    // 对用户展示的文案
}
class ToolResultEvent extends AgentEvent {
  final String name;
  final int durationMs;
  final bool success;
  final String display;
}
class RetryEvent extends AgentEvent {
  final String reason;     // "429" 等
  final int attempt;
  final int maxAttempts;
  final double waitSeconds;
}
class ErrorEvent extends AgentEvent { final String message; }
class DoneEvent extends AgentEvent {}
```

`AgentEvent.fromSse(String event, Map<String, dynamic> data)` 工厂根据 event 名分发。注意字段名要与 Python dataclass 一致：服务端 `duration_ms` / `wait_seconds` / `max_attempts` 为 snake_case，Dart 侧解析时需做映射。

### 2. SSE 解析（`core/api/sse_client.dart`）

参考 Web 端实现，用 `StreamTransformer`：

```dart
Stream<AgentEvent> chatStream({
  required String token,
  required String sessionId,
  required String message,
  CancelToken? cancelToken,
}) async* {
  final response = await _dio.post<ResponseBody>(
    '/agent/chat',
    data: {'message': message, 'session_id': sessionId},
    queryParameters: {'token': token},
    options: Options(responseType: ResponseType.stream),
    cancelToken: cancelToken,
  );

  String buffer = '';
  await for (final chunk in response.data!.stream.transform(utf8.decoder)) {
    buffer += chunk;
    final frames = buffer.split('\n\n');
    buffer = frames.removeLast();          // 末尾不完整帧保留
    for (final frame in frames) {
      final evt = _parseSseFrame(frame);
      if (evt != null) yield evt;
    }
  }
}
```

帧解析（按行扫 `event:` 和 `data:`）逻辑从 `../web/app.js` 翻译。

**注意**：服务端 `/agent/chat` 会在用户消息前自动加一行 `[发送时间: YYYY-MM-DD HH:MM:SS]\n` 再喂给 LLM（见 `server/app.py:386-387`），**客户端不需要自己拼时间**，直接发原文即可。

### 3. 消息列表状态驱动（`features/chat/chat_provider.dart`）

`ChatState` 持有 `List<ChatItem>`，`ChatItem` 是 sealed class：

```dart
sealed class ChatItem {}
class UserMsgItem extends ChatItem { final String content; }
class AssistantMsgItem extends ChatItem {
  String content;           // 可变，流式追加
  bool streaming;           // true 时显示 thinking dots
}
class ToolItem extends ChatItem {
  final String display;
  ToolStatus status;        // running / success / fail
  int? durationMs;
}
class InfoItem extends ChatItem { final String text; }
```

流式事件处理：

- `chunk` → 取列表最后的 AssistantMsgItem，追加 content，触发重建
- `tool_call` → 固化当前 AssistantMsgItem（streaming=false），新建 ToolItem + 新 AssistantMsgItem
- `tool_result` → 从列表倒序找同名 running ToolItem，更新 status/duration
- `retry` → 追加 InfoItem 显示等待信息
- `done` / `error` → streaming=false，释放 CancelToken，解锁输入框

### 4. 滚动跟随（`features/chat/chat_page.dart`）

- `ScrollController` 监听 `position.pixels` vs `position.maxScrollExtent`
- 阈值：距底部 < 80px → 跟随模式，每次列表变化后 `animateTo(max, 120ms)`
- 否则 → 用户查看历史，不自动滚，但显示"↓ 回到最新"浮动按钮
- 节流：用 `Timer` 16ms 防抖

### 5. Markdown 流式渲染权衡

Web 端策略：流式阶段用纯文本，done 后才渲染 Markdown（避免半截代码块 ``` 被错误解析）。

Flutter 端同样策略：

- `AssistantMsgItem.streaming == true` → `SelectableText(content)`
- `streaming == false` → `MarkdownBody(data: content, ...)`

### 6. API 统一响应处理（`core/api/api_client.dart`）

Dio 响应拦截器：

- HTTP 2xx 且 body 是 `{code, msg, data}` → `code == 0` 返回 `data`，否则抛 `ApiException(code, msg)`
- HTTP 401 → 清 SharedPreferences + 通知 auth provider 跳登录页
- HTTP 400 / 403 / 404 → 抛特定异常让业务层决定（见 §8 会话归属）

服务端即便是异常也会统一包成 `{code, msg, data}`（见 `server/app.py:266-290`），所以 `dio` 收到 4xx/5xx 时 response.data 仍是合法 JSON，`onError` 里要尝试读 `data['code']` 和 `data['msg']` 再包成 `ApiException`。

### 7. baseUrl 配置（`core/constants.dart`）

```dart
// 后端地址：根据运行环境改这里
// - Windows 桌面 / iOS 模拟器: http://127.0.0.1:8000
// - Android 模拟器:            http://10.0.2.2:8000
// - Android 真机:              http://<电脑局域网 IP>:8000
// - Web (CORS 已开):          http://localhost:8000
const String baseUrl = 'http://10.0.2.2:8000';
```

### 8. 会话归属严格校验

服务端对所有涉及 `session_id` 的接口都调用 `_check_session_ownership`（`server/app.py:137-159`）：

| 场景 | HTTP | code | msg |
|---|---|---|---|
| `session_id` 为空字符串 | 400 | 400 | session_id 不能为空 |
| `session_id` 在 DB 中不存在 | 404 | 404 | session 不存在 |
| `session_id` 属于别的用户 | 403 | 403 | 无权访问该 session |

客户端处理约定：

- 403 / 404 → 清除本地 `sessionId`，调 `/agent/session` 取最近会话，再不行就 `/agent/reset` 新建
- 401 → token 失效，回登录页

### 9. Agent 池与并发（服务端行为，客户端需知）

`server/app.py` 的 Agent 池键为 `(username, session_id)` 元组（`app.py:79-83`）：

- **同一用户可以同时挂载多个会话 Agent**（每个 session 一个），不会互相阻塞
- 同一 `(username, session_id)` 的 chat / reset / clear 通过 `asyncio.Lock` 串行化
- 闲置超 `LIGHT_AGENT_IDLE_TIMEOUT`（默认 1800s）的 Agent 会被驱逐并 `save_state()` 落盘
- 驱逐后下次访问自动重建，状态从 `context_snapshots` 恢复

这意味着 Flutter 端切换会话**不用担心**后端阻塞——服务端会为新 session 独立建 Agent。但客户端仍应在切 session 时 `cancelToken.cancel()` 中断旧 SSE 流，避免：

1. 已下发的 chunk 继续写入 UI 上
2. 占用的 HTTP 连接浪费

### 10. 登录后自动进入最近会话

`POST /agent/login` 的实现逻辑（`server/app.py:311-356`）：

1. 写入 users + tokens 表，生成新 token
2. 查 `memory.get_latest_session(username)` 取该用户最近活跃 session
3. 若有 → 返回该 session_id；若无 → 立刻创建一个新 Agent 并返回其 session_id

所以客户端 `LoginPage` 登录成功后直接用返回的 `session_id` 进入 `ChatPage`，不需要再调 `/agent/session`。

## 服务端接口 & 数据模型速查

### 统一响应包装

所有 JSON 接口（除 SSE 和静态资源）：

```json
{ "code": 0, "msg": "ok", "data": { ... } }
```

- `code == 0` → 成功，`data` 为业务数据
- `code != 0` → 失败，通常等于 HTTP 状态码，`msg` 为错误描述

### 接口清单（与 `server/app.py` 一致）

| 接口 | 方法 | 参数 | 响应 data | 备注 |
|---|---|---|---|---|
| `/agent/login` | POST | body: `{username}` | `{token, username, session_id}` | 无需 token；自动恢复最近会话 |
| `/agent/logout` | POST | `?token=xxx` | `null` | 驱逐该用户所有 Agent 并清 token |
| `/agent/chat` | POST | `?token=xxx` + body: `{message, session_id}` | **SSE 流**（不走包装） | 见下表 SSE 事件 |
| `/agent/reset` | POST | `?token=xxx` + body: `{session_id}` | `{session_id}` | 返回的是**新**会话 ID；旧会话不删 |
| `/agent/clear` | POST | `?token=xxx` + body: `{session_id}` | `{session_id}` | 清空历史，session_id 继续可用 |
| `/agent/session` | GET | `?token=xxx&session_id=xxx`（可空） | `{username, session_id}` | session_id 为空→返回最近会话 |
| `/agent/sessions` | GET | `?token=xxx` | `{sessions: [...]}` | 按 updated_at 倒序 |
| `/agent/session/delete` | POST | `?token=xxx` + body: `{session_id}` | `{session_id}` | 删 session + 历史 + 快照 |
| `/agent/history` | GET | `?token=xxx&session_id=xxx` | `{messages: [...]}` | session_id 必填 |
| `/agent/reload` | POST | 无 | `{cleared: N}` | **开发用**，清空 Agent 池强制重建 |
| `/agent/index` | GET | 无 | HTML（不包装） | Web 端页面 |
| `/agent/static/*` | GET | 无 | 静态资源 | 挂 `web/` 目录 |
| `/health` | GET | 无 | `{status, users_online}` | `users_online` 实际是池中 Agent 总数 |

除 `/agent/login` 外，所有接口通过 URL query 传 token：`?token=xxx`。

### SSE 事件格式

每个事件两行 `event: <type>\ndata: <JSON>\n\n`。字段定义来自 `src/events.py`：

| event | data 字段 | 说明 |
|---|---|---|
| `thinking` | `{type}` | 开始思考（UI 显示三点动画） |
| `chunk` | `{text}` | 文本增量，追加到当前 Assistant 气泡 |
| `tool_call` | `{type, name, display}` | 工具开始调用；`name` 内部名，`display` 展示文案 |
| `tool_result` | `{type, name, duration_ms, success, display}` | 工具结束 |
| `retry` | `{type, reason, attempt, max_attempts, wait_seconds}` | LLM 429 重试（指数退避） |
| `error` | `{type, message}` | 错误事件 |
| `done` | `{}` | 本轮流结束（由 FastAPI 手动 yield，见 `app.py:400`） |

**chunk 事件特殊**：服务端把 LLM 输出的纯字符串包装成 `event: chunk\ndata: {"text": "..."}`（`app.py:399`），与 AgentEvent dataclass 不同源。客户端解析时需把 `chunk` 也当作事件处理。

### Message 字段（/agent/history 返回）

| 字段 | 类型 | 说明 |
|---|---|---|
| role | String | `user` / `assistant` / `tool` / `system` |
| content | String | 可能为空 |
| tool_calls | List? | 仅 `role=assistant` 且调工具时有 |
| tool_call_id | String? | 仅 `role=tool` 时有 |

**注意**：history 不返回 timestamp；客户端如需显示时间，只能靠当时 send 时现存。

### SessionInfo 字段（/agent/sessions 返回）

| 字段 | 类型 | 说明 |
|---|---|---|
| session_id | String | `username_xxxxxxxx` |
| created_at | double | Unix 秒（含毫秒小数） |
| updated_at | double | Unix 秒 |

**无 title 字段**，客户端需自生成（例如取该 session 第一条 user message 的前 20 字）。Dart 转换：`DateTime.fromMillisecondsSinceEpoch((value * 1000).round())`。

## 实现顺序（分阶段）

### 阶段 1：脚手架 + 数据模型

- `flutter create` 生成项目（已完成）
- 配置 `pubspec.yaml` 依赖
- 实现 `core/constants.dart`、`core/models/*`、`core/storage/auth_storage.dart`

### 阶段 2：网络层

- `core/api/api_client.dart`（Dio + 拦截器 + 统一响应拆包）
- `core/api/api_exception.dart`
- `core/api/sse_client.dart`（SSE 解析）

### 阶段 3：登录流程闭环

- `features/auth/*`（repository / provider / page）
- `main.dart` + `app.dart`：根据 token 切 LoginPage / ChatPage
- 验证：能登录，token 写入 SharedPreferences，APP 重启自动进聊天页

### 阶段 4：对话核心（单 session）

- `features/chat/chat_repository.dart`（/history + /chat SSE）
- `features/chat/chat_provider.dart`
- `features/chat/chat_page.dart` + message/tool/thinking/info 4 个 widget
- 验证：能发消息、流式回复、工具调用气泡、Markdown 渲染

### 阶段 5：会话管理

- `features/sessions/*`（列表 + 新建 + 删除 + 清空）
- 侧边 Drawer 嵌入 chat_page
- 切换会话时 `cancelToken.cancel()` 中断旧 SSE
- 验证：能新建、切换、删除、清空

### 阶段 6：细节打磨

- 滚动跟随 + 回到底部按钮
- 错误提示（401 自动跳登录、403/404 session 失效重建、网络错误 toast）
- 加载态骨架屏
- 深色主题

## 验证方法

### 服务端启动

```powershell
cd D:\projects\light_agent
python server\run.py --reload --host 0.0.0.0 --port 8000
```

### 客户端跑起来

```powershell
cd D:\projects\light_agent\flutter_chat

flutter run -d windows            # Windows 桌面（最快看效果）
flutter run -d emulator-5554      # Android 模拟器（baseUrl=10.0.2.2）
flutter run -d chrome             # Web（CORS 已开）
```

### 端到端冒烟测试

- [ ] 登录成功 → 进入聊天页（自动带上最近 session_id）
- [ ] 发送消息 → 流式回复，文字逐字出现
- [ ] 触发工具调用 → 工具气泡从 running 变 success
- [ ] 侧栏新建会话 → 调 `/agent/reset` 拿到新 session_id，聊天区清空
- [ ] 切换会话 → 旧 SSE 被 CancelToken 取消，新会话历史加载正确
- [ ] 删除当前会话 → 自动切到最近 session 或新建
- [ ] 清空当前会话 → 消息列表清空，session_id 不变
- [ ] APP 重启 → 保持登录态，自动进入上次的会话
- [ ] 退出登录 → 回登录页，token 已清
- [ ] Token 过期（模拟 401） → 回登录页
- [ ] 在别的用户的 session 上触发操作（模拟 403） → 自动清本地 sessionId 并切换

## 关键引用（服务端已存在，不要改）

- 接口定义：`../server/app.py`
- 事件结构：`../src/events.py`
- 历史格式：`../src/memory.py`（`get_all_messages` / `list_sessions_by_user` / `session_owner` / `get_latest_session`）
- Web 交互参考：`../web/app.js`
- 项目总体说明：`../CLAUDE.md`

## 预估工作量

6 个阶段，约 30 个 Dart 文件，合计 ~1500 行代码。每个阶段收尾都可运行可验证。