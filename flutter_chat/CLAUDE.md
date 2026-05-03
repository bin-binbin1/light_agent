# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

`flutter_chat` 是 Light Agent 的 Flutter 客户端，目标跨平台：Android + Windows 桌面 + Web。**复刻 `../web/app.js` 的全部功能并做状态驱动/流控增强**。服务端在 `../server/app.py`，不要在这里修改任何服务端代码；如需改协议先到 `../CLAUDE.md` 和 `../server/app.py` 对齐后再回来改客户端。

设计细节、目录结构、实现顺序全部写在 `TECHNICAL_PLAN.md`——**改动前先读那份文档**，避免与已定的方案冲突。本文件仅记录跨文件协作的非显然约定。

## 常用命令

```powershell
# 装依赖
cd D:\projects\light_agent\flutter_chat
flutter pub get

# 跑起来（baseUrl 走 --dart-define，别直接改 constants.dart 的默认值）
flutter run -d windows       --dart-define=BASE_URL=http://127.0.0.1:8000
flutter run -d emulator-5554 --dart-define=BASE_URL=http://10.0.2.2:8000
flutter run -d chrome                                          # 默认 localhost:8000
flutter run -d <真机 id>     --dart-define=BASE_URL=http://<开发机 IP>:8000

# 静态检查 / 测试
flutter analyze
flutter test

# Android 打 release APK（只出 arm64，体积小一半）
flutter build apk --release --target-platform android-arm64
# 或按 abi 拆包
flutter build apk --release --split-per-abi

# 服务端（另一个终端）
cd D:\projects\light_agent
python server\run.py --reload --host 0.0.0.0 --port 8000
```

## baseUrl 随运行环境切换

`lib/core/constants.dart` 里 `baseUrl` 是 `String.fromEnvironment('BASE_URL', defaultValue: 'http://localhost:8000')`——**通过 `--dart-define=BASE_URL=...` 注入**，不要直接改默认值（容易手滑提交到 git）。

| 运行目标 | baseUrl |
|---|---|
| Windows 桌面 / iOS 模拟器 | `http://127.0.0.1:8000` |
| Android 模拟器 | `http://10.0.2.2:8000` |
| Android 真机 | `http://<开发机局域网 IP>:8000`|
| Web (flutter run -d chrome) | `http://localhost:8000` |

**Android 侧换 IP 要同步改两处**：`--dart-define=BASE_URL=...` **和** `android/app/src/main/res/xml/network_security_config.xml` 的白名单。没加到白名单的 HTTP 明文请求会在 release 包里被直接拒（debug 默认允许明文，所以 debug 看不出问题，release 才炸）。

## 架构关键点（需跨文件理解）

### 1. 服务端 Agent 池按 `(username, session_id)` 键

`server/app.py` 的 `_agent_pool` 是 `(username, session_id) → Agent` 映射，**同一用户可以同时挂载多个会话 Agent**。对客户端的影响：

- 切会话不会被后端串行化，多 session 并发聊天是安全的
- 但仍应在切 session 时 `cancelToken.cancel()` 中断旧 SSE 流——避免旧 chunk 继续追加到已切走的 UI 上
- `/agent/reset` **不复用**旧 Agent 而是造新 Agent，所以 reset 响应里的 `session_id` **一定是新的**，必须更新本地存储
- `/agent/clear` 只清历史、保留 `session_id`；`/agent/session/delete` 连 session 一起删——客户端要分清别混用

### 2. 服务端会自动给用户消息加时间戳前缀

服务端在把 user message 喂给 LLM 之前，会拼 `[发送时间: YYYY-MM-DD HH:MM:SS]\n<原文>`（见 `server/app.py` 的 `/agent/chat` 处理逻辑）。**客户端不要再自己加**，否则会出现双重时间前缀污染 LLM 上下文。

### 3. 统一响应包装 + SSE 例外

`../server/app.py` 所有 JSON 端点（连异常都经过 `_http_exception_handler` / `_validation_exception_handler` / `_unhandled_exception_handler`）都返回：

```json
{ "code": 0, "msg": "ok", "data": { ... } }
```

所以 `dio` 的 `onError` 里**response.data 仍是合法 JSON**，应当尝试读 `data['code']` / `data['msg']` 包成 `ApiException`，而不是只看 HTTP 状态码。

唯一例外：`/agent/chat` 是 SSE（`text/event-stream`），`/agent/index` 和 `/agent/static/*` 是 HTML/静态，这三个不走 code/msg/data 包装。

### 4. `chunk` 事件与其他 AgentEvent 不同源

`src/events.py` 的 `AgentEvent` 子类（Thinking / ToolCall / ToolResult / Retry / Error）由 `agent.achat_stream()` yield，服务端 `dataclasses.asdict()` 序列化后包成 `event: <type>\ndata: <json>`。

**但 chunk 不是 AgentEvent**——`agent.achat_stream()` 对文本增量直接 yield 字符串，服务端单独包成 `event: chunk\ndata: {"text": "..."}`。还有 `done` 事件是服务端手动 yield 的结束标记，data 为空对象 `{}`。

客户端 `SseClient._parseSseFrame` 必须同时处理这三类。

### 5. 会话归属错误码处理

服务端凡是接收 `session_id` 的接口（`/agent/chat`、`/agent/reset`、`/agent/clear`、`/agent/session`、`/agent/session/delete`、`/agent/history`）都调 `_check_session_ownership`，分三种错误：

| 场景 | HTTP 状态 |
|---|---|
| `session_id` 为空字符串 | 400 |
| `session_id` 不存在 | 404 |
| `session_id` 是别人的 | 403 |

客户端约定：**401 → 回登录页；403/404 → 清本地 sessionId，改调 `GET /agent/session`（不传 session_id 会返回最近一条）拿到新的 sid 兜底，必要时再 `/agent/reset`**。不要把 403/404 直接弹错误 toast 给用户看。

### 6. 登录响应里的 session_id 可直接用

`POST /agent/login` 的响应 `data.session_id` 已经自动恢复了"该用户最近活跃的 session"，没有则当场 create 一个新的。客户端登录成功后**不需要再调 `/agent/session`**，直接进 ChatPage 即可。

### 7. SessionInfo 无 title，客户端自生成

`/agent/sessions` 返回的 `SessionInfo` 只有 `{session_id, created_at, updated_at}`，没有标题。侧栏列表显示时需客户端自生成，建议策略：

- 调 `/agent/history?session_id=xxx` 取第一条 `role=user` 的消息
- 取前 20 个字符作为标题，超长加 `...`
- 结果缓存到内存（`sessions_provider` 的 state 中），避免每次重绘都重新请求

### 8. 流式 Markdown 渲染权衡

流式阶段 `streaming=true` 时用纯文本渲染（`SelectableText`），`done` 后才切到 `MarkdownBody`。原因：流式中途可能出现半截代码块（```开头但还没收到结尾），Markdown 解析器会把后面所有文本当代码。Web 端也是这个策略，不要改。

## Android 打包注意点

- **`abiFilters` 管不了 Flutter engine 的 `.so`**：`android/app/build.gradle.kts` 里 `defaultConfig.ndk { abiFilters += ... }` 只过滤 NDK 自建产物和第三方 AAR 的 native 库。Flutter engine 的 `libflutter.so` / `libapp.so` 由 Flutter Gradle 插件按 `--target-platform` 参数打进去——想只出 arm64 必须用 `flutter build apk --target-platform android-arm64` 或 `--split-per-abi`，也可以在 `android { splits { abi { ... } } }` 里限制（`splits.abi` 对 engine 有效）。
- **`network_security_config.xml` 白名单**：`android/app/src/main/res/xml/network_security_config.xml` 里 `cleartextTrafficPermitted=true` 的域名白名单要包含实际用到的 IP。开发机换网段（公司/家里切换）要改这里+重装 APK，不是改一下 `--dart-define` 就够。
- **release 包签名**：`android/app/build.gradle.kts` 里 `release.signingConfig = signingConfigs.getByName("debug")` 是临时的占位，正式发版前要替换为自己的 keystore 配置。
- **`INTERNET` 权限放 `main/AndroidManifest.xml`**：release 构建不合并 `debug/AndroidManifest.xml`，权限必须在 main 里声明一次，否则 release 包网络请求会直接失败。
- **模拟器 ABI 匹配**：如果 gradle 里 `abiFilters` 限了 `arm64-v8a`，但模拟器镜像是 x86_64，会装不上。要么换模拟器镜像，要么 debug 构建临时不加 abiFilters。

## 常见坑

- **Android 模拟器连不上服务端**：默认 baseUrl 写错了。模拟器里 `127.0.0.1` 指向的是模拟器自己的 loopback，主机在 `10.0.2.2`。
- **Web 端 SSE 被代理截断**：本地开发直连服务端的 `localhost:8000` 没问题；若走 nginx 反代需加 `proxy_buffering off` 和 `X-Accel-Buffering: no`（服务端已经设了该响应头）。
- **CancelToken 未取消旧 SSE**：切 session 时如果忘记 `cancelToken.cancel()`，旧流的 chunk 会继续追加到已切走的 AssistantMsgItem 上，UI 看着很诡异。切 session 的入口都要统一经过同一个 `chat_provider` 方法来处理 cancel。
- **连续快速切会话触发 503-like 状态**：不是服务端的问题，是客户端 state 竞态——需要在 `chat_provider` 里用一个 `int _generation` 或类似机制，确保旧流的 yield 晚于新流启动时会被丢弃。
- **`flutter_markdown` 版本与 `intl` 冲突**：`pubspec.yaml` 指定 `intl: ^0.19.0`，若 `flutter_markdown` 升级到新 major 可能强制降 `intl`，跑 `flutter pub upgrade --major-versions` 前先看 pubspec.lock。
- **`../web/app.js` 的 SSE 解析代码是权威参考**：帧分隔、事件名识别、data 聚合的细节直接抄过来改成 Dart 即可。发现行为不一致先对 `app.js`。

## 关键引用

| 资源 | 路径 | 用途 |
|---|---|---|
| 技术方案（必读） | `TECHNICAL_PLAN.md` | 目录结构 / 设计决策 / 实现顺序 / 冒烟测试清单 |
| 服务端项目说明 | `../CLAUDE.md` | Light Agent 整体架构与服务端内部细节 |
| 服务端接口源码 | `../server/app.py` | 所有协议细节以此为准（行号会漂，以路由/函数名为准） |
| 服务端事件定义 | `../src/events.py` | AgentEvent dataclass 字段名 |
| Web 交互参考 | `../web/app.js` | SSE 解析 / Markdown 策略的权威实现 |
| 记忆层接口 | `../src/memory.py` | `get_all_messages` / `list_sessions_by_user` / `session_owner` 的字段定义 |