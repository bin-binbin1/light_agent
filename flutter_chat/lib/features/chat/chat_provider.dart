import 'dart:async';

import 'package:dio/dio.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/api/api_exception.dart';
import '../../core/models/agent_event.dart';
import '../../core/models/message.dart';
import '../auth/auth_provider.dart';
import 'chat_repository.dart';
import 'chat_state.dart';

final chatRepositoryProvider = Provider<ChatRepository>(
  (ref) => ChatRepository(apiClient: ref.watch(apiClientProvider)),
);

/// 聊天状态。按当前 sessionId 绑定——session 切换后新实例会自动创建。
final chatProvider =
    StateNotifierProvider.autoDispose<ChatController, ChatState>((ref) {
  final auth = ref.watch(authStateProvider);
  final ctrl = ChatController(
    ref: ref,
    repository: ref.watch(chatRepositoryProvider),
    token: auth.token ?? '',
    sessionId: auth.sessionId ?? '',
  );
  ref.onDispose(ctrl._disposeCancel);
  // 初始化：有 session 就拉一次历史
  if ((auth.token ?? '').isNotEmpty && (auth.sessionId ?? '').isNotEmpty) {
    // ignore: discarded_futures
    ctrl.loadHistory();
  }
  return ctrl;
});

class ChatController extends StateNotifier<ChatState> {
  ChatController({
    required this.ref,
    required this.repository,
    required this.token,
    required this.sessionId,
  }) : super(ChatState.initial);

  final Ref ref;
  final ChatRepository repository;
  final String token;
  final String sessionId;

  /// 流代数：每次发送递增。旧流 yield 的事件若代数不匹配会被丢弃。
  int _generation = 0;
  CancelToken? _activeCancel;

  /// 加载历史消息填充到列表。
  Future<void> loadHistory() async {
    if (token.isEmpty || sessionId.isEmpty) return;
    state = state.copyWith(isLoadingHistory: true, clearError: true);
    try {
      final messages = await repository.loadHistory(
        token: token,
        sessionId: sessionId,
      );
      state = state.copyWith(
        items: _messagesToItems(messages),
        isLoadingHistory: false,
      );
    } on ApiException catch (e) {
      if (e.isUnauthorized) {
        await ref.read(authStateProvider.notifier).forceLogout();
        return;
      }
      if (e.isSessionInvalid) {
        // 历史加载时 session 失效：清本地 sessionId，让上层重拉（sessions_provider 兜底）
        await ref.read(authStateProvider.notifier).setSessionId(null);
        return;
      }
      state = state.copyWith(
        isLoadingHistory: false,
        error: e.message,
      );
    } catch (e) {
      state = state.copyWith(
        isLoadingHistory: false,
        error: '加载历史失败: $e',
      );
    }
  }

  /// 发送一条消息 + 订阅 SSE。
  Future<void> sendMessage(String text) async {
    final msg = text.trim();
    if (msg.isEmpty) return;
    if (state.isStreaming) return;
    if (token.isEmpty || sessionId.isEmpty) return;

    // 取消上一条（理论上 isStreaming=false 时不存在，但保险起见）
    _activeCancel?.cancel('new message');

    final gen = ++_generation;
    final cancel = CancelToken();
    _activeCancel = cancel;

    // 1) 追加 user 气泡 + 占位 assistant 气泡
    final items = List<ChatItem>.of(state.items)
      ..add(UserMsgItem(content: msg))
      ..add(AssistantMsgItem());
    state = state.copyWith(
      items: items,
      isStreaming: true,
      clearError: true,
    );

    try {
      await for (final event in repository.chatStream(
        token: token,
        sessionId: sessionId,
        message: msg,
        cancelToken: cancel,
      )) {
        if (gen != _generation) break; // 代数过期（用户切 session / 发了新消息）
        _onEvent(event);
        if (event is DoneEvent || event is ErrorEvent) break;
      }
    } on ApiException catch (e) {
      if (e.isUnauthorized) {
        await ref.read(authStateProvider.notifier).forceLogout();
        return;
      }
      if (e.isSessionInvalid) {
        await ref.read(authStateProvider.notifier).setSessionId(null);
        return;
      }
      if (gen == _generation) _appendInfo('❌ ${e.message}');
    } catch (e) {
      if (gen == _generation) _appendInfo('❌ 连接异常: $e');
    } finally {
      if (gen == _generation) {
        _finalizeStream();
      }
    }
  }

  /// 切换 session 或发新消息前，中断旧流。
  void cancelActive() {
    _activeCancel?.cancel('cancelled by user');
    _activeCancel = null;
    _generation++;
    if (state.isStreaming) {
      _finalizeStream();
    }
  }

  void _onEvent(AgentEvent event) {
    switch (event) {
      case ChunkEvent(:final text):
        _appendChunk(text);
      case ThinkingEvent():
        // UI 侧 AssistantMsgItem.streaming=true 时自动显示 thinking dots
        break;
      case ToolCallEvent(:final name, :final display):
        _onToolCall(name: name, display: display);
      case ToolResultEvent(
          :final name,
          :final durationMs,
          :final success,
          :final display,
        ):
        _onToolResult(
          name: name,
          durationMs: durationMs,
          success: success,
          display: display,
        );
      case RetryEvent(
          :final attempt,
          :final maxAttempts,
          :final waitSeconds,
        ):
        _appendInfo(
          '⏳ 限流重试 $attempt/$maxAttempts（等待 ${waitSeconds.toStringAsFixed(1)}s）',
        );
      case ErrorEvent(:final message):
        _appendInfo('❌ $message');
      case DoneEvent():
        break;
    }
  }

  void _appendChunk(String text) {
    if (text.isEmpty) return;
    final items = List<ChatItem>.of(state.items);
    final last = items.isEmpty ? null : items.last;
    if (last is AssistantMsgItem) {
      last.content += text;
      state = state.copyWith(items: items);
    } else {
      final item = AssistantMsgItem(content: text);
      items.add(item);
      state = state.copyWith(items: items);
    }
  }

  void _onToolCall({required String name, required String display}) {
    final items = List<ChatItem>.of(state.items);
    // 固化当前 assistant 气泡：若为空则丢弃，否则停流并保留
    if (items.isNotEmpty && items.last is AssistantMsgItem) {
      final last = items.last as AssistantMsgItem;
      if (last.content.isEmpty) {
        items.removeLast();
      } else {
        last.streaming = false;
      }
    }
    items.add(ToolItem(name: name, display: display));
    items.add(AssistantMsgItem()); // 工具执行后可能还有 assistant 输出
    state = state.copyWith(items: items);
  }

  void _onToolResult({
    required String name,
    required int durationMs,
    required bool success,
    required String display,
  }) {
    final items = List<ChatItem>.of(state.items);
    // 倒序找同名 running ToolItem
    for (var i = items.length - 1; i >= 0; i--) {
      final item = items[i];
      if (item is ToolItem &&
          item.name == name &&
          item.status == ToolStatus.running) {
        item.status = success ? ToolStatus.success : ToolStatus.fail;
        item.durationMs = durationMs;
        if (display.isNotEmpty) item.display = display;
        break;
      }
    }
    state = state.copyWith(items: items);
  }

  void _appendInfo(String text) {
    final items = List<ChatItem>.of(state.items)..add(InfoItem(text));
    state = state.copyWith(items: items);
  }

  void _finalizeStream() {
    final items = List<ChatItem>.of(state.items);
    // 收尾：最后一个 Assistant 气泡停流；若为空则移除
    if (items.isNotEmpty && items.last is AssistantMsgItem) {
      final last = items.last as AssistantMsgItem;
      if (last.content.isEmpty) {
        items.removeLast();
      } else {
        last.streaming = false;
      }
    }
    _activeCancel = null;
    state = state.copyWith(items: items, isStreaming: false);
  }

  void _disposeCancel() {
    _activeCancel?.cancel('provider disposed');
    _activeCancel = null;
  }

  /// 把 /agent/history 的 Message 序列转换成 ChatItem 序列。
  ///
  /// 规则：
  /// - role=user → UserMsgItem
  /// - role=assistant，content 非空 → AssistantMsgItem(streaming=false)
  /// - role=assistant，有 tool_calls → 为每个 tool_call 生成 ToolItem(success)
  /// - role=tool → 附加到最近 ToolItem 的 display（历史里工具调用已结束，一律当 success）
  /// - role=system → 跳过（一般不展示）
  List<ChatItem> _messagesToItems(List<Message> messages) {
    final items = <ChatItem>[];
    for (final m in messages) {
      if (m.isUser) {
        items.add(UserMsgItem(content: _stripTimePrefix(m.content)));
      } else if (m.isAssistant) {
        if (m.content.isNotEmpty) {
          items.add(AssistantMsgItem(content: m.content, streaming: false));
        }
        final calls = m.toolCalls;
        if (calls != null) {
          for (final tc in calls) {
            items.add(ToolItem(
              name: tc.function.name,
              display: tc.function.name,
              status: ToolStatus.success,
            ));
          }
        }
      } else if (m.isTool) {
        // 历史中工具结果以 role=tool 出现，一般是 JSON，展示上不铺开
        continue;
      }
    }
    return items;
  }

  /// 服务端会给 user message 加 `[发送时间: YYYY-MM-DD HH:MM:SS]\n` 前缀，
  /// 历史回显时把这一行剥掉让用户看到原文。
  String _stripTimePrefix(String content) {
    if (content.startsWith('[发送时间: ')) {
      final idx = content.indexOf('\n');
      if (idx > 0 && idx + 1 < content.length) {
        return content.substring(idx + 1);
      }
    }
    return content;
  }
}