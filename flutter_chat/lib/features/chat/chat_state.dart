import '../../core/models/message.dart';

/// 工具调用状态。
enum ToolStatus { running, success, fail }

/// 聊天列表单个条目。Sealed class 让渲染分发写成穷尽式 switch。
sealed class ChatItem {
  const ChatItem();
}

class UserMsgItem extends ChatItem {
  UserMsgItem({required this.content});
  final String content;
}

/// Assistant 消息气泡。`content` / `streaming` 可变——流式阶段追加文本。
class AssistantMsgItem extends ChatItem {
  AssistantMsgItem({
    this.content = '',
    this.streaming = true,
    this.toolCalls,
  });

  String content;
  bool streaming;

  /// 从历史回填时会有 tool_calls（当前实现仅保留字段，不在 UI 展示）。
  List<ToolCall>? toolCalls;
}

class ToolItem extends ChatItem {
  ToolItem({
    required this.name,
    required this.display,
    this.status = ToolStatus.running,
    this.durationMs,
  });

  final String name;
  String display;
  ToolStatus status;
  int? durationMs;
}

class InfoItem extends ChatItem {
  const InfoItem(this.text);
  final String text;
}

/// 聊天页面状态。
class ChatState {
  const ChatState({
    this.items = const [],
    this.isStreaming = false,
    this.isLoadingHistory = false,
    this.error,
  });

  final List<ChatItem> items;
  final bool isStreaming;
  final bool isLoadingHistory;
  final String? error;

  ChatState copyWith({
    List<ChatItem>? items,
    bool? isStreaming,
    bool? isLoadingHistory,
    String? error,
    bool clearError = false,
  }) {
    return ChatState(
      items: items ?? this.items,
      isStreaming: isStreaming ?? this.isStreaming,
      isLoadingHistory: isLoadingHistory ?? this.isLoadingHistory,
      error: clearError ? null : (error ?? this.error),
    );
  }

  static const ChatState initial = ChatState();
}