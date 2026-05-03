/// /agent/history 返回的单条消息。
///
/// 字段定义来自 `../src/memory.py::_rows_to_messages`：
/// - role: 'user' / 'assistant' / 'tool' / 'system'
/// - content: 可能为空字符串
/// - tool_calls: 仅 role=assistant 且调工具时存在（OpenAI 原始数组）
/// - tool_call_id: 仅 role=tool 时存在
///
/// 注意：history 不返回 timestamp。
class Message {
  const Message({
    required this.role,
    required this.content,
    this.toolCalls,
    this.toolCallId,
  });

  final String role;
  final String content;
  final List<ToolCall>? toolCalls;
  final String? toolCallId;

  bool get isUser => role == 'user';
  bool get isAssistant => role == 'assistant';
  bool get isTool => role == 'tool';
  bool get isSystem => role == 'system';

  factory Message.fromJson(Map<String, dynamic> json) {
    final rawCalls = json['tool_calls'];
    List<ToolCall>? calls;
    if (rawCalls is List && rawCalls.isNotEmpty) {
      calls = rawCalls
          .whereType<Map<String, dynamic>>()
          .map(ToolCall.fromJson)
          .toList(growable: false);
    }
    return Message(
      role: (json['role'] as String?) ?? '',
      content: (json['content'] as String?) ?? '',
      toolCalls: calls,
      toolCallId: json['tool_call_id'] as String?,
    );
  }
}

/// OpenAI 风格的 tool_call 条目。
class ToolCall {
  const ToolCall({
    required this.id,
    required this.function,
    this.type = 'function',
  });

  final String id;
  final String type;
  final ToolCallFunction function;

  factory ToolCall.fromJson(Map<String, dynamic> json) {
    final func = json['function'];
    return ToolCall(
      id: (json['id'] as String?) ?? '',
      type: (json['type'] as String?) ?? 'function',
      function: func is Map<String, dynamic>
          ? ToolCallFunction.fromJson(func)
          : const ToolCallFunction(name: '', arguments: ''),
    );
  }
}

class ToolCallFunction {
  const ToolCallFunction({required this.name, required this.arguments});

  final String name;

  /// 服务端存的是 JSON 字符串（可能未必合法），UI 展示时按字符串处理即可。
  final String arguments;

  factory ToolCallFunction.fromJson(Map<String, dynamic> json) {
    final args = json['arguments'];
    return ToolCallFunction(
      name: (json['name'] as String?) ?? '',
      arguments: args is String ? args : (args == null ? '' : args.toString()),
    );
  }
}