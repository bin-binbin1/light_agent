/// SSE 事件。
///
/// 字段对应 `../server/app.py` 的序列化产物（通过 `dataclasses.asdict`），
/// 字段名来自 `../src/events.py` 的 dataclass（snake_case），
/// 唯一特例：`chunk` 事件不是 AgentEvent，服务端单独包成 `{text: "..."}`；
/// `done` 事件也是服务端手动 yield 的结束标记，data 为空对象 `{}`。
sealed class AgentEvent {
  const AgentEvent();

  /// 根据 SSE 帧的 event 名和 data JSON 构造事件。未知 event 返回 null。
  static AgentEvent? fromSse(String event, Map<String, dynamic> data) {
    switch (event) {
      case 'chunk':
        return ChunkEvent(text: (data['text'] as String?) ?? '');
      case 'thinking':
        return const ThinkingEvent();
      case 'tool_call':
        return ToolCallEvent(
          name: (data['name'] as String?) ?? '',
          display: (data['display'] as String?) ?? '',
        );
      case 'tool_result':
        return ToolResultEvent(
          name: (data['name'] as String?) ?? '',
          durationMs: (data['duration_ms'] as num?)?.toInt() ?? 0,
          success: (data['success'] as bool?) ?? true,
          display: (data['display'] as String?) ?? '',
        );
      case 'retry':
        return RetryEvent(
          reason: (data['reason'] as String?) ?? '',
          attempt: (data['attempt'] as num?)?.toInt() ?? 0,
          maxAttempts: (data['max_attempts'] as num?)?.toInt() ?? 0,
          waitSeconds: (data['wait_seconds'] as num?)?.toDouble() ?? 0.0,
        );
      case 'error':
        return ErrorEvent(message: (data['message'] as String?) ?? '');
      case 'done':
        return const DoneEvent();
      default:
        return null;
    }
  }
}

class ChunkEvent extends AgentEvent {
  const ChunkEvent({required this.text});
  final String text;
}

class ThinkingEvent extends AgentEvent {
  const ThinkingEvent();
}

class ToolCallEvent extends AgentEvent {
  const ToolCallEvent({required this.name, required this.display});
  final String name;
  final String display;
}

class ToolResultEvent extends AgentEvent {
  const ToolResultEvent({
    required this.name,
    required this.durationMs,
    required this.success,
    required this.display,
  });
  final String name;
  final int durationMs;
  final bool success;
  final String display;
}

class RetryEvent extends AgentEvent {
  const RetryEvent({
    required this.reason,
    required this.attempt,
    required this.maxAttempts,
    required this.waitSeconds,
  });
  final String reason;
  final int attempt;
  final int maxAttempts;
  final double waitSeconds;
}

class ErrorEvent extends AgentEvent {
  const ErrorEvent({required this.message});
  final String message;
}

class DoneEvent extends AgentEvent {
  const DoneEvent();
}