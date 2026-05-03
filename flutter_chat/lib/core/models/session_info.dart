/// /agent/sessions 返回的单个会话条目。
///
/// 服务端字段（见 `../src/memory.py::list_sessions_by_user`）：
/// - session_id: String，格式 `username_xxxxxxxx`
/// - created_at / updated_at: double，Unix 秒（含毫秒小数）
///
/// **注意**：服务端不返回 title，客户端自生成（取首条 user message 前 20 字）。
class SessionInfo {
  const SessionInfo({
    required this.sessionId,
    required this.createdAt,
    required this.updatedAt,
    this.title,
  });

  final String sessionId;
  final DateTime createdAt;
  final DateTime updatedAt;

  /// 客户端自生成的标题（本地缓存到 sessions_provider，非服务端字段）。
  final String? title;

  SessionInfo copyWith({String? title}) => SessionInfo(
        sessionId: sessionId,
        createdAt: createdAt,
        updatedAt: updatedAt,
        title: title ?? this.title,
      );

  factory SessionInfo.fromJson(Map<String, dynamic> json) {
    return SessionInfo(
      sessionId: (json['session_id'] as String?) ?? '',
      createdAt: _parseUnixSeconds(json['created_at']),
      updatedAt: _parseUnixSeconds(json['updated_at']),
    );
  }

  static DateTime _parseUnixSeconds(Object? value) {
    if (value is num) {
      return DateTime.fromMillisecondsSinceEpoch((value * 1000).round());
    }
    return DateTime.fromMillisecondsSinceEpoch(0);
  }
}