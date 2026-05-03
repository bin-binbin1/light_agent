import '../../core/api/api_client.dart';
import '../../core/models/session_info.dart';

class SessionRepository {
  SessionRepository(this._client);

  final ApiClient _client;

  /// GET /agent/sessions → 按 updated_at 倒序。
  Future<List<SessionInfo>> list(String token) async {
    final data = await _client.get(
      '/agent/sessions',
      queryParameters: {'token': token},
    );
    if (data is! Map) return const [];
    final raw = data['sessions'];
    if (raw is! List) return const [];
    return raw
        .whereType<Map>()
        .map((e) => SessionInfo.fromJson(Map<String, dynamic>.from(e)))
        .toList(growable: false);
  }

  /// POST /agent/reset → 返回新 session_id（旧会话数据保留）。
  Future<String> reset({required String token, required String sessionId}) async {
    final data = await _client.post(
      '/agent/reset',
      queryParameters: {'token': token},
      data: {'session_id': sessionId},
    );
    if (data is Map && data['session_id'] is String) {
      return data['session_id'] as String;
    }
    throw Exception('reset 响应格式错误');
  }

  /// POST /agent/clear → 清空指定 session 的历史，session_id 保持不变。
  Future<void> clear({required String token, required String sessionId}) async {
    await _client.post(
      '/agent/clear',
      queryParameters: {'token': token},
      data: {'session_id': sessionId},
    );
  }

  /// POST /agent/session/delete → 删 session 和所有相关数据（不可恢复）。
  Future<void> delete({required String token, required String sessionId}) async {
    await _client.post(
      '/agent/session/delete',
      queryParameters: {'token': token},
      data: {'session_id': sessionId},
    );
  }

  /// GET /agent/session?session_id=（空则取最近）→ 返回 {username, session_id}。
  Future<String?> currentSession({
    required String token,
    String? sessionId,
  }) async {
    final data = await _client.get(
      '/agent/session',
      queryParameters: {
        'token': token,
        if (sessionId != null && sessionId.isNotEmpty) 'session_id': sessionId,
      },
    );
    if (data is Map) return data['session_id'] as String?;
    return null;
  }

  /// 读取某 session 的首条 user 消息（客户端自生成会话标题用）。
  Future<String?> firstUserMessage({
    required String token,
    required String sessionId,
  }) async {
    final data = await _client.get(
      '/agent/history',
      queryParameters: {'token': token, 'session_id': sessionId},
    );
    if (data is! Map) return null;
    final raw = data['messages'];
    if (raw is! List) return null;
    for (final m in raw) {
      if (m is Map && m['role'] == 'user') {
        final content = (m['content'] as String?) ?? '';
        // 服务端给 user message 自动加 `[发送时间: ...]\n` 前缀，取第二行才是原文
        if (content.startsWith('[发送时间: ')) {
          final idx = content.indexOf('\n');
          if (idx > 0 && idx + 1 < content.length) {
            return content.substring(idx + 1);
          }
        }
        return content;
      }
    }
    return null;
  }
}