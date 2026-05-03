import '../../core/api/api_client.dart';

/// 登录接口返回。
class LoginResult {
  const LoginResult({
    required this.token,
    required this.username,
    required this.sessionId,
  });

  final String token;
  final String username;
  final String sessionId;

  factory LoginResult.fromJson(Map<String, dynamic> json) => LoginResult(
        token: (json['token'] as String?) ?? '',
        username: (json['username'] as String?) ?? '',
        sessionId: (json['session_id'] as String?) ?? '',
      );
}

class AuthRepository {
  AuthRepository(this._client);

  final ApiClient _client;

  /// POST /agent/login —— 无需 token，服务端自动恢复最近 session 或新建。
  Future<LoginResult> login(String username) async {
    final data = await _client.post(
      '/agent/login',
      data: {'username': username},
    );
    if (data is! Map) {
      throw Exception('登录响应格式错误');
    }
    return LoginResult.fromJson(Map<String, dynamic>.from(data));
  }

  /// POST /agent/logout —— 失败静默（TECHNICAL_PLAN：即便服务端没响应也要清本地凭据）。
  Future<void> logout(String token) async {
    try {
      await _client.post(
        '/agent/logout',
        queryParameters: {'token': token},
      );
    } catch (_) {
      // 忽略：本地 clear 是最终保证
    }
  }

  /// GET /agent/session —— 探活 + 取当前/最近 session。
  ///
  /// 返回 `{username, session_id}`；APP 启动带着本地 token 调这个能验证 token 是否仍有效。
  Future<Map<String, dynamic>> currentSession({
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
    if (data is Map) return Map<String, dynamic>.from(data);
    return const {};
  }
}