import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/api/api_client.dart';
import '../../core/storage/auth_storage.dart';
import 'auth_repository.dart';

/// 登录态 —— 全局单一来源。
///
/// - [token] 空 → 未登录
/// - [token] 非空 + [sessionId] 非空 → 已登录且持有会话
class AuthState {
  const AuthState({
    this.token,
    this.username,
    this.sessionId,
  });

  final String? token;
  final String? username;
  final String? sessionId;

  bool get isLoggedIn =>
      (token?.isNotEmpty ?? false) && (username?.isNotEmpty ?? false);

  AuthState copyWith({
    String? token,
    String? username,
    String? sessionId,
    bool clearToken = false,
    bool clearSessionId = false,
  }) {
    return AuthState(
      token: clearToken ? null : (token ?? this.token),
      username: clearToken ? null : (username ?? this.username),
      sessionId:
          clearSessionId ? null : (sessionId ?? this.sessionId),
    );
  }

  static const AuthState empty = AuthState();
}

/// 启动时加载一次的 [AuthStorage] 实例，通过 main.dart 注入 override。
final authStorageProvider = Provider<AuthStorage>((ref) {
  throw UnimplementedError('authStorageProvider 必须在 main.dart 里被 override');
});

/// 全局唯一 [ApiClient]。
final apiClientProvider = Provider<ApiClient>((ref) => ApiClient());

/// 登录相关接口封装。
final authRepositoryProvider = Provider<AuthRepository>(
  (ref) => AuthRepository(ref.watch(apiClientProvider)),
);

/// 当前登录态。初值从 [AuthStorage] 读本地凭据。
final authStateProvider =
    StateNotifierProvider<AuthController, AuthState>((ref) {
  final storage = ref.watch(authStorageProvider);
  return AuthController(
    ref: ref,
    storage: storage,
    initial: AuthState(
      token: storage.token,
      username: storage.username,
      sessionId: storage.sessionId,
    ),
  );
});

class AuthController extends StateNotifier<AuthState> {
  AuthController({
    required this.ref,
    required this.storage,
    required AuthState initial,
  }) : super(initial);

  final Ref ref;
  final AuthStorage storage;

  /// 登录：成功后写 storage + 更新 state。
  Future<void> login(String username) async {
    final trimmed = username.trim();
    if (trimmed.isEmpty) {
      throw Exception('用户名不能为空');
    }
    final repo = ref.read(authRepositoryProvider);
    final result = await repo.login(trimmed);
    await storage.saveCredentials(
      token: result.token,
      username: result.username,
      sessionId: result.sessionId,
    );
    state = AuthState(
      token: result.token,
      username: result.username,
      sessionId: result.sessionId,
    );
  }

  /// 登出：服务端失败不影响本地清理。
  Future<void> logout() async {
    final token = state.token;
    if (token != null && token.isNotEmpty) {
      await ref.read(authRepositoryProvider).logout(token);
    }
    await storage.clear();
    state = AuthState.empty;
  }

  /// 切换当前 session（由 sessions_provider 调用），同步持久化。
  Future<void> setSessionId(String? sessionId) async {
    await storage.setSessionId(sessionId);
    state = state.copyWith(
      sessionId: sessionId,
      clearSessionId: sessionId == null || sessionId.isEmpty,
    );
  }

  /// 凭据失效（401 或本地 session 403/404 兜不回来）→ 清光 + 回登录页。
  Future<void> forceLogout() async {
    await storage.clear();
    state = AuthState.empty;
  }
}