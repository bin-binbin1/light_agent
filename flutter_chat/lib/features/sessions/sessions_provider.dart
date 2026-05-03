import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/api/api_exception.dart';
import '../../core/models/session_info.dart';
import '../auth/auth_provider.dart';
import '../chat/chat_provider.dart';
import 'session_repository.dart';

final sessionRepositoryProvider = Provider<SessionRepository>(
  (ref) => SessionRepository(ref.watch(apiClientProvider)),
);

/// 会话列表 UI 状态。
class SessionsState {
  const SessionsState({
    this.sessions = const [],
    this.loading = false,
    this.error,
  });

  final List<SessionInfo> sessions;
  final bool loading;
  final String? error;

  SessionsState copyWith({
    List<SessionInfo>? sessions,
    bool? loading,
    String? error,
    bool clearError = false,
  }) {
    return SessionsState(
      sessions: sessions ?? this.sessions,
      loading: loading ?? this.loading,
      error: clearError ? null : (error ?? this.error),
    );
  }

  static const SessionsState initial = SessionsState();
}

final sessionsProvider =
    StateNotifierProvider<SessionsController, SessionsState>(
  (ref) => SessionsController(ref),
);

class SessionsController extends StateNotifier<SessionsState> {
  SessionsController(this.ref) : super(SessionsState.initial) {
    // token 变化（登录/登出）→ 重新加载
    ref.listen<String?>(
      authStateProvider.select((a) => a.token),
      (prev, next) {
        if ((next ?? '').isNotEmpty && prev != next) {
          refresh();
        } else if ((next ?? '').isEmpty) {
          state = SessionsState.initial;
        }
      },
    );
    // 启动时若已登录就立刻拉一次
    final token = ref.read(authStateProvider).token ?? '';
    if (token.isNotEmpty) {
      // ignore: discarded_futures
      refresh();
    }
  }

  final Ref ref;

  /// 客户端自生成的标题缓存：session_id → title。
  final Map<String, String> _titleCache = {};

  String _token() => ref.read(authStateProvider).token ?? '';

  Future<void> refresh() async {
    final token = _token();
    if (token.isEmpty) return;
    state = state.copyWith(loading: true, clearError: true);
    try {
      final list = await ref.read(sessionRepositoryProvider).list(token);
      // 把已缓存的 title 回填到新 list 上
      final merged = [
        for (final s in list)
          s.copyWith(title: _titleCache[s.sessionId]),
      ];
      state = state.copyWith(sessions: merged, loading: false);
    } on ApiException catch (e) {
      if (e.isUnauthorized) {
        await ref.read(authStateProvider.notifier).forceLogout();
        return;
      }
      state = state.copyWith(loading: false, error: e.message);
    } catch (e) {
      state = state.copyWith(loading: false, error: '加载会话失败: $e');
    }
  }

  /// 切换当前 session。先中断正在进行的 SSE 流再切。
  Future<void> switchTo(String sessionId) async {
    if (sessionId.isEmpty) return;
    final auth = ref.read(authStateProvider);
    if (auth.sessionId == sessionId) return;
    // 中断旧流，避免其 chunk 追加到已切走的 UI 上
    ref.read(chatProvider.notifier).cancelActive();
    await ref.read(authStateProvider.notifier).setSessionId(sessionId);
    // authStateProvider 变更后 chatProvider 被 autoDispose 重建并自动 loadHistory
  }

  /// 新建会话：/agent/reset → 拿新 session_id → 切过去 → 刷新列表。
  Future<void> createNew() async {
    final token = _token();
    final auth = ref.read(authStateProvider);
    final currentSid = auth.sessionId ?? '';
    if (token.isEmpty || currentSid.isEmpty) return;
    try {
      ref.read(chatProvider.notifier).cancelActive();
      final newSid = await ref
          .read(sessionRepositoryProvider)
          .reset(token: token, sessionId: currentSid);
      await ref.read(authStateProvider.notifier).setSessionId(newSid);
      await refresh();
    } on ApiException catch (e) {
      if (e.isUnauthorized) {
        await ref.read(authStateProvider.notifier).forceLogout();
        return;
      }
      if (e.isSessionInvalid) {
        // 当前 session 已失效，清本地、拉最近、还是不行就新建
        await _recoverAfterInvalidSession();
        return;
      }
      state = state.copyWith(error: e.message);
    } catch (e) {
      state = state.copyWith(error: '新建会话失败: $e');
    }
  }

  /// 清空当前会话历史（session_id 不变）。
  Future<void> clearCurrent() async {
    final token = _token();
    final auth = ref.read(authStateProvider);
    final sid = auth.sessionId ?? '';
    if (token.isEmpty || sid.isEmpty) return;
    try {
      ref.read(chatProvider.notifier).cancelActive();
      await ref
          .read(sessionRepositoryProvider)
          .clear(token: token, sessionId: sid);
      _titleCache.remove(sid);
      // 重新加载当前会话的空历史
      await ref.read(chatProvider.notifier).loadHistory();
      await refresh();
    } on ApiException catch (e) {
      if (e.isUnauthorized) {
        await ref.read(authStateProvider.notifier).forceLogout();
        return;
      }
      if (e.isSessionInvalid) {
        await _recoverAfterInvalidSession();
        return;
      }
      state = state.copyWith(error: e.message);
    } catch (e) {
      state = state.copyWith(error: '清空失败: $e');
    }
  }

  /// 删除指定 session。若删的是当前 session，自动 fallback 到最近/新建。
  Future<void> delete(String sessionId) async {
    final token = _token();
    if (token.isEmpty || sessionId.isEmpty) return;
    final auth = ref.read(authStateProvider);
    final deletingCurrent = auth.sessionId == sessionId;
    try {
      if (deletingCurrent) {
        ref.read(chatProvider.notifier).cancelActive();
      }
      await ref
          .read(sessionRepositoryProvider)
          .delete(token: token, sessionId: sessionId);
      _titleCache.remove(sessionId);

      if (deletingCurrent) {
        // 问服务端要最近 session；没有就新建
        final repo = ref.read(sessionRepositoryProvider);
        String? fallback;
        try {
          fallback = await repo.currentSession(token: token);
        } on ApiException {
          fallback = null;
        }
        if (fallback != null && fallback.isNotEmpty) {
          await ref.read(authStateProvider.notifier).setSessionId(fallback);
        } else {
          // 空仓库：靠 /agent/reset 从当前（已删掉的）session 新建是不行的，
          // 这里用 /agent/login 反流程太重。服务端一定还剩至少一个 session
          // （delete 只删指定条目，不会删到空为止）→ currentSession 理论上会有返回。
          // 兜底：置空 sessionId，ChatPage 会显示空态。
          await ref.read(authStateProvider.notifier).setSessionId(null);
        }
      }
      await refresh();
    } on ApiException catch (e) {
      if (e.isUnauthorized) {
        await ref.read(authStateProvider.notifier).forceLogout();
        return;
      }
      state = state.copyWith(error: e.message);
    } catch (e) {
      state = state.copyWith(error: '删除失败: $e');
    }
  }

  /// 为某 session 生成/获取标题（取首条 user 消息前 20 字）。
  Future<String?> ensureTitle(String sessionId) async {
    final cached = _titleCache[sessionId];
    if (cached != null) return cached;
    final token = _token();
    if (token.isEmpty) return null;
    try {
      final firstMsg = await ref
          .read(sessionRepositoryProvider)
          .firstUserMessage(token: token, sessionId: sessionId);
      if (firstMsg == null || firstMsg.isEmpty) return null;
      final title = firstMsg.length > 20 ? '${firstMsg.substring(0, 20)}…' : firstMsg;
      _titleCache[sessionId] = title;
      // 把 title 拍回 state
      final idx =
          state.sessions.indexWhere((s) => s.sessionId == sessionId);
      if (idx >= 0) {
        final updated = List<SessionInfo>.of(state.sessions);
        updated[idx] = updated[idx].copyWith(title: title);
        state = state.copyWith(sessions: updated);
      }
      return title;
    } catch (_) {
      return null;
    }
  }

  Future<void> _recoverAfterInvalidSession() async {
    final token = _token();
    await ref.read(authStateProvider.notifier).setSessionId(null);
    try {
      final sid = await ref
          .read(sessionRepositoryProvider)
          .currentSession(token: token);
      if (sid != null && sid.isNotEmpty) {
        await ref.read(authStateProvider.notifier).setSessionId(sid);
      }
      await refresh();
    } catch (_) {
      // 保持 sessionId 为空
    }
  }
}