import 'package:dio/dio.dart';

import '../../core/api/api_client.dart';
import '../../core/api/sse_client.dart';
import '../../core/models/agent_event.dart';
import '../../core/models/message.dart';

class ChatRepository {
  ChatRepository({
    required ApiClient apiClient,
    SseClient? sseClient,
  })  : _api = apiClient,
        _sse = sseClient ?? SseClient(apiClient);

  final ApiClient _api;
  final SseClient _sse;

  /// GET /agent/history —— 返回按时间升序的消息列表。
  Future<List<Message>> loadHistory({
    required String token,
    required String sessionId,
  }) async {
    final data = await _api.get(
      '/agent/history',
      queryParameters: {'token': token, 'session_id': sessionId},
    );
    if (data is! Map) return const [];
    final raw = data['messages'];
    if (raw is! List) return const [];
    return raw
        .whereType<Map>()
        .map((e) => Message.fromJson(Map<String, dynamic>.from(e)))
        .toList(growable: false);
  }

  /// POST /agent/chat —— SSE 流。注意：服务端会自动加时间戳前缀，客户端直接发原文。
  Stream<AgentEvent> chatStream({
    required String token,
    required String sessionId,
    required String message,
    CancelToken? cancelToken,
  }) {
    return _sse.chatStream(
      token: token,
      sessionId: sessionId,
      message: message,
      cancelToken: cancelToken,
    );
  }
}