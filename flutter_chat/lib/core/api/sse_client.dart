import 'dart:async';
import 'dart:convert';

import 'package:dio/dio.dart';

import '../models/agent_event.dart';
import 'api_client.dart';
import 'api_exception.dart';

/// SSE 客户端：`/agent/chat` 流式对话。
///
/// 帧格式与 `../web/app.js::parseSSEFrame` 对齐：
/// - 帧分隔：`\n\n`
/// - 行前缀：`event:` / `data:`
/// - 未知 event 会被忽略（由 [AgentEvent.fromSse] 决定）
class SseClient {
  SseClient(this._client);

  final ApiClient _client;

  /// 发起 `/agent/chat` 并 yield [AgentEvent] 流。
  ///
  /// 流结束条件：
  /// - 收到 `event: done`（yield [DoneEvent] 后自然结束）
  /// - 收到 `event: error`（yield [ErrorEvent] 后自然结束）
  /// - 服务端关闭连接
  /// - [cancelToken] 被取消（抛 [NetworkException]）
  Stream<AgentEvent> chatStream({
    required String token,
    required String sessionId,
    required String message,
    CancelToken? cancelToken,
  }) async* {
    final Response<ResponseBody> resp;
    try {
      resp = await _client.rawDio.post<ResponseBody>(
        '/agent/chat',
        data: {'message': message, 'session_id': sessionId},
        queryParameters: {'token': token},
        options: Options(
          responseType: ResponseType.stream,
          headers: {'Accept': 'text/event-stream'},
        ),
        cancelToken: cancelToken,
      );
    } on DioException catch (e) {
      if (e.type == DioExceptionType.cancel) {
        throw const NetworkException('请求已取消');
      }
      throw NetworkException(e.message ?? '连接失败');
    }

    final status = resp.statusCode ?? 0;
    if (status < 200 || status >= 300) {
      throw ApiException(
        code: status,
        message: 'HTTP $status',
        statusCode: status,
      );
    }

    final body = resp.data;
    if (body == null) {
      throw const NetworkException('无响应体');
    }

    // 分帧 + UTF-8 流式解码
    String buffer = '';
    final stream = body.stream
        .cast<List<int>>()
        .transform(const Utf8Decoder(allowMalformed: true));

    await for (final chunk in stream) {
      buffer += chunk;
      // 按 \n\n 切帧，最后一段可能不完整，留到下次
      final frames = buffer.split('\n\n');
      buffer = frames.removeLast();
      for (final frame in frames) {
        final event = _parseFrame(frame);
        if (event != null) {
          yield event;
          if (event is DoneEvent || event is ErrorEvent) return;
        }
      }
    }

    // 残留缓冲：按最后一帧再试一次
    if (buffer.trim().isNotEmpty) {
      final event = _parseFrame(buffer);
      if (event != null) yield event;
    }
  }

  AgentEvent? _parseFrame(String frame) {
    String eventName = 'message';
    final dataBuf = StringBuffer();
    for (final line in const LineSplitter().convert(frame)) {
      if (line.startsWith('event:')) {
        eventName = line.substring(6).trim();
      } else if (line.startsWith('data:')) {
        dataBuf.write(line.substring(5).trim());
      }
    }

    final dataStr = dataBuf.toString();
    if (dataStr.isEmpty && eventName == 'message') return null;

    Map<String, dynamic> data = const {};
    if (dataStr.isNotEmpty) {
      try {
        final decoded = jsonDecode(dataStr);
        if (decoded is Map<String, dynamic>) data = decoded;
      } catch (_) {
        // 容错：非 JSON 时跳过，保持与 web 端一致
      }
    }

    return AgentEvent.fromSse(eventName, data);
  }
}