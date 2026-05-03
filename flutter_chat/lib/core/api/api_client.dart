import 'package:dio/dio.dart';

import '../constants.dart';
import 'api_exception.dart';

/// Dio 封装：JSON 端点统一 `{code, msg, data}` 拆包。
///
/// - `get` / `post` 返回的就是 `data` 字段（已拆包）。
/// - 失败（code != 0 或 HTTP 4xx/5xx）统一抛 [ApiException]。
/// - SSE 端点请用 [rawDio] 直接走 `ResponseType.stream`，不经过拆包。
class ApiClient {
  ApiClient({Dio? dio})
      : rawDio = dio ??
            Dio(
              BaseOptions(
                baseUrl: baseUrl,
                connectTimeout: const Duration(seconds: 10),
                receiveTimeout: const Duration(seconds: 30),
                contentType: 'application/json',
                responseType: ResponseType.json,
                validateStatus: (_) => true, // 4xx/5xx 也走正常分支，统一在拆包里判断
              ),
            );

  /// 暴露原始 Dio（SSE 用）。
  final Dio rawDio;

  /// GET 并拆包。返回值即 `response.data['data']`，可能为 null / Map / List。
  Future<dynamic> get(
    String path, {
    Map<String, dynamic>? queryParameters,
    CancelToken? cancelToken,
  }) async {
    try {
      final resp = await rawDio.get<dynamic>(
        path,
        queryParameters: queryParameters,
        cancelToken: cancelToken,
      );
      return _unwrap(resp);
    } on DioException catch (e) {
      throw _toApiException(e);
    }
  }

  /// POST 并拆包。
  Future<dynamic> post(
    String path, {
    Object? data,
    Map<String, dynamic>? queryParameters,
    CancelToken? cancelToken,
  }) async {
    try {
      final resp = await rawDio.post<dynamic>(
        path,
        data: data,
        queryParameters: queryParameters,
        cancelToken: cancelToken,
      );
      return _unwrap(resp);
    } on DioException catch (e) {
      throw _toApiException(e);
    }
  }

  /// 拆 `{code, msg, data}`。业务失败抛 [ApiException]。
  dynamic _unwrap(Response<dynamic> resp) {
    final body = resp.data;
    final status = resp.statusCode ?? 0;

    if (body is Map) {
      final code = (body['code'] as num?)?.toInt();
      final msg = (body['msg'] as String?) ?? '';
      final data = body['data'];
      if (code == 0) return data;
      // 业务失败：优先用 body 里的 code/msg，状态码仅作辅助
      throw ApiException(
        code: code ?? status,
        message: msg.isNotEmpty ? msg : 'HTTP $status',
        statusCode: status,
      );
    }

    // 非 JSON 响应（理论上 JSON 端点不会走到这里）
    if (status >= 200 && status < 300) return body;
    throw ApiException(
      code: status,
      message: 'HTTP $status',
      statusCode: status,
    );
  }

  ApiException _toApiException(DioException e) {
    // 服务端异常也是合法 JSON，尝试再次拆一下
    final resp = e.response;
    if (resp != null && resp.data is Map) {
      final body = resp.data as Map;
      final code = (body['code'] as num?)?.toInt() ?? resp.statusCode ?? -1;
      final msg = (body['msg'] as String?) ??
          e.message ??
          'HTTP ${resp.statusCode}';
      return ApiException(
        code: code,
        message: msg,
        statusCode: resp.statusCode,
      );
    }
    if (e.type == DioExceptionType.cancel) {
      return const NetworkException('请求已取消');
    }
    return NetworkException(e.message ?? '网络错误');
  }
}