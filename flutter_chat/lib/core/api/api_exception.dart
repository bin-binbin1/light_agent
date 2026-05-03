/// API 调用异常。
///
/// 服务端所有 JSON 端点（含异常路径）都返回 `{code, msg, data}`，
/// 所以 Dio 拿到 4xx/5xx 时也能读出业务 code/msg。
class ApiException implements Exception {
  const ApiException({
    required this.code,
    required this.message,
    this.statusCode,
  });

  /// 业务 code（0=成功，非 0 通常等于 HTTP 状态码）。
  final int code;

  /// 业务描述（来自服务端 `msg` 字段）。
  final String message;

  /// HTTP 状态码（可能与 code 不同，例如网络层错误）。
  final int? statusCode;

  /// token 失效 / 未登录 → 客户端应清本地凭据并回登录页。
  bool get isUnauthorized => statusCode == 401 || code == 401;

  /// 会话无效（空 / 不存在 / 无权访问）→ 客户端应清 sessionId 后重拉。
  bool get isSessionInvalid {
    final sc = statusCode ?? code;
    return sc == 400 || sc == 403 || sc == 404;
  }

  @override
  String toString() => 'ApiException(code=$code, message=$message)';
}

/// 网络层错误（连不上 / 超时 / 取消 等）。
class NetworkException extends ApiException {
  const NetworkException(String message)
      : super(code: -1, message: message, statusCode: null);
}