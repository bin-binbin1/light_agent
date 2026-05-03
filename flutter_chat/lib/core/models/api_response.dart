/// 服务端统一响应包装：{code, msg, data}
/// - code == 0 → 成功，data 为业务数据
/// - code != 0 → 失败，通常等于 HTTP 状态码，msg 为错误描述
///
/// 注意：SSE 端点和静态资源不走此包装
class ApiResponse<T> {
  const ApiResponse({
    required this.code,
    required this.msg,
    this.data,
  });

  final int code;
  final String msg;
  final T? data;

  bool get isOk => code == 0;

  static ApiResponse<T> fromJson<T>(
    Map<String, dynamic> json,
    T Function(Object? data)? mapper,
  ) {
    final raw = json['data'];
    return ApiResponse<T>(
      code: (json['code'] as num?)?.toInt() ?? -1,
      msg: (json['msg'] as String?) ?? '',
      data: mapper == null ? raw as T? : mapper(raw),
    );
  }
}