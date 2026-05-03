// 后端地址：根据运行环境改这里
// - Windows 桌面 / iOS 模拟器: http://127.0.0.1:8000
// - Android 模拟器:            http://10.0.2.2:8000
// - Android 真机:              http://<电脑局域网 IP>:8000
// - Web (CORS 已开):          http://localhost:8000
const String baseUrl = 'http://localhost:8000';

// SharedPreferences Keys
class StorageKeys {
  static const String token = 'la_token';
  static const String username = 'la_username';
  static const String sessionId = 'la_session_id';
}