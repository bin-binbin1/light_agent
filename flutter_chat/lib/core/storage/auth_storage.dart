import 'package:shared_preferences/shared_preferences.dart';

import '../constants.dart';

class AuthStorage {
  AuthStorage(this._prefs);

  final SharedPreferences _prefs;

  static Future<AuthStorage> load() async {
    final prefs = await SharedPreferences.getInstance();
    return AuthStorage(prefs);
  }

  String? get token => _prefs.getString(StorageKeys.token);
  String? get username => _prefs.getString(StorageKeys.username);
  String? get sessionId => _prefs.getString(StorageKeys.sessionId);

  bool get isLoggedIn =>
      (token?.isNotEmpty ?? false) && (username?.isNotEmpty ?? false);

  Future<void> saveCredentials({
    required String token,
    required String username,
    String? sessionId,
  }) async {
    await _prefs.setString(StorageKeys.token, token);
    await _prefs.setString(StorageKeys.username, username);
    if (sessionId != null && sessionId.isNotEmpty) {
      await _prefs.setString(StorageKeys.sessionId, sessionId);
    } else {
      await _prefs.remove(StorageKeys.sessionId);
    }
  }

  Future<void> setSessionId(String? sessionId) async {
    if (sessionId == null || sessionId.isEmpty) {
      await _prefs.remove(StorageKeys.sessionId);
    } else {
      await _prefs.setString(StorageKeys.sessionId, sessionId);
    }
  }

  Future<void> clear() async {
    await _prefs.remove(StorageKeys.token);
    await _prefs.remove(StorageKeys.username);
    await _prefs.remove(StorageKeys.sessionId);
  }
}