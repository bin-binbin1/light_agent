import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'app.dart';
import 'core/constants.dart';
import 'core/storage/auth_storage.dart';
import 'features/auth/auth_provider.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  debugPrint('[BOOT] baseUrl = $baseUrl');
  final storage = await AuthStorage.load();

  runApp(
    ProviderScope(
      overrides: [
        authStorageProvider.overrideWithValue(storage),
      ],
      child: const LightAgentApp(),
    ),
  );
}