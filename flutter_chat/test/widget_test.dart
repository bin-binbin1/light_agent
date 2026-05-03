import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:flutter_chat/app.dart';
import 'package:flutter_chat/core/storage/auth_storage.dart';
import 'package:flutter_chat/features/auth/auth_provider.dart';

void main() {
  testWidgets('未登录时展示登录页', (WidgetTester tester) async {
    SharedPreferences.setMockInitialValues({});
    final storage = await AuthStorage.load();

    await tester.pumpWidget(
      ProviderScope(
        overrides: [authStorageProvider.overrideWithValue(storage)],
        child: const LightAgentApp(),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('Light Agent'), findsOneWidget);
    expect(find.byType(TextField), findsOneWidget);
    expect(find.text('登录'), findsOneWidget);
  });
}