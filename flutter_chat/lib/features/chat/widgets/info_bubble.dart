import 'package:flutter/material.dart';

import '../chat_state.dart';

/// 系统/retry/error 类通知：居中弱色条。
class InfoBubble extends StatelessWidget {
  const InfoBubble({super.key, required this.item});

  final InfoItem item;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Center(
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 6, horizontal: 12),
        padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 12),
        decoration: BoxDecoration(
          color: cs.surfaceContainerLow,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Text(
          item.text,
          style: TextStyle(color: cs.onSurfaceVariant, fontSize: 12),
          textAlign: TextAlign.center,
        ),
      ),
    );
  }
}