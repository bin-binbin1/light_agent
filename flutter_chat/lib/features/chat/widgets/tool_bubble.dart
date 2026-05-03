import 'package:flutter/material.dart';

import '../chat_state.dart';

/// 工具调用状态条：居中小气泡，根据 status 切图标。
class ToolBubble extends StatelessWidget {
  const ToolBubble({super.key, required this.item});

  final ToolItem item;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final (icon, iconColor, label) = _visualize(context);

    final durationText = item.durationMs == null
        ? ''
        : ' · ${(item.durationMs! / 1000).toStringAsFixed(1)}s';

    return Center(
      child: ConstrainedBox(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.85,
        ),
        child: Container(
          margin: const EdgeInsets.symmetric(vertical: 4, horizontal: 12),
          padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 12),
          decoration: BoxDecoration(
            color: cs.surfaceContainer,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: cs.outlineVariant),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (item.status == ToolStatus.running)
                const SizedBox(
                  width: 14,
                  height: 14,
                  child: CircularProgressIndicator(strokeWidth: 2),
                )
              else
                Icon(icon, size: 16, color: iconColor),
              const SizedBox(width: 8),
              Flexible(
                child: Text(
                  '$label$durationText',
                  style: TextStyle(
                    color: cs.onSurfaceVariant,
                    fontSize: 13,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  (IconData, Color, String) _visualize(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    switch (item.status) {
      case ToolStatus.running:
        return (Icons.hourglass_top, cs.primary, item.display);
      case ToolStatus.success:
        return (Icons.check_circle_outline, Colors.green, item.display);
      case ToolStatus.fail:
        return (Icons.error_outline, cs.error, item.display);
    }
  }
}