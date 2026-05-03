import 'package:flutter/material.dart';

/// 三点跳动动画，用于 AssistantMsgItem.streaming && content 为空时展示。
class ThinkingDots extends StatefulWidget {
  const ThinkingDots({super.key, this.color});

  final Color? color;

  @override
  State<ThinkingDots> createState() => _ThinkingDotsState();
}

class _ThinkingDotsState extends State<ThinkingDots>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final color = widget.color ?? Theme.of(context).colorScheme.onSurfaceVariant;
    return SizedBox(
      height: 20,
      child: AnimatedBuilder(
        animation: _controller,
        builder: (context, _) {
          return Row(
            mainAxisSize: MainAxisSize.min,
            children: List.generate(3, (i) => _dot(i, color)),
          );
        },
      ),
    );
  }

  Widget _dot(int index, Color color) {
    final t = (_controller.value + index * 0.2) % 1.0;
    final wave = (t < 0.5) ? (t * 2) : (2 - t * 2); // 0..1..0
    final opacity = 0.3 + wave * 0.7;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 2),
      child: Container(
        width: 6,
        height: 6,
        decoration: BoxDecoration(
          color: color.withValues(alpha: opacity),
          shape: BoxShape.circle,
        ),
      ),
    );
  }
}