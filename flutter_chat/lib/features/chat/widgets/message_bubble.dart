import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';

import '../chat_state.dart';
import 'thinking_dots.dart';

/// 用户消息气泡：右侧，主色底。
class UserBubble extends StatelessWidget {
  const UserBubble({super.key, required this.item});

  final UserMsgItem item;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Align(
      alignment: Alignment.centerRight,
      child: ConstrainedBox(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.75,
        ),
        child: Container(
          margin: const EdgeInsets.symmetric(vertical: 4, horizontal: 12),
          padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 14),
          decoration: BoxDecoration(
            color: cs.primary,
            borderRadius: const BorderRadius.only(
              topLeft: Radius.circular(16),
              topRight: Radius.circular(16),
              bottomLeft: Radius.circular(16),
              bottomRight: Radius.circular(4),
            ),
          ),
          child: SelectableText(
            item.content,
            style: TextStyle(color: cs.onPrimary, height: 1.4),
          ),
        ),
      ),
    );
  }
}

/// Assistant 消息气泡：左侧，surface 底。
///
/// 流式阶段用纯文本避免半截 Markdown 被错误解析；完成后切普通 Text（后续阶段 4 末
/// 尾 / 阶段 6 可以替换为 MarkdownBody）。
class AssistantBubble extends StatelessWidget {
  const AssistantBubble({super.key, required this.item});

  final AssistantMsgItem item;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final isEmptyStreaming = item.streaming && item.content.isEmpty;

    return Align(
      alignment: Alignment.centerLeft,
      child: ConstrainedBox(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.85,
        ),
        child: Container(
          margin: const EdgeInsets.symmetric(vertical: 4, horizontal: 12),
          padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 14),
          decoration: BoxDecoration(
            color: cs.surfaceContainerHighest,
            borderRadius: const BorderRadius.only(
              topLeft: Radius.circular(16),
              topRight: Radius.circular(16),
              bottomLeft: Radius.circular(4),
              bottomRight: Radius.circular(16),
            ),
          ),
          child: isEmptyStreaming
              ? const ThinkingDots()
              : (item.streaming
                  // 流式阶段用纯文本，避免半截代码块 ``` 被错误解析
                  ? SelectableText(
                      item.content,
                      style: TextStyle(color: cs.onSurface, height: 1.5),
                    )
                  // 流结束后才渲染 Markdown
                  : MarkdownBody(
                      data: item.content,
                      selectable: true,
                      styleSheet: MarkdownStyleSheet.fromTheme(Theme.of(context))
                          .copyWith(
                        p: TextStyle(color: cs.onSurface, height: 1.5),
                        code: TextStyle(
                          color: cs.onSurfaceVariant,
                          backgroundColor: cs.surfaceContainer,
                          fontFamily: 'monospace',
                          fontSize: 13,
                        ),
                        codeblockDecoration: BoxDecoration(
                          color: cs.surfaceContainer,
                          borderRadius: BorderRadius.circular(6),
                        ),
                        codeblockPadding: const EdgeInsets.all(10),
                      ),
                    )),
        ),
      ),
    );
  }
}