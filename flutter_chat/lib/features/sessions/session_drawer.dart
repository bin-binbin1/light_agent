import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/models/session_info.dart';
import '../auth/auth_provider.dart';
import 'sessions_provider.dart';

class SessionDrawer extends ConsumerWidget {
  const SessionDrawer({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final cs = Theme.of(context).colorScheme;
    final auth = ref.watch(authStateProvider);
    final state = ref.watch(sessionsProvider);
    final currentSid = auth.sessionId ?? '';

    return Drawer(
      child: Column(
        children: [
          // 顶部：用户 + 操作
          Container(
            padding: EdgeInsets.fromLTRB(
              16,
              MediaQuery.of(context).padding.top + 12,
              16,
              12,
            ),
            decoration: BoxDecoration(
              color: cs.surfaceContainerHighest,
            ),
            child: Row(
              children: [
                CircleAvatar(
                  backgroundColor: cs.primary,
                  child: Text(
                    (auth.username ?? '?').substring(0, 1).toUpperCase(),
                    style: TextStyle(color: cs.onPrimary),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        auth.username ?? '',
                        style: Theme.of(context).textTheme.titleMedium,
                        overflow: TextOverflow.ellipsis,
                      ),
                      Text(
                        '${state.sessions.length} 个会话',
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                              color: cs.onSurfaceVariant,
                            ),
                      ),
                    ],
                  ),
                ),
                IconButton(
                  icon: const Icon(Icons.refresh),
                  tooltip: '刷新',
                  onPressed: state.loading
                      ? null
                      : () => ref.read(sessionsProvider.notifier).refresh(),
                ),
              ],
            ),
          ),
          // 新会话按钮
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 12, 12, 8),
            child: SizedBox(
              width: double.infinity,
              child: FilledButton.icon(
                icon: const Icon(Icons.add),
                label: const Text('新会话'),
                onPressed: () async {
                  Navigator.of(context).maybePop();
                  await ref.read(sessionsProvider.notifier).createNew();
                },
              ),
            ),
          ),
          const Divider(height: 1),
          // 列表
          Expanded(
            child: _buildList(context, ref, state, currentSid),
          ),
        ],
      ),
    );
  }

  Widget _buildList(
    BuildContext context,
    WidgetRef ref,
    SessionsState state,
    String currentSid,
  ) {
    if (state.loading && state.sessions.isEmpty) {
      return const Center(child: CircularProgressIndicator());
    }
    if (state.sessions.isEmpty) {
      return Center(
        child: Text(
          '还没有会话~',
          style: TextStyle(
            color: Theme.of(context).colorScheme.onSurfaceVariant,
          ),
        ),
      );
    }
    return ListView.separated(
      itemCount: state.sessions.length,
      separatorBuilder: (_, _) => const Divider(height: 1),
      itemBuilder: (context, index) {
        final s = state.sessions[index];
        return _SessionTile(
          info: s,
          active: s.sessionId == currentSid,
          onTap: () async {
            Navigator.of(context).maybePop();
            await ref.read(sessionsProvider.notifier).switchTo(s.sessionId);
          },
          onDelete: () async {
            final confirmed = await _confirmDelete(context, s);
            if (confirmed != true) return;
            await ref.read(sessionsProvider.notifier).delete(s.sessionId);
          },
        );
      },
    );
  }

  Future<bool?> _confirmDelete(BuildContext context, SessionInfo s) {
    return showDialog<bool>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('删除会话?'),
          content: Text(
            '会话 ${_shortSid(s.sessionId)} 的所有历史会被清除，不可恢复。',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(false),
              child: const Text('取消'),
            ),
            FilledButton.tonal(
              onPressed: () => Navigator.of(context).pop(true),
              style: FilledButton.styleFrom(
                foregroundColor: Theme.of(context).colorScheme.error,
              ),
              child: const Text('删除'),
            ),
          ],
        );
      },
    );
  }
}

String _shortSid(String sid) {
  if (sid.length <= 18) return sid;
  return '${sid.substring(0, 10)}…${sid.substring(sid.length - 4)}';
}

String _formatRelativeTime(DateTime t) {
  final diff = DateTime.now().difference(t);
  if (diff.inSeconds < 60) return '刚刚';
  if (diff.inMinutes < 60) return '${diff.inMinutes} 分钟前';
  if (diff.inHours < 24) return '${diff.inHours} 小时前';
  if (diff.inDays < 7) return '${diff.inDays} 天前';
  return '${t.year}-${t.month.toString().padLeft(2, '0')}-${t.day.toString().padLeft(2, '0')}';
}

class _SessionTile extends ConsumerStatefulWidget {
  const _SessionTile({
    required this.info,
    required this.active,
    required this.onTap,
    required this.onDelete,
  });

  final SessionInfo info;
  final bool active;
  final VoidCallback onTap;
  final VoidCallback onDelete;

  @override
  ConsumerState<_SessionTile> createState() => _SessionTileState();
}

class _SessionTileState extends ConsumerState<_SessionTile> {
  @override
  void initState() {
    super.initState();
    // 首次渲染后补标题（不阻塞 UI）
    if (widget.info.title == null) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (!mounted) return;
        ref
            .read(sessionsProvider.notifier)
            .ensureTitle(widget.info.sessionId);
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final title = widget.info.title ?? _shortSid(widget.info.sessionId);
    return Material(
      color: widget.active ? cs.primaryContainer : Colors.transparent,
      child: InkWell(
        onTap: widget.onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 16),
          child: Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        fontWeight: widget.active
                            ? FontWeight.w600
                            : FontWeight.w400,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      _formatRelativeTime(widget.info.updatedAt),
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: cs.onSurfaceVariant,
                          ),
                    ),
                  ],
                ),
              ),
              IconButton(
                icon: const Icon(Icons.delete_outline, size: 20),
                tooltip: '删除',
                onPressed: widget.onDelete,
              ),
            ],
          ),
        ),
      ),
    );
  }
}