import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../auth/auth_provider.dart';
import '../sessions/session_drawer.dart';
import '../sessions/sessions_provider.dart';
import 'chat_provider.dart';
import 'chat_state.dart';
import 'widgets/info_bubble.dart';
import 'widgets/message_bubble.dart';
import 'widgets/tool_bubble.dart';

class ChatPage extends ConsumerStatefulWidget {
  const ChatPage({super.key});

  @override
  ConsumerState<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends ConsumerState<ChatPage> {
  final _textController = TextEditingController();
  final _focusNode = FocusNode();
  final _scrollController = ScrollController();

  /// 距底部多远算"接近底部"，小于此值时自动跟随
  static const double _followThreshold = 80;

  bool _follow = true;
  Timer? _scrollDebounce;
  int _lastItemCount = 0;

  @override
  void initState() {
    super.initState();
    _scrollController.addListener(_onUserScroll);
  }

  @override
  void dispose() {
    _scrollDebounce?.cancel();
    _textController.dispose();
    _focusNode.dispose();
    _scrollController.removeListener(_onUserScroll);
    _scrollController.dispose();
    super.dispose();
  }

  void _onUserScroll() {
    if (!_scrollController.hasClients) return;
    final pos = _scrollController.position;
    final near = pos.maxScrollExtent - pos.pixels <= _followThreshold;
    if (near != _follow) {
      setState(() => _follow = near);
    }
  }

  void _scheduleScrollToBottom() {
    if (!_follow) return;
    _scrollDebounce?.cancel();
    _scrollDebounce = Timer(const Duration(milliseconds: 16), () {
      if (!_scrollController.hasClients) return;
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 120),
        curve: Curves.easeOut,
      );
    });
  }

  void _jumpToBottom() {
    if (!_scrollController.hasClients) return;
    _scrollController.jumpTo(_scrollController.position.maxScrollExtent);
    setState(() => _follow = true);
  }

  Future<void> _send() async {
    final text = _textController.text;
    if (text.trim().isEmpty) return;
    final state = ref.read(chatProvider);
    if (state.isStreaming) return;
    _textController.clear();
    _follow = true; // 主动发消息时强制跟随
    await ref.read(chatProvider.notifier).sendMessage(text);
    if (mounted) _focusNode.requestFocus();
  }

  @override
  Widget build(BuildContext context) {
    final auth = ref.watch(authStateProvider);
    final state = ref.watch(chatProvider);

    // 监听 items 变化，列表增长时触发跟随
    if (state.items.length != _lastItemCount) {
      _lastItemCount = state.items.length;
      WidgetsBinding.instance.addPostFrameCallback((_) {
        _scheduleScrollToBottom();
      });
    }

    // 错误 → SnackBar（看过一次就清）
    ref.listen<String?>(chatProvider.select((s) => s.error), (_, next) {
      if (next != null && next.isNotEmpty) {
        ScaffoldMessenger.of(context)
          ..clearSnackBars()
          ..showSnackBar(SnackBar(content: Text(next)));
      }
    });
    ref.listen<String?>(sessionsProvider.select((s) => s.error), (_, next) {
      if (next != null && next.isNotEmpty) {
        ScaffoldMessenger.of(context)
          ..clearSnackBars()
          ..showSnackBar(SnackBar(content: Text(next)));
      }
    });

    return Scaffold(
      appBar: AppBar(
        title: Text(auth.username ?? ''),
        actions: [
          PopupMenuButton<_MenuAction>(
            tooltip: '更多',
            onSelected: (action) => _onMenuAction(action),
            itemBuilder: (context) => const [
              PopupMenuItem(
                value: _MenuAction.newSession,
                child: ListTile(
                  leading: Icon(Icons.add),
                  title: Text('新会话'),
                ),
              ),
              PopupMenuItem(
                value: _MenuAction.clearHistory,
                child: ListTile(
                  leading: Icon(Icons.cleaning_services_outlined),
                  title: Text('清空当前会话'),
                ),
              ),
              PopupMenuDivider(),
              PopupMenuItem(
                value: _MenuAction.logout,
                child: ListTile(
                  leading: Icon(Icons.logout),
                  title: Text('退出登录'),
                ),
              ),
            ],
          ),
        ],
      ),
      drawer: const SessionDrawer(),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(child: _buildBody(state)),
            _buildComposer(state),
          ],
        ),
      ),
      floatingActionButton: _follow
          ? null
          : FloatingActionButton.small(
              onPressed: _jumpToBottom,
              child: const Icon(Icons.arrow_downward),
            ),
    );
  }

  Future<void> _onMenuAction(_MenuAction action) async {
    switch (action) {
      case _MenuAction.newSession:
        await ref.read(sessionsProvider.notifier).createNew();
      case _MenuAction.clearHistory:
        final ok = await _confirm(
          title: '清空当前会话?',
          message: '历史记录会被全部清除，session_id 不变。',
          confirmText: '清空',
        );
        if (ok == true) {
          await ref.read(sessionsProvider.notifier).clearCurrent();
        }
      case _MenuAction.logout:
        final ok = await _confirm(
          title: '退出登录?',
          message: '退出后需重新输入用户名登录。',
          confirmText: '退出',
        );
        if (ok == true) {
          await ref.read(authStateProvider.notifier).logout();
        }
    }
  }

  Future<bool?> _confirm({
    required String title,
    required String message,
    required String confirmText,
  }) {
    return showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(title),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('取消'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: Text(confirmText),
          ),
        ],
      ),
    );
  }

  Widget _buildBody(ChatState state) {
    if (state.isLoadingHistory) {
      return const Center(child: CircularProgressIndicator());
    }
    if (state.items.isEmpty) {
      return Center(
        child: Text(
          '发消息开始和 Agent 聊聊吧～',
          style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
        ),
      );
    }
    return ListView.builder(
      controller: _scrollController,
      padding: const EdgeInsets.symmetric(vertical: 12),
      itemCount: state.items.length,
      itemBuilder: (context, index) {
        final item = state.items[index];
        return switch (item) {
          UserMsgItem() => UserBubble(item: item),
          AssistantMsgItem() => AssistantBubble(item: item),
          ToolItem() => ToolBubble(item: item),
          InfoItem() => InfoBubble(item: item),
        };
      },
    );
  }

  Widget _buildComposer(ChatState state) {
    final cs = Theme.of(context).colorScheme;
    final disabled = state.isStreaming;
    return Container(
      padding: const EdgeInsets.fromLTRB(12, 8, 12, 12),
      decoration: BoxDecoration(
        color: cs.surface,
        border: Border(top: BorderSide(color: cs.outlineVariant)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Expanded(
            child: Shortcuts(
              shortcuts: <ShortcutActivator, Intent>{
                LogicalKeySet(LogicalKeyboardKey.enter):
                    const _SendIntent(),
                LogicalKeySet(
                        LogicalKeyboardKey.shift, LogicalKeyboardKey.enter):
                    const _NewlineIntent(),
              },
              child: Actions(
                actions: <Type, Action<Intent>>{
                  _SendIntent:
                      CallbackAction<_SendIntent>(onInvoke: (_) {
                    _send();
                    return null;
                  }),
                  _NewlineIntent: CallbackAction<_NewlineIntent>(
                    onInvoke: (_) {
                      final old = _textController.value;
                      final sel = old.selection;
                      if (sel.isValid) {
                        final text = old.text.replaceRange(
                          sel.start,
                          sel.end,
                          '\n',
                        );
                        _textController.value = TextEditingValue(
                          text: text,
                          selection: TextSelection.collapsed(
                            offset: sel.start + 1,
                          ),
                        );
                      } else {
                        _textController.text = '${old.text}\n';
                      }
                      return null;
                    },
                  ),
                },
                child: TextField(
                  controller: _textController,
                  focusNode: _focusNode,
                  enabled: !disabled,
                  minLines: 1,
                  maxLines: 6,
                  textInputAction: TextInputAction.send,
                  decoration: InputDecoration(
                    hintText: disabled ? 'Agent 正在回复...' : '输入消息 (Enter 发送, Shift+Enter 换行)',
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    isDense: true,
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(width: 8),
          FilledButton(
            onPressed: disabled ? null : _send,
            style: FilledButton.styleFrom(
              padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
            ),
            child: disabled
                ? const SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Text('发送'),
          ),
        ],
      ),
    );
  }
}

class _SendIntent extends Intent {
  const _SendIntent();
}

class _NewlineIntent extends Intent {
  const _NewlineIntent();
}

enum _MenuAction { newSession, clearHistory, logout }