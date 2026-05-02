// ─── Light Agent Web App ───

const API = {
    base: location.origin,
    token: localStorage.getItem('la_token') || '',
    username: localStorage.getItem('la_username') || '',
};

// ─── 视图切换 ───
function showLogin() {
    document.getElementById('loginView').classList.remove('hidden');
    document.getElementById('chatView').classList.add('hidden');
    setTimeout(() => document.getElementById('usernameInput')?.focus(), 50);
}

function showChat() {
    document.getElementById('loginView').classList.add('hidden');
    document.getElementById('chatView').classList.remove('hidden');
    document.getElementById('currentUser').textContent = API.username;
    setTimeout(() => document.getElementById('messageInput')?.focus(), 50);
}

// ─── 启动 ───
window.onload = async () => {
    if (API.token && API.username) {
        try {
            const r = await fetch(`${API.base}/agent/session?token=${encodeURIComponent(API.token)}`);
            if (r.ok) {
                showChat();
                await loadHistory();
                return;
            }
        } catch (_) { /* fall through to login */ }
        localStorage.removeItem('la_token');
        localStorage.removeItem('la_username');
        API.token = '';
        API.username = '';
    }
    showLogin();
};

// ─── 登录 / 登出 ───
async function login() {
    const input = document.getElementById('usernameInput');
    const errEl = document.getElementById('loginError');
    const username = input.value.trim();
    errEl.textContent = '';

    if (!username) {
        errEl.textContent = '用户名不能为空';
        return;
    }

    try {
        const r = await fetch(`${API.base}/agent/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username }),
        });
        const json = await r.json().catch(() => ({}));
        if (!r.ok || json.code !== 0) {
            errEl.textContent = json.msg || `登录失败 (HTTP ${r.status})`;
            return;
        }
        const payload = json.data || {};
        API.token = payload.token;
        API.username = payload.username;
        localStorage.setItem('la_token', API.token);
        localStorage.setItem('la_username', API.username);
        showChat();
        clearChatContainer();
        await loadHistory();
    } catch (err) {
        errEl.textContent = `网络错误: ${err.message}`;
    }
}

async function logout() {
    if (!confirm('确定要退出登录？')) return;
    try {
        await fetch(`${API.base}/agent/logout?token=${encodeURIComponent(API.token)}`, {
            method: 'POST',
        });
    } catch (_) { /* ignore */ }
    localStorage.removeItem('la_token');
    localStorage.removeItem('la_username');
    API.token = '';
    API.username = '';
    clearChatContainer();
    showLogin();
}

async function resetSession() {
    if (!confirm('重置后当前对话上下文会清空（历史消息仍在数据库中），确定？')) return;
    try {
        const r = await fetch(`${API.base}/agent/reset?token=${encodeURIComponent(API.token)}`, {
            method: 'POST',
        });
        const json = await r.json().catch(() => ({}));
        if (!r.ok || json.code !== 0) {
            alert(`重置失败: ${json.msg || 'HTTP ' + r.status}`);
            return;
        }
        clearChatContainer();
        appendInfoBubble(json.msg || '对话已重置');
    } catch (err) {
        alert(`网络错误: ${err.message}`);
    }
}

function clearChatContainer() {
    document.getElementById('chatContainer').innerHTML = '';
}

// ─── 历史加载 ───
async function loadHistory() {
    try {
        const r = await fetch(`${API.base}/agent/history?token=${encodeURIComponent(API.token)}`);
        if (!r.ok) return;
        const json = await r.json().catch(() => ({}));
        if (json.code !== 0) return;
        const messages = (json.data && json.data.messages) || [];
        const container = document.getElementById('chatContainer');
        if (!messages || messages.length === 0) {
            container.innerHTML = `
                <div class="welcome">
                    <h1>🤖 Light Agent</h1>
                    <p class="hint">开始新的对话吧</p>
                </div>`;
            return;
        }
        container.innerHTML = '';
        for (const m of messages) {
            if (m.role === 'user') {
                appendMessage('user', m.content || '');
            } else if (m.role === 'assistant') {
                if (m.content) {
                    appendMessage('assistant', m.content);
                }
                if (m.tool_calls && m.tool_calls.length) {
                    for (const tc of m.tool_calls) {
                        const name = (tc.function && tc.function.name) || tc.name || 'tool';
                        const tb = appendToolBubble({ name, display: name });
                        updateToolBubble(tb, { display: name, success: true });
                    }
                }
            }
            // role === 'tool'（工具返回结果）不单独展示，让 LLM 的下一条回复承载
        }
        scrollToBottom();
    } catch (err) {
        console.error('loadHistory failed:', err);
    }
}

// ─── 发送消息（SSE 流式） ───
async function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    if (!message) return;

    input.value = '';
    autoResize(input);
    const btn = document.getElementById('sendBtn');
    btn.disabled = true;

    // 隐藏欢迎提示
    const welcome = document.getElementById('welcomeHint');
    if (welcome) welcome.remove();

    appendMessage('user', message);
    scrollToBottom();

    // 创建 assistant 气泡（先挂思考点点点）
    let { bubble: asstBubble, contentEl: asstContent } = appendAssistantBubble();
    showThinking(asstBubble);
    scrollToBottom();

    const toolMap = new Map();
    let asstText = '';
    let hasTextStarted = false;

    try {
        const resp = await fetch(`${API.base}/agent/chat?token=${encodeURIComponent(API.token)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message }),
        });

        if (resp.status === 401) {
            hideThinking(asstBubble);
            asstContent.textContent = '❌ 登录已失效，请重新登录';
            setTimeout(() => { localStorage.clear(); location.reload(); }, 1500);
            return;
        }

        if (!resp.ok || !resp.body) {
            hideThinking(asstBubble);
            asstContent.textContent = `❌ 连接失败: HTTP ${resp.status}`;
            return;
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';

        outer: while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });

            const frames = buf.split('\n\n');
            buf = frames.pop(); // 最后一段可能不完整

            for (const frame of frames) {
                const parsed = parseSSEFrame(frame);
                if (!parsed) continue;
                const { event, data } = parsed;

                switch (event) {
                    case 'thinking':
                        if (!hasTextStarted) showThinking(asstBubble);
                        break;

                    case 'chunk': {
                        if (!hasTextStarted) {
                            hideThinking(asstBubble);
                            hasTextStarted = true;
                        }
                        asstText += (data.text || '');
                        asstContent.textContent = asstText;
                        scrollToBottom();
                        break;
                    }

                    case 'tool_call': {
                        // 如当前气泡已有文本则固化 markdown；否则移除空气泡
                        if (asstText) {
                            asstContent.innerHTML = renderMarkdown(asstText);
                        } else {
                            asstBubble.remove();
                        }
                        const tb = appendToolBubble(data);
                        toolMap.set(data.name, tb);
                        // 为工具后续可能的 assistant 输出新建空气泡
                        ({ bubble: asstBubble, contentEl: asstContent } = appendAssistantBubble());
                        asstText = '';
                        hasTextStarted = false;
                        scrollToBottom();
                        break;
                    }

                    case 'tool_result': {
                        const tb = toolMap.get(data.name);
                        if (tb) updateToolBubble(tb, data);
                        scrollToBottom();
                        break;
                    }

                    case 'retry': {
                        const secs = (data.wait_seconds || 0).toFixed ? data.wait_seconds.toFixed(1) : data.wait_seconds;
                        appendInfoBubble(`⏳ 限流重试 ${data.attempt}/${data.max_attempts}（等待 ${secs}s）`);
                        scrollToBottom();
                        break;
                    }

                    case 'error':
                        hideThinking(asstBubble);
                        asstContent.textContent = `❌ ${data.message || '未知错误'}`;
                        break outer;

                    case 'done':
                        break outer;
                }
            }
        }

        // 流结束，最终文本走一次 markdown 渲染
        if (asstText) {
            asstContent.innerHTML = renderMarkdown(asstText);
        } else if (!hasTextStarted) {
            asstBubble.remove();
        }
    } catch (err) {
        if (asstContent && !asstContent.textContent) {
            hideThinking(asstBubble);
            asstContent.textContent = `❌ 连接异常: ${err.message}`;
        }
    } finally {
        btn.disabled = false;
        scrollToBottom();
    }
}

// ─── SSE 帧解析 ───
function parseSSEFrame(frame) {
    let event = 'message';
    let data = '';
    for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) {
            event = line.slice(6).trim();
        } else if (line.startsWith('data:')) {
            data += line.slice(5).trim();
        }
    }
    if (!data && event === 'message') return null;
    let parsed = {};
    try { parsed = data ? JSON.parse(data) : {}; } catch (_) { parsed = { raw: data }; }
    return { event, data: parsed };
}

// ─── 气泡构造 ───
function appendAssistantBubble() {
    const container = document.getElementById('chatContainer');
    const wrap = document.createElement('div');
    wrap.className = 'message assistant';
    wrap.innerHTML = '<div class="role">🤖 Agent</div><div class="content"></div>';
    container.appendChild(wrap);
    return { bubble: wrap, contentEl: wrap.querySelector('.content') };
}

function appendMessage(role, content) {
    const container = document.getElementById('chatContainer');
    const wrap = document.createElement('div');
    wrap.className = `message ${role}`;
    wrap.innerHTML = `
        <div class="role">${role === 'user' ? '👤 你' : '🤖 Agent'}</div>
        <div class="content"></div>`;
    wrap.querySelector('.content').innerHTML = renderMarkdown(content);
    container.appendChild(wrap);
}

function showThinking(bubble) {
    const content = bubble.querySelector('.content');
    if (!content || content.querySelector('.thinking-dots')) return;
    content.innerHTML = '<span class="thinking-dots"><span class="dot">.</span><span class="dot">.</span><span class="dot">.</span></span>';
}

function hideThinking(bubble) {
    const content = bubble.querySelector('.content');
    if (!content) return;
    if (content.querySelector('.thinking-dots')) content.innerHTML = '';
}

function appendToolBubble(evt) {
    const container = document.getElementById('chatContainer');
    const wrap = document.createElement('div');
    wrap.className = 'tool-bubble running';
    wrap.innerHTML = `
        <span class="tool-spinner"></span>
        <span class="tool-display">${escapeHtml(evt.display || evt.name || '调用工具中...')}</span>
        <span class="tool-duration"></span>`;
    container.appendChild(wrap);
    return {
        bubble: wrap,
        contentEl: wrap.querySelector('.tool-display'),
        durEl: wrap.querySelector('.tool-duration'),
    };
}

function updateToolBubble(tb, evt) {
    if (!tb) return;
    tb.bubble.classList.remove('running');
    tb.bubble.classList.add(evt.success ? 'success' : 'fail');
    if (evt.display) tb.contentEl.textContent = evt.display;
    if (evt.duration_ms != null) tb.durEl.textContent = ` (${evt.duration_ms}ms)`;
}

function appendInfoBubble(text) {
    const container = document.getElementById('chatContainer');
    const wrap = document.createElement('div');
    wrap.className = 'info-bubble';
    wrap.textContent = text;
    container.appendChild(wrap);
}

// ─── 渲染辅助 ───
function renderMarkdown(text) {
    let html = escapeHtml(text);
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    return html;
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str || '';
    return div.innerHTML;
}

function scrollToBottom() {
    const container = document.getElementById('chatContainer');
    container.scrollTop = container.scrollHeight;
}

// ─── 输入区交互 ───
function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    } else {
        setTimeout(() => autoResize(e.target), 0);
    }
}

function autoResize(el) {
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}
