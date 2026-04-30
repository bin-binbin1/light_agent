// ─── Light Agent Web App ───

const API = {
    server: 'http://localhost:8000',
    userId: 'web_user',

    async request(method, path, body = null) {
        const opts = {
            method,
            headers: { 'Content-Type': 'application/json' }
        };
        if (body) opts.body = JSON.stringify(body);
        const res = await fetch(`${this.server}${path}`, opts);
        return res.json();
    },

    get(path) { return this.request('GET', path); },
    post(path, body) { return this.request('POST', path, body); },
    del(path) { return this.request('DELETE', path); },
};

// ─── 状态 ───
let currentSessionId = null;
let sessions = [];

// ─── 初始化 ───
window.onload = () => {
    loadSettings();
    loadSessions();
};

// ─── 设置 ───
function openSettings() {
    document.getElementById('settingsModal').classList.add('show');
    document.getElementById('settingServer').value = API.server;
    document.getElementById('settingUserId').value = API.userId;
}

function closeSettings(e) {
    if (e && e.target !== e.currentTarget) return;
    document.getElementById('settingsModal').classList.remove('show');
}

function saveSettings() {
    API.server = document.getElementById('settingServer').value || 'http://localhost:8000';
    API.userId = document.getElementById('settingUserId').value || 'web_user';

    const settings = {
        server: API.server,
        userId: API.userId,
        provider: document.getElementById('settingProvider').value,
        model: document.getElementById('settingModel').value,
        apiKey: document.getElementById('settingApiKey').value,
        systemPrompt: document.getElementById('settingPrompt').value,
        contextWindow: document.getElementById('settingContext').value,
        temperature: document.getElementById('settingTemp').value,
    };

    localStorage.setItem('light_agent_settings', JSON.stringify(settings));
    closeSettings();
    loadSessions();
}

function loadSettings() {
    const saved = localStorage.getItem('light_agent_settings');
    if (!saved) return;

    const s = JSON.parse(saved);
    API.server = s.server || 'http://localhost:8000';
    API.userId = s.userId || 'web_user';

    document.getElementById('settingServer').value = API.server;
    document.getElementById('settingUserId').value = API.userId;
    if (s.provider) document.getElementById('settingProvider').value = s.provider;
    if (s.model) document.getElementById('settingModel').value = s.model;
    if (s.apiKey) document.getElementById('settingApiKey').value = s.apiKey;
    if (s.systemPrompt) document.getElementById('settingPrompt').value = s.systemPrompt;
    if (s.contextWindow) document.getElementById('settingContext').value = s.contextWindow;
    if (s.temperature) document.getElementById('settingTemp').value = s.temperature;
}

// ─── 会话管理 ───
async function loadSessions() {
    const res = await API.get(`/api/users/${API.userId}/sessions`);
    sessions = res.sessions || [];
    renderSessionList();
}

async function createSession() {
    const res = await API.post('/api/sessions', {
        user_id: API.userId,
        title: '新对话'
    });

    if (res.session_id) {
        await loadSessions();
        selectSession(res.session_id);
    }
}

function selectSession(sessionId) {
    currentSessionId = sessionId;
    renderSessionList();
    loadMessages();

    const s = sessions.find(s => s.session_id === sessionId);
    document.getElementById('sessionTitle').textContent = s ? s.title || '对话' : '对话';
}

async function deleteSession(sessionId, e) {
    e.stopPropagation();
    if (!confirm('确定删除此对话？')) return;

    await API.del(`/api/sessions/${sessionId}?user_id=${API.userId}`);

    if (currentSessionId === sessionId) {
        currentSessionId = null;
        document.getElementById('chatContainer').innerHTML = `
            <div class="welcome">
                <h1>🤖 Light Agent</h1>
                <p>轻量化 AI 对话助手</p>
                <p class="hint">点击「新建对话」开始</p>
            </div>`;
        document.getElementById('sessionTitle').textContent = '选择或新建对话';
    }

    await loadSessions();
}

function renderSessionList() {
    const list = document.getElementById('sessionList');
    list.innerHTML = sessions.map(s => `
        <div class="session-item ${s.session_id === currentSessionId ? 'active' : ''}"
             onclick="selectSession('${s.session_id}')">
            <span class="title">${s.title || s.session_id}</span>
            <button class="delete-btn" onclick="deleteSession('${s.session_id}', event)">×</button>
        </div>
    `).join('');
}

// ─── 消息 ───
async function loadMessages() {
    if (!currentSessionId) return;

    const res = await API.get(`/api/sessions/${currentSessionId}/messages?user_id=${API.userId}`);
    const messages = res.messages || [];

    const container = document.getElementById('chatContainer');
    container.innerHTML = messages.map(m => renderMessage(m.role, m.content)).join('');
    scrollToBottom();
}

function renderMessage(role, content) {
    // 简单 markdown 处理
    let html = escapeHtml(content);
    // 代码块
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    // 行内代码
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    return `
        <div class="message ${role}">
            <div class="role">${role === 'user' ? '👤 你' : '🤖 Agent'}</div>
            <div class="content">${html}</div>
        </div>`;
}

async function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    if (!message || !currentSessionId) return;

    input.value = '';
    const btn = document.getElementById('sendBtn');
    btn.disabled = true;

    // 显示用户消息
    appendMessage('user', message);
    scrollToBottom();

    // 创建 assistant 气泡（内容稍后流式追加；先挂一个 loading 指示）
    let asstBubble, asstContent;
    ({ bubble: asstBubble, contentEl: asstContent } = appendAssistantBubble());
    showThinking(asstBubble);
    scrollToBottom();

    const toolMap = new Map();       // name -> {bubble, contentEl} 工具气泡追踪
    let asstText = '';               // 当前 assistant 气泡累积的文本（工具调用后会重置）
    let hasTextStarted = false;      // 当前 assistant 气泡是否已收到第一帧文本

    try {
        const resp = await fetch(`${API.server}/api/chat/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: API.userId,
                session_id: currentSessionId,
                message: message,
            }),
        });

        if (!resp.ok || !resp.body) {
            hideThinking(asstBubble);
            asstContent.textContent = `❌ 连接失败: HTTP ${resp.status}`;
            throw new Error(`HTTP ${resp.status}`);
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';

        outer: while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });
            const parts = buf.split('\n\n');
            buf = parts.pop(); // 最后一段可能不完整

            for (const part of parts) {
                if (!part.startsWith('data:')) continue;
                const raw = part.slice(5).trim();
                if (!raw) continue;

                let evt;
                try { evt = JSON.parse(raw); }
                catch (_) { continue; }

                switch (evt.type) {
                    case 'thinking':
                        if (!hasTextStarted) showThinking(asstBubble);
                        break;

                    case 'text':
                        if (!hasTextStarted) {
                            hideThinking(asstBubble);
                            hasTextStarted = true;
                        }
                        asstText += (evt.content || '');
                        // 流式期间先纯文本追加，避免 markdown 半截解析
                        asstContent.textContent = asstText;
                        scrollToBottom();
                        break;

                    case 'tool_call': {
                        // 当前 assistant 气泡若是空（仅有 thinking 或没内容），就先移除，避免一个空气泡卡在工具之前
                        if (!asstText) {
                            asstBubble.remove();
                        } else {
                            // 已经有文本，先把它固化（markdown 渲染）
                            asstContent.innerHTML = renderMarkdown(asstText);
                        }
                        // 工具气泡插入到消息流
                        const tb = appendToolBubble(evt);
                        toolMap.set(evt.name, tb);
                        // 为工具之后可能的 assistant 输出新建一个空气泡
                        ({ bubble: asstBubble, contentEl: asstContent } = appendAssistantBubble());
                        asstText = '';
                        hasTextStarted = false;
                        scrollToBottom();
                        break;
                    }

                    case 'tool_result': {
                        const tb = toolMap.get(evt.name);
                        if (tb) updateToolBubble(tb, evt);
                        scrollToBottom();
                        break;
                    }

                    case 'retry':
                        appendInfoBubble(`⏳ 限流重试 ${evt.attempt}/${evt.max_attempts}（等待 ${evt.wait_seconds?.toFixed?.(1) || evt.wait_seconds}s）`);
                        scrollToBottom();
                        break;

                    case 'error':
                        hideThinking(asstBubble);
                        asstContent.textContent = `❌ ${evt.message || '未知错误'}`;
                        break outer;

                    case 'done':
                        break outer;
                }
            }
        }

        // 流结束，对最终 assistant 文本做一次 markdown 重渲染
        if (asstText) {
            asstContent.innerHTML = renderMarkdown(asstText);
        } else if (!hasTextStarted) {
            // 没有任何文本产出（比如全是工具调用最后没接话），把空气泡移除
            asstBubble.remove();
        }

        // 更新会话标题（如果是第一条消息）
        const msgs = document.querySelectorAll('.message');
        if (msgs.length <= 3) {
            const title = message.slice(0, 20) + (message.length > 20 ? '...' : '');
            sessions = sessions.map(s =>
                s.session_id === currentSessionId ? { ...s, title } : s
            );
            renderSessionList();
            document.getElementById('sessionTitle').textContent = title;
        }
    } catch (err) {
        if (asstContent && !asstContent.textContent) {
            asstContent.textContent = `❌ 连接失败: ${err.message}`;
        }
    }

    btn.disabled = false;
    scrollToBottom();
}

// ─── 气泡构造/操作 ───

function appendAssistantBubble() {
    const container = document.getElementById('chatContainer');
    const wrap = document.createElement('div');
    wrap.className = 'message assistant';
    wrap.innerHTML = '<div class="role">🤖 Agent</div><div class="content"></div>';
    container.appendChild(wrap);
    const contentEl = wrap.querySelector('.content');
    return { bubble: wrap, contentEl };
}

function showThinking(bubble) {
    const content = bubble.querySelector('.content');
    if (!content) return;
    if (content.querySelector('.thinking-dots')) return;
    content.innerHTML = '<span class="thinking-dots"><span class="dot">.</span><span class="dot">.</span><span class="dot">.</span></span>';
}

function hideThinking(bubble) {
    const content = bubble.querySelector('.content');
    if (!content) return;
    const dots = content.querySelector('.thinking-dots');
    if (dots) {
        content.innerHTML = '';
    }
}

function appendToolBubble(evt) {
    const container = document.getElementById('chatContainer');
    const wrap = document.createElement('div');
    wrap.className = 'tool-bubble running';
    wrap.innerHTML = `
        <span class="tool-spinner"></span>
        <span class="tool-display">${escapeHtml(evt.display || evt.name || '调用工具中...')}</span>
        <span class="tool-duration"></span>
    `;
    container.appendChild(wrap);
    return { bubble: wrap, contentEl: wrap.querySelector('.tool-display'), durEl: wrap.querySelector('.tool-duration') };
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

function renderMarkdown(text) {
    let html = escapeHtml(text);
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    return html;
}

function appendMessage(role, content) {
    const container = document.getElementById('chatContainer');
    container.insertAdjacentHTML('beforeend', renderMessage(role, content));
}

function showLoading() {
    const id = 'loading_' + Date.now();
    const container = document.getElementById('chatContainer');
    container.insertAdjacentHTML('beforeend', `
        <div class="message assistant" id="${id}">
            <div class="role">🤖 Agent</div>
            <div class="content loading"><span class="dot">.</span><span class="dot">.</span><span class="dot">.</span></div>
        </div>`);
    scrollToBottom();
    return id;
}

function removeLoading(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function scrollToBottom() {
    const container = document.getElementById('chatContainer');
    container.scrollTop = container.scrollHeight;
}

// ─── 工具函数 ───
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function switchModel() {
    // TODO: 切换模型
}
