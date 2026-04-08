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

    // 显示 loading
    const loadingId = showLoading();

    try {
        const res = await API.post('/api/chat', {
            user_id: API.userId,
            session_id: currentSessionId,
            message: message
        });

        removeLoading(loadingId);

        if (res.response) {
            appendMessage('assistant', res.response);

            // 更新会话标题（如果是第一条消息）
            const msgs = document.querySelectorAll('.message');
            if (msgs.length === 2) {
                const title = message.slice(0, 20) + (message.length > 20 ? '...' : '');
                sessions = sessions.map(s =>
                    s.session_id === currentSessionId ? {...s, title} : s
                );
                renderSessionList();
                document.getElementById('sessionTitle').textContent = title;
            }
        } else if (res.error) {
            appendMessage('assistant', `❌ 错误: ${res.error}`);
        }
    } catch (err) {
        removeLoading(loadingId);
        appendMessage('assistant', `❌ 连接失败: ${err.message}。请检查 Server URL 设置。`);
    }

    btn.disabled = false;
    scrollToBottom();
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
