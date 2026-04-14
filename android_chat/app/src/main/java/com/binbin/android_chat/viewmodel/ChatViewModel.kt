package com.binbin.android_chat.viewmodel

import android.app.Application
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.binbin.android_chat.data.*
import com.binbin.android_chat.settingsDataStore
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import java.util.UUID

class ChatViewModel(application: Application) : AndroidViewModel(application) {

    private val dao = ChatDatabase.getInstance(application).chatDao()
    private val apiClient = ApiClient()
    private val dataStore = application.settingsDataStore

    // 设置 keys
    companion object {
        val KEY_BASE_URL = stringPreferencesKey("base_url")
        val KEY_API_KEY = stringPreferencesKey("api_key")
        val KEY_MODEL = stringPreferencesKey("model")
        val KEY_SYSTEM_PROMPT = stringPreferencesKey("system_prompt")

        const val DEFAULT_BASE_URL = "https://api.openai.com/v1"
        const val DEFAULT_MODEL = "gpt-4o"
        const val DEFAULT_SYSTEM_PROMPT = "你是一个有用的 AI 助手。"
    }

    // UI 状态
    val conversations = dao.getAllConversations()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    private val _currentConversationId = MutableStateFlow<String?>(null)
    val currentConversationId: StateFlow<String?> = _currentConversationId

    val currentMessages: Flow<List<Message>> = _currentConversationId.flatMapLatest { id ->
        if (id != null) dao.getMessages(id) else flowOf(emptyList())
    }

    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading

    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error

    // 设置值
    val baseUrl = dataStore.data.map { it[KEY_BASE_URL] ?: DEFAULT_BASE_URL }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), DEFAULT_BASE_URL)
    val apiKey = dataStore.data.map { it[KEY_API_KEY] ?: "" }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), "")
    val model = dataStore.data.map { it[KEY_MODEL] ?: DEFAULT_MODEL }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), DEFAULT_MODEL)
    val systemPrompt = dataStore.data.map { it[KEY_SYSTEM_PROMPT] ?: DEFAULT_SYSTEM_PROMPT }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), DEFAULT_SYSTEM_PROMPT)

    init {
        // 监听设置变化，更新 API 客户端
        viewModelScope.launch {
            combine(baseUrl, apiKey, model) { url, key, m -> Triple(url, key, m) }
                .collect { (url, key, m) -> apiClient.updateConfig(url, key, m) }
        }
    }

    fun saveSettings(baseUrl: String, apiKey: String, model: String, systemPrompt: String) {
        viewModelScope.launch {
            dataStore.edit { prefs ->
                prefs[KEY_BASE_URL] = baseUrl
                prefs[KEY_API_KEY] = apiKey
                prefs[KEY_MODEL] = model
                prefs[KEY_SYSTEM_PROMPT] = systemPrompt
            }
        }
    }

    fun createConversation(): String {
        val id = UUID.randomUUID().toString().take(8)
        viewModelScope.launch {
            dao.insertConversation(Conversation(id = id, title = "新对话"))
        }
        _currentConversationId.value = id
        return id
    }

    fun selectConversation(id: String) {
        _currentConversationId.value = id
    }

    fun deleteConversation(id: String) {
        viewModelScope.launch {
            dao.deleteMessages(id)
            dao.deleteConversation(id)
            if (_currentConversationId.value == id) {
                _currentConversationId.value = null
            }
        }
    }

    fun sendMessage(content: String) {
        val convId = _currentConversationId.value ?: return
        if (_isLoading.value) return

        viewModelScope.launch {
            _isLoading.value = true
            _error.value = null

            // 存用户消息
            dao.insertMessage(Message(conversationId = convId, role = "user", content = content))

            // 创建占位助手消息
            val assistantMsgId = dao.insertMessage(
                Message(conversationId = convId, role = "assistant", content = "", isStreaming = true)
            )

            try {
                // 构建消息历史
                val history = buildMessageHistory(convId)
                var fullResponse = ""

                // 流式调用
                apiClient.chatStream(history).collect { token ->
                    fullResponse += token
                    dao.updateMessage(assistantMsgId, fullResponse, isStreaming = true)
                }

                // 标记完成
                dao.updateMessage(assistantMsgId, fullResponse, isStreaming = false)

                // 更新会话摘要
                val preview = if (fullResponse.length > 30) fullResponse.take(30) + "..." else fullResponse
                val title = if (content.length > 15) content.take(15) + "..." else content
                dao.updateConversation(convId, title, preview, System.currentTimeMillis())

            } catch (e: Exception) {
                val errorMsg = e.message ?: "未知错误"
                dao.updateMessage(assistantMsgId, "错误: $errorMsg", isStreaming = false)
                _error.value = errorMsg
            } finally {
                _isLoading.value = false
            }
        }
    }

    fun clearError() {
        _error.value = null
    }

    private suspend fun buildMessageHistory(convId: String): List<ApiClient.ChatMessage> {
        val messages = mutableListOf<ApiClient.ChatMessage>()

        // 系统提示词
        val prompt = systemPrompt.value
        if (prompt.isNotBlank()) {
            messages.add(ApiClient.ChatMessage("system", prompt))
        }

        // 最近消息（最多保留 20 轮）
        val recentMessages = dao.getMessages(convId).first().takeLast(40)
        for (msg in recentMessages) {
            if (msg.content.isNotBlank() && !msg.content.startsWith("错误:")) {
                messages.add(ApiClient.ChatMessage(msg.role, msg.content))
            }
        }

        return messages
    }
}
