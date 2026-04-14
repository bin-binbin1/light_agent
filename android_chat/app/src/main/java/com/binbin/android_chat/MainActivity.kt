package com.binbin.android_chat

import android.content.Context
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.*
import androidx.datastore.preferences.preferencesDataStore
import androidx.lifecycle.viewmodel.compose.viewModel
import com.binbin.android_chat.data.Message
import com.binbin.android_chat.ui.*
import com.binbin.android_chat.ui.theme.Android_chatTheme
import com.binbin.android_chat.viewmodel.ChatViewModel

// DataStore 扩展
val Context.settingsDataStore by preferencesDataStore(name = "settings")

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            Android_chatTheme {
                LightChatApp()
            }
        }
    }
}

@Composable
fun LightChatApp(vm: ChatViewModel = viewModel()) {
    // 简单的页面导航状态
    var currentScreen by remember { mutableStateOf("list") } // "list", "chat", "settings"

    val conversations by vm.conversations.collectAsState()
    val messages by vm.currentMessages.collectAsState(initial = emptyList())
    val isLoading by vm.isLoading.collectAsState()
    val baseUrl by vm.baseUrl.collectAsState()
    val apiKey by vm.apiKey.collectAsState()
    val model by vm.model.collectAsState()
    val systemPrompt by vm.systemPrompt.collectAsState()

    when (currentScreen) {
        "list" -> ConversationListScreen(
            conversations = conversations,
            onSelect = { id ->
                vm.selectConversation(id)
                currentScreen = "chat"
            },
            onNew = {
                vm.createConversation()
                currentScreen = "chat"
            },
            onDelete = { vm.deleteConversation(it) },
            onSettings = { currentScreen = "settings" }
        )

        "chat" -> ChatScreen(
            messages = messages,
            isLoading = isLoading,
            onSend = { vm.sendMessage(it) },
            onBack = { currentScreen = "list" }
        )

        "settings" -> SettingsScreen(
            currentBaseUrl = baseUrl,
            currentApiKey = apiKey,
            currentModel = model,
            currentSystemPrompt = systemPrompt,
            onSave = { url, key, m, prompt ->
                vm.saveSettings(url, key, m, prompt)
            },
            onBack = { currentScreen = "list" }
        )
    }
}
