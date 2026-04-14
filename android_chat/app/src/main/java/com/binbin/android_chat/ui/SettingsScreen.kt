package com.binbin.android_chat.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    currentBaseUrl: String,
    currentApiKey: String,
    currentModel: String,
    currentSystemPrompt: String,
    onSave: (baseUrl: String, apiKey: String, model: String, systemPrompt: String) -> Unit,
    onBack: () -> Unit
) {
    var baseUrl by remember { mutableStateOf(currentBaseUrl) }
    var apiKey by remember { mutableStateOf(currentApiKey) }
    var model by remember { mutableStateOf(currentModel) }
    var systemPrompt by remember { mutableStateOf(currentSystemPrompt) }
    var showApiKey by remember { mutableStateOf(false) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("设置", fontWeight = FontWeight.Bold) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, "返回")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // 连接模式说明
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f)
                )
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("连接模式", fontWeight = FontWeight.Medium, fontSize = 14.sp)
                    Spacer(Modifier.height(4.dp))
                    Text(
                        "有 API Key: 填写 Key 直连 LLM 服务商\n没有 Key: 填写中转服务器地址和 Token",
                        fontSize = 13.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        lineHeight = 20.sp
                    )
                }
            }

            // API 地址
            OutlinedTextField(
                value = baseUrl,
                onValueChange = { baseUrl = it },
                label = { Text("API 地址") },
                placeholder = { Text("https://api.openai.com/v1") },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true
            )

            // API Key
            OutlinedTextField(
                value = apiKey,
                onValueChange = { apiKey = it },
                label = { Text("API Key / Token") },
                placeholder = { Text("sk-... 或中转 Token") },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                visualTransformation = if (showApiKey) VisualTransformation.None
                else PasswordVisualTransformation(),
                trailingIcon = {
                    TextButton(onClick = { showApiKey = !showApiKey }) {
                        Text(if (showApiKey) "隐藏" else "显示", fontSize = 12.sp)
                    }
                }
            )

            // 模型
            OutlinedTextField(
                value = model,
                onValueChange = { model = it },
                label = { Text("模型") },
                placeholder = { Text("gpt-4o") },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true
            )

            // 系统提示词
            OutlinedTextField(
                value = systemPrompt,
                onValueChange = { systemPrompt = it },
                label = { Text("系统提示词") },
                placeholder = { Text("你是一个有用的 AI 助手。") },
                modifier = Modifier.fillMaxWidth(),
                minLines = 3,
                maxLines = 6
            )

            Spacer(Modifier.height(8.dp))

            // 保存按钮
            Button(
                onClick = { onSave(baseUrl, apiKey, model, systemPrompt); onBack() },
                modifier = Modifier.fillMaxWidth().height(48.dp)
            ) {
                Text("保存")
            }

            // 预设快捷配置
            Text(
                "快捷配置",
                fontWeight = FontWeight.Medium,
                fontSize = 14.sp,
                modifier = Modifier.padding(top = 8.dp)
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                AssistChip(
                    onClick = {
                        baseUrl = "https://api.openai.com/v1"
                        model = "gpt-4o"
                    },
                    label = { Text("OpenAI") },
                    modifier = Modifier.weight(1f)
                )
                AssistChip(
                    onClick = {
                        baseUrl = "https://api.deepseek.com/v1"
                        model = "deepseek-chat"
                    },
                    label = { Text("DeepSeek") },
                    modifier = Modifier.weight(1f)
                )
                AssistChip(
                    onClick = {
                        baseUrl = "https://api.moonshot.cn/v1"
                        model = "moonshot-v1-8k"
                    },
                    label = { Text("Kimi") },
                    modifier = Modifier.weight(1f)
                )
            }
        }
    }
}
