package com.binbin.android_chat.data

import com.google.gson.Gson
import com.google.gson.JsonParser
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.util.concurrent.TimeUnit

/**
 * OpenAI 兼容 API 客户端
 * 支持直连（用户自有 Key）和中转（服务端代理）两种模式
 */
class ApiClient(
    private var baseUrl: String = "https://api.openai.com/v1",
    private var apiKey: String = "",
    private var model: String = "gpt-4o"
) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    private val gson = Gson()
    private val jsonType = "application/json".toMediaType()

    fun updateConfig(baseUrl: String, apiKey: String, model: String) {
        this.baseUrl = baseUrl.trimEnd('/')
        this.apiKey = apiKey
        this.model = model
    }

    data class ChatMessage(val role: String, val content: String)

    /**
     * 流式调用 — 逐 token 返回
     */
    fun chatStream(messages: List<ChatMessage>): Flow<String> = flow {
        val payload = buildPayload(messages, stream = true)

        val request = Request.Builder()
            .url("$baseUrl/chat/completions")
            .addHeader("Authorization", "Bearer $apiKey")
            .addHeader("Content-Type", "application/json")
            .post(payload.toRequestBody(jsonType))
            .build()

        val response = client.newCall(request).execute()

        if (!response.isSuccessful) {
            val body = response.body?.string() ?: ""
            throw ApiException(response.code, "API 错误 (${response.code}): $body")
        }

        val source = response.body?.source() ?: throw ApiException(0, "Empty response")

        while (!source.exhausted()) {
            val line = source.readUtf8Line() ?: continue

            if (!line.startsWith("data: ")) continue
            val data = line.removePrefix("data: ").trim()
            if (data == "[DONE]") break

            try {
                val json = JsonParser.parseString(data).asJsonObject
                val choices = json.getAsJsonArray("choices")
                if (choices != null && choices.size() > 0) {
                    val delta = choices[0].asJsonObject.getAsJsonObject("delta")
                    val content = delta?.get("content")?.asString
                    if (content != null) {
                        emit(content)
                    }
                }
            } catch (_: Exception) {
                // 跳过解析失败的行
            }
        }

        response.close()
    }.flowOn(Dispatchers.IO)

    /**
     * 同步调用 — 一次返回完整结果
     */
    suspend fun chat(messages: List<ChatMessage>): String = withContext(Dispatchers.IO) {
        val payload = buildPayload(messages, stream = false)

        val request = Request.Builder()
            .url("$baseUrl/chat/completions")
            .addHeader("Authorization", "Bearer $apiKey")
            .addHeader("Content-Type", "application/json")
            .post(payload.toRequestBody(jsonType))
            .build()

        val response = client.newCall(request).execute()

        if (!response.isSuccessful) {
            val body = response.body?.string() ?: ""
            throw ApiException(response.code, "API 错误 (${response.code}): $body")
        }

        val body = response.body?.string() ?: throw ApiException(0, "Empty response")
        val json = JsonParser.parseString(body).asJsonObject
        val choices = json.getAsJsonArray("choices")

        if (choices != null && choices.size() > 0) {
            choices[0].asJsonObject
                .getAsJsonObject("message")
                ?.get("content")?.asString ?: ""
        } else {
            ""
        }
    }

    private fun buildPayload(messages: List<ChatMessage>, stream: Boolean): String {
        val map = mutableMapOf<String, Any>(
            "model" to model,
            "messages" to messages.map { mapOf("role" to it.role, "content" to it.content) },
            "stream" to stream
        )
        return gson.toJson(map)
    }

    class ApiException(val code: Int, message: String) : Exception(message)
}
