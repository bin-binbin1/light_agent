package com.binbin.android_chat.data

import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Dao
interface ChatDao {
    // 会话
    @Query("SELECT * FROM conversations ORDER BY lastTime DESC")
    fun getAllConversations(): Flow<List<Conversation>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertConversation(conversation: Conversation)

    @Query("UPDATE conversations SET title = :title, lastMessage = :lastMessage, lastTime = :lastTime WHERE id = :id")
    suspend fun updateConversation(id: String, title: String, lastMessage: String, lastTime: Long)

    @Query("DELETE FROM conversations WHERE id = :id")
    suspend fun deleteConversation(id: String)

    // 消息
    @Query("SELECT * FROM messages WHERE conversationId = :conversationId ORDER BY timestamp ASC")
    fun getMessages(conversationId: String): Flow<List<Message>>

    @Insert
    suspend fun insertMessage(message: Message): Long

    @Query("UPDATE messages SET content = :content, isStreaming = :isStreaming WHERE id = :id")
    suspend fun updateMessage(id: Long, content: String, isStreaming: Boolean)

    @Query("DELETE FROM messages WHERE conversationId = :conversationId")
    suspend fun deleteMessages(conversationId: String)
}
