"""
Memory 模块 v2 - 全量存储 + 智能检索
保留所有消息，超阈值时用精简版，支持 RAG 检索历史细节
"""

import json
import asyncio
import sqlite3
import time
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MemoryConfig:
    """记忆配置"""
    db_path: str = "memory.db"
    compress_threshold: float = 0.5   # 上下文使用率超过此值时启用精简模式
    keep_ratio: float = 0.3           # 精简时保留最近消息的比例
    idle_compress_hours: float = 6    # 闲置超过此小时数且未压缩过时触发
    rag_top_k: int = 5                # RAG 检索返回条数


class Memory:
    """对话记忆管理 - 全量存储 + 智能检索"""

    def __init__(self, config: MemoryConfig, llm=None, embedding_fn=None):
        self.config = config
        self.llm = llm
        self.embedding_fn = embedding_fn  # 可选的 embedding 函数
        self.conn = sqlite3.connect(config.db_path, check_same_thread=False)
        self._current_session: Optional[str] = None
        self._init_db()

    def _init_db(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                created_at REAL,
                updated_at REAL,
                context_window INTEGER DEFAULT 128000
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id TEXT,
                role TEXT,
                content TEXT,
                tool_calls TEXT,
                tool_call_id TEXT,
                timestamp REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id TEXT,
                summary TEXT,
                message_range_start INTEGER,
                message_range_end INTEGER,
                timestamp REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS message_index (
                message_id INTEGER PRIMARY KEY,
                session_id TEXT,
                user_id TEXT,
                keywords TEXT,
                chunk_index INTEGER,
                FOREIGN KEY (message_id) REFERENCES messages(id)
            );

            CREATE TABLE IF NOT EXISTS context_snapshots (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                context TEXT,
                updated_at REAL
            );
        """)
        self.conn.commit()

    # ─── 会话管理 ───

    def create_session(self, session_id: str, context_window: int = 128000, user_id: str = "default_user"):
        now = time.time()
        self.conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, user_id, created_at, updated_at, context_window) VALUES (?, ?, ?, ?, ?)",
            (session_id, user_id, now, now, context_window)
        )
        self.conn.commit()
        self._current_session = session_id

    def get_current_session(self) -> Optional[str]:
        return self._current_session

    def get_user_id(self, session_id: str) -> str:
        """从 session_id 获取 user_id"""
        cursor = self.conn.execute(
            "SELECT user_id FROM sessions WHERE session_id = ?",
            (session_id,)
        )
        result = cursor.fetchone()
        return result[0] if result else "default_user"

    def touch_session(self, session_id: str):
        self.conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (time.time(), session_id)
        )
        self.conn.commit()

    # ─── 消息存储（全量） ───

    def add_message(self, session_id: str, role: str, content: str,
                    tool_calls: Optional[List[Dict]] = None,
                    tool_call_id: Optional[str] = None):
        now = time.time()
        tc_json = json.dumps(tool_calls) if tool_calls else None
        user_id = self.get_user_id(session_id)
        cursor = self.conn.execute(
            "INSERT INTO messages (session_id, user_id, role, content, tool_calls, tool_call_id, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, user_id, role, content, tc_json, tool_call_id, now)
        )
        msg_id = cursor.lastrowid

        # 为用户和助手消息建立关键词索引
        if role in ("user", "assistant") and content:
            self._index_message(msg_id, session_id, content)

        self.conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))
        self.conn.commit()

    def _index_message(self, msg_id: int, session_id: str, content: str):
        """为消息建立关键词索引"""
        user_id = self.get_user_id(session_id)
        # 简单分词：按标点和空格切分，保留有意义的词
        import re
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+(?:\.\d+)?', content.lower())
        # 过滤太短的词
        keywords = [w for w in words if len(w) > 1]
        if keywords:
            keywords_str = "|".join(keywords[:50])  # 最多50个关键词
            self.conn.execute(
                "INSERT OR REPLACE INTO message_index (message_id, session_id, user_id, keywords) VALUES (?, ?, ?, ?)",
                (msg_id, session_id, user_id, keywords_str)
            )

    def get_all_messages(self, session_id: str) -> List[Dict]:
        """获取全量消息"""
        user_id = self.get_user_id(session_id)
        rows = self.conn.execute(
            "SELECT role, content, tool_calls, tool_call_id FROM messages WHERE session_id = ? AND user_id = ? ORDER BY id ASC",
            (session_id, user_id)
        ).fetchall()
        return self._rows_to_messages(rows)

    def get_recent_messages(self, session_id: str, keep_ratio: float = None) -> List[Dict]:
        """获取最近一部分消息"""
        user_id = self.get_user_id(session_id)
        ratio = keep_ratio or self.config.keep_ratio
        total = self.get_message_count(session_id)
        keep_count = max(int(total * ratio), 6)

        rows = self.conn.execute(
            "SELECT role, content, tool_calls, tool_call_id FROM messages WHERE session_id = ? AND user_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, user_id, keep_count)
        ).fetchall()
        rows.reverse()
        return self._rows_to_messages(rows)

    def _rows_to_messages(self, rows) -> List[Dict]:
        messages = []
        for role, content, tc_json, tc_id in rows:
            msg = {"role": role, "content": content}
            if tc_json:
                msg["tool_calls"] = json.loads(tc_json)
            if tc_id:
                msg["tool_call_id"] = tc_id
            messages.append(msg)
        return messages

    def get_message_count(self, session_id: str) -> int:
        user_id = self.get_user_id(session_id)
        row = self.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ? AND user_id = ?",
            (session_id, user_id)
        ).fetchone()
        return row[0]

    # ─── 压缩判断 ───

    def estimate_token_usage(self, messages: List[Dict]) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content", "") or ""
            total += len(content) * 1.5
            if "tool_calls" in msg:
                total += len(json.dumps(msg["tool_calls"])) * 1.2
        return int(total)

    def should_compress(self, session_id: str, context: List[Dict] = None) -> bool:
        row = self.conn.execute(
            "SELECT context_window FROM sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        if not row:
            return False
        context_window = row[0]
        msgs = context if context is not None else self.get_all_messages(session_id)
        estimated = self.estimate_token_usage(msgs)
        return estimated / context_window > self.config.compress_threshold

    def should_compress_idle(self, session_id: str) -> bool:
        # 如果 idle_compress_hours <= 0，则不进行闲置压缩
        if self.config.idle_compress_hours <= 0:
            return False

        if self.has_been_compressed(session_id):
            return False
        row = self.conn.execute(
            "SELECT updated_at FROM sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        if not row:
            return False
        idle_hours = (time.time() - row[0]) / 3600
        return idle_hours >= self.config.idle_compress_hours and self.get_message_count(session_id) > 0

    def has_been_compressed(self, session_id: str) -> bool:
        user_id = self.get_user_id(session_id)
        row = self.conn.execute(
            "SELECT COUNT(*) FROM summaries WHERE session_id = ? AND user_id = ?",
            (session_id, user_id)
        ).fetchone()
        return row[0] > 0

    # ─── 压缩 ───

    def compress(self, session_id: str, context: List[Dict] = None, summarizer_llm=None) -> List[Dict]:
        """压缩上下文：旧消息生成摘要，返回 [摘要] + [最近消息]"""
        llm = summarizer_llm or self.llm
        if not llm:
            raise ValueError("压缩需要提供 LLM 实例")

        msgs = context if context is not None else self.get_all_messages(session_id)
        total = len(msgs)
        keep_count = max(int(total * self.config.keep_ratio), 6)

        if total <= keep_count:
            return msgs

        to_compress = msgs[:total - keep_count]
        to_keep = msgs[total - keep_count:]

        summary = self._summarize(to_compress, llm)

        new_context = [{"role": "system", "content": f"[历史对话摘要]\n{summary}"}] + to_keep
        return new_context

    def _summarize(self, messages: List[Dict], llm) -> str:
        from .llm import Message

        conv_text = "\n".join([
            f"[{m['role']}]: {m.get('content', '')}"
            for m in messages if m.get("content")
        ])
        if len(conv_text) > 50000:
            conv_text = conv_text[:50000] + "\n... (已截断)"

        summary_messages = [
            Message(role="system", content="你是对话摘要助手。请将以下对话提炼为简洁的摘要，保留关键信息、决策、结论和重要细节。用中文输出。"),
            Message(role="user", content=f"请总结以下对话：\n\n{conv_text}"),
        ]
        response = llm.chat(summary_messages, temperature=0.3, max_tokens=3000)
        return response.content

    # ─── 上下文构建 ───

    def get_context_for_llm(self, session_id: str) -> List[Dict]:
        """获取上下文：摘要 + 最近消息"""
        user_id = self.get_user_id(session_id)
        messages = []

        # 最新摘要
        summary_row = self.conn.execute(
            "SELECT summary FROM summaries WHERE session_id = ? AND user_id = ? ORDER BY id DESC LIMIT 1",
            (session_id, user_id)
        ).fetchone()
        if summary_row:
            messages.append({
                "role": "system",
                "content": f"[历史对话摘要]\n{summary_row[0]}"
            })

        # 最近消息
        messages.extend(self.get_recent_messages(session_id))
        return messages

    # ─── 上下文快照（Write-Back Cache） ───

    def save_context(self, session_id: str, context: List[Dict]):
        """将内存上下文快照存入 SQLite"""
        user_id = self.get_user_id(session_id)
        ctx_json = json.dumps(context, ensure_ascii=False)
        self.conn.execute(
            "INSERT OR REPLACE INTO context_snapshots (session_id, user_id, context, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, user_id, ctx_json, time.time())
        )
        self.conn.commit()

    def load_context(self, session_id: str) -> List[Dict]:
        """从快照加载上下文，无快照则用 get_context_for_llm 兜底"""
        row = self.conn.execute(
            "SELECT context FROM context_snapshots WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        if row and row[0]:
            return json.loads(row[0])
        return self.get_context_for_llm(session_id)

    def delete_context_snapshot(self, session_id: str):
        """删除上下文快照"""
        self.conn.execute(
            "DELETE FROM context_snapshots WHERE session_id = ?",
            (session_id,)
        )
        self.conn.commit()

    # ─── RAG 检索 ───

    def search_messages(self, session_id: str, query: str, top_k: int = None) -> List[Dict]:
        """关键词检索历史消息"""
        user_id = self.get_user_id(session_id)
        k = top_k or self.config.rag_top_k

        import re
        query_words = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+(?:\.\d+)?', query.lower()))
        query_words = {w for w in query_words if len(w) > 1}

        if not query_words:
            return []

        # 构造 LIKE 查询
        conditions = []
        params = [session_id, user_id]
        for word in query_words:
            conditions.append("mi.keywords LIKE ?")
            params.append(f"%{word}%")

        where = " OR ".join(conditions)

        sql = f"""
            SELECT m.id, m.role, m.content, m.timestamp, mi.keywords
            FROM messages m
            JOIN message_index mi ON m.id = mi.message_id
            WHERE m.session_id = ? AND m.user_id = ? AND ({where})
            ORDER BY m.id DESC
            LIMIT ?
        """
        params.append(k * 3)  # 多取一些，后面排序

        rows = self.conn.execute(sql, params).fetchall()

        # 按关键词匹配度排序
        scored = []
        for msg_id, role, content, ts, keywords_str in rows:
            kw_set = set(keywords_str.split("|")) if keywords_str else set()
            score = len(query_words & kw_set)
            scored.append((score, msg_id, role, content, ts))

        scored.sort(key=lambda x: -x[0])

        results = []
        seen_contents = set()
        for score, msg_id, role, content, ts in scored[:k]:
            # 去重相似内容
            content_short = content[:100]
            if content_short in seen_contents:
                continue
            seen_contents.add(content_short)

            results.append({
                "message_id": msg_id,
                "role": role,
                "content": content,
                "timestamp": ts,
                "match_score": score,
                "time_str": time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
            })

        return results

    def search_by_time(self, session_id: str, start_time: float = None,
                       end_time: float = None, limit: int = 20) -> List[Dict]:
        """按时间范围检索"""
        conditions = ["session_id = ?"]
        params = [session_id]

        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)

        where = " AND ".join(conditions)
        sql = f"SELECT role, content, timestamp FROM messages WHERE {where} ORDER BY id DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()
        return [
            {"role": r[0], "content": r[1], "timestamp": r[2],
             "time_str": time.strftime("%Y-%m-%d %H:%M", time.localtime(r[2]))}
            for r in rows
        ]

    # ─── 会话管理 ───

    def clear_session(self, session_id: str):
        user_id = self.get_user_id(session_id)
        self.conn.execute("DELETE FROM message_index WHERE session_id = ? AND user_id = ?", (session_id, user_id))
        self.conn.execute("DELETE FROM messages WHERE session_id = ? AND user_id = ?", (session_id, user_id))
        self.conn.execute("DELETE FROM summaries WHERE session_id = ? AND user_id = ?", (session_id, user_id))
        self.conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        self.conn.commit()

    def list_sessions(self) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT session_id, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [
            {"session_id": r[0], "created_at": r[1], "updated_at": r[2]}
            for r in rows
        ]

    def get_latest_session(self, user_id: str) -> Optional[str]:
        """获取用户最近活跃的 session_id"""
        row = self.conn.execute(
            "SELECT session_id FROM sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT 1",
            (user_id,)
        ).fetchone()
        return row[0] if row else None

    def close(self):
        self.conn.close()

    # ─── 异步包装 ───

    async def aadd_message(self, session_id: str, role: str, content: str,
                           tool_calls=None, tool_call_id=None):
        await asyncio.to_thread(self.add_message, session_id, role, content, tool_calls, tool_call_id)

    async def atouch_session(self, session_id: str):
        await asyncio.to_thread(self.touch_session, session_id)

    async def ashould_compress(self, session_id: str, context: List[Dict] = None) -> bool:
        return await asyncio.to_thread(self.should_compress, session_id, context)

    async def ashould_compress_idle(self, session_id: str) -> bool:
        return await asyncio.to_thread(self.should_compress_idle, session_id)

    async def acompress(self, session_id: str, context: List[Dict] = None, summarizer_llm=None) -> List[Dict]:
        return await asyncio.to_thread(self.compress, session_id, context, summarizer_llm)

    async def aget_context_for_llm(self, session_id: str) -> List[Dict]:
        return await asyncio.to_thread(self.get_context_for_llm, session_id)

    async def asave_context(self, session_id: str, context: List[Dict]):
        await asyncio.to_thread(self.save_context, session_id, context)

    async def aload_context(self, session_id: str) -> List[Dict]:
        return await asyncio.to_thread(self.load_context, session_id)
