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

from .dialect import get_dialect


@dataclass
class MemoryConfig:
    """记忆配置"""
    db_path: str = "memory.db"
    compress_threshold: float = 0.5   # 上下文使用率超过此值时启用精简模式
    keep_ratio: float = 0.3           # 精简时保留最近消息的比例
    idle_compress_hours: float = 6    # 闲置超过此小时数且未压缩过时触发
    rag_top_k: int = 5                # RAG 检索返回条数
    dialect: str = "sqlite"           # SQL 方言: sqlite / mysql


class Memory:
    """对话记忆管理 - 全量存储 + 智能检索"""

    def __init__(self, config: MemoryConfig, llm=None, embedding_fn=None):
        self.config = config
        self.llm = llm
        self.embedding_fn = embedding_fn  # 可选的 embedding 函数
        self._d = get_dialect(config.dialect)
        self._ph = self._d["placeholder"]
        self.conn = sqlite3.connect(config.db_path, check_same_thread=False)
        self._current_session: Optional[str] = None
        self._init_db()
        self._maybe_migrate_keyword_index()

    def _init_db(self):
        pk = self._d["autoincrement_pk"]  # sqlite: "INTEGER PRIMARY KEY AUTOINCREMENT" / mysql: "BIGINT PRIMARY KEY AUTO_INCREMENT"
        stmts = [
            """CREATE TABLE IF NOT EXISTS sessions (
                session_id VARCHAR(64) PRIMARY KEY,
                user_id VARCHAR(64),
                created_at REAL,
                updated_at REAL,
                context_window INTEGER DEFAULT 128000
            )""",
            f"""CREATE TABLE IF NOT EXISTS messages (
                id {pk},
                session_id VARCHAR(64),
                user_id VARCHAR(64),
                role VARCHAR(32),
                content TEXT,
                tool_calls TEXT,
                tool_call_id VARCHAR(128),
                timestamp REAL
            )""",
            f"""CREATE TABLE IF NOT EXISTS summaries (
                id {pk},
                session_id VARCHAR(64),
                user_id VARCHAR(64),
                summary TEXT,
                message_range_start INTEGER,
                message_range_end INTEGER,
                timestamp REAL
            )""",
            # 老索引表（保留用于迁移回填，后续不再写入）
            """CREATE TABLE IF NOT EXISTS message_index (
                message_id INTEGER PRIMARY KEY,
                session_id VARCHAR(64),
                user_id VARCHAR(64),
                keywords TEXT,
                chunk_index INTEGER
            )""",
            # 新独立倒排表：每个关键词一行，可走 B-Tree 精确索引
            f"""CREATE TABLE IF NOT EXISTS keyword_index (
                id {pk},
                keyword VARCHAR(64) NOT NULL,
                message_id INTEGER NOT NULL,
                session_id VARCHAR(64) NOT NULL,
                user_id VARCHAR(64) NOT NULL
            )""",
            "CREATE INDEX IF NOT EXISTS ix_ki_lookup ON keyword_index(user_id, session_id, keyword)",
            "CREATE INDEX IF NOT EXISTS ix_ki_msg ON keyword_index(message_id)",
            """CREATE TABLE IF NOT EXISTS context_snapshots (
                session_id VARCHAR(64) PRIMARY KEY,
                user_id VARCHAR(64),
                context TEXT,
                updated_at REAL
            )""",
        ]
        if self._d["supports_executescript"]:
            self.conn.executescript(";\n".join(stmts))
        else:
            for s in stmts:
                self.conn.execute(s)
        self.conn.commit()

    # ─── 会话管理 ───

    def create_session(self, session_id: str, context_window: int = 128000, user_id: str = "default_user"):
        now = time.time()
        ph = self._ph
        self.conn.execute(
            f"{self._d['insert_or_ignore']} INTO sessions (session_id, user_id, created_at, updated_at, context_window) VALUES ({ph}, {ph}, {ph}, {ph}, {ph})",
            (session_id, user_id, now, now, context_window)
        )
        self.conn.commit()
        self._current_session = session_id

    def get_current_session(self) -> Optional[str]:
        return self._current_session

    def get_user_id(self, session_id: str) -> str:
        """从 session_id 获取 user_id"""
        ph = self._ph
        cursor = self.conn.execute(
            f"SELECT user_id FROM sessions WHERE session_id = {ph}",
            (session_id,)
        )
        result = cursor.fetchone()
        return result[0] if result else "default_user"

    def touch_session(self, session_id: str):
        ph = self._ph
        self.conn.execute(
            f"UPDATE sessions SET updated_at = {ph} WHERE session_id = {ph}",
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
        ph = self._ph
        cursor = self.conn.execute(
            f"INSERT INTO messages (session_id, user_id, role, content, tool_calls, tool_call_id, timestamp) VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})",
            (session_id, user_id, role, content, tc_json, tool_call_id, now)
        )
        msg_id = cursor.lastrowid

        # 为用户和助手消息建立关键词索引
        if role in ("user", "assistant") and content:
            self._index_message(msg_id, session_id, content)

        self.conn.execute(
            f"UPDATE sessions SET updated_at = {ph} WHERE session_id = {ph}",
            (now, session_id)
        )
        self.conn.commit()

    @staticmethod
    def _tokenize(content: str) -> List[str]:
        """统一分词逻辑：中文连续串 / 英文串 / 数字，过滤长度 <= 1 的词，最多 50 个"""
        import re
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+(?:\.\d+)?', content.lower())
        # 去重保序
        seen = set()
        kws = []
        for w in words:
            if len(w) > 1 and w not in seen:
                seen.add(w)
                kws.append(w)
                if len(kws) >= 50:
                    break
        return kws

    def _index_message(self, msg_id: int, session_id: str, content: str):
        """为消息建立关键词索引（写独立倒排表 keyword_index）

        策略：先删除该 msg_id 的旧行（兼容 SQLite/MySQL 双库，不用 upsert），
        再批量 executemany 插入每个 keyword 一行。
        """
        user_id = self.get_user_id(session_id)
        keywords = self._tokenize(content)
        if not keywords:
            return
        ph = self._ph
        # 先删（幂等，便于重复建索引）
        self.conn.execute(
            f"DELETE FROM keyword_index WHERE message_id = {ph}",
            (msg_id,)
        )
        rows = [(kw, msg_id, session_id, user_id) for kw in keywords]
        self.conn.executemany(
            f"INSERT INTO keyword_index (keyword, message_id, session_id, user_id) VALUES ({ph}, {ph}, {ph}, {ph})",
            rows
        )

    def _maybe_migrate_keyword_index(self):
        """首次启动检测：若新表 keyword_index 空而老表 message_index 非空，一次性回填。

        幂等：完成后新表非空，下次启动不再触发。
        """
        try:
            row = self.conn.execute("SELECT 1 FROM keyword_index LIMIT 1").fetchone()
            if row:
                return  # 新表已有数据，跳过
            row = self.conn.execute("SELECT 1 FROM message_index LIMIT 1").fetchone()
            if not row:
                return  # 老表也空，无需迁移
        except Exception:
            return  # 表可能还不存在（极早期），静默跳过

        ph = self._ph
        cursor = self.conn.execute(
            "SELECT message_id, session_id, user_id, keywords FROM message_index"
        )
        total = 0
        batch: List[Tuple] = []
        for msg_id, sid, uid, kws_str in cursor.fetchall():
            if not kws_str:
                continue
            for kw in kws_str.split("|"):
                kw = kw.strip()
                if len(kw) > 1:
                    batch.append((kw, msg_id, sid, uid or "default_user"))
            if len(batch) >= 500:
                self.conn.executemany(
                    f"INSERT INTO keyword_index (keyword, message_id, session_id, user_id) VALUES ({ph}, {ph}, {ph}, {ph})",
                    batch
                )
                total += len(batch)
                batch = []
        if batch:
            self.conn.executemany(
                f"INSERT INTO keyword_index (keyword, message_id, session_id, user_id) VALUES ({ph}, {ph}, {ph}, {ph})",
                batch
            )
            total += len(batch)
        self.conn.commit()
        if total > 0:
            try:
                from .agent_logging import default_logger as _lg
                _lg.system(f"keyword_index migrated {total} rows from message_index")
            except Exception:
                pass

    def get_all_messages(self, session_id: str) -> List[Dict]:
        """获取全量消息"""
        user_id = self.get_user_id(session_id)
        ph = self._ph
        rows = self.conn.execute(
            f"SELECT role, content, tool_calls, tool_call_id FROM messages WHERE session_id = {ph} AND user_id = {ph} ORDER BY id ASC",
            (session_id, user_id)
        ).fetchall()
        return self._rows_to_messages(rows)

    def get_recent_messages(self, session_id: str, keep_ratio: float = None) -> List[Dict]:
        """获取最近一部分消息"""
        user_id = self.get_user_id(session_id)
        ratio = keep_ratio or self.config.keep_ratio
        total = self.get_message_count(session_id)
        keep_count = max(int(total * ratio), 6)

        ph = self._ph
        rows = self.conn.execute(
            f"SELECT role, content, tool_calls, tool_call_id FROM messages WHERE session_id = {ph} AND user_id = {ph} ORDER BY id DESC LIMIT {ph}",
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
        ph = self._ph
        row = self.conn.execute(
            f"SELECT COUNT(*) FROM messages WHERE session_id = {ph} AND user_id = {ph}",
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
        ph = self._ph
        row = self.conn.execute(
            f"SELECT context_window FROM sessions WHERE session_id = {ph}",
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
        ph = self._ph
        row = self.conn.execute(
            f"SELECT updated_at FROM sessions WHERE session_id = {ph}",
            (session_id,)
        ).fetchone()
        if not row:
            return False
        idle_hours = (time.time() - row[0]) / 3600
        return idle_hours >= self.config.idle_compress_hours and self.get_message_count(session_id) > 0

    def has_been_compressed(self, session_id: str) -> bool:
        user_id = self.get_user_id(session_id)
        ph = self._ph
        row = self.conn.execute(
            f"SELECT COUNT(*) FROM summaries WHERE session_id = {ph} AND user_id = {ph}",
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
        ph = self._ph
        summary_row = self.conn.execute(
            f"SELECT summary FROM summaries WHERE session_id = {ph} AND user_id = {ph} ORDER BY id DESC LIMIT 1",
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
        ph = self._ph
        self.conn.execute(
            f"{self._d['insert_or_replace']} INTO context_snapshots (session_id, user_id, context, updated_at) VALUES ({ph}, {ph}, {ph}, {ph})",
            (session_id, user_id, ctx_json, time.time())
        )
        self.conn.commit()

    def load_context(self, session_id: str) -> List[Dict]:
        """从快照加载上下文，无快照则用 get_context_for_llm 兜底"""
        ph = self._ph
        row = self.conn.execute(
            f"SELECT context FROM context_snapshots WHERE session_id = {ph}",
            (session_id,)
        ).fetchone()
        if row and row[0]:
            return json.loads(row[0])
        return self.get_context_for_llm(session_id)

    def delete_context_snapshot(self, session_id: str):
        """删除上下文快照"""
        ph = self._ph
        self.conn.execute(
            f"DELETE FROM context_snapshots WHERE session_id = {ph}",
            (session_id,)
        )
        self.conn.commit()

    # ─── RAG 检索 ───

    def search_messages(self, session_id: str, query: str, top_k: int = None) -> List[Dict]:
        """关键词检索历史消息（走独立倒排表 keyword_index）

        步骤：
        1. 对 query 分词，过滤短词
        2. 在 keyword_index 上用 IN (...) + GROUP BY 聚合匹配分数（走 (user_id, session_id, keyword) 索引）
        3. JOIN messages 拉回原文，按匹配度降序、再按 id 降序
        """
        user_id = self.get_user_id(session_id)
        k = top_k or self.config.rag_top_k

        query_words = list({w for w in self._tokenize(query)})
        if not query_words:
            return []

        ph = self._ph
        in_placeholders = ",".join([ph] * len(query_words))
        # 先在倒排表上聚合分数（限制候选集 = k*3，避免全表聚合）
        candidate_sql = f"""
            SELECT ki.message_id, COUNT(DISTINCT ki.keyword) AS score
            FROM keyword_index ki
            WHERE ki.user_id = {ph} AND ki.session_id = {ph}
              AND ki.keyword IN ({in_placeholders})
            GROUP BY ki.message_id
            ORDER BY score DESC, ki.message_id DESC
            LIMIT {ph}
        """
        params = [user_id, session_id] + query_words + [k * 3]
        cand_rows = self.conn.execute(candidate_sql, params).fetchall()
        if not cand_rows:
            return []

        msg_ids = [r[0] for r in cand_rows]
        score_map = {r[0]: r[1] for r in cand_rows}

        # JOIN messages 拉原文
        in_ids = ",".join([ph] * len(msg_ids))
        msg_sql = f"""
            SELECT id, role, content, timestamp
            FROM messages
            WHERE session_id = {ph} AND user_id = {ph} AND id IN ({in_ids})
        """
        msg_rows = self.conn.execute(msg_sql, [session_id, user_id] + msg_ids).fetchall()

        scored = [
            (score_map.get(mid, 0), mid, role, content, ts)
            for (mid, role, content, ts) in msg_rows
        ]
        # 按分数 DESC、id DESC（新消息优先）排序
        scored.sort(key=lambda x: (-x[0], -x[1]))

        results = []
        seen_contents = set()
        for score, msg_id, role, content, ts in scored[:k]:
            content_short = (content or "")[:100]
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
        ph = self._ph
        conditions = [f"session_id = {ph}"]
        params = [session_id]

        if start_time:
            conditions.append(f"timestamp >= {ph}")
            params.append(start_time)
        if end_time:
            conditions.append(f"timestamp <= {ph}")
            params.append(end_time)

        where = " AND ".join(conditions)
        sql = f"SELECT role, content, timestamp FROM messages WHERE {where} ORDER BY id DESC LIMIT {ph}"
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
        ph = self._ph
        self.conn.execute(f"DELETE FROM message_index WHERE session_id = {ph} AND user_id = {ph}", (session_id, user_id))
        self.conn.execute(f"DELETE FROM keyword_index WHERE session_id = {ph} AND user_id = {ph}", (session_id, user_id))
        self.conn.execute(f"DELETE FROM messages WHERE session_id = {ph} AND user_id = {ph}", (session_id, user_id))
        self.conn.execute(f"DELETE FROM summaries WHERE session_id = {ph} AND user_id = {ph}", (session_id, user_id))
        self.conn.execute(f"DELETE FROM sessions WHERE session_id = {ph}", (session_id,))
        self.conn.commit()

    def clear_session_messages(self, session_id: str):
        """清空 session 的消息/摘要/索引/快照，但保留 sessions 行，使 session_id 继续可用"""
        user_id = self.get_user_id(session_id)
        ph = self._ph
        now = time.time()
        self.conn.execute(f"DELETE FROM message_index WHERE session_id = {ph} AND user_id = {ph}", (session_id, user_id))
        self.conn.execute(f"DELETE FROM keyword_index WHERE session_id = {ph} AND user_id = {ph}", (session_id, user_id))
        self.conn.execute(f"DELETE FROM messages WHERE session_id = {ph} AND user_id = {ph}", (session_id, user_id))
        self.conn.execute(f"DELETE FROM summaries WHERE session_id = {ph} AND user_id = {ph}", (session_id, user_id))
        self.conn.execute(f"DELETE FROM context_snapshots WHERE session_id = {ph}", (session_id,))
        self.conn.execute(f"UPDATE sessions SET updated_at = {ph} WHERE session_id = {ph}", (now, session_id))
        self.conn.commit()

    def session_owner(self, session_id: str) -> Optional[str]:
        """返回该 session_id 的 user_id；不存在则 None"""
        ph = self._ph
        row = self.conn.execute(
            f"SELECT user_id FROM sessions WHERE session_id = {ph}",
            (session_id,)
        ).fetchone()
        return row[0] if row else None

    def list_sessions(self) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT session_id, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [
            {"session_id": r[0], "created_at": r[1], "updated_at": r[2]}
            for r in rows
        ]

    def list_sessions_by_user(self, user_id: str) -> List[Dict]:
        """按 updated_at 倒序列出该用户的所有 session"""
        ph = self._ph
        rows = self.conn.execute(
            f"SELECT session_id, created_at, updated_at FROM sessions WHERE user_id = {ph} ORDER BY updated_at DESC",
            (user_id,)
        ).fetchall()
        return [
            {"session_id": r[0], "created_at": r[1], "updated_at": r[2]}
            for r in rows
        ]

    def get_latest_session(self, user_id: str) -> Optional[str]:
        """获取用户最近活跃的 session_id"""
        ph = self._ph
        row = self.conn.execute(
            f"SELECT session_id FROM sessions WHERE user_id = {ph} ORDER BY updated_at DESC LIMIT 1",
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

    async def aclear_session_messages(self, session_id: str):
        await asyncio.to_thread(self.clear_session_messages, session_id)
