"""
Session Manager - 会话管理
单库，user_id + session_id 双层隔离
"""

import uuid
import time
import sqlite3
from typing import Optional, List, Dict
from .memory import Memory, MemoryConfig


class SessionManager:
    """会话管理器 - 多用户多会话隔离"""

    def __init__(self, db_path: str = "memory.db", default_context_window: int = 128000,
                 compress_threshold: float = 0.5, keep_ratio: float = 0.3,
                 idle_compress_hours: float = 6):
        self.db_path = db_path
        self.default_context_window = default_context_window
        self.conn = sqlite3.connect(db_path)
        self._init_db()

        # 缓存当前活跃的 Memory 实例
        self._memory_cache: Dict[str, Memory] = {}

        # 全局配置
        self.compress_threshold = compress_threshold
        self.keep_ratio = keep_ratio
        self.idle_compress_hours = idle_compress_hours

    def _init_db(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                created_at REAL,
                last_active_at REAL,
                display_name TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                title TEXT DEFAULT '',
                created_at REAL,
                updated_at REAL,
                context_window INTEGER DEFAULT 128000,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
        """)
        self.conn.commit()

    # ─── 用户管理 ───

    def ensure_user(self, user_id: str, display_name: str = ""):
        """确保用户存在"""
        now = time.time()
        self.conn.execute(
            "INSERT OR IGNORE INTO users (user_id, created_at, last_active_at, display_name) VALUES (?, ?, ?, ?)",
            (user_id, now, now, display_name)
        )
        self.conn.execute(
            "UPDATE users SET last_active_at = ?, display_name = CASE WHEN ? != '' THEN ? ELSE display_name END WHERE user_id = ?",
            (now, display_name, display_name, user_id)
        )
        self.conn.commit()

    def get_user_info(self, user_id: str) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT user_id, created_at, last_active_at, display_name FROM users WHERE user_id = ?",
            (user_id,)
        ).fetchone()
        if not row:
            return None
        return {
            "user_id": row[0],
            "created_at": row[1],
            "last_active_at": row[2],
            "display_name": row[3]
        }

    def list_users(self) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT user_id, created_at, last_active_at, display_name FROM users ORDER BY last_active_at DESC"
        ).fetchall()
        return [
            {"user_id": r[0], "created_at": r[1], "last_active_at": r[2], "display_name": r[3]}
            for r in rows
        ]

    # ─── 会话管理 ───

    def create_session(self, user_id: str, session_id: str = None,
                       title: str = "", context_window: int = None) -> str:
        """创建会话，返回 session_id"""
        self.ensure_user(user_id)
        sid = session_id or str(uuid.uuid4())[:12]
        cw = context_window or self.default_context_window
        now = time.time()

        self.conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, user_id, title, created_at, updated_at, context_window) VALUES (?, ?, ?, ?, ?, ?)",
            (sid, user_id, title, now, now, cw)
        )
        self.conn.commit()
        return sid

    def get_session(self, session_id: str, user_id: str = None) -> Optional[Dict]:
        """获取会话信息"""
        if user_id:
            row = self.conn.execute(
                "SELECT session_id, user_id, title, created_at, updated_at, context_window FROM sessions WHERE session_id = ? AND user_id = ?",
                (session_id, user_id)
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT session_id, user_id, title, created_at, updated_at, context_window FROM sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()

        if not row:
            return None
        return {
            "session_id": row[0],
            "user_id": row[1],
            "title": row[2],
            "created_at": row[3],
            "updated_at": row[4],
            "context_window": row[5]
        }

    def list_sessions(self, user_id: str) -> List[Dict]:
        """列出用户的所有会话"""
        rows = self.conn.execute(
            "SELECT session_id, title, created_at, updated_at FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,)
        ).fetchall()
        return [
            {"session_id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]}
            for r in rows
        ]

    def delete_session(self, session_id: str, user_id: str = None):
        """删除会话及其所有数据"""
        # 先关掉缓存的 Memory
        cache_key = f"{user_id}:{session_id}" if user_id else session_id
        if cache_key in self._memory_cache:
            self._memory_cache[cache_key].close()
            del self._memory_cache[cache_key]

        if user_id:
            self.conn.execute("DELETE FROM sessions WHERE session_id = ? AND user_id = ?", (session_id, user_id))
        else:
            self.conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        self.conn.commit()

        # 清理 memory 表中的数据
        memory = Memory(MemoryConfig(db_path=self.db_path))
        memory.clear_session(session_id)
        memory.close()

    def touch_session(self, session_id: str):
        """更新会话活跃时间"""
        self.conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (time.time(), session_id))
        self.conn.commit()

    def update_session_title(self, session_id: str, title: str):
        """更新会话标题"""
        self.conn.execute("UPDATE sessions SET title = ? WHERE session_id = ?", (title, session_id))
        self.conn.commit()

    # ─── Memory 实例获取 ───

    def get_memory(self, user_id: str, session_id: str, llm=None) -> Memory:
        """获取会话的 Memory 实例"""
        cache_key = f"{user_id}:{session_id}"

        if cache_key not in self._memory_cache:
            # 确保会话存在
            session = self.get_session(session_id, user_id)
            if not session:
                raise ValueError(f"会话不存在: user={user_id}, session={session_id}")

            config = MemoryConfig(
                db_path=self.db_path,
                compress_threshold=self.compress_threshold,
                keep_ratio=self.keep_ratio,
                idle_compress_hours=self.idle_compress_hours
            )
            memory = Memory(config, llm=llm)
            memory.create_session(session_id, session["context_window"])
            self._memory_cache[cache_key] = memory

        return self._memory_cache[cache_key]

    # ─── 统计 ───

    def get_stats(self, user_id: str = None) -> Dict:
        """获取统计信息"""
        if user_id:
            sessions = self.conn.execute("SELECT COUNT(*) FROM sessions WHERE user_id = ?", (user_id,)).fetchone()[0]
            return {"user_id": user_id, "sessions": sessions}
        else:
            users = self.conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            sessions = self.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            return {"users": users, "sessions": sessions}

    def close(self):
        for mem in self._memory_cache.values():
            mem.close()
        self._memory_cache.clear()
        self.conn.close()
