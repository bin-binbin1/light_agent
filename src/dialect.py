"""
Dialect 模块 - SQL 方言适配

当前只需要 SQLite；MySQL 相关常量为后期迁移预留。
用法：
    from .dialect import get_dialect
    d = get_dialect("sqlite")
    ph = d["placeholder"]        # "?" / "%s"
    sql = f"INSERT INTO x VALUES ({ph}, {ph})"
    conn.execute(sql, (a, b))
"""

from typing import Dict, Any


DIALECTS: Dict[str, Dict[str, Any]] = {
    "sqlite": {
        "placeholder": "?",
        "insert_or_ignore": "INSERT OR IGNORE",
        "insert_or_replace": "INSERT OR REPLACE",
        "autoincrement_pk": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "supports_executescript": True,
        "text_type": "TEXT",
        "bigint_type": "INTEGER",
    },
    "mysql": {
        "placeholder": "%s",
        "insert_or_ignore": "INSERT IGNORE",
        "insert_or_replace": "REPLACE",
        "autoincrement_pk": "BIGINT PRIMARY KEY AUTO_INCREMENT",
        "supports_executescript": False,
        "text_type": "TEXT",
        "bigint_type": "BIGINT",
    },
}


def get_dialect(name: str) -> Dict[str, Any]:
    """获取方言配置字典，未知方言 fallback 到 sqlite"""
    if name not in DIALECTS:
        return DIALECTS["sqlite"]
    return DIALECTS[name]
