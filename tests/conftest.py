"""pytest 兼容层：把硬编码的 /tmp/ 路径改写到系统 tempdir，让测试能在 Windows 上跑。
仅测试环境使用，不影响生产。"""

import os
import sqlite3
import tempfile


def _map_tmp(p):
    if isinstance(p, str) and p.startswith("/tmp/"):
        return os.path.join(tempfile.gettempdir(), p[len("/tmp/"):])
    return p


_real_connect = sqlite3.connect
def _wrapped_connect(db, *a, **kw):
    return _real_connect(_map_tmp(db), *a, **kw)
sqlite3.connect = _wrapped_connect

_real_remove = os.remove
def _wrapped_remove(p):
    return _real_remove(_map_tmp(p))
os.remove = _wrapped_remove
