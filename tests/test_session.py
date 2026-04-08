"""Session Manager 测试"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.session import SessionManager


def test_basic():
    db = "/tmp/test_session.db"
    if os.path.exists(db):
        os.remove(db)

    sm = SessionManager(db_path=db)

    # 创建用户和会话
    sm.ensure_user("user_001", "张三")
    sid = sm.create_session("user_001", title="第一次对话")

    # 验证
    info = sm.get_session(sid, "user_001")
    assert info is not None
    assert info["user_id"] == "user_001"
    assert info["title"] == "第一次对话"

    # Memory 隔离
    mem = sm.get_memory("user_001", sid)
    mem.add_message(sid, "user", "你好")
    mem.add_message(sid, "assistant", "你好张三")

    msgs = mem.get_all_messages(sid)
    assert len(msgs) == 2

    sm.close()
    os.remove(db)
    print("✅ test_basic passed")


def test_multi_user_isolation():
    db = "/tmp/test_session2.db"
    if os.path.exists(db):
        os.remove(db)

    sm = SessionManager(db_path=db)

    # 用户 A
    sm.ensure_user("user_a", "Alice")
    sid_a = sm.create_session("user_a")
    mem_a = sm.get_memory("user_a", sid_a)
    mem_a.add_message(sid_a, "user", "用户A的秘密")

    # 用户 B
    sm.ensure_user("user_b", "Bob")
    sid_b = sm.create_session("user_b")
    mem_b = sm.get_memory("user_b", sid_b)
    mem_b.add_message(sid_b, "user", "用户B的内容")

    # 验证隔离
    msgs_a = mem_a.get_all_messages(sid_a)
    msgs_b = mem_b.get_all_messages(sid_b)
    assert len(msgs_a) == 1 and msgs_a[0]["content"] == "用户A的秘密"
    assert len(msgs_b) == 1 and msgs_b[0]["content"] == "用户B的内容"

    # 验证用户A不能访问用户B的会话
    try:
        sm.get_memory("user_a", sid_b)
        assert False, "应该报错"
    except ValueError:
        pass

    # 列出会话
    sessions_a = sm.list_sessions("user_a")
    sessions_b = sm.list_sessions("user_b")
    assert len(sessions_a) == 1
    assert len(sessions_b) == 1
    assert sessions_a[0]["session_id"] != sessions_b[0]["session_id"]

    sm.close()
    os.remove(db)
    print("✅ test_multi_user_isolation passed")


def test_multi_session():
    db = "/tmp/test_session3.db"
    if os.path.exists(db):
        os.remove(db)

    sm = SessionManager(db_path=db)
    sm.ensure_user("user_001")

    # 同一用户多个会话
    s1 = sm.create_session("user_001", title="工作对话")
    s2 = sm.create_session("user_001", title="闲聊")

    mem1 = sm.get_memory("user_001", s1)
    mem2 = sm.get_memory("user_001", s2)

    mem1.add_message(s1, "user", "工作相关")
    mem2.add_message(s2, "user", "闲聊内容")

    # 会话间隔离
    assert len(mem1.get_all_messages(s1)) == 1
    assert len(mem2.get_all_messages(s2)) == 1
    assert mem1.get_all_messages(s1)[0]["content"] != mem2.get_all_messages(s2)[0]["content"]

    sessions = sm.list_sessions("user_001")
    assert len(sessions) == 2

    sm.close()
    os.remove(db)
    print("✅ test_multi_session passed")


def test_stats():
    db = "/tmp/test_session4.db"
    if os.path.exists(db):
        os.remove(db)

    sm = SessionManager(db_path=db)
    sm.ensure_user("u1")
    sm.ensure_user("u2")
    sm.create_session("u1")
    sm.create_session("u1")
    sm.create_session("u2")

    stats = sm.get_stats()
    assert stats["users"] == 2
    assert stats["sessions"] == 3

    stats_u1 = sm.get_stats("u1")
    assert stats_u1["sessions"] == 2

    sm.close()
    os.remove(db)
    print("✅ test_stats passed")


if __name__ == "__main__":
    test_basic()
    test_multi_user_isolation()
    test_multi_session()
    test_stats()
    print("\n全部测试通过 ✅")
