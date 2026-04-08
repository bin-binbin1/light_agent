"""Memory 模块测试"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory import Memory, MemoryConfig


def test_basic():
    config = MemoryConfig(db_path="/tmp/test_memory.db")
    memory = Memory(config)

    # 清理旧数据
    memory.clear_session("test")

    memory.create_session("test")
    memory.add_message("test", "user", "你好")
    memory.add_message("test", "assistant", "你好，有什么可以帮你？")

    msgs = memory.get_all_messages("test")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "你好"

    count = memory.get_message_count("test")
    assert count == 2

    memory.clear_session("test")
    memory.close()

    os.remove("/tmp/test_memory.db")
    print("✅ test_basic passed")


def test_sessions():
    config = MemoryConfig(db_path="/tmp/test_memory2.db")
    memory = Memory(config)

    memory.create_session("s1")
    memory.create_session("s2")
    memory.add_message("s1", "user", "消息1")
    memory.add_message("s2", "user", "消息2")

    sessions = memory.list_sessions()
    assert len(sessions) == 2

    memory.close()
    os.remove("/tmp/test_memory2.db")
    print("✅ test_sessions passed")


if __name__ == "__main__":
    test_basic()
    test_sessions()
    print("\n全部测试通过 ✅")
