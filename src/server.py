"""
Server 模块 - HTTP 服务接口
基于 Flask，轻量级 API 服务
"""

import json
import os
import time
import threading
from collections import OrderedDict
from typing import Optional

try:
    from flask import Flask, request, jsonify, Response, stream_with_context
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

from .session import SessionManager
from .agent import Agent, AgentConfig
from .memory import MemoryConfig
from .llm import LLMFactory, LLMType, Message
from .tools import create_default_tools
from .logging import Logger, LogConfig, LogLevel
from .config import Config


class LRUAgentCache:
    """线程安全的 LRU Agent 缓存，淘汰时自动释放资源"""

    def __init__(self, max_size: int = 500):
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._max_size = max_size

    def get(self, key: str) -> Optional[Agent]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key: str, agent: Agent):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = agent
            else:
                if len(self._cache) >= self._max_size:
                    _, evicted = self._cache.popitem(last=False)
                    self._cleanup(evicted)
                self._cache[key] = agent

    def _cleanup(self, agent: Agent):
        """释放被淘汰 Agent 持有的资源"""
        try:
            if hasattr(agent, 'memory') and hasattr(agent.memory, 'close'):
                agent.memory.close()
        except Exception:
            pass


def create_app(config_path: str = "config/config.json") -> Flask:
    """创建 Flask 应用"""
    if not HAS_FLASK:
        raise ImportError("需要安装 Flask: pip install flask")

    app = Flask(__name__)
    config = Config(config_path)

    # 初始化
    log_config = LogConfig(
        level=LogLevel.DEBUG if config.get("debug") else LogLevel.INFO
    )
    logger = Logger(log_config)

    sm = SessionManager(
        db_path=config.get("memory_db", "memory.db"),
        compress_threshold=config.compress_threshold,
        keep_ratio=config.keep_ratio,
        idle_compress_hours=config.idle_compress_hours,
    )

    # Agent LRU 缓存（线程安全，淘汰时自动释放资源）
    _agent_cache = LRUAgentCache(max_size=config.get("agent_cache_size", 500))

    def get_agent(user_id: str, session_id: str) -> Agent:
        cache_key = f"{user_id}:{session_id}"
        agent = _agent_cache.get(cache_key)
        if agent is None:
            llm = LLMFactory.create(config.provider, config.api_key, config.model or None)
            mem = sm.get_memory(user_id, session_id, llm=llm)
            agent_config = AgentConfig(
                name=config.get("name", "assistant"),
                system_prompt=config.get("system_prompt", ""),
                context_window=config.context_window,
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 4096),
            )
            agent = Agent(llm, agent_config, tools=create_default_tools(memory=mem), logger=logger)
            agent.memory = mem
            agent.session_id = session_id
            _agent_cache.put(cache_key, agent)
        return agent

    # ─── API 路由 ───

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "time": time.time()})

    # ─── 用户管理 ───

    @app.route("/api/users", methods=["POST"])
    def create_user():
        data = request.json or {}
        user_id = data.get("user_id")
        display_name = data.get("display_name", "")
        if not user_id:
            return jsonify({"error": "user_id required"}), 400
        sm.ensure_user(user_id, display_name)
        return jsonify({"ok": True, "user_id": user_id})

    @app.route("/api/users", methods=["GET"])
    def list_users():
        users = sm.list_users()
        return jsonify({"users": users})

    @app.route("/api/users/<user_id>", methods=["GET"])
    def get_user(user_id):
        info = sm.get_user_info(user_id)
        if not info:
            return jsonify({"error": "user not found"}), 404
        return jsonify(info)

    # ─── 会话管理 ───

    @app.route("/api/sessions", methods=["POST"])
    def create_session():
        data = request.json or {}
        user_id = data.get("user_id")
        if not user_id:
            return jsonify({"error": "user_id required"}), 400

        title = data.get("title", "")
        context_window = data.get("context_window")
        session_id = sm.create_session(user_id, title=title, context_window=context_window)
        return jsonify({"ok": True, "session_id": session_id})

    @app.route("/api/users/<user_id>/sessions", methods=["GET"])
    def list_sessions(user_id):
        sessions = sm.list_sessions(user_id)
        return jsonify({"sessions": sessions})

    @app.route("/api/sessions/<session_id>", methods=["GET"])
    def get_session(session_id):
        user_id = request.args.get("user_id")
        info = sm.get_session(session_id, user_id)
        if not info:
            return jsonify({"error": "session not found"}), 404
        return jsonify(info)

    @app.route("/api/sessions/<session_id>", methods=["DELETE"])
    def delete_session(session_id):
        user_id = request.args.get("user_id")
        sm.delete_session(session_id, user_id)
        return jsonify({"ok": True})

    # ─── 对话 ───

    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.json or {}
        user_id = data.get("user_id")
        session_id = data.get("session_id")
        message = data.get("message", "")

        if not user_id or not session_id or not message:
            return jsonify({"error": "user_id, session_id, message required"}), 400

        # 确保会话存在
        if not sm.get_session(session_id, user_id):
            sm.create_session(user_id, session_id=session_id)

        try:
            agent = get_agent(user_id, session_id)
            response = agent.chat(message)
            sm.touch_session(session_id)
            return jsonify({"response": response})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/chat/stream", methods=["POST"])
    def chat_stream():
        """流式输出（如果模型支持）"""
        data = request.json or {}
        user_id = data.get("user_id")
        session_id = data.get("session_id")
        message = data.get("message", "")

        if not user_id or not session_id or not message:
            return jsonify({"error": "user_id, session_id, message required"}), 400

        if not sm.get_session(session_id, user_id):
            sm.create_session(user_id, session_id=session_id)

        def generate():
            try:
                agent = get_agent(user_id, session_id)
                # 目前先返回完整响应，后续实现真正的流式
                response = agent.chat(message)
                sm.touch_session(session_id)
                # 按字符模拟流式
                for char in response:
                    yield f"data: {json.dumps({'content': char})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    # ─── 消息历史 ───

    @app.route("/api/sessions/<session_id>/messages", methods=["GET"])
    def get_messages(session_id):
        user_id = request.args.get("user_id")
        if not user_id:
            return jsonify({"error": "user_id required"}), 400

        try:
            mem = sm.get_memory(user_id, session_id)
            messages = mem.get_all_messages(session_id)
            return jsonify({"messages": messages})
        except ValueError as e:
            return jsonify({"error": str(e)}), 404

    # ─── 记忆搜索 ───

    @app.route("/api/search", methods=["POST"])
    def search_memory():
        data = request.json or {}
        user_id = data.get("user_id")
        session_id = data.get("session_id")
        query = data.get("query", "")
        top_k = data.get("top_k", 5)

        if not user_id or not session_id or not query:
            return jsonify({"error": "user_id, session_id, query required"}), 400

        try:
            mem = sm.get_memory(user_id, session_id)
            results = mem.search_messages(session_id, query, top_k=top_k)
            return jsonify({"results": results})
        except ValueError as e:
            return jsonify({"error": str(e)}), 404

    # ─── 统计 ───

    @app.route("/api/stats", methods=["GET"])
    def stats():
        user_id = request.args.get("user_id")
        s = sm.get_stats(user_id)
        return jsonify(s)

    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, config_path: str = "config/config.json",
               debug: bool = False, workers: int = 4, server: str = "auto"):
    """启动服务

    Args:
        server: "auto" 自动检测最佳服务器, "waitress" 强制用 waitress, "flask" 用开发服务器
    """
    app = create_app(config_path)
    print(f"Light Agent Server: http://{host}:{port}")
    print(f"   POST /api/chat          - 对话")
    print(f"   POST /api/chat/stream   - 流式对话")
    print(f"   POST /api/sessions      - 创建会话")
    print(f"   POST /api/users         - 创建用户")
    print(f"   POST /api/search        - 搜索记忆")
    print(f"   GET  /api/health        - 健康检查")

    # 自动选择服务器
    if server == "auto":
        if debug:
            server = "flask"
        else:
            try:
                import waitress  # noqa: F401
                server = "waitress"
            except ImportError:
                server = "flask"

    if server == "waitress":
        import waitress
        print(f"   Server: waitress ({workers} threads)")
        waitress.serve(app, host=host, port=port, threads=workers)
    else:
        if not debug:
            print("   WARNING: Flask 开发服务器不适合生产环境，请安装 waitress: pip install waitress")
        app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server()
