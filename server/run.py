"""
Light Agent FastAPI Server - 启动入口

用法:
    python server/run.py
    python server/run.py --host 0.0.0.0 --port 8000
    python server/run.py --reload                   # 开发模式（代码热重载）
    python server/run.py --workers 4                # 多 worker（注意：agent_pool 每 worker 独立）
"""

import argparse
import os
import sys

# 确保项目根目录在 sys.path（这样 `from src.xxx` 可用）
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)


def main():
    parser = argparse.ArgumentParser(description="Light Agent FastAPI Server")
    parser.add_argument("--host", default="0.0.0.0", help="绑定地址（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=8000, help="端口（默认 8000）")
    parser.add_argument("--reload", action="store_true", help="开发模式：代码变更自动重载")
    parser.add_argument("--workers", type=int, default=1,
                        help="worker 数（默认 1；>1 时每个 worker 独立维护 agent_pool，"
                             "多用户访问请用反向代理做 sticky session）")
    parser.add_argument("--log-level", default="info",
                        choices=["critical", "error", "warning", "info", "debug", "trace"])
    args = parser.parse_args()

    import uvicorn

    # reload 和 workers 互斥（uvicorn 限制）
    workers = 1 if args.reload else args.workers

    uvicorn.run(
        "server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=workers,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
