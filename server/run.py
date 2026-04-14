"""
启动入口 - 运行 Light Agent Server

用法:
    python server/run.py
    python server/run.py --port 8080 --debug
"""

import sys
import os
import argparse

# 确保项目根目录在 sys.path 中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from server.app import run_server


def main():
    parser = argparse.ArgumentParser(description="Light Agent Server")
    parser.add_argument("--host", default="0.0.0.0", help="绑定地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="端口 (默认: 8000)")
    parser.add_argument("--config", default="config/config.json", help="配置文件路径")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--workers", type=int, default=4, help="工作线程数 (默认: 4)")
    parser.add_argument("--server", default="auto", choices=["auto", "waitress", "flask"],
                        help="服务器类型 (默认: auto)")
    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        config_path=args.config,
        debug=args.debug,
        workers=args.workers,
        server=args.server,
    )


if __name__ == "__main__":
    main()
