"""
WSGI 入口 - 用于 gunicorn 部署

用法:
    gunicorn -w 4 -k gevent wsgi:app
"""

from app import create_app

app = create_app("config/config.json")
