"""
Proxy 模块 - OpenAI 兼容的 LLM 中转接口
让没有 API Key 的客户端通过服务端调用 LLM

接口: POST /v1/chat/completions
格式: 完全兼容 OpenAI API，客户端可直接用 openai SDK 对接
"""

import json
import os
import time
import requests as http_requests
from typing import Optional, TYPE_CHECKING

try:
    from flask import Blueprint, request, jsonify, Response, stream_with_context
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

if TYPE_CHECKING:
    from flask import Blueprint

from src.config import Config
from src.llm import LLMFactory


def create_proxy_blueprint(config: Config):
    """创建中转代理蓝图"""

    bp = Blueprint("proxy", __name__)

    # 从配置读取合法 token 列表和模型映射
    proxy_tokens = config.get("proxy_tokens", [])
    proxy_models = config.get("proxy_models", {})

    def _get_token_from_header() -> Optional[str]:
        """从 Authorization header 提取 Bearer token"""
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:]
        return None

    def _verify_token(token: str) -> bool:
        """校验用户 token"""
        if not proxy_tokens:
            return False
        return token in proxy_tokens

    def _resolve_model(model_name: str) -> dict:
        """解析模型名 → 实际的 provider + api_key + model

        优先查 proxy_models 映射，找不到则用默认 provider。
        返回: {"provider": ..., "api_key": ..., "model": ...}
        """
        if model_name in proxy_models:
            mapping = proxy_models[model_name]
            provider = mapping.get("provider", config.provider)
            # api_key 优先从环境变量取，其次用配置文件的
            api_key_env = mapping.get("api_key_env", "")
            api_key = os.getenv(api_key_env, "") if api_key_env else ""
            if not api_key:
                api_key = mapping.get("api_key", config.api_key)
            model = mapping.get("model", model_name)
            return {"provider": provider, "api_key": api_key, "model": model}

        # 无映射，用默认 provider
        return {
            "provider": config.provider,
            "api_key": config.api_key,
            "model": model_name or config.model,
        }

    def _build_llm_payload(data: dict, resolved: dict) -> tuple:
        """构建发给 LLM 的请求 payload，返回 (base_url, headers, payload)"""
        provider_info = LLMFactory.PROVIDERS.get(resolved["provider"])
        if not provider_info:
            return None, None, None

        base_url = provider_info["base_url"]
        headers = {
            "Authorization": f"Bearer {resolved['api_key']}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": resolved["model"],
            "messages": data.get("messages", []),
        }

        # 透传可选参数
        for key in ("temperature", "max_tokens", "top_p", "frequency_penalty",
                     "presence_penalty", "stop", "tools", "tool_choice", "stream"):
            if key in data:
                payload[key] = data[key]

        return base_url, headers, payload

    # ─── 中转路由 ───

    @bp.route("/v1/chat/completions", methods=["POST"])
    def chat_completions():
        """OpenAI 兼容的 chat completions 中转"""

        # 鉴权
        token = _get_token_from_header()
        if not token:
            return jsonify({"error": {"message": "Missing Authorization header", "type": "auth_error"}}), 401

        if not _verify_token(token):
            return jsonify({"error": {"message": "Invalid token", "type": "auth_error"}}), 401

        if not proxy_tokens:
            return jsonify({"error": {"message": "Proxy not configured", "type": "server_error"}}), 403

        data = request.json or {}
        model_name = data.get("model", "")
        is_stream = data.get("stream", False)

        # 解析模型
        resolved = _resolve_model(model_name)
        if not resolved["api_key"]:
            return jsonify({"error": {"message": "Server API key not configured for this model", "type": "server_error"}}), 500

        base_url, headers, payload = _build_llm_payload(data, resolved)
        if not base_url:
            return jsonify({"error": {"message": f"Unknown provider: {resolved['provider']}", "type": "invalid_request_error"}}), 400

        url = f"{base_url}/chat/completions"

        try:
            if is_stream:
                return _stream_proxy(url, headers, payload)
            else:
                return _sync_proxy(url, headers, payload)
        except http_requests.Timeout:
            return jsonify({"error": {"message": "LLM request timeout", "type": "timeout_error"}}), 504
        except Exception as e:
            return jsonify({"error": {"message": str(e), "type": "server_error"}}), 500

    @bp.route("/v1/models", methods=["GET"])
    def list_models():
        """返回可用模型列表"""
        token = _get_token_from_header()
        if not token or not _verify_token(token):
            return jsonify({"error": {"message": "Unauthorized", "type": "auth_error"}}), 401

        models = []
        for model_name in proxy_models:
            models.append({
                "id": model_name,
                "object": "model",
                "owned_by": proxy_models[model_name].get("provider", config.provider),
            })

        # 如果没有映射配置，返回默认模型
        if not models and config.model:
            models.append({
                "id": config.model,
                "object": "model",
                "owned_by": config.provider,
            })

        return jsonify({"object": "list", "data": models})

    def _sync_proxy(url: str, headers: dict, payload: dict) -> Response:
        """同步中转"""
        resp = http_requests.post(url, headers=headers, json=payload, timeout=120)
        # 透传 LLM 响应
        return Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get("Content-Type", "application/json"),
        )

    def _stream_proxy(url: str, headers: dict, payload: dict) -> Response:
        """流式中转（SSE 透传）"""
        payload["stream"] = True

        def generate():
            with http_requests.post(url, headers=headers, json=payload,
                                    stream=True, timeout=120) as resp:
                for line in resp.iter_lines():
                    if line:
                        yield line.decode("utf-8") + "\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
        )

    return bp
