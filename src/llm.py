"""
LLM 统一接口 - 封装多家大模型 API
支持: DeepSeek, Kimi, OpenAI, MiniMax, Grok
能力: 文本、视觉、语音、生图
"""

import json
import base64
import requests
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# ─── 数据结构 ───

class LLMType(Enum):
    """LLM 能力类型"""
    TEXT = "text"           # 纯文本对话
    VISION = "vision"       # 图像理解
    TTS = "tts"            # 文字转语音
    STT = "stt"            # 语音转文字
    IMAGE_GEN = "image_gen" # 图像生成
    VIDEO = "video"         # 视频理解


@dataclass
class Message:
    """统一消息格式"""
    role: str  # system, user, assistant, tool
    content: Union[str, List[Dict]]  # str 或 multimodal content list
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None

    @staticmethod
    def text(role: str, content: str) -> "Message":
        return Message(role=role, content=content)

    @staticmethod
    def image(role: str, text: str, image_url: str = None, image_path: str = None) -> "Message":
        """创建包含图片的消息"""
        content = [{"type": "text", "text": text}]
        if image_url:
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        elif image_path:
            b64 = _encode_image(image_path)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        return Message(role=role, content=content)

    @staticmethod
    def audio(role: str, text: str, audio_data: bytes = None, audio_url: str = None) -> "Message":
        """创建包含音频的消息（部分模型支持）"""
        content = [{"type": "text", "text": text}]
        if audio_data:
            b64 = base64.b64encode(audio_data).decode()
            content.append({"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}})
        elif audio_url:
            content.append({"type": "audio_url", "audio_url": {"url": audio_url}})
        return Message(role=role, content=content)


@dataclass
class ToolCall:
    """工具调用"""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    """统一响应格式"""
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    usage: Optional[Dict[str, int]] = None
    raw: Optional[Dict] = None


@dataclass
class TTSResponse:
    """TTS 响应"""
    audio_data: bytes
    format: str = "mp3"
    duration: Optional[float] = None


@dataclass
class ImageGenResponse:
    """图像生成响应"""
    images: List[bytes]  # 图片数据列表
    format: str = "png"
    revised_prompt: Optional[str] = None


# ─── 辅助函数 ───

def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _msg_to_dict(msg: Message) -> Dict:
    d = {"role": msg.role, "content": msg.content}
    if msg.tool_calls:
        d["tool_calls"] = msg.tool_calls
    if msg.tool_call_id:
        d["tool_call_id"] = msg.tool_call_id
    return d


# ─── LLM 基类 ───

class BaseLLM(ABC):
    """LLM 基类"""

    def __init__(self, api_key: str, model: str, base_url: str):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    @abstractmethod
    def chat(self, messages: List[Message], tools: Optional[List[Dict]] = None,
             temperature: float = 0.7, max_tokens: int = 4096) -> LLMResponse:
        """文本/视觉对话"""
        pass

    def supports(self, capability: LLMType) -> bool:
        """检查是否支持某能力"""
        return capability in self.capabilities

    @property
    def capabilities(self) -> List[LLMType]:
        """子类覆盖，声明支持的能力"""
        return [LLMType.TEXT]


# ─── OpenAI 兼容实现 ───

class OpenAICompatibleLLM(BaseLLM):
    """OpenAI 兼容接口 - 文本 + 视觉"""

    def __init__(self, api_key: str, model: str, base_url: str,
                 vision_model: str = None, capabilities: List[LLMType] = None):
        super().__init__(api_key, model, base_url)
        self.vision_model = vision_model or model
        self._capabilities = capabilities or [LLMType.TEXT]

    @property
    def capabilities(self) -> List[LLMType]:
        return self._capabilities

    def chat(self, messages: List[Message], tools: Optional[List[Dict]] = None,
             temperature: float = 0.7, max_tokens: int = 4096) -> LLMResponse:

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 检测是否有图片，切换到视觉模型
        has_image = any(
            isinstance(m.content, list) and
            any(c.get("type") == "image_url" for c in m.content if isinstance(c, dict))
            for m in messages
        )
        model = self.vision_model if has_image else self.model

        payload = {
            "model": model,
            "messages": [_msg_to_dict(m) for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            payload["tools"] = tools

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            return self._parse_response(data)
        except requests.exceptions.HTTPError as e:
            # 获取详细的错误信息
            error_detail = ""
            try:
                error_data = response.json()
                error_detail = json.dumps(error_data, ensure_ascii=False)
            except:
                error_detail = response.text if hasattr(response, 'text') else str(e)

            # 构建详细的错误信息
            error_msg = f"HTTP {response.status_code}: {response.reason}\n错误详情: {error_detail}"

            # 返回包含错误信息的响应
            return LLMResponse(
                content="",
                tool_calls=None,
                usage=None,
                raw={
                    "error": True,
                    "status_code": response.status_code,
                    "reason": response.reason,
                    "error_detail": error_detail,
                    "error_message": error_msg
                }
            )

    def _parse_response(self, data: Dict) -> LLMResponse:
        # 检查是否有错误
        if "error" in data:
            error_detail = json.dumps(data["error"], ensure_ascii=False)
            error_msg = f"API 错误: {error_detail}"

            return LLMResponse(
                content="",
                tool_calls=None,
                usage=None,
                raw={
                    "error": True,
                    "error_detail": error_detail,
                    "error_message": error_msg
                }
            )

        # 检查是否有 choices
        if "choices" not in data or not data["choices"]:
            error_msg = "API 返回了空的 choices"
            return LLMResponse(
                content="",
                tool_calls=None,
                usage=None,
                raw={
                    "error": True,
                    "error_message": error_msg,
                    "response_data": data
                }
            )

        choice = data["choices"][0]["message"]
        content = choice.get("content", "") or ""

        tool_calls = None
        if "tool_calls" in choice and choice["tool_calls"]:
            tool_calls = []
            for tc in choice["tool_calls"]:
                func = tc.get("function", {})
                args_raw = func.get("arguments", "{}")

                # 统一解析 arguments（可能是 JSON 字符串或已解析的 dict）
                if isinstance(args_raw, str):
                    try:
                        args = json.loads(args_raw)
                    except json.JSONDecodeError:
                        # 有些模型返回不标准的 JSON，尝试修复
                        args_raw = args_raw.strip().strip("'\"")
                        try:
                            args = json.loads(args_raw)
                        except:
                            args = {"_raw": args_raw}
                else:
                    args = args_raw

                # 统一获取 tool_call_id（有些模型可能叫 call_id 或没有）
                tc_id = tc.get("id") or tc.get("call_id") or f"call_{len(tool_calls)}"

                tool_calls.append(ToolCall(
                    id=tc_id,
                    name=func.get("name", ""),
                    arguments=args
                ))

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=data.get("usage"),
            raw=data
        )


# ─── TTS 专用接口 ───

class OpenAITTS:
    """OpenAI TTS 接口"""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1",
                 model: str = "tts-1", voice: str = "alloy"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.voice = voice

    def synthesize(self, text: str, voice: str = None, model: str = None,
                   response_format: str = "mp3", speed: float = 1.0) -> TTSResponse:
        """文字转语音"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model or self.model,
            "input": text,
            "voice": voice or self.voice,
            "response_format": response_format,
            "speed": speed
        }

        response = requests.post(
            f"{self.base_url}/audio/speech",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        return TTSResponse(
            audio_data=response.content,
            format=response_format
        )


# ─── STT 专用接口 ───

class OpenAISTT:
    """OpenAI STT (Whisper) 接口"""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1",
                 model: str = "whisper-1"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def transcribe(self, audio_path: str, language: str = None,
                   prompt: str = None) -> str:
        """语音转文字"""
        headers = {"Authorization": f"Bearer {self.api_key}"}

        with open(audio_path, "rb") as f:
            files = {"file": (audio_path, f, "audio/wav")}
            data = {"model": self.model}
            if language:
                data["language"] = language
            if prompt:
                data["prompt"] = prompt

            response = requests.post(
                f"{self.base_url}/audio/transcriptions",
                headers=headers,
                files=files,
                data=data,
                timeout=60
            )
            response.raise_for_status()
            return response.json().get("text", "")

    def translate(self, audio_path: str) -> str:
        """语音翻译为英文"""
        headers = {"Authorization": f"Bearer {self.api_key}"}

        with open(audio_path, "rb") as f:
            files = {"file": (audio_path, f, "audio/wav")}
            data = {"model": self.model}

            response = requests.post(
                f"{self.base_url}/audio/translations",
                headers=headers,
                files=files,
                data=data,
                timeout=60
            )
            response.raise_for_status()
            return response.json().get("text", "")


# ─── 图像生成专用接口 ───

class OpenAIImageGen:
    """OpenAI DALL-E 图像生成"""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1",
                 model: str = "dall-e-3"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def generate(self, prompt: str, size: str = "1024x1024",
                 quality: str = "standard", n: int = 1) -> ImageGenResponse:
        """生成图像"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "n": n,
            "response_format": "b64_json"
        }

        response = requests.post(
            f"{self.base_url}/images/generations",
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        data = response.json()

        images = []
        revised_prompt = data.get("data", [{}])[0].get("revised_prompt")
        for item in data.get("data", []):
            if "b64_json" in item:
                images.append(base64.b64decode(item["b64_json"]))

        return ImageGenResponse(
            images=images,
            format="png",
            revised_prompt=revised_prompt
        )


# ─── 工厂 ───

class LLMFactory:
    """LLM 工厂"""

    PROVIDERS = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-4o",
            "vision_model": "gpt-4o",
            "tts_model": "tts-1",
            "stt_model": "whisper-1",
            "image_model": "dall-e-3",
            "capabilities": [LLMType.TEXT, LLMType.VISION],
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "default_model": "deepseek-chat",
            "capabilities": [LLMType.TEXT],
        },
        "kimi": {
            "base_url": "https://api.moonshot.cn/v1",
            "default_model": "moonshot-v1-8k",
            "vision_model": "moonshot-v1-8k-vision-preview",
            "capabilities": [LLMType.TEXT, LLMType.VISION],
        },
        "minimax": {
            "base_url": "https://api.minimaxi.com/v1",
            "default_model": "MiniMax-M2.5",
            "capabilities": [LLMType.TEXT],
        },
        "grok": {
            "base_url": "https://api.x.ai/v1",
            "default_model": "grok-2-latest",
            "capabilities": [LLMType.TEXT],
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "default_model": "openai/gpt-4o",
            "capabilities": [LLMType.TEXT, LLMType.VISION],
        },
    }

    @classmethod
    def create(cls, provider: str, api_key: str, model: str = None) -> OpenAICompatibleLLM:
        if provider not in cls.PROVIDERS:
            raise ValueError(f"不支持的 provider: {provider}，可选: {list(cls.PROVIDERS.keys())}")

        config = cls.PROVIDERS[provider]
        return OpenAICompatibleLLM(
            api_key=api_key,
            model=model or config["default_model"],
            base_url=config["base_url"],
            vision_model=config.get("vision_model"),
            capabilities=config.get("capabilities", [LLMType.TEXT])
        )

    @classmethod
    def create_tts(cls, provider: str, api_key: str, voice: str = "alloy") -> Optional[OpenAITTS]:
        config = cls.PROVIDERS.get(provider, {})
        if "tts_model" not in config:
            return None
        return OpenAITTS(
            api_key=api_key,
            base_url=config["base_url"],
            model=config["tts_model"],
            voice=voice
        )

    @classmethod
    def create_stt(cls, provider: str, api_key: str) -> Optional[OpenAISTT]:
        config = cls.PROVIDERS.get(provider, {})
        if "stt_model" not in config:
            return None
        return OpenAISTT(
            api_key=api_key,
            base_url=config["base_url"],
            model=config["stt_model"]
        )

    @classmethod
    def create_image_gen(cls, provider: str, api_key: str) -> Optional[OpenAIImageGen]:
        config = cls.PROVIDERS.get(provider, {})
        if "image_model" not in config:
            return None
        return OpenAIImageGen(
            api_key=api_key,
            base_url=config["base_url"],
            model=config["image_model"]
        )

    @classmethod
    def list_providers(cls) -> List[str]:
        return list(cls.PROVIDERS.keys())

    @classmethod
    def get_capabilities(cls, provider: str) -> List[LLMType]:
        return cls.PROVIDERS.get(provider, {}).get("capabilities", [])


class ToolBuilder:
    """统一工具定义构建器"""

    @staticmethod
    def define(name: str, description: str, parameters: Dict[str, Any],
               required: List[str] = None) -> Dict:
        """构建标准工具定义"""
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required or list(parameters.keys())
                }
            }
        }

    @staticmethod
    def param(type_: str, description: str, **kwargs) -> Dict:
        """构建参数定义"""
        p = {"type": type_, "description": description}
        if type_ == "string" and "enum" in kwargs:
            p["enum"] = kwargs["enum"]
        if type_ == "number" or type_ == "integer":
            if "minimum" in kwargs:
                p["minimum"] = kwargs["minimum"]
            if "maximum" in kwargs:
                p["maximum"] = kwargs["maximum"]
        return p

    @staticmethod
    def from_function(func, name: str = None, description: str = None) -> Dict:
        """从 Python 函数自动构建工具定义（需 type hints）"""
        import inspect

        sig = inspect.signature(func)
        params = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_info = {"type": "string", "description": f"参数 {param_name}"}

            # 从 type hint 推断类型
            if param.annotation != inspect.Parameter.empty:
                type_map = {
                    str: "string",
                    int: "integer",
                    float: "number",
                    bool: "boolean",
                    list: "array",
                    dict: "object",
                }
                param_info["type"] = type_map.get(param.annotation, "string")

            params[param_name] = param_info

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return ToolBuilder.define(
            name=name or func.__name__,
            description=description or func.__doc__ or f"工具 {func.__name__}",
            parameters=params,
            required=required
        )
