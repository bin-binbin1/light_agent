"""
Router 模块 - 多 Agent 路由 + 能力路由
支持按名称、意图、能力类型分发到不同 Agent/LLM
"""

from typing import Dict, Optional, Callable, List
from .agent import Agent, AgentConfig
from .llm import BaseLLM, LLMFactory, LLMType
from .agent_logging import Logger, default_logger


class ModelPool:
    """模型池 - 按能力管理多个 LLM 实例"""

    def __init__(self):
        self._models: Dict[str, BaseLLM] = {}       # name -> LLM
        self._capability_map: Dict[LLMType, List[str]] = {}  # capability -> [model names]

    def register(self, name: str, llm: BaseLLM):
        """注册模型"""
        self._models[name] = llm
        for cap in llm.capabilities:
            if cap not in self._capability_map:
                self._capability_map[cap] = []
            if name not in self._capability_map[cap]:
                self._capability_map[cap].append(name)

    def get(self, name: str) -> Optional[BaseLLM]:
        return self._models.get(name)

    def get_by_capability(self, capability: LLMType, preferred: str = None) -> Optional[BaseLLM]:
        """按能力获取模型，可指定首选"""
        if preferred and preferred in self._models:
            llm = self._models[preferred]
            if capability in llm.capabilities:
                return llm

        names = self._capability_map.get(capability, [])
        if names:
            return self._models[names[0]]
        return None

    def list_models(self) -> List[str]:
        return list(self._models.keys())

    def list_capabilities(self) -> Dict[str, List[str]]:
        result = {}
        for name, llm in self._models.items():
            result[name] = [c.value for c in llm.capabilities]
        return result


class Router:
    """多 Agent 路由器"""

    def __init__(self, logger: Optional[Logger] = None):
        self._agents: Dict[str, Agent] = {}
        self._rules: list = []
        self._default_agent: Optional[str] = None
        self.logger = logger or default_logger

    def register(self, name: str, agent: Agent, default: bool = False):
        """注册 Agent"""
        self._agents[name] = agent
        if default or self._default_agent is None:
            self._default_agent = name
        self.logger.system(f"Agent '{name}' 已注册{' (default)' if default else ''}")

    def unregister(self, name: str):
        if name in self._agents:
            del self._agents[name]
            if self._default_agent == name:
                self._default_agent = next(iter(self._agents), None)

    def add_rule(self, rule: Callable[[str], Optional[str]]):
        """添加路由规则"""
        self._rules.append(rule)

    def route(self, user_input: str) -> Optional[str]:
        """路由到 Agent 名称"""
        for rule in self._rules:
            result = rule(user_input)
            if result and result in self._agents:
                return result
        return self._default_agent

    def chat(self, user_input: str, agent_name: Optional[str] = None) -> str:
        """路由并对话"""
        if agent_name is None:
            agent_name = self.route(user_input)

        if agent_name is None or agent_name not in self._agents:
            return f"未找到可用的 Agent。已注册: {list(self._agents.keys())}"

        agent = self._agents[agent_name]
        self.logger.system(f"路由到 Agent '{agent_name}'")
        return agent.chat(user_input)

    def get_agent(self, name: str) -> Optional[Agent]:
        return self._agents.get(name)

    def list_agents(self) -> List[str]:
        return list(self._agents.keys())


class CapabilityRouter:
    """能力路由器 - 根据请求自动选择具备对应能力的模型"""

    def __init__(self, model_pool: ModelPool, logger: Optional[Logger] = None):
        self.model_pool = model_pool
        self.logger = logger or default_logger
        self._rules: List[Callable[[str], Optional[LLMType]]] = []

    def add_rule(self, rule: Callable[[str], Optional[LLMType]]):
        """添加能力判断规则"""
        self._rules.append(rule)

    def detect_capability(self, user_input: str, has_image: bool = False,
                          has_audio: bool = False) -> LLMType:
        """检测需要的能力"""
        if has_image:
            return LLMType.VISION
        if has_audio:
            return LLMType.STT

        for rule in self._rules:
            result = rule(user_input)
            if result:
                return result

        return LLMType.TEXT

    def get_llm(self, capability: LLMType, preferred: str = None) -> Optional[BaseLLM]:
        """获取具备指定能力的 LLM"""
        llm = self.model_pool.get_by_capability(capability, preferred)
        if llm:
            self.logger.system(f"能力路由: {capability.value} → {llm.model}")
        else:
            self.logger.error(f"没有支持 {capability.value} 的模型")
        return llm

    def route(self, user_input: str, has_image: bool = False,
              has_audio: bool = False, preferred_model: str = None) -> Optional[BaseLLM]:
        """完整路由：输入 → 能力检测 → 模型选择"""
        capability = self.detect_capability(user_input, has_image, has_audio)
        return self.get_llm(capability, preferred_model)


# ─── 预置规则 ───

def keyword_rule(keywords: Dict[str, str]):
    """关键词路由规则"""
    def rule(user_input: str) -> Optional[str]:
        for keyword, agent_name in keywords.items():
            if keyword in user_input:
                return agent_name
        return None
    return rule


def intent_rule(llm: BaseLLM, intent_map: Dict[str, str]):
    """意图路由规则（用 LLM 判断意图）"""
    def rule(user_input: str) -> Optional[str]:
        intents = ", ".join(intent_map.keys())
        prompt = f"用户输入: {user_input}\n\n请判断用户意图属于以下哪个类别: {intents}\n只回复类别名称。"

        response = llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=50
        )

        intent = response.content.strip().lower()
        for key, agent_name in intent_map.items():
            if key.lower() in intent:
                return agent_name
        return None
    return rule


def image_gen_rule():
    """图像生成能力判断规则"""
    keywords = ["画", "生成图片", "图片", "画一个", "画一张", "生成一个图", "draw", "generate image", "生图"]
    def rule(user_input: str) -> Optional[LLMType]:
        for kw in keywords:
            if kw in user_input.lower():
                return LLMType.IMAGE_GEN
        return None
    return rule


def tts_rule():
    """语音合成能力判断规则"""
    keywords = ["读出来", "语音", "念", "朗读", "tts", "说一下", "语音合成"]
    def rule(user_input: str) -> Optional[LLMType]:
        for kw in keywords:
            if kw in user_input.lower():
                return LLMType.TTS
        return None
    return rule
