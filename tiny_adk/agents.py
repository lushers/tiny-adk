"""Agent - AI 代理的核心定义"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional

from pydantic import BaseModel, Field, PrivateAttr

from .flows import SimpleFlow
from .models import BaseLlm
from .tools import BaseTool, Tool

if TYPE_CHECKING:
    from .events import Event
    from .flows import BaseFlow
    from .session import InvocationContext

logger = logging.getLogger(__name__)


class Agent(BaseModel):
    """
    Agent 配置容器（使用 Pydantic）
    
    核心设计理念:
    - Agent 是配置，不包含状态
    - Agent 定义了"是谁"、"会什么"、"怎么做"
    - 实际执行由 Runner 负责
    - Flow 在初始化时创建（通过 model_post_init）
    
    这种分离使得:
    1. Agent 可以被复用
    2. 同一个 Agent 可以并发执行多个会话
    3. Agent 配置可以被序列化、版本化、A/B 测试
    """
    
    model_config = {"arbitrary_types_allowed": True}
    
    # 基本配置
    name: str
    instruction: str = ""
    description: str = ""
    
    # 模型配置（支持两种方式）
    model: str | Any = "gpt-4"  # 可以是模型名称字符串或 BaseLlm 实例
    temperature: float = 0.7
    max_tokens: int = 2000
    max_iterations: int = 10  # 默认允许 10 次迭代（工具调用循环）
    
    # 工具
    tools: list[BaseTool | Tool] = Field(default_factory=list)
    
    # 子代理（Multi-Agent 支持）
    sub_agents: list['Agent'] = Field(default_factory=list)
    
    # 私有字段：Flow 实例（在 model_post_init 中创建）
    _flow: Optional[Any] = PrivateAttr(default=None)
    _llm: Optional[Any] = PrivateAttr(default=None)
    
    def model_post_init(self, __context: Any) -> None:
        """Pydantic 初始化完成后的钩子"""
        super().model_post_init(__context)
        
        # 创建 Flow（只创建一次）
        # 根据是否有子代理选择不同的 Flow
        # 目前只有 SimpleFlow，后续可以添加 AutoFlow 支持 Multi-Agent
        if self.sub_agents:
            # TODO: 使用 AutoFlow 支持 Agent 转移
            self._flow = SimpleFlow(max_iterations=self.max_iterations)
            logger.debug(f"[Agent {self.name}] Created SimpleFlow (with sub_agents)")
        else:
            self._flow = SimpleFlow(max_iterations=self.max_iterations)
            logger.debug(f"[Agent {self.name}] Created SimpleFlow")
        
        # 如果 model 是 BaseLlm 实例，保存引用
        if isinstance(self.model, BaseLlm):
            self._llm = self.model
    
    @property
    def flow(self) -> 'BaseFlow':
        """返回 Flow 实例"""
        if self._flow is None:
            # Fallback：如果没有初始化（不应该发生）
            self._flow = SimpleFlow(max_iterations=self.max_iterations)
        return self._flow
    
    @property
    def llm(self) -> Optional[BaseLlm]:
        """返回 LLM 实例（如果有）"""
        return self._llm
    
    def get_model_name(self) -> str:
        """获取模型名称"""
        if isinstance(self.model, str):
            return self.model
        elif hasattr(self.model, 'model'):
            return self.model.model
        return "unknown"
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            'name': self.name,
            'instruction': self.instruction,
            'model': self.get_model_name(),
            'description': self.description,
            'tools': [t.to_function_declaration() for t in self.tools],
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'max_iterations': self.max_iterations,
        }
    
    def get_system_prompt(self) -> str:
        """
        构建系统提示词
        
        将 agent 的身份和能力转换为 LLM 可理解的提示词
        """
        prompt = f"You are {self.name}.\n\n"
        prompt += f"{self.instruction}\n\n"
        
        if self.tools:
            prompt += "You have access to the following tools:\n"
            for tool in self.tools:
                prompt += f"- {tool.name}: {tool.description}\n"
        
        return prompt
    
    async def run_async(
        self, 
        context: 'InvocationContext'
    ) -> AsyncGenerator['Event', None]:
        """
        Agent 执行入口：委托给 Flow
        
        Args:
            context: InvocationContext（运行时状态）
            
        Yields:
            Event 对象
        """
        logger.info(f"[Agent {self.name}] Starting execution")
        
        # 获取 LLM
        llm = self._llm or context.run_config.get('llm')
        if llm is None:
            raise ValueError(f"Agent {self.name} has no LLM configured")
        
        # 委托给 Flow
        async for event in self.flow.run_stream_async(self, context.session, llm):
            yield event
