"""LlmAgent - LLM 驱动的 Agent"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Iterator, Optional, Union

from pydantic import Field, PrivateAttr
from typing_extensions import override

from .base_agent import BaseAgent
from ..flows import SimpleFlow
from ..models import BaseLlm
from ..tools import BaseTool, Tool

if TYPE_CHECKING:
    from ..events import Event
    from ..flows import BaseFlow
    from ..session import Session

logger = logging.getLogger(__name__)


class LlmAgent(BaseAgent):
    """
    LLM 驱动的 Agent
    
    职责：
    - 管理 LLM 相关配置（model, temperature, tools 等）
    - 委托给 Flow 执行实际的 LLM 交互
    - 处理 Agent 跳转逻辑
    """
    
    # === LLM 配置 ===
    instruction: str = ''
    """Agent 的指令/系统提示词"""
    
    model: Union[str, BaseLlm] = 'gpt-4'
    """模型名称或 LLM 实例"""
    
    temperature: float = 0.7
    max_tokens: int = 2000
    max_iterations: int = 10
    
    # === 工具 ===
    tools: list[Union[BaseTool, Tool]] = Field(default_factory=list)
    """可用工具列表"""
    
    # === Agent 跳转控制 ===
    disallow_transfer_to_parent: bool = False
    """禁止跳转回父 Agent"""
    
    disallow_transfer_to_peers: bool = False
    """禁止跳转到同级 Agent"""
    
    # === 私有字段 ===
    _flow: Optional[Any] = PrivateAttr(default=None)
    _llm: Optional[BaseLlm] = PrivateAttr(default=None)
    
    @override
    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._flow = SimpleFlow(max_iterations=self.max_iterations)
        # 如果 model 是 BaseLlm 实例，保存引用
        if isinstance(self.model, BaseLlm):
            self._llm = self.model
        logger.debug(f"[LlmAgent {self.name}] Created with model={self.get_model_name()}")
    
    # === 属性 ===
    
    @property
    def flow(self) -> 'BaseFlow':
        """返回 Flow 实例"""
        if self._flow is None:
            self._flow = SimpleFlow(max_iterations=self.max_iterations)
        return self._flow
    
    @property
    def llm(self) -> Optional[BaseLlm]:
        """返回 LLM 实例（如果有）"""
        return self._llm
    
    @property
    def canonical_model(self) -> BaseLlm:
        """获取解析后的 LLM 实例"""
        if isinstance(self.model, BaseLlm):
            return self.model
        # 如果是字符串，需要从祖先继承或创建
        ancestor = self.parent_agent
        while ancestor is not None:
            if isinstance(ancestor, LlmAgent) and isinstance(ancestor.model, BaseLlm):
                return ancestor.model
            ancestor = ancestor.parent_agent
        raise ValueError(f"No LLM configured for agent '{self.name}'")
    
    def get_model_name(self) -> str:
        """获取模型名称"""
        if isinstance(self.model, str):
            return self.model
        elif hasattr(self.model, 'model'):
            return self.model.model
        return "unknown"
    
    # === 可跳转的 Agent ===
    
    def get_transferable_agents(self) -> list['BaseAgent']:
        """
        获取可跳转到的 Agent 列表
        
        注意：只有当父 Agent 是 LlmAgent 时，才允许跳转到父 Agent 或同级 Agent。
        如果父 Agent 是编排器（SequentialAgent、LoopAgent），执行顺序由编排器控制，
        不允许子 Agent 自行跳转。
        """
        agents = []
        
        # 子 Agent - 总是可以跳转
        agents.extend(self.sub_agents)
        
        # 检查父 Agent 是否是 LlmAgent（不是编排器）
        parent_is_llm_agent = isinstance(self.parent_agent, LlmAgent)
        
        # 父 Agent - 只有父 Agent 是 LlmAgent 时才允许
        if not self.disallow_transfer_to_parent and self.parent_agent and parent_is_llm_agent:
            agents.append(self.parent_agent)
        
        # 同级 Agent - 只有父 Agent 是 LlmAgent 时才允许
        if not self.disallow_transfer_to_peers and self.parent_agent and parent_is_llm_agent:
            for sibling in self.parent_agent.sub_agents:
                if sibling.name != self.name:
                    agents.append(sibling)
        
        return agents
    
    # === 系统提示词 ===
    
    def get_system_prompt(self) -> str:
        """构建系统提示词"""
        prompt = f"You are {self.name}.\n\n{self.instruction}\n\n"
        
        if self.tools:
            prompt += "You have access to the following tools:\n"
            for tool in self.tools:
                prompt += f"- {tool.name}: {tool.description}\n"
            prompt += "\n"
        
        transferable = self.get_transferable_agents()
        if transferable:
            prompt += "You can transfer control to the following agents:\n"
            for agent in transferable:
                desc = agent.description or f"Agent {agent.name}"
                prompt += f"- {agent.name}: {desc}\n"
            prompt += "\nUse the 'transfer_to_agent' tool to hand over.\n"
        
        return prompt
    
    # === 执行 ===
    
    @override
    async def _run_async_impl(
        self,
        session: 'Session',
        llm: Optional[BaseLlm] = None,
    ) -> AsyncGenerator['Event', None]:
        """LLM Agent 的核心执行逻辑"""
        from ..events import EventType
        
        # 获取 LLM
        if llm is None:
            if isinstance(self.model, BaseLlm):
                llm = self.model
            else:
                raise ValueError(f"Agent '{self.name}' has no LLM configured")
        
        # 检查待处理的跳转
        transfer_target = self._check_pending_transfer(session)
        if transfer_target:
            logger.info(f"[{self.name}] Resuming transfer to {transfer_target.name}")
            async for event in transfer_target.run_async(session, llm=llm):
                yield event
            return
        
        # 委托给 Flow 执行
        async for event in self.flow.run_async(self, session, llm, stream=True):
            yield event
            
            # 检查跳转
            if self._should_transfer(event):
                target_name = event.metadata.get('transfer_to_agent')
                target_agent = self.root_agent.find_agent(target_name)
                if target_agent:
                    logger.info(f"[{self.name}] Transferring to {target_name}")
                    async for sub_event in target_agent.run_async(session, llm=llm):
                        yield sub_event
                    return
    
    def run(
        self,
        session: 'Session',
        llm: Optional[BaseLlm] = None,
    ) -> Iterator['Event']:
        """同步执行"""
        if llm is None and isinstance(self.model, BaseLlm):
            llm = self.model
        if llm is None:
            raise ValueError(f"Agent '{self.name}' has no LLM configured")
        
        for event in self.flow.run(self, session, llm, stream=True):
            yield event
    
    def _check_pending_transfer(self, session: 'Session') -> Optional['BaseAgent']:
        """检查待处理的跳转"""
        if not session.events:
            return None
        
        from ..events import EventType
        for event in reversed(session.events):
            if event.event_type == EventType.AGENT_TRANSFER:
                target_name = event.content.get('target_agent')
                if target_name and target_name != self.name:
                    return self.root_agent.find_agent(target_name)
                break
        return None
    
    def _should_transfer(self, event: 'Event') -> bool:
        """检查事件是否触发跳转"""
        from ..events import EventType
        return (
            event.event_type == EventType.AGENT_TRANSFER
            and event.metadata.get('transfer_to_agent')
        )
    
    # === 序列化 ===
    
    @override
    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            'instruction': self.instruction,
            'model': self.get_model_name(),
            'tools': [t.to_function_declaration() for t in self.tools],
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'max_iterations': self.max_iterations,
        })
        return base


# 别名，保持向后兼容
Agent = LlmAgent

