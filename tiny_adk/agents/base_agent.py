"""BaseAgent - Agent 基类，定义配置和生命周期框架"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from ..events import Event
    from ..session import Session

logger = logging.getLogger(__name__)

# 回调类型定义（使用简化类型避免 Pydantic 前向引用问题）
BeforeAgentCallback = Callable[..., Any]  # (agent, session) -> Optional[str]
AfterAgentCallback = Callable[..., Any]   # (agent, session) -> None


class BaseAgent(BaseModel):
    """
    Agent 基类 - 纯配置容器 + 树形结构 + 生命周期框架
    
    设计理念（借鉴 Google ADK）：
    - Agent 是配置，不包含状态
    - run_async 是模板方法，子类实现 _run_async_impl
    - 支持 before/after 回调钩子
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')
    
    # === 基本配置 ===
    name: str
    """Agent 名称，必须是有效的 Python 标识符"""
    
    description: str = ''
    """Agent 描述，用于 LLM 决定是否委托给此 Agent"""
    
    # === 树形结构 ===
    sub_agents: list['BaseAgent'] = Field(default_factory=list)
    """子 Agent 列表"""
    
    parent_agent: Optional['BaseAgent'] = Field(default=None, exclude=True)
    """父 Agent（自动设置，不序列化）"""
    
    # === 生命周期回调 ===
    before_agent_callback: Optional[BeforeAgentCallback] = None
    """Agent 执行前的回调，返回内容则跳过执行"""
    
    after_agent_callback: Optional[AfterAgentCallback] = None
    """Agent 执行后的回调"""
    
    # === 验证器 ===
    
    @field_validator('name', mode='after')
    @classmethod
    def validate_name(cls, value: str) -> str:
        """验证 Agent 名称"""
        if not value.isidentifier():
            raise ValueError(
                f"Invalid agent name: '{value}'. "
                "Must be a valid Python identifier."
            )
        if value == 'user':
            raise ValueError("Agent name cannot be 'user' (reserved).")
        return value
    
    def model_post_init(self, __context: Any) -> None:
        """初始化后设置父子关系"""
        self._set_parent_for_sub_agents()
    
    def _set_parent_for_sub_agents(self) -> None:
        """为所有子 Agent 设置 parent_agent"""
        for sub_agent in self.sub_agents:
            if sub_agent.parent_agent is not None:
                raise ValueError(
                    f"Agent '{sub_agent.name}' already has parent "
                    f"'{sub_agent.parent_agent.name}', cannot add to '{self.name}'"
                )
            sub_agent.parent_agent = self
    
    # === 树形结构操作 ===
    
    @property
    def root_agent(self) -> 'BaseAgent':
        """获取根 Agent"""
        root = self
        while root.parent_agent is not None:
            root = root.parent_agent
        return root
    
    def find_agent(self, name: str) -> Optional['BaseAgent']:
        """在当前 Agent 及其后代中查找"""
        if self.name == name:
            return self
        return self.find_sub_agent(name)
    
    def find_sub_agent(self, name: str) -> Optional['BaseAgent']:
        """在后代中查找"""
        for sub_agent in self.sub_agents:
            if result := sub_agent.find_agent(name):
                return result
        return None
    
    # === 兼容性属性 ===
    
    @property
    def llm(self) -> Any:
        """
        LLM 实例（默认返回 None）
        
        这个属性是为了与 Runner 兼容。
        对于编排类 Agent（SequentialAgent、LoopAgent），返回 None。
        对于 LlmAgent，会被重写返回实际的 LLM 实例。
        """
        return None
    
    # === 执行入口（模板方法） ===
    
    async def run_async(
        self,
        session: 'Session',
        llm: Any = None,  # 保持与 Runner 兼容
    ) -> AsyncGenerator['Event', None]:
        """
        执行入口 - 模板方法
        
        处理生命周期回调，具体执行逻辑委托给 _run_async_impl
        """
        from ..events import Event, EventType
        
        logger.debug(f"[{self.name}] Starting execution")
        
        # before 回调
        if self.before_agent_callback:
            result = self.before_agent_callback(self, session)
            if hasattr(result, '__await__'):
                result = await result
            if result:
                # 回调返回内容，跳过执行
                yield Event(
                    event_type=EventType.TEXT_CHUNK,
                    author=self.name,
                    content=result,
                )
                return
        
        # 执行核心逻辑
        async for event in self._run_async_impl(session, llm=llm):
            yield event
        
        # after 回调
        if self.after_agent_callback:
            result = self.after_agent_callback(self, session)
            if hasattr(result, '__await__'):
                await result
        
        logger.debug(f"[{self.name}] Execution completed")
    
    async def _run_async_impl(
        self,
        session: 'Session',
        llm: Any = None,
    ) -> AsyncGenerator['Event', None]:
        """
        核心执行逻辑 - 子类必须实现
        """
        raise NotImplementedError(
            f"_run_async_impl not implemented for {type(self).__name__}"
        )
        yield  # 保持为 AsyncGenerator
    
    # === 序列化 ===
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'description': self.description,
            'sub_agents': [a.name for a in self.sub_agents],
        }

