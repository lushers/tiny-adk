"""
Runner - 使用三层架构的执行引擎

Runner 只负责:
- 会话管理
- 配置解析
- 编排 Flow 执行
- 用户消息处理

三层架构:
┌────────────────────────────────────────┐
│  Runner: 配置 + 会话管理 + 编排         │
├────────────────────────────────────────┤
│  Flow: Reason-Act 循环 + 工具执行       │
├────────────────────────────────────────┤
│  Model: LLM 抽象 + 请求/响应格式化       │
└────────────────────────────────────────┘
"""

from __future__ import annotations

from typing import AsyncIterator, Iterator

from .agents import Agent
from .config import Config, get_config
from .events import Event, EventType
from .flows import BaseFlow, SimpleFlow
from .models import BaseLlm, OpenAILlm
from .session import Session


class Runner:
    """
    Runner - 简化版执行引擎
    
    核心设计理念:
    - Runner 只负责编排，不包含具体的 LLM 调用逻辑
    - LLM 调用由 Model 层处理
    - Reason-Act 循环由 Flow 层处理
    - Runner 管理配置和会话
    """
    
    def __init__(
        self,
        llm: BaseLlm | None = None,
        flow: BaseFlow | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        default_model: str | None = None,
        show_thinking: bool | None = None,
        show_request: bool | None = None,
        config: Config | None = None,
    ):
        """
        初始化 Runner
        
        Args:
            llm: LLM 实例（如果不提供，会根据配置自动创建）
            flow: Flow 实例（如果不提供，默认使用 SimpleFlow）
            api_base: API 地址
            api_key: API 密钥
            default_model: 默认模型名称
            show_thinking: 是否显示思考过程
            show_request: 是否显示 API 请求详情（调试用）
            config: 配置对象
        """
        # 获取配置
        self._config = config or get_config()
        
        # 合并配置
        self.api_base = api_base if api_base is not None else self._config.llm.api_base
        self.api_key = api_key if api_key is not None else self._config.llm.api_key
        self.default_model = default_model if default_model is not None else self._config.llm.model
        self.show_thinking = show_thinking if show_thinking is not None else self._config.runner.show_thinking
        self.show_request = show_request if show_request is not None else self._config.runner.show_request
        
        # 初始化 LLM
        self.llm = llm or self._create_llm()
        
        # 初始化 Flow
        self.flow = flow or SimpleFlow()
    
    def _create_llm(self) -> BaseLlm:
        """根据配置创建 LLM 实例"""
        if not self.api_base:
            raise ValueError(
                "未配置 LLM API。请在配置文件 (tiny_adk.yaml) 中设置 llm.api_base，"
                "或在创建 Runner 时传入 api_base 参数，"
                "或直接传入 llm 实例。"
            )
        return OpenAILlm(
            api_base=self.api_base,
            api_key=self.api_key,
            model=self.default_model,
            show_thinking=self.show_thinking,
            show_request=self.show_request,
        )
    
    # ==================== 公共 API ====================
    
    def run(
        self,
        agent: Agent,
        session: Session,
        user_message: str,
    ) -> str:
        """
        同步执行一轮对话
        
        Args:
            agent: 要执行的 Agent
            session: 会话对象
            user_message: 用户消息
        
        Returns:
            Agent 的最终响应
        """
        # 1. 记录用户消息
        session.add_event(Event(
            event_type=EventType.USER_MESSAGE,
            content=user_message,
        ))
        
        # 2. 委托给 Flow 执行
        return self.flow.run(agent, session, self.llm)
    
    def run_stream(
        self,
        agent: Agent,
        session: Session,
        user_message: str,
    ) -> Iterator[Event]:
        """
        同步流式执行
        
        Yields:
            执行过程中的事件
        """
        # 1. 记录用户消息
        session.add_event(Event(
            event_type=EventType.USER_MESSAGE,
            content=user_message,
        ))
        
        # 2. 委托给 Flow 执行
        yield from self.flow.run_stream(agent, session, self.llm)
    
    async def run_async(
        self,
        agent: Agent,
        session: Session,
        user_message: str,
    ) -> str:
        """
        异步执行一轮对话
        
        Args:
            agent: 要执行的 Agent
            session: 会话对象
            user_message: 用户消息
        
        Returns:
            Agent 的最终响应
        """
        # 1. 记录用户消息
        session.add_event(Event(
            event_type=EventType.USER_MESSAGE,
            content=user_message,
        ))
        
        # 2. 委托给 Flow 执行
        return await self.flow.run_async(agent, session, self.llm)
    
    async def run_stream_async(
        self,
        agent: Agent,
        session: Session,
        user_message: str,
    ) -> AsyncIterator[Event]:
        """
        异步流式执行
        
        Yields:
            执行过程中的事件
        """
        # 1. 记录用户消息
        session.add_event(Event(
            event_type=EventType.USER_MESSAGE,
            content=user_message,
        ))
        
        # 2. 委托给 Flow 执行
        async for event in self.flow.run_stream_async(agent, session, self.llm):
            yield event


# ==================== 便捷类 ====================

class InMemoryRunner(Runner):
    """
    内存版 Runner - 用于测试和快速原型
    
    自动使用内存中的会话服务，无需外部依赖
    """
    
    def __init__(
        self,
        agent: Agent | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._default_agent = agent
    
    def quick_run(
        self,
        user_message: str,
        agent: Agent | None = None,
        session: Session | None = None,
    ) -> str:
        """
        快速运行 - 自动创建会话
        
        Args:
            user_message: 用户消息
            agent: Agent（如果不提供，使用默认 Agent）
            session: 会话（如果不提供，自动创建）
        
        Returns:
            Agent 的响应
        """
        agent = agent or self._default_agent
        if agent is None:
            raise ValueError("需要提供 agent 参数或在初始化时设置默认 agent")
        
        session = session or Session(session_id="quick_session")
        return self.run(agent, session, user_message)

