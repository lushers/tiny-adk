"""
Runner - 无状态执行引擎（参考 ADK 设计）

Runner 职责（单一职责）:
- 绑定特定的 Agent 和 App
- 编排 Flow 执行
- 追加事件到 Session（通过 SessionService.append_event）

架构:
┌────────────────────────────────────────┐
│  Runner: 执行编排（绑定 Agent）          │
├────────────────────────────────────────┤
│  Flow: Reason-Act 循环 + 工具执行       │
├────────────────────────────────────────┤
│  Model: LLM 抽象 + 请求/响应格式化       │
└────────────────────────────────────────┘

设计理念（参考 ADK）:
- Runner 绑定特定的 app_name 和 agent
- Runner 本身无状态（Agent 是只读配置）
- Session 必须预先存在（由调用方创建）
- 事件追加是原子操作
- 支持高并发（状态通过 session 隔离）
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import AsyncIterator, Iterator, Optional

from .agents import Agent
from .config import Config, get_config
from .events import Event, EventType
from .flows import SimpleFlow
from .models import BaseLlm, OpenAILlm
from .session import Session, SessionService, InvocationContext

logger = logging.getLogger(__name__)


class Runner:
    """
    无状态 Runner（参考 ADK 设计）
    
    核心设计理念:
    - Runner 绑定特定的 app_name 和 agent
    - Runner 本身无状态（Agent 是只读配置，不是运行时状态）
    - Session 必须预先存在（不存在则抛出 ValueError）
    - 事件通过 append_event 原子追加
    - 支持高并发（状态通过 user_id + session_id 隔离）
    
    使用方式:
        # 1. 创建 Agent
        agent = Agent(name="助手", instruction="...")
        
        # 2. 创建 Runner（绑定 Agent）
        session_service = SessionService()
        runner = Runner(
            app_name="my_app",
            agent=agent,
            session_service=session_service,
        )
        
        # 3. 创建 Session
        session = await runner.session_service.create_session(
            app_name="my_app",
            user_id="u1",
            session_id="s1"
        )
        
        # 4. 执行（不需要传 agent）
        async for event in runner.run_async(user_id="u1", session_id="s1", message="你好"):
            print(event)
    """
    
    app_name: str
    """应用名称"""
    agent: Agent
    """绑定的 Agent（只读配置）"""
    session_service: SessionService
    """Session 持久化服务"""
    
    def __init__(
        self,
        app_name: str,
        agent: Agent,
        session_service: SessionService,
        config: Config | None = None,
    ):
        """
        初始化 Runner
        
        Args:
            app_name: 应用名称（用于 Session 隔离）
            agent: 绑定的 Agent 实例
            session_service: Session 持久化服务
            config: 配置对象
        """
        self.app_name = app_name
        self.agent = agent
        self.session_service = session_service
        self._config = config or get_config()
    
    # ==================== 异步 API ====================
    
    async def run_async(
        self,
        user_id: str,
        session_id: str,
        message: str,
        trace_id: Optional[str] = None,
        stream: bool = False,
    ) -> AsyncIterator[Event]:
        """
        异步执行
        
        前置条件: Session 必须已存在（通过 session_service.create_session 创建）
        
        Args:
            user_id: 用户 ID
            session_id: 会话 ID
            message: 用户消息
            trace_id: 追踪 ID（可选）
            stream: 是否流式返回
            
        Yields:
            Event 对象
            
        Raises:
            ValueError: 如果 Session 不存在
        """
        agent = self.agent
        
        # 1. 创建 InvocationContext
        ctx = InvocationContext(
            invocation_id=str(uuid.uuid4()),
            trace_id=trace_id,
            agent=agent,
            user_message=message,
            run_config={'stream': stream, 'llm': agent.llm},
        )
        
        logger.info(
            f"[Runner] START app={self.app_name} invocation_id={ctx.invocation_id} "
            f"trace_id={ctx.trace_id} agent={agent.name}"
        )
        
        try:
            # 2. 获取 Session（不创建）
            ctx.session = await self.session_service.get_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id
            )
            if not ctx.session:
                raise ValueError(f"Session not found: {session_id}")
            
            # 3. 追加用户消息（原子操作）
            user_event = Event(
                event_type=EventType.USER_MESSAGE,
                content=message,
            )
            await self.session_service.append_event(ctx.session, user_event)
            
            # 4. 获取 LLM
            llm = agent.llm or self._create_llm()
            
            # 5. 执行（使用 Agent 的 run_async，支持多 Agent 跳转）
            event_count = 0
            current_agent = agent
            
            if stream:
                async for event in current_agent.run_async(ctx.session, llm):
                    event_count += 1
                    # 非 partial 事件追加到 Session
                    if not getattr(event, 'partial', False):
                        await self.session_service.append_event(ctx.session, event)
                    yield event
            else:
                # 非流式：收集所有事件，返回最后的响应
                final_content = ""
                async for event in current_agent.run_async(ctx.session, llm):
                    event_count += 1
                    if not getattr(event, 'partial', False):
                        await self.session_service.append_event(ctx.session, event)
                    if event.event_type == EventType.MODEL_RESPONSE:
                        final_content = event.content or ""
                
                if final_content:
                    yield Event(
                        event_type=EventType.MODEL_RESPONSE,
                        content=final_content,
                        author=current_agent.name,
                    )
            
            # 6. 记录成功
            duration = time.time() - ctx.start_time
            logger.info(
                f"[Runner] SUCCESS invocation_id={ctx.invocation_id} "
                f"duration={duration:.2f}s events={event_count}"
            )
            
        except Exception as e:
            duration = time.time() - ctx.start_time
            logger.error(
                f"[Runner] FAILED invocation_id={ctx.invocation_id} "
                f"error={str(e)} duration={duration:.2f}s",
                exc_info=True
            )
            raise
    
    # ==================== 同步 API ====================
    
    def run(
        self,
        user_id: str,
        session_id: str,
        message: str,
        trace_id: Optional[str] = None,
    ) -> str:
        """
        同步执行（非流式）
        
        前置条件: Session 必须已存在
        
        Args:
            user_id: 用户 ID
            session_id: 会话 ID
            message: 用户消息
            trace_id: 追踪 ID（可选）
            
        Returns:
            Agent 的响应
            
        Raises:
            ValueError: 如果 Session 不存在
        """
        agent = self.agent
        
        ctx = InvocationContext(
            invocation_id=str(uuid.uuid4()),
            trace_id=trace_id,
            agent=agent,
            user_message=message,
        )
        
        logger.info(
            f"[Runner] START app={self.app_name} invocation_id={ctx.invocation_id} agent={agent.name}"
        )
        
        try:
            # 获取 Session（不创建）
            ctx.session = self.session_service.get_session_sync(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id
            )
            if not ctx.session:
                raise ValueError(f"Session not found: {session_id}")
            
            # 追加用户消息（原子操作）
            user_event = Event(
                event_type=EventType.USER_MESSAGE,
                content=message,
            )
            self.session_service.append_event_sync(ctx.session, user_event)
            
            # 获取 LLM
            llm = agent.llm or self._create_llm()
            
            # 执行：收集所有事件并添加到 session（非流式）
            final_content = ""
            event_count = 0
            for event in agent.flow.run(agent, ctx.session, llm, stream=False):
                event_count += 1
                # 非 partial 事件追加到 Session
                if not getattr(event, 'partial', False):
                    self.session_service.append_event_sync(ctx.session, event)
                if event.event_type == EventType.MODEL_RESPONSE:
                    final_content = event.content or ""
            
            duration = time.time() - ctx.start_time
            logger.info(
                f"[Runner] SUCCESS invocation_id={ctx.invocation_id} "
                f"duration={duration:.2f}s events={event_count}"
            )
            
            return final_content
            
        except Exception as e:
            duration = time.time() - ctx.start_time
            logger.error(
                f"[Runner] FAILED invocation_id={ctx.invocation_id} "
                f"error={str(e)} duration={duration:.2f}s",
                exc_info=True
            )
            raise
    
    def run_stream(
        self,
        user_id: str,
        session_id: str,
        message: str,
        trace_id: Optional[str] = None,
    ) -> Iterator[Event]:
        """
        同步流式执行
        
        前置条件: Session 必须已存在
        
        Args:
            user_id: 用户 ID
            session_id: 会话 ID
            message: 用户消息
            trace_id: 追踪 ID（可选）
            
        Yields:
            Event 对象
            
        Raises:
            ValueError: 如果 Session 不存在
        """
        agent = self.agent
        
        ctx = InvocationContext(
            invocation_id=str(uuid.uuid4()),
            trace_id=trace_id,
            agent=agent,
            user_message=message,
        )
        
        logger.info(
            f"[Runner] START app={self.app_name} invocation_id={ctx.invocation_id} agent={agent.name}"
        )
        
        try:
            # 获取 Session（不创建）
            ctx.session = self.session_service.get_session_sync(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id
            )
            if not ctx.session:
                raise ValueError(f"Session not found: {session_id}")
            
            # 追加用户消息（原子操作）
            user_event = Event(
                event_type=EventType.USER_MESSAGE,
                content=message,
            )
            self.session_service.append_event_sync(ctx.session, user_event)
            
            # 获取 LLM
            llm = agent.llm or self._create_llm()
            
            # 执行（流式）
            event_count = 0
            for event in agent.flow.run(agent, ctx.session, llm, stream=True):
                event_count += 1
                # 非 partial 事件追加到 Session
                if not getattr(event, 'partial', False):
                    self.session_service.append_event_sync(ctx.session, event)
                yield event
            
            duration = time.time() - ctx.start_time
            logger.info(
                f"[Runner] SUCCESS invocation_id={ctx.invocation_id} "
                f"duration={duration:.2f}s events={event_count}"
            )
            
        except Exception as e:
            duration = time.time() - ctx.start_time
            logger.error(
                f"[Runner] FAILED invocation_id={ctx.invocation_id} "
                f"error={str(e)} duration={duration:.2f}s",
                exc_info=True
            )
            raise
    
    # ==================== 便利方法（用于调试）====================
    
    async def run_debug(
        self,
        message: str,
        user_id: str = "debug_user",
        session_id: str = "debug_session",
    ) -> list[Event]:
        """
        调试用便利方法 - 自动处理 Session 创建
        
        注意：仅用于调试和测试，生产环境请使用 run_async
        
        Args:
            message: 用户消息
            user_id: 用户 ID（默认 debug_user）
            session_id: 会话 ID（默认 debug_session）
            
        Returns:
            事件列表
        """
        # 获取或创建 Session
        session = await self.session_service.get_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id
        )
        if not session:
            session = await self.session_service.create_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id
            )
            logger.info(f"[Runner] Created debug session: {session_id}")
        
        events = []
        async for event in self.run_async(user_id, session_id, message):
            events.append(event)
        return events
    
    # ==================== 辅助方法 ====================
    
    def _create_llm(self) -> BaseLlm:
        """根据配置创建 LLM 实例"""
        if not self._config.llm.api_base:
            raise ValueError(
                "未配置 LLM API。请在配置文件 (tiny_adk.yaml) 中设置 llm.api_base，"
                "或在 Agent 中直接传入 LLM 实例。"
            )
        return OpenAILlm(
            api_base=self._config.llm.api_base,
            api_key=self._config.llm.api_key,
            model=self._config.llm.model,
            show_thinking=self._config.runner.show_thinking,
            show_request=self._config.runner.show_request,
            log_level=self._config.runner.log_level,
        )
