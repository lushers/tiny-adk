"""
Runner - 无状态执行引擎（参考 ADK 设计）

Runner 职责（单一职责）:
- 编排 Flow 执行
- 追加事件到 Session（通过 SessionService.append_event）

架构:
┌────────────────────────────────────────┐
│  Runner: 执行编排                       │
├────────────────────────────────────────┤
│  Flow: Reason-Act 循环 + 工具执行       │
├────────────────────────────────────────┤
│  Model: LLM 抽象 + 请求/响应格式化       │
└────────────────────────────────────────┘

设计理念（参考 ADK）:
- Runner 完全无状态
- Session 必须预先存在（由调用方创建）
- 事件追加是原子操作
- Agent 作为参数传入
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
    - Runner 只负责执行，不负责 Session 生命周期
    - Session 必须预先存在（不存在则抛出 ValueError）
    - 事件通过 append_event 原子追加
    - Agent 作为参数传入
    
    使用方式:
        # 1. 创建服务
        session_service = SessionService()
        runner = Runner(session_service=session_service)
        
        # 2. 显式创建 Session
        session = await session_service.create_session(user_id="u1", session_id="s1")
        
        # 3. 执行
        async for event in runner.run_async(agent, "u1", "s1", "你好"):
            print(event)
    """
    
    def __init__(
        self,
        session_service: SessionService,
        config: Config | None = None,
    ):
        """
        初始化 Runner
        
        Args:
            session_service: Session 持久化服务（必须提供）
            config: 配置对象
        """
        self.session_service = session_service
        self._config = config or get_config()
    
    # ==================== 异步 API ====================
    
    async def run_async(
        self,
        agent: Agent,
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
            agent: Agent 实例
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
        # 1. 创建 InvocationContext
        ctx = InvocationContext(
            invocation_id=str(uuid.uuid4()),
            trace_id=trace_id,
            agent=agent,
            user_message=message,
            run_config={'stream': stream, 'llm': agent.llm},
        )
        
        logger.info(
            f"[Runner] START invocation_id={ctx.invocation_id} "
            f"trace_id={ctx.trace_id} agent={agent.name}"
        )
        
        try:
            # 2. 获取 Session（不创建）
            ctx.session = await self.session_service.get_session(
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
            
            # 5. 执行
            event_count = 0
            if stream:
                async for event in agent.flow.run_stream_async(agent, ctx.session, llm):
                    event_count += 1
                    # 非 partial 事件追加到 Session
                    if not getattr(event, 'partial', False):
                        await self.session_service.append_event(ctx.session, event)
                    yield event
            else:
                result = await agent.flow.run_async(agent, ctx.session, llm)
                final_event = Event(
                    event_type=EventType.MODEL_RESPONSE,
                    content=result,
                )
                await self.session_service.append_event(ctx.session, final_event)
                event_count += 1
                yield final_event
            
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
        agent: Agent,
        user_id: str,
        session_id: str,
        message: str,
        trace_id: Optional[str] = None,
    ) -> str:
        """
        同步执行（非流式）
        
        前置条件: Session 必须已存在
        
        Args:
            agent: Agent 实例
            user_id: 用户 ID
            session_id: 会话 ID
            message: 用户消息
            trace_id: 追踪 ID（可选）
            
        Returns:
            Agent 的响应
            
        Raises:
            ValueError: 如果 Session 不存在
        """
        ctx = InvocationContext(
            invocation_id=str(uuid.uuid4()),
            trace_id=trace_id,
            agent=agent,
            user_message=message,
        )
        
        logger.info(
            f"[Runner] START invocation_id={ctx.invocation_id} agent={agent.name}"
        )
        
        try:
            # 获取 Session（不创建）
            ctx.session = self.session_service.get_session_sync(
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
            
            # 执行
            result = agent.flow.run(agent, ctx.session, llm)
            
            # 追加响应事件
            final_event = Event(
                event_type=EventType.MODEL_RESPONSE,
                content=result,
            )
            self.session_service.append_event_sync(ctx.session, final_event)
            
            duration = time.time() - ctx.start_time
            logger.info(
                f"[Runner] SUCCESS invocation_id={ctx.invocation_id} "
                f"duration={duration:.2f}s"
            )
            
            return result
            
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
        agent: Agent,
        user_id: str,
        session_id: str,
        message: str,
        trace_id: Optional[str] = None,
    ) -> Iterator[Event]:
        """
        同步流式执行
        
        前置条件: Session 必须已存在
        
        Args:
            agent: Agent 实例
            user_id: 用户 ID
            session_id: 会话 ID
            message: 用户消息
            trace_id: 追踪 ID（可选）
            
        Yields:
            Event 对象
            
        Raises:
            ValueError: 如果 Session 不存在
        """
        ctx = InvocationContext(
            invocation_id=str(uuid.uuid4()),
            trace_id=trace_id,
            agent=agent,
            user_message=message,
        )
        
        logger.info(
            f"[Runner] START invocation_id={ctx.invocation_id} agent={agent.name}"
        )
        
        try:
            # 获取 Session（不创建）
            ctx.session = self.session_service.get_session_sync(
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
            
            # 执行
            event_count = 0
            for event in agent.flow.run_stream(agent, ctx.session, llm):
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
        agent: Agent,
        message: str,
        user_id: str = "debug_user",
        session_id: str = "debug_session",
    ) -> list[Event]:
        """
        调试用便利方法 - 自动处理 Session 创建
        
        注意：仅用于调试和测试，生产环境请使用 run_async
        
        Args:
            agent: Agent 实例
            message: 用户消息
            user_id: 用户 ID（默认 debug_user）
            session_id: 会话 ID（默认 debug_session）
            
        Returns:
            事件列表
        """
        # 获取或创建 Session
        session = await self.session_service.get_session(user_id, session_id)
        if not session:
            session = await self.session_service.create_session(user_id, session_id)
            logger.info(f"[Runner] Created debug session: {session_id}")
        
        events = []
        async for event in self.run_async(agent, user_id, session_id, message):
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
        )
