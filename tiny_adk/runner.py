"""
Runner - 无状态执行引擎

Runner 只负责:
- 会话管理（通过 SessionService）
- 编排 Flow 执行
- 用户消息处理

架构:
┌────────────────────────────────────────┐
│  Runner: 会话管理 + 编排               │
├────────────────────────────────────────┤
│  Flow: Reason-Act 循环 + 工具执行       │
├────────────────────────────────────────┤
│  Model: LLM 抽象 + 请求/响应格式化       │
└────────────────────────────────────────┘

设计理念:
- Runner 完全无状态
- Agent 作为参数传入
- 支持 InvocationContext 进行追踪
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
    无状态 Runner
    
    核心设计理念:
    - Runner 只持有基础服务（SessionService）
    - Agent 和配置作为参数传入
    - 每次调用都是独立的
    - 支持追踪和监控
    
    这种设计的好处:
    1. Runner 可以被多个 Agent 复用
    2. Agent 可以动态切换
    3. 更好的可测试性
    4. 更清晰的职责分离
    """
    
    def __init__(
        self,
        session_service: SessionService | None = None,
        config: Config | None = None,
    ):
        """
        初始化 Runner
        
        Args:
            session_service: Session 持久化服务（如果不提供，使用内存存储）
            config: 配置对象
        """
        self.session_service = session_service or SessionService()
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
        
        Args:
            agent: Agent 实例
            user_id: 用户 ID
            session_id: 会话 ID
            message: 用户消息
            trace_id: 追踪 ID（可选）
            stream: 是否流式返回
            
        Yields:
            Event 对象
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
            # 2. 加载 Session
            ctx.session = await self.session_service.get_session(
                user_id=user_id,
                session_id=session_id
            )
            
            # 3. 添加用户消息
            user_event = Event(
                event_type=EventType.USER_MESSAGE,
                content=message,
            )
            ctx.session.add_event(user_event)
            
            # 4. 获取 LLM
            llm = agent.llm or self._create_llm()
            
            # 5. 执行
            event_count = 0
            if stream:
                async for event in agent.flow.run_stream_async(agent, ctx.session, llm):
                    event_count += 1
                    yield event
            else:
                result = await agent.flow.run_async(agent, ctx.session, llm)
                final_event = Event(
                    event_type=EventType.MODEL_RESPONSE,
                    content=result,
                )
                event_count += 1
                yield final_event
            
            # 6. 保存 Session
            await self.session_service.save_session(ctx.session)
            
            # 7. 记录成功
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
        
        Args:
            agent: Agent 实例
            user_id: 用户 ID
            session_id: 会话 ID
            message: 用户消息
            trace_id: 追踪 ID（可选）
            
        Returns:
            Agent 的响应
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
            # 加载 Session
            ctx.session = self.session_service.get_session_sync(
                user_id=user_id,
                session_id=session_id
            )
            
            # 添加用户消息
            ctx.session.add_event(Event(
                event_type=EventType.USER_MESSAGE,
                content=message,
            ))
            
            # 获取 LLM
            llm = agent.llm or self._create_llm()
            
            # 执行
            result = agent.flow.run(agent, ctx.session, llm)
            
            # 保存 Session
            self.session_service.save_session_sync(ctx.session)
            
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
        
        Args:
            agent: Agent 实例
            user_id: 用户 ID
            session_id: 会话 ID
            message: 用户消息
            trace_id: 追踪 ID（可选）
            
        Yields:
            Event 对象
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
            # 加载 Session
            ctx.session = self.session_service.get_session_sync(
                user_id=user_id,
                session_id=session_id
            )
            
            # 添加用户消息
            ctx.session.add_event(Event(
                event_type=EventType.USER_MESSAGE,
                content=message,
            ))
            
            # 获取 LLM
            llm = agent.llm or self._create_llm()
            
            # 执行
            event_count = 0
            for event in agent.flow.run_stream(agent, ctx.session, llm):
                event_count += 1
                yield event
            
            # 保存 Session
            self.session_service.save_session_sync(ctx.session)
            
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
    
    def get_session(self, user_id: str, session_id: str) -> Session:
        """获取 Session（同步）"""
        return self.session_service.get_session_sync(user_id, session_id)
    
    async def get_session_async(self, user_id: str, session_id: str) -> Session:
        """获取 Session（异步）"""
        return await self.session_service.get_session(user_id, session_id)


# 向后兼容别名
StatelessRunner = Runner
