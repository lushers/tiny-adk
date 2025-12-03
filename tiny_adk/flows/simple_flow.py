"""简单的 Reason-Act 循环实现"""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, AsyncIterator, Iterator

from .base_flow import BaseFlow

if TYPE_CHECKING:
    from ..agents import Agent
    from ..models import BaseLlm
    from ..session import Session

from ..events import Event, EventType


class SimpleFlow(BaseFlow):
    """
    简单的 Reason-Act 循环实现
    
    这是最基本的 Flow 实现，执行以下循环：
    1. 构建请求 -> 调用 LLM -> 获取响应
    2. 如果响应包含工具调用 -> 执行工具 -> 回到步骤 1
    3. 如果响应是最终文本 -> 返回结果
    
    特点:
    - 支持同步/异步、流式/非流式
    - 支持工具调用和多轮循环
    - 自动处理思考内容分离
    """
    
    # ==================== 同步非流式 ====================
    
    def run(
        self,
        agent: Agent,
        session: Session,
        llm: BaseLlm,
    ) -> str:
        """同步执行"""
        return self._reason_act_loop(agent, session, llm, iteration=0)
    
    def _reason_act_loop(
        self,
        agent: Agent,
        session: Session,
        llm: BaseLlm,
        iteration: int,
    ) -> str:
        """Reason-Act 循环"""
        # 检查迭代次数
        if iteration >= self.max_iterations:
            error_msg = f"⚠️ 达到最大迭代次数限制 ({self.max_iterations})"
            session.add_event(Event(
                event_type=EventType.ERROR,
                content={'error': error_msg, 'iteration': iteration},
            ))
            return error_msg
        
        # 构建请求并调用 LLM
        request = self.build_request(agent, session)
        response = llm.generate(request)
        
        # 记录响应
        session.add_event(Event(
            event_type=EventType.MODEL_RESPONSE,
            content=response.content,
            metadata={
                'model': response.model,
                'thinking': response.thinking,
                'finish_reason': response.finish_reason,
            },
        ))
        
        # 如果有工具调用，执行工具并继续循环
        if response.has_tool_calls():
            for tool_call in response.tool_calls:
                self._execute_tool(agent, session, tool_call)
            return self._reason_act_loop(agent, session, llm, iteration + 1)
        
        return response.content
    
    # ==================== 同步流式 ====================
    
    def run_stream(
        self,
        agent: Agent,
        session: Session,
        llm: BaseLlm,
    ) -> Iterator[Event]:
        """同步流式执行"""
        yield from self._reason_act_loop_stream(agent, session, llm, iteration=0)
    
    def _reason_act_loop_stream(
        self,
        agent: Agent,
        session: Session,
        llm: BaseLlm,
        iteration: int,
    ) -> Iterator[Event]:
        """流式 Reason-Act 循环"""
        # 检查迭代次数
        if iteration >= self.max_iterations:
            error_msg = f"⚠️ 达到最大迭代次数限制 ({self.max_iterations})"
            event = Event(
                event_type=EventType.ERROR,
                content={'error': error_msg, 'iteration': iteration},
            )
            session.add_event(event)
            yield event
            return
        
        # 构建请求
        request = self.build_request(agent, session)
        
        # 流式调用 LLM
        final_response = None
        for response in llm.generate_stream(request):
            if response.partial:
                # 流式增量
                yield Event(
                    event_type=EventType.MODEL_RESPONSE_DELTA,
                    content=response.delta,
                    metadata={'chunk_index': response.metadata.get('chunk_index')},
                )
            else:
                # 完整响应
                final_response = response
                event = Event(
                    event_type=EventType.MODEL_RESPONSE,
                    content=response.content,
                    metadata={
                        'model': response.model,
                        'thinking': response.thinking,
                        'finish_reason': response.finish_reason,
                    },
                )
                session.add_event(event)
                yield event
        
        # 处理工具调用
        if final_response and final_response.has_tool_calls():
            for tool_call in final_response.tool_calls:
                yield from self._execute_tool_stream(agent, session, tool_call)
            yield from self._reason_act_loop_stream(agent, session, llm, iteration + 1)
    
    # ==================== 异步非流式 ====================
    
    async def run_async(
        self,
        agent: Agent,
        session: Session,
        llm: BaseLlm,
    ) -> str:
        """异步执行"""
        return await self._reason_act_loop_async(agent, session, llm, iteration=0)
    
    async def _reason_act_loop_async(
        self,
        agent: Agent,
        session: Session,
        llm: BaseLlm,
        iteration: int,
    ) -> str:
        """异步 Reason-Act 循环"""
        # 检查迭代次数
        if iteration >= self.max_iterations:
            error_msg = f"⚠️ 达到最大迭代次数限制 ({self.max_iterations})"
            session.add_event(Event(
                event_type=EventType.ERROR,
                content={'error': error_msg, 'iteration': iteration},
            ))
            return error_msg
        
        # 构建请求并调用 LLM
        request = self.build_request(agent, session)
        response = await llm.generate_async(request)
        
        # 记录响应
        session.add_event(Event(
            event_type=EventType.MODEL_RESPONSE,
            content=response.content,
            metadata={
                'model': response.model,
                'thinking': response.thinking,
                'finish_reason': response.finish_reason,
            },
        ))
        
        # 如果有工具调用，执行工具并继续循环
        if response.has_tool_calls():
            for tool_call in response.tool_calls:
                await self._execute_tool_async(agent, session, tool_call)
            return await self._reason_act_loop_async(agent, session, llm, iteration + 1)
        
        return response.content
    
    # ==================== 异步流式 ====================
    
    async def run_stream_async(
        self,
        agent: Agent,
        session: Session,
        llm: BaseLlm,
    ) -> AsyncIterator[Event]:
        """异步流式执行"""
        async for event in self._reason_act_loop_stream_async(agent, session, llm, iteration=0):
            yield event
    
    async def _reason_act_loop_stream_async(
        self,
        agent: Agent,
        session: Session,
        llm: BaseLlm,
        iteration: int,
    ) -> AsyncIterator[Event]:
        """异步流式 Reason-Act 循环"""
        # 检查迭代次数
        if iteration >= self.max_iterations:
            error_msg = f"⚠️ 达到最大迭代次数限制 ({self.max_iterations})"
            event = Event(
                event_type=EventType.ERROR,
                content={'error': error_msg, 'iteration': iteration},
            )
            session.add_event(event)
            yield event
            return
        
        # 构建请求
        request = self.build_request(agent, session)
        
        # 流式调用 LLM
        final_response = None
        async for response in llm.generate_stream_async(request):
            if response.partial:
                yield Event(
                    event_type=EventType.MODEL_RESPONSE_DELTA,
                    content=response.delta,
                    metadata={'chunk_index': response.metadata.get('chunk_index')},
                )
            else:
                final_response = response
                event = Event(
                    event_type=EventType.MODEL_RESPONSE,
                    content=response.content,
                    metadata={
                        'model': response.model,
                        'thinking': response.thinking,
                        'finish_reason': response.finish_reason,
                    },
                )
                session.add_event(event)
                yield event
        
        # 处理工具调用
        if final_response and final_response.has_tool_calls():
            for tool_call in final_response.tool_calls:
                async for event in self._execute_tool_stream_async(agent, session, tool_call):
                    yield event
            async for event in self._reason_act_loop_stream_async(agent, session, llm, iteration + 1):
                yield event
    
    # ==================== 工具执行 ====================
    
    def _execute_tool(self, agent, session, tool_call) -> None:
        """执行工具调用"""
        session.add_event(Event(
            event_type=EventType.TOOL_CALL,
            content={
                'id': tool_call.id,
                'name': tool_call.name,
                'arguments': tool_call.arguments,
            },
        ))
        
        tool = self.find_tool(agent, tool_call.name)
        if tool:
            try:
                result = tool.execute(**tool_call.arguments)
                session.add_event(Event(
                    event_type=EventType.TOOL_RESPONSE,
                    content={
                        'call_id': tool_call.id,
                        'name': tool_call.name,
                        'result': str(result),
                    },
                ))
            except Exception as e:
                session.add_event(Event(
                    event_type=EventType.ERROR,
                    content={'tool': tool_call.name, 'error': str(e)},
                ))
    
    def _execute_tool_stream(self, agent, session, tool_call) -> Iterator[Event]:
        """流式执行工具"""
        event = Event(
            event_type=EventType.TOOL_CALL,
            content={
                'id': tool_call.id,
                'name': tool_call.name,
                'arguments': tool_call.arguments,
            },
        )
        session.add_event(event)
        yield event
        
        tool = self.find_tool(agent, tool_call.name)
        if tool:
            try:
                result = tool.execute(**tool_call.arguments)
                event = Event(
                    event_type=EventType.TOOL_RESPONSE,
                    content={
                        'call_id': tool_call.id,
                        'name': tool_call.name,
                        'result': str(result),
                    },
                )
                session.add_event(event)
                yield event
            except Exception as e:
                event = Event(
                    event_type=EventType.ERROR,
                    content={'tool': tool_call.name, 'error': str(e)},
                )
                session.add_event(event)
                yield event
    
    async def _execute_tool_async(self, agent, session, tool_call) -> None:
        """异步执行工具"""
        session.add_event(Event(
            event_type=EventType.TOOL_CALL,
            content={
                'id': tool_call.id,
                'name': tool_call.name,
                'arguments': tool_call.arguments,
            },
        ))
        
        tool = self.find_tool(agent, tool_call.name)
        if tool:
            try:
                if inspect.iscoroutinefunction(tool.func):
                    result = await tool.func(**tool_call.arguments)
                else:
                    result = await asyncio.to_thread(tool.execute, **tool_call.arguments)
                
                session.add_event(Event(
                    event_type=EventType.TOOL_RESPONSE,
                    content={
                        'call_id': tool_call.id,
                        'name': tool_call.name,
                        'result': str(result),
                    },
                ))
            except Exception as e:
                session.add_event(Event(
                    event_type=EventType.ERROR,
                    content={'tool': tool_call.name, 'error': str(e)},
                ))
    
    async def _execute_tool_stream_async(self, agent, session, tool_call) -> AsyncIterator[Event]:
        """异步流式执行工具"""
        event = Event(
            event_type=EventType.TOOL_CALL,
            content={
                'id': tool_call.id,
                'name': tool_call.name,
                'arguments': tool_call.arguments,
            },
        )
        session.add_event(event)
        yield event
        
        tool = self.find_tool(agent, tool_call.name)
        if tool:
            try:
                if inspect.iscoroutinefunction(tool.func):
                    result = await tool.func(**tool_call.arguments)
                else:
                    result = await asyncio.to_thread(tool.execute, **tool_call.arguments)
                
                event = Event(
                    event_type=EventType.TOOL_RESPONSE,
                    content={
                        'call_id': tool_call.id,
                        'name': tool_call.name,
                        'result': str(result),
                    },
                )
                session.add_event(event)
                yield event
            except Exception as e:
                event = Event(
                    event_type=EventType.ERROR,
                    content={'tool': tool_call.name, 'error': str(e)},
                )
                session.add_event(event)
                yield event

