"""简单的 Reason-Act 循环实现（支持多 Agent）"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING, AsyncIterator, Iterator, Optional

from .base_flow import BaseFlow

if TYPE_CHECKING:
    from ..agents import Agent
    from ..models import BaseLlm
    from ..session import Session

from ..events import Event, EventType, EventActions

logger = logging.getLogger(__name__)


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
        
        logger.debug(f"[SimpleFlow] Iteration {iteration + 1}")
        
        # 构建请求并调用 LLM
        request = self.build_request(agent, session)
        response = llm.generate(request)
        
        # 记录响应
        session.add_event(Event(
            event_type=EventType.MODEL_RESPONSE,
            content=response.content,
            author=agent.name,
            metadata={
                'model': response.model,
                'thinking': response.thinking,
                'finish_reason': response.finish_reason,
            },
        ))
        
        # 如果有工具调用，执行工具并继续循环
        if response.has_function_calls():
            for fc in response.function_calls:
                self._execute_tool(agent, session, fc)
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
            # 由 Runner 统一追加事件
            yield event
            return
        
        logger.debug(f"[SimpleFlow] Stream iteration {iteration + 1}")
        
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
                    author=agent.name,
                    metadata={
                        'model': response.model,
                        'thinking': response.thinking,
                        'finish_reason': response.finish_reason,
                    },
                )
                # 注意：不在这里调用 session.add_event(event)
                # 由 Runner 统一负责事件追加，避免重复
                yield event
        
        # 处理工具调用
        if final_response and final_response.has_function_calls():
            for fc in final_response.function_calls:
                yield from self._execute_tool_stream(agent, session, fc)
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
        
        logger.debug(f"[SimpleFlow] Async iteration {iteration + 1}")
        
        # 构建请求并调用 LLM
        request = self.build_request(agent, session)
        response = await llm.generate_async(request)
        
        # 记录响应
        session.add_event(Event(
            event_type=EventType.MODEL_RESPONSE,
            content=response.content,
            author=agent.name,
            metadata={
                'model': response.model,
                'thinking': response.thinking,
                'finish_reason': response.finish_reason,
            },
        ))
        
        # 如果有工具调用，执行工具并继续循环
        if response.has_function_calls():
            for fc in response.function_calls:
                await self._execute_tool_async(agent, session, fc)
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
            # 由 Runner 统一追加事件
            yield event
            return
        
        logger.debug(f"[SimpleFlow] Async stream iteration {iteration + 1}")
        
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
                    author=agent.name,
                    metadata={
                        'model': response.model,
                        'thinking': response.thinking,
                        'finish_reason': response.finish_reason,
                    },
                )
                # 注意：不在这里调用 session.add_event(event)
                # 由 Runner 统一负责事件追加，避免重复
                yield event
        
        # 处理工具调用
        if final_response and final_response.has_function_calls():
            for fc in final_response.function_calls:
                async for event in self._execute_tool_stream_async(agent, session, fc):
                    yield event
            async for event in self._reason_act_loop_stream_async(agent, session, llm, iteration + 1):
                yield event
    
    # ==================== 工具执行 ====================
    
    def _execute_tool(self, agent, session, function_call) -> None:
        """执行工具调用"""
        # 获取工具调用信息（FunctionCall 对象）
        call_id = function_call.id
        call_name = function_call.name
        call_args = function_call.args if hasattr(function_call, 'args') else function_call.arguments
        
        session.add_event(Event(
            event_type=EventType.TOOL_CALL,
            content={
                'id': call_id,
                'name': call_name,
                'arguments': call_args,
            },
        ))
        
        tool = self.find_tool(agent, call_name)
        if tool:
            try:
                # 使用新接口：run(args, context)
                if hasattr(tool, 'run'):
                    result = tool.run(call_args, None)
                else:
                    # 兼容旧接口
                    result = tool.execute(**call_args)
                
                session.add_event(Event(
                    event_type=EventType.TOOL_RESPONSE,
                    content={
                        'call_id': call_id,
                        'name': call_name,
                        'result': str(result),
                    },
                ))
            except Exception as e:
                logger.error(f"Tool {call_name} execution failed: {e}")
                session.add_event(Event(
                    event_type=EventType.ERROR,
                    content={'tool': call_name, 'error': str(e)},
                ))
    
    def _execute_tool_stream(self, agent, session, function_call) -> Iterator[Event]:
        """流式执行工具"""
        call_id = function_call.id
        call_name = function_call.name
        call_args = function_call.args if hasattr(function_call, 'args') else function_call.arguments
        
        event = Event(
            event_type=EventType.TOOL_CALL,
            content={
                'id': call_id,
                'name': call_name,
                'arguments': call_args,
            },
        )
        # 由 Runner 统一追加事件
        yield event
        
        tool = self.find_tool(agent, call_name)
        if tool:
            try:
                if hasattr(tool, 'run'):
                    result = tool.run(call_args, None)
                else:
                    result = tool.execute(**call_args)
                
                event = Event(
                    event_type=EventType.TOOL_RESPONSE,
                    content={
                        'call_id': call_id,
                        'name': call_name,
                        'result': str(result),
                    },
                )
                # 由 Runner 统一追加事件
                yield event
            except Exception as e:
                logger.error(f"Tool {call_name} execution failed: {e}")
                event = Event(
                    event_type=EventType.ERROR,
                    content={'tool': call_name, 'error': str(e)},
                )
                # 由 Runner 统一追加事件
                yield event
    
    async def _execute_tool_async(self, agent, session, function_call) -> None:
        """异步执行工具"""
        call_id = function_call.id
        call_name = function_call.name
        call_args = function_call.args if hasattr(function_call, 'args') else function_call.arguments
        
        session.add_event(Event(
            event_type=EventType.TOOL_CALL,
            content={
                'id': call_id,
                'name': call_name,
                'arguments': call_args,
            },
        ))
        
        tool = self.find_tool(agent, call_name)
        if tool:
            try:
                # 使用新接口：run_async(args, context)
                if hasattr(tool, 'run_async'):
                    result = await tool.run_async(call_args, None)
                elif hasattr(tool, 'func') and inspect.iscoroutinefunction(tool.func):
                    result = await tool.func(**call_args)
                elif hasattr(tool, 'execute'):
                    result = await asyncio.to_thread(tool.execute, **call_args)
                else:
                    result = await asyncio.to_thread(tool.run, call_args, None)
                
                session.add_event(Event(
                    event_type=EventType.TOOL_RESPONSE,
                    content={
                        'call_id': call_id,
                        'name': call_name,
                        'result': str(result),
                    },
                ))
            except Exception as e:
                logger.error(f"Tool {call_name} execution failed: {e}")
                session.add_event(Event(
                    event_type=EventType.ERROR,
                    content={'tool': call_name, 'error': str(e)},
                ))
    
    async def _execute_tool_stream_async(self, agent, session, function_call) -> AsyncIterator[Event]:
        """异步流式执行工具（支持 Agent 跳转）"""
        call_id = function_call.id
        call_name = function_call.name
        call_args = function_call.args if hasattr(function_call, 'args') else function_call.arguments
        
        event = Event(
            event_type=EventType.TOOL_CALL,
            content={
                'id': call_id,
                'name': call_name,
                'arguments': call_args,
            },
            author=agent.name,
        )
        # 由 Runner 统一追加事件
        yield event
        
        tool = self.find_tool(agent, call_name)
        if tool:
            try:
                if hasattr(tool, 'run_async'):
                    result = await tool.run_async(call_args, None)
                elif hasattr(tool, 'func') and inspect.iscoroutinefunction(tool.func):
                    result = await tool.func(**call_args)
                elif hasattr(tool, 'execute'):
                    result = await asyncio.to_thread(tool.execute, **call_args)
                else:
                    result = await asyncio.to_thread(tool.run, call_args, None)
                
                # 检查是否是 Agent 跳转
                if isinstance(result, dict) and result.get('transfer'):
                    target_agent = result.get('target_agent', '')
                    reason = result.get('reason', '')
                    
                    event = Event(
                        event_type=EventType.AGENT_TRANSFER,
                        content={
                            'from_agent': agent.name,
                            'target_agent': target_agent,
                            'reason': reason,
                        },
                        author=agent.name,
                        actions=EventActions(transfer_to_agent=target_agent),
                        metadata={'transfer_to_agent': target_agent},
                    )
                    # 由 Runner 统一追加事件
                    yield event
                    return
                
                # 检查是否是 escalate
                if isinstance(result, dict) and result.get('escalate'):
                    reason = result.get('reason', 'Task completed')
                    
                    event = Event(
                        event_type=EventType.TOOL_RESPONSE,
                        content={
                            'call_id': call_id,
                            'name': call_name,
                            'result': reason,
                        },
                        author=agent.name,
                        actions=EventActions(escalate=True),
                        metadata={'escalate': True},
                    )
                    # 由 Runner 统一追加事件
                    yield event
                    return
                
                # 普通工具响应
                event = Event(
                    event_type=EventType.TOOL_RESPONSE,
                    content={
                        'call_id': call_id,
                        'name': call_name,
                        'result': str(result),
                    },
                    author=agent.name,
                )
                # 由 Runner 统一追加事件
                yield event
            except Exception as e:
                logger.error(f"Tool {call_name} execution failed: {e}")
                event = Event(
                    event_type=EventType.ERROR,
                    content={'tool': call_name, 'error': str(e)},
                    author=agent.name,
                )
                # 由 Runner 统一追加事件
                yield event
