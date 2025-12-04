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
    
    设计理念（借鉴 Google ADK）：
    - 统一接口：run(stream) / run_async(stream)，通过 stream 参数控制流式/非流式
    - LLM 返回生成器，Flow 负责 LlmResponse -> Event 转换
    - 所有事件只 yield，由 Runner 负责持久化
    
    关于 stream 参数：
    - stream=False: LLM 使用非流式 API，只 yield 一个 MODEL_RESPONSE
    - stream=True: LLM 使用流式 API，yield 多个 MODEL_RESPONSE_DELTA + 最后一个 MODEL_RESPONSE
    """
    
    # ==================== 同步执行 ====================
    
    def run(
        self,
        agent: 'Agent',
        session: 'Session',
        llm: 'BaseLlm',
        stream: bool = False,
    ) -> Iterator[Event]:
        """
        同步执行
        
        Args:
            stream: 是否流式生成（传递给 LLM）
        """
        yield from self._reason_act_loop(agent, session, llm, stream=stream, iteration=0)
    
    def _reason_act_loop(
        self,
        agent: 'Agent',
        session: 'Session',
        llm: 'BaseLlm',
        stream: bool,
        iteration: int,
    ) -> Iterator[Event]:
        """同步 Reason-Act 循环"""
        if iteration >= self.max_iterations:
            yield Event(
                event_type=EventType.ERROR,
                content={'error': f"达到最大迭代次数限制 ({self.max_iterations})"},
            )
            return
        
        logger.debug(f"[SimpleFlow] Iteration {iteration + 1}")
        
        # 构建请求
        request = self.build_request(agent, session)
        
        # 调用 LLM（统一生成器接口）
        final_response = None
        for response in llm.generate(request, stream=stream):
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
                yield Event(
                    event_type=EventType.MODEL_RESPONSE,
                    content=response.content,
                    author=agent.name,
                    metadata={
                        'model': response.model,
                        'thinking': response.thinking,
                        'finish_reason': response.finish_reason,
                    },
                )
        
        # 处理工具调用
        if final_response and final_response.has_function_calls():
            for fc in final_response.function_calls:
                yield from self._execute_tool(agent, session, fc)
            yield from self._reason_act_loop(agent, session, llm, stream, iteration + 1)
    
    # ==================== 异步执行 ====================
    
    async def run_async(
        self,
        agent: 'Agent',
        session: 'Session',
        llm: 'BaseLlm',
        stream: bool = False,
    ) -> AsyncIterator[Event]:
        """
        异步执行
        
        Args:
            stream: 是否流式生成（传递给 LLM）
        """
        async for event in self._reason_act_loop_async(agent, session, llm, stream=stream, iteration=0):
            yield event
    
    async def _reason_act_loop_async(
        self,
        agent: 'Agent',
        session: 'Session',
        llm: 'BaseLlm',
        stream: bool,
        iteration: int,
    ) -> AsyncIterator[Event]:
        """异步 Reason-Act 循环"""
        if iteration >= self.max_iterations:
            yield Event(
                event_type=EventType.ERROR,
                content={'error': f"达到最大迭代次数限制 ({self.max_iterations})"},
            )
            return
        
        logger.debug(f"[SimpleFlow] Async iteration {iteration + 1}")
        
        # 构建请求
        request = self.build_request(agent, session)
        
        # 调用 LLM（统一生成器接口）
        final_response = None
        async for response in llm.generate_async(request, stream=stream):
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
                yield Event(
                    event_type=EventType.MODEL_RESPONSE,
                    content=response.content,
                    author=agent.name,
                    metadata={
                        'model': response.model,
                        'thinking': response.thinking,
                        'finish_reason': response.finish_reason,
                    },
                )
        
        # 处理工具调用
        if final_response and final_response.has_function_calls():
            for fc in final_response.function_calls:
                async for event in self._execute_tool_async(agent, session, fc):
                    yield event
            async for event in self._reason_act_loop_async(agent, session, llm, stream, iteration + 1):
                yield event
    
    # ==================== 工具执行（只 yield 事件）====================
    
    def _execute_tool(
        self, 
        agent: 'Agent', 
        session: 'Session', 
        function_call,
    ) -> Iterator[Event]:
        """同步执行工具"""
        call_id = function_call.id
        call_name = function_call.name
        call_args = function_call.args if hasattr(function_call, 'args') else function_call.arguments
        
        # yield 工具调用事件
        yield Event(
            event_type=EventType.TOOL_CALL,
            content={
                'id': call_id,
                'name': call_name,
                'arguments': call_args,
            },
            author=agent.name,
        )
        
        tool = self.find_tool(agent, call_name)
        if tool:
            try:
                if hasattr(tool, 'run'):
                    result = tool.run(call_args, None)
                else:
                    result = tool.execute(**call_args)
                
                yield Event(
                    event_type=EventType.TOOL_RESPONSE,
                    content={
                        'call_id': call_id,
                        'name': call_name,
                        'result': str(result),
                    },
                )
            except Exception as e:
                logger.error(f"Tool {call_name} execution failed: {e}")
                yield Event(
                    event_type=EventType.ERROR,
                    content={'tool': call_name, 'error': str(e)},
                )
    
    async def _execute_tool_async(
        self, 
        agent: 'Agent', 
        session: 'Session', 
        function_call,
    ) -> AsyncIterator[Event]:
        """异步执行工具（支持 Agent 跳转）"""
        call_id = function_call.id
        call_name = function_call.name
        call_args = function_call.args if hasattr(function_call, 'args') else function_call.arguments
        
        # yield 工具调用事件
        yield Event(
            event_type=EventType.TOOL_CALL,
            content={
                'id': call_id,
                'name': call_name,
                'arguments': call_args,
            },
            author=agent.name,
        )
        
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
                if call_name == 'transfer_to_agent':
                    target_agent_name = call_args.get('agent_name')
                    yield Event(
                        event_type=EventType.AGENT_TRANSFER,
                        content={
                            'call_id': call_id,
                            'name': call_name,
                            'result': str(result),
                            'target_agent': target_agent_name,
                        },
                        metadata={'transfer_to_agent': target_agent_name},
                    )
                # 检查是否是 escalate
                elif call_name == 'escalate':
                    yield Event(
                        event_type=EventType.TOOL_RESPONSE,
                        content={
                            'call_id': call_id,
                            'name': call_name,
                            'result': str(result),
                        },
                        metadata={'escalate': True},
                    )
                else:
                    yield Event(
                        event_type=EventType.TOOL_RESPONSE,
                        content={
                            'call_id': call_id,
                            'name': call_name,
                            'result': str(result),
                        },
                    )
            except Exception as e:
                logger.error(f"Tool {call_name} execution failed: {e}")
                yield Event(
                    event_type=EventType.ERROR,
                    content={'tool': call_name, 'error': str(e)},
                )
