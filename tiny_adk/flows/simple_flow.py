"""简单的 Reason-Act 循环实现（支持多 Agent + Memory）"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, Optional

from .base_flow import BaseFlow

if TYPE_CHECKING:
    from ..agents import Agent
    from ..models import BaseLlm
    from ..session import Session
    from ..memory import BaseMemoryService

from ..events import Event, EventType, EventActions

logger = logging.getLogger(__name__)


# 检查 Memory 模块是否可用
try:
    from ..memory import PreloadMemoryTool, MemoryToolContext
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    PreloadMemoryTool = None
    MemoryToolContext = None


class SimpleFlow(BaseFlow):
    """
    简单的 Reason-Act 循环实现（支持 Memory）
    
    设计理念（借鉴 Google ADK）：
    - 统一接口：run(stream) / run_async(stream)，通过 stream 参数控制流式/非流式
    - LLM 返回生成器，Flow 负责 LlmResponse -> Event 转换
    - 所有事件只 yield，由 Runner 负责持久化
    
    Memory 支持：
    - 如果 Agent 包含 PreloadMemoryTool，会在 LLM 请求前自动注入相关记忆
    - 如果 Agent 包含 LoadMemoryTool，模型可以主动调用搜索记忆
    - 通过 memory_context 传递 Memory 服务和用户信息
    
    关于 stream 参数：
    - stream=False: LLM 使用非流式 API，只 yield 一个 MODEL_RESPONSE
    - stream=True: LLM 使用流式 API，yield 多个 MODEL_RESPONSE_DELTA + 最后一个 MODEL_RESPONSE
    """
    
    # Memory 上下文（运行时设置）
    _memory_context: Optional[Any] = None
    
    def set_memory_context(
        self,
        memory_service: Optional['BaseMemoryService'] = None,
        app_name: str = "",
        user_id: str = "",
        session_id: Optional[str] = None,
        user_query: str = "",
    ) -> None:
        """
        设置 Memory 上下文
        
        在 Runner.run_async 调用 Flow 之前设置，用于 preload_memory 和 load_memory
        """
        if MEMORY_AVAILABLE and memory_service:
            self._memory_context = MemoryToolContext(
                memory_service=memory_service,
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                user_query=user_query,
            )
        else:
            self._memory_context = None
    
    # ==================== Memory 预处理 ====================
    
    def _get_preload_memory_text(self, agent: 'Agent') -> Optional[str]:
        """
        获取 preload_memory 的预加载文本
        
        如果 Agent 包含 PreloadMemoryTool，调用其 process_llm_request 获取记忆文本
        """
        if not MEMORY_AVAILABLE or not self._memory_context:
            return None
        
        for tool in agent.tools:
            if isinstance(tool, PreloadMemoryTool):
                try:
                    # 同步版本
                    return tool.process_llm_request_sync(self._memory_context)
                except Exception as e:
                    logger.warning(f"Preload memory failed: {e}")
                    return None
        
        return None
    
    async def _get_preload_memory_text_async(self, agent: 'Agent') -> Optional[str]:
        """异步版本的预加载"""
        if not MEMORY_AVAILABLE or not self._memory_context:
            return None
        
        for tool in agent.tools:
            if isinstance(tool, PreloadMemoryTool):
                try:
                    return await tool.process_llm_request(self._memory_context)
                except Exception as e:
                    logger.warning(f"Preload memory failed: {e}")
                    return None
        
        return None
    
    def build_request(
        self,
        agent: 'Agent',
        session: 'Session',
    ):
        """
        构建 LLM 请求（覆盖基类方法，添加 Memory 支持）
        
        如果 Agent 包含 PreloadMemoryTool，会自动注入相关记忆到系统指令
        """
        from ..models import LlmRequest
        
        request = LlmRequest(
            model=agent.get_model_name() if hasattr(agent, 'get_model_name') else agent.model,
            temperature=agent.temperature,
            max_tokens=agent.max_tokens,
        )
        
        # 获取系统提示
        system_prompt = agent.get_system_prompt()
        
        # 预加载记忆（如果有 PreloadMemoryTool）
        preload_text = self._get_preload_memory_text(agent)
        if preload_text:
            system_prompt = system_prompt + "\n\n" + preload_text
            logger.debug("[SimpleFlow] Injected preloaded memory into system prompt")
        
        # 添加系统提示
        request.add_message("system", system_prompt)
        
        # 添加历史消息
        for msg in session.get_conversation_history():
            if msg.get("tool_calls"):
                request.add_tool_call_message(
                    role=msg["role"],
                    content=msg.get("content"),
                    tool_calls=msg["tool_calls"],
                )
            elif msg.get("role") == "tool":
                request.add_tool_response_message(
                    tool_call_id=msg.get("tool_call_id", ""),
                    name=msg.get("name", ""),
                    content=msg.get("content", ""),
                )
            else:
                request.messages.append(msg)
        
        # 添加工具定义（排除 PreloadMemoryTool，它不暴露给模型）
        if agent.tools:
            tools_for_llm = []
            for tool in agent.tools:
                # 跳过 PreloadMemoryTool（它有 is_preprocessing 标记）
                if getattr(tool, 'is_preprocessing', False):
                    continue
                tools_for_llm.append(self._tool_to_openai_format(tool))
            
            if tools_for_llm:
                request.tools = tools_for_llm
        
        return request
    
    async def build_request_async(
        self,
        agent: 'Agent',
        session: 'Session',
    ):
        """异步版本的构建请求"""
        from ..models import LlmRequest
        
        request = LlmRequest(
            model=agent.get_model_name() if hasattr(agent, 'get_model_name') else agent.model,
            temperature=agent.temperature,
            max_tokens=agent.max_tokens,
        )
        
        system_prompt = agent.get_system_prompt()
        
        # 异步预加载记忆
        preload_text = await self._get_preload_memory_text_async(agent)
        if preload_text:
            system_prompt = system_prompt + "\n\n" + preload_text
            logger.debug("[SimpleFlow] Injected preloaded memory into system prompt")
        
        request.add_message("system", system_prompt)
        
        for msg in session.get_conversation_history():
            if msg.get("tool_calls"):
                request.add_tool_call_message(
                    role=msg["role"],
                    content=msg.get("content"),
                    tool_calls=msg["tool_calls"],
                )
            elif msg.get("role") == "tool":
                request.add_tool_response_message(
                    tool_call_id=msg.get("tool_call_id", ""),
                    name=msg.get("name", ""),
                    content=msg.get("content", ""),
                )
            else:
                request.messages.append(msg)
        
        if agent.tools:
            tools_for_llm = []
            for tool in agent.tools:
                if getattr(tool, 'is_preprocessing', False):
                    continue
                tools_for_llm.append(self._tool_to_openai_format(tool))
            
            if tools_for_llm:
                request.tools = tools_for_llm
        
        return request
    
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
        
        # 构建请求（使用异步版本以支持 preload memory）
        request = await self.build_request_async(agent, session)
        
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
                # 传递 memory_context 给工具（如果有）
                context = self._memory_context
                
                if hasattr(tool, 'run'):
                    result = tool.run(call_args, context)
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
        """异步执行工具（支持 Agent 跳转 + Memory）"""
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
                # 传递 memory_context 给工具（如果有）
                context = self._memory_context
                
                if hasattr(tool, 'run_async'):
                    result = await tool.run_async(call_args, context)
                elif hasattr(tool, 'func') and inspect.iscoroutinefunction(tool.func):
                    result = await tool.func(**call_args)
                elif hasattr(tool, 'execute'):
                    result = await asyncio.to_thread(tool.execute, **call_args)
                else:
                    result = await asyncio.to_thread(tool.run, call_args, context)
                
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
