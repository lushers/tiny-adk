"""Flow 抽象基类"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator

if TYPE_CHECKING:
    from ..agents import Agent
    from ..events import Event
    from ..models import BaseLlm, LlmRequest
    from ..session import Session, InvocationContext
    from ..tools import BaseTool, Tool

logger = logging.getLogger(__name__)


class BaseFlow(ABC):
    """
    Flow 抽象基类
    
    Flow 负责编排 LLM 调用循环（Reason-Act Loop）：
    1. 构建 LLM 请求
    2. 调用 LLM
    3. 处理响应（执行工具调用或返回最终结果）
    4. 如果有工具调用，重复步骤 1-3
    
    设计理念:
    - Flow 是无状态的，所有状态在 Session 中
    - Flow 不管理会话，只执行单次完整的交互循环
    - Flow 可以被不同的 Runner 复用
    - Flow 在 Agent 初始化时创建（通过 model_post_init）
    
    API 设计（统一接口）:
    - run(stream: bool) -> Iterator[Event]: 同步执行
    - run_async(stream: bool) -> AsyncIterator[Event]: 异步执行
    - stream=False: 非流式，只 yield 一个 MODEL_RESPONSE 事件
    - stream=True: 流式，yield 多个 MODEL_RESPONSE_DELTA + 最后一个 MODEL_RESPONSE
    """
    
    def __init__(self, max_iterations: int = 10):
        """
        初始化 Flow
        
        Args:
            max_iterations: 最大循环次数，防止无限循环
        """
        self.max_iterations = max_iterations
        logger.debug(f"[{self.__class__.__name__}] Created with max_iterations={max_iterations}")
    
    # ==================== 核心抽象方法 ====================
    
    @abstractmethod
    def run(
        self,
        agent: Agent,
        session: Session,
        llm: BaseLlm,
        stream: bool = False,
    ) -> Iterator[Event]:
        """
        同步执行
        
        Args:
            agent: 要执行的 Agent
            session: 会话对象
            llm: LLM 实例
            stream: 是否流式生成（传递给 LLM）
                - False: 只 yield 一个 MODEL_RESPONSE 事件
                - True: yield 多个 MODEL_RESPONSE_DELTA + 最后一个 MODEL_RESPONSE
        
        Yields:
            执行过程中的事件
        """
        pass
    
    @abstractmethod
    async def run_async(
        self,
        agent: Agent,
        session: Session,
        llm: BaseLlm,
        stream: bool = False,
    ) -> AsyncIterator[Event]:
        """
        异步执行
        
        Args:
            agent: 要执行的 Agent
            session: 会话对象
            llm: LLM 实例
            stream: 是否流式生成（传递给 LLM）
                - False: 只 yield 一个 MODEL_RESPONSE 事件
                - True: yield 多个 MODEL_RESPONSE_DELTA + 最后一个 MODEL_RESPONSE
        
        Yields:
            执行过程中的事件
        """
        pass
    
    # ==================== 辅助方法 ====================
    
    def build_request(
        self,
        agent: Agent,
        session: Session,
    ) -> LlmRequest:
        """
        构建 LLM 请求
        
        子类可以覆盖此方法来自定义请求构建逻辑
        """
        from ..models import LlmRequest
        
        request = LlmRequest(
            model=agent.get_model_name() if hasattr(agent, 'get_model_name') else agent.model,
            temperature=agent.temperature,
            max_tokens=agent.max_tokens,
        )
        
        # 添加系统提示
        request.add_message("system", agent.get_system_prompt())
        
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
        
        # 添加工具定义
        if agent.tools:
            request.tools = [self._tool_to_openai_format(tool) for tool in agent.tools]
        
        return request
    
    def find_tool(self, agent: Agent, tool_name: str) -> BaseTool | Tool | None:
        """查找工具"""
        for tool in agent.tools:
            if tool.name == tool_name:
                return tool
        return None
    
    def _tool_to_openai_format(self, tool: BaseTool | Tool) -> dict[str, Any]:
        """将 Tool 转换为 OpenAI function calling 格式"""
        properties = {}
        required = []
        
        # 获取参数信息（Tool 类有 parameters 属性）
        parameters = getattr(tool, 'parameters', None) or {}
        
        for param_name, param_info in parameters.items():
            param_type = param_info.get('type', 'string')
            
            type_mapping = {
                'str': 'string',
                'int': 'integer',
                'float': 'number',
                'bool': 'boolean',
                'list': 'array',
                'dict': 'object',
            }
            json_type = type_mapping.get(param_type, 'string')
            
            prop = {
                'type': json_type,
                'description': param_info.get('description', f'参数 {param_name}'),
            }
            
            # 如果有枚举值（如 available_agents），添加 enum
            if 'enum' in param_info:
                prop['enum'] = param_info['enum']
            
            properties[param_name] = prop
            
            if 'default' not in param_info:
                required.append(param_name)
        
        # 使用工具的 to_function_declaration 获取描述（可能包含额外信息）
        if hasattr(tool, 'to_function_declaration'):
            decl = tool.to_function_declaration()
            description = decl.get('description', tool.description)
        else:
            description = tool.description
        
        return {
            'type': 'function',
            'function': {
                'name': tool.name,
                'description': description,
                'parameters': {
                    'type': 'object',
                    'properties': properties,
                    'required': required,
                },
            },
        }
