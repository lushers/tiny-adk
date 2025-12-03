"""
tiny_adk - 简化版 Agent Development Kit

核心组件:
- Agent: 定义 AI agent 的蓝图（Pydantic BaseModel）
- Runner: 无状态执行引擎（绑定 Agent）
- Tool/BaseTool: 可调用的工具函数
- Session: 会话状态管理
- SessionService: Session 持久化服务
- InvocationContext: 单次调用上下文
- Event: 事件系统
- Config: 配置管理

架构:
- Runner: 执行编排（绑定 Agent）
- Flow: Reason-Act 循环 + 工具执行
- Model: LLM 抽象 + 请求/响应格式化

Web 服务:
- 请使用独立的 web 模块: from web import AgentService
"""

from .agents import Agent
from .config import Config, LLMConfig, RunnerConfig, get_config, set_config
from .events import Event, EventType
from .runner import Runner
from .session import Session, SessionService, InvocationContext
from .tools import Tool, BaseTool, tool

# Flow 层
from .flows import BaseFlow, SimpleFlow

# Model 层
from .models import BaseLlm, LlmRequest, LlmResponse, OpenAILlm
from .models import FunctionCall, ToolCall

__all__ = [
    # 核心组件
    'Agent',
    'Config',
    'LLMConfig',
    'RunnerConfig',
    'Event',
    'EventType',
    'Runner',
    'Session',
    'SessionService',
    'InvocationContext',
    'Tool',
    'BaseTool',
    'tool',
    'get_config',
    'set_config',
    
    # Flow 层
    'BaseFlow',
    'SimpleFlow',
    
    # Model 层
    'BaseLlm',
    'LlmRequest',
    'LlmResponse',
    'OpenAILlm',
    'FunctionCall',
    'ToolCall',
]

__version__ = '0.4.0'
