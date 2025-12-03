"""
tiny_adk - 简化版 Agent Development Kit

核心组件:
- Agent: 定义 AI agent 的蓝图
- Runner: 执行引擎
- Tool: 可调用的工具函数
- Session: 会话状态管理
- Event: 事件系统
- Config: 配置管理

三层架构:
- Runner: 会话管理 + 编排
- Flow: Reason-Act 循环 + 工具执行
- Model: LLM 抽象 + 请求/响应格式化
"""

from .agents import Agent
from .config import Config, LLMConfig, RunnerConfig, get_config, set_config
from .events import Event, EventType
from .runner import Runner, InMemoryRunner
from .session import Session
from .tools import Tool, tool

# Flow 层
from .flows import BaseFlow, SimpleFlow

# Model 层
from .models import BaseLlm, LlmRequest, LlmResponse, OpenAILlm

__all__ = [
    # 核心组件
    'Agent',
    'Config',
    'LLMConfig',
    'RunnerConfig',
    'Event',
    'EventType',
    'Runner',
    'InMemoryRunner',
    'Session',
    'Tool',
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
]

__version__ = '0.2.0'
