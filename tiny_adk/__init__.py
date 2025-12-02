"""
Simplified ADK - 保留核心概念的简化版本

核心组件:
- Agent: 定义 AI agent 的蓝图
- Runner: 执行引擎
- Tool: 可调用的工具函数
- Session: 会话状态管理
- Event: 事件系统
- Config: 配置管理
"""

from .agents import Agent
from .config import Config, LLMConfig, RunnerConfig, get_config, set_config
from .events import Event, EventType
from .runner import Runner
from .session import Session
from .tools import Tool, tool

__all__ = [
    'Agent',
    'Config',
    'LLMConfig',
    'RunnerConfig',
    'Event',
    'EventType',
    'Runner',
    'Session',
    'Tool',
    'tool',
    'get_config',
    'set_config',
]

__version__ = '0.1.0-simplified'

