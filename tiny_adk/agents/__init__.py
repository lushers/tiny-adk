"""agents 模块 - Agent 定义"""

from .base_agent import BaseAgent, BeforeAgentCallback, AfterAgentCallback
from .llm_agent import LlmAgent, Agent
from .sequential_agent import SequentialAgent
from .loop_agent import LoopAgent

__all__ = [
    # 基类
    'BaseAgent',
    # LLM Agent
    'LlmAgent',
    'Agent',  # LlmAgent 的别名
    # 编排 Agent
    'SequentialAgent',
    'LoopAgent',
    # 回调类型
    'BeforeAgentCallback',
    'AfterAgentCallback',
]

