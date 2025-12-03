"""Agent - AI 代理的核心定义"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .tools import Tool


@dataclass
class Agent:
  """
  Agent - 定义 AI 代理的蓝图
  
  核心设计理念:
  - Agent 是配置，不包含状态
  - Agent 定义了"是谁"、"会什么"、"怎么做"
  - 实际执行由 Runner 负责
  
  这种分离使得:
  1. Agent 可以被复用
  2. 同一个 Agent 可以并发执行多个会话
  3. Agent 配置可以被序列化、版本化、A/B 测试
  """
  name: str
  instruction: str
  model: str = 'gpt-4'
  description: str = ''
  tools: list[Tool] = field(default_factory=list)
  temperature: float = 0.7
  max_tokens: int = 2000
  max_iterations: int = 10  # 默认允许 10 次迭代（工具调用循环）
  
  def to_dict(self) -> dict[str, Any]:
    """转换为字典（用于序列化）"""
    return {
        'name': self.name,
        'instruction': self.instruction,
        'model': self.model,
        'description': self.description,
        'tools': [t.to_function_declaration() for t in self.tools],
        'temperature': self.temperature,
        'max_tokens': self.max_tokens,
        'max_iterations': self.max_iterations,
    }
  
  def get_system_prompt(self) -> str:
    """
    构建系统提示词
    
    将 agent 的身份和能力转换为 LLM 可理解的提示词
    """
    prompt = f"You are {self.name}.\n\n"
    prompt += f"{self.instruction}\n\n"
    
    if self.tools:
      prompt += "You have access to the following tools:\n"
      for tool in self.tools:
        prompt += f"- {tool.name}: {tool.description}\n"
    
    return prompt

