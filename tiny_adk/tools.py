"""工具系统 - Agent 可以调用的函数"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
  """
  工具 - 封装可被 agent 调用的函数
  
  核心设计理念: 工具是带有描述的函数，LLM 可以理解并调用
  """
  name: str
  description: str
  func: Callable[..., Any]
  parameters: dict[str, Any] | None = None
  
  def __post_init__(self):
    """自动从函数签名提取参数信息"""
    if self.parameters is None:
      self.parameters = self._extract_parameters()
  
  def _extract_parameters(self) -> dict[str, Any]:
    """从函数签名提取参数定义"""
    sig = inspect.signature(self.func)
    params = {}
    
    for name, param in sig.parameters.items():
      param_info = {'name': name}
      
      # 提取类型注解
      if param.annotation != inspect.Parameter.empty:
        param_info['type'] = param.annotation.__name__
      
      # 提取默认值
      if param.default != inspect.Parameter.empty:
        param_info['default'] = param.default
      
      params[name] = param_info
    
    return params
  
  def execute(self, **kwargs: Any) -> Any:
    """执行工具函数"""
    return self.func(**kwargs)
  
  def to_function_declaration(self) -> dict[str, Any]:
    """
    转换为函数声明格式（供 LLM 理解）
    这是与 LLM 交互的关键 - 让 LLM 知道有哪些工具可用
    """
    return {
        'name': self.name,
        'description': self.description,
        'parameters': self.parameters or {},
    }


def tool(name: str | None = None, description: str | None = None):
  """
  装饰器 - 将普通函数转换为 Tool
  
  用法:
    @tool(description="搜索网页")
    def search(query: str) -> str:
      return f"搜索结果: {query}"
  """
  def decorator(func: Callable[..., Any]) -> Tool:
    tool_name = name or func.__name__
    tool_desc = description or func.__doc__ or f"Function {func.__name__}"
    return Tool(name=tool_name, description=tool_desc, func=func)
  
  return decorator

