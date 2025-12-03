"""工具系统 - Agent 可以调用的函数"""

from __future__ import annotations

import inspect
import re
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
    """从函数签名和 docstring 提取参数定义"""
    sig = inspect.signature(self.func)
    docstring = inspect.getdoc(self.func) or ''
    
    # 解析 docstring 中的参数描述（支持 Google 风格）
    param_descriptions = self._parse_docstring_params(docstring)
    
    params = {}
    
    for name, param in sig.parameters.items():
      param_info = {'name': name}
      
      # 提取类型注解
      if param.annotation != inspect.Parameter.empty:
        param_info['type'] = param.annotation.__name__
      
      # 提取默认值
      if param.default != inspect.Parameter.empty:
        param_info['default'] = param.default
      
      # 从 docstring 获取参数描述
      if name in param_descriptions:
        param_info['description'] = param_descriptions[name]
      
      params[name] = param_info
    
    return params
  
  def _parse_docstring_params(self, docstring: str) -> dict[str, str]:
    """
    解析 docstring 中的参数描述
    
    支持 Google 风格:
      Args:
        city: 城市名称
        date: 查询日期
    
    和简单风格:
      :param city: 城市名称
    """
    descriptions = {}
    
    if not docstring:
      return descriptions
    
    # Google 风格: Args: 部分
    args_match = re.search(r'Args?:\s*\n((?:\s+\w+.*\n?)+)', docstring, re.IGNORECASE)
    if args_match:
      args_section = args_match.group(1)
      # 匹配 "param_name: description" 或 "param_name (type): description"
      for match in re.finditer(r'^\s+(\w+)(?:\s*\([^)]*\))?:\s*(.+?)(?=\n\s+\w+|\n\n|\Z)', 
                                args_section, re.MULTILINE | re.DOTALL):
        param_name = match.group(1)
        description = match.group(2).strip().replace('\n', ' ')
        descriptions[param_name] = description
    
    # Sphinx 风格: :param name: description
    for match in re.finditer(r':param\s+(\w+):\s*(.+?)(?=:|$)', docstring, re.MULTILINE):
      param_name = match.group(1)
      description = match.group(2).strip()
      descriptions[param_name] = description
    
    return descriptions
  
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

