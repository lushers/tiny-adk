"""工具系统 - Agent 可以调用的函数"""

from __future__ import annotations

import asyncio
import inspect
import re
from typing import Any, Callable

from pydantic import BaseModel, Field


class BaseTool(BaseModel):
    """
    工具基类（使用 Pydantic）
    
    核心设计理念: 工具是带有描述的函数，LLM 可以理解并调用
    
    使用 Pydantic 的好处:
    - 配置验证
    - 序列化/反序列化
    - 类型检查
    """
    
    model_config = {"arbitrary_types_allowed": True}
    
    name: str
    description: str
    
    async def run_async(self, args: dict, context: Any = None) -> Any:
        """
        异步执行工具（子类应覆盖此方法）
        
        Args:
            args: 工具参数
            context: 执行上下文（InvocationContext）
            
        Returns:
            工具执行结果
        """
        raise NotImplementedError(f"Tool {self.name} must implement run_async")
    
    def run(self, args: dict, context: Any = None) -> Any:
        """
        同步执行工具（默认调用 run_async）
        
        Args:
            args: 工具参数
            context: 执行上下文
            
        Returns:
            工具执行结果
        """
        return asyncio.run(self.run_async(args, context))
    
    def to_function_declaration(self) -> dict[str, Any]:
        """
        转换为函数声明格式（供 LLM 理解）
        这是与 LLM 交互的关键 - 让 LLM 知道有哪些工具可用
        """
        return {
            'name': self.name,
            'description': self.description,
        }


class Tool(BaseTool):
    """
    函数包装工具
    
    将普通 Python 函数包装为 Tool，自动提取参数信息
    """
    
    func: Callable[..., Any]
    parameters: dict[str, Any] | None = None
    
    def model_post_init(self, __context: Any) -> None:
        """初始化后自动提取参数信息"""
        super().model_post_init(__context)
        if self.parameters is None:
            object.__setattr__(self, 'parameters', self._extract_parameters())
    
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
    
    async def run_async(self, args: dict, context: Any = None) -> Any:
        """异步执行工具函数"""
        if inspect.iscoroutinefunction(self.func):
            return await self.func(**args)
        else:
            return await asyncio.to_thread(self.func, **args)
    
    def run(self, args: dict, context: Any = None) -> Any:
        """同步执行工具函数"""
        return self.func(**args)
    
    def execute(self, **kwargs: Any) -> Any:
        """执行工具函数（兼容旧接口）"""
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


# ==================== 内置工具：Agent 跳转 ====================

class TransferToAgentTool(BaseTool):
    """
    内置工具：跳转到另一个 Agent
    
    让 LLM 可以主动决定将控制权交给其他 Agent。
    这是多 Agent 协作的核心机制。
    
    使用场景：
    - 主 Agent 识别到需要专业处理，跳转到专家 Agent
    - 任务完成后返回给父 Agent
    - 在多个专家 Agent 之间路由
    """
    
    name: str = "transfer_to_agent"
    description: str = "Transfer control to another agent. Use this when the task requires expertise from a different agent."
    available_agents: list[str] = Field(default_factory=list)
    """可以跳转到的 Agent 名称列表"""
    
    # parameters 在 model_post_init 中动态生成
    parameters: dict[str, Any] = Field(default_factory=dict)
    
    def model_post_init(self, __context: Any) -> None:
        """初始化后动态生成 parameters，包含可用 agent 枚举"""
        super().model_post_init(__context)
        
        agent_name_param = {
            'type': 'str',
            'description': 'The name of the agent to transfer control to',
        }
        
        # 如果有可用 agent 列表，添加 enum 约束
        if self.available_agents:
            agent_name_param['enum'] = self.available_agents
        
        self.parameters = {
            'agent_name': agent_name_param,
            'reason': {
                'type': 'str',
                'description': 'Brief reason for the transfer',
                'default': '',
            },
        }
    
    async def run_async(self, args: dict, context: Any = None) -> dict[str, Any]:
        """
        执行 Agent 跳转
        
        返回一个特殊的结果，Flow 会识别并触发跳转
        """
        agent_name = args.get('agent_name', '')
        reason = args.get('reason', '')
        
        if not agent_name:
            return {'error': 'agent_name is required'}
        
        if self.available_agents and agent_name not in self.available_agents:
            return {
                'error': f"Agent '{agent_name}' is not available. Available agents: {self.available_agents}"
            }
        
        # 返回跳转信号
        return {
            'transfer': True,
            'target_agent': agent_name,
            'reason': reason,
        }
    
    def to_function_declaration(self) -> dict[str, Any]:
        """生成函数声明，包含可用 Agent 列表"""
        desc = self.description
        if self.available_agents:
            desc += f"\n\nAvailable agents: {', '.join(self.available_agents)}"
        
        return {
            'name': self.name,
            'description': desc,
            'parameters': self.parameters,
        }


class EscalateTool(BaseTool):
    """
    内置工具：向上级报告/退出循环
    
    用于 LoopAgent 场景，让 Agent 可以主动退出循环。
    """
    
    name: str = "escalate"
    description: str = "Signal that the current task is complete or needs to be escalated to a higher-level agent."
    
    parameters: dict[str, Any] = Field(default_factory=lambda: {
        'reason': {
            'type': 'str',
            'description': 'Reason for escalation or completion status',
            'default': 'Task completed',
        },
    })
    
    async def run_async(self, args: dict, context: Any = None) -> dict[str, Any]:
        """执行 escalate"""
        reason = args.get('reason', 'Task completed')
        return {
            'escalate': True,
            'reason': reason,
        }


def create_transfer_tool(available_agents: list[str]) -> TransferToAgentTool:
    """
    创建 transfer_to_agent 工具
    
    Args:
        available_agents: 可以跳转到的 Agent 名称列表
        
    Returns:
        TransferToAgentTool 实例
    """
    return TransferToAgentTool(available_agents=available_agents)


def create_escalate_tool() -> EscalateTool:
    """创建 escalate 工具"""
    return EscalateTool()
