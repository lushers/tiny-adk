"""
Flow 层 - 流程控制层

这一层负责编排 LLM 调用和工具执行的循环，提供：
- BaseFlow: 流程控制的抽象基类
- SimpleFlow: 简单的 Reason-Act 循环实现

设计理念:
- Flow 管理 "思考-行动" 循环
- Flow 负责工具调用的执行
- Flow 不关心具体使用哪个 LLM（由 Model 层提供）
- Flow 不关心会话管理（由 Runner 层管理）
- Flow 在 Agent 初始化时创建（通过 model_post_init）
"""

from .base_flow import BaseFlow
from .simple_flow import SimpleFlow

__all__ = [
    'BaseFlow',
    'SimpleFlow',
]
