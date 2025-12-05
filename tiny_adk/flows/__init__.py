"""
Flow 层 - 流程控制层

这一层负责编排 LLM 调用和工具执行的循环，提供：
- BaseFlow: 流程控制的抽象基类
- SimpleFlow: 简单的 Reason-Act 循环实现
- RequestProcessor: 请求处理器协议
- ResponseProcessor: 响应处理器协议

设计理念:
- Flow 管理 "思考-行动" 循环
- Flow 负责工具调用的执行
- Flow 不关心具体使用哪个 LLM（由 Model 层提供）
- Flow 不关心会话管理（由 Runner 层管理）
- Flow 在 Agent 初始化时创建（通过 model_post_init）

扩展机制:
- request_processors: 请求处理器链（在 LLM 调用前执行）
- response_processors: 响应处理器链（在 LLM 调用后执行）
- before_model_callback / after_model_callback: 模型调用回调
- before_tool_callback / after_tool_callback: 工具调用回调
"""

from .base_flow import BaseFlow, RequestProcessor, ResponseProcessor
from .simple_flow import SimpleFlow

__all__ = [
    'BaseFlow',
    'SimpleFlow',
    'RequestProcessor',
    'ResponseProcessor',
]
