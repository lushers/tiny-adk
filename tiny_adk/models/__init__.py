"""
Model 层 - LLM 抽象层

这一层负责统一不同 LLM 提供商的接口，提供：
- BaseLlm: 所有 LLM 的抽象基类（Pydantic BaseModel）
- LlmRequest: 标准化的请求格式
- LlmResponse: 标准化的响应格式
- FunctionCall: 工具调用信息
- 具体实现: OpenAILlm 等

设计理念:
- 调用方（Flow 层）不需要关心具体是哪个 LLM
- 新增 LLM 提供商只需实现 BaseLlm 接口
- 统一的请求/响应格式便于处理和测试
- 使用 Pydantic 进行配置管理和验证
"""

from .base_llm import BaseLlm
from .llm_request import LlmRequest
from .llm_response import LlmResponse, FunctionCall, ToolCall
from .openai_llm import OpenAILlm

__all__ = [
    'BaseLlm',
    'LlmRequest', 
    'LlmResponse',
    'FunctionCall',
    'ToolCall',  # 向后兼容
    'OpenAILlm',
]
