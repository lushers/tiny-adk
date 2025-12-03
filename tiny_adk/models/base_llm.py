"""LLM 抽象基类"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator

from .llm_request import LlmRequest
from .llm_response import LlmResponse


class BaseLlm(ABC):
    """
    LLM 抽象基类
    
    所有 LLM 实现都需要继承这个类，并实现以下方法：
    - generate: 同步生成（非流式）
    - generate_stream: 同步流式生成
    - generate_async: 异步生成（非流式）
    - generate_stream_async: 异步流式生成
    
    设计理念:
    - 统一接口：不同 LLM 提供商有相同的调用方式
    - 请求/响应标准化：使用 LlmRequest 和 LlmResponse
    - 支持同步/异步、流式/非流式四种模式
    """
    
    def __init__(self, model: str = ""):
        """
        初始化 LLM
        
        Args:
            model: 模型名称（如果不在请求中指定，则使用这个默认值）
        """
        self.model = model
    
    @abstractmethod
    def generate(self, request: LlmRequest) -> LlmResponse:
        """
        同步生成（非流式）
        
        Args:
            request: LLM 请求
        
        Returns:
            完整的 LLM 响应
        """
        pass
    
    @abstractmethod
    def generate_stream(self, request: LlmRequest) -> Iterator[LlmResponse]:
        """
        同步流式生成
        
        Args:
            request: LLM 请求
        
        Yields:
            流式响应片段，最后一个响应的 partial=False 表示完成
        """
        pass
    
    @abstractmethod
    async def generate_async(self, request: LlmRequest) -> LlmResponse:
        """
        异步生成（非流式）
        
        Args:
            request: LLM 请求
        
        Returns:
            完整的 LLM 响应
        """
        pass
    
    @abstractmethod
    async def generate_stream_async(
        self, request: LlmRequest
    ) -> AsyncIterator[LlmResponse]:
        """
        异步流式生成
        
        Args:
            request: LLM 请求
        
        Yields:
            流式响应片段，最后一个响应的 partial=False 表示完成
        """
        pass
    
    def get_model(self, request: LlmRequest) -> str:
        """获取实际使用的模型名称"""
        return request.model or self.model
    
    @classmethod
    def supported_models(cls) -> list[str]:
        """
        返回支持的模型列表（可以是正则表达式）
        
        子类可以覆盖这个方法来声明支持的模型
        """
        return []

