"""LLM 抽象基类"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Iterator

from pydantic import BaseModel

from .llm_request import LlmRequest
from .llm_response import LlmResponse


class BaseLlm(BaseModel, ABC):
    """
    LLM 抽象基类（使用 Pydantic）
    
    设计理念（借鉴 Google ADK）：
    - 统一生成器接口：无论流式/非流式，都返回 Iterator/AsyncIterator
    - 通过 stream 参数区分：非流式只 yield 一次，流式 yield 多次
    - 让 Flow 可以统一处理，无需区分流式/非流式
    
    子类需要实现：
    - generate: 同步生成（统一接口）
    - generate_async: 异步生成（统一接口）
    """
    
    model_config = {"arbitrary_types_allowed": True}
    
    # 模型名称
    model: str = ""
    
    @abstractmethod
    def generate(
        self, 
        request: LlmRequest, 
        stream: bool = False,
    ) -> Iterator[LlmResponse]:
        """
        同步生成（统一接口）
        
        Args:
            request: LLM 请求
            stream: 是否流式生成
        
        Yields:
            - stream=False: 只 yield 一次完整响应 (partial=False)
            - stream=True: yield 多个增量响应 (partial=True) + 最后一个完整响应 (partial=False)
        """
        raise NotImplementedError
    
    @abstractmethod
    async def generate_async(
        self, 
        request: LlmRequest, 
        stream: bool = False,
    ) -> AsyncIterator[LlmResponse]:
        """
        异步生成（统一接口）
        
        Args:
            request: LLM 请求
            stream: 是否流式生成
        
        Yields:
            - stream=False: 只 yield 一次完整响应 (partial=False)
            - stream=True: yield 多个增量响应 (partial=True) + 最后一个完整响应 (partial=False)
        """
        raise NotImplementedError
        yield  # 保持为 AsyncIterator
    
    def get_model(self, request: LlmRequest) -> str:
        """获取实际使用的模型名称"""
        return request.model or self.model
    
    @classmethod
    def supported_models(cls) -> list[str]:
        """返回支持的模型列表（可以是正则表达式）"""
        return []
