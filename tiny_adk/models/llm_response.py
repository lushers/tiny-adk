"""LLM 响应的标准化格式"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """工具调用信息"""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LlmResponse:
    """
    标准化的 LLM 响应格式
    
    这个类统一了不同 LLM 的响应格式，使 Flow 层可以用一致的方式处理响应。
    
    Attributes:
        content: 文本内容
        tool_calls: 工具调用列表
        thinking: 思考过程（如果模型支持）
        raw_content: 原始未处理的内容
        finish_reason: 完成原因 (stop, tool_calls, length, etc.)
        model: 实际使用的模型名称
        usage: token 使用统计
        error: 错误信息（如果有）
        partial: 是否是流式响应的部分内容
        delta: 流式响应中的增量内容
    """
    
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str = ""
    raw_content: str = ""
    finish_reason: str | None = None
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    
    # 流式响应相关
    partial: bool = False
    delta: str = ""
    
    # 元数据
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def has_tool_calls(self) -> bool:
        """是否包含工具调用"""
        return len(self.tool_calls) > 0
    
    def is_error(self) -> bool:
        """是否是错误响应"""
        return self.error is not None
    
    def is_complete(self) -> bool:
        """是否是完整响应（非流式部分）"""
        return not self.partial
    
    @classmethod
    def from_error(cls, error: str) -> LlmResponse:
        """从错误创建响应"""
        return cls(error=error, content=f"LLM 调用失败: {error}")
    
    @classmethod
    def create_delta(cls, delta: str, chunk_index: int = 0) -> LlmResponse:
        """创建流式增量响应"""
        return cls(
            partial=True,
            delta=delta,
            metadata={"chunk_index": chunk_index},
        )

