"""LLM 响应的标准化格式"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FunctionCall:
    """
    工具/函数调用信息
    
    统一使用 FunctionCall 命名，与新架构保持一致
    """
    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    
    # 兼容旧的 arguments 字段
    @property
    def arguments(self) -> dict[str, Any]:
        """兼容旧代码使用 arguments 字段"""
        return self.args
    
    @arguments.setter
    def arguments(self, value: dict[str, Any]) -> None:
        self.args = value


# 向后兼容：保留 ToolCall 作为别名
ToolCall = FunctionCall


@dataclass
class LlmResponse:
    """
    标准化的 LLM 响应格式
    
    这个类统一了不同 LLM 的响应格式，使 Flow 层可以用一致的方式处理响应。
    
    Attributes:
        content: 文本内容
        function_calls: 工具调用列表
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
    function_calls: list[FunctionCall] = field(default_factory=list)
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
    
    # 向后兼容：tool_calls 属性
    @property
    def tool_calls(self) -> list[FunctionCall]:
        """兼容旧代码使用 tool_calls 字段"""
        return self.function_calls
    
    @tool_calls.setter
    def tool_calls(self, value: list[FunctionCall]) -> None:
        self.function_calls = value
    
    def has_function_calls(self) -> bool:
        """是否包含工具调用"""
        return len(self.function_calls) > 0
    
    def has_tool_calls(self) -> bool:
        """是否包含工具调用（兼容旧命名）"""
        return self.has_function_calls()
    
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
