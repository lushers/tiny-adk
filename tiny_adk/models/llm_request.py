"""LLM 请求的标准化格式"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LlmRequest:
    """
    标准化的 LLM 请求格式
    
    这个类将 Agent 的配置和消息历史转换为 LLM 可以理解的格式。
    不同的 LLM 实现会将这个统一格式转换为各自的 API 格式。
    
    Attributes:
        model: 模型名称
        messages: 消息列表 (OpenAI 格式: role, content)
        tools: 工具定义列表 (OpenAI function calling 格式)
        temperature: 温度参数
        max_tokens: 最大 token 数
        stream: 是否启用流式输出
    """
    
    model: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = False
    
    # 扩展配置（不同 LLM 可能有特定配置）
    extra_config: dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str) -> None:
        """添加一条消息"""
        self.messages.append({"role": role, "content": content})
    
    def add_tool_call_message(
        self,
        role: str,
        content: str | None,
        tool_calls: list[dict[str, Any]],
    ) -> None:
        """添加包含工具调用的 assistant 消息"""
        msg = {"role": role, "content": content, "tool_calls": tool_calls}
        self.messages.append(msg)
    
    def add_tool_response_message(
        self,
        tool_call_id: str,
        name: str,
        content: str,
    ) -> None:
        """添加工具响应消息"""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content,
        })
    
    def to_openai_format(self) -> dict[str, Any]:
        """转换为 OpenAI API 格式"""
        params = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
        }
        
        if self.tools:
            params["tools"] = self.tools
            params["tool_choice"] = "auto"
        
        params.update(self.extra_config)
        return params

