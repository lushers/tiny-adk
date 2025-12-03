"""OpenAI 兼容的 LLM 实现"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncIterator, Iterator

from .base_llm import BaseLlm
from .llm_request import LlmRequest
from .llm_response import LlmResponse, ToolCall


class ThinkingFilter:
    """流式思考内容过滤器 - 实时过滤 <think> 标签"""
    
    def __init__(self):
        self.buffer = ''
        self.in_thinking = False
        self.thinking_content = ''
        self.clean_content = ''
    
    def process_delta(self, delta: str) -> str:
        """处理流式内容片段，过滤 thinking 内容"""
        self.buffer += delta
        output = ''
        
        while self.buffer:
            if not self.in_thinking:
                think_start_idx = self.buffer.find('<think>')
                
                if think_start_idx == -1:
                    if len(self.buffer) > 10:
                        output_part = self.buffer[:-7]
                        output += output_part
                        self.clean_content += output_part
                        self.buffer = self.buffer[-7:]
                    break
                else:
                    if think_start_idx > 0:
                        output_part = self.buffer[:think_start_idx]
                        output += output_part
                        self.clean_content += output_part
                    self.buffer = self.buffer[think_start_idx + 7:]
                    self.in_thinking = True
            else:
                think_end_idx = self.buffer.find('</think>')
                
                if think_end_idx == -1:
                    if len(self.buffer) > 10:
                        thinking_part = self.buffer[:-8]
                        self.thinking_content += thinking_part
                        self.buffer = self.buffer[-8:]
                    break
                else:
                    self.thinking_content += self.buffer[:think_end_idx]
                    self.buffer = self.buffer[think_end_idx + 8:]
                    self.in_thinking = False
        
        return output
    
    def finalize(self) -> tuple[str, str, str]:
        """返回 (clean_content, thinking_content, remaining_buffer)"""
        remaining = ''
        
        if self.buffer:
            if self.in_thinking:
                self.thinking_content += self.buffer
            else:
                remaining = self.buffer
                self.clean_content += self.buffer
            self.buffer = ''
        
        return self.clean_content.strip(), self.thinking_content.strip(), remaining


class OpenAILlm(BaseLlm):
    """
    OpenAI 兼容的 LLM 实现
    
    支持:
    - OpenAI API
    - vLLM (OpenAI 兼容模式)
    - 其他 OpenAI 兼容的 API
    
    Attributes:
        api_base: API 地址
        api_key: API 密钥
        model: 默认模型名称
        show_thinking: 是否显示思考过程
        show_request: 是否显示 API 请求详情
    """
    
    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str = "",
        show_thinking: bool = False,
        show_request: bool = False,
        client: Any = None,
    ):
        super().__init__(model)
        self.api_base = api_base
        self.api_key = api_key
        self.show_thinking = show_thinking
        self.show_request = show_request
        self._client = client
    
    @property
    def client(self) -> Any:
        """获取 OpenAI 客户端（懒加载）"""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
            )
        return self._client
    
    @classmethod
    def supported_models(cls) -> list[str]:
        return [
            r"gpt-.*",
            r"o1-.*",
            r"chatgpt-.*",
        ]
    
    def generate(self, request: LlmRequest) -> LlmResponse:
        """同步非流式生成"""
        try:
            params = request.to_openai_format()
            params["model"] = self.get_model(request)
            params["stream"] = False
            
            self._log_request(params)
            response = self.client.chat.completions.create(**params)
            return self._parse_response(response)
        except Exception as e:
            return LlmResponse.from_error(str(e))
    
    def generate_stream(self, request: LlmRequest) -> Iterator[LlmResponse]:
        """同步流式生成"""
        try:
            params = request.to_openai_format()
            params["model"] = self.get_model(request)
            params["stream"] = True
            
            self._log_request(params)
            stream = self.client.chat.completions.create(**params)
            yield from self._process_stream(stream)
        except Exception as e:
            yield LlmResponse.from_error(str(e))
    
    async def generate_async(self, request: LlmRequest) -> LlmResponse:
        """异步非流式生成"""
        try:
            params = request.to_openai_format()
            params["model"] = self.get_model(request)
            params["stream"] = False
            
            self._log_request(params)
            # 使用线程池执行同步调用
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                **params
            )
            return self._parse_response(response)
        except Exception as e:
            return LlmResponse.from_error(str(e))
    
    async def generate_stream_async(
        self, request: LlmRequest
    ) -> AsyncIterator[LlmResponse]:
        """异步流式生成"""
        try:
            params = request.to_openai_format()
            params["model"] = self.get_model(request)
            params["stream"] = True
            
            self._log_request(params)
            stream = await asyncio.to_thread(
                self.client.chat.completions.create,
                **params
            )
            
            for response in self._process_stream(stream):
                yield response
                await asyncio.sleep(0)  # 让出控制权
        except Exception as e:
            yield LlmResponse.from_error(str(e))
    
    def _parse_response(self, response: Any) -> LlmResponse:
        """解析 OpenAI 响应"""
        choice = response.choices[0]
        message = choice.message
        
        raw_content = message.content or ""
        clean_content, thinking = self._extract_thinking(raw_content)
        
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))
        
        return LlmResponse(
            content=clean_content,
            tool_calls=tool_calls,
            thinking=thinking,
            raw_content=raw_content,
            finish_reason=choice.finish_reason,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            } if response.usage else {},
        )
    
    def _process_stream(self, stream: Any) -> Iterator[LlmResponse]:
        """处理流式响应"""
        full_content = ""
        tool_calls_data: list[dict[str, Any]] = []
        finish_reason = None
        model_name = None
        chunk_index = 0
        
        thinking_filter = ThinkingFilter()
        
        for chunk in stream:
            if not chunk.choices:
                continue
            
            choice = chunk.choices[0]
            delta = choice.delta
            
            if chunk.model:
                model_name = chunk.model
            
            # 处理内容
            if delta.content:
                full_content += delta.content
                filtered_delta = thinking_filter.process_delta(delta.content)
                
                if self.show_thinking:
                    yield LlmResponse.create_delta(delta.content, chunk_index)
                elif filtered_delta:
                    yield LlmResponse.create_delta(filtered_delta, chunk_index)
                chunk_index += 1
            
            # 处理工具调用
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    tc_index = tc.index
                    tc_id = tc.id
                    tc_name = tc.function.name if tc.function else None
                    tc_args = tc.function.arguments if tc.function else ""
                    
                    if tc_index < len(tool_calls_data):
                        existing = tool_calls_data[tc_index]
                        if tc_args:
                            existing["arguments"] = existing.get("arguments", "") + tc_args
                        if tc_id:
                            existing["id"] = tc_id
                        if tc_name:
                            existing["name"] = tc_name
                    else:
                        tool_calls_data.append({
                            "id": tc_id,
                            "name": tc_name,
                            "arguments": tc_args or "",
                        })
            
            if choice.finish_reason:
                finish_reason = choice.finish_reason
        
        # 完成过滤
        clean_content, thinking, remaining = thinking_filter.finalize()
        
        if not self.show_thinking and remaining:
            yield LlmResponse.create_delta(remaining, chunk_index)
        
        # 解析工具调用
        tool_calls = []
        for tc in tool_calls_data:
            if tc.get("name"):
                try:
                    args = json.loads(tc.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(
                    id=tc.get("id", "call_unknown"),
                    name=tc["name"],
                    arguments=args,
                ))
        
        # 返回最终完整响应
        yield LlmResponse(
            content=clean_content,
            tool_calls=tool_calls,
            thinking=thinking,
            raw_content=full_content,
            finish_reason=finish_reason,
            model=model_name or self.model,
            partial=False,
        )
    
    def _extract_thinking(self, raw_content: str) -> tuple[str, str]:
        """提取并分离思考内容"""
        if not raw_content:
            return "", ""
        
        think_pattern = r'<think>(.*?)</think>'
        thinking_parts = re.findall(think_pattern, raw_content, re.DOTALL)
        clean_content = re.sub(think_pattern, '', raw_content, flags=re.DOTALL).strip()
        thinking_content = '\n'.join(part.strip() for part in thinking_parts) if thinking_parts else ''
        
        return clean_content, thinking_content
    
    def _log_request(self, params: dict[str, Any]) -> None:
        """打印 API 请求详情（调试用）"""
        if not self.show_request:
            return
        
        print("\n" + "=" * 60)
        print("📤 LLM API Request")
        print("=" * 60)
        print(f"🔗 API Base: {self.api_base}")
        print(f"🤖 Model: {params.get('model', 'N/A')}")
        print(f"🌊 Stream: {params.get('stream', False)}")
        
        # 先打印工具定义
        tools = params.get("tools", [])
        if tools:
            print(f"\n🔧 Tools ({len(tools)}):")
            for t in tools:
                func = t.get("function", {})
                print(f"  - {func.get('name', 'unknown')}: {func.get('description', 'N/A')[:50]}...")
        
        # 再打印消息
        messages = params.get("messages", [])
        print(f"\n📝 Messages ({len(messages)}):")
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            
            # 处理 assistant 调用工具的情况
            if role == "assistant" and not content and tool_calls:
                print(f"  [{i+1}] {role}: [调用工具]")
                for tc in tool_calls:
                    func = tc.get("function", {})
                    tc_name = func.get("name", "?")
                    tc_args = func.get("arguments", "{}")
                    # 如果 arguments 是字符串，尝试解析为更友好的格式
                    if isinstance(tc_args, str) and len(tc_args) > 100:
                        tc_args = tc_args[:100] + "..."
                    print(f"        → {tc_name}({tc_args})")
            # 处理 tool 响应消息
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                if isinstance(content, str) and len(content) > 100:
                    content = content[:100] + "..."
                print(f"  [{i+1}] {role} ({tool_call_id[:12]}...): {content}")
            else:
                # 截断过长的内容
                if isinstance(content, str) and len(content) > 200:
                    content = content[:200] + "..."
                print(f"  [{i+1}] {role}: {content}")
        
        print("=" * 60 + "\n")

