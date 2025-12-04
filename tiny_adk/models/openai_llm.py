"""OpenAI 兼容的 LLM 实现"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncIterator, Iterator

from .base_llm import BaseLlm
from .llm_request import LlmRequest
from .llm_response import LlmResponse, FunctionCall


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
    OpenAI 兼容的 LLM 实现（使用 Pydantic）
    
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
    
    api_base: str | None = None
    api_key: str | None = None
    show_thinking: bool = False
    show_request: bool = False
    log_level: str = "normal"  # minimal | normal | verbose
    
    # 私有字段（不在 Pydantic 模式中）
    _client: Any = None
    
    def model_post_init(self, __context: Any) -> None:
        """Pydantic 初始化完成后的钩子"""
        super().model_post_init(__context)
        self._client = None
    
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
            result = self._parse_response(response)
            self._log_response(result)
            return result
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
            result = self._parse_response(response)
            self._log_response(result)
            return result
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
        
        function_calls = []
        
        # 1. 标准 OpenAI 格式的 tool_calls
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                function_calls.append(FunctionCall(
                    id=tc.id,
                    name=tc.function.name,
                    args=args,
                ))
        
        # 2. MiniMax XML 格式的工具调用（在 content 中）
        if not function_calls and self._has_xml_tool_calls(raw_content):
            xml_calls = self._parse_minimax_tool_calls(raw_content)
            function_calls.extend(xml_calls)
            # 从 content 中移除工具调用 XML
            clean_content = self._remove_minimax_tool_calls(clean_content)
        
        return LlmResponse(
            content=clean_content,
            function_calls=function_calls,
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
        function_calls = []
        for tc in tool_calls_data:
            if tc.get("name"):
                try:
                    args = json.loads(tc.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                function_calls.append(FunctionCall(
                    id=tc.get("id", "call_unknown"),
                    name=tc["name"],
                    args=args,
                ))
        
        # 如果没有标准工具调用，检查 MiniMax XML 格式
        if not function_calls and self._has_xml_tool_calls(full_content):
            function_calls = self._parse_minimax_tool_calls(full_content)
            # 从 content 中移除工具调用 XML
            clean_content = self._remove_minimax_tool_calls(clean_content)
        
        # 返回最终完整响应
        final_response = LlmResponse(
            content=clean_content,
            function_calls=function_calls,
            thinking=thinking,
            raw_content=full_content,
            finish_reason=finish_reason,
            model=model_name or self.model,
            partial=False,
        )
        self._log_response(final_response)
        yield final_response
    
    def _extract_thinking(self, raw_content: str) -> tuple[str, str]:
        """提取并分离思考内容"""
        if not raw_content:
            return "", ""
        
        think_pattern = r'<think>(.*?)</think>'
        thinking_parts = re.findall(think_pattern, raw_content, re.DOTALL)
        clean_content = re.sub(think_pattern, '', raw_content, flags=re.DOTALL).strip()
        thinking_content = '\n'.join(part.strip() for part in thinking_parts) if thinking_parts else ''
        
        return clean_content, thinking_content
    
    def _parse_minimax_tool_calls(self, content: str) -> list[FunctionCall]:
        """
        解析 MiniMax 模型的 XML 格式工具调用
        
        支持多种格式:
        1. <minimax:tool_call><invoke name="...">...</invoke></minimax:tool_call>
        2. <invoke name="..."><parameter name="...">...</parameter></invoke>
        3. <invoke><tool_name><param>value</param></tool_name></invoke>
        """
        function_calls = []
        call_index = 0
        
        # 格式 1: <minimax:tool_call>...</minimax:tool_call>
        tool_call_pattern = r'<minimax:tool_call>(.*?)</minimax:tool_call>'
        for block in re.findall(tool_call_pattern, content, re.DOTALL):
            fc = self._parse_invoke_block(block, call_index)
            if fc:
                function_calls.append(fc)
                call_index += 1
        
        # 格式 2 & 3: 独立的 <invoke>...</invoke>（不在 minimax:tool_call 内）
        # 先移除已处理的 minimax:tool_call 块
        remaining = re.sub(tool_call_pattern, '', content, flags=re.DOTALL)
        
        # 匹配所有 <invoke>...</invoke>
        invoke_pattern = r'<invoke[^>]*>(.*?)</invoke>'
        for block in re.findall(invoke_pattern, remaining, re.DOTALL):
            fc = self._parse_invoke_block(block, call_index)
            if fc:
                function_calls.append(fc)
                call_index += 1
        
        return function_calls
    
    def _parse_invoke_block(self, block: str, index: int) -> FunctionCall | None:
        """解析单个 invoke 块"""
        # 格式 A: <invoke name="tool_name"><parameter name="...">...</parameter></invoke>
        invoke_name_match = re.search(r'<invoke\s+name="([^"]+)"', block)
        if invoke_name_match:
            func_name = invoke_name_match.group(1)
            param_pattern = r'<parameter\s+name="([^"]+)"[^>]*>(.*?)</parameter>'
            params = re.findall(param_pattern, block, re.DOTALL)
            args = {name: value.strip() for name, value in params}
            return FunctionCall(id=f"call_minimax_{index}", name=func_name, args=args)
        
        # 格式 B: <invoke><tool_name><param1>value1</param1></tool_name></invoke>
        # 或: <invoke><transfer_to_agent><agent>name</agent><args>...</args></transfer_to_agent></invoke>
        tool_match = re.search(r'<(\w+)>(.*?)</\1>', block, re.DOTALL)
        if tool_match:
            func_name = tool_match.group(1)
            inner_content = tool_match.group(2)
            
            # 解析内部参数
            args = {}
            param_matches = re.findall(r'<(\w+)>(.*?)</\1>', inner_content, re.DOTALL)
            for param_name, param_value in param_matches:
                args[param_name] = param_value.strip()
            
            # 特殊处理: transfer_to_agent 的参数映射
            if func_name == 'transfer_to_agent':
                if 'agent' in args:
                    args['agent_name'] = args.pop('agent')
                if 'args' in args:
                    # 解析 args 中可能嵌套的参数
                    args_content = args.pop('args')
                    # 提取 <task>...</task> 或其他嵌套参数
                    nested = re.findall(r'<(\w+)>(.*?)</\1>', args_content, re.DOTALL)
                    if nested:
                        for param_name, param_value in nested:
                            args[param_name] = param_value.strip()
                    else:
                        args['reason'] = args_content.strip()
            
            return FunctionCall(id=f"call_minimax_{index}", name=func_name, args=args)
        
        return None
    
    def _has_xml_tool_calls(self, content: str) -> bool:
        """检查内容是否包含 XML 格式的工具调用"""
        return '<minimax:tool_call>' in content or '<invoke>' in content or '<invoke ' in content
    
    def _remove_minimax_tool_calls(self, content: str) -> str:
        """从 content 中移除 MiniMax XML 格式的工具调用"""
        # 移除 <minimax:tool_call>...</minimax:tool_call> 块
        clean = re.sub(r'<minimax:tool_call>.*?</minimax:tool_call>', '', content, flags=re.DOTALL)
        # 移除独立的 <invoke>...</invoke> 块
        clean = re.sub(r'<invoke[^>]*>.*?</invoke>', '', clean, flags=re.DOTALL)
        return clean.strip()
    
    def _log_request(self, params: dict[str, Any]) -> None:
        """打印 API 请求详情（调试用）"""
        if not self.show_request:
            return
        
        level = self.log_level
        
        # ========== minimal: 一行摘要 ==========
        if level == "minimal":
            tools = params.get("tools", [])
            tool_names = [t.get("function", {}).get("name", "?") for t in tools]
            msgs = params.get("messages", [])
            last_user = next((m["content"][:50] for m in reversed(msgs) if m.get("role") == "user"), "")
            print(f"📤 INPUT | model={params.get('model')} | tools={tool_names} | user=\"{last_user}...\"")
            return
        
        # ========== normal / verbose ==========
        print("\n" + "=" * 60)
        print("📤 LLM INPUT")
        print("=" * 60)
        print(f"🤖 Model: {params.get('model', 'N/A')} | Stream: {params.get('stream', False)}")
        
        # 打印工具定义
        tools = params.get("tools", [])
        if tools:
            tool_names = [t.get("function", {}).get("name", "?") for t in tools]
            print(f"🔧 Tools: {tool_names}")
            
            # verbose 模式显示工具详情
            if level == "verbose":
                for t in tools:
                    func = t.get("function", {})
                    func_params = func.get("parameters", {})
                    print(f"   📌 {func.get('name')}: {func.get('description', '')[:60]}...")
                    if func_params.get("properties"):
                        print(f"      参数: {list(func_params['properties'].keys())}")
        
        # 打印消息
        messages = params.get("messages", [])
        print(f"\n📝 Messages ({len(messages)}):")
        
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            
            # normal 模式：简洁显示
            if level == "normal":
                if role == "system":
                    # system 消息只显示前 80 字符
                    preview = str(content).replace('\n', ' ')[:80]
                    print(f"  [{i+1}] SYSTEM: {preview}...")
                elif role == "user":
                    print(f"  [{i+1}] USER: {content}")
                elif role == "assistant":
                    if tool_calls:
                        tc_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                        print(f"  [{i+1}] ASSISTANT: [调用工具: {tc_names}]")
                    else:
                        preview = str(content).replace('\n', ' ')[:60]
                        print(f"  [{i+1}] ASSISTANT: {preview}...")
                elif role == "tool":
                    result = str(content)[:40]
                    print(f"  [{i+1}] TOOL: {result}...")
            
            # verbose 模式：完整显示
            else:
                print(f"\n  [{i+1}] 【{role.upper()}】")
                if role == "assistant" and tool_calls:
                    if content:
                        for line in str(content).split('\n'):
                            print(f"      {line}")
                    print("      🔧 工具调用:")
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        print(f"         → {func.get('name')}({func.get('arguments', '{}')})")
                elif role == "tool":
                    print(f"      (call_id: {msg.get('tool_call_id', '')})")
                    print(f"      {content}")
                else:
                    if content:
                        for line in str(content).split('\n'):
                            print(f"      {line}")
        
        print("=" * 60)
    
    def _log_response(self, response: 'LlmResponse') -> None:
        """打印 API 响应详情（调试用）"""
        if not self.show_request:
            return
        
        level = self.log_level
        
        # ========== minimal: 一行摘要 ==========
        if level == "minimal":
            content_preview = str(response.content or "").replace('\n', ' ')[:50]
            tool_names = [fc.name for fc in response.function_calls] if response.function_calls else []
            if tool_names:
                print(f"📥 OUTPUT | tools={tool_names} | content=\"{content_preview}...\"")
            else:
                print(f"📥 OUTPUT | content=\"{content_preview}...\"")
            return
        
        # ========== normal / verbose ==========
        print("\n" + "=" * 60)
        print("📥 LLM OUTPUT")
        print("=" * 60)
        
        # 基本信息
        if level == "verbose":
            print(f"🤖 Model: {response.model} | Finish: {response.finish_reason}")
        
        # 思考内容
        if response.thinking:
            if self.show_thinking:
                if level == "verbose":
                    print(f"\n💭 Thinking:")
                    for line in response.thinking.split('\n'):
                        print(f"    {line}")
                else:
                    thinking_preview = response.thinking.replace('\n', ' ')[:100]
                    print(f"💭 Thinking: {thinking_preview}...")
            else:
                print(f"💭 Thinking: (已隐藏)")
        
        # 主要内容
        if response.content:
            print(f"\n📝 Content:")
            if level == "verbose":
                for line in response.content.split('\n'):
                    print(f"    {line}")
            else:
                # normal: 显示完整内容但更紧凑
                content = response.content.strip()
                if len(content) > 200:
                    print(f"    {content[:200]}...")
                    print(f"    (共 {len(content)} 字符)")
                else:
                    print(f"    {content}")
        
        # 工具调用
        if response.function_calls:
            print(f"\n🔧 Tool Calls:")
            for fc in response.function_calls:
                if level == "verbose":
                    print(f"    📌 {fc.name}")
                    print(f"       ID: {fc.id}")
                    print(f"       Args: {json.dumps(fc.args, ensure_ascii=False)}")
                else:
                    args_str = json.dumps(fc.args, ensure_ascii=False)
                    if len(args_str) > 60:
                        args_str = args_str[:60] + "..."
                    print(f"    → {fc.name}({args_str})")
        
        # token 使用（仅 verbose）
        if level == "verbose" and response.usage:
            print(f"\n📊 Usage: prompt={response.usage.get('prompt_tokens', 'N/A')} | completion={response.usage.get('completion_tokens', 'N/A')} | total={response.usage.get('total_tokens', 'N/A')}")
        
        print("=" * 60 + "\n")
