"""Runner - Agent 的执行引擎"""

from __future__ import annotations

import json
from typing import Any, Iterator

from .agents import Agent
from .config import Config, get_config
from .events import Event, EventType
from .session import Session
from .tools import Tool


class Runner:
  """
  Runner - 无状态的执行引擎
  
  核心设计理念:
  - Runner 负责编排执行流程
  - Runner 不保存状态，所有状态在 Session 中
  - Runner 管理 "Reason-Act" 循环：
    1. 从 Session 加载历史
    2. 调用 LLM 推理
    3. 执行工具调用
    4. 保存事件到 Session
    5. 重复直到完成
  
  这种设计使得:
  - Runner 可以是单例或短生命周期对象
  - Session 可以跨 Runner 实例持久化
  - 容易实现分布式执行
  """
  
  def __init__(
      self,
      llm_client: Any | None = None,
      api_base: str | None = None,
      api_key: str | None = None,
      default_model: str | None = None,
      show_thinking: bool | None = None,
      show_request: bool | None = None,
      config: Config | None = None,
  ):
    """
    初始化 Runner
    
    配置优先级: 直接传入的参数 > 传入的 config > 全局配置 (环境变量/配置文件)
    
    Args:
      llm_client: LLM 客户端（OpenAI 兼容的客户端）
      api_base: vLLM server 的 API 地址（如 "http://localhost:8000/v1"）
      api_key: API 密钥（vLLM 通常用 "EMPTY"）
      default_model: 默认模型名称
      show_thinking: 是否打印模型的思考过程（默认 False）
      show_request: 是否打印请求参数（默认 False）
      config: 可选的配置对象，不传则使用全局配置
    """
    # 获取配置（优先使用传入的，否则使用全局配置）
    self._config = config or get_config()
    
    self.llm_client = llm_client
    # 使用配置文件的值作为默认，参数可以覆盖
    self.api_base = api_base if api_base is not None else self._config.llm.api_base
    self.api_key = api_key if api_key is not None else self._config.llm.api_key
    self.default_model = default_model if default_model is not None else self._config.llm.model
    self.show_thinking = show_thinking if show_thinking is not None else self._config.runner.show_thinking
    self.show_request = show_request if show_request is not None else self._config.runner.show_request
    
    # 如果没有提供 client，自动创建
    if not llm_client and self.api_base:
      self._init_openai_client()
  
  def run(
      self,
      agent: Agent,
      session: Session,
      user_message: str,
  ) -> str:
    """
    同步执行一轮对话
    
    Args:
      agent: 要执行的 Agent
      session: 会话对象
      user_message: 用户消息
    
    Returns:
      Agent 的最终响应
    """
    # 1. 记录用户消息事件
    session.add_event(Event(
        event_type=EventType.USER_MESSAGE,
        content=user_message,
    ))
    
    # 2. 执行 Reason-Act 循环
    response = self._reason_act_loop(agent, session)
    
    return response
  
  def run_stream(
      self,
      agent: Agent,
      session: Session,
      user_message: str,
  ) -> Iterator[Event]:
    """
    流式执行 - 实时返回事件
    
    只返回 Agent 的响应和工具调用事件，不返回用户消息
    用户消息会被记录到 session 中，但不会 yield
    """
    # 1. 记录用户消息到 session（不 yield）
    session.add_event(Event(
        event_type=EventType.USER_MESSAGE,
        content=user_message,
    ))
    
    # 2. 执行并流式返回 Agent 的事件
    yield from self._reason_act_loop_stream(agent, session)
  
  def _reason_act_loop(
      self,
      agent: Agent,
      session: Session,
  ) -> str:
    """
    Reason-Act 循环的简化实现
    
    真实的 ADK 实现会：
    - 调用实际的 LLM API
    - 处理函数调用
    - 处理错误和重试
    - 支持多轮工具调用
    
    这里用简化的逻辑展示核心流程
    """
    # 构建请求
    messages = self._build_messages(agent, session)
    
    # 模拟 LLM 调用（实际应该调用 self.llm_client）
    if self.llm_client is None:
      response = self._mock_llm_response(agent, messages)
    else:
      response = self._call_llm(agent, messages)
    
    # 记录模型响应
    session.add_event(Event(
        event_type=EventType.MODEL_RESPONSE,
        content=response['content'],
        metadata=response.get('metadata', {}),
    ))
    
    # 如果有工具调用，执行工具
    if 'tool_calls' in response:
      for tool_call in response['tool_calls']:
        self._execute_tool(agent, session, tool_call)
      
      # 递归继续循环（工具执行后让 LLM 继续）
      return self._reason_act_loop(agent, session)
    
    return response['content']
  
  def _reason_act_loop_stream(
      self,
      agent: Agent,
      session: Session,
  ) -> Iterator[Event]:
    """流式版本的 Reason-Act 循环 - 真正的流式输出"""
    messages = self._build_messages(agent, session)
    
    # 使用流式调用或模拟响应
    if self.llm_client is None:
      # 模拟流式响应
      response = self._mock_llm_response(agent, messages)
      event = Event(
          event_type=EventType.MODEL_RESPONSE,
          content=response['content'],
          metadata=response.get('metadata', {}),
      )
      session.add_event(event)
      yield event
    else:
      # 真正的流式 LLM 调用
      full_content = ''
      full_response = None
      
      # 逐步接收流式响应
      for chunk in self._call_llm_stream(agent, messages):
        if chunk.get('type') == 'content':
          # 流式内容片段
          full_content += chunk['delta']
          # 实时 yield 内容事件（可选：用于实时显示）
          yield Event(
              event_type=EventType.MODEL_RESPONSE_DELTA,
              content=chunk['delta'],
              metadata={'type': 'delta'},
          )
        elif chunk.get('done'):
          # 完整响应
          full_response = chunk
      
      # 保存完整响应到 session
      if full_response:
        response = full_response
        event = Event(
            event_type=EventType.MODEL_RESPONSE,
            content=response['content'],
            metadata=response.get('metadata', {}),
        )
        session.add_event(event)
        yield event
      else:
        # 如果没有收到完整响应，使用累积的内容
        response = {'content': full_content}
        event = Event(
            event_type=EventType.MODEL_RESPONSE,
            content=full_content,
            metadata={},
        )
        session.add_event(event)
        yield event
    
    # 处理工具调用
    if 'tool_calls' in response:
      for tool_call in response['tool_calls']:
        yield from self._execute_tool_stream(agent, session, tool_call)
      
      # 继续循环
      yield from self._reason_act_loop_stream(agent, session)
  
  def _build_messages(
      self,
      agent: Agent,
      session: Session,
  ) -> list[dict[str, Any]]:
    """
    构建发送给 LLM 的消息列表
    
    核心转换：Session Events -> LLM Messages
    """
    messages = [
        {'role': 'system', 'content': agent.get_system_prompt()}
    ]
    
    # 添加历史对话
    messages.extend(session.get_conversation_history())
    
    return messages
  
  def _init_openai_client(self):
    """初始化 OpenAI 兼容的客户端"""
    try:
      from openai import OpenAI
      self.llm_client = OpenAI(
          base_url=self.api_base,
          api_key=self.api_key,
      )
      print(f"✅ 已连接到 LLM: {self.api_base}")
    except ImportError:
      raise ImportError(
          "需要安装 openai 包: pip install openai"
      )
  
  def _extract_thinking_content(self, raw_content: str) -> tuple[str, str]:
    """
    提取并分离思考内容
    
    将 <think>...</think> 标签内的内容提取出来，不放入对话历史
    这符合主流设计（ADK/Anthropic/OpenAI）的做法
    
    Args:
      raw_content: 原始模型输出
    
    Returns:
      (clean_content, thinking_content) - 清洗后的内容和思考过程
    """
    import re
    
    if not raw_content:
      return '', ''
    
    # 提取 <think> 标签内容
    think_pattern = r'<think>(.*?)</think>'
    thinking_parts = re.findall(think_pattern, raw_content, re.DOTALL)
    
    # 移除 <think> 标签，只保留实际输出
    clean_content = re.sub(think_pattern, '', raw_content, flags=re.DOTALL).strip()
    
    # 合并所有思考内容
    thinking_content = '\n'.join(part.strip() for part in thinking_parts) if thinking_parts else ''
    
    return clean_content, thinking_content
  
  class _StreamThinkingFilter:
    """流式思考内容过滤器 - 实时过滤 <think> 标签"""
    
    def __init__(self):
      self.buffer = ''  # 缓冲区
      self.in_thinking = False  # 是否在思考模式
      self.thinking_content = ''  # 累积的思考内容
      self.clean_content = ''  # 清洗后的内容
    
    def process_delta(self, delta: str) -> str:
      """
      处理流式内容片段，过滤 thinking 内容
      
      Returns:
        应该输出的内容（可能为空字符串）
      """
      self.buffer += delta
      output = ''
      
      while self.buffer:
        if not self.in_thinking:
          # 不在思考模式，检查是否遇到 <think> 标签
          think_start_idx = self.buffer.find('<think>')
          
          if think_start_idx == -1:
            # 没有 <think>，但可能在末尾有部分标签，保留少量缓冲
            if len(self.buffer) > 10:
              # 输出除了最后几个字符外的所有内容
              output_part = self.buffer[:-7]  # 保留 7 个字符（'<think>' 的长度）
              output += output_part
              self.clean_content += output_part
              self.buffer = self.buffer[-7:]
            break
          else:
            # 找到 <think>，输出之前的内容
            if think_start_idx > 0:
              output_part = self.buffer[:think_start_idx]
              output += output_part
              self.clean_content += output_part
            # 进入思考模式，跳过 <think> 标签
            self.buffer = self.buffer[think_start_idx + 7:]
            self.in_thinking = True
        else:
          # 在思考模式，查找 </think> 标签
          think_end_idx = self.buffer.find('</think>')
          
          if think_end_idx == -1:
            # 没有 </think>，保留少量缓冲
            if len(self.buffer) > 10:
              thinking_part = self.buffer[:-8]  # 保留 8 个字符（'</think>' 的长度）
              self.thinking_content += thinking_part
              self.buffer = self.buffer[-8:]
            break
          else:
            # 找到 </think>，保存思考内容
            self.thinking_content += self.buffer[:think_end_idx]
            # 退出思考模式，跳过 </think> 标签
            self.buffer = self.buffer[think_end_idx + 8:]
            self.in_thinking = False
      
      return output
    
    def finalize(self) -> tuple[str, str, str]:
      """
      完成处理，返回清洗后的内容、思考内容和剩余缓冲
      
      Returns:
        (clean_content, thinking_content, remaining_buffer)
      """
      remaining = ''
      
      # 处理剩余缓冲区
      if self.buffer:
        if self.in_thinking:
          self.thinking_content += self.buffer
        else:
          # 非思考模式的缓冲区内容应该输出
          remaining = self.buffer
          self.clean_content += self.buffer
        self.buffer = ''
      
      return self.clean_content.strip(), self.thinking_content.strip(), remaining
  
  def _call_llm_stream(
      self,
      agent: Agent,
      messages: list[dict[str, Any]],
  ) -> Iterator[dict[str, Any]]:
    """
    流式调用 LLM - 实时返回生成的内容片段
    
    Yields:
      包含 'delta' (内容片段) 或 'done' (完整响应) 的字典
    """
    if not self.llm_client:
      raise ValueError(
          "未配置 LLM 客户端。请在初始化 Runner 时提供 llm_client 或 api_base"
      )
    
    # 准备工具定义（如果有）
    tools = None
    if agent.tools:
      tools = [self._tool_to_openai_format(tool) for tool in agent.tools]
    
    try:
      # 使用 agent.model，如果为空或是默认值 'gpt-4'，则使用 runner 的默认模型
      model_to_use = agent.model
      if not model_to_use or model_to_use == 'gpt-4':
        model_to_use = self.default_model
      
      # 构建请求参数
      request_params = {
          'model': model_to_use,
          'messages': messages,
          'temperature': agent.temperature,
          'max_tokens': agent.max_tokens,
          'stream': True,  # 启用流式模式
      }
      
      # 可选：打印请求参数
      if self.show_request:
        print('--------------------------------')
        print('LLM 流式请求参数:')
        print({**request_params, 'stream': True})
        print('--------------------------------')
      
      # 如果有工具，添加工具定义
      if tools:
        request_params['tools'] = tools
        request_params['tool_choice'] = 'auto'
      
      # 流式调用 API
      stream = self.llm_client.chat.completions.create(**request_params)
      
      # 收集完整响应
      full_content = ''
      tool_calls_data = []
      finish_reason = None
      model_name = None
      
      # 创建思考内容过滤器（用于清洗对话历史）
      thinking_filter = self._StreamThinkingFilter()
      
      # 逐步处理流式响应
      for chunk in stream:
        if not chunk.choices:
          continue
        
        choice = chunk.choices[0]
        delta = choice.delta
        
        # 保存模型名称
        if chunk.model:
          model_name = chunk.model
        
        # 处理内容片段
        if delta.content:
          full_content += delta.content
          
          # 始终通过过滤器处理（用于生成清洗后的内容保存到 session）
          filtered_delta = thinking_filter.process_delta(delta.content)
          
          # 根据 show_thinking 决定输出内容
          if self.show_thinking:
            # 显示 thinking：原样输出所有内容
            yield {
                'delta': delta.content,  # 原始内容，包含 <think> 标签
                'type': 'content',
            }
          else:
            # 不显示 thinking：只输出过滤后的内容
            if filtered_delta:
              yield {
                  'delta': filtered_delta,
                  'type': 'content',
              }
        
        # 处理工具调用
        if delta.tool_calls:
          for tc in delta.tool_calls:
            tool_calls_data.append({
                'id': tc.id,
                'name': tc.function.name if tc.function else None,
                'arguments': tc.function.arguments if tc.function else None,
            })
        
        # 处理完成原因
        if choice.finish_reason:
          finish_reason = choice.finish_reason
      
      # 完成过滤，获取清洗后的内容、思考内容和剩余缓冲
      clean_content, thinking, remaining = thinking_filter.finalize()
      
      # 如果不显示 thinking，需要输出剩余缓冲区内容（最后几个字符）
      # 如果显示 thinking，剩余内容已经在上面原样输出了
      if not self.show_thinking and remaining:
        yield {
            'delta': remaining,
            'type': 'content',
        }
      
      # 返回完整响应
      result = {
          'done': True,
          'content': clean_content,
          'raw_content': full_content,
          'metadata': {
              'model': model_name or model_to_use,
              'finish_reason': finish_reason,
              'thinking': thinking,
          },
      }
      
      # 如果有工具调用，添加到结果中
      if tool_calls_data:
        # 合并工具调用数据
        merged_tool_calls = []
        for tc in tool_calls_data:
          if tc.get('name'):
            merged_tool_calls.append({
                'id': tc.get('id', 'call_unknown'),
                'name': tc['name'],
                'arguments': json.loads(tc.get('arguments', '{}')),
            })
        
        if merged_tool_calls:
          result['tool_calls'] = merged_tool_calls
      
      yield result
    
    except Exception as e:
      # 错误处理
      yield {
          'done': True,
          'content': f"LLM 流式调用失败: {str(e)}",
          'metadata': {'error': str(e)},
      }
  
  def _call_llm(
      self,
      agent: Agent,
      messages: list[dict[str, Any]],
  ) -> dict[str, Any]:
    """
    调用 OpenAI 兼容的 LLM（vLLM server）- 非流式版本
    
    支持:
    - 文本生成
    - 函数调用（如果模型支持）
    - 思考内容分离（不污染对话历史）
    """
    if not self.llm_client:
      raise ValueError(
          "未配置 LLM 客户端。请在初始化 Runner 时提供 llm_client 或 api_base"
      )
    
    # 准备工具定义（如果有）
    tools = None
    if agent.tools:
      tools = [self._tool_to_openai_format(tool) for tool in agent.tools]
    
    # 调用 LLM
    try:
      # 使用 agent.model，如果为空或是默认值 'gpt-4'，则使用 runner 的默认模型
      model_to_use = agent.model
      if not model_to_use or model_to_use == 'gpt-4':
        model_to_use = self.default_model
      
      # 构建请求参数
      request_params = {
          'model': model_to_use,
          'messages': messages,
          'temperature': agent.temperature,
          'max_tokens': agent.max_tokens,
      }

      # 可选：打印请求参数
      if self.show_request:
        print('--------------------------------')
        print('LLM 请求参数:')
        print(request_params)
        print('--------------------------------')
      
      # 如果有工具，添加工具定义
      if tools:
        request_params['tools'] = tools
        request_params['tool_choice'] = 'auto'
      
      # 调用 API
      response = self.llm_client.chat.completions.create(**request_params)
      
      # 解析响应
      choice = response.choices[0]
      message = choice.message
      
      # 提取原始内容
      raw_content = message.content or ''
      
      # 分离思考内容和实际输出
      clean_content, thinking = self._extract_thinking_content(raw_content)
      
      # 可选：打印思考过程（用于调试）
      if thinking and self.show_thinking:
        print('--------------------------------')
        print('💭 Agent 思考过程:')
        print(thinking)
        print('--------------------------------')
      
      # 检查是否有工具调用
      if message.tool_calls:
        return {
            'content': clean_content,  # 只保存清洗后的内容
            'tool_calls': [
                {
                    'id': tc.id,
                    'name': tc.function.name,
                    'arguments': json.loads(tc.function.arguments),
                }
                for tc in message.tool_calls
            ],
            'metadata': {
                'model': response.model,
                'finish_reason': choice.finish_reason,
                'thinking': thinking,  # 思考过程放在 metadata 中
                'raw_content': raw_content,  # 保留原始内容用于调试
            },
        }
      
      # 普通文本响应
      return {
          'content': clean_content,  # 只保存清洗后的内容
          'metadata': {
              'model': response.model,
              'finish_reason': choice.finish_reason,
              'thinking': thinking,  # 思考过程放在 metadata 中
              'raw_content': raw_content,  # 保留原始内容用于调试
              'usage': {
                  'prompt_tokens': response.usage.prompt_tokens,
                  'completion_tokens': response.usage.completion_tokens,
                  'total_tokens': response.usage.total_tokens,
              } if response.usage else {},
          },
      }
    
    except Exception as e:
      # 错误处理
      return {
          'content': f"LLM 调用失败: {str(e)}",
          'metadata': {'error': str(e)},
      }
  
  def _tool_to_openai_format(self, tool: Tool) -> dict[str, Any]:
    """
    将 Tool 转换为 OpenAI function calling 格式
    
    OpenAI 格式:
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "获取天气",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "城市名"}
          },
          "required": ["city"]
        }
      }
    }
    """
    # 构建参数定义
    properties = {}
    required = []
    
    for param_name, param_info in tool.parameters.items():
      param_type = param_info.get('type', 'string')
      
      # 转换 Python 类型到 JSON Schema 类型
      type_mapping = {
          'str': 'string',
          'int': 'integer',
          'float': 'number',
          'bool': 'boolean',
          'list': 'array',
          'dict': 'object',
      }
      json_type = type_mapping.get(param_type, 'string')
      
      properties[param_name] = {
          'type': json_type,
          'description': param_info.get('description', f'参数 {param_name}'),
      }
      
      # 如果没有默认值，则为必需参数
      if 'default' not in param_info:
        required.append(param_name)
    
    return {
        'type': 'function',
        'function': {
            'name': tool.name,
            'description': tool.description,
            'parameters': {
                'type': 'object',
                'properties': properties,
                'required': required,
            },
        },
    }
  
  # Mock 响应标签
  MOCK_LABEL = "[MOCK]"
  
  def _mock_llm_response(
      self,
      agent: Agent,
      messages: list[dict[str, Any]],
  ) -> dict[str, Any]:
    """模拟 LLM 响应（用于演示）"""
    last_message = messages[-1] if messages else {}
    
    # 检查最后一条消息的角色
    last_role = last_message.get('role', '')
    last_content = last_message.get('content', '')
    
    # Mock 元数据
    mock_metadata = {'model': agent.model, 'is_mock': True}
    
    # 如果上一条是工具响应，说明工具已经执行完毕，应该返回最终答案
    if last_role == 'tool':
      tool_result = last_content
      return {
          'content': f"{self.MOCK_LABEL} 根据查询结果：{tool_result}",
          'metadata': mock_metadata,
      }
    
    # 如果是用户消息且包含特定关键词，模拟工具调用
    if last_role == 'user' and agent.tools:
      content_str = str(last_content)
      
      # 天气查询
      if '天气' in content_str:
        # 提取城市名（简单模拟）
        city = '北京'  # 默认
        for c in ['北京', '上海', '深圳', '成都']:
          if c in content_str:
            city = c
            break
        
        return {
            'content': None,
            'tool_calls': [{
                'id': 'call_weather_123',
                'name': 'get_weather',
                'arguments': {'city': city},
            }],
            'metadata': mock_metadata,
        }
      
      # 计算请求
      if '计算' in content_str or '=' in content_str:
        # 简单提取表达式
        import re
        expr_match = re.search(r'(\d+\s*[\+\-\*/]\s*\d+)', content_str)
        if expr_match:
          expression = expr_match.group(1).replace(' ', '')
          return {
              'content': None,
              'tool_calls': [{
                  'id': 'call_calc_123',
                  'name': 'calculate',
                  'arguments': {'expression': expression},
              }],
              'metadata': mock_metadata,
          }
      
      # 搜索请求
      if '搜索' in content_str or '查找' in content_str:
        return {
            'content': None,
            'tool_calls': [{
                'id': 'call_search_123',
                'name': 'web_search',
                'arguments': {'query': content_str},
            }],
            'metadata': mock_metadata,
        }
    
    # 默认文本响应
    return {
        'content': f"{self.MOCK_LABEL} 我是 {agent.name}。收到消息: {last_content}",
        'metadata': mock_metadata,
    }
  
  def _execute_tool(
      self,
      agent: Agent,
      session: Session,
      tool_call: dict[str, Any],
  ) -> None:
    """执行工具调用"""
    # 记录工具调用事件
    session.add_event(Event(
        event_type=EventType.TOOL_CALL,
        content=tool_call,
    ))
    
    # 查找并执行工具
    tool = self._find_tool(agent, tool_call['name'])
    if tool:
      try:
        # 解析参数
        args = tool_call.get('arguments', {})
        if isinstance(args, str):
          args = json.loads(args)
        
        # 执行工具
        result = tool.execute(**args)
        
        # 记录工具响应
        session.add_event(Event(
            event_type=EventType.TOOL_RESPONSE,
            content={
                'call_id': tool_call.get('id'),
                'name': tool_call['name'],
                'result': str(result),
            },
        ))
      except Exception as e:
        # 记录错误
        session.add_event(Event(
            event_type=EventType.ERROR,
            content={
                'tool': tool_call['name'],
                'error': str(e),
            },
        ))
  
  def _execute_tool_stream(
      self,
      agent: Agent,
      session: Session,
      tool_call: dict[str, Any],
  ) -> Iterator[Event]:
    """流式执行工具"""
    # 工具调用事件
    event = Event(event_type=EventType.TOOL_CALL, content=tool_call)
    session.add_event(event)
    yield event
    
    # 执行工具
    tool = self._find_tool(agent, tool_call['name'])
    if tool:
      try:
        args = tool_call.get('arguments', {})
        if isinstance(args, str):
          args = json.loads(args)
        
        result = tool.execute(**args)
        
        event = Event(
            event_type=EventType.TOOL_RESPONSE,
            content={
                'call_id': tool_call.get('id'),
                'name': tool_call['name'],
                'result': str(result),
            },
        )
        session.add_event(event)
        yield event
      except Exception as e:
        event = Event(
            event_type=EventType.ERROR,
            content={'tool': tool_call['name'], 'error': str(e)},
        )
        session.add_event(event)
        yield event
  
  def _find_tool(self, agent: Agent, tool_name: str) -> Tool | None:
    """查找工具"""
    for tool in agent.tools:
      if tool.name == tool_name:
        return tool
    return None

