"""会话管理 - 维护对话历史和状态"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .events import Event


@dataclass
class Session:
  """
  会话 - 维护一次完整对话的所有事件和状态
  
  核心设计理念:
  - Session 是有状态的，存储所有历史事件
  - Runner 是无状态的，每次执行从 Session 加载历史
  - 这种分离使得会话可以持久化、恢复、跨进程共享
  """
  session_id: str = field(default_factory=lambda: str(uuid4()))
  events: list[Event] = field(default_factory=list)
  metadata: dict[str, Any] = field(default_factory=dict)
  
  def add_event(self, event: Event) -> None:
    """添加事件到会话"""
    self.events.append(event)
  
  def get_events(self) -> list[Event]:
    """获取所有事件"""
    return self.events
  
  def get_conversation_history(self) -> list[dict[str, Any]]:
    """
    获取对话历史（用于传给 LLM 的格式）
    
    将事件转换为 LLM 可理解的消息格式
    符合 OpenAI Chat Completion API 规范
    """
    history = []
    pending_tool_calls = []
    
    for event in self.events:
      if event.event_type.value == 'user_message':
        history.append({
            'role': 'user',
            'content': event.content,
        })
      
      elif event.event_type.value == 'model_response':
        # 如果有 content，添加普通助手消息
        if event.content:
          history.append({
              'role': 'assistant',
              'content': event.content,
          })
      
      elif event.event_type.value == 'tool_call':
        # 收集工具调用
        tool_call = event.content
        # 转换为 OpenAI 格式
        pending_tool_calls.append({
            'id': tool_call.get('id', 'call_unknown'),
            'type': 'function',
            'function': {
                'name': tool_call.get('name', ''),
                'arguments': self._dict_to_json(tool_call.get('arguments', {})),
            }
        })
      
      elif event.event_type.value == 'tool_response':
        # 如果有待处理的工具调用，先添加 assistant 消息
        if pending_tool_calls:
          history.append({
              'role': 'assistant',
              'content': None,
              'tool_calls': pending_tool_calls,
          })
          pending_tool_calls = []
        
        # 添加工具响应
        history.append({
            'role': 'tool',
            'content': event.content.get('result', ''),
            'tool_call_id': event.content.get('call_id', 'call_unknown'),
        })
    
    # 如果还有未处理的工具调用
    if pending_tool_calls:
      history.append({
          'role': 'assistant',
          'content': None,
          'tool_calls': pending_tool_calls,
      })
    
    return history
  
  def _dict_to_json(self, obj: Any) -> str:
    """将字典转为 JSON 字符串"""
    import json
    if isinstance(obj, str):
      return obj
    return json.dumps(obj, ensure_ascii=False)
  
  def clear(self) -> None:
    """清空会话"""
    self.events.clear()
  
  def to_dict(self) -> dict[str, Any]:
    """序列化为字典"""
    return {
        'session_id': self.session_id,
        'events': [e.to_dict() for e in self.events],
        'metadata': self.metadata,
    }
  
  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> Session:
    """从字典反序列化"""
    return cls(
        session_id=data['session_id'],
        events=[Event.from_dict(e) for e in data['events']],
        metadata=data.get('metadata', {}),
    )

