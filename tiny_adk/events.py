"""事件系统 - 追踪 agent 执行过程中的所有操作"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class EventType(Enum):
  """事件类型"""
  USER_MESSAGE = 'user_message'
  MODEL_REQUEST = 'model_request'
  MODEL_RESPONSE = 'model_response'
  MODEL_RESPONSE_DELTA = 'model_response_delta'  # 流式响应的增量内容
  TOOL_CALL = 'tool_call'
  TOOL_RESPONSE = 'tool_response'
  ERROR = 'error'


@dataclass
class Event:
  """
  事件 - 记录 agent 执行过程中的每一步
  
  核心设计理念: 所有操作都是事件，事件组成会话历史
  """
  event_type: EventType
  content: Any
  timestamp: datetime = field(default_factory=datetime.now)
  metadata: dict[str, Any] = field(default_factory=dict)
  
  def to_dict(self) -> dict[str, Any]:
    """转换为字典格式"""
    return {
        'event_type': self.event_type.value,
        'content': self.content,
        'timestamp': self.timestamp.isoformat(),
        'metadata': self.metadata,
    }
  
  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> Event:
    """从字典创建事件"""
    return cls(
        event_type=EventType(data['event_type']),
        content=data['content'],
        timestamp=datetime.fromisoformat(data['timestamp']),
        metadata=data.get('metadata', {}),
    )

