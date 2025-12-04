"""事件系统 - 追踪 agent 执行过程中的所有操作"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class EventType(Enum):
    """事件类型"""
    USER_MESSAGE = 'user_message'
    MODEL_REQUEST = 'model_request'
    MODEL_RESPONSE = 'model_response'
    MODEL_RESPONSE_DELTA = 'model_response_delta'  # 流式响应的增量内容
    TOOL_CALL = 'tool_call'
    TOOL_RESPONSE = 'tool_response'
    AGENT_TRANSFER = 'agent_transfer'  # Agent 跳转事件
    ERROR = 'error'


@dataclass
class EventActions:
    """
    事件动作 - 描述事件触发的后续动作
    
    用于多 Agent 场景下的控制流：
    - transfer_to_agent: 跳转到指定 Agent
    - escalate: 向上级 Agent 报告/退出循环
    - state_delta: 状态变更
    """
    transfer_to_agent: Optional[str] = None
    """跳转到的目标 Agent 名称"""
    
    escalate: bool = False
    """是否向上级 Agent 报告（用于退出 LoopAgent）"""
    
    state_delta: dict[str, Any] = field(default_factory=dict)
    """状态变更"""
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            'transfer_to_agent': self.transfer_to_agent,
            'escalate': self.escalate,
            'state_delta': self.state_delta,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'EventActions':
        """从字典创建"""
        return cls(
            transfer_to_agent=data.get('transfer_to_agent'),
            escalate=data.get('escalate', False),
            state_delta=data.get('state_delta', {}),
        )


@dataclass
class Event:
    """
    事件 - 记录 agent 执行过程中的每一步
    
    核心设计理念: 所有操作都是事件，事件组成会话历史
    
    Multi-Agent 扩展:
    - author: 产生此事件的 Agent 名称
    - actions: 事件触发的动作（如 Agent 跳转）
    """
    event_type: EventType
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Multi-Agent 支持
    author: Optional[str] = None
    """产生此事件的 Agent 名称"""
    
    actions: EventActions = field(default_factory=EventActions)
    """事件触发的动作"""
    
    # 流式支持
    partial: bool = False
    """是否是流式的部分事件（不应被持久化）"""
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式"""
        result = {
            'event_type': self.event_type.value,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
        }
        if self.author:
            result['author'] = self.author
        if self.actions.transfer_to_agent or self.actions.escalate or self.actions.state_delta:
            result['actions'] = self.actions.to_dict()
        if self.partial:
            result['partial'] = self.partial
        return result
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        """从字典创建事件"""
        actions = EventActions.from_dict(data.get('actions', {}))
        return cls(
            event_type=EventType(data['event_type']),
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {}),
            author=data.get('author'),
            actions=actions,
            partial=data.get('partial', False),
        )
    
    # ==================== 便捷方法 ====================
    
    def is_transfer(self) -> bool:
        """是否是 Agent 跳转事件"""
        return self.event_type == EventType.AGENT_TRANSFER or bool(self.actions.transfer_to_agent)
    
    def is_final_response(self) -> bool:
        """是否是最终响应（非流式增量）"""
        return self.event_type == EventType.MODEL_RESPONSE and not self.partial
    
    def get_transfer_target(self) -> Optional[str]:
        """获取跳转目标 Agent 名称"""
        if self.event_type == EventType.AGENT_TRANSFER:
            return self.content.get('target_agent') if isinstance(self.content, dict) else None
        return self.actions.transfer_to_agent


# ==================== 事件工厂函数 ====================

def create_transfer_event(
    from_agent: str,
    to_agent: str,
    reason: str = "",
) -> Event:
    """创建 Agent 跳转事件"""
    return Event(
        event_type=EventType.AGENT_TRANSFER,
        content={
            'from_agent': from_agent,
            'target_agent': to_agent,
            'reason': reason,
        },
        author=from_agent,
        actions=EventActions(transfer_to_agent=to_agent),
    )


def create_escalate_event(
    agent_name: str,
    reason: str = "",
) -> Event:
    """创建 escalate 事件（用于退出 LoopAgent）"""
    return Event(
        event_type=EventType.MODEL_RESPONSE,
        content=reason or "Task completed, escalating.",
        author=agent_name,
        actions=EventActions(escalate=True),
        metadata={'escalate': True},
    )
