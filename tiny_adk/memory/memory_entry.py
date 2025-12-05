"""
MemoryEntry - 记忆条目数据结构

设计理念：
- 简洁但信息完整
- 支持多种内容类型
- 可序列化/反序列化
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from enum import Enum


class MemoryType(Enum):
    """记忆类型"""
    # 短期记忆：会话级别，随会话结束可能清理
    SHORT_TERM = "short_term"
    # 长期记忆：跨会话持久化，重要信息
    LONG_TERM = "long_term"
    # 实体记忆：关于特定实体的结构化信息
    ENTITY = "entity"
    # 事件记忆：用户-Agent对话历史
    EVENT = "event"


@dataclass
class MemoryEntry:
    """
    记忆条目 - 存储单条记忆的数据结构
    
    核心字段：
    - content: 记忆内容（字符串或结构化数据）
    - memory_type: 记忆类型
    - timestamp: 创建时间
    - metadata: 自定义元数据
    
    关联字段：
    - app_name: 应用名称
    - user_id: 用户 ID
    - session_id: 会话 ID（短期记忆必需）
    - author: 记忆来源（user/agent_name）
    """
    
    # 核心字段
    content: str
    """记忆内容"""
    
    memory_type: MemoryType = MemoryType.SHORT_TERM
    """记忆类型"""
    
    timestamp: datetime = field(default_factory=datetime.now)
    """创建时间"""
    
    metadata: dict[str, Any] = field(default_factory=dict)
    """自定义元数据"""
    
    # 唯一标识
    id: Optional[str] = None
    """记忆 ID（由存储层生成）"""
    
    # 关联字段
    app_name: str = ""
    """应用名称"""
    
    user_id: str = ""
    """用户 ID"""
    
    session_id: Optional[str] = None
    """会话 ID（短期记忆必需）"""
    
    author: Optional[str] = None
    """记忆来源（user/agent名称）"""
    
    # 向量搜索支持
    embedding: Optional[list[float]] = None
    """向量嵌入（用于语义搜索）"""
    
    # 重要性/权重
    importance: float = 1.0
    """重要性权重（0-1），影响检索排序"""
    
    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            'id': self.id,
            'content': self.content,
            'memory_type': self.memory_type.value,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'app_name': self.app_name,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'author': self.author,
            'importance': self.importance,
            # 不序列化 embedding（太大）
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        """从字典反序列化"""
        return cls(
            id=data.get('id'),
            content=data['content'],
            memory_type=MemoryType(data.get('memory_type', 'short_term')),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now(),
            metadata=data.get('metadata', {}),
            app_name=data.get('app_name', ''),
            user_id=data.get('user_id', ''),
            session_id=data.get('session_id'),
            author=data.get('author'),
            importance=data.get('importance', 1.0),
        )
    
    @classmethod
    def from_event(
        cls,
        event: Any,  # Event type
        app_name: str = "",
        user_id: str = "",
        session_id: str = "",
    ) -> MemoryEntry:
        """从 Event 创建记忆条目"""
        # 提取内容
        content = event.content if isinstance(event.content, str) else str(event.content)
        
        return cls(
            content=content,
            memory_type=MemoryType.EVENT,
            timestamp=event.timestamp if hasattr(event, 'timestamp') else datetime.now(),
            metadata={'event_type': event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type)},
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            author=getattr(event, 'author', None),
        )
    
    def __str__(self) -> str:
        """友好的字符串表示"""
        author_str = f" by {self.author}" if self.author else ""
        return f"[{self.memory_type.value}]{author_str}: {self.content[:50]}..."

