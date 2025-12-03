"""会话管理 - 维护对话历史和状态"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

from .events import Event

if TYPE_CHECKING:
    from .agents import Agent


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
    user_id: str = ""  # 用户 ID
    events: list[Event] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)  # 会话状态
    
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
            'user_id': self.user_id,
            'events': [e.to_dict() for e in self.events],
            'metadata': self.metadata,
            'state': self.state,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        """从字典反序列化"""
        return cls(
            session_id=data['session_id'],
            user_id=data.get('user_id', ''),
            events=[Event.from_dict(e) for e in data['events']],
            metadata=data.get('metadata', {}),
            state=data.get('state', {}),
        )


# ==================== SessionService ====================

class SessionService:
    """
    Session 持久化服务
    
    负责 Session 的存储和检索，支持:
    - 内存存储（默认，用于测试和开发）
    - 可扩展为数据库、Redis 等持久化方案
    """
    
    def __init__(self):
        self._sessions: dict[tuple[str, str], Session] = {}
    
    async def get_session(
        self, 
        user_id: str, 
        session_id: str
    ) -> Session:
        """
        获取或创建 Session
        
        Args:
            user_id: 用户 ID
            session_id: 会话 ID
            
        Returns:
            Session 实例
        """
        key = (user_id, session_id)
        if key not in self._sessions:
            self._sessions[key] = Session(
                user_id=user_id,
                session_id=session_id
            )
        return self._sessions[key]
    
    def get_session_sync(
        self, 
        user_id: str, 
        session_id: str
    ) -> Session:
        """同步版本的获取 Session"""
        key = (user_id, session_id)
        if key not in self._sessions:
            self._sessions[key] = Session(
                user_id=user_id,
                session_id=session_id
            )
        return self._sessions[key]
    
    async def save_session(self, session: Session) -> None:
        """
        保存 Session
        
        Args:
            session: Session 实例
        """
        key = (session.user_id, session.session_id)
        self._sessions[key] = session
    
    def save_session_sync(self, session: Session) -> None:
        """同步版本的保存 Session"""
        key = (session.user_id, session.session_id)
        self._sessions[key] = session
    
    async def delete_session(
        self, 
        user_id: str, 
        session_id: str
    ) -> bool:
        """
        删除 Session
        
        Args:
            user_id: 用户 ID
            session_id: 会话 ID
            
        Returns:
            是否成功删除
        """
        key = (user_id, session_id)
        if key in self._sessions:
            del self._sessions[key]
            return True
        return False
    
    async def list_sessions(self, user_id: str) -> list[Session]:
        """
        列出用户的所有 Session
        
        Args:
            user_id: 用户 ID
            
        Returns:
            Session 列表
        """
        return [
            session 
            for (uid, _), session in self._sessions.items() 
            if uid == user_id
        ]


# ==================== InvocationContext ====================

@dataclass
class InvocationContext:
    """
    单次调用的临时上下文
    
    核心设计理念:
    - InvocationContext 是短暂的，只存在于一次调用期间
    - 它持有所有执行需要的引用
    - 用于追踪和调试
    """
    
    # 追踪信息
    invocation_id: str = field(default_factory=lambda: str(uuid4()))
    trace_id: Optional[str] = None
    
    # 引用（运行时设置）
    agent: Optional['Agent'] = None
    session: Optional[Session] = None
    
    # 用户消息（简单字符串）
    user_message: str = ""
    
    # 运行配置
    run_config: dict[str, Any] = field(default_factory=dict)
    
    # 时间戳
    start_time: float = field(default_factory=time.time)
    
    @property
    def elapsed_time(self) -> float:
        """获取已用时间（秒）"""
        return time.time() - self.start_time
