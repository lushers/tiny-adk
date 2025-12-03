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
    Session 持久化服务（参考 ADK 设计）
    
    设计原则:
    - get_session: 只获取，不存在返回 None
    - create_session: 显式创建
    - append_event: 原子操作追加事件
    
    这种分离使得:
    1. 行为可预测，没有隐式副作用
    2. 易于扩展到分布式存储（PostgreSQL、Redis 等）
    3. 调用方明确控制 Session 生命周期
    """
    
    def __init__(self):
        self._sessions: dict[tuple[str, str], Session] = {}
    
    # ==================== 获取（纯获取，不创建）====================
    
    async def get_session(
        self, 
        user_id: str, 
        session_id: str
    ) -> Optional[Session]:
        """
        获取 Session
        
        Args:
            user_id: 用户 ID
            session_id: 会话 ID
            
        Returns:
            Session 实例，不存在返回 None
        """
        key = (user_id, session_id)
        return self._sessions.get(key)
    
    def get_session_sync(
        self, 
        user_id: str, 
        session_id: str
    ) -> Optional[Session]:
        """同步版本的获取 Session"""
        key = (user_id, session_id)
        return self._sessions.get(key)
    
    # ==================== 创建（显式创建）====================
    
    async def create_session(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        state: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Session:
        """
        创建新 Session
        
        Args:
            user_id: 用户 ID
            session_id: 会话 ID（可选，不提供则自动生成）
            state: 初始状态
            metadata: 元数据
            
        Returns:
            新创建的 Session
            
        Raises:
            ValueError: 如果 Session 已存在
        """
        session_id = session_id or str(uuid4())
        key = (user_id, session_id)
        
        if key in self._sessions:
            raise ValueError(f"Session already exists: {session_id}")
        
        session = Session(
            user_id=user_id,
            session_id=session_id,
            state=state or {},
            metadata=metadata or {},
        )
        self._sessions[key] = session
        return session
    
    def create_session_sync(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        state: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Session:
        """同步版本的创建 Session"""
        session_id = session_id or str(uuid4())
        key = (user_id, session_id)
        
        if key in self._sessions:
            raise ValueError(f"Session already exists: {session_id}")
        
        session = Session(
            user_id=user_id,
            session_id=session_id,
            state=state or {},
            metadata=metadata or {},
        )
        self._sessions[key] = session
        return session
    
    # ==================== 追加事件（原子操作）====================
    
    async def append_event(
        self, 
        session: Session, 
        event: Event
    ) -> None:
        """
        追加单个事件到 Session（原子操作）
        
        这是核心的事件追加方法，而不是批量保存整个 Session。
        对于持久化存储，这里会是一个增量写入（如 INSERT INTO events）
        而不是全量更新（如 UPDATE session SET events = ...）
        
        Args:
            session: Session 实例
            event: 要追加的事件
        """
        session.add_event(event)
        # 内存存储不需要额外操作
        # 对于数据库存储，这里会是: INSERT INTO events (session_id, ...) VALUES (...)
    
    def append_event_sync(
        self, 
        session: Session, 
        event: Event
    ) -> None:
        """同步版本的追加事件"""
        session.add_event(event)
    
    # ==================== 删除 ====================
    
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
    
    def delete_session_sync(
        self, 
        user_id: str, 
        session_id: str
    ) -> bool:
        """同步版本的删除 Session"""
        key = (user_id, session_id)
        if key in self._sessions:
            del self._sessions[key]
            return True
        return False
    
    # ==================== 列表查询 ====================
    
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
    
    def list_sessions_sync(self, user_id: str) -> list[Session]:
        """同步版本的列出 Session"""
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
