"""
BaseMemoryService - 记忆服务抽象基类

设计理念（参考 ADK）：
- 简洁的 API：add + search + clear
- 按 (app_name, user_id) 隔离
- 支持异步和同步操作
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from .memory_entry import MemoryEntry, MemoryType

if TYPE_CHECKING:
    from ..session import Session


@dataclass
class SearchResult:
    """搜索结果"""
    entries: list[MemoryEntry] = field(default_factory=list)
    """匹配的记忆条目"""
    
    total_count: int = 0
    """总匹配数（可能大于返回的条目数）"""
    
    def __iter__(self):
        return iter(self.entries)
    
    def __len__(self):
        return len(self.entries)
    
    def to_context_string(self, max_entries: int = 5) -> str:
        """
        转换为可插入 prompt 的上下文字符串
        
        用于将检索到的记忆注入到 LLM 的上下文中
        """
        if not self.entries:
            return ""
        
        lines = ["[Relevant Memories]"]
        for i, entry in enumerate(self.entries[:max_entries]):
            author = f" ({entry.author})" if entry.author else ""
            time_str = entry.timestamp.strftime("%Y-%m-%d %H:%M")
            lines.append(f"- [{time_str}]{author}: {entry.content}")
        
        if len(self.entries) > max_entries:
            lines.append(f"... and {len(self.entries) - max_entries} more")
        
        return "\n".join(lines)


class BaseMemoryService(ABC):
    """
    记忆服务抽象基类
    
    核心方法：
    - add: 添加记忆
    - add_session: 将整个 Session 添加到记忆
    - search: 搜索记忆
    - clear: 清空记忆
    
    设计原则：
    - 所有操作都需要 (app_name, user_id) 标识
    - 支持按 session_id 进一步隔离（短期记忆）
    - 同时提供异步和同步版本
    """
    
    # ==================== 添加记忆 ====================
    
    @abstractmethod
    async def add(
        self,
        entry: MemoryEntry,
    ) -> str:
        """
        添加单条记忆
        
        Args:
            entry: 记忆条目
            
        Returns:
            记忆 ID
        """
        pass
    
    def add_sync(self, entry: MemoryEntry) -> str:
        """同步版本的添加"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 在已有事件循环中，直接调用内部实现
                return self._add_sync_impl(entry)
            return loop.run_until_complete(self.add(entry))
        except RuntimeError:
            return asyncio.run(self.add(entry))
    
    def _add_sync_impl(self, entry: MemoryEntry) -> str:
        """同步实现（子类可重写以优化）"""
        import asyncio
        return asyncio.run(self.add(entry))
    
    async def add_session(
        self,
        session: 'Session',
        memory_type: MemoryType = MemoryType.SHORT_TERM,
    ) -> list[str]:
        """
        将整个 Session 的事件添加到记忆
        
        Args:
            session: Session 实例
            memory_type: 记忆类型
            
        Returns:
            添加的记忆 ID 列表
        """
        from ..events import EventType
        
        ids = []
        for event in session.events:
            # 只添加有实际内容的事件
            if not event.content:
                continue
            
            # 跳过部分事件类型
            if event.event_type in (EventType.MODEL_REQUEST, EventType.MODEL_RESPONSE_DELTA):
                continue
            
            entry = MemoryEntry.from_event(
                event,
                app_name=session.app_name,
                user_id=session.user_id,
                session_id=session.session_id,
            )
            entry.memory_type = memory_type
            
            memory_id = await self.add(entry)
            ids.append(memory_id)
        
        return ids
    
    def add_session_sync(
        self,
        session: 'Session',
        memory_type: MemoryType = MemoryType.SHORT_TERM,
    ) -> list[str]:
        """同步版本的添加 Session"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return self._add_session_sync_impl(session, memory_type)
            return loop.run_until_complete(self.add_session(session, memory_type))
        except RuntimeError:
            return asyncio.run(self.add_session(session, memory_type))
    
    def _add_session_sync_impl(
        self,
        session: 'Session',
        memory_type: MemoryType = MemoryType.SHORT_TERM,
    ) -> list[str]:
        """同步实现"""
        from ..events import EventType
        
        ids = []
        for event in session.events:
            if not event.content:
                continue
            if event.event_type in (EventType.MODEL_REQUEST, EventType.MODEL_RESPONSE_DELTA):
                continue
            
            entry = MemoryEntry.from_event(
                event,
                app_name=session.app_name,
                user_id=session.user_id,
                session_id=session.session_id,
            )
            entry.memory_type = memory_type
            
            memory_id = self.add_sync(entry)
            ids.append(memory_id)
        
        return ids
    
    # ==================== 搜索记忆 ====================
    
    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        app_name: str,
        user_id: str,
        session_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> SearchResult:
        """
        搜索记忆
        
        Args:
            query: 搜索查询（关键词或语义）
            app_name: 应用名称
            user_id: 用户 ID
            session_id: 会话 ID（可选，用于限制搜索范围）
            memory_type: 记忆类型过滤
            limit: 返回数量限制
            score_threshold: 相关性阈值（0-1）
            
        Returns:
            SearchResult 包含匹配的记忆条目
        """
        pass
    
    def search_sync(
        self,
        query: str,
        *,
        app_name: str,
        user_id: str,
        session_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> SearchResult:
        """同步版本的搜索"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return self._search_sync_impl(
                    query,
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id,
                    memory_type=memory_type,
                    limit=limit,
                    score_threshold=score_threshold,
                )
            return loop.run_until_complete(self.search(
                query,
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                memory_type=memory_type,
                limit=limit,
                score_threshold=score_threshold,
            ))
        except RuntimeError:
            return asyncio.run(self.search(
                query,
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                memory_type=memory_type,
                limit=limit,
                score_threshold=score_threshold,
            ))
    
    def _search_sync_impl(
        self,
        query: str,
        *,
        app_name: str,
        user_id: str,
        session_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> SearchResult:
        """同步实现（子类可重写）"""
        import asyncio
        return asyncio.run(self.search(
            query,
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            memory_type=memory_type,
            limit=limit,
            score_threshold=score_threshold,
        ))
    
    # ==================== 获取记忆 ====================
    
    async def get(
        self,
        memory_id: str,
    ) -> Optional[MemoryEntry]:
        """
        根据 ID 获取记忆
        
        Args:
            memory_id: 记忆 ID
            
        Returns:
            记忆条目，不存在返回 None
        """
        # 默认实现：子类应重写以提供更高效的实现
        return None
    
    def get_sync(self, memory_id: str) -> Optional[MemoryEntry]:
        """同步版本的获取"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return None  # 默认实现
            return loop.run_until_complete(self.get(memory_id))
        except RuntimeError:
            return asyncio.run(self.get(memory_id))
    
    # ==================== 清空记忆 ====================
    
    @abstractmethod
    async def clear(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
    ) -> int:
        """
        清空记忆
        
        Args:
            app_name: 应用名称
            user_id: 用户 ID
            session_id: 会话 ID（可选，只清空特定会话）
            memory_type: 记忆类型（可选，只清空特定类型）
            
        Returns:
            删除的记忆数量
        """
        pass
    
    def clear_sync(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
    ) -> int:
        """同步版本的清空"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return self._clear_sync_impl(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id,
                    memory_type=memory_type,
                )
            return loop.run_until_complete(self.clear(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                memory_type=memory_type,
            ))
        except RuntimeError:
            return asyncio.run(self.clear(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                memory_type=memory_type,
            ))
    
    def _clear_sync_impl(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
    ) -> int:
        """同步实现"""
        import asyncio
        return asyncio.run(self.clear(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            memory_type=memory_type,
        ))

