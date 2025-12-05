"""
MemoryManager - 记忆管理器（聚合器）

设计理念（参考 CrewAI ContextualMemory）：
- 统一管理多种记忆类型
- 自动构建上下文
- 与 Agent 执行流程集成

功能：
- 聚合短期记忆（会话级）和长期记忆（用户级）
- 自动提取重要信息存入长期记忆
- 为 Agent 构建相关上下文
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from .base_memory_service import BaseMemoryService, SearchResult
from .memory_entry import MemoryEntry, MemoryType
from .in_memory_service import InMemoryService


if TYPE_CHECKING:
    from ..session import Session


logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """记忆配置"""
    
    # 短期记忆配置
    short_term_enabled: bool = True
    """是否启用短期记忆"""
    
    short_term_limit: int = 20
    """短期记忆搜索限制"""
    
    # 长期记忆配置
    long_term_enabled: bool = True
    """是否启用长期记忆"""
    
    long_term_limit: int = 10
    """长期记忆搜索限制"""
    
    # 自动提取配置
    auto_extract: bool = False
    """是否自动提取重要信息到长期记忆"""
    
    importance_threshold: float = 0.7
    """重要性阈值（用于自动提取）"""
    
    # 搜索配置
    score_threshold: float = 0.3
    """搜索相关性阈值"""


class MemoryManager:
    """
    记忆管理器 - 聚合并管理多种记忆类型
    
    核心功能：
    1. 统一接口管理短期/长期记忆
    2. 自动为 Agent 构建相关上下文
    3. 会话结束时将重要信息转入长期记忆
    
    示例:
        # 创建记忆管理器
        manager = MemoryManager(
            short_term=InMemoryService(),
            long_term=VectorMemoryService(db_path="./memory.db"),
        )
        
        # 添加记忆
        manager.add(
            content="用户喜欢Python编程",
            app_name="my_app",
            user_id="u1",
            memory_type=MemoryType.LONG_TERM,
        )
        
        # 构建上下文（用于注入到 prompt）
        context = manager.build_context(
            query="帮我写Python代码",
            app_name="my_app",
            user_id="u1",
            session_id="s1",
        )
        print(context)
        # [Relevant Memories]
        # - [2024-01-01 10:00] (user): 用户喜欢Python编程
    """
    
    def __init__(
        self,
        short_term: Optional[BaseMemoryService] = None,
        long_term: Optional[BaseMemoryService] = None,
        config: Optional[MemoryConfig] = None,
    ):
        """
        初始化记忆管理器
        
        Args:
            short_term: 短期记忆服务（默认 InMemoryService）
            long_term: 长期记忆服务（默认与短期共享）
            config: 记忆配置
        """
        self.short_term = short_term or InMemoryService()
        self.long_term = long_term or self.short_term
        self.config = config or MemoryConfig()
    
    # ==================== 添加记忆 ====================
    
    async def add(
        self,
        content: str,
        *,
        app_name: str,
        user_id: str,
        session_id: Optional[str] = None,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        author: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        importance: float = 1.0,
    ) -> str:
        """
        添加记忆
        
        Args:
            content: 记忆内容
            app_name: 应用名称
            user_id: 用户 ID
            session_id: 会话 ID（短期记忆必需）
            memory_type: 记忆类型
            author: 记忆来源
            metadata: 自定义元数据
            importance: 重要性权重
            
        Returns:
            记忆 ID
        """
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            author=author,
            metadata=metadata or {},
            importance=importance,
        )
        
        # 根据类型选择存储
        if memory_type == MemoryType.LONG_TERM:
            return await self.long_term.add(entry)
        else:
            return await self.short_term.add(entry)
    
    def add_sync(
        self,
        content: str,
        *,
        app_name: str,
        user_id: str,
        session_id: Optional[str] = None,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        author: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        importance: float = 1.0,
    ) -> str:
        """同步版本的添加"""
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            author=author,
            metadata=metadata or {},
            importance=importance,
        )
        
        if memory_type == MemoryType.LONG_TERM:
            return self.long_term.add_sync(entry)
        else:
            return self.short_term.add_sync(entry)
    
    async def add_from_session(
        self,
        session: 'Session',
        memory_type: MemoryType = MemoryType.SHORT_TERM,
    ) -> list[str]:
        """
        将 Session 的事件添加到记忆
        
        Args:
            session: Session 实例
            memory_type: 记忆类型
            
        Returns:
            添加的记忆 ID 列表
        """
        service = self.long_term if memory_type == MemoryType.LONG_TERM else self.short_term
        return await service.add_session(session, memory_type)
    
    def add_from_session_sync(
        self,
        session: 'Session',
        memory_type: MemoryType = MemoryType.SHORT_TERM,
    ) -> list[str]:
        """同步版本"""
        service = self.long_term if memory_type == MemoryType.LONG_TERM else self.short_term
        return service.add_session_sync(session, memory_type)
    
    # ==================== 搜索记忆 ====================
    
    async def search(
        self,
        query: str,
        *,
        app_name: str,
        user_id: str,
        session_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> SearchResult:
        """
        搜索记忆
        
        Args:
            query: 搜索查询
            app_name: 应用名称
            user_id: 用户 ID
            session_id: 会话 ID（可选）
            memory_type: 记忆类型（可选，不指定则搜索所有类型）
            limit: 返回数量限制
            
        Returns:
            SearchResult 包含匹配的记忆
        """
        all_entries = []
        
        # 搜索短期记忆
        if self.config.short_term_enabled and memory_type in (None, MemoryType.SHORT_TERM, MemoryType.EVENT):
            short_result = await self.short_term.search(
                query,
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                memory_type=memory_type if memory_type != MemoryType.LONG_TERM else None,
                limit=self.config.short_term_limit,
                score_threshold=self.config.score_threshold,
            )
            all_entries.extend(short_result.entries)
        
        # 搜索长期记忆（如果是不同的服务）
        if (self.config.long_term_enabled 
            and self.long_term is not self.short_term
            and memory_type in (None, MemoryType.LONG_TERM)):
            long_result = await self.long_term.search(
                query,
                app_name=app_name,
                user_id=user_id,
                session_id=None,  # 长期记忆不按 session 过滤
                memory_type=MemoryType.LONG_TERM if memory_type is None else memory_type,
                limit=self.config.long_term_limit,
                score_threshold=self.config.score_threshold,
            )
            all_entries.extend(long_result.entries)
        
        # 去重并按时间排序
        seen_ids = set()
        unique_entries = []
        for entry in all_entries:
            if entry.id not in seen_ids:
                seen_ids.add(entry.id)
                unique_entries.append(entry)
        
        unique_entries.sort(key=lambda x: x.timestamp, reverse=True)
        
        return SearchResult(
            entries=unique_entries[:limit],
            total_count=len(unique_entries),
        )
    
    def search_sync(
        self,
        query: str,
        *,
        app_name: str,
        user_id: str,
        session_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> SearchResult:
        """同步版本的搜索"""
        all_entries = []
        
        if self.config.short_term_enabled and memory_type in (None, MemoryType.SHORT_TERM, MemoryType.EVENT):
            short_result = self.short_term.search_sync(
                query,
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                memory_type=memory_type if memory_type != MemoryType.LONG_TERM else None,
                limit=self.config.short_term_limit,
                score_threshold=self.config.score_threshold,
            )
            all_entries.extend(short_result.entries)
        
        if (self.config.long_term_enabled 
            and self.long_term is not self.short_term
            and memory_type in (None, MemoryType.LONG_TERM)):
            long_result = self.long_term.search_sync(
                query,
                app_name=app_name,
                user_id=user_id,
                session_id=None,
                memory_type=MemoryType.LONG_TERM if memory_type is None else memory_type,
                limit=self.config.long_term_limit,
                score_threshold=self.config.score_threshold,
            )
            all_entries.extend(long_result.entries)
        
        seen_ids = set()
        unique_entries = []
        for entry in all_entries:
            if entry.id not in seen_ids:
                seen_ids.add(entry.id)
                unique_entries.append(entry)
        
        unique_entries.sort(key=lambda x: x.timestamp, reverse=True)
        
        return SearchResult(
            entries=unique_entries[:limit],
            total_count=len(unique_entries),
        )
    
    # ==================== 构建上下文 ====================
    
    async def build_context(
        self,
        query: str,
        *,
        app_name: str,
        user_id: str,
        session_id: Optional[str] = None,
        max_entries: int = 5,
    ) -> str:
        """
        为 Agent 构建记忆上下文
        
        根据查询搜索相关记忆，并格式化为可注入 prompt 的字符串
        
        Args:
            query: 当前查询/任务描述
            app_name: 应用名称
            user_id: 用户 ID
            session_id: 会话 ID
            max_entries: 最大返回条目数
            
        Returns:
            格式化的上下文字符串
        """
        result = await self.search(
            query,
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            limit=max_entries,
        )
        
        return result.to_context_string(max_entries)
    
    def build_context_sync(
        self,
        query: str,
        *,
        app_name: str,
        user_id: str,
        session_id: Optional[str] = None,
        max_entries: int = 5,
    ) -> str:
        """同步版本的构建上下文"""
        result = self.search_sync(
            query,
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            limit=max_entries,
        )
        
        return result.to_context_string(max_entries)
    
    # ==================== 清空记忆 ====================
    
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
            session_id: 会话 ID（可选）
            memory_type: 记忆类型（可选）
            
        Returns:
            删除的记忆数量
        """
        count = 0
        
        # 清空短期记忆
        if memory_type in (None, MemoryType.SHORT_TERM, MemoryType.EVENT):
            count += await self.short_term.clear(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                memory_type=memory_type,
            )
        
        # 清空长期记忆（如果是不同服务）
        if (self.long_term is not self.short_term
            and memory_type in (None, MemoryType.LONG_TERM)):
            count += await self.long_term.clear(
                app_name=app_name,
                user_id=user_id,
                session_id=None,  # 长期记忆不按 session 过滤
                memory_type=MemoryType.LONG_TERM if memory_type is None else memory_type,
            )
        
        return count
    
    def clear_sync(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
    ) -> int:
        """同步版本的清空"""
        count = 0
        
        if memory_type in (None, MemoryType.SHORT_TERM, MemoryType.EVENT):
            count += self.short_term.clear_sync(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                memory_type=memory_type,
            )
        
        if (self.long_term is not self.short_term
            and memory_type in (None, MemoryType.LONG_TERM)):
            count += self.long_term.clear_sync(
                app_name=app_name,
                user_id=user_id,
                session_id=None,
                memory_type=MemoryType.LONG_TERM if memory_type is None else memory_type,
            )
        
        return count
    
    # ==================== 会话生命周期 ====================
    
    async def on_session_end(
        self,
        session: 'Session',
        extract_to_long_term: bool = False,
    ) -> None:
        """
        会话结束时的回调
        
        可选择将重要信息提取到长期记忆
        
        Args:
            session: 结束的会话
            extract_to_long_term: 是否提取到长期记忆
        """
        if extract_to_long_term and self.config.auto_extract:
            # 这里可以添加提取重要信息的逻辑
            # 例如使用 LLM 总结对话要点
            logger.info(
                f"Session {session.session_id} ended. "
                f"Auto-extract to long-term memory is enabled but not implemented."
            )
    
    def on_session_end_sync(
        self,
        session: 'Session',
        extract_to_long_term: bool = False,
    ) -> None:
        """同步版本"""
        if extract_to_long_term and self.config.auto_extract:
            logger.info(
                f"Session {session.session_id} ended. "
                f"Auto-extract to long-term memory is enabled but not implemented."
            )

