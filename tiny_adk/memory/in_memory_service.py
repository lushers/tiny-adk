"""
InMemoryService - 内存存储实现

设计理念（参考 ADK InMemoryMemoryService）：
- 仅用于开发和测试
- 使用关键词匹配（非语义搜索）
- 线程安全

特点：
- 快速原型开发
- 无外部依赖
- 数据存储在内存中，重启丢失
"""

from __future__ import annotations

import re
import threading
from typing import Optional
from uuid import uuid4

from typing_extensions import override

from .base_memory_service import BaseMemoryService, SearchResult
from .memory_entry import MemoryEntry, MemoryType


def _user_key(app_name: str, user_id: str) -> str:
    """生成用户唯一键"""
    return f"{app_name}/{user_id}"


def _extract_words(text: str) -> set[str]:
    """提取文本中的单词（小写）"""
    # 匹配中英文
    words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', text.lower())
    return set(words)


class InMemoryService(BaseMemoryService):
    """
    内存存储的记忆服务实现
    
    使用关键词匹配进行搜索，仅用于开发和测试。
    
    数据结构：
    - _memories: dict[user_key, dict[memory_id, MemoryEntry]]
    
    示例:
        service = InMemoryService()
        
        # 添加记忆
        entry = MemoryEntry(
            content="用户喜欢Python编程",
            app_name="my_app",
            user_id="u1",
        )
        memory_id = await service.add(entry)
        
        # 搜索记忆
        result = await service.search(
            "Python",
            app_name="my_app",
            user_id="u1",
        )
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        # {user_key: {memory_id: MemoryEntry}}
        self._memories: dict[str, dict[str, MemoryEntry]] = {}
    
    @override
    async def add(self, entry: MemoryEntry) -> str:
        """添加记忆到内存存储"""
        user_key = _user_key(entry.app_name, entry.user_id)
        
        # 生成 ID
        if not entry.id:
            entry.id = str(uuid4())
        
        with self._lock:
            if user_key not in self._memories:
                self._memories[user_key] = {}
            self._memories[user_key][entry.id] = entry
        
        return entry.id
    
    @override
    def _add_sync_impl(self, entry: MemoryEntry) -> str:
        """同步添加（直接实现，无需事件循环）"""
        user_key = _user_key(entry.app_name, entry.user_id)
        
        if not entry.id:
            entry.id = str(uuid4())
        
        with self._lock:
            if user_key not in self._memories:
                self._memories[user_key] = {}
            self._memories[user_key][entry.id] = entry
        
        return entry.id
    
    @override
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
        使用关键词匹配搜索记忆
        
        匹配规则：查询词中任一词出现在记忆内容中即匹配
        """
        user_key = _user_key(app_name, user_id)
        
        with self._lock:
            user_memories = self._memories.get(user_key, {})
        
        query_words = _extract_words(query)
        if not query_words:
            return SearchResult(entries=[], total_count=0)
        
        matched = []
        
        for entry in user_memories.values():
            # 过滤 session_id
            if session_id and entry.session_id != session_id:
                continue
            
            # 过滤 memory_type
            if memory_type and entry.memory_type != memory_type:
                continue
            
            # 关键词匹配
            content_words = _extract_words(entry.content)
            if not content_words:
                continue
            
            # 计算匹配分数（匹配词数 / 查询词数）
            matches = query_words & content_words
            if matches:
                score = len(matches) / len(query_words)
                if score >= score_threshold:
                    matched.append((score, entry))
        
        # 按分数和时间排序
        matched.sort(key=lambda x: (-x[0], -x[1].timestamp.timestamp()))
        
        # 限制数量
        entries = [entry for _, entry in matched[:limit]]
        
        return SearchResult(
            entries=entries,
            total_count=len(matched),
        )
    
    @override
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
        """同步搜索（直接实现）"""
        user_key = _user_key(app_name, user_id)
        
        with self._lock:
            user_memories = self._memories.get(user_key, {})
        
        query_words = _extract_words(query)
        if not query_words:
            return SearchResult(entries=[], total_count=0)
        
        matched = []
        
        for entry in user_memories.values():
            if session_id and entry.session_id != session_id:
                continue
            if memory_type and entry.memory_type != memory_type:
                continue
            
            content_words = _extract_words(entry.content)
            if not content_words:
                continue
            
            matches = query_words & content_words
            if matches:
                score = len(matches) / len(query_words)
                if score >= score_threshold:
                    matched.append((score, entry))
        
        matched.sort(key=lambda x: (-x[0], -x[1].timestamp.timestamp()))
        entries = [entry for _, entry in matched[:limit]]
        
        return SearchResult(entries=entries, total_count=len(matched))
    
    @override
    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """根据 ID 获取记忆"""
        with self._lock:
            for user_memories in self._memories.values():
                if memory_id in user_memories:
                    return user_memories[memory_id]
        return None
    
    @override
    async def clear(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
    ) -> int:
        """清空记忆"""
        user_key = _user_key(app_name, user_id)
        deleted = 0
        
        with self._lock:
            if user_key not in self._memories:
                return 0
            
            if session_id is None and memory_type is None:
                # 清空该用户的所有记忆
                deleted = len(self._memories[user_key])
                del self._memories[user_key]
            else:
                # 选择性删除
                to_delete = []
                for memory_id, entry in self._memories[user_key].items():
                    if session_id and entry.session_id != session_id:
                        continue
                    if memory_type and entry.memory_type != memory_type:
                        continue
                    to_delete.append(memory_id)
                
                for memory_id in to_delete:
                    del self._memories[user_key][memory_id]
                deleted = len(to_delete)
        
        return deleted
    
    @override
    def _clear_sync_impl(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
    ) -> int:
        """同步清空（直接实现）"""
        user_key = _user_key(app_name, user_id)
        deleted = 0
        
        with self._lock:
            if user_key not in self._memories:
                return 0
            
            if session_id is None and memory_type is None:
                deleted = len(self._memories[user_key])
                del self._memories[user_key]
            else:
                to_delete = []
                for memory_id, entry in self._memories[user_key].items():
                    if session_id and entry.session_id != session_id:
                        continue
                    if memory_type and entry.memory_type != memory_type:
                        continue
                    to_delete.append(memory_id)
                
                for memory_id in to_delete:
                    del self._memories[user_key][memory_id]
                deleted = len(to_delete)
        
        return deleted
    
    # ==================== 便利方法 ====================
    
    def count(
        self,
        app_name: str,
        user_id: str,
        session_id: Optional[str] = None,
    ) -> int:
        """统计记忆数量"""
        user_key = _user_key(app_name, user_id)
        
        with self._lock:
            user_memories = self._memories.get(user_key, {})
            
            if session_id is None:
                return len(user_memories)
            
            return sum(
                1 for entry in user_memories.values()
                if entry.session_id == session_id
            )
    
    def list_all(
        self,
        app_name: str,
        user_id: str,
        session_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
    ) -> list[MemoryEntry]:
        """列出所有记忆（用于调试）"""
        user_key = _user_key(app_name, user_id)
        
        with self._lock:
            user_memories = self._memories.get(user_key, {})
            
            result = []
            for entry in user_memories.values():
                if session_id and entry.session_id != session_id:
                    continue
                if memory_type and entry.memory_type != memory_type:
                    continue
                result.append(entry)
            
            # 按时间排序
            result.sort(key=lambda x: x.timestamp, reverse=True)
            return result

