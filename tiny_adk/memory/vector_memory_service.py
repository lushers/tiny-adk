"""
VectorMemoryService - 向量存储实现（语义搜索）

设计理念：
- 使用向量嵌入进行语义搜索
- 支持多种 Embedding 提供者
- 可选持久化到 SQLite

特点：
- 语义理解，非精确匹配
- 支持相似度阈值过滤
- 可扩展的 Embedding 接口
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

from typing_extensions import override

from .base_memory_service import BaseMemoryService, SearchResult
from .memory_entry import MemoryEntry, MemoryType


logger = logging.getLogger(__name__)


# ==================== Embedding 类型 ====================

EmbeddingFunc = Callable[[str], list[float]]
"""Embedding 函数类型: text -> vector"""


def _default_embedding_func(text: str) -> list[float]:
    """
    默认的简单 Embedding 实现（基于字符统计）
    
    注意：仅用于测试，生产环境请使用真正的 Embedding 模型
    """
    # 简单的字符频率向量（128维）
    vector = [0.0] * 128
    for char in text.lower():
        idx = ord(char) % 128
        vector[idx] += 1.0
    
    # 归一化
    norm = sum(x * x for x in vector) ** 0.5
    if norm > 0:
        vector = [x / norm for x in vector]
    
    return vector


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """计算余弦相似度"""
    if len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


# ==================== VectorMemoryService ====================

class VectorMemoryService(BaseMemoryService):
    """
    基于向量的记忆服务实现
    
    支持：
    - 语义搜索（通过 Embedding）
    - 可选 SQLite 持久化
    - 自定义 Embedding 函数
    
    示例:
        # 使用默认 Embedding（仅测试用）
        service = VectorMemoryService()
        
        # 使用 OpenAI Embedding
        from openai import OpenAI
        client = OpenAI()
        
        def openai_embed(text: str) -> list[float]:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        
        service = VectorMemoryService(embedding_func=openai_embed)
        
        # 使用 SQLite 持久化
        service = VectorMemoryService(
            db_path="./memory.db",
            embedding_func=openai_embed,
        )
    """
    
    def __init__(
        self,
        embedding_func: Optional[EmbeddingFunc] = None,
        db_path: Optional[str] = None,
    ):
        """
        初始化向量记忆服务
        
        Args:
            embedding_func: Embedding 函数，将文本转换为向量
            db_path: SQLite 数据库路径（可选，不提供则使用内存存储）
        """
        self._embedding_func = embedding_func or _default_embedding_func
        self._db_path = db_path
        self._lock = threading.Lock()
        
        # 内存存储（db_path 为 None 时使用）
        # {user_key: {memory_id: (MemoryEntry, embedding)}}
        self._memories: dict[str, dict[str, tuple[MemoryEntry, list[float]]]] = {}
        
        # 初始化数据库
        if self._db_path:
            self._init_db()
    
    def _init_db(self) -> None:
        """初始化 SQLite 数据库"""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    app_name TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    author TEXT,
                    importance REAL DEFAULT 1.0,
                    embedding TEXT NOT NULL,
                    metadata TEXT,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_user 
                ON memories(app_name, user_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session 
                ON memories(app_name, user_id, session_id)
            """)
            conn.commit()
    
    def _user_key(self, app_name: str, user_id: str) -> str:
        """生成用户唯一键"""
        return f"{app_name}/{user_id}"
    
    @override
    async def add(self, entry: MemoryEntry) -> str:
        """添加记忆（带向量嵌入）"""
        # 生成 ID
        if not entry.id:
            entry.id = str(uuid4())
        
        # 计算 Embedding
        embedding = self._embedding_func(entry.content)
        entry.embedding = embedding
        
        if self._db_path:
            self._save_to_db(entry, embedding)
        else:
            self._save_to_memory(entry, embedding)
        
        return entry.id
    
    @override
    def _add_sync_impl(self, entry: MemoryEntry) -> str:
        """同步添加"""
        if not entry.id:
            entry.id = str(uuid4())
        
        embedding = self._embedding_func(entry.content)
        entry.embedding = embedding
        
        if self._db_path:
            self._save_to_db(entry, embedding)
        else:
            self._save_to_memory(entry, embedding)
        
        return entry.id
    
    def _save_to_memory(
        self, entry: MemoryEntry, embedding: list[float]
    ) -> None:
        """保存到内存"""
        user_key = self._user_key(entry.app_name, entry.user_id)
        
        with self._lock:
            if user_key not in self._memories:
                self._memories[user_key] = {}
            self._memories[user_key][entry.id] = (entry, embedding)
    
    def _save_to_db(
        self, entry: MemoryEntry, embedding: list[float]
    ) -> None:
        """保存到 SQLite"""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO memories 
                (id, app_name, user_id, session_id, content, memory_type, 
                 author, importance, embedding, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id,
                entry.app_name,
                entry.user_id,
                entry.session_id,
                entry.content,
                entry.memory_type.value,
                entry.author,
                entry.importance,
                json.dumps(embedding),
                json.dumps(entry.metadata),
                entry.timestamp.isoformat(),
            ))
            conn.commit()
    
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
        """语义搜索记忆"""
        # 计算查询向量
        query_embedding = self._embedding_func(query)
        
        if self._db_path:
            return self._search_from_db(
                query_embedding,
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                memory_type=memory_type,
                limit=limit,
                score_threshold=score_threshold,
            )
        else:
            return self._search_from_memory(
                query_embedding,
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                memory_type=memory_type,
                limit=limit,
                score_threshold=score_threshold,
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
        """同步搜索"""
        query_embedding = self._embedding_func(query)
        
        if self._db_path:
            return self._search_from_db(
                query_embedding,
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                memory_type=memory_type,
                limit=limit,
                score_threshold=score_threshold,
            )
        else:
            return self._search_from_memory(
                query_embedding,
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                memory_type=memory_type,
                limit=limit,
                score_threshold=score_threshold,
            )
    
    def _search_from_memory(
        self,
        query_embedding: list[float],
        *,
        app_name: str,
        user_id: str,
        session_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> SearchResult:
        """从内存搜索"""
        user_key = self._user_key(app_name, user_id)
        
        with self._lock:
            user_memories = self._memories.get(user_key, {})
        
        scored_entries = []
        
        for entry, embedding in user_memories.values():
            # 过滤
            if session_id and entry.session_id != session_id:
                continue
            if memory_type and entry.memory_type != memory_type:
                continue
            
            # 计算相似度
            score = _cosine_similarity(query_embedding, embedding)
            if score >= score_threshold:
                scored_entries.append((score, entry))
        
        # 排序
        scored_entries.sort(key=lambda x: -x[0])
        
        entries = [entry for _, entry in scored_entries[:limit]]
        
        return SearchResult(
            entries=entries,
            total_count=len(scored_entries),
        )
    
    def _search_from_db(
        self,
        query_embedding: list[float],
        *,
        app_name: str,
        user_id: str,
        session_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> SearchResult:
        """从数据库搜索"""
        # 构建查询
        sql = """
            SELECT id, content, memory_type, author, importance, 
                   embedding, metadata, timestamp, session_id
            FROM memories 
            WHERE app_name = ? AND user_id = ?
        """
        params: list = [app_name, user_id]
        
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        
        if memory_type:
            sql += " AND memory_type = ?"
            params.append(memory_type.value)
        
        scored_entries = []
        
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            
            for row in cursor.fetchall():
                embedding = json.loads(row[5])
                score = _cosine_similarity(query_embedding, embedding)
                
                if score >= score_threshold:
                    entry = MemoryEntry(
                        id=row[0],
                        content=row[1],
                        memory_type=MemoryType(row[2]),
                        author=row[3],
                        importance=row[4],
                        metadata=json.loads(row[6]) if row[6] else {},
                        timestamp=datetime.fromisoformat(row[7]),
                        session_id=row[8],
                        app_name=app_name,
                        user_id=user_id,
                    )
                    scored_entries.append((score, entry))
        
        scored_entries.sort(key=lambda x: -x[0])
        entries = [entry for _, entry in scored_entries[:limit]]
        
        return SearchResult(
            entries=entries,
            total_count=len(scored_entries),
        )
    
    @override
    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """根据 ID 获取记忆"""
        if self._db_path:
            return self._get_from_db(memory_id)
        else:
            with self._lock:
                for user_memories in self._memories.values():
                    if memory_id in user_memories:
                        return user_memories[memory_id][0]
            return None
    
    def _get_from_db(self, memory_id: str) -> Optional[MemoryEntry]:
        """从数据库获取"""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, app_name, user_id, session_id, content, 
                       memory_type, author, importance, metadata, timestamp
                FROM memories WHERE id = ?
            """, (memory_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return MemoryEntry(
                id=row[0],
                app_name=row[1],
                user_id=row[2],
                session_id=row[3],
                content=row[4],
                memory_type=MemoryType(row[5]),
                author=row[6],
                importance=row[7],
                metadata=json.loads(row[8]) if row[8] else {},
                timestamp=datetime.fromisoformat(row[9]),
            )
    
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
        if self._db_path:
            return self._clear_from_db(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                memory_type=memory_type,
            )
        else:
            return self._clear_from_memory(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                memory_type=memory_type,
            )
    
    @override
    def _clear_sync_impl(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
    ) -> int:
        """同步清空"""
        if self._db_path:
            return self._clear_from_db(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                memory_type=memory_type,
            )
        else:
            return self._clear_from_memory(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                memory_type=memory_type,
            )
    
    def _clear_from_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
    ) -> int:
        """从内存清空"""
        user_key = self._user_key(app_name, user_id)
        deleted = 0
        
        with self._lock:
            if user_key not in self._memories:
                return 0
            
            if session_id is None and memory_type is None:
                deleted = len(self._memories[user_key])
                del self._memories[user_key]
            else:
                to_delete = []
                for memory_id, (entry, _) in self._memories[user_key].items():
                    if session_id and entry.session_id != session_id:
                        continue
                    if memory_type and entry.memory_type != memory_type:
                        continue
                    to_delete.append(memory_id)
                
                for memory_id in to_delete:
                    del self._memories[user_key][memory_id]
                deleted = len(to_delete)
        
        return deleted
    
    def _clear_from_db(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
    ) -> int:
        """从数据库清空"""
        sql = "DELETE FROM memories WHERE app_name = ? AND user_id = ?"
        params: list = [app_name, user_id]
        
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        
        if memory_type:
            sql += " AND memory_type = ?"
            params.append(memory_type.value)
        
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            deleted = cursor.rowcount
            conn.commit()
        
        return deleted
    
    # ==================== Embedding 辅助方法 ====================
    
    def set_embedding_func(self, func: EmbeddingFunc) -> None:
        """设置 Embedding 函数"""
        self._embedding_func = func
    
    @staticmethod
    def create_openai_embedding_func(
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
    ) -> EmbeddingFunc:
        """
        创建 OpenAI Embedding 函数
        
        Args:
            api_key: OpenAI API Key（可选，默认从环境变量读取）
            model: Embedding 模型名称
            
        Returns:
            Embedding 函数
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "请安装 openai 包: pip install openai"
            )
        
        client = OpenAI(api_key=api_key)
        
        def embed(text: str) -> list[float]:
            response = client.embeddings.create(
                model=model,
                input=text,
            )
            return response.data[0].embedding
        
        return embed

