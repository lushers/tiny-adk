"""
Memory 模块 - 为 Agent 提供记忆能力

设计理念（融合 ADK 和 CrewAI 的优点）：
- 简洁的 API 设计（参考 ADK）
- 分层记忆类型（参考 CrewAI，但更精简）
- 与 Session/Event 系统深度集成

记忆类型：
- ShortTermMemory: 会话级记忆（当前对话上下文）
- LongTermMemory: 用户级记忆（跨会话持久化）

存储后端：
- InMemoryStorage: 内存存储（开发/测试）
- VectorStorage: 向量存储（语义搜索）
- SQLiteStorage: SQLite存储（持久化）

Memory 工具：
- preload_memory_tool: 自动预加载（推荐）
- load_memory_tool: 模型主动调用
"""

from .memory_entry import MemoryEntry
from .base_memory_service import BaseMemoryService, SearchResult
from .in_memory_service import InMemoryService
from .vector_memory_service import VectorMemoryService
from .memory_manager import MemoryManager
from .memory_tools import (
    MemoryToolContext,
    PreloadMemoryTool,
    LoadMemoryTool,
    preload_memory_tool,
    load_memory_tool,
)

__all__ = [
    # 数据结构
    'MemoryEntry',
    'SearchResult',
    # 服务
    'BaseMemoryService',
    'InMemoryService',
    'VectorMemoryService',
    'MemoryManager',
    # 工具
    'MemoryToolContext',
    'PreloadMemoryTool',
    'LoadMemoryTool',
    'preload_memory_tool',
    'load_memory_tool',
]

