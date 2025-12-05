"""
Memory 工具 - 让 Agent 能够访问记忆

提供两种核心工具：
1. PreloadMemoryTool: 自动预加载，在每次 LLM 请求前注入相关记忆
2. LoadMemoryTool: 让模型主动调用搜索记忆

参考 ADK 设计：
- preload_memory: process_llm_request 钩子，自动执行
- load_memory: 作为普通工具供模型调用
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from ..tools import BaseTool

if TYPE_CHECKING:
    from .base_memory_service import BaseMemoryService, SearchResult


logger = logging.getLogger(__name__)


# ==================== ToolContext: 工具执行上下文 ====================

@dataclass
class MemoryToolContext:
    """
    Memory 工具上下文
    
    提供工具访问 memory 的能力。在工具执行时传入。
    
    设计参考 ADK 的 ToolContext.search_memory()
    """
    
    memory_service: Optional['BaseMemoryService'] = None
    """Memory 服务实例"""
    
    app_name: str = ""
    """应用名称"""
    
    user_id: str = ""
    """用户 ID"""
    
    session_id: Optional[str] = None
    """会话 ID"""
    
    user_query: str = ""
    """当前用户查询（用于 preload）"""
    
    async def search_memory(self, query: str) -> 'SearchResult':
        """
        搜索记忆（跨会话）
        
        注意：Memory 是跨会话的长期记忆，搜索时**不**按 session_id 过滤。
        这样可以检索到所有历史会话中的相关信息。
        
        Args:
            query: 搜索查询
            
        Returns:
            SearchResult 包含匹配的记忆
        """
        from .base_memory_service import SearchResult
        
        if not self.memory_service:
            logger.warning("Memory service not configured")
            return SearchResult(entries=[], total_count=0)
        
        # 注意：不传递 session_id，以便跨会话搜索
        return await self.memory_service.search(
            query,
            app_name=self.app_name,
            user_id=self.user_id,
            # session_id=None,  # 跨会话搜索
        )
    
    def search_memory_sync(self, query: str) -> 'SearchResult':
        """同步版本的搜索（跨会话）"""
        from .base_memory_service import SearchResult
        
        if not self.memory_service:
            logger.warning("Memory service not configured")
            return SearchResult(entries=[], total_count=0)
        
        # 注意：不传递 session_id，以便跨会话搜索
        return self.memory_service.search_sync(
            query,
            app_name=self.app_name,
            user_id=self.user_id,
            # session_id=None,  # 跨会话搜索
        )


# ==================== PreloadMemoryTool: 自动预加载 ====================

class PreloadMemoryTool(BaseTool):
    """
    自动预加载记忆工具
    
    特点：
    - 不会暴露给模型调用
    - 在每次 LLM 请求前自动执行
    - 将相关记忆注入到系统指令中
    
    使用方式：
        agent = Agent(
            tools=[preload_memory_tool],  # 添加到工具列表
            ...
        )
    
    工作原理：
        1. 在 Flow 发送 LLM 请求前，检测到 PreloadMemoryTool
        2. 调用 process_llm_request() 获取记忆
        3. 将记忆注入到系统指令中
        4. 模型直接使用上下文回答，无需调用工具
    """
    
    name: str = "preload_memory"
    description: str = "Automatically preload relevant memories (internal use only)"
    
    # 标记：这是一个预处理工具，不暴露给模型
    is_preprocessing: bool = True
    
    async def process_llm_request(
        self,
        context: MemoryToolContext,
    ) -> Optional[str]:
        """
        处理 LLM 请求前的预加载
        
        Args:
            context: Memory 工具上下文
            
        Returns:
            要注入到 prompt 的记忆文本，没有记忆则返回 None
        """
        user_query = context.user_query
        if not user_query:
            return None
        
        try:
            result = await context.search_memory(user_query)
        except Exception as e:
            logger.warning(f"Failed to preload memory: {e}")
            return None
        
        if not result.entries:
            return None
        
        # 格式化记忆
        memory_lines = []
        for entry in result.entries:
            time_str = entry.timestamp.strftime("%Y-%m-%d %H:%M") if entry.timestamp else ""
            author_str = f"{entry.author}: " if entry.author else ""
            
            if time_str:
                memory_lines.append(f"[{time_str}] {author_str}{entry.content}")
            else:
                memory_lines.append(f"{author_str}{entry.content}")
        
        if not memory_lines:
            return None
        
        # 构建注入文本
        memory_text = "\n".join(memory_lines)
        return f"""The following content is from your previous conversations with the user.
They may be useful for answering the user's current query.
<PAST_CONVERSATIONS>
{memory_text}
</PAST_CONVERSATIONS>
"""
    
    def process_llm_request_sync(
        self,
        context: MemoryToolContext,
    ) -> Optional[str]:
        """同步版本的预加载"""
        user_query = context.user_query
        if not user_query:
            return None
        
        try:
            result = context.search_memory_sync(user_query)
        except Exception as e:
            logger.warning(f"Failed to preload memory: {e}")
            return None
        
        if not result.entries:
            return None
        
        memory_lines = []
        for entry in result.entries:
            time_str = entry.timestamp.strftime("%Y-%m-%d %H:%M") if entry.timestamp else ""
            author_str = f"{entry.author}: " if entry.author else ""
            
            if time_str:
                memory_lines.append(f"[{time_str}] {author_str}{entry.content}")
            else:
                memory_lines.append(f"{author_str}{entry.content}")
        
        if not memory_lines:
            return None
        
        memory_text = "\n".join(memory_lines)
        return f"""The following content is from your previous conversations with the user.
They may be useful for answering the user's current query.
<PAST_CONVERSATIONS>
{memory_text}
</PAST_CONVERSATIONS>
"""
    
    async def run_async(self, args: dict, context: Any = None) -> Any:
        """PreloadMemoryTool 不应该被模型调用"""
        return {"error": "This tool is for internal use only"}
    
    def to_function_declaration(self) -> dict[str, Any]:
        """PreloadMemoryTool 不生成函数声明（不暴露给模型）"""
        return {}


# ==================== LoadMemoryTool: 模型主动调用 ====================

class LoadMemoryTool(BaseTool):
    """
    加载记忆工具（模型主动调用）
    
    特点：
    - 作为普通工具暴露给模型
    - 模型决定何时需要查询记忆
    - 返回搜索结果供模型使用
    
    使用方式：
        agent = Agent(
            tools=[load_memory_tool],  # 添加到工具列表
            instruction="当需要查找历史信息时，使用 load_memory 工具。",
        )
    
    vs PreloadMemoryTool:
        - LoadMemoryTool: 模型主动调用，可能需要额外一轮交互
        - PreloadMemoryTool: 自动执行，零延迟，100%可靠
    """
    
    name: str = "load_memory"
    description: str = "Search your memory for past conversations with the user. Use this when you need to recall previous information."
    
    parameters: dict[str, Any] = {
        'query': {
            'type': 'str',
            'description': 'Search query to find relevant memories',
        },
    }
    
    # 注入的系统指令
    memory_instruction: str = """
You have memory capabilities. When the user asks about something from previous conversations, 
use the load_memory tool to search for relevant information.
"""
    
    async def run_async(self, args: dict, context: Any = None) -> dict[str, Any]:
        """
        执行记忆搜索
        
        Args:
            args: 包含 'query' 参数
            context: MemoryToolContext 实例
            
        Returns:
            搜索结果
        """
        query = args.get('query', '')
        if not query:
            return {"error": "query is required"}
        
        if not isinstance(context, MemoryToolContext):
            return {"error": "Memory context not available"}
        
        try:
            result = await context.search_memory(query)
            
            # 格式化结果
            memories = []
            for entry in result.entries:
                memories.append({
                    'content': entry.content,
                    'author': entry.author,
                    'timestamp': entry.timestamp.isoformat() if entry.timestamp else None,
                })
            
            return {
                'found': len(memories),
                'memories': memories,
            }
            
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
            return {"error": str(e)}
    
    def run(self, args: dict, context: Any = None) -> dict[str, Any]:
        """同步版本"""
        query = args.get('query', '')
        if not query:
            return {"error": "query is required"}
        
        if not isinstance(context, MemoryToolContext):
            return {"error": "Memory context not available"}
        
        try:
            result = context.search_memory_sync(query)
            
            memories = []
            for entry in result.entries:
                memories.append({
                    'content': entry.content,
                    'author': entry.author,
                    'timestamp': entry.timestamp.isoformat() if entry.timestamp else None,
                })
            
            return {
                'found': len(memories),
                'memories': memories,
            }
            
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
            return {"error": str(e)}
    
    def to_function_declaration(self) -> dict[str, Any]:
        """生成函数声明"""
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
        }


# ==================== 工具实例 ====================

preload_memory_tool = PreloadMemoryTool()
"""预加载记忆工具实例（推荐使用）"""

load_memory_tool = LoadMemoryTool()
"""加载记忆工具实例"""

