"""
示例 8: Memory 系统

演示 tiny_adk 的记忆系统使用：
1. 短期记忆（会话级）
2. 长期记忆（跨会话）
3. 向量搜索（语义搜索）
4. 集成到 Agent 工作流
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_adk import (
    Agent,
    Runner,
    Session,
    SessionService,
    # Memory 相关
    MemoryEntry,
    MemoryType,
    InMemoryService,
    VectorMemoryService,
    MemoryManager,
)


# ==================== 示例 1: 基础内存存储 ====================

async def basic_memory_example():
    """基础记忆存储示例"""
    print("\n" + "=" * 50)
    print("示例 1: 基础内存存储")
    print("=" * 50)
    
    # 创建内存存储服务
    memory_service = InMemoryService()
    
    # 添加记忆
    entry1 = MemoryEntry(
        content="用户喜欢Python编程",
        memory_type=MemoryType.LONG_TERM,
        app_name="demo_app",
        user_id="user_001",
        author="assistant",
    )
    id1 = await memory_service.add(entry1)
    print(f"添加记忆 1: {id1}")
    
    entry2 = MemoryEntry(
        content="用户正在学习机器学习",
        memory_type=MemoryType.SHORT_TERM,
        app_name="demo_app",
        user_id="user_001",
        session_id="session_001",
        author="user",
    )
    id2 = await memory_service.add(entry2)
    print(f"添加记忆 2: {id2}")
    
    entry3 = MemoryEntry(
        content="用户问了关于TensorFlow的问题",
        memory_type=MemoryType.SHORT_TERM,
        app_name="demo_app",
        user_id="user_001",
        session_id="session_001",
        author="user",
    )
    id3 = await memory_service.add(entry3)
    print(f"添加记忆 3: {id3}")
    
    # 搜索记忆（关键词匹配）
    print("\n搜索 'Python':")
    result = await memory_service.search(
        "Python",
        app_name="demo_app",
        user_id="user_001",
    )
    for entry in result:
        print(f"  - [{entry.memory_type.value}] {entry.content}")
    
    print("\n搜索 '机器学习 TensorFlow':")
    result = await memory_service.search(
        "机器学习 TensorFlow",
        app_name="demo_app",
        user_id="user_001",
    )
    for entry in result:
        print(f"  - [{entry.memory_type.value}] {entry.content}")


# ==================== 示例 2: 向量记忆存储 ====================

async def vector_memory_example():
    """向量记忆存储示例（语义搜索）"""
    print("\n" + "=" * 50)
    print("示例 2: 向量记忆存储（语义搜索）")
    print("=" * 50)
    
    # 创建向量存储服务
    # 注意：默认使用简单的字符频率向量，生产环境请使用真正的 Embedding
    memory_service = VectorMemoryService()
    
    # 添加记忆
    memories = [
        "用户是一名后端开发工程师",
        "用户对人工智能很感兴趣",
        "用户常用的编程语言是Python和Go",
        "用户最近在研究大语言模型",
    ]
    
    for content in memories:
        entry = MemoryEntry(
            content=content,
            memory_type=MemoryType.LONG_TERM,
            app_name="demo_app",
            user_id="user_001",
        )
        await memory_service.add(entry)
        print(f"添加: {content}")
    
    # 语义搜索
    print("\n搜索 'AI和LLM':")
    result = await memory_service.search(
        "AI和LLM",
        app_name="demo_app",
        user_id="user_001",
        limit=3,
    )
    for entry in result:
        print(f"  - {entry.content}")


# ==================== 示例 3: MemoryManager 聚合器 ====================

async def memory_manager_example():
    """MemoryManager 聚合器示例"""
    print("\n" + "=" * 50)
    print("示例 3: MemoryManager 聚合器")
    print("=" * 50)
    
    # 创建独立的短期和长期记忆存储
    short_term = InMemoryService()
    long_term = VectorMemoryService()
    
    # 创建 MemoryManager
    memory = MemoryManager(
        short_term=short_term,
        long_term=long_term,
    )
    
    # 添加长期记忆（用户偏好）
    await memory.add(
        "用户偏好使用中文回复",
        app_name="demo_app",
        user_id="user_001",
        memory_type=MemoryType.LONG_TERM,
        author="system",
    )
    await memory.add(
        "用户是高级Python开发者，无需过多解释基础概念",
        app_name="demo_app",
        user_id="user_001",
        memory_type=MemoryType.LONG_TERM,
        author="system",
    )
    print("添加长期记忆: 用户偏好")
    
    # 添加短期记忆（当前会话）
    await memory.add(
        "用户问了关于Python异步编程async的问题",
        app_name="demo_app",
        user_id="user_001",
        session_id="session_001",
        memory_type=MemoryType.SHORT_TERM,
        author="user",
    )
    await memory.add(
        "助手解释了asyncio和协程的概念",
        app_name="demo_app",
        user_id="user_001",
        session_id="session_001",
        memory_type=MemoryType.SHORT_TERM,
        author="assistant",
    )
    print("添加短期记忆: 当前对话")
    
    # 搜索记忆（演示多类型聚合搜索）
    print("\n搜索 'Python async':")
    result = await memory.search(
        "Python async",
        app_name="demo_app",
        user_id="user_001",
        session_id="session_001",
    )
    for entry in result:
        print(f"  - [{entry.memory_type.value}] ({entry.author}) {entry.content}")
    
    # 构建上下文（用于注入 prompt）
    context = await memory.build_context(
        "Python async 异步",
        app_name="demo_app",
        user_id="user_001",
        session_id="session_001",
    )
    print("\n构建的上下文（可注入到 prompt）:")
    print(context if context else "（无相关记忆）")


# ==================== 示例 4: 与 Session 集成 ====================

async def session_integration_example():
    """与 Session 集成示例"""
    print("\n" + "=" * 50)
    print("示例 4: 与 Session 集成")
    print("=" * 50)
    
    # 创建 SessionService
    session_service = SessionService()
    
    # 创建 Session
    session = await session_service.create_session(
        app_name="demo_app",
        user_id="user_001",
        session_id="session_with_memory",
    )
    
    # 模拟一些对话事件
    from tiny_adk.events import Event, EventType
    
    events = [
        Event(event_type=EventType.USER_MESSAGE, content="你好，我想学习Python", author="user"),
        Event(event_type=EventType.MODEL_RESPONSE, content="你好！Python是一门很棒的编程语言。你有什么具体想学的吗？", author="assistant"),
        Event(event_type=EventType.USER_MESSAGE, content="我想学习异步编程", author="user"),
        Event(event_type=EventType.MODEL_RESPONSE, content="异步编程是Python中非常重要的概念。让我给你介绍asyncio...", author="assistant"),
    ]
    
    for event in events:
        session.add_event(event)
        print(f"添加事件: {event.event_type.value}")
    
    # 将 Session 事件存入记忆
    memory_service = InMemoryService()
    ids = await memory_service.add_session(session)
    print(f"\n将 Session 存入记忆，共 {len(ids)} 条")
    
    # 搜索记忆
    print("\n搜索 'Python 学习':")
    result = await memory_service.search(
        "Python 学习",
        app_name="demo_app",
        user_id="user_001",
    )
    for entry in result:
        print(f"  - [{entry.author}] {entry.content[:50]}...")


# ==================== 示例 5: 持久化存储 ====================

def persistent_storage_example():
    """持久化存储示例（SQLite）"""
    print("\n" + "=" * 50)
    print("示例 5: 持久化存储（SQLite）")
    print("=" * 50)
    
    import tempfile
    import os
    
    # 创建临时数据库文件
    db_path = os.path.join(tempfile.gettempdir(), "tiny_adk_memory.db")
    
    # 创建带持久化的向量存储
    memory_service = VectorMemoryService(db_path=db_path)
    
    # 添加记忆
    entry = MemoryEntry(
        content="这条记忆会被持久化到SQLite",
        memory_type=MemoryType.LONG_TERM,
        app_name="demo_app",
        user_id="user_001",
    )
    memory_id = memory_service.add_sync(entry)
    print(f"添加记忆: {memory_id}")
    print(f"数据库路径: {db_path}")
    
    # 搜索（验证持久化）
    result = memory_service.search_sync(
        "持久化",
        app_name="demo_app",
        user_id="user_001",
    )
    print(f"搜索结果: {len(result)} 条")
    for entry in result:
        print(f"  - {entry.content}")


# ==================== 示例 6: 使用 OpenAI Embedding ====================

async def openai_embedding_example():
    """使用 OpenAI Embedding 进行语义搜索"""
    print("\n" + "=" * 50)
    print("示例 6: OpenAI Embedding（需要 API Key）")
    print("=" * 50)
    
    import os
    
    # 检查 API Key
    if not os.environ.get("OPENAI_API_KEY"):
        print("跳过：未设置 OPENAI_API_KEY 环境变量")
        return
    
    try:
        # 创建 OpenAI Embedding 函数
        embed_func = VectorMemoryService.create_openai_embedding_func()
        
        # 创建向量存储
        memory_service = VectorMemoryService(embedding_func=embed_func)
        
        # 添加记忆
        memories = [
            "用户是一名数据科学家",
            "用户精通统计学和机器学习",
            "用户使用 PyTorch 进行深度学习研究",
        ]
        
        for content in memories:
            entry = MemoryEntry(
                content=content,
                app_name="demo_app",
                user_id="user_001",
            )
            await memory_service.add(entry)
        
        # 语义搜索
        result = await memory_service.search(
            "神经网络训练",  # 语义相关但词汇不完全匹配
            app_name="demo_app",
            user_id="user_001",
        )
        
        print("搜索 '神经网络训练' 的结果:")
        for entry in result:
            print(f"  - {entry.content}")
            
    except ImportError:
        print("跳过：未安装 openai 包")
    except Exception as e:
        print(f"跳过：{e}")


# ==================== 主函数 ====================

async def main():
    """运行所有示例"""
    await basic_memory_example()
    await vector_memory_example()
    await memory_manager_example()
    await session_integration_example()
    persistent_storage_example()
    # await openai_embedding_example()
    
    print("\n" + "=" * 50)
    print("所有示例完成！")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())

