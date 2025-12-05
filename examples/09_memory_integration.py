"""
示例 9: Memory 系统集成 - 在 Agent 运行中使用记忆

演示如何在真实 Agent 工作流中使用 Memory：
1. 基础：保存会话到 Memory，搜索历史
2. preload_memory_tool: 自动预加载（推荐！）
3. load_memory_tool: 模型主动调用
4. 自定义工具中使用 MemoryToolContext

参考 ADK 的 Memory 设计
"""

import asyncio
import time
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_adk import (
    Agent,
    Runner,
    Session,
    SessionService,
    Event,
    EventType,
    # Memory 相关
    InMemoryService,
    MemoryToolContext,
    preload_memory_tool,
    load_memory_tool,
)


# ============================================================================
# 演示 1: Memory 基础 - 保存和搜索历史
# ============================================================================

async def demo_basic_memory():
    """演示 Memory 的基本使用：保存会话并搜索历史"""
    print("\n" + "=" * 60)
    print("📚 演示 1: Memory 基础 - 保存和搜索历史")
    print("=" * 60 + "\n")
    
    # 创建服务
    session_service = SessionService()
    memory_service = InMemoryService()
    
    # 创建会话并添加一些对话事件
    session = await session_service.create_session(
        app_name="memory_demo",
        user_id="alice",
        session_id="session_001"
    )
    
    print("📝 步骤 1: 创建会话并添加对话事件...")
    print("-" * 50)
    
    events = [
        Event(event_type=EventType.USER_MESSAGE, content="你好！我叫 Alice，我是一名 Python 开发者。", author="user"),
        Event(event_type=EventType.MODEL_RESPONSE, content="你好 Alice！很高兴认识你，作为 Python 开发者你有什么问题想问吗？", author="assistant"),
        Event(event_type=EventType.USER_MESSAGE, content="我最喜欢的框架是 FastAPI。", author="user"),
        Event(event_type=EventType.MODEL_RESPONSE, content="FastAPI 是个很棒的选择！它的异步支持和自动文档生成非常强大。", author="assistant"),
    ]
    
    for event in events:
        session.add_event(event)
        print(f"   [{event.author}]: {event.content[:50]}...")
    
    # 将会话保存到 Memory
    print("\n📝 步骤 2: 将会话保存到 Memory...")
    print("-" * 50)
    
    ids = await memory_service.add_session(session)
    print(f"   ✅ 保存了 {len(ids)} 条记忆")
    
    # 搜索 Memory
    print("\n📝 步骤 3: 搜索 Memory...")
    print("-" * 50)
    
    result = await memory_service.search(
        "Python FastAPI",  # 关键词
        app_name="memory_demo",
        user_id="alice",
    )
    
    print(f"   🔍 搜索 'Python FastAPI':")
    print(f"   📄 找到 {len(result.entries)} 条相关记忆:")
    for entry in result.entries:
        print(f"      - [{entry.author}]: {entry.content[:40]}...")


# ============================================================================
# 演示 2: preload_memory_tool - 自动预加载（推荐！）
# ============================================================================

async def demo_preload_memory():
    """演示 preload_memory_tool 的使用"""
    print("\n" + "=" * 60)
    print("🚀 演示 2: preload_memory_tool - 自动预加载（推荐！）")
    print("=" * 60 + "\n")
    
    print("📖 preload_memory vs load_memory 对比:")
    print("-" * 50)
    print("""
   ┌─────────────────┬──────────────────────────────────┐
   │  load_memory    │       preload_memory (推荐)      │
   ├─────────────────┼──────────────────────────────────┤
   │ 模型主动调用     │ 自动执行，不依赖模型判断          │
   │ 可能忘记/出错    │ 100% 可靠执行                    │
   │ 需要额外轮次     │ 零额外延迟                       │
   │ 适合精确控制     │ 适合需要稳定召回的场景            │
   └─────────────────┴──────────────────────────────────┘
    """)
    
    # 准备历史数据
    session_service = SessionService()
    memory_service = InMemoryService()
    
    print("📝 步骤 1: 准备历史记忆...")
    print("-" * 50)
    
    history_session = await session_service.create_session(
        app_name="preload_demo",
        user_id="bob",
        session_id="history"
    )
    
    history_events = [
        Event(event_type=EventType.USER_MESSAGE, content="My favorite color is blue.", author="user"),
        Event(event_type=EventType.MODEL_RESPONSE, content="Blue is a nice color!", author="assistant"),
        Event(event_type=EventType.USER_MESSAGE, content="My pet dog's name is Max.", author="user"),
        Event(event_type=EventType.MODEL_RESPONSE, content="Max is a lovely name!", author="assistant"),
    ]
    
    for event in history_events:
        history_session.add_event(event)
        print(f"   [{event.author}]: {event.content}")
    
    await memory_service.add_session(history_session)
    print("\n   ✅ 历史数据已保存到 Memory")
    
    print("\n📝 步骤 2: 模拟 preload_memory_tool 的工作...")
    print("-" * 50)
    
    # 创建 Memory 上下文
    # 注意：InMemoryService 使用关键词匹配，所以查询中需要包含记忆中的关键词
    context = MemoryToolContext(
        memory_service=memory_service,
        app_name="preload_demo",
        user_id="bob",
        session_id="new_session",
        user_query="Tell me about my favorite color blue",  # 包含 "blue" 关键词
    )
    
    # 调用 preload_memory_tool 的预处理方法
    preload_text = await preload_memory_tool.process_llm_request(context)
    
    print("   用户查询: 'Tell me about my favorite color blue'")
    print("\n   📋 preload_memory 生成的上下文（会注入到 prompt）:")
    print("-" * 50)
    if preload_text:
        print(preload_text)
    else:
        print("   (无相关记忆)")
    
    print("\n   ✅ 这段文本会自动注入到 LLM 的系统指令中，")
    print("      模型可以直接使用这些信息回答，无需调用工具！")


# ============================================================================
# 演示 3: load_memory_tool - 模型主动调用
# ============================================================================

async def demo_load_memory():
    """演示 load_memory_tool 的使用"""
    print("\n" + "=" * 60)
    print("🔍 演示 3: load_memory_tool - 模型主动调用")
    print("=" * 60 + "\n")
    
    # 准备历史数据
    session_service = SessionService()
    memory_service = InMemoryService()
    
    history_session = await session_service.create_session(
        app_name="load_demo",
        user_id="charlie",
        session_id="history"
    )
    
    print("📝 步骤 1: 准备历史数据...")
    print("-" * 50)
    
    history_events = [
        Event(event_type=EventType.USER_MESSAGE, content="请记住我的订单号是 ORDER-12345", author="user"),
        Event(event_type=EventType.MODEL_RESPONSE, content="好的，我记住了你的订单号 ORDER-12345", author="assistant"),
    ]
    
    for event in history_events:
        history_session.add_event(event)
        print(f"   [{event.author}]: {event.content}")
    
    await memory_service.add_session(history_session)
    print("\n   ✅ 历史数据已保存\n")
    
    print("📝 步骤 2: 模拟 load_memory_tool 被模型调用...")
    print("-" * 50)
    
    # 创建 Memory 上下文
    context = MemoryToolContext(
        memory_service=memory_service,
        app_name="load_demo",
        user_id="charlie",
    )
    
    # 模拟模型调用 load_memory_tool
    print("   🤖 模型决定调用 load_memory 工具...")
    print(f"   🔧 调用: load_memory(query='ORDER')")
    
    result = await load_memory_tool.run_async(
        args={"query": "ORDER"},
        context=context,
    )
    
    print(f"\n   📋 工具返回:")
    print(f"      找到 {result['found']} 条记忆:")
    for mem in result['memories']:
        print(f"      - [{mem['author']}]: {mem['content']}")
    
    print("\n   ℹ️  与 preload_memory 的区别：")
    print("      - load_memory 需要模型主动调用")
    print("      - 需要额外一轮 LLM 交互")
    print("      - 但可以精确控制搜索关键词")


# ============================================================================
# 演示 4: 在 Runner 中使用 Memory
# ============================================================================

async def demo_runner_with_memory():
    """演示在 Runner 中使用 Memory"""
    print("\n" + "=" * 60)
    print("🏃 演示 4: 在 Runner 中使用 Memory")
    print("=" * 60 + "\n")
    
    print("📖 Runner 集成 Memory 的方式:")
    print("-" * 50)
    print("""
   ┌─────────────────────────────────────────────────────┐
   │                    Runner                           │
   │  ┌─────────────────────────────────────────────┐   │
   │  │ session_service: SessionService              │   │
   │  │ memory_service: BaseMemoryService (可选)     │   │
   │  └─────────────────────────────────────────────┘   │
   │                      ↓                              │
   │  ┌─────────────────────────────────────────────┐   │
   │  │ Agent (tools=[preload_memory_tool, ...])    │   │
   │  └─────────────────────────────────────────────┘   │
   │                      ↓                              │
   │  ┌─────────────────────────────────────────────┐   │
   │  │ Flow: 在 LLM 请求前自动调用 preload         │   │
   │  └─────────────────────────────────────────────┘   │
   └─────────────────────────────────────────────────────┘
    """)
    
    # 创建服务
    session_service = SessionService()
    memory_service = InMemoryService()
    
    # 创建 Agent（带 preload_memory_tool）
    agent = Agent(
        name="memory_agent",
        instruction="""你是一个有记忆能力的助手。
如果 PAST_CONVERSATIONS 中有相关信息，请使用它来回答问题。""",
        tools=[preload_memory_tool],  # 自动预加载
    )
    
    # 创建 Runner（带 memory_service）
    runner = Runner(
        app_name="runner_demo",
        agent=agent,
        session_service=session_service,
        memory_service=memory_service,  # 传入 memory_service
    )
    
    print("   ✅ Runner 创建完成，已配置 memory_service")
    print("\n   使用示例代码:")
    print("-" * 50)
    print("""
    # 1. 创建 Runner 时传入 memory_service
    runner = Runner(
        app_name="my_app",
        agent=agent,
        session_service=session_service,
        memory_service=InMemoryService(),  # 或 VectorMemoryService
    )
    
    # 2. 在 Agent 中添加 memory 工具
    agent = Agent(
        tools=[preload_memory_tool],  # 自动预加载
        # 或 tools=[load_memory_tool],  # 模型主动调用
        ...
    )
    
    # 3. 会话结束后保存到 Memory
    await memory_service.add_session(session)
    """)


# ============================================================================
# 演示 5: Memory 使用总结
# ============================================================================

def demo_summary():
    """Memory 使用总结"""
    print("\n" + "=" * 60)
    print("📖 Memory 系统使用总结")
    print("=" * 60)
    print("""

1️⃣  Memory vs Session 的区别:
   - Session: 单次对话的上下文（短期记忆）
   - Memory: 跨多个会话的历史记录（长期记忆）

2️⃣  Memory Service 的核心方法:
   - add_session(session): 保存会话到记忆
   - search(query, ...): 搜索记忆

3️⃣  两种 Memory 工具:
   
   ┌─────────────────────┬────────────────────────────────┐
   │  preload_memory_tool│     load_memory_tool           │
   │       (推荐！)       │                                │
   ├─────────────────────┼────────────────────────────────┤
   │ 自动执行            │ 模型主动调用                    │
   │ 在 LLM 请求前注入   │ 需要额外一轮交互                │
   │ 100% 可靠           │ 可能忘记调用                    │
   │ 零延迟              │ 可精确控制查询                  │
   └─────────────────────┴────────────────────────────────┘

4️⃣  集成步骤:
   
   Step 1: 创建 Memory Service
   ```python
   memory_service = InMemoryService()  # 开发用
   # 或
   memory_service = VectorMemoryService(db_path="./memory.db")  # 生产用
   ```
   
   Step 2: 创建 Runner 时传入
   ```python
   runner = Runner(
       ...,
       memory_service=memory_service,
   )
   ```
   
   Step 3: 在 Agent 中添加工具
   ```python
   agent = Agent(
       tools=[preload_memory_tool],  # 推荐
       ...
   )
   ```
   
   Step 4: 会话结束后保存
   ```python
   await memory_service.add_session(session)
   ```

5️⃣  可用的 Memory Service:
   - InMemoryService: 内存存储，关键词匹配（开发/测试）
   - VectorMemoryService: 向量存储，语义搜索（生产环境）

""")


# ============================================================================
# 主函数
# ============================================================================

async def main():
    """运行所有演示"""
    print("\n🎓 " + "=" * 56)
    print("   Memory 系统集成演示 - 在 Agent 运行中使用记忆")
    print("=" * 60)
    
    await demo_basic_memory()
    await demo_preload_memory()
    await demo_load_memory()
    await demo_runner_with_memory()
    demo_summary()
    
    print("=" * 60)
    print("🎓 演示完成！")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

