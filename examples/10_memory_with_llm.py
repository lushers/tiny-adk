#!/usr/bin/env python3
"""
示例 10: Memory + LLM 真实集成

演示两种 Memory 使用模式：

1. preload_memory (自动注入)
   - 每次 LLM 请求前自动搜索相关记忆
   - 将记忆注入到系统提示词中
   - 模型直接使用上下文回答，无需调用工具

2. load_memory (模型自动判断)
   - 作为普通工具暴露给模型
   - 模型根据问题类型自主判断是否需要查询记忆
   - 需要历史信息时调用工具，否则直接回答

运行方式:
    python examples/10_memory_with_llm.py           # 运行所有演示
    python examples/10_memory_with_llm.py preload   # 只运行 preload 演示
    python examples/10_memory_with_llm.py load      # 只运行 load 演示

需要配置 LLM API (通过 tiny_adk.yaml 或环境变量)
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_adk import (
    Agent,
    Runner,
    Session,
    SessionService,
    Event,
    EventType,
    # Memory
    InMemoryService,
    preload_memory_tool,
    load_memory_tool,
)


async def demo_memory_with_llm():
    """Memory + LLM 完整演示"""
    print("\n" + "=" * 60)
    print("🧠 Memory + LLM 真实集成演示")
    print("=" * 60 + "\n")
    
    # ==================== 准备服务 ====================
    session_service = SessionService()
    memory_service = InMemoryService()
    
    # ==================== 步骤 1: 创建历史会话 ====================
    print("📝 步骤 1: 创建历史会话并保存到 Memory...")
    print("-" * 50)
    
    # 模拟之前的对话
    history_session = await session_service.create_session(
        app_name="memory_llm_demo",
        user_id="alice",
        session_id="history_session"
    )
    
    # 添加历史对话事件
    history_events = [
        Event(
            event_type=EventType.USER_MESSAGE,
            content="My name is Alice and I am a Python developer.",
            author="user"
        ),
        Event(
            event_type=EventType.MODEL_RESPONSE,
            content="Nice to meet you, Alice! As a Python developer, what kind of projects do you work on?",
            author="assistant"
        ),
        Event(
            event_type=EventType.USER_MESSAGE,
            content="I mainly work on machine learning projects using PyTorch.",
            author="user"
        ),
        Event(
            event_type=EventType.MODEL_RESPONSE,
            content="That's great! PyTorch is an excellent framework for ML projects.",
            author="assistant"
        ),
    ]
    
    for event in history_events:
        history_session.add_event(event)
        print(f"   [{event.author}]: {event.content[:50]}...")
    
    # 保存到 Memory
    await memory_service.add_session(history_session)
    print("\n   ✅ 历史会话已保存到 Memory")
    
    # ==================== 步骤 2: 创建带 Memory 的 Agent ====================
    print("\n📝 步骤 2: 创建带 preload_memory_tool 的 Agent...")
    print("-" * 50)
    
    agent = Agent(
        name="memory_assistant",
        instruction="""You are a helpful assistant with memory capabilities.
When answering questions, use the information from PAST_CONVERSATIONS if relevant.
Be concise and direct in your responses.""",
        tools=[preload_memory_tool],  # 自动预加载记忆
    )
    
    print(f"   Agent: {agent.name}")
    print(f"   Tools: {[t.name for t in agent.tools]}")
    
    # ==================== 步骤 3: 创建 Runner ====================
    print("\n📝 步骤 3: 创建 Runner 并配置 memory_service...")
    print("-" * 50)
    
    runner = Runner(
        app_name="memory_llm_demo",
        agent=agent,
        session_service=session_service,
        memory_service=memory_service,  # 配置 Memory 服务
    )
    
    print("   ✅ Runner 已创建并配置 memory_service")
    
    # ==================== 步骤 4: 新会话测试 ====================
    print("\n📝 步骤 4: 在新会话中测试 Memory 召回...")
    print("-" * 50)
    
    # 创建新会话
    new_session = await session_service.create_session(
        app_name="memory_llm_demo",
        user_id="alice",
        session_id="new_session"
    )
    
    # 发送查询
    query = "What is my name and what do I work on?"
    print(f"\n👤 User: {query}")
    print("\n🔄 Processing...")
    print("   (preload_memory_tool 将自动在 LLM 请求前注入相关记忆)")
    
    try:
        print("\n🤖 Assistant:")
        async for event in runner.run_async(
            user_id="alice",
            session_id="new_session",
            message=query,
            stream=True,
        ):
            if event.event_type == EventType.MODEL_RESPONSE_DELTA:
                # 流式输出
                print(event.content, end="", flush=True)
            elif event.event_type == EventType.MODEL_RESPONSE:
                # 完整响应
                if event.content:
                    print(event.content)
        
        print("\n")
        print("=" * 50)
        print("✅ 演示完成！")
        print("=" * 50)
        print("""
注意观察：
1. 模型正确回答了用户的名字 (Alice) 和工作内容 (Python/ML/PyTorch)
2. 这些信息来自历史会话，通过 preload_memory_tool 自动注入
3. 模型不需要调用任何工具，直接从上下文获取信息

工作原理：
1. Runner 在调用 Flow 前设置 memory_context
2. SimpleFlow.build_request_async 检测 PreloadMemoryTool
3. 调用 preload_memory_tool.process_llm_request 搜索记忆
4. 将记忆文本注入到系统提示词中
5. 模型直接使用上下文回答
""")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n⚠️  请确保已配置 LLM API:")
        print("   - 创建 tiny_adk.yaml 配置文件")
        print("   - 或设置相关环境变量")
        print("\n示例 tiny_adk.yaml:")
        print("""
llm:
  api_base: "https://api.openai.com/v1"
  api_key: "your-api-key"
  model: "gpt-4"
""")


async def demo_load_memory_with_llm():
    """
    演示模型自动判断是否需要查询 Memory
    
    模型会根据问题类型自行决定：
    - 需要历史信息 -> 调用 load_memory 工具
    - 不需要历史信息 -> 直接回答
    """
    print("\n" + "=" * 60)
    print("🔍 load_memory_tool 演示 - 模型自动判断是否查询记忆")
    print("=" * 60 + "\n")
    
    session_service = SessionService()
    memory_service = InMemoryService()
    
    # ==================== 准备丰富的历史数据 ====================
    print("📝 步骤 1: 准备历史记忆数据...")
    print("-" * 50)
    
    history_session = await session_service.create_session(
        app_name="smart_assistant",
        user_id="charlie",
        session_id="history"
    )
    
    # 多种类型的历史信息
    history_events = [
        Event(
            event_type=EventType.USER_MESSAGE,
            content="My favorite color is blue and I love hiking on weekends.",
            author="user"
        ),
        Event(
            event_type=EventType.MODEL_RESPONSE,
            content="That's nice! Blue is a calming color, and hiking is great exercise.",
            author="assistant"
        ),
        Event(
            event_type=EventType.USER_MESSAGE,
            content="I have a dog named Max, he's a golden retriever.",
            author="user"
        ),
        Event(
            event_type=EventType.MODEL_RESPONSE,
            content="Golden retrievers are wonderful companions! Max must be a great hiking buddy.",
            author="assistant"
        ),
        Event(
            event_type=EventType.USER_MESSAGE,
            content="My birthday is on March 15th.",
            author="user"
        ),
        Event(
            event_type=EventType.MODEL_RESPONSE,
            content="I'll remember that! March 15th - that's coming up in spring.",
            author="assistant"
        ),
    ]
    
    for event in history_events:
        history_session.add_event(event)
        if event.author == "user":
            print(f"   💬 用户说过: {event.content[:60]}...")
    
    await memory_service.add_session(history_session)
    print("\n   ✅ 历史记忆已保存\n")
    
    # ==================== 创建 Agent ====================
    print("📝 步骤 2: 创建带 load_memory 的智能 Agent...")
    print("-" * 50)
    
    agent = Agent(
        name="smart_assistant",
        instruction="""You are a helpful assistant with memory capabilities.

You have access to a load_memory tool that can search your past conversations with the user.

IMPORTANT: Use load_memory ONLY when the user asks about something from previous conversations,
such as their preferences, personal information, or things they told you before.

For general questions (like "what's 2+2" or "what's the weather like"), 
answer directly WITHOUT using the tool.

When you do use load_memory, search with relevant keywords from the user's question.""",
        tools=[load_memory_tool],
    )
    
    print(f"   Agent: {agent.name}")
    print(f"   Tools: {[t.name for t in agent.tools]}")
    
    runner = Runner(
        app_name="smart_assistant",
        agent=agent,
        session_service=session_service,
        memory_service=memory_service,
    )
    
    # ==================== 测试不同类型的问题 ====================
    print("\n📝 步骤 3: 测试模型的自动判断能力...")
    print("-" * 50)
    
    # 测试问题列表
    test_queries = [
        {
            "query": "What is my dog's name?",
            "expected": "应该调用 load_memory（需要查询历史）",
            "keywords": "dog, pet, name"
        },
        {
            "query": "What is 15 + 27?",
            "expected": "不应该调用工具（简单计算）",
            "keywords": None
        },
        {
            "query": "When is my birthday?",
            "expected": "应该调用 load_memory（需要查询历史）",
            "keywords": "birthday, date"
        },
    ]
    
    for i, test in enumerate(test_queries, 1):
        query = test["query"]
        expected = test["expected"]
        
        print(f"\n{'='*50}")
        print(f"🧪 测试 {i}: {query}")
        print(f"   预期行为: {expected}")
        print("=" * 50)
        
        # 为每个测试创建新会话
        test_session = await session_service.create_session(
            app_name="smart_assistant",
            user_id="charlie",
            session_id=f"test_{i}"
        )
        
        print(f"\n👤 User: {query}\n")
        
        tool_called = False
        try:
            async for event in runner.run_async(
                user_id="charlie",
                session_id=f"test_{i}",
                message=query,
                stream=False,  # 非流式更清晰
            ):
                if event.event_type == EventType.TOOL_CALL:
                    tool_called = True
                    content = event.content
                    print(f"   🔧 Tool Call: {content['name']}")
                    print(f"      Query: {content['arguments']}")
                    
                elif event.event_type == EventType.TOOL_RESPONSE:
                    content = event.content
                    result = content['result']
                    # 截断显示
                    if len(result) > 150:
                        result = result[:150] + "..."
                    print(f"   📋 Tool Result: {result}")
                    
                elif event.event_type == EventType.MODEL_RESPONSE:
                    if event.content:
                        print(f"\n🤖 Assistant: {event.content}")
            
            # 判断结果
            if tool_called:
                print("\n   ✅ 模型调用了 load_memory 工具")
            else:
                print("\n   ✅ 模型直接回答，未调用工具")
                
        except Exception as e:
            print(f"\n   ❌ Error: {e}")
    
    # ==================== 总结 ====================
    print("\n" + "=" * 60)
    print("📊 演示总结")
    print("=" * 60)
    print("""
工作原理：
1. load_memory_tool 作为普通工具暴露给模型
2. 模型根据系统指令和问题类型自主判断
3. 需要历史信息时 -> 调用 load_memory 搜索记忆
4. 不需要时 -> 直接回答

与 preload_memory 的区别：
┌─────────────────┬───────────────────┬───────────────────┐
│                 │ preload_memory    │ load_memory       │
├─────────────────┼───────────────────┼───────────────────┤
│ 调用方式        │ 自动（每次请求）   │ 模型主动调用       │
│ 暴露给模型      │ ❌ 不暴露          │ ✅ 作为工具暴露    │
│ 搜索时机        │ 请求前            │ 模型决定时         │
│ 适用场景        │ 100% 需要记忆     │ 按需查询          │
│ 额外延迟        │ 有（每次都搜索）   │ 可能无（不一定搜索）│
└─────────────────┴───────────────────┴───────────────────┘
""")


async def main():
    """主函数"""
    import sys
    
    print("\n🎓 " + "=" * 54)
    print("   Memory + LLM 真实集成演示")
    print("=" * 58 + "\n")
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        demo = sys.argv[1]
        if demo == "preload":
            await demo_memory_with_llm()
        elif demo == "load":
            await demo_load_memory_with_llm()
        else:
            print(f"未知演示: {demo}")
            print("可用选项: preload, load")
    else:
        # 默认运行两个演示
        print("运行演示 1: preload_memory (自动注入)")
        print("-" * 60)
        await demo_memory_with_llm()
        
        print("\n" + "=" * 60)
        print("运行演示 2: load_memory (模型自动判断)")
        print("=" * 60)
        await demo_load_memory_with_llm()
    
    print("\n✅ 所有演示完成！")


if __name__ == "__main__":
    asyncio.run(main())
