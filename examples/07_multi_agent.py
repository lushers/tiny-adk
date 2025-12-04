"""
示例 7: 多 Agent 协作

展示 tiny-adk 的多 Agent 功能：
1. Agent 树形结构（parent_agent / sub_agents）
2. transfer_to_agent 工具实现 Agent 跳转
3. SequentialAgent 顺序执行多个 Agent
4. LoopAgent 循环执行直到满足条件

运行方式:
    python examples/07_multi_agent.py
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_adk import (
    Agent, 
    SequentialAgent, 
    LoopAgent,
    Runner, 
    SessionService,
    create_transfer_tool,
    create_escalate_tool,
)


# ==================== 示例 1: Agent 树形结构 ====================

def example_agent_tree():
    """
    演示 Agent 树形结构
    
    展示 parent_agent、root_agent、find_agent 的使用
    """
    print("=" * 60)
    print("示例 1: Agent 树形结构")
    print("=" * 60)
    
    # 创建 Agent 树
    #
    #       root
    #      /    \
    #   math    language
    #            /    \
    #       english  chinese
    #
    
    english = Agent(name="english", description="English expert")
    chinese = Agent(name="chinese", description="Chinese expert")
    
    language = Agent(
        name="language",
        description="Language expert",
        sub_agents=[english, chinese],
    )
    
    math = Agent(name="math", description="Math expert")
    
    root = Agent(
        name="root",
        description="Root agent",
        sub_agents=[math, language],
    )
    
    # 展示树形结构
    print("\nAgent 树形结构:")
    print(f"  root")
    print(f"    ├── math (parent: {math.parent_agent.name})")
    print(f"    └── language (parent: {language.parent_agent.name})")
    print(f"          ├── english (parent: {english.parent_agent.name})")
    print(f"          └── chinese (parent: {chinese.parent_agent.name})")
    
    # 测试 root_agent
    print(f"\n从 english 获取 root: {english.root_agent.name}")
    print(f"从 chinese 获取 root: {chinese.root_agent.name}")
    
    # 测试 find_agent
    print(f"\n从 root 查找 'chinese': {root.find_agent('chinese').name}")
    print(f"从 root 查找 'math': {root.find_agent('math').name}")
    print(f"从 language 查找 'english': {language.find_agent('english').name}")
    
    # 测试 get_transferable_agents
    print(f"\n从 english 可以跳转到: {[a.name for a in english.get_transferable_agents()]}")
    print(f"从 language 可以跳转到: {[a.name for a in language.get_transferable_agents()]}")
    
    print("\n✅ 树形结构测试完成")


# ==================== 示例 2: 简单的多 Agent 跳转 ====================

async def example_transfer():
    """
    演示 transfer_to_agent 功能
    
    场景：主 Agent 识别到编程问题，跳转到编程专家 Agent
    """
    print("\n" + "=" * 60)
    print("示例 2: Agent 跳转 (transfer_to_agent)")
    print("=" * 60)
    
    # 创建专家 Agent
    coder = Agent(
        name="coder",
        description="编程专家，擅长解决代码问题",
        instruction="你是一位编程专家。用户会问你编程相关的问题，请给出专业的解答。",
        model="QuantTrio/MiniMax-M2-AWQ",
    )
    
    # 创建主 Agent，可以跳转到 coder
    main_agent = Agent(
        name="assistant",
        description="智能助手，可以协调多个专家",
        instruction="""你是一个智能助手。

当用户询问编程相关问题时，使用 transfer_to_agent 工具将任务交给 coder 专家。
当用户询问其他问题时，直接回答。""",
        model="QuantTrio/MiniMax-M2-AWQ",
        sub_agents=[coder],  # coder 是子 Agent
        tools=[
            create_transfer_tool(available_agents=["coder"]),
        ],
    )
    
    # 创建 Runner
    session_service = SessionService()
    runner = Runner(
        app_name="multi_agent_demo",
        agent=main_agent,
        session_service=session_service,
    )
    
    # 创建 Session
    await session_service.create_session(
        app_name="multi_agent_demo",
        user_id="user_1",
        session_id="session_transfer",
    )
    
    # 测试跳转
    print("\n📝 用户: 请帮我写一个 Python 快速排序函数")
    print("-" * 40)
    
    async for event in runner.run_async(
        user_id="user_1",
        session_id="session_transfer",
        message="请帮我写一个 Python 快速排序函数",
        stream=True,
    ):
        if event.event_type.value == 'model_response':
            print(f"🤖 [{event.author or 'agent'}]: {event.content}")
        elif event.event_type.value == 'agent_transfer':
            print(f"🔄 [跳转] {event.content.get('from_agent')} -> {event.content.get('target_agent')}")
    
    print("\n✅ Agent 跳转测试完成")


# ==================== 示例 3: SequentialAgent 顺序执行 ====================

async def example_sequential():
    """
    演示 SequentialAgent 功能
    
    场景：写作流水线 - 规划 -> 写作 -> 审核
    """
    print("\n" + "=" * 60)
    print("示例 3: 顺序执行 (SequentialAgent)")
    print("=" * 60)
    
    # 创建三个专业 Agent
    planner = Agent(
        name="planner",
        description="规划专家",
        instruction="你是一个规划专家。根据用户需求，列出3个要点作为写作大纲。只输出大纲，不要写正文。",
        model="QuantTrio/MiniMax-M2-AWQ",
    )
    
    writer = Agent(
        name="writer",
        description="写作专家",
        instruction="你是一个写作专家。根据之前的对话历史中的大纲，写出简短的内容（50字以内）。",
        model="QuantTrio/MiniMax-M2-AWQ",
    )
    
    reviewer = Agent(
        name="reviewer",
        description="审核专家",
        instruction="你是一个审核专家。检查之前写的内容，给出一句话评价。",
        model="QuantTrio/MiniMax-M2-AWQ",
    )
    
    # 创建顺序执行 Agent
    pipeline = SequentialAgent(
        name="writing_pipeline",
        description="写作流水线",
        instruction="",
        sub_agents=[planner, writer, reviewer],
        model="QuantTrio/MiniMax-M2-AWQ",
    )
    
    # 创建 Runner
    session_service = SessionService()
    runner = Runner(
        app_name="sequential_demo",
        agent=pipeline,
        session_service=session_service,
    )
    
    # 创建 Session
    await session_service.create_session(
        app_name="sequential_demo",
        user_id="user_1",
        session_id="session_seq",
    )
    
    # 测试顺序执行
    print("\n📝 用户: 请写一篇关于人工智能的短文")
    print("-" * 40)
    
    current_agent = None
    async for event in runner.run_async(
        user_id="user_1",
        session_id="session_seq",
        message="请写一篇关于人工智能的短文（50字左右）",
        stream=True,
    ):
        # 检查是否切换了 Agent（SequentialAgent 自动顺序执行，非工具调用）
        agent_name = event.author
        if event.event_type.value == 'model_response':
            if agent_name and agent_name != current_agent:
                if current_agent:
                    print(f"➡️ [顺序执行] {current_agent} -> {agent_name}")
                current_agent = agent_name
            print(f"🤖 [{agent_name or 'agent'}]: {event.content}")
    
    print("\n✅ 顺序执行测试完成")


# ==================== 示例 4: 内置工具展示 ====================

def example_builtin_tools():
    """
    演示内置的多 Agent 工具
    """
    print("\n" + "=" * 60)
    print("示例 4: 内置工具")
    print("=" * 60)
    
    # transfer_to_agent 工具
    transfer_tool = create_transfer_tool(available_agents=["coder", "writer"])
    print(f"\n1. TransferToAgentTool:")
    print(f"   名称: {transfer_tool.name}")
    print(f"   描述: {transfer_tool.description}")
    print(f"   可跳转: {transfer_tool.available_agents}")
    
    # escalate 工具
    escalate_tool = create_escalate_tool()
    print(f"\n2. EscalateTool:")
    print(f"   名称: {escalate_tool.name}")
    print(f"   描述: {escalate_tool.description}")
    
    print("\n✅ 内置工具展示完成")


# ==================== 主函数 ====================

async def main():
    """运行所有示例"""
    print("=" * 60)
    print("tiny-adk 多 Agent 示例")
    print("=" * 60)
    
    # 示例 1: Agent 树形结构（不需要 LLM）
    # example_agent_tree()
    
    # 示例 4: 内置工具展示（不需要 LLM）
    # example_builtin_tools()
    
    # 以下示例需要 LLM
    # print("\n" + "=" * 60)
    # print("以下示例需要 LLM，请确保已配置 tiny_adk.yaml")
    # print("=" * 60)
    
    try:
        # 示例 2: Agent 跳转
        await example_transfer()
    except Exception as e:
        print(f"\n⚠️ 示例 2 失败: {e}")
    
    try:
        # 示例 3: 顺序执行
        await example_sequential()
    except Exception as e:
        print(f"\n⚠️ 示例 3 失败: {e}")
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
