"""示例 5: 异步流式执行 - 实时获取事件"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_adk import Agent, EventType, Runner, Session, tool


@tool(description='执行mock的任务')
async def mock_task(task_name: str) -> str:
    """
    mock任务, 从用户query中提取task_name并执行
    
    Args:
        task_name: 要执行的任务名称
    """
    print(f'      [开始执行任务: {task_name}]')
    await asyncio.sleep(1)  # 模拟异步 IO 操作
    print(f'      [任务完成: {task_name}]')
    return f'任务 "{task_name}" 已完成'


@tool(description='获取指定城市的当前天气信息')
def get_weather(city: str) -> str:
    """
    查询城市天气
    
    Args:
        city: 要查询天气的城市名称，如 "北京"、"上海"
    """
    return f'{city}: 晴天，25°C'


async def main():
    """
    异步流式执行示例
    
    run_stream_async 方法可以：
    1. 实时接收模型生成的内容（token by token）
    2. 不阻塞事件循环
    3. 同时处理多个流式请求
    """
    agent = Agent(
        name='流式助手',
        model='QuantTrio/MiniMax-M2-AWQ',
        instruction='你可以执行任务和查询天气。',
        tools=[mock_task, get_weather],
    )
    
    session = Session()
    # 使用配置文件中的 show_request 设置，不再硬编码
    runner = Runner()
    
    print('=== 异步流式执行示例 ===\n')
    
    # 示例 1: 基础异步流式
    print('--- 示例 1: 异步流式输出 ---')
    user_msg = '你好，请介绍一下你自己'
    print(f'📝 用户: {user_msg}')
    print('🤖 Agent: ', end='', flush=True)
    
    async for event in runner.run_stream_async(
        agent=agent,
        session=session,
        user_message=user_msg,
    ):
        if event.event_type == EventType.MODEL_RESPONSE_DELTA:
            # 流式内容片段 - 实时打印
            print(event.content, end='', flush=True)
        
        elif event.event_type == EventType.MODEL_RESPONSE:
            # 完整响应
            print(f'\n   [响应完成] 时间: {event.timestamp.strftime("%H:%M:%S")}\n')
    
    # 示例 2: 带工具调用的异步流式
    print('--- 示例 2: 异步流式 + 工具调用 ---')
    session2 = Session()
    user_msg = '帮我执行一个数据分析任务'
    print(f'📝 用户: {user_msg}')
    print('🤖 Agent: ', end='', flush=True)
    
    async for event in runner.run_stream_async(
        agent=agent,
        session=session2,
        user_message=user_msg,
    ):
        if event.event_type == EventType.MODEL_RESPONSE_DELTA:
            print(event.content, end='', flush=True)
        
        elif event.event_type == EventType.MODEL_RESPONSE:
            print(f'\n   [响应完成]\n')
        
        elif event.event_type == EventType.TOOL_CALL:
            print(f'\n🔧 调用工具: {event.content["name"]}')
            print(f'   参数: {event.content.get("arguments", {})}')
        
        elif event.event_type == EventType.TOOL_RESPONSE:
            print(f'✅ 工具结果: {event.content["result"]}\n')
            print('🤖 Agent: ', end='', flush=True)
        
        elif event.event_type == EventType.ERROR:
            print(f'\n❌ 错误: {event.content}\n')
    
    # 示例 3: 并发流式处理
    print('\n--- 示例 3: 并发流式处理 ---')
    print('同时向两个不同的 session 发送请求...\n')
    
    async def stream_query(query: str, session_name: str):
        """在独立 session 中执行流式查询"""
        s = Session()
        responses = []  # 收集所有响应
        tool_calls = []  # 收集工具调用
        
        print(f'  [{session_name}] 开始: {query}')
        
        async for event in runner.run_stream_async(agent, s, query):
            if event.event_type == EventType.MODEL_RESPONSE:
                responses.append(event.content or '')
            elif event.event_type == EventType.TOOL_CALL:
                tool_calls.append(event.content.get('name', 'unknown'))
        
        # 合并所有响应用于展示
        all_responses = ' | '.join(r for r in responses if r)
        print(f'  [{session_name}] 完成: {all_responses[:50]}...')
        return {'responses': responses, 'tool_calls': tool_calls}
    
    # 并发执行两个流式请求
    results = await asyncio.gather(
        stream_query('北京天气怎么样？', 'Session-A'),
        stream_query('上海天气怎么样？', 'Session-B'),
    )
    
    print(f'\n✅ 所有并发请求完成！')
    
    for name, result in [('Session-A', results[0]), ('Session-B', results[1])]:
        responses = result['responses']
        tool_calls = result['tool_calls']
        print(f'   {name}:')
        print(f'     响应数量: {len(responses)} 个')
        print(f'     工具调用: {len(tool_calls)} 次 → {tool_calls}')
        non_empty = [r for r in responses if r]
        if non_empty:
            print(f'     最终响应: {non_empty[-1][:50]}...')
    
    # 显示会话统计
    print(f'\n--- 会话统计 ---')
    print(f'示例 1 会话事件数: {len(session.events)}')
    print(f'示例 2 会话事件数: {len(session2.events)}')


if __name__ == '__main__':
    asyncio.run(main())
