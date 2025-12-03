"""示例 5: 异步流式执行 - 实时获取事件"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_adk import Agent, EventType, Runner, tool


@tool(description='执行mock的任务')
async def mock_task(task_name: str) -> str:
    """mock任务"""
    print(f'      [开始执行任务: {task_name}]')
    await asyncio.sleep(1)
    print(f'      [任务完成: {task_name}]')
    return f'任务 "{task_name}" 已完成'


@tool(description='获取指定城市的当前天气信息')
def get_weather(city: str) -> str:
    """查询城市天气"""
    return f'{city}: 晴天，25°C'


async def main():
    """异步流式执行示例"""
    agent = Agent(
        name='流式助手',
        model='QuantTrio/MiniMax-M2-AWQ',
        instruction='你可以执行任务和查询天气。',
        tools=[mock_task, get_weather],
    )
    
    runner = Runner()
    user_id = 'user_001'
    
    print('=== 异步流式执行示例 ===\n')
    
    # 示例 1: 基础异步流式
    print('--- 示例 1: 异步流式输出 ---')
    user_msg = '你好，请介绍一下你自己'
    print(f'📝 用户: {user_msg}')
    print('🤖 Agent: ', end='', flush=True)
    
    async for event in runner.run_async(
        agent=agent,
        user_id=user_id,
        session_id='stream_1',
        message=user_msg,
        stream=True,
    ):
        if event.event_type == EventType.MODEL_RESPONSE_DELTA:
            print(event.content, end='', flush=True)
        elif event.event_type == EventType.MODEL_RESPONSE:
            print(f'\n   [响应完成] 时间: {event.timestamp.strftime("%H:%M:%S")}\n')
    
    # 示例 2: 带工具调用的异步流式
    print('--- 示例 2: 异步流式 + 工具调用 ---')
    user_msg = '帮我执行一个数据分析任务'
    print(f'📝 用户: {user_msg}')
    print('🤖 Agent: ', end='', flush=True)
    
    async for event in runner.run_async(
        agent=agent,
        user_id=user_id,
        session_id='stream_2',
        message=user_msg,
        stream=True,
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
    
    async def stream_query(query: str, session_id: str, label: str):
        """并发流式查询"""
        responses = []
        tool_calls = []
        
        print(f'  [{label}] 开始: {query}')
        
        async for event in runner.run_async(
            agent=agent,
            user_id=user_id,
            session_id=session_id,
            message=query,
            stream=True,
        ):
            if event.event_type == EventType.MODEL_RESPONSE:
                responses.append(event.content or '')
            elif event.event_type == EventType.TOOL_CALL:
                tool_calls.append(event.content.get('name', 'unknown'))
        
        all_responses = ' | '.join(r for r in responses if r)
        print(f'  [{label}] 完成: {all_responses[:50]}...')
        return {'responses': responses, 'tool_calls': tool_calls}
    
    results = await asyncio.gather(
        stream_query('北京天气怎么样？', 'concurrent_a', 'Session-A'),
        stream_query('上海天气怎么样？', 'concurrent_b', 'Session-B'),
    )
    
    print(f'\n✅ 所有并发请求完成！')
    
    for label, result in [('Session-A', results[0]), ('Session-B', results[1])]:
        print(f'   {label}:')
        print(f'     响应数量: {len(result["responses"])} 个')
        print(f'     工具调用: {len(result["tool_calls"])} 次 → {result["tool_calls"]}')


if __name__ == '__main__':
    asyncio.run(main())
