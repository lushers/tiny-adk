"""示例 4: 异步执行 - 使用 async/await"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_adk import Agent, Runner, tool


@tool(description='获取指定城市的当前天气信息')
def get_weather(city: str) -> str:
    """查询城市天气"""
    weather_data = {
        '北京': '晴天，25°C',
        '上海': '多云，22°C',
        '深圳': '雨天，28°C',
    }
    return weather_data.get(city, f'{city} 的天气信息暂时无法获取')


@tool(description='在数据库中搜索相关记录')
async def search_database(query: str) -> str:
    """异步数据库查询"""
    await asyncio.sleep(0.5)
    return f'数据库查询结果: 找到 3 条关于 "{query}" 的记录'


@tool(description='发送通知消息给用户')
async def send_notification(message: str) -> str:
    """异步发送通知"""
    await asyncio.sleep(0.2)
    return f'通知已发送: {message}'


async def main():
    """异步主函数"""
    agent = Agent(
        name='异步助手',
        model='QuantTrio/MiniMax-M2-AWQ',
        instruction='你是一个智能助手，可以查询天气、搜索数据库和发送通知。',
        tools=[get_weather, search_database, send_notification],
    )
    
    runner = Runner()
    user_id = 'user_001'
    
    print('=== 异步执行示例 ===\n')
    
    # 示例 1: 基础异步调用
    print('--- 示例 1: 基础异步调用 ---')
    print('📝 用户: 你好，介绍一下你自己')
    
    # 收集所有事件，获取最终响应
    response = None
    async for event in runner.run_async(
        agent=agent,
        user_id=user_id,
        session_id='session_1',
        message='你好，介绍一下你自己',
    ):
        if event.event_type.value == 'model_response':
            response = event.content
    print(f'🤖 Agent: {response}\n')
    
    # 示例 2: 调用同步工具
    print('--- 示例 2: 调用同步工具 ---')
    print('📝 用户: 北京天气怎么样？')
    
    response = None
    async for event in runner.run_async(
        agent=agent,
        user_id=user_id,
        session_id='session_2',
        message='北京天气怎么样？',
    ):
        if event.event_type.value == 'model_response':
            response = event.content
    print(f'🤖 Agent: {response}\n')
    
    # 示例 3: 调用异步工具
    print('--- 示例 3: 调用异步工具 ---')
    print('📝 用户: 帮我搜索一下 Python 教程')
    
    response = None
    async for event in runner.run_async(
        agent=agent,
        user_id=user_id,
        session_id='session_3',
        message='帮我搜索一下 Python 教程',
    ):
        if event.event_type.value == 'model_response':
            response = event.content
    print(f'🤖 Agent: {response}\n')
    
    # 示例 4: 并发执行多个任务
    print('--- 示例 4: 并发执行多个任务 ---')
    
    async def query_weather(city: str, sid: str) -> str:
        """并发查询天气"""
        response = None
        async for event in runner.run_async(
            agent=agent,
            user_id=user_id,
            session_id=sid,
            message=f'{city}天气',
        ):
            if event.event_type.value == 'model_response':
                response = event.content
        return response
    
    print('同时查询 3 个城市的天气...')
    
    results = await asyncio.gather(
        query_weather('北京', 'concurrent_1'),
        query_weather('上海', 'concurrent_2'),
        query_weather('深圳', 'concurrent_3'),
    )
    
    for city, result in zip(['北京', '上海', '深圳'], results):
        print(f'  📍 {city}: {result[:50] if result else "无结果"}...')
    
    print('\n✅ 所有任务完成！')


if __name__ == '__main__':
    asyncio.run(main())
