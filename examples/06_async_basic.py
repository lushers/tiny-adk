"""示例 6: 异步执行 - 使用 async/await"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_adk import Agent, Runner, Session, tool


# 定义同步工具
@tool(description='获取指定城市的当前天气信息')
def get_weather(city: str) -> str:
  """
  查询城市天气
  
  Args:
    city: 要查询天气的城市名称，如 "北京"、"上海"、"深圳"
  """
  weather_data = {
      '北京': '晴天，25°C',
      '上海': '多云，22°C',
      '深圳': '雨天，28°C',
  }
  return weather_data.get(city, f'{city} 的天气信息暂时无法获取')


# 定义异步工具
@tool(description='在数据库中搜索相关记录')
async def search_database(query: str) -> str:
  """
  异步数据库查询
  
  异步工具可以执行非阻塞的 IO 操作，如：
  - 数据库查询
  - HTTP 请求
  - 文件读写
  
  Args:
    query: 搜索关键词
  """
  # 模拟异步数据库查询
  await asyncio.sleep(0.5)
  return f'数据库查询结果: 找到 3 条关于 "{query}" 的记录'


@tool(description='发送通知消息给用户')
async def send_notification(message: str) -> str:
  """
  异步发送通知
  
  Args:
    message: 要发送的通知内容
  """
  await asyncio.sleep(0.2)  # 模拟网络请求
  return f'通知已发送: {message}'


async def main():
  """
  异步主函数
  
  使用 async/await 语法可以：
  1. 并发执行多个 Agent 任务
  2. 不阻塞事件循环
  3. 更好地利用 IO 等待时间
  """
  # 创建 Agent，同时支持同步和异步工具
  agent = Agent(
      name='异步助手',
      model='QuantTrio/MiniMax-M2-AWQ',
      instruction='你是一个智能助手，可以查询天气、搜索数据库和发送通知。',
      tools=[get_weather, search_database, send_notification],
  )
  
  session = Session()
  runner = Runner()
  
  print('=== 异步执行示例 ===\n')
  
  # 示例 1: 基础异步调用
  print('--- 示例 1: 基础异步调用 ---')
  print('📝 用户: 你好，介绍一下你自己')
  
  response = await runner.run_async(
      agent=agent,
      session=session,
      user_message='你好，介绍一下你自己',
  )
  print(f'🤖 Agent: {response}\n')
  
  # 示例 2: 调用同步工具（通过异步方法）
  print('--- 示例 2: 调用同步工具 ---')
  print('📝 用户: 北京天气怎么样？')
  
  response = await runner.run_async(
      agent=agent,
      session=session,
      user_message='北京天气怎么样？',
  )
  print(f'🤖 Agent: {response}\n')
  
  # 示例 3: 调用异步工具
  print('--- 示例 3: 调用异步工具 ---')
  print('📝 用户: 帮我搜索一下 Python 教程')
  
  # 创建新的 session 来隔离上下文
  session2 = Session()
  response = await runner.run_async(
      agent=agent,
      session=session2,
      user_message='帮我搜索一下 Python 教程',
  )
  print(f'🤖 Agent: {response}\n')
  
  # 示例 4: 并发执行多个任务
  print('--- 示例 4: 并发执行多个任务 ---')
  
  async def query_with_agent(query: str, session_id: str) -> str:
    """在独立的 session 中执行查询"""
    s = Session()
    s.session_id = session_id
    return await runner.run_async(agent, s, query)
  
  # 并发执行 3 个查询
  queries = [
      ('北京天气', 'session_1'),
      ('上海天气', 'session_2'),
      ('深圳天气', 'session_3'),
  ]
  
  print('同时查询 3 个城市的天气...')
  
  # 使用 asyncio.gather 并发执行
  results = await asyncio.gather(*[
      query_with_agent(q, sid) for q, sid in queries
  ])
  
  for (query, _), result in zip(queries, results):
    print(f'  📍 {query}: {result[:50]}...')
  
  print('\n✅ 所有任务完成！')
  
  # 显示会话统计
  print(f'\n--- 会话统计 ---')
  print(f'主会话事件数: {len(session.events)}')
  print(f'Session 2 事件数: {len(session2.events)}')


if __name__ == '__main__':
  asyncio.run(main())

