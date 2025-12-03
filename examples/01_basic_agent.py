"""示例 1: 最基础的 Agent 用法"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # 添加项目根目录到路径

from tiny_adk import Agent, Runner, Session


def main():
  # 1. 创建 Agent - 定义"是谁"和"做什么"
  agent = Agent(
      name='助手pil',
      instruction='你是一个友好的助手，由philister提供，帮助用户解答问题。',
      model='QuantTrio/MiniMax-M2-AWQ',  # 使用真实模型
      max_iterations=10,
  )
  
  # 2. 创建 Session - 维护对话状态
  session = Session()
  
  # 3. 创建 Runner - 执行引擎
  runner = Runner()
  
  # 4. 执行对话
  print('=== 第一轮对话 ===')
  response1 = runner.run(
      agent=agent,
      session=session,
      user_message='你好！',
  )
  print(f'Agent: {response1}')
  
  # 5. 继续对话（Session 保存了历史）
  print('\n=== 第二轮对话 ===')
  response2 = runner.run(
      agent=agent,
      session=session,
      user_message='你叫什么名字？',
  )
  print(f'Agent: {response2}')
  
  # 6. 查看会话历史
  print('\n=== 会话历史 ===')
  for i, event in enumerate(session.get_events(), 1):
    print(f'{i}. [{event.event_type.value}] {event.content}')


if __name__ == '__main__':
  main()

