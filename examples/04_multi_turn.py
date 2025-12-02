"""示例 4: 多轮对话 - Session 的核心价值"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_adk import Agent, Runner, Session


def main():
  agent = Agent(
      name='记忆助手',
      model='QuantTrio/MiniMax-M2-AWQ',  # 使用真实模型
      instruction='你要记住用户告诉你的信息，并在后续对话中使用。',
  )
  
  # 同一个 Session 保持对话上下文
  session = Session()
  runner = Runner()
  
  # 模拟多轮对话
  conversation = [
      '我叫张三，今年25岁。',
      '我喜欢编程和阅读。',
      '请告诉我，我的名字是什么？我多大了？',
      '我的兴趣爱好是什么？',
  ]
  
  print('=== 多轮对话演示 ===\n')
  
  for i, user_msg in enumerate(conversation, 1):
    print(f'第 {i} 轮:')
    print(f'  用户: {user_msg}')
    
    response = runner.run(agent, session, user_msg)
    print(f'  Agent: {response}\n')
  
  # 展示 Session 的价值
  print('=== Session 保存了什么？ ===')
  print(f'Session ID: {session.session_id}')
  print(f'事件数量: {len(session.events)}\n')
  
  # 可以序列化保存
  session_data = session.to_dict()
  print('可以序列化保存到数据库或文件:')
  print(f'  {list(session_data.keys())}\n')
  
  # 可以从序列化数据恢复
  restored_session = Session.from_dict(session_data)
  print(f'恢复后的 Session ID: {restored_session.session_id}')
  print(f'恢复后的事件数量: {len(restored_session.events)}')
  print(f'恢复后的事件: {restored_session.events}')

if __name__ == '__main__':
  main()

