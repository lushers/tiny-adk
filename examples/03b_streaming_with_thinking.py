"""示例 3b: 流式执行 - 显示思考过程"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_adk import Agent, EventType, Runner, Session


def main():
  agent = Agent(
      name='助手',
      model='QuantTrio/MiniMax-M2-AWQ',  # 使用真实模型
      instruction='你是一个友好的助手，帮助用户解答问题。',
  )
  
  session = Session()
  # 创建 Runner，启用 thinking 显示
  runner = Runner(
      show_thinking=True,  # ✅ 显示思考过程（原样输出所有内容，包括 <think> 标签）
  )
  
  print('=== 流式执行示例 (显示思考过程) ===\n')
  
  # 使用流式 API - 实时获取每个事件
  user_msg = '你好！'
  print(f'📝 用户: {user_msg}')
  print('🤖 Agent: ', end='', flush=True)
  
  for event in runner.run_stream(
      agent=agent,
      session=session,
      user_message=user_msg,
  ):
    # 根据事件类型做不同处理
    if event.event_type == EventType.MODEL_RESPONSE_DELTA:
      # 流式内容片段 - 实时打印，包含 <think> 标签
      print(event.content, end='', flush=True)
    
    elif event.event_type == EventType.MODEL_RESPONSE:
      # 完整响应
      print()  # 换行


if __name__ == '__main__':
  main()

