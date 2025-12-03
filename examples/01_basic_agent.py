"""示例 1: 最基础的 Agent 用法"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_adk import Agent, Runner


def main():
    """
    基础用法示例
    
    展示 tiny_adk 的核心组件:
    - Agent: 定义"是谁"和"做什么"
    - Runner: 无状态执行引擎
    """
    # 1. 创建 Agent - 定义"是谁"和"做什么"
    agent = Agent(
        name='助手',
        instruction='你是一个友好的助手，帮助用户解答问题。',
        model='QuantTrio/MiniMax-M2-AWQ',
        max_iterations=10,
    )
    
    # 2. 创建 Runner - 无状态执行引擎
    runner = Runner()
    
    # 3. 执行对话（使用 user_id 和 session_id）
    user_id = 'user_001'
    session_id = 'session_001'
    
    print('=== 第一轮对话 ===')
    response1 = runner.run(
        agent=agent,
        user_id=user_id,
        session_id=session_id,
        message='你好！',
    )
    print(f'Agent: {response1}')
    
    # 4. 继续对话（同一个 session_id 保持历史）
    print('\n=== 第二轮对话 ===')
    response2 = runner.run(
        agent=agent,
        user_id=user_id,
        session_id=session_id,
        message='你叫什么名字？',
    )
    print(f'Agent: {response2}')
    
    # 5. 查看会话历史
    print('\n=== 会话历史 ===')
    session = runner.get_session(user_id, session_id)
    for i, event in enumerate(session.get_events(), 1):
        print(f'{i}. [{event.event_type.value}] {event.content}')


if __name__ == '__main__':
    main()
