"""示例 1: 最基础的 Agent 用法"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_adk import Agent, Runner, SessionService


def main():
    """
    基础用法示例
    
    展示 tiny_adk 的核心组件:
    - Agent: 定义"是谁"和"做什么"
    - SessionService: Session 持久化服务
    - Runner: 无状态执行引擎
    
    设计理念（参考 ADK）:
    - Session 必须显式创建（通过 SessionService.create_session）
    - Runner 只负责执行，不负责 Session 生命周期
    """
    # 1. 创建 Agent - 定义"是谁"和"做什么"
    agent = Agent(
        name='助手',
        instruction='你是一个友好的助手，帮助用户解答问题。',
        model='QuantTrio/MiniMax-M2-AWQ',
        max_iterations=10,
    )
    
    # 2. 创建 SessionService 和 Runner
    session_service = SessionService()
    runner = Runner(session_service=session_service)
    
    # 3. 定义用户和会话 ID
    user_id = 'user_001'
    session_id = 'session_001'
    
    # 4. 显式创建 Session（ADK 设计：Session 必须预先存在）
    session_service.create_session_sync(user_id=user_id, session_id=session_id)
    
    # 5. 执行对话
    print('=== 第一轮对话 ===')
    response1 = runner.run(
        agent=agent,
        user_id=user_id,
        session_id=session_id,
        message='你好！',
    )
    print(f'Agent: {response1}')
    
    # 6. 继续对话（同一个 session_id 保持历史）
    print('\n=== 第二轮对话 ===')
    response2 = runner.run(
        agent=agent,
        user_id=user_id,
        session_id=session_id,
        message='你叫什么名字？',
    )
    print(f'Agent: {response2}')
    
    # 7. 查看会话历史
    print('\n=== 会话历史 ===')
    session = session_service.get_session_sync(user_id, session_id)
    for i, event in enumerate(session.get_events(), 1):
        print(f'{i}. [{event.event_type.value}] {event.content}')


if __name__ == '__main__':
    main()
