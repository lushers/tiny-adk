"""示例 1: 最基础的 Agent 用法（ADK 风格）"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_adk import Agent, Runner, SessionService


def main():
    """
    基础用法示例（参考 ADK 设计）
    
    展示 tiny_adk 的核心组件:
    - Agent: 定义"是谁"和"做什么"（只读配置）
    - SessionService: Session 持久化服务
    - Runner: 无状态执行引擎（绑定特定 Agent）
    
    设计理念:
    - Runner 绑定 app_name 和 agent
    - Session 通过 (app_name, user_id, session_id) 唯一标识
    - Session 必须显式创建
    """
    # 1. 创建 Agent - 定义"是谁"和"做什么"
    agent = Agent(
        name='助手',
        instruction='你是一个友好的助手，帮助用户解答问题。',
        model='QuantTrio/MiniMax-M2-AWQ',
        max_iterations=10,
    )
    
    # 2. 创建 Runner（绑定 Agent）
    session_service = SessionService()
    runner = Runner(
        app_name="hello_app",
        agent=agent,
        session_service=session_service,
    )
    
    # 3. 定义用户和会话 ID
    user_id = 'user_001'
    session_id = 'session_001'
    
    # 4. 显式创建 Session
    session_service.create_session_sync(
        app_name="hello_app",
        user_id=user_id,
        session_id=session_id
    )
    print(f"✅ 创建会话: {session_id}\n")
    
    # 5. 执行对话（不需要传 agent）
    print('=== 第一轮对话 ===')
    response1 = runner.run(
        user_id=user_id,
        session_id=session_id,
        message='你好！',
    )
    print(f'Agent: {response1}')
    
    # 6. 继续对话（同一个 session_id 保持历史）
    print('\n=== 第二轮对话 ===')
    response2 = runner.run(
        user_id=user_id,
        session_id=session_id,
        message='你叫什么名字？',
    )
    print(f'Agent: {response2}')
    
    # 7. 查看会话历史
    print('\n=== 会话历史 ===')
    session = session_service.get_session_sync(
        app_name="hello_app",
        user_id=user_id,
        session_id=session_id
    )
    for i, event in enumerate(session.get_events(), 1):
        print(f'{i}. [{event.event_type.value}] {event.content}')


if __name__ == '__main__':
    main()
