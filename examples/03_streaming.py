"""示例 3: 流式执行 - 实时获取事件"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_adk import Agent, EventType, Runner, SessionService, tool


@tool(description='执行耗时任务')
def slow_task(task_name: str) -> str:
    """模拟耗时任务"""
    import time
    time.sleep(1)
    return f'任务 "{task_name}" 已完成'


def main():
    agent = Agent(
        name='异步助手',
        model='QuantTrio/MiniMax-M2-AWQ',
        instruction='你可以执行耗时任务。',
        tools=[slow_task],
    )
    
    # 创建 SessionService 和 Runner
    session_service = SessionService()
    runner = Runner(session_service=session_service)
    
    user_id = 'user_001'
    session_id = 'stream_session'
    
    # 显式创建 Session
    session_service.create_session_sync(user_id=user_id, session_id=session_id)
    
    print('=== 流式执行示例 ===')
    user_msg = '帮我执行一个数据分析任务'
    print(f'📝 用户: {user_msg}')
    print('🤖 Agent: ', end='', flush=True)
    
    # 使用流式 API
    for event in runner.run_stream(
        agent=agent,
        user_id=user_id,
        session_id=session_id,
        message=user_msg,
    ):
        if event.event_type == EventType.MODEL_RESPONSE_DELTA:
            # 流式内容片段
            print(event.content, end='', flush=True)
        
        elif event.event_type == EventType.MODEL_RESPONSE:
            print(f'\n   [响应完成] 时间: {event.timestamp.strftime("%H:%M:%S")}\n')
        
        elif event.event_type == EventType.TOOL_CALL:
            print(f'\n🔧 调用工具: {event.content["name"]}')
            print(f'   参数: {event.content.get("arguments", {})}')
            print(f'   时间: {event.timestamp.strftime("%H:%M:%S")}\n')
        
        elif event.event_type == EventType.TOOL_RESPONSE:
            print(f'✅ 工具结果: {event.content["result"]}')
            print(f'   时间: {event.timestamp.strftime("%H:%M:%S")}\n')
            print('🤖 Agent: ', end='', flush=True)
        
        elif event.event_type == EventType.ERROR:
            print(f'\n❌ 错误: {event.content}')
            print(f'   时间: {event.timestamp.strftime("%H:%M:%S")}\n')


if __name__ == '__main__':
    main()
