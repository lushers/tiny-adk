"""示例 3: 流式执行 - 实时获取事件"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_adk import Agent, EventType, Runner, Session, tool


@tool(description='执行耗时任务')
def slow_task(task_name: str) -> str:
    """模拟耗时任务"""
    import time
    time.sleep(1)  # 模拟延迟
    return f'任务 "{task_name}" 已完成'


def main():
    agent = Agent(
        name='异步助手',
        model='QuantTrio/MiniMax-M2-AWQ',  # 使用真实模型
        instruction='你可以执行耗时任务。',
        tools=[slow_task],
    )
    
    session = Session()
    # 创建 Runner，可选择性显示调试信息
    runner = Runner(
        show_thinking=False,  # 不显示思考过程
    )
    
    print('=== 流式执行示例 ===')
    print('用户: 帮我执行一个数据分析任务\n')
    
    # 使用流式 API - 实时获取每个事件
    user_msg = '帮我执行一个数据分析任务'
    print(f'📝 用户: {user_msg}')
    print('🤖 Agent: ', end='', flush=True)
    
    for event in runner.run_stream(
        agent=agent,
        session=session,
        user_message=user_msg,
    ):
        # 根据事件类型做不同处理
        if event.event_type == EventType.MODEL_RESPONSE_DELTA:
            # 流式内容片段 - 实时打印，不换行
            print(event.content, end='', flush=True)
        
        elif event.event_type == EventType.MODEL_RESPONSE:
            # 完整响应 - 已经通过 delta 打印了，这里只标记完成
            print(f'\n   [响应完成] 时间: {event.timestamp.strftime("%H:%M:%S")}\n')
        
        elif event.event_type == EventType.TOOL_CALL:
            print(f'\n🔧 调用工具: {event.content["name"]}')
            print(f'   参数: {event.content.get("arguments", {})}')
            print(f'   时间: {event.timestamp.strftime("%H:%M:%S")}\n')
        
        elif event.event_type == EventType.TOOL_RESPONSE:
            print(f'✅ 工具结果: {event.content["result"]}')
            print(f'   时间: {event.timestamp.strftime("%H:%M:%S")}\n')
            print('🤖 Agent: ', end='', flush=True)  # 准备接收下一轮响应
        
        elif event.event_type == EventType.ERROR:
            print(f'\n❌ 错误: {event.content}')
            print(f'   时间: {event.timestamp.strftime("%H:%M:%S")}\n')


if __name__ == '__main__':
    main()
