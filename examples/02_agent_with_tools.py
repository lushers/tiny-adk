"""示例 2: 带工具的 Agent"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_adk import Agent, Runner, Session, tool


# 定义工具函数
@tool(description='搜索互联网获取实时信息')
def web_search(query: str) -> str:
  """模拟网页搜索"""
  # 实际应该调用真实的搜索 API
  return f'搜索 "{query}" 的结果：这是模拟的搜索结果。'


@tool(description='获取指定城市的天气信息')
def get_weather(city: str) -> str:
  """模拟天气查询"""
  # 实际应该调用天气 API
  weather_data = {
      '北京': '晴天，25°C',
      '上海': '多云，22°C',
      '深圳': '雨天，28°C',
  }
  return weather_data.get(city, f'{city} 的天气信息暂时无法获取')


@tool(description='计算数学表达式')
def calculator(expression: str) -> str:
  """安全的计算器"""
  try:
    # 注意：eval 有安全风险，这里仅用于演示
    result = eval(expression, {'__builtins__': {}}, {})
    return f'{expression} = {result}'
  except Exception as e:
    return f'计算错误: {e}'


def main():
  # 创建带工具的 Agent
  agent = Agent(
      name='智能助手',
      model='QuantTrio/MiniMax-M2-AWQ',  # 使用真实模型
      instruction='你是一个智能助手，可以搜索信息、查天气、做计算。请根据用户需求选择合适的工具。',
      tools=[web_search, get_weather, calculator],  # 赋予 Agent 能力
  )
  
  session = Session()
  runner = Runner()  # 会自动连接到硬编码的 API
  
  # 测试不同的工具调用
  test_messages = [
      '北京今天天气怎么样？',
      '帮我搜索一下 Python 教程',
      '计算 123 + 456',
  ]
  
  for i, message in enumerate(test_messages, 1):
    print(f'\n=== 对话 {i} ===')
    print(f'用户: {message}')
    
    response = runner.run(agent, session, message)
    print(f'Agent: {response}')
    
    # 显示使用了哪些工具
    tool_calls = [
        e for e in session.get_events()
        if e.event_type.value == 'tool_call'
    ]
    if tool_calls:
      print(f'  (调用了工具: {tool_calls[-1].content["name"]})')


if __name__ == '__main__':
  main()

