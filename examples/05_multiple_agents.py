"""示例 5: 多 Agent 协作"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_adk import Agent, Runner, Session, tool


# 专业化的工具
@tool(description='分析 Python 代码质量')
def analyze_code(code: str) -> str:
  """代码分析工具"""
  lines = len(code.split('\n'))
  return f'代码共 {lines} 行，建议添加注释和类型提示。'


@tool(description='编写单元测试')
def write_tests(function_name: str) -> str:
  """测试生成工具"""
  return f'''
def test_{function_name}():
    # 测试正常情况
    assert {function_name}(valid_input) == expected_output
    
    # 测试边界情况
    assert {function_name}(edge_case) == edge_output
'''


def main():
  # 创建专业化的 Agents
  code_reviewer = Agent(
      name='代码审查员',
      model='QuantTrio/MiniMax-M2-AWQ',  # 使用真实模型
      instruction='你专注于代码审查，给出改进建议。',
      tools=[analyze_code],
  )
  
  test_writer = Agent(
      name='测试工程师',
      model='QuantTrio/MiniMax-M2-AWQ',  # 使用真实模型
      instruction='你专注于编写单元测试。',
      tools=[write_tests],
  )
  
  coordinator = Agent(
      name='项目协调员',
      model='QuantTrio/MiniMax-M2-AWQ',  # 使用真实模型
      instruction='你负责协调团队工作，分配任务。',
  )
  
  # 不同的 Agents 可以用在不同的场景
  runner = Runner()
  
  print('=== 场景 1: 代码审查 ===')
  review_session = Session()
  code = '''
def add(a, b):
    return a + b
'''
  response = runner.run(
      code_reviewer,
      review_session,
      f'请审查这段代码：\n{code}',
  )
  print(f'{code_reviewer.name}: {response}\n')
  
  print('=== 场景 2: 编写测试 ===')
  test_session = Session()
  response = runner.run(
      test_writer,
      test_session,
      '为 add 函数编写测试',
  )
  print(f'{test_writer.name}: {response}\n')
  
  print('=== 场景 3: 项目协调 ===')
  coord_session = Session()
  response = runner.run(
      coordinator,
      coord_session,
      '我们需要开发一个新功能，该怎么分工？',
  )
  print(f'{coordinator.name}: {response}\n')
  
  print('=== 核心思想 ===')
  print('1. 不同的 Agent 有不同的能力（工具）和角色（指令）')
  print('2. 每个 Agent 可以有独立的 Session（隔离的对话上下文）')
  print('3. Runner 可以执行任何 Agent（无状态、可复用）')
  print('4. 这种设计支持复杂的多 Agent 协作系统')


if __name__ == '__main__':
  main()

