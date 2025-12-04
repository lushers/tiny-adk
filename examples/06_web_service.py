"""示例 6: Web 服务 - 提供 API 和 Web 界面"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_adk import Agent, tool
from web import AgentService


# 定义工具函数
@tool(description='获取指定城市的天气信息')
def get_weather(city: str) -> str:
    """查询城市天气"""
    weather_data = {
        '北京': '晴天，25°C，空气质量良好',
        '上海': '多云，22°C，有轻度雾霾',
        '深圳': '雨天，28°C，建议带伞',
        '广州': '阴天，26°C，湿度较高',
        '杭州': '晴天，23°C，适合出行',
    }
    return weather_data.get(city, f'{city} 的天气信息暂时无法获取')


@tool(description='计算数学表达式')
def calculator(expression: str) -> str:
    """安全的计算器"""
    try:
        # 只允许基本数学运算
        allowed = set('0123456789+-*/.() ')
        if not all(c in allowed for c in expression):
            return '不支持的表达式'
        result = eval(expression, {'__builtins__': {}}, {})
        return f'{expression} = {result}'
    except Exception as e:
        return f'计算错误: {e}'


@tool(description='搜索知识库获取信息')
def search_knowledge(query: str) -> str:
    """模拟知识库搜索"""
    knowledge = {
        'python': 'Python 是一种解释型、高级、通用的编程语言。由 Guido van Rossum 创建。',
        'fastapi': 'FastAPI 是一个现代、快速的 Python Web 框架，用于构建 API。',
        'agent': 'Agent 是一种能够感知环境并采取行动以实现目标的智能体。',
        'llm': 'LLM (Large Language Model) 是在大规模文本数据上训练的神经网络模型。',
    }
    
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    
    return f'未找到关于 "{query}" 的相关信息'


def main():
    """启动 Web 服务"""
    
    # 1. 创建 Agent
    agent = Agent(
        name='智能助手',
        model='QuantTrio/MiniMax-M2-AWQ',
        instruction='''你是一个智能助手，可以：
1. 查询城市天气
2. 进行数学计算
3. 搜索知识库

请根据用户需求选择合适的工具，并友好地回答问题。
回答时请使用中文。''',
        tools=[get_weather, calculator, search_knowledge],
    )
    
    # 2. 创建服务
    service = AgentService(
        app_name="智能助手",
        agent=agent,
    )
    
    # 3. 启动服务
    print("""
╔══════════════════════════════════════════════════════════╗
║                  tiny_adk Web 服务                       ║
╠══════════════════════════════════════════════════════════╣
║  API 端点:                                               ║
║    POST /api/sessions         - 创建会话                 ║
║    GET  /api/sessions/{u}/{s} - 获取会话                 ║
║    POST /api/chat             - 非流式对话               ║
║    POST /api/chat/stream      - 流式对话 (SSE)           ║
║    DELETE /api/sessions/{u}/{s} - 删除会话               ║
║                                                          ║
║  Web 界面:                                               ║
║    GET /                      - 聊天界面                 ║
║    GET /docs                  - API 文档                 ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    service.run(host="0.0.0.0", port=8000)


if __name__ == '__main__':
    main()
