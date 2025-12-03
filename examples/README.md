# tiny_adk 示例

这是 tiny_adk 三层架构的示例集合。

## 架构概览

```
┌────────────────────────────────────────┐
│  Runner: 会话管理 + 编排               │
├────────────────────────────────────────┤
│  Flow: Reason-Act 循环 + 工具执行       │
├────────────────────────────────────────┤
│  Model: LLM 抽象 + 请求/响应格式化       │
└────────────────────────────────────────┘
```

## 示例列表

| 示例 | 描述 | 核心概念 |
|------|------|---------|
| `01_basic_agent.py` | 基础 Agent 用法 | Runner, Agent |
| `02_agent_with_tools.py` | 带工具的 Agent | 工具调用, Flow |
| `03_streaming.py` | 流式执行 | 流式输出, 事件 |
| `04_async_basic.py` | 异步执行 | async/await |
| `05_async_streaming.py` | 异步流式 | 异步 + 流式 |

## 快速开始

```python
from tiny_adk import Agent, Runner

# 创建组件
agent = Agent(name='助手', instruction='你是一个助手')
runner = Runner()

# 执行对话（使用 user_id 和 session_id）
response = runner.run(
    agent=agent,
    user_id='user_001',
    session_id='session_001',
    message='你好',
)

# 获取会话历史
session = runner.get_session('user_001', 'session_001')
for event in session.events:
    print(event.event_type, event.content)
```

## Runner API

### 同步执行

```python
# 非流式
response = runner.run(
    agent=agent,
    user_id='user_001',
    session_id='session_001',
    message='你好',
)

# 流式
for event in runner.run_stream(
    agent=agent,
    user_id='user_001',
    session_id='session_001',
    message='你好',
):
    if event.event_type == EventType.MODEL_RESPONSE_DELTA:
        print(event.content, end='', flush=True)
```

### 异步执行

```python
# 非流式
async for event in runner.run_async(
    agent=agent,
    user_id='user_001',
    session_id='session_001',
    message='你好',
):
    if event.event_type.value == 'model_response':
        print(event.content)

# 流式
async for event in runner.run_async(
    agent=agent,
    user_id='user_001',
    session_id='session_001',
    message='你好',
    stream=True,  # 启用流式
):
    if event.event_type == EventType.MODEL_RESPONSE_DELTA:
        print(event.content, end='', flush=True)
```

## 可扩展性

### 1. Agent 持有 LLM 实例

```python
from tiny_adk import Agent, OpenAILlm

llm = OpenAILlm(
    api_base="http://localhost:8000/v1",
    model="your-model",
)

agent = Agent(
    name='助手',
    model=llm,  # 传入 LLM 实例
    instruction='你是一个助手',
)
```

### 2. 自定义 SessionService

```python
from tiny_adk import Runner, SessionService

# 可以实现自己的 SessionService（如 Redis、数据库）
runner = Runner(session_service=SessionService())
```

### 3. 自定义 LLM

```python
from tiny_adk.models import BaseLlm

class MyCustomLlm(BaseLlm):
    model: str = "my-model"
    
    def generate(self, request): ...
    def generate_stream(self, request): ...
    async def generate_async(self, request): ...
    async def generate_stream_async(self, request): ...
```

## 运行示例

```bash
# 运行单个示例
python examples/01_basic_agent.py
python examples/02_agent_with_tools.py
python examples/03_streaming.py
python examples/04_async_basic.py
python examples/05_async_streaming.py
```
