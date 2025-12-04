# tiny_adk 示例

这是 tiny_adk 三层架构的示例集合。

## 架构概览

```
┌────────────────────────────────────────┐
│  Runner: 执行编排（绑定 Agent）          │
├────────────────────────────────────────┤
│  Flow: Reason-Act 循环 + 工具执行       │
├────────────────────────────────────────┤
│  Model: LLM 抽象 + 请求/响应格式化       │
└────────────────────────────────────────┘
```

## 示例列表

| 示例 | 描述 | 核心概念 |
|------|------|---------|
| `01_basic_agent.py` | 基础 Agent 用法 | Runner, Agent, Session |
| `02_agent_with_tools.py` | 带工具的 Agent | 工具调用, Flow |
| `03_streaming.py` | 流式执行 | 流式输出, 事件 |
| `04_async_basic.py` | 异步执行 | async/await |
| `05_async_streaming.py` | 异步流式 | 异步 + 流式 |
| `06_web_service.py` | **Web 服务** | FastAPI, SSE, Web 界面 |

## 快速开始（ADK 风格）

```python
from tiny_adk import Agent, Runner, SessionService

# 1. 创建 Agent
agent = Agent(
    name='助手',
    instruction='你是一个助手',
    model='your-model',
)

# 2. 创建 Runner（绑定 Agent）
session_service = SessionService()
runner = Runner(
    app_name="my_app",
    agent=agent,
    session_service=session_service,
)

# 3. 创建 Session
session_service.create_session_sync(
    app_name="my_app",
    user_id='user_001',
    session_id='session_001'
)

# 4. 执行对话（不需要传 agent）
response = runner.run(
    user_id='user_001',
    session_id='session_001',
    message='你好',
)

# 5. 获取会话历史
session = session_service.get_session_sync(
    app_name="my_app",
    user_id='user_001',
    session_id='session_001'
)
for event in session.events:
    print(event.event_type, event.content)
```

## Web 服务快速启动

```python
from tiny_adk import Agent
from web import AgentService  # 独立的 web 模块

# 创建 Agent
agent = Agent(
    name='智能助手',
    instruction='你是一个智能助手',
    model='your-model',
)

# 创建并启动服务
service = AgentService(
    app_name="my_chatbot",
    agent=agent,
)
service.run(host="0.0.0.0", port=8000)
```

访问：
- **Web 界面**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs

## API 端点

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/sessions` | 创建会话 |
| GET | `/api/sessions/{user_id}/{session_id}` | 获取会话 |
| POST | `/api/chat` | 非流式对话 |
| POST | `/api/chat/stream` | 流式对话 (SSE) |
| DELETE | `/api/sessions/{user_id}/{session_id}` | 删除会话 |
| GET | `/` | Web 聊天界面 |

## Runner API

### 同步执行

```python
# 非流式
response = runner.run(
    user_id='user_001',
    session_id='session_001',
    message='你好',
)

# 流式
for event in runner.run_stream(
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
    user_id='user_001',
    session_id='session_001',
    message='你好',
):
    if event.event_type.value == 'model_response':
        print(event.content)

# 流式
async for event in runner.run_async(
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
    llm=llm,  # 传入 LLM 实例
    instruction='你是一个助手',
)
```

### 2. 自定义 SessionService

```python
from tiny_adk import Runner, SessionService

# 可以实现自己的 SessionService（如 Redis、数据库）
class RedisSessionService(SessionService):
    # 重写方法连接 Redis
    pass

runner = Runner(
    app_name="my_app",
    agent=agent,
    session_service=RedisSessionService(),
)
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
# 安装依赖
pip install -r requirements.txt

# 运行单个示例
python examples/01_basic_agent.py
python examples/02_agent_with_tools.py
python examples/03_streaming.py
python examples/04_async_basic.py
python examples/05_async_streaming.py

# 启动 Web 服务
python examples/06_web_service.py
```
