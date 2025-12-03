# tiny_adk 示例

这是 tiny_adk 三层架构的示例集合。

## 架构概览

```
┌────────────────────────────────────────┐
│  Runner: 配置 + 会话管理 + 编排         │  (~200 行)
├────────────────────────────────────────┤
│  Flow: Reason-Act 循环 + 工具执行       │  (~300 行)
├────────────────────────────────────────┤
│  Model: LLM 抽象 + 请求/响应格式化       │  (~400 行)
└────────────────────────────────────────┘
```

## 示例列表

| 示例 | 描述 | 核心概念 |
|------|------|---------|
| `01_basic_agent.py` | 基础 Agent 用法 | Runner, Session |
| `02_agent_with_tools.py` | 带工具的 Agent | 工具调用, Flow |
| `03_streaming.py` | 流式执行 | 流式输出, 事件 |
| `04_async_basic.py` | 异步执行 | async/await |
| `05_async_streaming.py` | 异步流式 | 异步 + 流式 |
| `06_custom_llm.py` | 自定义 LLM | BaseLlm, 扩展性 |
| `07_architecture_comparison.py` | 架构说明 | 设计理念 |

## 快速开始

```python
from tiny_adk import Agent, Session, Runner

# 创建组件
agent = Agent(name='助手', instruction='你是一个助手')
session = Session()
runner = Runner()  # 自动根据配置创建 LLM

# 执行对话
response = runner.run(agent, session, '你好')
```

## 可扩展性

### 1. 指定 LLM
```python
from tiny_adk import Runner, OpenAILlm

runner = Runner(llm=OpenAILlm(
    api_base="http://localhost:8000/v1",
    model="your-model",
))
```

### 2. 自定义 Flow
```python
from tiny_adk import SimpleFlow, Runner

runner = Runner(
    flow=SimpleFlow(max_iterations=5),
)
```

### 3. 自定义 LLM
```python
# 实现自己的 LLM（见 06_custom_llm.py）
from tiny_adk.models import BaseLlm

class MyCustomLlm(BaseLlm):
    def generate(self, request): ...
    def generate_stream(self, request): ...
    async def generate_async(self, request): ...
    async def generate_stream_async(self, request): ...
```

## 运行示例

```bash
# 运行单个示例
python examples/01_basic_agent.py

# 运行自定义 LLM 示例
python examples/06_custom_llm.py
```
