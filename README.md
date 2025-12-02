# Tiny ADK - 简化版 Agent Development Kit

这是 Google ADK 的简化版本，保留了核心概念和框架设计，但用更简洁的代码实现。

## 🎯 设计目标

- **保留核心概念**：Agent, Runner, Tool, Session, Event, Config
- **简化实现**：从 2600+ 测试、236+ 文件简化到 ~1000 行核心代码
- **清晰易懂**：每个概念都有详细注释说明设计理念
- **完整可运行**：虽然简化，但保持了完整的执行流程

## 📦 核心组件

### 1. Agent - AI 代理的蓝图

```python
from tiny_adk import Agent

agent = Agent(
    name='助手',
    instruction='你是一个友好的助手',
    model='gpt-4',
    tools=[...]  # 可选的工具列表
)
```

**核心设计理念**：
- Agent 是**配置**，不包含状态
- 定义了"是谁"（name）、"会什么"（tools）、"怎么做"（instruction）
- 可以被复用、序列化、版本化

### 2. Session - 会话状态管理

```python
from tiny_adk import Session

session = Session()
# Session 存储所有对话事件
# 可以序列化、持久化、恢复
```

**核心设计理念**：
- Session 是**有状态的**，存储所有历史事件
- 支持多轮对话的上下文
- 可以跨进程、跨时间持久化

### 3. Runner - 无状态执行引擎

```python
from tiny_adk import Runner

runner = Runner()
response = runner.run(agent, session, "用户消息")
```

**核心设计理念**：
- Runner 是**无状态的**，不保存对话历史
- 负责编排 "Reason-Act" 循环
- 每次执行从 Session 加载历史，结果保存回 Session

### 4. Tool - 可调用的函数

```python
from tiny_adk import tool

@tool(description='搜索网页')
def search(query: str) -> str:
    return f"搜索结果: {query}"
```

**核心设计理念**：
- Tool 是带描述的函数，让 LLM 理解和调用
- 自动提取函数签名和参数信息
- 执行结果会记录为事件

### 5. Event - 事件系统

```python
from tiny_adk import Event, EventType

# 所有操作都是事件
event = Event(
    event_type=EventType.USER_MESSAGE,
    content="你好"
)
```

**核心设计理念**：
- 一切皆事件：用户消息、模型响应、工具调用都是事件
- 事件组成会话历史
- 支持流式处理、实时监控、审计追踪

### 6. Config - 配置管理

```python
from tiny_adk import Config, get_config

# 自动加载配置（从 tiny_adk.yaml 或环境变量）
config = get_config()

# 或手动加载指定配置文件
config = Config.load(config_file="my_config.yaml")
```

**核心设计理念**：
- 支持 YAML/JSON 配置文件
- 支持环境变量覆盖
- 配置优先级：代码参数 > 环境变量 > 配置文件 > 默认值

## 🚀 快速开始

### 基础示例（不需要真实 LLM）

```python
from tiny_adk import Agent, Runner, Session

# 1. 创建 Agent
agent = Agent(
    name='助手',
    instruction='你是一个友好的助手'
)

# 2. 创建 Session 和 Runner
session = Session()
runner = Runner()  # 使用模拟的 LLM 响应

# 3. 执行对话
response = runner.run(agent, session, "你好！")
print(response)
```

### 🔥 使用真实 LLM

**方式一：使用配置文件（推荐）**

创建 `tiny_adk.yaml` 配置文件：

```yaml
llm:
  api_base: "http://localhost:8000/v1"
  api_key: "EMPTY"
  model: "Qwen/Qwen2.5-7B-Instruct"

runner:
  show_thinking: false
  show_request: false
```

然后直接使用：

```python
from tiny_adk import Agent, Runner, Session

# Runner 会自动读取 tiny_adk.yaml 配置
runner = Runner()

agent = Agent(
    name='智能助手',
    instruction='你是一个有帮助的AI助手。',
)

session = Session()
response = runner.run(agent, session, "介绍一下 Python")
print(response)
```

**方式二：代码中直接配置**

```python
from tiny_adk import Agent, Runner, Session

# 连接到 vLLM server
runner = Runner(
    api_base='http://localhost:8000/v1',
    api_key='EMPTY',
)

agent = Agent(
    name='智能助手',
    model='Qwen/Qwen2.5-7B-Instruct',
    instruction='你是一个有帮助的AI助手。',
)

session = Session()
response = runner.run(agent, session, "介绍一下 Python")
print(response)
```

**安装依赖**:
```bash
pip install openai pyyaml  # OpenAI 兼容客户端 + YAML 支持
```

### 示例 2: 带工具的 Agent

```python
from tiny_adk import Agent, Runner, Session, tool

# 定义工具
@tool(description='查询天气')
def get_weather(city: str) -> str:
    return f"{city}：晴天，25°C"

# 创建带工具的 Agent
agent = Agent(
    name='天气助手',
    instruction='你可以查询天气',
    tools=[get_weather]
)

session = Session()
runner = Runner()

response = runner.run(agent, session, "北京天气怎么样？")
```

### 示例 3: 流式执行

```python
# 实时获取每个事件
for event in runner.run_stream(agent, session, "你好"):
    print(f"[{event.event_type.value}] {event.content}")
```

## ⚙️ 配置说明

### 配置文件

复制 `tiny_adk.example.yaml` 为 `tiny_adk.yaml` 并根据需要修改：

```yaml
# LLM API 配置
llm:
  api_base: "http://localhost:8000/v1"
  api_key: "EMPTY"
  model: "your-model-name"
  temperature: 0.7
  max_tokens: null
  timeout: 60.0

# Runner 运行时配置
runner:
  show_thinking: false  # 是否显示模型思考过程
  show_request: false   # 是否显示 API 请求详情
  max_iterations: 10    # 最大迭代次数
```

### 环境变量

也可以通过环境变量配置（会覆盖配置文件）：

```bash
export TINY_ADK_API_BASE="http://localhost:8000/v1"
export TINY_ADK_API_KEY="EMPTY"
export TINY_ADK_MODEL="your-model-name"
export TINY_ADK_SHOW_THINKING="true"
```

## 🏗️ 架构设计

### Agent ↔ Runner ↔ Session 的分离

这是 ADK 最核心的设计模式：

```
┌─────────────────────────────────────────────────┐
│                                                 │
│  Agent (配置)      Runner (无状态)   Session (状态) │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐ │
│  │ name    │      │ run()   │      │ events  │ │
│  │ tools   │─────>│ stream()│<────>│ history │ │
│  │ prompt  │      │ loop()  │      │ data    │ │
│  └─────────┘      └─────────┘      └─────────┘ │
│                                                 │
└─────────────────────────────────────────────────┘
```

**为什么这样设计？**

1. **Agent 无状态** → 可以并发执行多个会话
2. **Runner 无状态** → 可以是单例或短生命周期
3. **Session 有状态** → 可以持久化、恢复、迁移
4. **分离关注点** → 配置、执行、状态各自独立

### Reason-Act 循环

Runner 的核心执行流程：

```
1. 从 Session 加载历史
     ↓
2. 调用 LLM 推理（Reason）
     ↓
3. 是否需要工具？
     ├─ 是 → 执行工具（Act）→ 回到步骤 2
     └─ 否 → 返回最终响应
     ↓
4. 保存事件到 Session
```

### 事件驱动架构

```
用户输入 → Event(USER_MESSAGE)
    ↓
LLM 调用 → Event(MODEL_REQUEST)
    ↓
LLM 响应 → Event(MODEL_RESPONSE)
    ↓
工具调用 → Event(TOOL_CALL)
    ↓
工具结果 → Event(TOOL_RESPONSE)
    ↓
所有事件存入 Session
```

## 📚 完整示例

查看 `examples/` 目录：

1. **01_basic_agent.py** - 基础对话
2. **02_agent_with_tools.py** - 工具调用
3. **03_streaming.py** - 流式执行
4. **03b_streaming_with_thinking.py** - 流式执行（显示思考过程）
5. **04_multi_turn.py** - 多轮对话
6. **05_multiple_agents.py** - 多 Agent 协作

## 🔍 与完整版 ADK 的对比

| 特性 | 完整版 ADK | Tiny ADK |
|------|-----------|----------|
| 核心概念 | ✅ Agent, Runner, Session, Tool, Event | ✅ 完全保留 + Config |
| 代码量 | ~50,000 行 | ~1000 行核心代码 |
| Agent 类型 | LlmAgent, LoopAgent, ParallelAgent, Sequential... | 统一的 Agent |
| 工具集成 | 50+ 内置工具（Google Search, BigQuery...） | 自定义工具接口 |
| 模型支持 | Gemini, Anthropic, LiteLLM, 本地模型 | OpenAI 兼容 API |
| Session 后端 | 内存、Vertex AI、Spanner | 仅内存 |
| 配置管理 | 复杂配置系统 | YAML/环境变量 |
| 评估框架 | 完整的评估系统（47 文件） | 无 |
| CLI/Web UI | 完整的工具链 | 无 |
| 生产特性 | 遥测、认证、A2A 协议、上下文缓存... | 无 |

## 💡 核心思想总结

### 1. 配置与状态分离
- Agent = 配置（可复用）
- Session = 状态（可持久化）
- Runner = 执行（无状态）

### 2. 事件驱动
- 一切操作都是事件
- 事件可观察、可审计、可回放

### 3. 工具即函数
- 工具是带描述的函数
- LLM 根据描述决定何时调用

### 4. Reason-Act 循环
- LLM 推理 → 决定动作 → 执行 → 继续推理
- 循环直到得出最终答案

## 🎓 学习路径

1. **理解核心概念** - 阅读各个模块的代码注释
2. **运行示例** - 从 01 到 05 逐个运行
3. **自定义工具** - 尝试添加自己的工具
4. **扩展功能** - 参考完整版 ADK 添加需要的功能

## 📖 代码导读

- `config.py` (~250 行) - 配置管理系统
- `events.py` (~50 行) - 事件系统定义
- `tools.py` (~80 行) - 工具接口和装饰器
- `agents.py` (~60 行) - Agent 配置类
- `session.py` (~130 行) - 会话管理
- `runner.py` (~880 行) - 核心执行引擎

**建议阅读顺序**：config → events → tools → agents → session → runner

## 🔗 参考资源

- [完整版 ADK](https://github.com/google/adk-python)
- [ADK 文档](https://google.github.io/adk-docs)

## ⚠️ 限制说明

这是一个教学性质的简化版本，**不应用于生产环境**。缺少：
- 错误处理和重试
- 并发控制
- 安全性保护
- 性能优化
- 持久化存储
- 监控和日志

如需生产级别的功能，请使用完整版 ADK。

## 📝 许可证

与完整版 ADK 相同：Apache License 2.0
