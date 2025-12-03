# Tiny ADK

这是受到 Google ADK 启发的简单版本，保留了部分核心概念和框架设计，但用更简洁的代码实现。

## 🎯 设计目标

- **保留核心概念**：Agent, Runner, Tool, Session, Event, Config
- **简化实现**：从 2600+ 测试、236+ 文件简化到 ~1000 行核心代码；现在发现越改越多了~
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

这里相较于原本的runner，完全是无状态的；只在run时传入相应的agent, 目前发现也有一定的问题主要就是tracking不好进行

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
7. **06_async_basic.py** - 异步执行（async/await）
8. **07_async_streaming.py** - 异步流式执行

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
