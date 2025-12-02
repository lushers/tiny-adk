# 示例代码说明

这些示例展示了简化版 ADK 的核心功能和用法。

## 📚 示例列表

### 01_basic_agent.py - 基础入门

**学习目标**:
- 理解 Agent、Session、Runner 三大核心组件
- 掌握基本的对话流程
- 了解会话历史如何保存

**核心概念**:
```python
Agent    → 定义"是谁"和"能做什么"
Session  → 保存对话历史
Runner   → 执行 Agent
```

**运行**:
```bash
cd examples
python 01_basic_agent.py
```

---

### 02_agent_with_tools.py - 工具调用

**学习目标**:
- 使用 @tool 装饰器定义工具
- 理解 Agent 如何调用工具
- 观察 Reason-Act 循环

**核心概念**:
```python
@tool(description="...")  → 让 LLM 理解工具用途
Agent(tools=[...])        → 赋予 Agent 能力
Runner 自动编排           → 何时调用工具由 LLM 决定
```

**运行**:
```bash
python 02_agent_with_tools.py
```

---

### 03_streaming.py - 流式执行

**学习目标**:
- 使用 run_stream() 实时获取事件
- 理解事件驱动架构
- 观察每一步的执行过程

**核心概念**:
```python
一切皆事件  → USER_MESSAGE, TOOL_CALL, MODEL_RESPONSE
流式返回    → 实时观察 Agent 的思考过程
```

**运行**:
```bash
python 03_streaming.py
```

---

### 04_multi_turn.py - 多轮对话

**学习目标**:
- 理解 Session 的价值
- 掌握会话序列化和恢复
- 理解为什么 Runner 要设计成无状态

**核心概念**:
```python
Session 保存状态      → 支持多轮对话上下文
Runner 无状态        → 每次从 Session 加载历史
可序列化            → 会话可以保存、恢复、迁移
```

**运行**:
```bash
python 04_multi_turn.py
```

---

### 05_multiple_agents.py - 多 Agent 协作

**学习目标**:
- 创建专业化的 Agent
- 理解不同 Agent 的职责分工
- 了解如何组合多个 Agent

**核心概念**:
```python
专业化 Agent  → 不同角色有不同能力
独立 Session  → 每个对话独立的上下文
Runner 复用   → 同一个 Runner 可执行任何 Agent
```

**运行**:
```bash
python 05_multiple_agents.py
```

---

## 🎓 建议学习顺序

1. **第一步**: 运行 `01_basic_agent.py`
   - 理解三大核心组件
   - 掌握基本流程

2. **第二步**: 运行 `02_agent_with_tools.py`
   - 学习工具定义和使用
   - 观察工具调用过程

3. **第三步**: 运行 `03_streaming.py`
   - 理解事件系统
   - 观察执行细节

4. **第四步**: 运行 `04_multi_turn.py`
   - 理解 Session 的重要性
   - 掌握状态管理

5. **第五步**: 运行 `05_multiple_agents.py`
   - 理解多 Agent 系统
   - 学习如何组织复杂应用

## 💡 常见问题

### Q: 如何添加自己的工具？

A: 非常简单：
```python
@tool(description="你的工具描述")
def your_tool(param1: str, param2: int) -> str:
    # 实现你的逻辑
    return "结果"

agent = Agent(tools=[your_tool])
```

### Q: Session 如何持久化？

A: 使用序列化：
```python
# 保存
import json
with open('session.json', 'w') as f:
    json.dump(session.to_dict(), f)

# 恢复
with open('session.json', 'r') as f:
    data = json.load(f)
    session = Session.from_dict(data)
```

### Q: 如何实现异步执行？

A: 将 Runner 的方法改为 async:
```python
async def run_async(self, agent, session, message):
    # 异步实现
    response = await self._call_llm_async(...)
    return response
```

### Q: 错误处理怎么做？

A: 在 Runner 中捕获异常并记录为事件：
```python
try:
    result = tool.execute(**args)
except Exception as e:
    session.add_event(Event(
        event_type=EventType.ERROR,
        content={'error': str(e)}
    ))
```

## 🔧 扩展建议

想要增强功能？试试这些：

1. **集成真实 LLM**
   - OpenAI API
   - Anthropic Claude
   - 本地模型（Ollama, vLLM(demo中本地部署了server)）

2. **添加持久化**
   - SQLite 数据库
   - Redis
   - 文件系统

3. **实现异步**
   - async/await
   - asyncio

4. **添加监控**
   - 日志记录
   - 性能追踪
   - 错误告警

5. **Web 界面**
   - FastAPI 后端
   - React/Vue 前端
   - WebSocket 实时通信

