"""Agent - AI 代理的核心定义（支持多 Agent）"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Iterator, Optional

from pydantic import BaseModel, Field, PrivateAttr, field_validator

from .flows import SimpleFlow
from .models import BaseLlm
from .tools import BaseTool, Tool

if TYPE_CHECKING:
    from .events import Event
    from .flows import BaseFlow
    from .session import InvocationContext, Session

logger = logging.getLogger(__name__)


class Agent(BaseModel):
    """
    Agent 配置容器（使用 Pydantic，支持多 Agent）
    
    核心设计理念:
    - Agent 是配置，不包含状态
    - Agent 定义了"是谁"、"会什么"、"怎么做"
    - 实际执行由 Runner 负责
    - 支持树形结构：parent_agent / sub_agents
    
    Multi-Agent 支持:
    - sub_agents: 子 Agent 列表
    - parent_agent: 父 Agent（自动设置）
    - transfer_to_agent: 通过内置工具切换到其他 Agent
    - SequentialAgent: 顺序执行多个 Agent
    """
    
    model_config = {"arbitrary_types_allowed": True}
    
    # 基本配置
    name: str
    instruction: str = ""
    description: str = ""
    
    # 模型配置
    model: str | Any = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    max_iterations: int = 10
    
    # 工具
    tools: list[BaseTool | Tool] = Field(default_factory=list)
    
    # 多 Agent 支持
    sub_agents: list['Agent'] = Field(default_factory=list)
    parent_agent: Optional['Agent'] = Field(default=None, exclude=True)  # 不序列化
    
    # Agent 跳转控制
    disallow_transfer_to_parent: bool = False
    """禁止 LLM 控制跳转回父 Agent"""
    disallow_transfer_to_peers: bool = False
    """禁止 LLM 控制跳转到同级 Agent"""
    
    # 私有字段
    _flow: Optional[Any] = PrivateAttr(default=None)
    _llm: Optional[Any] = PrivateAttr(default=None)
    
    @field_validator('name', mode='after')
    @classmethod
    def validate_name(cls, value: str) -> str:
        """验证 Agent 名称必须是有效标识符"""
        if not value.isidentifier():
            raise ValueError(
                f"Invalid agent name: '{value}'. "
                "Agent name must be a valid Python identifier."
            )
        if value == 'user':
            raise ValueError("Agent name cannot be 'user' (reserved).")
        return value
    
    def model_post_init(self, __context: Any) -> None:
        """Pydantic 初始化完成后的钩子"""
        super().model_post_init(__context)
        
        # 设置子 Agent 的 parent_agent
        self._set_parent_for_sub_agents()
        
        # 创建 Flow
        self._flow = SimpleFlow(max_iterations=self.max_iterations)
        logger.debug(f"[Agent {self.name}] Created SimpleFlow")
        
        # 如果 model 是 BaseLlm 实例，保存引用
        if isinstance(self.model, BaseLlm):
            self._llm = self.model
    
    def _set_parent_for_sub_agents(self) -> None:
        """为所有子 Agent 设置 parent_agent"""
        for sub_agent in self.sub_agents:
            if sub_agent.parent_agent is not None:
                raise ValueError(
                    f"Agent '{sub_agent.name}' already has a parent agent "
                    f"(current: '{sub_agent.parent_agent.name}', trying to add: '{self.name}')"
                )
            sub_agent.parent_agent = self
    
    # ==================== 树形结构操作 ====================
    
    @property
    def root_agent(self) -> 'Agent':
        """获取根 Agent"""
        root = self
        while root.parent_agent is not None:
            root = root.parent_agent
        return root
    
    def find_agent(self, name: str) -> Optional['Agent']:
        """在当前 Agent 及其所有后代中查找指定名称的 Agent"""
        if self.name == name:
            return self
        return self.find_sub_agent(name)
    
    def find_sub_agent(self, name: str) -> Optional['Agent']:
        """在后代中查找指定名称的 Agent"""
        for sub_agent in self.sub_agents:
            if result := sub_agent.find_agent(name):
                return result
        return None
    
    def get_transferable_agents(self) -> list['Agent']:
        """
        获取可以跳转到的 Agent 列表
        
        包括：
        - 子 Agent（如果有）
        - 父 Agent（如果允许）
        - 同级 Agent（如果允许）
        """
        agents = []
        
        # 子 Agent
        agents.extend(self.sub_agents)
        
        # 父 Agent
        if not self.disallow_transfer_to_parent and self.parent_agent:
            agents.append(self.parent_agent)
        
        # 同级 Agent（兄弟）
        if not self.disallow_transfer_to_peers and self.parent_agent:
            for sibling in self.parent_agent.sub_agents:
                if sibling.name != self.name:
                    agents.append(sibling)
        
        return agents
    
    # ==================== 属性访问 ====================
    
    @property
    def flow(self) -> 'BaseFlow':
        """返回 Flow 实例"""
        if self._flow is None:
            self._flow = SimpleFlow(max_iterations=self.max_iterations)
        return self._flow
    
    @property
    def llm(self) -> Optional[BaseLlm]:
        """返回 LLM 实例（如果有）"""
        return self._llm
    
    def get_model_name(self) -> str:
        """获取模型名称"""
        if isinstance(self.model, str):
            return self.model
        elif hasattr(self.model, 'model'):
            return self.model.model
        return "unknown"
    
    # ==================== 序列化 ====================
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            'name': self.name,
            'instruction': self.instruction,
            'model': self.get_model_name(),
            'description': self.description,
            'tools': [t.to_function_declaration() for t in self.tools],
            'sub_agents': [a.name for a in self.sub_agents],
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'max_iterations': self.max_iterations,
        }
    
    # ==================== 系统提示词 ====================
    
    def get_system_prompt(self) -> str:
        """
        构建系统提示词
        
        包含：Agent 身份、指令、工具列表、可跳转的 Agent 列表
        """
        prompt = f"You are {self.name}.\n\n"
        prompt += f"{self.instruction}\n\n"
        
        if self.tools:
            prompt += "You have access to the following tools:\n"
            for tool in self.tools:
                prompt += f"- {tool.name}: {tool.description}\n"
            prompt += "\n"
        
        # 添加可跳转的 Agent 信息
        transferable = self.get_transferable_agents()
        if transferable:
            prompt += "You can transfer control to the following agents:\n"
            for agent in transferable:
                desc = agent.description or f"Agent {agent.name}"
                prompt += f"- {agent.name}: {desc}\n"
            prompt += "\nUse the 'transfer_to_agent' tool to hand over to another agent.\n"
        
        return prompt
    
    # ==================== 执行 ====================
    
    async def run_async(
        self, 
        session: 'Session',
        llm: Optional[BaseLlm] = None,
    ) -> AsyncGenerator['Event', None]:
        """
        Agent 执行入口
        
        Args:
            session: Session 对象
            llm: LLM 实例（可选，如果 Agent 自带则使用自带的）
            
        Yields:
            Event 对象
        """
        from .events import Event, EventType
        
        logger.info(f"[Agent {self.name}] Starting execution")
        
        # 获取 LLM
        llm = llm or self._llm
        if llm is None:
            raise ValueError(f"Agent {self.name} has no LLM configured")
        
        # 检查是否有待处理的 Agent 跳转
        transfer_target = self._check_pending_transfer(session)
        if transfer_target:
            logger.info(f"[Agent {self.name}] Resuming transfer to {transfer_target.name}")
            async for event in transfer_target.run_async(session, llm):
                yield event
            return
        
        # 委托给 Flow 执行
        async for event in self.flow.run_stream_async(self, session, llm):
            yield event
            
            # 检查是否需要跳转到其他 Agent
            if self._should_transfer(event):
                target_name = event.metadata.get('transfer_to_agent')
                target_agent = self.root_agent.find_agent(target_name)
                if target_agent:
                    logger.info(f"[Agent {self.name}] Transferring to {target_name}")
                    async for sub_event in target_agent.run_async(session, llm):
                        yield sub_event
                    return
    
    def run(
        self, 
        session: 'Session',
        llm: Optional[BaseLlm] = None,
    ) -> Iterator['Event']:
        """同步执行（非流式）"""
        from .events import Event, EventType
        
        logger.info(f"[Agent {self.name}] Starting execution (sync)")
        
        llm = llm or self._llm
        if llm is None:
            raise ValueError(f"Agent {self.name} has no LLM configured")
        
        for event in self.flow.run_stream(self, session, llm):
            yield event
    
    def _check_pending_transfer(self, session: 'Session') -> Optional['Agent']:
        """检查是否有待处理的 Agent 跳转"""
        if not session.events:
            return None
        
        # 查找最后一个 AGENT_TRANSFER 事件
        from .events import EventType
        for event in reversed(session.events):
            if event.event_type == EventType.AGENT_TRANSFER:
                target_name = event.content.get('target_agent')
                if target_name and target_name != self.name:
                    return self.root_agent.find_agent(target_name)
                break
        return None
    
    def _should_transfer(self, event: 'Event') -> bool:
        """检查事件是否触发 Agent 跳转"""
        from .events import EventType
        return (
            event.event_type == EventType.AGENT_TRANSFER
            and event.metadata.get('transfer_to_agent')
        )


class SequentialAgent(Agent):
    """
    顺序执行 Agent - 按顺序运行所有子 Agent
    
    用于流水线式的任务处理，如：
    - 分析 → 生成 → 审核
    - 规划 → 执行 → 总结
    
    Example:
        pipeline = SequentialAgent(
            name="pipeline",
            sub_agents=[
                Agent(name="analyzer", instruction="分析用户需求"),
                Agent(name="generator", instruction="生成解决方案"),
                Agent(name="reviewer", instruction="审核结果"),
            ]
        )
    """
    
    async def run_async(
        self, 
        session: 'Session',
        llm: Optional[BaseLlm] = None,
    ) -> AsyncGenerator['Event', None]:
        """顺序执行所有子 Agent"""
        from .events import Event, EventType
        
        if not self.sub_agents:
            logger.warning(f"[SequentialAgent {self.name}] No sub_agents to run")
            return
        
        logger.info(f"[SequentialAgent {self.name}] Starting sequential execution")
        
        for i, sub_agent in enumerate(self.sub_agents):
            logger.info(f"[SequentialAgent {self.name}] Running sub_agent {i+1}/{len(self.sub_agents)}: {sub_agent.name}")
            
            async for event in sub_agent.run_async(session, llm or self._llm):
                # 标记事件来源
                event.metadata['sequential_agent'] = self.name
                event.metadata['sequential_index'] = i
                yield event
        
        logger.info(f"[SequentialAgent {self.name}] Completed all sub_agents")


class LoopAgent(Agent):
    """
    循环执行 Agent - 重复运行子 Agent 直到满足条件
    
    停止条件：
    - 达到 max_loop_iterations
    - 子 Agent 返回 escalate 信号
    
    Example:
        refiner = LoopAgent(
            name="refiner",
            max_loop_iterations=3,
            sub_agents=[
                Agent(name="writer", instruction="写作"),
                Agent(name="critic", instruction="评审并决定是否需要继续"),
            ]
        )
    """
    
    max_loop_iterations: int = 5
    """最大循环次数"""
    
    async def run_async(
        self, 
        session: 'Session',
        llm: Optional[BaseLlm] = None,
    ) -> AsyncGenerator['Event', None]:
        """循环执行子 Agent"""
        from .events import Event, EventType
        
        if not self.sub_agents:
            logger.warning(f"[LoopAgent {self.name}] No sub_agents to run")
            return
        
        logger.info(f"[LoopAgent {self.name}] Starting loop execution (max={self.max_loop_iterations})")
        
        for loop_idx in range(self.max_loop_iterations):
            logger.info(f"[LoopAgent {self.name}] Loop iteration {loop_idx + 1}/{self.max_loop_iterations}")
            should_exit = False
            
            for sub_agent in self.sub_agents:
                async for event in sub_agent.run_async(session, llm or self._llm):
                    event.metadata['loop_agent'] = self.name
                    event.metadata['loop_iteration'] = loop_idx
                    yield event
                    
                    # 检查 escalate 信号
                    if event.metadata.get('escalate'):
                        logger.info(f"[LoopAgent {self.name}] Received escalate signal, exiting loop")
                        should_exit = True
                        break
                
                if should_exit:
                    break
            
            if should_exit:
                break
        
        logger.info(f"[LoopAgent {self.name}] Loop completed")
