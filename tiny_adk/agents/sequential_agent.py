"""SequentialAgent - 顺序执行子 Agent（编排器）"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator

from typing_extensions import override

from .base_agent import BaseAgent

if TYPE_CHECKING:
    from ..events import Event
    from ..session import Session

logger = logging.getLogger(__name__)


class SequentialAgent(BaseAgent):
    """
    顺序执行 Agent - 按顺序运行所有子 Agent
    
    这是一个编排器（shell agent），它本身不调用 LLM，
    只是按顺序执行子 Agent。
    
    设计原则：
    - 组合优于继承：通过组合子 Agent 来工作
    - 单一职责：只负责编排子 Agent 的执行顺序
    - 避免不必要的复杂性：不继承 LlmAgent 的 LLM 相关属性
    
    适用场景：
    - 流水线处理：分析 → 生成 → 审核
    - 多阶段任务：规划 → 执行 → 总结
    
    Example:
        pipeline = SequentialAgent(
            name="pipeline",
            sub_agents=[
                LlmAgent(name="analyzer", instruction="分析需求"),
                LlmAgent(name="generator", instruction="生成方案"),
                LlmAgent(name="reviewer", instruction="审核结果"),
            ]
        )
    """
    
    @override
    async def _run_async_impl(
        self,
        session: 'Session',
        llm: Any = None,
    ) -> AsyncGenerator['Event', None]:
        """顺序执行所有子 Agent"""
        if not self.sub_agents:
            logger.warning(f"[{self.name}] No sub_agents to run")
            return
        
        logger.info(f"[{self.name}] Starting sequential execution of {len(self.sub_agents)} agents")
        
        for i, sub_agent in enumerate(self.sub_agents):
            logger.info(f"[{self.name}] Running {i+1}/{len(self.sub_agents)}: {sub_agent.name}")
            
            async for event in sub_agent.run_async(session, llm):
                # 标记来源
                event.metadata['sequential_agent'] = self.name
                event.metadata['sequential_index'] = i
                yield event
        
        logger.info(f"[{self.name}] Sequential execution completed")
