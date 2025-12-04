"""LoopAgent - 循环执行子 Agent（编排器）"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional

from typing_extensions import override

from .base_agent import BaseAgent

if TYPE_CHECKING:
    from ..events import Event
    from ..session import Session

logger = logging.getLogger(__name__)


class LoopAgent(BaseAgent):
    """
    循环执行 Agent - 重复运行子 Agent 直到满足条件
    
    这是一个编排器（shell agent），它本身不调用 LLM，
    只是循环执行子 Agent 直到满足退出条件。
    
    设计原则：
    - 组合优于继承：通过组合子 Agent 来工作
    - 单一职责：只负责循环编排子 Agent 的执行
    - 避免不必要的复杂性：不继承 LlmAgent 的 LLM 相关属性
    
    停止条件：
    - 达到 max_iterations
    - 子 Agent 返回 escalate 信号
    
    Example:
        refiner = LoopAgent(
            name="refiner",
            max_iterations=3,
            sub_agents=[
                LlmAgent(name="writer", instruction="写作"),
                LlmAgent(name="critic", instruction="评审，满意则 escalate"),
            ]
        )
    """
    
    max_iterations: Optional[int] = 5
    """最大循环次数，None 表示无限循环直到 escalate"""
    
    @override
    async def _run_async_impl(
        self,
        session: 'Session',
        llm: Any = None,
    ) -> AsyncGenerator['Event', None]:
        """循环执行子 Agent"""
        if not self.sub_agents:
            logger.warning(f"[{self.name}] No sub_agents to run")
            return
        
        logger.info(f"[{self.name}] Starting loop (max={self.max_iterations})")
        
        loop_idx = 0
        should_exit = False
        
        while not should_exit:
            # 检查迭代次数
            if self.max_iterations and loop_idx >= self.max_iterations:
                logger.info(f"[{self.name}] Max iterations reached")
                break
            
            logger.info(f"[{self.name}] Loop iteration {loop_idx + 1}")
            
            for sub_agent in self.sub_agents:
                async for event in sub_agent.run_async(session, llm):
                    event.metadata['loop_agent'] = self.name
                    event.metadata['loop_iteration'] = loop_idx
                    yield event
                    
                    # 检查 escalate 信号
                    if event.metadata.get('escalate'):
                        logger.info(f"[{self.name}] Received escalate signal")
                        should_exit = True
                        break
                
                if should_exit:
                    break
            
            loop_idx += 1
        
        logger.info(f"[{self.name}] Loop completed after {loop_idx} iterations")
