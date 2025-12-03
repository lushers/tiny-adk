"""
Web 服务模块

提供 REST API 和 Web 界面：
- AgentService: 服务封装类
- create_api_router: API 路由工厂

使用方式:
    from tiny_adk import Agent
    from web import AgentService
    
    agent = Agent(name="助手", instruction="...")
    service = AgentService(app_name="my_app", agent=agent)
    service.run(host="0.0.0.0", port=8000)
    
    # 访问 /sessions 查看所有 session 和事件
"""

from .app import AgentService
from .api import create_api_router

__all__ = ['AgentService', 'create_api_router']
