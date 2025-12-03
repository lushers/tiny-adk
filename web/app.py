"""
Agent Service - Web 服务封装

提供 REST API 和 Web 界面的统一入口。

使用方式:
    from tiny_adk import Agent
    from web import AgentService
    
    agent = Agent(name="助手", instruction="...")
    service = AgentService(app_name="my_app", agent=agent)
    service.run(host="0.0.0.0", port=8000)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response

from tiny_adk import Agent, Runner, SessionService
from .api import create_api_router

logger = logging.getLogger(__name__)

# 模板目录
TEMPLATES_DIR = Path(__file__).parent / "templates"


class AgentService:
    """
    Agent 服务封装
    
    提供 FastAPI 应用，可以直接用 Uvicorn 运行。
    
    目录结构:
        tiny-adk/
        ├── tiny_adk/       # 核心库
        ├── web/            # Web 服务（本模块）
        │   ├── __init__.py
        │   ├── app.py      # 本文件
        │   ├── api.py      # API 路由
        │   └── templates/
        │       ├── chat.html
        │       └── sessions.html
        └── examples/
    """
    
    def __init__(
        self,
        app_name: str,
        agent: Agent,
        session_service: Optional[SessionService] = None,
    ):
        """
        初始化服务
        
        Args:
            app_name: 应用名称
            agent: Agent 实例
            session_service: Session 服务（可选，默认使用内存存储）
        """
        self.app_name = app_name
        self.agent = agent
        self.session_service = session_service or SessionService()
        self.runner = Runner(
            app_name=app_name,
            agent=agent,
            session_service=self.session_service,
        )
        
        # 创建 FastAPI 应用
        self.app = FastAPI(
            title=f"{app_name} - Agent Service",
            description="tiny_adk Agent Service API",
            version="0.4.0",
        )
        
        # 注册路由
        self._register_routes()
    
    def _register_routes(self):
        """注册路由"""
        # 注册 API 路由
        api_router = create_api_router(
            app_name=self.app_name,
            runner=self.runner,
            session_service=self.session_service,
        )
        self.app.include_router(api_router)
        
        # 注册 Web 界面路由
        @self.app.get("/", response_class=HTMLResponse)
        async def index():
            """Web 聊天界面"""
            return self._render_template("chat.html")
        
        @self.app.get("/sessions", response_class=HTMLResponse)
        async def sessions():
            """Session 浏览界面"""
            return self._render_template("sessions.html")
        
        @self.app.get("/favicon.ico")
        async def favicon():
            """返回 favicon"""
            # 1x1 透明 PNG
            transparent_png = bytes([
                0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
                0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
                0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
                0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
                0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,
                0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
                0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
                0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
                0x42, 0x60, 0x82
            ])
            return Response(content=transparent_png, media_type="image/png")
    
    def _render_template(self, template_name: str) -> str:
        """
        渲染模板
        
        Args:
            template_name: 模板文件名
            
        Returns:
            渲染后的 HTML
        """
        template_path = TEMPLATES_DIR / template_name
        
        if not template_path.exists():
            return f"<h1>Template not found: {template_name}</h1>"
        
        html = template_path.read_text(encoding="utf-8")
        
        # 简单的模板变量替换
        html = html.replace("{{ app_name }}", self.app_name)
        html = html.replace("{{ agent_name }}", self.agent.name)
        
        return html
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """
        启动服务
        
        Args:
            host: 监听地址
            port: 监听端口
            **kwargs: 传递给 uvicorn.run 的其他参数
        """
        import uvicorn
        
        print(f"\n🚀 启动 {self.app_name} 服务...")
        print(f"   Agent: {self.agent.name}")
        print(f"   地址: http://{host}:{port}")
        print(f"   Sessions: http://{host}:{port}/sessions")
        print(f"   API 文档: http://{host}:{port}/docs")
        print()
        
        uvicorn.run(self.app, host=host, port=port, **kwargs)
