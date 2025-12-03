"""
API 路由定义

提供 REST API 端点：
- POST /api/sessions - 创建 Session
- GET /api/sessions - 列出所有 Sessions
- GET /api/sessions/{user_id}/{session_id} - 获取 Session 详情
- DELETE /api/sessions/{user_id}/{session_id} - 删除 Session
- POST /api/chat - 非流式对话
- POST /api/chat/stream - 流式对话 (SSE)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

if TYPE_CHECKING:
    from tiny_adk import Runner, SessionService

from tiny_adk import EventType


# ==================== Request/Response Models ====================

class CreateSessionRequest(BaseModel):
    """创建 Session 请求"""
    user_id: str
    session_id: Optional[str] = None


class ChatRequest(BaseModel):
    """聊天请求"""
    user_id: str
    session_id: str
    message: str


class ChatResponse(BaseModel):
    """聊天响应"""
    response: str
    events: list[dict[str, Any]]


class SessionInfo(BaseModel):
    """Session 信息"""
    app_name: str
    user_id: str
    session_id: str
    event_count: int
    events: list[dict[str, Any]]


class SessionSummary(BaseModel):
    """Session 摘要（用于列表）"""
    app_name: str
    user_id: str
    session_id: str
    event_count: int
    first_message: Optional[str] = None
    last_message: Optional[str] = None


# ==================== Router Factory ====================

def create_api_router(
    app_name: str,
    runner: 'Runner',
    session_service: 'SessionService',
) -> APIRouter:
    """
    创建 API 路由
    
    Args:
        app_name: 应用名称
        runner: Runner 实例
        session_service: SessionService 实例
        
    Returns:
        FastAPI APIRouter
    """
    router = APIRouter(prefix="/api", tags=["API"])
    
    # ==================== Session 管理 ====================
    
    @router.post("/sessions", response_model=SessionInfo)
    async def create_session(request: CreateSessionRequest):
        """创建新 Session"""
        try:
            session = await session_service.create_session(
                app_name=app_name,
                user_id=request.user_id,
                session_id=request.session_id,
            )
            return SessionInfo(
                app_name=app_name,
                user_id=session.user_id,
                session_id=session.session_id,
                event_count=len(session.events),
                events=[],
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @router.get("/sessions")
    async def list_sessions(
        user_id: Optional[str] = Query(None, description="按用户ID过滤"),
    ):
        """
        列出所有 Sessions
        
        返回 session 列表，包含基本信息和消息摘要
        """
        # 获取所有 sessions（通过访问内部存储）
        all_sessions = []
        for key, session in session_service._sessions.items():
            if key[0] != app_name:
                continue
            if user_id and session.user_id != user_id:
                continue
            
            # 获取第一条和最后一条用户消息
            first_msg = None
            last_msg = None
            for event in session.events:
                if event.event_type.value == 'user_message':
                    if first_msg is None:
                        first_msg = event.content[:100] if event.content else None
                    last_msg = event.content[:100] if event.content else None
            
            all_sessions.append({
                'app_name': session.app_name,
                'user_id': session.user_id,
                'session_id': session.session_id,
                'event_count': len(session.events),
                'first_message': first_msg,
                'last_message': last_msg,
            })
        
        # 按事件数量倒序排列
        all_sessions.sort(key=lambda s: s['event_count'], reverse=True)
        
        return {
            'sessions': all_sessions,
            'total': len(all_sessions),
        }
    
    @router.get("/sessions/{user_id}/{session_id}", response_model=SessionInfo)
    async def get_session(user_id: str, session_id: str):
        """获取 Session 详情，包含所有事件"""
        session = await session_service.get_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionInfo(
            app_name=app_name,
            user_id=session.user_id,
            session_id=session.session_id,
            event_count=len(session.events),
            events=[e.to_dict() for e in session.events],
        )
    
    @router.delete("/sessions/{user_id}/{session_id}")
    async def delete_session(user_id: str, session_id: str):
        """删除 Session"""
        deleted = await session_service.delete_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "deleted"}
    
    @router.delete("/sessions")
    async def clear_all_sessions():
        """清空当前 app 的所有 Sessions"""
        keys_to_delete = [
            key for key in session_service._sessions.keys()
            if key[0] == app_name
        ]
        for key in keys_to_delete:
            del session_service._sessions[key]
        
        return {"status": "cleared", "deleted_count": len(keys_to_delete)}
    
    # ==================== 对话 ====================
    
    @router.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """非流式对话"""
        # 检查 session 是否存在，不存在则创建
        session = await session_service.get_session(
            app_name=app_name,
            user_id=request.user_id,
            session_id=request.session_id,
        )
        if not session:
            session = await session_service.create_session(
                app_name=app_name,
                user_id=request.user_id,
                session_id=request.session_id,
            )
        
        events = []
        response_text = ""
        
        async for event in runner.run_async(
            user_id=request.user_id,
            session_id=request.session_id,
            message=request.message,
        ):
            events.append(event.to_dict())
            if event.event_type == EventType.MODEL_RESPONSE:
                response_text = event.content or ""
        
        return ChatResponse(
            response=response_text,
            events=events,
        )
    
    @router.post("/chat/stream")
    async def chat_stream(request: ChatRequest):
        """流式对话 (SSE)"""
        # 检查 session 是否存在，不存在则创建
        session = await session_service.get_session(
            app_name=app_name,
            user_id=request.user_id,
            session_id=request.session_id,
        )
        if not session:
            session = await session_service.create_session(
                app_name=app_name,
                user_id=request.user_id,
                session_id=request.session_id,
            )
        
        async def event_generator():
            try:
                async for event in runner.run_async(
                    user_id=request.user_id,
                    session_id=request.session_id,
                    message=request.message,
                    stream=True,
                ):
                    # SSE 格式
                    data = json.dumps(event.to_dict(), ensure_ascii=False)
                    yield f"data: {data}\n\n"
                
                # 发送结束标记
                yield "data: [DONE]\n\n"
            except Exception as e:
                error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
                yield f"data: {error_data}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    
    return router
