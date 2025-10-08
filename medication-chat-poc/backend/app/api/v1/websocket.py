# WebSocket API for real-time medical chat

import json
import logging
from typing import Dict, List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from datetime import datetime

from app.services.medical_ai_service import MedicalAIService

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    """Manage WebSocket connections"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_data: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_data[session_id] = {
            "connected_at": datetime.now().isoformat(),
            "message_count": 0,
            "last_activity": datetime.now().isoformat()
        }
        logger.info(f"🔗 WebSocket connected: {session_id}")

    def disconnect(self, session_id: str):
        """Remove WebSocket connection"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.session_data:
            del self.session_data[session_id]
        logger.info(f"❌ WebSocket disconnected: {session_id}")

    async def send_personal_message(self, message: dict, session_id: str):
        """Send message to specific session"""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_text(json.dumps(message, ensure_ascii=False))
                # Update session activity
                if session_id in self.session_data:
                    self.session_data[session_id]["last_activity"] = datetime.now().isoformat()
                    self.session_data[session_id]["message_count"] += 1
            except Exception as e:
                logger.error(f"Error sending message to {session_id}: {e}")
                self.disconnect(session_id)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected sessions"""
        disconnected = []
        for session_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message, ensure_ascii=False))
            except Exception as e:
                logger.error(f"Error broadcasting to {session_id}: {e}")
                disconnected.append(session_id)

        # Remove disconnected sessions
        for session_id in disconnected:
            self.disconnect(session_id)

    def get_stats(self) -> Dict:
        """Get connection statistics"""
        return {
            "active_connections": len(self.active_connections),
            "sessions": list(self.active_connections.keys()),
            "session_data": self.session_data
        }


# Global connection manager
manager = ConnectionManager()


def get_medical_ai_service(request: Request) -> MedicalAIService:
    """Get medical AI service from app state"""
    return request.app.state.medical_ai_service


@router.websocket("/chat/{session_id}")
async def websocket_medical_chat(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time medical chat"""

    await manager.connect(websocket, session_id)

    # Send welcome message
    welcome_message = {
        "type": "system",
        "message": "🏥 เชื่อมต่อกับระบบปรึกษาทางการแพทย์แล้ว",
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "status": "connected"
    }
    await manager.send_personal_message(welcome_message, session_id)

    try:
        # Get medical AI service (we'll need to initialize it differently for WebSocket)
        medical_ai = MedicalAIService()
        await medical_ai.initialize()

        while True:
            # Receive message from client
            data = await websocket.receive_text()

            try:
                message_data = json.loads(data)
                user_message = message_data.get("message", "")
                message_type = message_data.get("type", "chat")

                if not user_message:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": "ข้อความว่าง กรุณาระบุอาการหรือคำถาม",
                        "timestamp": datetime.now().isoformat()
                    }, session_id)
                    continue

                # Send typing indicator
                await manager.send_personal_message({
                    "type": "typing",
                    "message": "💭 กำลังวิเคราะห์อาการ...",
                    "timestamp": datetime.now().isoformat()
                }, session_id)

                # Process medical consultation
                if message_type == "chat":
                    start_time = datetime.now()

                    result = await medical_ai.process_medical_consultation(
                        message=user_message,
                        session_id=session_id
                    )

                    processing_time = (datetime.now() - start_time).total_seconds() * 1000

                    # Send medical response
                    response_message = {
                        "type": "medical_response",
                        "message": result.get("message", ""),
                        "diagnosis": result.get("diagnosis"),
                        "treatment": result.get("treatment"),
                        "triage": result.get("triage"),
                        "disclaimer": result.get("disclaimer"),
                        "recommendation": result.get("recommendation"),
                        "session_id": session_id,
                        "processing_time_ms": processing_time,
                        "timestamp": datetime.now().isoformat()
                    }

                elif message_type == "emergency_check":
                    # Quick emergency assessment
                    emergency_result = await medical_ai.check_emergency_keywords(user_message)

                    response_message = {
                        "type": "emergency_assessment",
                        "is_emergency": emergency_result["is_emergency"],
                        "keywords": emergency_result["keywords"],
                        "message": "🚨 ตรวจพบคำหลักฉุกเฉิน โทร 1669 ทันที" if emergency_result["is_emergency"] else "ไม่พบสัญญาณฉุกเฉิน",
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat()
                    }

                elif message_type == "symptom_check":
                    # Quick symptom check
                    triage_result = await medical_ai.perform_triage_assessment(user_message)

                    response_message = {
                        "type": "symptom_assessment",
                        "urgency": triage_result["urgency"],
                        "triage_level": triage_result["triage_level"],
                        "risk_score": triage_result["risk_score"],
                        "message": f"ระดับความเร่งด่วน: {triage_result['urgency']}",
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat()
                    }

                else:
                    response_message = {
                        "type": "error",
                        "message": f"ประเภทข้อความไม่รองรับ: {message_type}",
                        "timestamp": datetime.now().isoformat()
                    }

                await manager.send_personal_message(response_message, session_id)

            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "รูปแบบข้อความไม่ถูกต้อง กรุณาส่ง JSON",
                    "timestamp": datetime.now().isoformat()
                }, session_id)
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await manager.send_personal_message({
                    "type": "error",
                    "message": "เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่",
                    "timestamp": datetime.now().isoformat()
                }, session_id)

    except WebSocketDisconnect:
        manager.disconnect(session_id)
        logger.info(f"🔌 WebSocket client {session_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}")
        manager.disconnect(session_id)
    finally:
        # Cleanup resources
        if 'medical_ai' in locals():
            await medical_ai.cleanup()


@router.get("/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics"""
    return {
        "success": True,
        "data": manager.get_stats(),
        "timestamp": datetime.now().isoformat()
    }


@router.post("/broadcast")
async def broadcast_message(message_data: dict):
    """Broadcast message to all connected WebSocket clients"""

    broadcast_message = {
        "type": "broadcast",
        "message": message_data.get("message", ""),
        "sender": message_data.get("sender", "system"),
        "timestamp": datetime.now().isoformat()
    }

    await manager.broadcast(broadcast_message)

    return {
        "success": True,
        "message": "Message broadcasted to all connected clients",
        "active_connections": len(manager.active_connections),
        "timestamp": datetime.now().isoformat()
    }


@router.delete("/session/{session_id}")
async def disconnect_session(session_id: str):
    """Manually disconnect a WebSocket session"""

    if session_id in manager.active_connections:
        # Send disconnect message
        await manager.send_personal_message({
            "type": "system",
            "message": "การเชื่อมต่อถูกปิดโดยระบบ",
            "timestamp": datetime.now().isoformat()
        }, session_id)

        # Close connection
        websocket = manager.active_connections[session_id]
        await websocket.close()
        manager.disconnect(session_id)

        return {
            "success": True,
            "message": f"Session {session_id} disconnected",
            "timestamp": datetime.now().isoformat()
        }
    else:
        return {
            "success": False,
            "message": f"Session {session_id} not found",
            "timestamp": datetime.now().isoformat()
        }