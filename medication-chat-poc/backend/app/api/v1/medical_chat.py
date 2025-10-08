# Medical Chat API - Agentic AI Medical Consultation Endpoints

import logging
import time
from datetime import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse

from app.schemas.medical_chat import (
    MedicalChatRequest,
    MedicalChatResponse,
    EmergencyResponse,
    ErrorResponse,
    UrgencyLevel,
    ProcessingMetadata
)
from app.services.medical_ai_service import MedicalAIService
from app.util.rate_limiter import RateLimiter
from app.util.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()
rate_limiter = RateLimiter()


def get_medical_ai_service(request: Request) -> MedicalAIService:
    """Get medical AI service from app state"""
    return request.app.state.medical_ai_service


@router.post(
    "/",
    response_model=MedicalChatResponse,
    responses={
        200: {"model": MedicalChatResponse, "description": "Successful medical consultation"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="🩺 Medical Consultation Chat",
    description="""
    **Agentic AI Medical Consultation**

    This endpoint provides comprehensive medical consultation using an advanced multi-agent AI system.

    ### Features:
    - 🤖 **Multi-Agent Analysis**: Specialized agents for triage, diagnosis, and treatment
    - 🗣️ **Thai Language Support**: Full support for Thai language and regional dialects
    - 🚨 **Emergency Detection**: Automatic detection of emergency situations
    - 📊 **Risk Assessment**: Clinical risk scoring and urgency determination
    - 💊 **Treatment Planning**: Medication recommendations with safety checks
    - 🔍 **RAG-Enhanced**: Knowledge base integration for evidence-based responses

    ### Safety Features:
    - Emergency escalation protocols
    - Medical disclaimer enforcement
    - Regional dialect emergency detection
    - Professional oversight integration

    ### Input Processing:
    1. **Triage Agent**: Initial assessment and urgency determination
    2. **Diagnostic Agent**: Medical analysis with differential diagnosis
    3. **Treatment Agent**: Treatment recommendations and safety checks

    ⚠️ **Medical Disclaimer**: This system provides general health information only.
    Always consult qualified healthcare professionals for medical advice.
    """
)
async def medical_chat(
    request: MedicalChatRequest,
    ai_service: MedicalAIService = Depends(get_medical_ai_service),
    client_request: Request = None
) -> MedicalChatResponse:
    """
    Process medical consultation request through agentic AI system
    """
    start_time = time.time()
    client_ip = client_request.client.host if client_request else "unknown"

    # Rate limiting
    if not await rate_limiter.check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="มีการใช้งานเกินขีดจำกัด กรุณาลองใหม่อีกครั้งในภายหลัง"
        )

    logger.info(f"🩺 Medical chat request from {client_ip}: {request.message[:100]}...")

    try:
        # Process through agentic AI system
        result = await ai_service.process_medical_consultation(
            message=request.message,
            conversation_history=request.conversation_history,
            patient_info=request.patient_info,
            preferred_language=request.preferred_language,
            session_id=request.session_id,
            include_reasoning=request.include_reasoning
        )

        processing_time = int((time.time() - start_time) * 1000)

        # Handle emergency responses
        if result.get("type") == "emergency":
            emergency_response = EmergencyResponse(
                message=result.get("message", "Emergency situation detected"),
                urgency=UrgencyLevel.CRITICAL,
                recommendations=result.get("recommendations", []),
                warning=result.get("warning", "Seek immediate medical attention"),
                detected_keywords=result.get("emergency_keywords", []),
                agent_reasoning=result.get("agent_reasoning_chain")
            )

            logger.warning(f"🚨 Emergency detected for {client_ip}")
            return emergency_response

        # Build comprehensive response
        response = MedicalChatResponse(
            message=result.get("message", ""),
            type=result.get("type", "comprehensive_analysis"),
            triage=result.get("triage"),
            diagnosis=result.get("diagnosis"),
            treatment=result.get("treatment"),
            agent_reasoning_chain=result.get("agent_reasoning_chain") if request.include_reasoning else None,
            metadata=ProcessingMetadata(
                processing_time_ms=processing_time,
                translation_used=result.get("translation_used", False),
                detected_language=result.get("detected_language", "unknown"),
                detected_dialects=result.get("detected_dialects", []),
                agents_used=len(result.get("agent_reasoning_chain", [])),
                rag_results_count=result.get("rag_results_count", 0)
            ),
            session_id=request.session_id,
            disclaimer="ข้อมูลนี้เป็นเพียงข้อมูลทั่วไปเท่านั้น ไม่ใช่การวินิจฉัยทางการแพทย์ กรุณาปรึกษาแพทย์หรือผู้เชี่ยวชาญด้านสุขภาพสำหรับการวินิจฉัยและการรักษาที่เหมาะสม",
            recommendation=result.get("recommendation", "Please consult a healthcare professional for proper medical advice")
        )

        logger.info(f"✅ Medical chat completed for {client_ip} in {processing_time}ms")
        return response

    except ValueError as e:
        logger.error(f"❌ Validation error for {client_ip}: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.error(f"❌ Medical chat error for {client_ip}: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred during medical consultation. Please try again."
        )


@router.post(
    "/emergency-check",
    response_model=Dict[str, Any],
    summary="🚨 Emergency Keyword Detection",
    description="""
    **Emergency Detection System**

    Quickly check if a message contains emergency keywords across multiple languages and Thai dialects.

    ### Supported Detection:
    - Standard Thai emergency keywords
    - Northern Thai (ล้านนา) dialects
    - Isan (อีสาน) dialects
    - Southern Thai dialects
    - English emergency keywords

    ### Use Cases:
    - Pre-screening messages for urgency
    - Triage workflow automation
    - Emergency protocol activation
    """
)
async def emergency_check(
    message: str,
    ai_service: MedicalAIService = Depends(get_medical_ai_service)
) -> Dict[str, Any]:
    """
    Check for emergency keywords in a message
    """
    try:
        result = await ai_service.check_emergency_keywords(message)

        return {
            "is_emergency": result["is_emergency"],
            "detected_keywords": result["keywords"],
            "detected_dialects": result.get("dialects", []),
            "confidence": result.get("confidence", 0),
            "recommendation": result.get("recommendation", ""),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Emergency check error: {e}")
        raise HTTPException(status_code=500, detail="Emergency check failed")


@router.post(
    "/triage",
    response_model=Dict[str, Any],
    summary="🏥 Medical Triage Assessment",
    description="""
    **AI-Powered Medical Triage**

    Perform initial triage assessment to determine urgency and priority level.

    ### Assessment Factors:
    - Symptom severity analysis
    - Vital signs evaluation (if provided)
    - Emergency keyword detection
    - Risk scoring algorithm
    - Thai dialect processing

    ### Triage Levels:
    1. **Resuscitation** (Red) - Immediate
    2. **Emergency** (Orange) - Within 10 minutes
    3. **Urgent** (Yellow) - Within 30 minutes
    4. **Semi-urgent** (Green) - Within 60 minutes
    5. **Non-urgent** (Blue) - Within 120 minutes
    """
)
async def triage_assessment(
    request: MedicalChatRequest,
    ai_service: MedicalAIService = Depends(get_medical_ai_service)
) -> Dict[str, Any]:
    """
    Perform medical triage assessment
    """
    try:
        result = await ai_service.perform_triage_assessment(
            message=request.message,
            patient_info=request.patient_info,
        )

        return {
            "triage_level": result["triage_level"],
            "urgency": result["urgency"],
            "emergency_detected": result["emergency_detected"],
            "risk_score": result["risk_score"],
            "recommendations": result["recommendations"],
            "emergency_keywords": result.get("emergency_keywords", []),
            "reasoning": result.get("reasoning", ""),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Triage assessment error: {e}")
        raise HTTPException(status_code=500, detail="Triage assessment failed")


@router.get(
    "/conversation/{session_id}",
    response_model=List[Dict[str, Any]],
    summary="💬 Get Conversation History",
    description="""
    **Conversation History Retrieval**

    Retrieve the conversation history for a specific session.

    ### Features:
    - Session-based conversation tracking
    - Message timestamps
    - Agent reasoning preservation
    - Translation history
    """
)
async def get_conversation_history(
    session_id: str,
    limit: int = 50,
    ai_service: MedicalAIService = Depends(get_medical_ai_service)
) -> List[Dict[str, Any]]:
    """
    Get conversation history for a session
    """
    try:
        history = await ai_service.get_conversation_history(session_id, limit)
        return history

    except Exception as e:
        logger.error(f"❌ Error retrieving conversation history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation history")


@router.delete(
    "/conversation/{session_id}",
    response_model=Dict[str, str],
    summary="🗑️ Clear Conversation History",
    description="""
    **Clear Conversation History**

    Clear the conversation history for a specific session.

    ### Use Cases:
    - Privacy protection
    - Session reset
    - New patient consultation
    """
)
async def clear_conversation_history(
    session_id: str,
    ai_service: MedicalAIService = Depends(get_medical_ai_service)
) -> Dict[str, str]:
    """
    Clear conversation history for a session
    """
    try:
        await ai_service.clear_conversation_history(session_id)
        return {
            "status": "success",
            "message": f"Conversation history cleared for session {session_id}",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Error clearing conversation history: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear conversation history")


@router.get(
    "/stats",
    response_model=Dict[str, Any],
    summary="📊 Medical AI Statistics",
    description="""
    **System Statistics**

    Get system performance and usage statistics.

    ### Metrics:
    - Total consultations processed
    - Emergency cases detected
    - Average response time
    - Agent performance metrics
    - Language distribution
    """
)
async def get_system_stats(
    ai_service: MedicalAIService = Depends(get_medical_ai_service)
) -> Dict[str, Any]:
    """
    Get system statistics
    """
    try:
        stats = await ai_service.get_system_statistics()
        return stats

    except Exception as e:
        logger.error(f"❌ Error retrieving system stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system statistics")