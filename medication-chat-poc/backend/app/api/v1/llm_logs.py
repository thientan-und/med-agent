"""
LLM Logs API Endpoints
=====================
API endpoints for viewing LLM model interactions and responses
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import logging

from app.services.medical_ai_service import MedicalAIService
from app.services.llm_logger import llm_logger

router = APIRouter(prefix="/llm", tags=["LLM Logs"])
logger = logging.getLogger(__name__)

@router.get("/interactions", response_model=List[Dict[str, Any]])
async def get_llm_interactions(
    limit: int = Query(10, ge=1, le=100, description="Number of interactions to retrieve"),
    model_name: Optional[str] = Query(None, description="Filter by model name")
):
    """Get recent LLM interactions with detailed logs"""
    try:
        interactions = llm_logger.get_recent_interactions(limit=limit)

        # Filter by model name if specified
        if model_name:
            filtered_interactions = []
            for interaction in interactions:
                if interaction.get('type') == 'complete_interaction':
                    request_data = interaction.get('data', {}).get('request', {})
                    if request_data.get('model_name') == model_name:
                        filtered_interactions.append(interaction)
                elif interaction.get('type') == 'request':
                    request_data = interaction.get('data', {})
                    if request_data.get('model_name') == model_name:
                        filtered_interactions.append(interaction)
            interactions = filtered_interactions

        return interactions
    except Exception as e:
        logger.error(f"Failed to get LLM interactions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics", response_model=Dict[str, Any])
async def get_llm_statistics():
    """Get LLM interaction statistics"""
    try:
        stats = llm_logger.get_statistics()
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        logger.error(f"Failed to get LLM statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/responses/{request_id}")
async def get_llm_response(request_id: str):
    """Get specific LLM response by request ID"""
    try:
        interactions = llm_logger.get_recent_interactions(limit=1000)

        for interaction in interactions:
            if interaction.get('type') == 'complete_interaction':
                request_data = interaction.get('data', {}).get('request', {})
                if request_data.get('request_id') == request_id:
                    return {
                        "success": True,
                        "data": interaction
                    }
            elif interaction.get('type') == 'response':
                response_data = interaction.get('data', {})
                if response_data.get('request_id') == request_id:
                    return {
                        "success": True,
                        "data": interaction
                    }

        raise HTTPException(status_code=404, detail="LLM interaction not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get LLM response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models", response_model=List[str])
async def get_available_models():
    """Get list of available LLM models"""
    try:
        stats = llm_logger.get_statistics()
        model_usage = stats.get('model_usage', {})
        return list(model_usage.keys())
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test-interaction")
async def test_llm_interaction(
    request_data: dict
):
    """Test LLM interaction with logging"""
    try:
        symptoms = request_data.get("symptoms", "")
        model_name = request_data.get("model_name", "medllama2")

        if not symptoms:
            raise HTTPException(status_code=400, detail="Symptoms required")

        # Initialize medical AI service for testing
        medical_ai = MedicalAIService()
        await medical_ai.initialize()

        # Use the medical AI service to process symptoms
        result = await medical_ai.process_medical_consultation(
            message=symptoms,
            session_id=f"test_{model_name}"
        )

        await medical_ai.cleanup()

        # Get the most recent LLM logs
        recent_logs = llm_logger.get_recent_interactions(limit=1)

        return {
            "success": True,
            "consultation_result": {
                "type": result.get("type"),
                "message_length": len(result.get("message", "")),
                "diagnosis": result.get("diagnosis"),
                "treatment": result.get("treatment")
            },
            "llm_logs": recent_logs
        }
    except Exception as e:
        logger.error(f"Failed to test LLM interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/medical-responses", response_model=List[Dict[str, Any]])
async def get_medical_responses(
    limit: int = Query(10, ge=1, le=50, description="Number of medical responses to retrieve")
):
    """Get recent medical diagnosis responses from LLM"""
    try:
        interactions = llm_logger.get_recent_interactions(limit=limit * 2)  # Get more to filter

        medical_responses = []
        for interaction in interactions:
            # Check both complete_interaction and response types
            if interaction.get('type') in ['complete_interaction', 'response']:
                if interaction.get('type') == 'complete_interaction':
                    interaction_data = interaction.get('data', {})
                    request = interaction_data.get('request', {})
                    response = interaction_data.get('response', {})

                    # Check if it's a medical consultation
                    context = request.get('context', {})
                    if context.get('consultation_type') == 'common_illness':
                        medical_responses.append({
                            "request_id": request.get('request_id'),
                            "timestamp": request.get('timestamp'),
                            "model_name": request.get('model_name'),
                            "symptoms": request.get('prompt', '').replace('Medical consultation for symptoms: ', ''),
                            "diagnosis_response": response.get('response_text', ''),
                            "response_time_ms": response.get('response_time_ms'),
                            "tokens_used": response.get('tokens_used'),
                            "confidence_score": response.get('confidence_score'),
                            "session_id": context.get('session_id'),
                            "language": context.get('language')
                        })
                elif interaction.get('type') == 'response':
                    # Handle response-only logs (which seem to be the actual format)
                    response_data = interaction.get('data', {})

                    # Check if this is a medical response by looking at response content
                    response_text = response_data.get('response_text', '')
                    if any(keyword in response_text for keyword in ['อาการ', 'diagnosis', 'ปรึกษาแพทย์', 'medical', 'ICD']):
                        medical_responses.append({
                            "request_id": response_data.get('request_id'),
                            "timestamp": response_data.get('response_timestamp'),
                            "model_name": "medllama2",  # Default model for medical responses
                            "symptoms": "Medical symptoms query",  # We don't have original prompt in response-only logs
                            "diagnosis_response": response_text,
                            "response_time_ms": response_data.get('response_time_ms'),
                            "tokens_used": response_data.get('tokens_used'),
                            "confidence_score": response_data.get('confidence_score'),
                            "session_id": "unknown",
                            "language": "thai" if any(ord(c) >= 0x0E00 and ord(c) <= 0x0E7F for c in response_text) else "english"
                        })

                if len(medical_responses) >= limit:
                    break

        return medical_responses
    except Exception as e:
        logger.error(f"Failed to get medical responses: {e}")
        raise HTTPException(status_code=500, detail=str(e))