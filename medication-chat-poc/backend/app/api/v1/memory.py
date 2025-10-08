"""
Memory and Learning API Endpoints
==================================
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from app.services.memory_agent import memory_agent
from app.schemas.medical_chat import PatientInfo

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/memory", tags=["Memory & Learning"])


@router.post("/store_interaction")
async def store_interaction(
    session_id: str,
    symptoms: str,
    diagnosis: Dict[str, Any],
    treatment: Dict[str, Any],
    patient_id: Optional[str] = None,
    doctor_feedback: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Store an interaction for learning"""
    try:
        await memory_agent.store_interaction(
            session_id=session_id,
            patient_id=patient_id,
            symptoms=symptoms,
            diagnosis=diagnosis,
            treatment=treatment,
            doctor_feedback=doctor_feedback
        )

        return {
            "status": "success",
            "message": "Interaction stored successfully",
            "memory_stats": memory_agent.get_memory_stats()
        }
    except Exception as e:
        logger.error(f"Error storing interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/learn_from_feedback")
async def learn_from_doctor_feedback(
    session_id: str,
    original_diagnosis: Dict[str, Any],
    doctor_feedback: Dict[str, Any]
) -> Dict[str, Any]:
    """Learn from doctor feedback on a diagnosis"""
    try:
        await memory_agent.learn_from_doctor_feedback(
            session_id=session_id,
            original_diagnosis=original_diagnosis,
            doctor_feedback=doctor_feedback
        )

        return {
            "status": "success",
            "message": "Learned from doctor feedback",
            "corrections_count": len(memory_agent.doctor_corrections)
        }
    except Exception as e:
        logger.error(f"Error learning from feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patient_history/{patient_id}")
async def get_patient_history(patient_id: str) -> Dict[str, Any]:
    """Get patient medical history and context"""
    try:
        if patient_id in memory_agent.patient_memories:
            patient_memory = memory_agent.patient_memories[patient_id]
            return {
                "patient_id": patient_id,
                "medical_history": patient_memory.medical_history[-10:],  # Last 10 visits
                "risk_factors": patient_memory.risk_factors,
                "allergies": patient_memory.allergies,
                "successful_treatments": patient_memory.successful_treatments[-5:],
                "total_interactions": patient_memory.total_interactions,
                "last_visit": patient_memory.last_visit.isoformat() if patient_memory.last_visit else None
            }
        else:
            return {
                "patient_id": patient_id,
                "message": "No history found for this patient"
            }
    except Exception as e:
        logger.error(f"Error getting patient history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory_stats")
async def get_memory_statistics() -> Dict[str, Any]:
    """Get memory system statistics"""
    try:
        stats = memory_agent.get_memory_stats()
        return {
            "status": "success",
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/consolidate_memory")
async def consolidate_memory() -> Dict[str, Any]:
    """Trigger memory consolidation and optimization"""
    try:
        await memory_agent.consolidate_memory()
        return {
            "status": "success",
            "message": "Memory consolidation completed",
            "memory_stats": memory_agent.get_memory_stats()
        }
    except Exception as e:
        logger.error(f"Error consolidating memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/similar_cases")
async def find_similar_cases(
    symptoms: str = Query(..., description="Symptoms to match"),
    limit: int = Query(5, description="Maximum number of cases to return")
) -> Dict[str, Any]:
    """Find similar cases from memory"""
    try:
        similar_cases = await memory_agent._find_similar_cases(symptoms, limit)
        return {
            "status": "success",
            "query_symptoms": symptoms,
            "similar_cases": similar_cases,
            "count": len(similar_cases)
        }
    except Exception as e:
        logger.error(f"Error finding similar cases: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/learning_patterns")
async def get_learning_patterns() -> Dict[str, Any]:
    """Get learned diagnosis patterns"""
    try:
        # Get top patterns for each symptom
        top_patterns = {}
        for symptom, diagnoses in memory_agent.diagnosis_patterns.items():
            if diagnoses:
                # Sort by pattern strength
                sorted_diagnoses = sorted(diagnoses.items(), key=lambda x: x[1], reverse=True)
                top_patterns[symptom] = sorted_diagnoses[:3]  # Top 3 for each symptom

        return {
            "status": "success",
            "learned_patterns": top_patterns,
            "total_patterns": len(memory_agent.diagnosis_patterns),
            "doctor_corrections": memory_agent.doctor_corrections[-10:]  # Last 10 corrections
        }
    except Exception as e:
        logger.error(f"Error getting learning patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))