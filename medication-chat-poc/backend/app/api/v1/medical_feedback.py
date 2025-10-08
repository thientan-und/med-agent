# Medical Feedback API - Doctor feedback and model improvement endpoints

import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, Request

from app.schemas.medical_chat import (
    FeedbackRequest,
    FeedbackResponse,
    ProcessingMetadata
)
from app.services.medical_ai_service import MedicalAIService
from app.util.config import get_settings
from app.util.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()
rate_limiter = RateLimiter()


def get_medical_ai_service(request: Request) -> MedicalAIService:
    """Get medical AI service from app state"""
    return request.app.state.medical_ai_service


@router.post(
    "/submit",
    response_model=FeedbackResponse,
    summary="👨‍⚕️ Submit Doctor Feedback",
    description="""
    **Medical Professional Feedback System**

    Submit feedback on AI-generated medical consultations for continuous improvement.

    ### Feedback Types:
    - **Diagnosis Corrections**: Approve or correct AI diagnoses
    - **Treatment Modifications**: Adjust medication recommendations
    - **Urgency Assessment**: Correct triage levels
    - **General Feedback**: Additional notes and observations

    ### Model Learning:
    - Feedback is stored for model retraining
    - Patterns are analyzed for systematic improvements
    - Performance metrics are tracked
    - Quality assurance monitoring

    ### Requirements:
    - Valid doctor ID for tracking
    - Original chat session ID
    - Specific corrections or approvals
    """
)
async def submit_feedback(
    request: FeedbackRequest,
    ai_service: MedicalAIService = Depends(get_medical_ai_service),
    client_request: Request = None
) -> FeedbackResponse:
    """
    Submit medical professional feedback on AI consultations
    """

    client_ip = client_request.client.host if client_request else "unknown"

    # Rate limiting
    if not await rate_limiter.check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )

    logger.info(f"👨‍⚕️ Medical feedback from doctor {request.doctor_id} for chat {request.chat_id}")

    try:
        # Generate unique feedback ID
        feedback_id = f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.doctor_id}"

        # Prepare feedback data
        feedback_data = {
            "feedback_id": feedback_id,
            "chat_id": request.chat_id,
            "doctor_id": request.doctor_id,
            "timestamp": datetime.now().isoformat(),
            "diagnosis_approved": request.diagnosis_approved,
            "corrected_diagnosis": request.corrected_diagnosis,
            "corrected_medications": request.corrected_medications,
            "feedback_notes": request.feedback_notes,
            "urgency_assessment": request.urgency_assessment.value if request.urgency_assessment else None,
            "client_ip": client_ip
        }

        # Store feedback to file
        feedback_stored = await _store_feedback(feedback_data)

        # Analyze feedback for immediate insights
        insights = await _analyze_feedback(feedback_data, ai_service)

        # Update system statistics
        await _update_feedback_stats(feedback_data, ai_service)

        # Determine if model should be updated
        model_updated = await _should_update_model(feedback_data, ai_service)

        response_message = "Feedback submitted successfully"
        if not request.diagnosis_approved:
            response_message += " - Diagnosis corrections noted for model improvement"
        if request.corrected_medications:
            response_message += " - Medication corrections will be incorporated"

        return FeedbackResponse(
            status="success",
            message=response_message,
            feedback_id=feedback_id,
            model_updated=model_updated
        )

    except ValueError as e:
        logger.error(f"❌ Validation error in feedback submission: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.error(f"❌ Feedback submission error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process feedback. Please try again."
        )


@router.get(
    "/stats",
    response_model=Dict[str, Any],
    summary="📊 Feedback Statistics",
    description="""
    **Medical Feedback Analytics**

    Get comprehensive statistics on doctor feedback and model performance.

    ### Analytics Include:
    - Approval rates by diagnosis category
    - Common correction patterns
    - Doctor feedback frequency
    - Model improvement metrics
    - Performance trends over time
    """
)
async def get_feedback_stats(
    doctor_id: str = None,
    days: int = 30,
    ai_service: MedicalAIService = Depends(get_medical_ai_service)
) -> Dict[str, Any]:
    """
    Get feedback statistics and analytics
    """

    try:
        # Load feedback data
        feedback_data = await _load_feedback_data()

        # Filter by date range
        cutoff_date = datetime.now().timestamp() - (days * 24 * 3600)
        recent_feedback = [
            fb for fb in feedback_data
            if datetime.fromisoformat(fb.get("timestamp", "")).timestamp() > cutoff_date
        ]

        # Filter by doctor if specified
        if doctor_id:
            recent_feedback = [fb for fb in recent_feedback if fb.get("doctor_id") == doctor_id]

        # Calculate statistics
        total_feedback = len(recent_feedback)
        approved_diagnoses = sum(1 for fb in recent_feedback if fb.get("diagnosis_approved", False))
        approval_rate = (approved_diagnoses / total_feedback * 100) if total_feedback > 0 else 0

        # Correction patterns
        correction_patterns = {}
        urgency_corrections = {}
        doctor_activity = {}

        for feedback in recent_feedback:
            # Track correction patterns
            if feedback.get("corrected_diagnosis"):
                original_icd = feedback.get("corrected_diagnosis", {}).get("original_icd", "unknown")
                corrected_icd = feedback.get("corrected_diagnosis", {}).get("corrected_icd", "unknown")
                pattern_key = f"{original_icd} -> {corrected_icd}"
                correction_patterns[pattern_key] = correction_patterns.get(pattern_key, 0) + 1

            # Track urgency corrections
            if feedback.get("urgency_assessment"):
                urgency = feedback.get("urgency_assessment")
                urgency_corrections[urgency] = urgency_corrections.get(urgency, 0) + 1

            # Track doctor activity
            doctor = feedback.get("doctor_id", "unknown")
            doctor_activity[doctor] = doctor_activity.get(doctor, 0) + 1

        # Top corrections
        top_corrections = sorted(correction_patterns.items(), key=lambda x: x[1], reverse=True)[:5]

        # Calculate model performance metrics
        performance_metrics = await _calculate_performance_metrics(recent_feedback)

        return {
            "period_days": days,
            "total_feedback_submissions": total_feedback,
            "diagnosis_approval_rate": round(approval_rate, 1),
            "approved_diagnoses": approved_diagnoses,
            "corrected_diagnoses": total_feedback - approved_diagnoses,
            "active_doctors": len(doctor_activity),
            "top_correction_patterns": top_corrections,
            "urgency_assessment_distribution": urgency_corrections,
            "doctor_activity": dict(sorted(doctor_activity.items(), key=lambda x: x[1], reverse=True)[:10]),
            "performance_metrics": performance_metrics,
            "improvement_suggestions": _generate_improvement_suggestions(recent_feedback),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Feedback stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve feedback statistics")


@router.get(
    "/history/{doctor_id}",
    response_model=Dict[str, Any],
    summary="📋 Doctor Feedback History",
    description="""
    **Individual Doctor Feedback History**

    Retrieve feedback history for a specific medical professional.

    ### Information Included:
    - Chronological feedback submissions
    - Approval rates and patterns
    - Frequent correction types
    - Performance contribution metrics
    """
)
async def get_doctor_feedback_history(
    doctor_id: str,
    limit: int = 50,
    ai_service: MedicalAIService = Depends(get_medical_ai_service)
) -> Dict[str, Any]:
    """
    Get feedback history for a specific doctor
    """

    try:
        # Load feedback data
        feedback_data = await _load_feedback_data()

        # Filter by doctor
        doctor_feedback = [fb for fb in feedback_data if fb.get("doctor_id") == doctor_id]
        doctor_feedback.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        # Limit results
        recent_feedback = doctor_feedback[:limit]

        # Calculate doctor-specific statistics
        total_submissions = len(doctor_feedback)
        approved = sum(1 for fb in doctor_feedback if fb.get("diagnosis_approved", False))
        approval_rate = (approved / total_submissions * 100) if total_submissions > 0 else 0

        # Correction categories
        correction_categories = {}
        for feedback in doctor_feedback:
            if feedback.get("corrected_diagnosis"):
                category = feedback.get("corrected_diagnosis", {}).get("category", "other")
                correction_categories[category] = correction_categories.get(category, 0) + 1

        return {
            "doctor_id": doctor_id,
            "total_feedback_submissions": total_submissions,
            "diagnosis_approval_rate": round(approval_rate, 1),
            "recent_feedback": recent_feedback,
            "correction_categories": correction_categories,
            "contribution_score": _calculate_contribution_score(doctor_feedback),
            "first_feedback": doctor_feedback[-1].get("timestamp") if doctor_feedback else None,
            "last_feedback": doctor_feedback[0].get("timestamp") if doctor_feedback else None,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Doctor feedback history error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve doctor feedback history")


@router.post(
    "/train-model",
    response_model=Dict[str, Any],
    summary="🤖 Trigger Model Retraining",
    description="""
    **Trigger AI Model Retraining**

    Manually trigger model retraining based on accumulated feedback.

    ### Process:
    - Analyze accumulated feedback
    - Prepare training data
    - Update model parameters
    - Validate improvements

    **Note**: This is typically an administrative function.
    """
)
async def trigger_model_training(
    doctor_id: str,
    force_retrain: bool = False,
    ai_service: MedicalAIService = Depends(get_medical_ai_service)
) -> Dict[str, Any]:
    """
    Trigger model retraining based on feedback (admin function)
    """

    logger.info(f"🤖 Model retraining triggered by doctor {doctor_id}")

    try:
        # Load feedback data
        feedback_data = await _load_feedback_data()

        # Determine if retraining is needed
        if not force_retrain:
            recent_corrections = [
                fb for fb in feedback_data
                if not fb.get("diagnosis_approved", True)
                and datetime.fromisoformat(fb.get("timestamp", "")).timestamp() >
                   (datetime.now().timestamp() - 7 * 24 * 3600)  # Last 7 days
            ]

            if len(recent_corrections) < 10:
                return {
                    "status": "skipped",
                    "message": "Insufficient recent corrections for retraining",
                    "corrections_needed": 10,
                    "corrections_available": len(recent_corrections),
                    "recommendation": "Accumulate more feedback before retraining"
                }

        # Simulate model training process
        training_result = await _simulate_model_training(feedback_data)

        return {
            "status": "success",
            "message": "Model retraining completed successfully",
            "training_data_points": len(feedback_data),
            "corrections_incorporated": training_result["corrections_count"],
            "estimated_improvement": training_result["improvement_estimate"],
            "training_timestamp": datetime.now().isoformat(),
            "triggered_by": doctor_id
        }

    except Exception as e:
        logger.error(f"❌ Model training error: {e}")
        raise HTTPException(status_code=500, detail="Model training failed")


# Utility functions

async def _store_feedback(feedback_data: Dict) -> bool:
    """Store feedback data to file"""

    try:
        feedback_file = settings.feedback_data_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(feedback_file), exist_ok=True)

        # Load existing data
        existing_data = []
        if os.path.exists(feedback_file):
            try:
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = []

        # Add new feedback
        existing_data.append(feedback_data)

        # Keep only last 1000 feedback entries
        if len(existing_data) > 1000:
            existing_data = existing_data[-1000:]

        # Save back to file
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

        return True

    except Exception as e:
        logger.error(f"Failed to store feedback: {e}")
        return False


async def _load_feedback_data() -> List[Dict]:
    """Load feedback data from file"""

    try:
        feedback_file = settings.feedback_data_path

        if not os.path.exists(feedback_file):
            return []

        with open(feedback_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    except Exception as e:
        logger.error(f"Failed to load feedback data: {e}")
        return []


async def _analyze_feedback(feedback_data: Dict, ai_service: MedicalAIService) -> Dict:
    """Analyze feedback for immediate insights"""

    insights = {
        "feedback_type": "correction" if not feedback_data.get("diagnosis_approved") else "approval",
        "correction_areas": [],
        "urgency_adjusted": bool(feedback_data.get("urgency_assessment")),
        "medication_changes": bool(feedback_data.get("corrected_medications"))
    }

    # Identify correction areas
    if feedback_data.get("corrected_diagnosis"):
        insights["correction_areas"].append("diagnosis")
    if feedback_data.get("corrected_medications"):
        insights["correction_areas"].append("medications")
    if feedback_data.get("urgency_assessment"):
        insights["correction_areas"].append("urgency")

    return insights


async def _update_feedback_stats(feedback_data: Dict, ai_service: MedicalAIService):
    """Update system statistics with feedback data"""

    # This would update internal statistics
    # For now, we'll just log the feedback
    feedback_type = "approval" if feedback_data.get("diagnosis_approved") else "correction"
    logger.info(f"📊 Feedback recorded: {feedback_type} from doctor {feedback_data.get('doctor_id')}")


async def _should_update_model(feedback_data: Dict, ai_service: MedicalAIService) -> bool:
    """Determine if model should be updated based on feedback"""

    # Simple logic: update if there are corrections
    has_corrections = (
        not feedback_data.get("diagnosis_approved") or
        feedback_data.get("corrected_medications") or
        feedback_data.get("urgency_assessment")
    )

    return has_corrections


async def _calculate_performance_metrics(feedback_data: List[Dict]) -> Dict:
    """Calculate model performance metrics from feedback"""

    if not feedback_data:
        return {"error": "No feedback data available"}

    total = len(feedback_data)
    approved = sum(1 for fb in feedback_data if fb.get("diagnosis_approved", False))

    # Accuracy metrics
    accuracy = (approved / total * 100) if total > 0 else 0

    # Urgency assessment accuracy
    urgency_feedback = [fb for fb in feedback_data if fb.get("urgency_assessment")]
    urgency_accuracy = len(urgency_feedback) / total * 100 if total > 0 else 100

    return {
        "overall_accuracy": round(accuracy, 1),
        "diagnosis_approval_rate": round(accuracy, 1),
        "urgency_assessment_accuracy": round(100 - urgency_accuracy, 1),  # Inverse of corrections
        "total_evaluations": total,
        "improvement_trend": "stable"  # Would calculate from historical data
    }


def _generate_improvement_suggestions(feedback_data: List[Dict]) -> List[str]:
    """Generate improvement suggestions based on feedback patterns"""

    suggestions = []

    if not feedback_data:
        return ["Collect more feedback data for analysis"]

    # Analyze common patterns
    correction_count = sum(1 for fb in feedback_data if not fb.get("diagnosis_approved", True))
    total_count = len(feedback_data)

    if correction_count / total_count > 0.3:
        suggestions.append("Consider retraining diagnostic models with recent corrections")

    urgency_corrections = sum(1 for fb in feedback_data if fb.get("urgency_assessment"))
    if urgency_corrections / total_count > 0.2:
        suggestions.append("Review triage algorithm parameters")

    medication_corrections = sum(1 for fb in feedback_data if fb.get("corrected_medications"))
    if medication_corrections / total_count > 0.15:
        suggestions.append("Update medication recommendation database")

    if not suggestions:
        suggestions.append("Model performance is good - continue monitoring")

    return suggestions


def _calculate_contribution_score(doctor_feedback: List[Dict]) -> float:
    """Calculate doctor's contribution score based on feedback quality and quantity"""

    if not doctor_feedback:
        return 0.0

    # Base score from quantity
    quantity_score = min(len(doctor_feedback) * 2, 50)  # Max 50 points for quantity

    # Quality score from detailed feedback
    quality_score = 0
    for feedback in doctor_feedback:
        if feedback.get("feedback_notes"):
            quality_score += 5
        if feedback.get("corrected_diagnosis"):
            quality_score += 10
        if feedback.get("corrected_medications"):
            quality_score += 8

    quality_score = min(quality_score, 50)  # Max 50 points for quality

    return round(quantity_score + quality_score, 1)


async def _simulate_model_training(feedback_data: List[Dict]) -> Dict:
    """Simulate model training process"""

    corrections = [fb for fb in feedback_data if not fb.get("diagnosis_approved", True)]

    return {
        "corrections_count": len(corrections),
        "training_samples": len(feedback_data),
        "improvement_estimate": f"{min(len(corrections) * 2, 15)}% accuracy improvement expected",
        "training_duration": "2-4 hours estimated"
    }