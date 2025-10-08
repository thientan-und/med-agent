"""
Model Evaluation API Endpoints
==============================
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from app.services.model_evaluator import model_evaluator
from app.services.medical_ai_service import MedicalAIService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/evaluate", tags=["Model Evaluation"])


@router.post("/run_evaluation")
async def run_model_evaluation() -> Dict[str, Any]:
    """Run comprehensive model evaluation with zero-shot and few-shot learning"""
    try:
        # Initialize medical AI service
        medical_ai = MedicalAIService()

        # Run evaluation
        evaluation_results = await model_evaluator.run_zero_shot_evaluation(medical_ai)

        # Get improvement suggestions
        suggestions = model_evaluator.get_improvement_suggestions()

        return {
            "status": "success",
            "evaluation": evaluation_results,
            "improvement_suggestions": suggestions,
            "message": f"Evaluation complete with {evaluation_results['overall_metrics']['overall_accuracy']:.2%} accuracy"
        }
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/few_shot_examples")
async def get_few_shot_examples(n_examples: int = 5) -> Dict[str, Any]:
    """Get few-shot learning examples from evaluation"""
    try:
        examples = model_evaluator.get_few_shot_examples(n_examples)
        return {
            "status": "success",
            "examples": examples,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting few-shot examples: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluation_history")
async def get_evaluation_history(limit: int = 10) -> Dict[str, Any]:
    """Get evaluation history and trends"""
    try:
        history = model_evaluator.evaluation_history[-limit:]

        return {
            "status": "success",
            "history": [model_evaluator._result_to_dict(h) for h in history],
            "accuracy_trend": model_evaluator.accuracy_over_time,
            "total_evaluations": len(model_evaluator.evaluation_history)
        }
    except Exception as e:
        logger.error(f"Error getting evaluation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test_specific_case")
async def test_specific_case(
    symptoms: str,
    expected_diagnosis: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Test a specific case and learn from it"""
    try:
        medical_ai = MedicalAIService()

        # Get AI diagnosis
        ai_response = await medical_ai.process_common_illness_consultation(
            message=symptoms,
            conversation_history=[],
            patient_info=None,
            vital_signs=None,
            preferred_language="thai",
            session_id=f"test_{datetime.now().timestamp()}"
        )

        # Compare with expected if provided
        if expected_diagnosis:
            match = ai_response.get("diagnosis", {}).get("primary_diagnosis", {}).get("icd_code") == expected_diagnosis.get("icd_code")

            if not match:
                # Learn from the mistake
                await model_evaluator._learn_from_mistake(
                    test_case=None,  # Ad-hoc test case
                    evaluation=None,
                    ai_response=ai_response
                )

        return {
            "status": "success",
            "symptoms": symptoms,
            "ai_diagnosis": ai_response.get("diagnosis"),
            "ai_treatment": ai_response.get("treatment"),
            "expected_diagnosis": expected_diagnosis,
            "match": match if expected_diagnosis else None
        }
    except Exception as e:
        logger.error(f"Error testing specific case: {e}")
        raise HTTPException(status_code=500, detail=str(e))