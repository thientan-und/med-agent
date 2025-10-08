# Medical Diagnosis API - Specialized diagnosis endpoints

import logging
from datetime import datetime
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, Request

from app.schemas.medical_chat import (
    DiagnosisRequest,
    DiagnosisResponse,
    PatientInfo,
    ProcessingMetadata
)
from app.services.medical_ai_service import MedicalAIService
from app.util.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)
router = APIRouter()
rate_limiter = RateLimiter()


def get_medical_ai_service(request: Request) -> MedicalAIService:
    """Get medical AI service from app state"""
    return request.app.state.medical_ai_service


@router.post(
    "/analyze",
    response_model=DiagnosisResponse,
    summary="🔬 Medical Diagnosis Analysis",
    description="""
    **AI-Powered Medical Diagnosis**

    Perform comprehensive medical diagnosis analysis using agentic AI system.

    ### Analysis Process:
    1. **Symptom Processing**: Parse and normalize symptoms
    2. **Knowledge Search**: RAG-enhanced search through medical knowledge base
    3. **Differential Diagnosis**: Generate multiple possible diagnoses
    4. **Risk Assessment**: Calculate clinical risk scores
    5. **Confidence Scoring**: Provide confidence levels for each diagnosis

    ### Input Requirements:
    - **Symptoms**: List of symptoms (required)
    - **Patient Info**: Demographics and medical history (optional)
    - **Vital Signs**: Current vital signs (optional)
    - **Medical History**: Relevant medical history (optional)

    ### AI Features:
    - Multi-agent diagnostic reasoning
    - ICD-10 code mapping
    - Thai language support
    - Evidence-based recommendations
    """
)
async def analyze_diagnosis(
    request: DiagnosisRequest,
    ai_service: MedicalAIService = Depends(get_medical_ai_service),
    client_request: Request = None
) -> DiagnosisResponse:
    """
    Analyze symptoms and provide medical diagnosis recommendations
    """

    start_time = datetime.now()
    client_ip = client_request.client.host if client_request else "unknown"

    # Rate limiting
    if not await rate_limiter.check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )

    logger.info(f"🔬 Diagnosis analysis request from {client_ip}: {request.symptoms}")

    try:
        # Prepare input for diagnostic agent
        symptoms_text = " ".join(request.symptoms)

        case_data = {
            "message": symptoms_text,
            "patient_info": request.patient_info,
            "medical_history": request.medical_history,
            "duration": request.duration,
            "severity": request.severity
        }

        # Run diagnostic analysis through agentic AI
        diagnostic_agent = ai_service.agents["diagnostic"]
        diagnosis_result = await diagnostic_agent.analyze_symptoms(case_data)

        # Run triage assessment for risk scoring
        triage_agent = ai_service.agents["triage"]
        triage_result = await triage_agent.assess_urgency(case_data)

        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

        # Build response
        response = DiagnosisResponse(
            primary_diagnosis=diagnosis_result.get("primary_diagnosis"),
            differential_diagnoses=diagnosis_result.get("differential_diagnoses", []),
            confidence=diagnosis_result.get("confidence", 0.0),
            risk_score=triage_result.get("risk_score", 0),
            urgency=triage_result.get("urgency", "low"),
            recommendations=_generate_diagnosis_recommendations(
                diagnosis_result, triage_result, request.symptoms
            ),
            reasoning=diagnosis_result.get("reasoning", ""),
            metadata=ProcessingMetadata(
                processing_time_ms=processing_time,
                translation_used=False,
                detected_language="unknown",
                agents_used=2,
                rag_results_count=len(diagnosis_result.get("differential_diagnoses", []))
            )
        )

        logger.info(f"✅ Diagnosis analysis completed for {client_ip} in {processing_time}ms")
        return response

    except ValueError as e:
        logger.error(f"❌ Validation error for {client_ip}: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.error(f"❌ Diagnosis analysis error for {client_ip}: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred during diagnosis analysis. Please try again."
        )


@router.post(
    "/symptom-checker",
    response_model=Dict[str, Any],
    summary="🩺 Symptom Checker",
    description="""
    **Quick Symptom Assessment**

    Fast symptom checking with preliminary assessment and recommendations.

    ### Features:
    - Rapid symptom analysis
    - Urgency assessment
    - Basic recommendations
    - Emergency detection

    ### Use Cases:
    - Initial patient screening
    - Self-assessment tools
    - Triage support
    """
)
async def symptom_checker(
    symptoms: List[str],
    age: int = None,
    gender: str = None,
    ai_service: MedicalAIService = Depends(get_medical_ai_service)
) -> Dict[str, Any]:
    """
    Quick symptom checking and assessment
    """

    try:
        # Prepare basic case data
        symptoms_text = " ".join(symptoms)
        patient_info = PatientInfo(age=age, gender=gender) if age or gender else None

        case_data = {
            "message": symptoms_text,
            "patient_info": patient_info
        }

        # Quick triage assessment
        triage_result = await ai_service.perform_triage_assessment(
            message=symptoms_text,
            patient_info=patient_info
        )

        # Emergency check
        emergency_result = await ai_service.check_emergency_keywords(symptoms_text)

        # Generate quick recommendations
        recommendations = []
        if emergency_result["is_emergency"]:
            recommendations = [
                "โทร 1669 เพื่อขอความช่วยเหลือฉุกเฉิน",
                "ไปโรงพยาบาลทันที",
                "อย่าขับรถไปเอง ให้คนอื่นพาไป"
            ]
        elif triage_result["urgency"] in ["high", "critical"]:
            recommendations = [
                "ควรไปโรงพยาบาลโดยเร็ว",
                "สังเกตอาการอย่างใกล้ชิด",
                "โทรแจ้งโรงพยาบาลก่อนไป"
            ]
        else:
            recommendations = [
                "สามารถดูแลตัวเองที่บ้านได้",
                "สังเกตอาการต่อไป",
                "หากอาการแย่ลง ให้ปรึกษาแพทย์"
            ]

        return {
            "symptoms_analyzed": symptoms,
            "urgency_level": triage_result["urgency"],
            "risk_score": triage_result["risk_score"],
            "emergency_detected": emergency_result["is_emergency"],
            "emergency_keywords": emergency_result["keywords"],
            "recommendations": recommendations,
            "triage_level": triage_result["triage_level"],
            "next_steps": _get_next_steps(triage_result["urgency"]),
            "disclaimer": "ข้อมูลนี้เป็นเพียงการประเมินเบื้องต้น กรุณาปรึกษาแพทย์เพื่อการวินิจฉัยที่แม่นยำ",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Symptom checker error: {e}")
        raise HTTPException(status_code=500, detail="Symptom checking failed")


@router.get(
    "/conditions/search",
    response_model=Dict[str, Any],
    summary="🔍 Search Medical Conditions",
    description="""
    **Medical Conditions Search**

    Search through the medical knowledge base for conditions and diagnoses.

    ### Search Capabilities:
    - Keyword search
    - ICD-10 code lookup
    - Category filtering
    - Thai/English search
    """
)
async def search_conditions(
    query: str,
    limit: int = 10,
    category: str = None,
    ai_service: MedicalAIService = Depends(get_medical_ai_service)
) -> Dict[str, Any]:
    """
    Search medical conditions database
    """

    try:
        if not ai_service.diagnoses:
            return {
                "results": [],
                "count": 0,
                "message": "Medical knowledge base not available"
            }

        # Filter and search diagnoses
        matches = []
        query_lower = query.lower()

        for diagnosis in ai_service.diagnoses:
            score = 0

            # Check name matches
            if query_lower in diagnosis.name_en.lower():
                score += 50
            if query in diagnosis.name_th:
                score += 50

            # Check ICD code
            if diagnosis.icd_code and query.upper() in diagnosis.icd_code:
                score += 100

            # Check category filter
            if category and diagnosis.category.lower() != category.lower():
                continue

            # Check description
            if diagnosis.description:
                if query_lower in diagnosis.description.lower():
                    score += 25

            if score > 0:
                matches.append({
                    "icd_code": diagnosis.icd_code,
                    "english_name": diagnosis.name_en,
                    "thai_name": diagnosis.name_th,
                    "category": diagnosis.category,
                    "description": diagnosis.description,
                    "relevance_score": score
                })

        # Sort by relevance and limit results
        matches.sort(key=lambda x: x["relevance_score"], reverse=True)
        results = matches[:limit]

        return {
            "query": query,
            "results": results,
            "count": len(results),
            "total_available": len(matches),
            "categories": list(set(d.category for d in ai_service.diagnoses if d.category)),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Condition search error: {e}")
        raise HTTPException(status_code=500, detail="Condition search failed")


@router.get(
    "/conditions/{icd_code}",
    response_model=Dict[str, Any],
    summary="📋 Get Condition Details",
    description="""
    **Get Medical Condition Details**

    Retrieve detailed information about a specific medical condition by ICD-10 code.
    """
)
async def get_condition_details(
    icd_code: str,
    ai_service: MedicalAIService = Depends(get_medical_ai_service)
) -> Dict[str, Any]:
    """
    Get detailed information about a specific medical condition
    """

    try:
        # Find condition by ICD code
        condition = None
        for diagnosis in ai_service.diagnoses:
            if diagnosis.icd_code and diagnosis.icd_code.upper() == icd_code.upper():
                condition = diagnosis
                break

        if not condition:
            raise HTTPException(status_code=404, detail=f"Condition with ICD code {icd_code} not found")

        # Find related treatments
        related_treatments = []
        condition_name_lower = condition.name_en.lower()
        for treatment in ai_service.treatments:
            if condition_name_lower in treatment.description.lower() if treatment.description else False:
                related_treatments.append({
                    "english_name": treatment.name_en,
                    "thai_name": treatment.name_th,
                    "category": treatment.category,
                    "description": treatment.description
                })

        return {
            "icd_code": condition.icd_code,
            "english_name": condition.name_en,
            "thai_name": condition.name_th,
            "category": condition.category,
            "description": condition.description,
            "related_treatments": related_treatments[:5],  # Limit to 5
            "icd_category": _get_icd_category(condition.icd_code),
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Get condition details error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve condition details")


# Utility functions

def _generate_diagnosis_recommendations(
    diagnosis_result: Dict,
    triage_result: Dict,
    symptoms: List[str]
) -> List[str]:
    """Generate recommendations based on diagnosis and triage results"""

    recommendations = []

    # Based on urgency
    urgency = triage_result.get("urgency", "low")
    if urgency in ["critical", "high"]:
        recommendations.append("ไปโรงพยาบาลโดยเร็วเพื่อรับการตรวจและรักษา")
        recommendations.append("นำข้อมูลการตรวจและยาที่ใช้ไปด้วย")
    elif urgency == "medium":
        recommendations.append("ควรนัดพบแพทย์ภายใน 24-48 ชั่วโมง")
        recommendations.append("สังเกตอาการและบันทึกการเปลี่ยนแปลง")
    else:
        recommendations.append("สามารถดูแลตัวเองที่บ้านได้")
        recommendations.append("หากอาการไม่ดีขึ้นภายใน 3-5 วัน ให้ปรึกษาแพทย์")

    # Based on primary diagnosis
    primary_diagnosis = diagnosis_result.get("primary_diagnosis")
    if primary_diagnosis:
        category = primary_diagnosis.get("category", "").lower()
        if "respiratory" in category or any("cough" in s.lower() for s in symptoms):
            recommendations.append("พักผ่อนให้เพียงพอและดื่มน้ำอุ่น")
        if "fever" in " ".join(symptoms).lower():
            recommendations.append("คลายร้อนและดื่มน้ำมาก ๆ")

    # General recommendations
    recommendations.extend([
        "บันทึกอาการและการเปลี่ยนแปลง",
        "หลีกเลี่ยงการออกแรงหนัก",
        "รับประทานอาหารที่มีประโยชน์"
    ])

    return recommendations


def _get_next_steps(urgency: str) -> List[str]:
    """Get next steps based on urgency level"""

    steps = {
        "critical": [
            "โทร 1669 ทันที",
            "ไปโรงพยาบาลด่วนที่สุด",
            "เตรียมข้อมูลผู้ป่วยและยา"
        ],
        "high": [
            "ไปโรงพยาบาลโดยเร็ว",
            "โทรแจ้งโรงพยาบาลก่อน",
            "เตรียมเอกสารการรักษา"
        ],
        "medium": [
            "นัดพบแพทย์ภายใน 1-2 วัน",
            "สังเกตอาการอย่างใกล้ชิด",
            "บันทึกการเปลี่ยนแปลง"
        ],
        "low": [
            "ดูแลตัวเองที่บ้าน",
            "สังเกตอาการต่อไป",
            "พบแพทย์หากอาการแย่ลง"
        ]
    }

    return steps.get(urgency, steps["low"])


def _get_icd_category(icd_code: str) -> str:
    """Get ICD-10 category description"""

    if not icd_code:
        return "Unknown"

    # Extract first character to determine category
    first_char = icd_code[0].upper()

    categories = {
        "A": "Infectious and parasitic diseases",
        "B": "Infectious and parasitic diseases",
        "C": "Neoplasms",
        "D": "Blood disorders / Neoplasms",
        "E": "Endocrine disorders",
        "F": "Mental disorders",
        "G": "Nervous system",
        "H": "Eye and ear disorders",
        "I": "Circulatory system",
        "J": "Respiratory system",
        "K": "Digestive system",
        "L": "Skin disorders",
        "M": "Musculoskeletal system",
        "N": "Genitourinary system",
        "O": "Pregnancy and childbirth",
        "P": "Perinatal conditions",
        "Q": "Congenital malformations",
        "R": "Symptoms and signs",
        "S": "Injury and poisoning",
        "T": "Injury and poisoning",
        "V": "External causes",
        "W": "External causes",
        "X": "External causes",
        "Y": "External causes",
        "Z": "Health status factors"
    }

    return categories.get(first_char, "Unknown category")