#!/usr/bin/env python3
"""
Case Review Recommendation System
=================================
Flags cases that may need human review despite matching RAG knowledge
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class ReviewReason(Enum):
    KEYWORD_VS_CONTEXT_MISMATCH = "keyword_vs_context_mismatch"
    TEMPORAL_SEQUENCE_IGNORED = "temporal_sequence_ignored"
    WORK_RELATED_SYMPTOMS = "work_related_symptoms"
    AGE_INAPPROPRIATE_DIAGNOSIS = "age_inappropriate_diagnosis"
    SYMPTOM_CONTEXT_DISCONNECT = "symptom_context_disconnect"

@dataclass
class CaseReviewFlag:
    reason: ReviewReason
    confidence: float
    explanation: str
    suggested_alternative: Optional[str] = None

class CaseReviewAnalyzer:
    """Analyzes cases to determine if they need human review"""

    def __init__(self):
        self.review_patterns = self._initialize_review_patterns()

    def _initialize_review_patterns(self) -> Dict[str, Any]:
        """Initialize patterns that trigger case review"""
        return {
            "temporal_disconnect": {
                "pattern": ["วันนี้", "today", "หลังจาก", "after"],
                "symptoms": ["ปวดหัว", "ปวดตา", "เหนื่อย", "อ่อนเพลีย"],
                "triggers": ["ทำงานหนัก", "hard work", "เครียด", "stress"]
            },
            "work_related_context": {
                "occupational": ["ทำงาน", "work", "งาน", "job"],
                "symptoms": ["headache", "ปวดหัว", "eye pain", "ปวดตา", "fatigue", "เหนื่อย"],
                "inappropriate_diagnoses": ["Allergic Reaction", "ปฏิกิริยาแพ้"]
            },
            "age_symptom_mismatch": {
                "young_healthy": {"age_range": (18, 35), "health_status": ["ไม่มีประวัติ", "สุขภาพดี"]},
                "serious_diagnoses": ["stroke", "heart attack", "cancer", "โรคหัวใจ", "มะเร็ง"]
            }
        }

    def analyze_case(
        self,
        message: str,
        patient_context: str,
        ai_diagnosis: Dict[str, Any]
    ) -> List[CaseReviewFlag]:
        """Analyze case and return review flags if needed"""

        flags = []

        # Check for temporal sequence issues
        temporal_flag = self._check_temporal_sequence(message, patient_context, ai_diagnosis)
        if temporal_flag:
            flags.append(temporal_flag)

        # Check for work-related context mismatches
        work_flag = self._check_work_context_mismatch(message, patient_context, ai_diagnosis)
        if work_flag:
            flags.append(work_flag)

        # Check for age-inappropriate diagnoses
        age_flag = self._check_age_appropriateness(patient_context, ai_diagnosis)
        if age_flag:
            flags.append(age_flag)

        return flags

    def _check_temporal_sequence(
        self,
        message: str,
        patient_context: str,
        ai_diagnosis: Dict[str, Any]
    ) -> Optional[CaseReviewFlag]:
        """Check if temporal sequence suggests different diagnosis"""

        full_text = f"{message} {patient_context}".lower()

        # Look for temporal indicators
        temporal_indicators = ["วันนี้", "today", "หลังจาก", "after", "แล้ว"]
        work_activities = ["ทำงานหนัก", "hard work", "ทำงาน", "work"]
        then_symptoms = ["ปวดหัว", "ปวดตา", "headache", "eye pain"]

        has_temporal = any(indicator in full_text for indicator in temporal_indicators)
        has_work = any(work in full_text for work in work_activities)
        has_symptoms = any(symptom in full_text for symptom in then_symptoms)

        if has_temporal and has_work and has_symptoms:
            diagnosis_name = ai_diagnosis.get('english_name', '')

            # If diagnosis doesn't match temporal context
            work_related_diagnoses = ['tension', 'headache', 'strain', 'stress', 'fatigue']
            is_work_related = any(term in diagnosis_name.lower() for term in work_related_diagnoses)

            if not is_work_related:
                return CaseReviewFlag(
                    reason=ReviewReason.TEMPORAL_SEQUENCE_IGNORED,
                    confidence=0.85,
                    explanation="ผู้ป่วยระบุลำดับเวลา 'วันนี้ทำงานหนัก แล้วปวดหัว ปวดตา' แต่ AI วินิจฉัยโรคที่ไม่เกี่ยวข้องกับการทำงาน",
                    suggested_alternative="Tension headache, Eye strain, Work-related fatigue"
                )

        return None

    def _check_work_context_mismatch(
        self,
        message: str,
        patient_context: str,
        ai_diagnosis: Dict[str, Any]
    ) -> Optional[CaseReviewFlag]:
        """Check if work context suggests different diagnosis"""

        full_text = f"{message} {patient_context}".lower()

        work_indicators = ["ทำงาน", "work", "งาน", "job", "office", "computer"]
        has_work_context = any(indicator in full_text for indicator in work_indicators)

        if has_work_context:
            diagnosis_name = ai_diagnosis.get('english_name', '')

            # Check if diagnosis is inappropriate for work context
            if diagnosis_name == "Allergic Reaction" and "ปวดหัว" in message and "ปวดตา" in message:
                return CaseReviewFlag(
                    reason=ReviewReason.WORK_RELATED_SYMPTOMS,
                    confidence=0.90,
                    explanation="ผู้ป่วยมีอาการปวดหัว ปวดตา ในบริบทการทำงาน แต่ AI วินิจฉัย Allergic Reaction ซึ่งไม่สอดคล้องกับบริบท",
                    suggested_alternative="Tension headache, Computer eye strain, Work stress"
                )

        return None

    def _check_age_appropriateness(
        self,
        patient_context: str,
        ai_diagnosis: Dict[str, Any]
    ) -> Optional[CaseReviewFlag]:
        """Check if diagnosis is age-appropriate"""

        # Extract age from context
        age = None
        if "อายุ" in patient_context:
            try:
                age_part = patient_context.split("อายุ")[1].split()[0]
                age = int(age_part)
            except:
                pass

        if age and 18 <= age <= 35:
            health_indicators = ["ไม่มีประวัติ", "สุขภาพดี", "healthy"]
            is_healthy = any(indicator in patient_context for indicator in health_indicators)

            if is_healthy:
                diagnosis_name = ai_diagnosis.get('english_name', '')
                serious_conditions = ['emergency', 'crisis', 'attack', 'stroke']

                if any(serious in diagnosis_name.lower() for serious in serious_conditions):
                    return CaseReviewFlag(
                        reason=ReviewReason.AGE_INAPPROPRIATE_DIAGNOSIS,
                        confidence=0.75,
                        explanation=f"ผู้ป่วยอายุ {age} ปี สุขภาพดี แต่ได้รับการวินิจฉัยโรคร้าย/ฉุกเฉินซึ่งไม่เหมาะสมกับกลุ่มอายุ",
                        suggested_alternative="Consider common conditions for young healthy adults"
                    )

        return None

    def generate_recommendation(
        self,
        message: str,
        patient_context: str,
        ai_diagnosis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate case review recommendation"""

        flags = self.analyze_case(message, patient_context, ai_diagnosis)

        if flags:
            return {
                "needs_review": True,
                "recommendation": "เคสนี้ไม่ครอบคลุม - ต้องการการตรวจสอบโดยแพทย์",
                "flags": [
                    {
                        "reason": flag.reason.value,
                        "confidence": flag.confidence,
                        "explanation": flag.explanation,
                        "suggested_alternative": flag.suggested_alternative
                    }
                    for flag in flags
                ],
                "priority": "high" if any(f.confidence > 0.8 for f in flags) else "medium",
                "thai_message": "การวินิจฉัยนี้อาจไม่สอดคล้องกับบริบทของผู้ป่วย แนะนำให้แพทย์ตรวจสอบเพิ่มเติม"
            }
        else:
            return {
                "needs_review": False,
                "recommendation": "การวินิจฉัยสอดคล้องกับบริบท",
                "flags": [],
                "priority": "none"
            }

def test_case_review():
    """Test the case review system with the real message"""

    analyzer = CaseReviewAnalyzer()

    # Your real case
    message = "ฉันอายุ 28 สูง 170 หนัก 65 ไม่มีประวัติโรคประจำตัว การแพ้ยา แพ้อาหาร วันนี้ทำงานหนัก แล้วปวดหัว ปวดตา ตอนนี้ฉันเป็นโรคอะไร"
    patient_context = "อายุ 28 สูง 170 หนัก 65 ไม่มีประวัติโรคประจำตัว ไม่แพ้ยา ไม่แพ้อาหาร ทำงานหนัก"
    ai_diagnosis = {
        "english_name": "Allergic Reaction",
        "thai_name": "ปฏิกิริยาแพ้",
        "confidence": 95,
        "icd_code": "T78.40"
    }

    recommendation = analyzer.generate_recommendation(message, patient_context, ai_diagnosis)

    print("🔍 CASE REVIEW ANALYSIS")
    print("=" * 50)
    print(f"📝 Message: {message}")
    print(f"👤 Context: {patient_context}")
    print(f"🤖 AI Diagnosis: {ai_diagnosis['english_name']}")
    print()

    if recommendation["needs_review"]:
        print("⚠️  REVIEW NEEDED")
        print(f"📋 Recommendation: {recommendation['recommendation']}")
        print(f"🔺 Priority: {recommendation['priority'].upper()}")
        print(f"💬 Thai Message: {recommendation['thai_message']}")
        print()

        print("🚩 FLAGS DETECTED:")
        for i, flag in enumerate(recommendation["flags"], 1):
            print(f"   {i}. {flag['reason']}")
            print(f"      Confidence: {flag['confidence']*100:.1f}%")
            print(f"      Explanation: {flag['explanation']}")
            if flag['suggested_alternative']:
                print(f"      Suggested: {flag['suggested_alternative']}")
            print()
    else:
        print("✅ NO REVIEW NEEDED")
        print(f"📋 Recommendation: {recommendation['recommendation']}")

    # Save recommendation
    report = {
        "timestamp": datetime.now().isoformat(),
        "case": {
            "message": message,
            "patient_context": patient_context,
            "ai_diagnosis": ai_diagnosis
        },
        "review_analysis": recommendation
    }

    filename = f"case_review_recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"📁 Analysis saved to: {filename}")

if __name__ == "__main__":
    test_case_review()