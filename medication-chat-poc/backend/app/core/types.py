# Core Precision-Oriented Medical AI Types
# Implements strict contracts for all agent interactions

from pydantic import BaseModel, Field, confloat, validator
from typing import List, Dict, Optional, Literal, Union
from enum import Enum
from datetime import datetime


class TriageLevel(str, Enum):
    """Standardized triage levels"""
    RESUSCITATION = "resuscitation"  # 1 - Red - Immediate
    EMERGENCY = "emergency"          # 2 - Orange - Within 10 min
    URGENT = "urgent"               # 3 - Yellow - Within 30 min
    SEMI_URGENT = "semi_urgent"     # 4 - Green - Within 60 min
    NON_URGENT = "non_urgent"       # 5 - Blue - Within 120 min


class RoutingReason(str, Enum):
    """Evidence-based routing rationales"""
    CHEST_PAIN_RISK = "chest_pain_risk"
    FEVER_HEADACHE_REDFLAGS = "fever_headache_redflags"
    NEURO_DEFICIT = "neuro_deficit"
    ABDOMINAL_PAIN = "abdominal_pain"
    RESPIRATORY_DISTRESS = "respiratory_distress"
    CARDIAC_SYMPTOMS = "cardiac_symptoms"
    EMERGENCY_KEYWORDS = "emergency_keywords"
    BASIC_SYMPTOMS = "basic_symptoms"


class Evidence(BaseModel):
    """Structured evidence for medical reasoning"""
    for_: List[str] = Field(default=[], alias="for", description="Evidence supporting the diagnosis")
    against: List[str] = Field(default=[], description="Evidence against the diagnosis")
    citations: List[str] = Field(default=[], description="Guideline IDs, KB references, calculator refs")

    @validator('citations')
    def validate_citations(cls, v):
        """Ensure citations follow expected format"""
        valid_prefixes = ['guideline:', 'icd:', 'calculator:', 'kb:', 'study:', 'system:']
        for citation in v:
            if not any(citation.startswith(prefix) for prefix in valid_prefixes):
                raise ValueError(f"Invalid citation format: {citation}")
        return v


class DxCandidate(BaseModel):
    """Diagnosis candidate with evidence and probability"""
    icd10: str = Field(..., description="ICD-10 code")
    label: str = Field(..., description="Human-readable diagnosis name")
    p: confloat(ge=0.0, le=1.0) = Field(..., description="Probability estimate")
    evidence: Evidence = Field(..., description="Supporting/opposing evidence")

    @validator('icd10')
    def validate_icd10(cls, v):
        """Basic ICD-10 format validation"""
        if not v or len(v) < 3:
            raise ValueError("ICD-10 code must be at least 3 characters")
        return v.upper()


class Calculator(BaseModel):
    """Medical calculator with structured inputs/outputs"""
    name: str = Field(..., description="Calculator name (HEART, PERC, etc.)")
    inputs_used: Dict[str, Union[str, int, float, bool]] = Field(..., description="Actual inputs provided")
    score: Union[int, float] = Field(..., description="Calculated score")
    risk_band: str = Field(..., description="Risk stratification result")
    reference: str = Field(..., description="Citation for calculator")
    confidence: confloat(ge=0.0, le=1.0) = Field(..., description="Confidence in inputs completeness")


class Test(BaseModel):
    """Diagnostic test recommendation"""
    name: str = Field(..., description="Test name")
    rationale: str = Field(..., description="Why this test is needed")
    voi_score: confloat(ge=0.0, le=1.0) = Field(..., description="Value of information score")
    urgency: TriageLevel = Field(..., description="How urgently needed")


class Treatment(BaseModel):
    """Treatment recommendation with evidence"""
    medication: Optional[str] = Field(None, description="Medication name")
    dosage: Optional[str] = Field(None, description="Dosage information")
    instructions: str = Field(..., description="Treatment instructions")
    contraindications: List[str] = Field(default=[], description="Known contraindications")
    evidence: Evidence = Field(..., description="Supporting evidence")
    safety_score: confloat(ge=0.0, le=1.0) = Field(..., description="Safety confidence")


class Uncertainty(BaseModel):
    """Uncertainty quantification"""
    diagnostic_coverage: confloat(ge=0.0, le=1.0) = Field(..., description="Probability true dx in differential")
    safety_certainty: confloat(ge=0.0, le=1.0) = Field(..., description="Probability no red flags missed")
    abstention_reason: Optional[str] = Field(None, description="Why abstaining if applicable")
    prediction_set_size: int = Field(..., description="Size of prediction set for coverage")


class DiagnosisCard(BaseModel):
    """Complete precision-oriented diagnosis output"""
    patient_id: str = Field(..., description="Patient identifier")
    language: Literal["thai", "english", "auto"] = Field(..., description="Primary language")

    # Core medical content
    triage: Dict[str, Union[TriageLevel, str]] = Field(..., description="Triage level and rationale")
    differential: List[DxCandidate] = Field(..., min_items=1, description="Differential diagnosis ranked by probability")
    calculators: List[Calculator] = Field(default=[], description="Applied medical calculators")
    tests: List[Test] = Field(default=[], description="Recommended diagnostic tests")
    treatment_candidates: List[Treatment] = Field(default=[], description="Treatment recommendations")

    # Precision metrics
    uncertainty: Uncertainty = Field(..., description="Uncertainty quantification")
    overall_confidence: confloat(ge=0.0, le=1.0) = Field(..., description="Overall diagnostic confidence")

    # Routing and processing metadata
    routing_reasons: List[RoutingReason] = Field(default=[], description="Why specific tools were called")
    processing_metadata: Dict[str, Union[str, int, float]] = Field(default={}, description="Processing details")

    # Timestamps and identifiers
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: str = Field(..., description="Session identifier")

    @validator('differential')
    def validate_differential_probabilities(cls, v):
        """Ensure probabilities are reasonable"""
        if len(v) > 1:
            total_p = sum(dx.p for dx in v)
            if total_p > 1.1:  # Allow small floating point errors
                raise ValueError(f"Total probability {total_p} exceeds 1.0")
        return v

    @validator('treatment_candidates')
    def validate_treatment_evidence(cls, v):
        """Ensure treatments have guideline citations"""
        for treatment in v:
            if not any(citation.startswith('guideline:') for citation in treatment.evidence.citations):
                raise ValueError(f"Treatment '{treatment.instructions}' missing guideline citation")
        return v


class RouteSignals(BaseModel):
    """Signals used for evidence-first routing"""
    chest_pain: bool = False
    fever: bool = False
    severe_headache: bool = False
    breathing_difficulty: bool = False
    neurological_deficit: bool = False
    abdominal_pain: bool = False
    emergency_keywords: List[str] = Field(default=[])
    # Note: vital_signs removed - inappropriate for consultation scope

    @classmethod
    def from_symptoms(cls, symptoms: str) -> "RouteSignals":
        """Extract routing signals from symptom text"""
        symptoms_lower = symptoms.lower()

        return cls(
            chest_pain=any(keyword in symptoms_lower for keyword in [
                "chest pain", "ปวดหน้าอก", "เจ็บหน้าอก", "แน่นหน้าอก"
            ]),
            fever=any(keyword in symptoms_lower for keyword in [
                "fever", "ไข้", "มีไข้", "ตัวร้อน"
            ]),
            severe_headache=any(keyword in symptoms_lower for keyword in [
                "severe headache", "ปวดหัวรุนแรง", "ปวดหัวมาก", "หัวปวดแสบ"
            ]),
            breathing_difficulty=any(keyword in symptoms_lower for keyword in [
                "shortness of breath", "หายใจไม่ออก", "หายใจลำบาก", "อึดอัด"
            ]),
            neurological_deficit=any(keyword in symptoms_lower for keyword in [
                "paralysis", "อัมพาต", "พูดไม่ได้", "มึนงง", "ชัก"
            ]),
            abdominal_pain=any(keyword in symptoms_lower for keyword in [
                "abdominal pain", "ปวดท้อง", "เจ็บท้อง", "ท้องเสียว"
            ]),
            emergency_keywords=[
                keyword for keyword in [
                    "emergency", "urgent", "ฉุกเฉิน", "เร่งด่วน", "รุนแรง"
                ] if keyword in symptoms_lower
            ]
        )


class VOIQuestion(BaseModel):
    """Value of Information question"""
    question: str = Field(..., description="Question to ask")
    voi_score: confloat(ge=0.0, le=1.0) = Field(..., description="Expected value of information")
    expected_delta_p: confloat(ge=0.0, le=1.0) = Field(..., description="Expected change in top diagnosis probability")
    category: str = Field(..., description="Question category (vitals, history, etc.)")


class PrecisionPlan(BaseModel):
    """Execution plan with success criteria"""
    steps: List[str] = Field(..., description="Ordered execution steps")
    success_criteria: Dict[str, str] = Field(..., description="Success criteria per step")
    routing_reasons: List[RoutingReason] = Field(..., description="Why specific routes were chosen")
    max_questions: int = Field(default=3, description="Maximum VOI questions to ask")
    abstention_threshold: confloat(ge=0.0, le=1.0) = Field(default=0.7, description="Confidence threshold for abstention")


class CriticResult(BaseModel):
    """Result from precision critic"""
    passed: bool = Field(..., description="Whether all checks passed")
    failed_rules: List[str] = Field(default=[], description="Names of failed validation rules")
    actions: List[str] = Field(default=[], description="Required actions (request_info, abstain, escalate)")
    rationale: str = Field(..., description="Explanation of critic decision")


# Validation Functions
def validate_diagnosis_card(card: DiagnosisCard) -> CriticResult:
    """Comprehensive validation of diagnosis card"""
    failed_rules = []
    actions = []

    # Rule: Treatment must have guideline citations
    for treatment in card.treatment_candidates:
        if not any(c.startswith('guideline:') for c in treatment.evidence.citations):
            failed_rules.append("treatment_without_guideline")
            actions.append("request_info")

    # Rule: High-risk diagnoses need supporting evidence
    for dx in card.differential[:3]:  # Top 3
        if dx.icd10.startswith(('I2', 'I4', 'G0', 'R06')) and not dx.evidence.for_:
            failed_rules.append("high_risk_without_evidence")
            actions.append("request_info")

    # Rule: Safety certainty threshold
    if card.uncertainty.safety_certainty < 0.85:
        failed_rules.append("low_safety_certainty")
        actions.append("escalate")

    # Rule: Calculator inputs must be captured
    for calc in card.calculators:
        required_fields = calc.inputs_used.keys()
        # This would check against captured patient data in real implementation
        if calc.confidence < 0.8:
            failed_rules.append("incomplete_calculator_inputs")
            actions.append("request_info")

    passed = len(failed_rules) == 0

    return CriticResult(
        passed=passed,
        failed_rules=failed_rules,
        actions=actions,
        rationale=f"Checked {len(card.differential)} diagnoses, {len(card.calculators)} calculators, {len(card.treatment_candidates)} treatments"
    )


# Export main types
__all__ = [
    'DiagnosisCard',
    'DxCandidate',
    'Evidence',
    'Calculator',
    'Test',
    'Treatment',
    'Uncertainty',
    'RouteSignals',
    'VOIQuestion',
    'PrecisionPlan',
    'CriticResult',
    'TriageLevel',
    'RoutingReason',
    'validate_diagnosis_card'
]