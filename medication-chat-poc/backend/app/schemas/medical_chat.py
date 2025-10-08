# Pydantic schemas for Medical Chat API

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class UrgencyLevel(str, Enum):
    """Medical urgency levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TriageLevel(int, Enum):
    """Medical triage levels"""
    RESUSCITATION = 1
    EMERGENCY = 2
    URGENT = 3
    SEMI_URGENT = 4
    NON_URGENT = 5


class DiagnosisConfidence(str, Enum):
    """Diagnosis confidence levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class Language(str, Enum):
    """Supported languages"""
    THAI = "thai"
    ENGLISH = "english"
    AUTO = "auto"


# Request Models

class PatientInfo(BaseModel):
    """Patient information"""
    age: Optional[int] = Field(None, ge=0, le=150, description="Patient age")
    gender: Optional[str] = Field(None, description="Patient gender")
    allergies: Optional[List[str]] = Field(default_factory=list, description="Known allergies")
    conditions: Optional[List[str]] = Field(default_factory=list, description="Existing medical conditions")
    medications: Optional[List[str]] = Field(default_factory=list, description="Current medications")



class ConversationMessage(BaseModel):
    """Individual conversation message"""
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="Message timestamp")


class MedicalChatRequest(BaseModel):
    """Medical chat request"""
    message: str = Field(..., min_length=1, max_length=5000, description="Patient's message or symptoms")
    conversation_history: Optional[List[ConversationMessage]] = Field(
        default_factory=list,
        description="Previous conversation messages"
    )
    patient_info: Optional[PatientInfo] = Field(None, description="Patient demographic and medical information")
    preferred_language: Optional[Language] = Field(Language.AUTO, description="Preferred response language")
    session_id: Optional[str] = Field(None, description="Session identifier for tracking")
    include_reasoning: bool = Field(False, description="Include AI reasoning chain in response")

    @validator('message')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()


class DiagnosisRequest(BaseModel):
    """Diagnosis analysis request"""
    symptoms: List[str] = Field(..., min_items=1, description="List of symptoms")
    patient_info: Optional[PatientInfo] = Field(None, description="Patient information")
    medical_history: Optional[str] = Field(None, description="Relevant medical history")
    duration: Optional[str] = Field(None, description="Symptom duration")
    severity: Optional[str] = Field(None, description="Symptom severity description")


class FeedbackRequest(BaseModel):
    """Doctor feedback request"""
    chat_id: str = Field(..., description="Chat session identifier")
    diagnosis_approved: bool = Field(..., description="Whether the AI diagnosis was approved")
    corrected_diagnosis: Optional[Dict[str, Any]] = Field(None, description="Corrected diagnosis if not approved")
    corrected_medications: Optional[List[Dict[str, Any]]] = Field(None, description="Corrected medications")
    feedback_notes: Optional[str] = Field(None, description="Additional feedback notes")
    doctor_id: str = Field(..., description="Doctor identifier")
    urgency_assessment: Optional[UrgencyLevel] = Field(None, description="Doctor's urgency assessment")


# Response Models

class Medication(BaseModel):
    """Medication information"""
    english_name: str = Field(..., description="Medication name in English")
    thai_name: str = Field(..., description="Medication name in Thai")
    dosage: Optional[str] = Field(None, description="Recommended dosage")
    instructions: str = Field(..., description="Usage instructions")
    category: Optional[str] = Field(None, description="Medication category")


class Diagnosis(BaseModel):
    """Medical diagnosis information"""
    icd_code: str = Field(..., description="ICD-10 diagnosis code")
    english_name: str = Field(..., description="Diagnosis name in English")
    thai_name: str = Field(..., description="Diagnosis name in Thai")
    confidence: float = Field(..., ge=0, le=100, description="Confidence percentage")
    category: Optional[str] = Field(None, description="Medical category")


class AgentReasoning(BaseModel):
    """Agent reasoning step"""
    agent: str = Field(..., description="Agent name")
    step: int = Field(..., description="Reasoning step number")
    reasoning: str = Field(..., description="Agent's reasoning")
    action: str = Field(..., description="Action taken")
    observation: str = Field(..., description="Observation from action")
    confidence: float = Field(..., ge=0, le=100, description="Confidence in reasoning")


class TriageAssessment(BaseModel):
    """Triage assessment result"""
    urgency: UrgencyLevel = Field(..., description="Urgency level")
    triage_level: TriageLevel = Field(..., description="Triage level (1-5)")
    emergency_detected: bool = Field(..., description="Whether emergency was detected")
    emergency_keywords: List[str] = Field(default_factory=list, description="Detected emergency keywords")
    recommendation: str = Field(..., description="Triage recommendation")


class DiagnosisAssessment(BaseModel):
    """Diagnosis assessment result"""
    primary_diagnosis: Optional[Diagnosis] = Field(None, description="Primary diagnosis")
    differential_diagnoses: List[Diagnosis] = Field(default_factory=list, description="Alternative diagnoses")
    confidence: float = Field(..., ge=0, le=100, description="Overall diagnostic confidence")
    reasoning: str = Field(..., description="Diagnostic reasoning")


class TreatmentPlan(BaseModel):
    """Treatment plan recommendation"""
    medications: List[Medication] = Field(default_factory=list, description="Recommended medications")
    lifestyle_recommendations: List[str] = Field(default_factory=list, description="Lifestyle recommendations")
    follow_up_instructions: str = Field(..., description="Follow-up instructions")
    safety_warnings: List[str] = Field(default_factory=list, description="Safety warnings")
    contraindications: List[str] = Field(default_factory=list, description="Contraindications")


class ProcessingMetadata(BaseModel):
    """Processing metadata"""
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    translation_used: bool = Field(..., description="Whether translation was used")
    detected_language: str = Field(..., description="Detected input language")
    detected_dialects: List[str] = Field(default_factory=list, description="Detected Thai dialects")
    agents_used: int = Field(..., description="Number of agents involved")
    rag_results_count: int = Field(..., description="Number of RAG search results")


class MedicalChatResponse(BaseModel):
    """Medical chat response"""
    message: str = Field(..., description="AI response message")
    type: str = Field(..., description="Response type (emergency, comprehensive_analysis, etc.)")

    # Assessment results
    triage: Optional[TriageAssessment] = Field(None, description="Triage assessment")
    diagnosis: Optional[DiagnosisAssessment] = Field(None, description="Diagnosis assessment")
    treatment: Optional[TreatmentPlan] = Field(None, description="Treatment recommendations")

    # Agent information
    agent_reasoning_chain: Optional[List[AgentReasoning]] = Field(None, description="Agent reasoning chain")

    # Metadata
    metadata: ProcessingMetadata = Field(..., description="Processing metadata")
    session_id: Optional[str] = Field(None, description="Session identifier")

    # Disclaimers and recommendations
    disclaimer: str = Field(..., description="Medical disclaimer")
    recommendation: str = Field(..., description="Final recommendation")

    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class EmergencyResponse(BaseModel):
    """Emergency response"""
    type: str = Field(default="emergency", description="Response type")
    message: str = Field(..., description="Emergency message")
    urgency: UrgencyLevel = Field(UrgencyLevel.CRITICAL, description="Urgency level")
    recommendations: List[str] = Field(..., description="Emergency recommendations")
    emergency_contacts: List[str] = Field(
        default_factory=lambda: ["1669 (Emergency Services)", "Local Hospital"],
        description="Emergency contact numbers"
    )
    warning: str = Field(..., description="Emergency warning message")
    detected_keywords: List[str] = Field(default_factory=list, description="Emergency keywords detected")
    agent_reasoning: Optional[List[AgentReasoning]] = Field(None, description="Agent reasoning for emergency")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    metadata: Optional[ProcessingMetadata] = Field(None, description="Processing metadata")
    disclaimer: str = Field(default="ข้อมูลนี้เป็นเพียงข้อมูลทั่วไปเท่านั้น ไม่ใช่การวินิจฉัยทางการแพทย์ กรุณาปรึกษาแพทย์หรือผู้เชี่ยวชาญด้านสุขภาพสำหรับการวินิจฉัยและการรักษาที่เหมาะสม", description="Medical disclaimer")
    recommendation: str = Field(default="โทร 1669 ทันที - สถานการณ์ฉุกเฉิน", description="Primary recommendation")


class DiagnosisResponse(BaseModel):
    """Diagnosis response"""
    primary_diagnosis: Optional[Diagnosis] = Field(None, description="Primary diagnosis")
    differential_diagnoses: List[Diagnosis] = Field(default_factory=list, description="Differential diagnoses")
    confidence: float = Field(..., ge=0, le=100, description="Overall confidence")
    risk_score: int = Field(..., ge=0, le=100, description="Risk score")
    urgency: UrgencyLevel = Field(..., description="Urgency level")
    recommendations: List[str] = Field(default_factory=list, description="Medical recommendations")
    reasoning: str = Field(..., description="Diagnostic reasoning")
    metadata: ProcessingMetadata = Field(..., description="Processing metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class FeedbackResponse(BaseModel):
    """Feedback response"""
    status: str = Field(..., description="Feedback processing status")
    message: str = Field(..., description="Response message")
    feedback_id: str = Field(..., description="Feedback record identifier")
    model_updated: bool = Field(..., description="Whether the AI model was updated")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    services: Dict[str, str] = Field(..., description="Individual service statuses")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


# Error Models

class ErrorResponse(BaseModel):
    """Error response"""
    error: bool = Field(True, description="Error flag")
    message: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


# Export all models
__all__ = [
    # Enums
    "UrgencyLevel", "TriageLevel", "DiagnosisConfidence", "Language",

    # Request models
    "PatientInfo", "ConversationMessage", "MedicalChatRequest",
    "DiagnosisRequest", "FeedbackRequest",

    # Response models
    "Medication", "Diagnosis", "AgentReasoning", "TriageAssessment",
    "DiagnosisAssessment", "TreatmentPlan", "ProcessingMetadata",
    "MedicalChatResponse", "EmergencyResponse", "DiagnosisResponse",
    "FeedbackResponse", "HealthCheckResponse", "ErrorResponse"
]