# Medical AI Service - Core agentic AI business logic
# Implements multi-agent medical reasoning system

import logging
import asyncio
import json
import csv
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from app.util.config import get_settings, MedicalConstants
from app.schemas.medical_chat import (
    PatientInfo, ConversationMessage,
    UrgencyLevel, TriageLevel, DiagnosisConfidence
)

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class AgentThought:
    agent: str
    step: int
    reasoning: str
    action: str
    observation: str
    confidence: float


@dataclass
class MedicalKnowledgeItem:
    id: str
    name_en: str
    name_th: str
    category: str
    description: str
    icd_code: Optional[str] = None


class MedicalAIService:
    """Core Medical AI Service with agentic reasoning capabilities"""

    def __init__(self):
        self.ollama_url = settings.ollama_url
        self.seallm_model = settings.seallm_model
        self.medllama_model = settings.medllama_model

        # Medical knowledge bases
        self.medicines: List[MedicalKnowledgeItem] = []
        self.diagnoses: List[MedicalKnowledgeItem] = []
        self.treatments: List[MedicalKnowledgeItem] = []

        # Conversation storage
        self.conversation_history: Dict[str, List[Dict]] = {}
        self.system_stats = {
            "total_consultations": 0,
            "emergency_cases": 0,
            "avg_response_time": 0,
            "languages_detected": {}
        }

        # Agent system
        self.agents = {
            "triage": TriageAgent(),
            "diagnostic": DiagnosticAgent(),
            "treatment": TreatmentAgent(),
            "coordinator": CoordinatorAgent()
        }

    async def initialize(self):
        """Initialize medical AI service"""
        logger.info("🚀 Initializing Medical AI Service...")

        try:
            # Load medical knowledge bases
            await self._load_medical_data()

            # Initialize agents
            for agent_name, agent in self.agents.items():
                await agent.initialize(self)
                logger.info(f"✅ {agent_name} agent initialized")

            # Test AI models
            await self._test_ai_models()

            logger.info("✅ Medical AI Service initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Medical AI Service: {e}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up Medical AI Service...")
        for agent in self.agents.values():
            if hasattr(agent, 'cleanup'):
                await agent.cleanup()

    async def process_medical_consultation(
        self,
        message: str,
        conversation_history: Optional[List[ConversationMessage]] = None,
        patient_info: Optional[PatientInfo] = None,
        preferred_language: str = "auto",
        session_id: Optional[str] = None,
        include_reasoning: bool = False
    ) -> Dict[str, Any]:
        """Process medical consultation through agentic AI system"""

        logger.info(f"🩺 Processing medical consultation: {message[:100]}...")
        start_time = datetime.now()

        try:
            # Detect language and translate if needed
            language_info = await self._detect_and_translate(message)
            processed_message = language_info.get("translated_message", message)

            # Quick emergency check
            emergency_result = await self.check_emergency_keywords(processed_message)
            if emergency_result["is_emergency"]:
                return await self._handle_emergency_response(
                    emergency_result, language_info, include_reasoning
                )

            # Multi-agent processing
            coordinator = self.agents["coordinator"]
            result = await coordinator.process_medical_case({
                "message": processed_message,
                "original_message": message,
                "conversation_history": conversation_history or [],
                "patient_info": patient_info,
                "language_info": language_info,
                "session_id": session_id
            })

            # Translate response back if needed
            if language_info.get("detected_language") == "thai":
                result["message"] = await self._translate_to_thai(result["message"])

            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_stats(processing_time, language_info.get("detected_language"))

            # Store conversation
            if session_id:
                await self._store_conversation(session_id, message, result)

            return result

        except Exception as e:
            logger.error(f"❌ Medical consultation processing failed: {e}")
            return {
                "type": "error",
                "message": "เกิดข้อผิดพลาดในระหว่างการประมวลผล กรุณาลองใหม่อีกครั้ง",
                "error": str(e)
            }

    async def check_emergency_keywords(self, message: str) -> Dict[str, Any]:
        """Check for emergency keywords across languages and dialects"""

        emergency_keywords = []
        detected_dialects = []
        is_emergency = False
        confidence = 0.0

        message_lower = message.lower()

        # Check English emergency keywords
        for keyword in MedicalConstants.EMERGENCY_KEYWORDS["english"]:
            if keyword in message_lower:
                emergency_keywords.append(keyword)
                is_emergency = True
                confidence += 0.4

        # Check Thai standard keywords
        for keyword in MedicalConstants.EMERGENCY_KEYWORDS["thai_standard"]:
            if keyword in message:
                emergency_keywords.append(keyword)
                is_emergency = True
                confidence += 0.5

        # Check dialect keywords
        for dialect, keywords in MedicalConstants.EMERGENCY_KEYWORDS.items():
            if dialect.startswith("thai_") and dialect != "thai_standard":
                for keyword in keywords:
                    if keyword in message:
                        emergency_keywords.append(keyword)
                        detected_dialects.append(dialect.replace("thai_", ""))
                        is_emergency = True
                        confidence += 0.6

        # Enhanced emergency symptom detection
        emergency_symptoms = [
            "chest pain", "ปวดหน้าอก", "เจ็บหน้าอก", "หายใจไม่ออก", "หายใจไม่อิ่ม", "can't breathe",
            "unconscious", "หมดสติ", "stroke", "โรคหลอดเลือดสมอง", "หัวใจวาย", "heart attack",
            "แน่นหน้าอก", "เหงื่อออก", "หน้าซีด", "ผื่นลมพิษ", "ริมฝีปากบวม", "หายใจลำบาก",
            "รุนแรง", "severe", "เจ็บมาก", "ปวดมาก", "unbearable", "intense"
        ]

        for symptom in emergency_symptoms:
            if symptom in message_lower or symptom in message:
                emergency_keywords.append(symptom)
                is_emergency = True
                confidence += 0.7

        confidence = min(confidence, 1.0)

        return {
            "is_emergency": is_emergency,
            "keywords": emergency_keywords,
            "dialects": detected_dialects,
            "confidence": confidence,
            "recommendation": "โทร 1669 เพื่อขอความช่วยเหลือฉุกเฉิน" if is_emergency else ""
        }

    async def perform_triage_assessment(
        self,
        message: str,
        patient_info: Optional[PatientInfo] = None
    ) -> Dict[str, Any]:
        """Perform medical triage assessment"""

        triage_agent = self.agents["triage"]
        result = await triage_agent.assess_urgency({
            "message": message,
            "patient_info": patient_info
        })

        return result

    async def get_conversation_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""

        if session_id not in self.conversation_history:
            return []

        history = self.conversation_history[session_id]
        return history[-limit:] if limit > 0 else history

    async def clear_conversation_history(self, session_id: str):
        """Clear conversation history for a session"""

        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
            logger.info(f"🗑️ Cleared conversation history for session {session_id}")

    async def get_system_statistics(self) -> Dict[str, Any]:
        """Get system performance statistics"""

        return {
            "total_consultations": self.system_stats["total_consultations"],
            "emergency_cases_detected": self.system_stats["emergency_cases"],
            "average_response_time_ms": self.system_stats["avg_response_time"],
            "languages_detected": self.system_stats["languages_detected"],
            "knowledge_base": {
                "medicines_count": len(self.medicines),
                "diagnoses_count": len(self.diagnoses),
                "treatments_count": len(self.treatments)
            },
            "agents_status": {
                agent_name: "active" for agent_name in self.agents.keys()
            },
            "uptime": datetime.now().isoformat()
        }

    # Private methods

    async def _load_medical_data(self):
        """Load medical knowledge from CSV files"""

        try:
            # Load medicines
            if os.path.exists(settings.medicine_data_path):
                with open(settings.medicine_data_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        medicine = MedicalKnowledgeItem(
                            id=row.get('id', ''),
                            name_en=row.get('english_name', ''),
                            name_th=row.get('thai_name', ''),
                            category=row.get('category', ''),
                            description=row.get('description', '')
                        )
                        self.medicines.append(medicine)
                logger.info(f"📊 Loaded {len(self.medicines)} medicines")

            # Load diagnoses
            if os.path.exists(settings.diagnosis_data_path):
                with open(settings.diagnosis_data_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        diagnosis = MedicalKnowledgeItem(
                            id=row.get('id', ''),
                            name_en=row.get('english_name', ''),
                            name_th=row.get('thai_name', ''),
                            category=row.get('category', ''),
                            description=row.get('description', ''),
                            icd_code=row.get('icd_code', '')
                        )
                        self.diagnoses.append(diagnosis)
                logger.info(f"📊 Loaded {len(self.diagnoses)} diagnoses")

            # Load treatments
            if os.path.exists(settings.treatment_data_path):
                with open(settings.treatment_data_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        treatment = MedicalKnowledgeItem(
                            id=row.get('id', ''),
                            name_en=row.get('english_name', ''),
                            name_th=row.get('thai_name', ''),
                            category=row.get('category', ''),
                            description=row.get('description', '')
                        )
                        self.treatments.append(treatment)
                logger.info(f"📊 Loaded {len(self.treatments)} treatments")

        except Exception as e:
            logger.warning(f"⚠️ Could not load some medical data: {e}")

    async def _test_ai_models(self):
        """Test AI model connectivity"""

        try:
            import aiohttp

            # Test Ollama connection
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_url}/api/tags") as response:
                    if response.status == 200:
                        models = await response.json()
                        logger.info(f"✅ Ollama connected, {len(models.get('models', []))} models available")
                    else:
                        logger.warning(f"⚠️ Ollama connection issue: {response.status}")

        except Exception as e:
            logger.warning(f"⚠️ AI model testing failed: {e}")

    async def _detect_and_translate(self, message: str) -> Dict[str, Any]:
        """Detect language and translate if needed"""

        # Simple Thai detection
        thai_chars = sum(1 for char in message if '\u0e00' <= char <= '\u0e7f')
        total_chars = len([c for c in message if c.isalpha()])

        if total_chars > 0 and thai_chars / total_chars > 0.3:
            detected_language = "thai"
            # Translate to English for processing
            translated_message = await self._translate_to_english(message)
        else:
            detected_language = "english"
            translated_message = message

        return {
            "detected_language": detected_language,
            "translated_message": translated_message,
            "original_message": message,
            "translation_used": detected_language == "thai"
        }

    async def _translate_to_english(self, thai_text: str) -> str:
        """Translate Thai text to English using SeaLLM"""

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.ollama_url}/api/generate", json={
                    "model": self.seallm_model,
                    "prompt": f"Translate this Thai medical text to English: {thai_text}",
                    "stream": False
                }) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", thai_text)
                    else:
                        logger.warning("Translation failed, using original text")
                        return thai_text

        except Exception as e:
            logger.warning(f"Translation error: {e}")
            return thai_text

    async def _translate_to_thai(self, english_text: str) -> str:
        """Translate English text to Thai using SeaLLM"""

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.ollama_url}/api/generate", json={
                    "model": self.seallm_model,
                    "prompt": f"Translate this English medical text to Thai: {english_text}",
                    "stream": False
                }) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", english_text)
                    else:
                        logger.warning("Translation failed, using original text")
                        return english_text

        except Exception as e:
            logger.warning(f"Translation error: {e}")
            return english_text

    async def _handle_emergency_response(
        self,
        emergency_result: Dict,
        language_info: Dict,
        include_reasoning: bool
    ) -> Dict[str, Any]:
        """Handle emergency response"""

        emergency_message = (
            "🚨 ตรวจพบสถานการณ์ฉุกเฉิน กรุณาติดต่อหน่วยการแพทย์ฉุกเฉินทันที\n\n"
            "โทร: 1669 (บริการการแพทย์ฉุกเฉิน)\n"
            "หรือไปโรงพยาบาลใกล้เคียงทันที"
        )

        recommendations = [
            "โทร 1669 เพื่อขอความช่วยเหลือฉุกเฉิน",
            "ไปโรงพยาบาลใกล้เคียงทันที",
            "อย่าขับรถไปเอง ให้คนอื่นพาไป",
            "เตรียมข้อมูลผู้ป่วยและยาที่กินประจำ"
        ]

        self.system_stats["emergency_cases"] += 1

        response = {
            "type": "emergency",
            "message": emergency_message,
            "recommendations": recommendations,
            "warning": "สถานการณ์ฉุกเฉิน - ต้องการความช่วยเหลือทางการแพทย์ทันที",
            "emergency_keywords": emergency_result["keywords"],
            "detected_dialects": emergency_result["dialects"]
        }

        if include_reasoning:
            response["agent_reasoning_chain"] = [
                AgentThought(
                    agent="EmergencyDetector",
                    step=1,
                    reasoning=f"Detected emergency keywords: {emergency_result['keywords']}",
                    action="Escalate to emergency protocols",
                    observation="Emergency situation confirmed",
                    confidence=emergency_result["confidence"] * 100
                ).__dict__
            ]

        return response

    async def _store_conversation(self, session_id: str, message: str, response: Dict):
        """Store conversation in memory"""

        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []

        conversation_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_message": message,
            "ai_response": response.get("message", ""),
            "type": response.get("type", ""),
            "metadata": {
                "agents_used": len(response.get("agent_reasoning_chain", [])),
                "emergency_detected": response.get("type") == "emergency"
            }
        }

        self.conversation_history[session_id].append(conversation_entry)

        # Keep only last 50 messages per session
        if len(self.conversation_history[session_id]) > 50:
            self.conversation_history[session_id] = self.conversation_history[session_id][-50:]

    def _update_stats(self, processing_time: float, language: str):
        """Update system statistics"""

        self.system_stats["total_consultations"] += 1

        # Update average processing time
        current_avg = self.system_stats["avg_response_time"]
        total_consultations = self.system_stats["total_consultations"]
        self.system_stats["avg_response_time"] = (
            (current_avg * (total_consultations - 1) + processing_time) / total_consultations
        )

        # Update language statistics
        if language not in self.system_stats["languages_detected"]:
            self.system_stats["languages_detected"][language] = 0
        self.system_stats["languages_detected"][language] += 1


# Agent Classes

class BaseAgent:
    """Base class for medical AI agents"""

    def __init__(self):
        self.name = ""
        self.role = ""
        self.tools = {}
        self.memory = []
        self.medical_service = None

    async def initialize(self, medical_service: MedicalAIService):
        """Initialize agent with medical service reference"""
        self.medical_service = medical_service

    async def think(self, input_data: Dict) -> List[AgentThought]:
        """Agent reasoning process"""
        raise NotImplementedError

    async def act(self, thought: AgentThought, context: Dict) -> Dict[str, Any]:
        """Execute agent action"""
        raise NotImplementedError


class TriageAgent(BaseAgent):
    """Agent specialized in medical triage and urgency assessment"""

    def __init__(self):
        super().__init__()
        self.name = "TriageAgent"
        self.role = "Medical triage and urgency assessment"

    async def assess_urgency(self, case_data: Dict) -> Dict[str, Any]:
        """Assess medical urgency and triage level"""

        message = case_data.get("message", "")
        patient_info = case_data.get("patient_info")

        # Calculate urgency score
        urgency_score = 0
        risk_factors = []

        # Age factor (more sensitive scoring)
        if patient_info and patient_info.age:
            if patient_info.age > 65:
                urgency_score += 10
                risk_factors.append("Advanced age")
            elif patient_info.age < 5:
                urgency_score += 15
                risk_factors.append("Young child")
            elif patient_info.age < 2:
                urgency_score += 25
                risk_factors.append("Infant")

        # Note: Vital signs assessment removed - inappropriate for consultation scope

        # Enhanced symptom severity keywords
        emergency_keywords = [
            "chest pain", "ปวดหน้าอก", "เจ็บหน้าอก", "แน่นหน้าอก",
            "can't breathe", "หายใจไม่ออก", "หายใจไม่อิ่ม", "หายใจลำบาก",
            "unconscious", "หมดสติ", "stroke", "โรคหลอดเลือดสมอง"
        ]

        high_severity_keywords = [
            "severe", "intense", "unbearable", "รุนแรง", "เจ็บมาก", "ปวดมาก",
            "เหงื่อออก", "หน้าซีด", "ผื่นลมพิษ", "ริมฝีปากบวม"
        ]

        pediatric_keywords = [
            "ลูก", "เด็ก", "child", "ซึม", "กินข้าวไม่ได้", "ผื่นแดง"
        ]

        for keyword in emergency_keywords:
            if keyword.lower() in message.lower() or keyword in message:
                urgency_score += 40
                risk_factors.append(f"Emergency symptom: {keyword}")

        for keyword in high_severity_keywords:
            if keyword.lower() in message.lower() or keyword in message:
                urgency_score += 25
                risk_factors.append(f"High severity symptom: {keyword}")

        for keyword in pediatric_keywords:
            if keyword.lower() in message.lower() or keyword in message:
                urgency_score += 15
                risk_factors.append(f"Pediatric concern: {keyword}")

        # Specific medical conditions
        diabetes_keywords = ["ปัสสาวะบ่อย", "กระหายน้ำ", "น้ำหนักลด", "ตาพร่ามัว"]
        allergy_keywords = ["ผื่นลมพิษ", "คันมาก", "ริมฝีปากบวม", "allergic"]
        gastric_keywords = ["ปวดท้อง", "แสบร้อน", "คลื่นไส้", "อาเจียน"]
        migraine_keywords = ["ปวดศีรษะ", "ไมเกรน", "กลัวแสง", "กลัวเสียง"]

        condition_found = False
        for keywords in [diabetes_keywords, allergy_keywords, gastric_keywords, migraine_keywords]:
            matches = sum(1 for kw in keywords if kw in message)
            if matches >= 2:
                urgency_score += 20
                risk_factors.append(f"Multiple symptoms suggesting specific condition")
                condition_found = True
                break

        # Determine triage level (lowered thresholds)
        if urgency_score >= 50:
            triage_level = TriageLevel.RESUSCITATION
            urgency = UrgencyLevel.CRITICAL
        elif urgency_score >= 35:
            triage_level = TriageLevel.EMERGENCY
            urgency = UrgencyLevel.HIGH
        elif urgency_score >= 25:
            triage_level = TriageLevel.URGENT
            urgency = UrgencyLevel.HIGH
        elif urgency_score >= 15:
            triage_level = TriageLevel.SEMI_URGENT
            urgency = UrgencyLevel.MEDIUM
        else:
            triage_level = TriageLevel.NON_URGENT
            urgency = UrgencyLevel.LOW

        return {
            "triage_level": triage_level.value,
            "urgency": urgency.value,
            "emergency_detected": urgency_score >= 60,
            "risk_score": min(urgency_score, 100),
            "risk_factors": risk_factors,
            "recommendation": " / ".join(self._get_triage_recommendations(triage_level)),
            "reasoning": f"Urgency score: {urgency_score}/100 based on symptoms, vitals, and demographics"
        }

    def _get_triage_recommendations(self, triage_level: TriageLevel) -> List[str]:
        """Get recommendations based on triage level"""

        recommendations = {
            TriageLevel.RESUSCITATION: [
                "โทร 1669 ทันที",
                "ไปโรงพยาบาลด่วนที่สุด",
                "เตรียมทำ CPR หากจำเป็น"
            ],
            TriageLevel.EMERGENCY: [
                "ไปโรงพยาบาลทันที",
                "อย่าชักช้า",
                "โทรแจ้งโรงพยาบาลก่อนไป"
            ],
            TriageLevel.URGENT: [
                "ควรไปโรงพยาบาลภายใน 30 นาที",
                "สังเกตอาการอย่างใกล้ชิด"
            ],
            TriageLevel.SEMI_URGENT: [
                "ไปพบแพทย์ภายใน 1 ชั่วโมง",
                "สามารถรอได้แต่ควรติดตาม"
            ],
            TriageLevel.NON_URGENT: [
                "สามารถพบแพทย์ภายใน 2 ชั่วโมง",
                "ดูแลตัวเองที่บ้านได้"
            ]
        }

        return recommendations.get(triage_level, ["ปรึกษาแพทย์"])


class DiagnosticAgent(BaseAgent):
    """Agent specialized in medical diagnosis and differential diagnosis"""

    def __init__(self):
        super().__init__()
        self.name = "DiagnosticAgent"
        self.role = "Medical diagnosis and analysis"

    async def analyze_symptoms(self, case_data: Dict) -> Dict[str, Any]:
        """Analyze symptoms and provide differential diagnosis"""

        message = case_data.get("message", "")

        # Search medical knowledge base for matching conditions
        matching_diagnoses = self._search_diagnoses(message)

        # Calculate confidence for each diagnosis
        primary_diagnosis = None
        differential_diagnoses = []

        if matching_diagnoses:
            primary_diagnosis = matching_diagnoses[0]
            differential_diagnoses = matching_diagnoses[1:5]  # Top 5 alternatives

        return {
            "primary_diagnosis": primary_diagnosis,
            "differential_diagnoses": differential_diagnoses,
            "confidence": 75.0 if primary_diagnosis else 0.0,
            "reasoning": f"Based on symptom analysis of: {message[:100]}..."
        }

    def _search_diagnoses(self, symptoms: str) -> List[Dict]:
        """Search diagnosis database for matching conditions with enhanced logic"""

        # Enhanced symptom-to-diagnosis mapping when no database available
        symptom_diagnosis_map = {
            "common_cold": {
                "keywords": ["ไข้", "คัดจมูก", "น้ำมูกใส", "ไอ", "เจ็บคอ", "ปวดเมื่อย", "fever", "runny nose", "cough"],
                "icd_code": "J00", "thai_name": "ไข้หวัด", "english_name": "Common Cold"
            },
            "diabetes": {
                "keywords": ["ปัสสาวะบ่อย", "กระหายน้ำ", "น้ำหนักลด", "อ่อนเพลีย", "ตาพร่ามัว", "frequent urination", "thirst", "weight loss"],
                "icd_code": "E11", "thai_name": "เบาหวาน", "english_name": "Diabetes Mellitus"
            },
            "chest_pain": {
                "keywords": ["ปวดหน้าอก", "เจ็บหน้าอก", "แน่นหน้าอก", "chest pain", "เหงื่อออก", "หน้าซีด"],
                "icd_code": "R06.02", "thai_name": "อาการปวดหน้าอก", "english_name": "Chest Pain"
            },
            "allergic_reaction": {
                "keywords": ["ผื่นลมพิษ", "คันมาก", "ริมฝีปากบวม", "หายใจลำบาก", "allergic", "hives", "swelling"],
                "icd_code": "T78.40", "thai_name": "ปฏิกิริยาแพ้", "english_name": "Allergic Reaction"
            },
            "gastritis": {
                "keywords": ["ปวดท้อง", "แสบร้อน", "คลื่นไส้", "อาเจียน", "ลิ้นปี่", "gastric", "nausea", "vomiting"],
                "icd_code": "K29.70", "thai_name": "กระเพาะอักเสบ", "english_name": "Gastritis"
            },
            "migraine": {
                "keywords": ["ปวดศีรษะ", "ไมเกรน", "คลื่นไส้", "กลัวแสง", "กลัวเสียง", "migraine", "headache"],
                "icd_code": "G43.909", "thai_name": "ไมเกรน", "english_name": "Migraine"
            },
            "uti": {
                "keywords": ["ปัสสาวะแสบ", "ปัสสาวะบ่อย", "ปวดท้องน้อย", "urinary", "burning", "frequent"],
                "icd_code": "N39.0", "thai_name": "ติดเชื้อทางเดินปัสสาวะ", "english_name": "Urinary Tract Infection"
            },
            "fever": {
                "keywords": ["ไข้สูง", "ไข้", "39 องศา", "fever", "temperature", "ซึม", "ผื่นแดง"],
                "icd_code": "R50.9", "thai_name": "ไข้", "english_name": "Fever"
            }
        }

        matches = []
        symptoms_lower = symptoms.lower()

        # Check database first if available
        if self.medical_service and self.medical_service.diagnoses:
            for diagnosis in self.medical_service.diagnoses:
                score = 0

                # Check name matches
                if diagnosis.name_en and diagnosis.name_en.lower() in symptoms_lower:
                    score += 40
                if diagnosis.name_th and diagnosis.name_th in symptoms:
                    score += 40

                # Check description matches
                if diagnosis.description:
                    description_words = diagnosis.description.lower().split()
                    symptom_words = symptoms_lower.split()
                    common_words = set(description_words) & set(symptom_words)
                    score += len(common_words) * 3

                if score > 15:
                    matches.append({
                        "icd_code": diagnosis.icd_code or "R69",
                        "english_name": diagnosis.name_en or "Unknown condition",
                        "thai_name": diagnosis.name_th or "ไม่ทราบโรค",
                        "confidence": min(score, 100),
                        "category": diagnosis.category or "General"
                    })

        # Use symptom mapping as fallback
        if not matches:
            for condition, data in symptom_diagnosis_map.items():
                score = 0
                matched_keywords = []

                for keyword in data["keywords"]:
                    if keyword.lower() in symptoms_lower or keyword in symptoms:
                        score += 15
                        matched_keywords.append(keyword)

                if score > 20:  # At least 2 keywords matched
                    matches.append({
                        "icd_code": data["icd_code"],
                        "english_name": data["english_name"],
                        "thai_name": data["thai_name"],
                        "confidence": min(score, 95),
                        "category": "Clinical Assessment",
                        "matched_keywords": matched_keywords
                    })

        return sorted(matches, key=lambda x: x["confidence"], reverse=True)


class TreatmentAgent(BaseAgent):
    """Agent specialized in treatment planning and medication recommendations"""

    def __init__(self):
        super().__init__()
        self.name = "TreatmentAgent"
        self.role = "Treatment planning and medication recommendations"

    async def recommend_treatment(self, case_data: Dict) -> Dict[str, Any]:
        """Recommend treatment plan based on diagnosis"""

        diagnosis = case_data.get("diagnosis")
        patient_info = case_data.get("patient_info")

        medications = self._recommend_medications(diagnosis, patient_info)
        lifestyle_recommendations = self._get_lifestyle_recommendations(diagnosis)

        return {
            "medications": medications,
            "lifestyle_recommendations": lifestyle_recommendations,
            "follow_up_instructions": "ติดตามอาการและกลับมาพบแพทย์หากอาการไม่ดีขึ้น",
            "safety_warnings": [
                "ไม่ควรใช้ยานี้หากแพ้ส่วนประกอบ",
                "หากมีอาการข้างเคียง ให้หยุดยาและปรึกษาแพทย์"
            ],
            "contraindications": []
        }

    def _recommend_medications(self, diagnosis: Optional[Dict], patient_info: Optional[PatientInfo]) -> List[Dict]:
        """Recommend appropriate medications"""

        if not diagnosis:
            return []

        # Basic pain relief for common conditions
        medications = []

        # General pain and fever medication
        medications.append({
            "english_name": "Paracetamol",
            "thai_name": "พาราเซตามอล",
            "dosage": "500mg",
            "instructions": "รับประทานทุก 6 ชั่วโมง หลังอาหาร",
            "category": "pain_relief"
        })

        return medications

    def _get_lifestyle_recommendations(self, diagnosis: Optional[Dict]) -> List[str]:
        """Get lifestyle recommendations"""

        return [
            "พักผ่อนให้เพียงพอ",
            "ดื่มน้ำให้มาก",
            "หลีกเลี่ยงการออกแรงหนัก",
            "รับประทานอาหารที่มีประโยชน์"
        ]


class CoordinatorAgent(BaseAgent):
    """Agent that coordinates multiple agents for comprehensive medical analysis"""

    def __init__(self):
        super().__init__()
        self.name = "CoordinatorAgent"
        self.role = "Multi-agent coordination and workflow management"

    async def process_medical_case(self, case_data: Dict) -> Dict[str, Any]:
        """Coordinate multiple agents to process medical case"""

        reasoning_chain = []

        # Step 1: Triage assessment
        triage_agent = self.medical_service.agents["triage"]
        triage_result = await triage_agent.assess_urgency(case_data)

        reasoning_chain.append(AgentThought(
            agent="TriageAgent",
            step=1,
            reasoning=f"Assessed urgency based on symptoms and vitals",
            action="Calculate triage score and urgency level",
            observation=f"Triage level: {triage_result['triage_level']}, Risk score: {triage_result['risk_score']}",
            confidence=85.0
        ).__dict__)

        # Step 2: Diagnostic analysis
        diagnostic_agent = self.medical_service.agents["diagnostic"]
        diagnosis_result = await diagnostic_agent.analyze_symptoms(case_data)

        reasoning_chain.append(AgentThought(
            agent="DiagnosticAgent",
            step=2,
            reasoning="Analyzed symptoms against medical knowledge base",
            action="Search for matching diagnoses and calculate confidence",
            observation=f"Primary diagnosis confidence: {diagnosis_result['confidence']}%",
            confidence=diagnosis_result['confidence']
        ).__dict__)

        # Step 3: Treatment planning
        treatment_agent = self.medical_service.agents["treatment"]
        treatment_data = {**case_data, "diagnosis": diagnosis_result.get("primary_diagnosis")}
        treatment_result = await treatment_agent.recommend_treatment(treatment_data)

        reasoning_chain.append(AgentThought(
            agent="TreatmentAgent",
            step=3,
            reasoning="Developed treatment plan based on diagnosis and patient profile",
            action="Recommend medications and lifestyle changes",
            observation=f"Recommended {len(treatment_result['medications'])} medications",
            confidence=70.0
        ).__dict__)

        # Generate comprehensive response
        response_message = self._generate_response_message(
            triage_result, diagnosis_result, treatment_result, case_data
        )

        return {
            "type": "comprehensive_analysis",
            "message": response_message,
            "triage": triage_result,
            "diagnosis": diagnosis_result,
            "treatment": treatment_result,
            "agent_reasoning_chain": reasoning_chain,
            "rag_results_count": len(diagnosis_result.get("differential_diagnoses", [])),
            "recommendation": "กรุณาปรึกษาแพทย์เพื่อการวินิจฉัยและการรักษาที่เหมาะสม"
        }

    def _generate_response_message(self, triage: Dict, diagnosis: Dict, treatment: Dict, case_data: Dict) -> str:
        """Generate comprehensive response message"""

        message_parts = []

        # Triage information
        urgency = triage.get("urgency", "medium")
        if urgency in ["critical", "high"]:
            message_parts.append("🚨 อาการของคุณต้องการความสนใจทางการแพทย์")
        else:
            message_parts.append("📋 ผลการประเมินเบื้องต้น")

        # Diagnosis information
        primary_diagnosis = diagnosis.get("primary_diagnosis")
        if primary_diagnosis:
            message_parts.append(f"🩺 การวินิจฉัยเบื้องต้น: {primary_diagnosis.get('thai_name', 'ไม่ระบุ')}")
            confidence = primary_diagnosis.get("confidence", 0)
            message_parts.append(f"📊 ความเชื่อมั่น: {confidence:.0f}%")

        # Treatment recommendations
        medications = treatment.get("medications", [])
        if medications:
            message_parts.append("💊 คำแนะนำการรักษาเบื้องต้น:")
            for med in medications[:2]:  # Show only first 2 medications
                message_parts.append(f"- {med.get('thai_name', med.get('english_name', ''))}")

        # Lifestyle recommendations
        lifestyle = treatment.get("lifestyle_recommendations", [])
        if lifestyle:
            message_parts.append("🏃‍♂️ คำแนะนำการดูแลตัวเอง:")
            for rec in lifestyle[:3]:  # Show only first 3 recommendations
                message_parts.append(f"- {rec}")

        # Medical disclaimer
        message_parts.append("\n⚠️ ข้อมูลนี้เป็นเพียงคำแนะนำเบื้องต้น กรุณาปรึกษาแพทย์เพื่อการวินิจฉัยและการรักษาที่เหมาะสม")

        return "\n".join(message_parts)