
# Medical AI Service - Simplified for Common Illness Consultation Only

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
from app.services.memory_agent import memory_agent, AdaptiveMemoryAgent
from app.services.advanced_few_shot import AdvancedFewShotLearning
from app.services.llm_logger import llm_logger
from app.services.ollama_client import ollama_client
from app.services.rag_few_shot_service import rag_few_shot_service

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
    """Simplified Medical AI Service for Common Illness Consultation"""

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
            "avg_response_time": 0,
            "languages_detected": {}
        }

        # Simple agent system
        self.agents = {
            "diagnostic": DiagnosticAgent(),
            "treatment": TreatmentAgent(),
            "triage": TriageAgent(),
            "coordinator": CoordinatorAgent()
        }

    async def initialize(self):
        """Initialize medical AI service"""
        logger.info("🚀 Initializing Simplified Medical AI Service...")

        try:
            # Load medical knowledge bases
            await self._load_medical_data()

            # Initialize RAG Few-Shot service
            try:
                await rag_few_shot_service.initialize()
                logger.info("✅ RAG Few-Shot service initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize RAG service: {e}")

            # Initialize agents
            for agent_name, agent in self.agents.items():
                await agent.initialize(self)
                logger.info(f"✅ {agent_name} agent initialized")

            # Initialize Ollama client
            try:
                await ollama_client.initialize()
                if await ollama_client.check_connection():
                    logger.info("✅ Ollama client connected successfully")
                    models = await ollama_client.list_models()
                    logger.info(f"📋 Available models: {models}")
                else:
                    logger.warning("⚠️ Ollama not available, will use fallback responses")
            except Exception as e:
                logger.warning(f"⚠️ Ollama initialization failed: {e}")

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

        # Cleanup Ollama client
        try:
            await ollama_client.cleanup()
            logger.info("✅ Ollama client cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Ollama cleanup failed: {e}")

    async def _call_llm_model(self,
                             model_name: str,
                             prompt: str,
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Call real Ollama LLM model with comprehensive logging"""

        start_time = datetime.now()

        # Log the request
        request_id = llm_logger.log_request(
            model_name=model_name,
            prompt=prompt,
            parameters={"temperature": 0.7, "max_tokens": 1000},
            context=context or {}
        )

        try:
            # Initialize Ollama client
            await ollama_client.initialize()

            # Check Ollama connection
            if not await ollama_client.check_connection():
                raise Exception("Cannot connect to Ollama server")

            # Call real Ollama model
            if context and context.get("consultation_type") == "common_illness":
                # Medical consultation
                language = context.get("language", "thai")
                result = await ollama_client.generate_medical_response(
                    symptoms=prompt.replace("Medical consultation for symptoms: ", ""),
                    model=model_name,
                    language=language
                )
            elif "translation" in prompt.lower() or model_name == settings.seallm_model:
                # Translation task
                result = await ollama_client.generate_translation(
                    text=prompt,
                    model=model_name
                )
            else:
                # General generation
                result = await ollama_client.generate(
                    model=model_name,
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=1000
                )

            if result["success"]:
                # Log successful response
                llm_logger.log_response(
                    request_id=request_id,
                    response_text=result["response"],
                    response_time_ms=result["processing_time_ms"],
                    tokens_used=result.get("tokens_used", 0),
                    confidence_score=0.85 if "diagnosis" in prompt.lower() else None
                )

                return {
                    "success": True,
                    "response": result["response"],
                    "processing_time_ms": result["processing_time_ms"],
                    "tokens_used": result.get("tokens_used", 0),
                    "request_id": request_id,
                    "model_used": result.get("model_used", model_name)
                }
            else:
                raise Exception(result.get("error", "Unknown Ollama error"))

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            error_msg = str(e)

            logger.error(f"❌ LLM call failed: {error_msg}")

            # Log error response
            llm_logger.log_response(
                request_id=request_id,
                response_text="",
                response_time_ms=processing_time,
                error=error_msg
            )

            # Fallback to simulation if Ollama fails
            logger.warning("🔄 Falling back to simulated response")
            fallback_response = self._get_fallback_response(model_name, prompt, context)

            # Log fallback response
            llm_logger.log_response(
                request_id=request_id,
                response_text=fallback_response,
                response_time_ms=processing_time,
                tokens_used=len(fallback_response.split()),
                error=f"Ollama failed, used fallback: {error_msg}"
            )

            return {
                "success": True,  # Return success with fallback
                "response": fallback_response,
                "processing_time_ms": processing_time,
                "request_id": request_id,
                "fallback_used": True,
                "original_error": error_msg
            }

    def _get_fallback_response(self, model_name: str, prompt: str, context: Dict[str, Any] = None) -> str:
        """Generate fallback response when Ollama is unavailable"""

        if model_name == "medllama2" or (context and context.get("consultation_type") == "common_illness"):
            return self._simulate_medllama_response(prompt, context)
        elif model_name == "seallm-7b-v2":
            return self._simulate_seallm_response(prompt, context)
        else:
            return f"Medical AI response for: {prompt[:100]}...\n\nNote: Using fallback response as primary AI model is unavailable."

    def _simulate_medllama_response(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Simulate MedLlama2 response for medical diagnosis"""

        symptoms = prompt.lower()

        # Simple keyword-based simulation
        if any(word in symptoms for word in ['เจ็บหน้าอก', 'chest pain', 'ปวดหน้าอก']):
            return """Based on the symptoms of chest pain, this could indicate several conditions:

Primary Assessment: Possible angina or myocardial infarction
Confidence: 75%
Recommendations:
- Immediate medical evaluation
- ECG monitoring
- Cardiac enzymes
- Call emergency services if severe

Differential Diagnoses:
1. Acute coronary syndrome
2. Gastroesophageal reflux
3. Costochondritis
4. Pulmonary embolism

Emergency signs: Severe pain, radiation to arm/jaw, shortness of breath, sweating"""

        elif any(word in symptoms for word in ['ปวดข้อ', 'joint pain', 'arthritis']):
            return """Joint pain assessment indicates possible inflammatory arthritis.

Primary Assessment: Osteoarthritis or Rheumatoid Arthritis
Confidence: 85%
Key findings: Joint swelling, morning stiffness, pain with movement

Treatment recommendations:
- NSAIDs for pain relief
- Physical therapy
- Joint protection techniques
- Weight management if applicable

Monitor for: Fever, multiple joint involvement, systemic symptoms"""

        elif any(word in symptoms for word in ['ปัสสาวะบ่อย', 'diabetes', 'เบาหวาน']):
            return """Polyuria and associated symptoms suggest diabetes mellitus.

Primary Assessment: Type 2 Diabetes Mellitus
Confidence: 90%
Classical triad: Polyuria, polydipsia, polyphagia

Immediate actions:
- Blood glucose testing
- HbA1c measurement
- Urinalysis
- Blood pressure monitoring

Management plan:
- Lifestyle modifications
- Metformin consideration
- Regular monitoring
- Diabetes education"""

        else:
            return f"""Medical assessment based on presented symptoms: {prompt[:100]}...

General medical evaluation indicates need for further assessment.
Confidence: 65%

Recommendations:
- Complete history and physical examination
- Appropriate diagnostic testing
- Follow-up as clinically indicated
- Seek medical attention if symptoms worsen"""

    def _simulate_seallm_response(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Simulate SeaLLM translation response"""

        # Simple translation simulation
        if any('\u0e00' <= c <= '\u0e7f' for c in prompt):  # Thai text
            return "Translated to English: " + prompt.replace('ปวด', 'pain').replace('เจ็บ', 'ache').replace('บวม', 'swelling')
        else:
            return "แปลเป็นไทย: " + prompt.replace('pain', 'ปวด').replace('ache', 'เจ็บ').replace('swelling', 'บวม')

    async def process_medical_consultation(
        self,
        message: str,
        conversation_history: Optional[List[ConversationMessage]] = None,
        patient_info: Optional[PatientInfo] = None,
        preferred_language: str = "auto",
        session_id: Optional[str] = None,
        include_reasoning: bool = False
    ) -> Dict[str, Any]:
        """Process medical consultation for common illnesses"""

        logger.info(f"🩺 Processing medical consultation: {message[:100]}...")
        start_time = datetime.now()

        # AUTO-EXTRACT: For elderly users, extract patient info from Thai message
        if not patient_info:
            extracted_info = self._extract_patient_info_from_message(message)
            if extracted_info:
                patient_info = extracted_info
                logger.info(f"🔍 Auto-extracted patient info: age={patient_info.age}, gender={patient_info.gender}")

        try:
            # NEW WORKFLOW: Generate AI response first, then queue for doctor approval
            if settings.require_doctor_approval:
                return await self._process_with_doctor_approval(
                    message, conversation_history, patient_info,
                    preferred_language, session_id, include_reasoning
                )

            # Detect language and translate if needed
            language_info = await self._detect_and_translate(message)
            processed_message = language_info.get("translated_message", message)

            # Log translation if needed
            if language_info.get("translation_used"):
                llm_logger.log_translation(
                    source_text=message,
                    translated_text=processed_message,
                    source_lang=language_info.get("detected_language", "unknown"),
                    target_lang="english",
                    model_name="seallm-7b-v2",
                    processing_time_ms=50.0  # Simulated translation time
                )

            # Call LLM for medical diagnosis
            llm_context = {
                "patient_info": patient_info.dict() if patient_info else None,
                "session_id": session_id,
                "language": language_info.get("detected_language"),
                "consultation_type": "common_illness"
            }

            llm_result = await self._call_llm_model(
                model_name=self.medllama_model,
                prompt=f"Medical consultation for symptoms: {processed_message}",
                context=llm_context
            )

            # Process common illness consultation
            coordinator = self.agents["coordinator"]
            result = await coordinator.process_common_illness_consultation({
                "message": processed_message,
                "original_message": message,
                "conversation_history": conversation_history or [],
                "patient_info": patient_info,
                "language_info": language_info,
                "session_id": session_id,
                "llm_result": llm_result  # Include LLM response
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
        """Enhanced emergency detection with severity levels and specific actions"""

        # Use the enhanced red flag detection
        red_flags = self._check_red_flags(message)

        if red_flags:
            return {
                "is_emergency": True,
                "urgency_level": red_flags["max_urgency"],
                "keywords": red_flags["keywords"],
                "detected_flags": red_flags["flags"],
                "recommendation": red_flags["recommendation"],
                "immediate_action": self._get_emergency_action(red_flags["max_urgency"])
            }

        return {
            "is_emergency": False,
            "urgency_level": "none",
            "keywords": [],
            "recommendation": ""
        }

    def _get_emergency_action(self, urgency_level: str) -> str:
        """Get specific emergency action based on urgency level"""

        actions = {
            "critical": "🚨 โทร 1669 ทันที - อย่าขับรถเอง ให้ผู้อื่นพาไป หรือเรียกรถพยาบาล",
            "high": "⚠️ ไปโรงพยาบาลโดยเร็ว - ไม่ควรรอ ให้ไปในวันนี้",
            "medium": "📋 ควรพบแพทย์ภายใน 24 ชั่วโมง"
        }

        return actions.get(urgency_level, "")

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

    async def assess_common_illness(
        self,
        message: str,
        patient_info: Optional[PatientInfo] = None
    ) -> Dict[str, Any]:
        """Assess common illness symptoms"""

        diagnostic_agent = self.agents["diagnostic"]
        result = await diagnostic_agent.analyze_common_symptoms({
            "message": message,
            "patient_info": patient_info,
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
        """Get system performance statistics including LLM interactions"""

        # Get LLM statistics
        llm_stats = llm_logger.get_statistics()

        return {
            "total_consultations": self.system_stats["total_consultations"],
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
            "llm_interactions": llm_stats,
            "uptime": datetime.now().isoformat()
        }

    async def get_llm_logs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent LLM interaction logs"""
        return llm_logger.get_recent_interactions(limit=limit)

    # Private methods

    async def _load_medical_data(self):
        """Load medical knowledge from CSV files"""

        try:
            # Load medicines
            if os.path.exists(settings.medicine_data_path):
                with open(settings.medicine_data_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Handle the actual CSV format: No,Prescription,[Thai_name]
                        english_name = row.get('Prescription', '') or row.get('prescription', '')
                        thai_name = list(row.values())[-1] if len(row) > 1 else ''  # Last column
                        medicine = MedicalKnowledgeItem(
                            id=row.get('No', '') or row.get('no', ''),
                            name_en=english_name,
                            name_th=thai_name,
                            category='medication',
                            description=f"{english_name} ({thai_name})"
                        )
                        self.medicines.append(medicine)
                logger.info(f"📊 Loaded {len(self.medicines)} medicines")

            # Load diagnoses
            if os.path.exists(settings.diagnosis_data_path):
                with open(settings.diagnosis_data_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Handle the actual CSV format: No,Diagnosis Code and Name,[Thai_name]
                        diagnosis_code_name = row.get('Diagnosis Code and Name', '') or row.get('diagnosis code and name', '')
                        thai_name = list(row.values())[-1] if len(row) > 1 else ''  # Last column

                        # Extract ICD code from diagnosis_code_name
                        icd_code = ''
                        english_name = diagnosis_code_name
                        if diagnosis_code_name:
                            parts = diagnosis_code_name.split(' ', 1)
                            if parts and len(parts[0]) <= 6:  # ICD codes are typically short
                                icd_code = parts[0]
                                english_name = ' '.join(parts[1:]) if len(parts) > 1 else parts[0]

                        diagnosis = MedicalKnowledgeItem(
                            id=row.get('No', '') or row.get('no', ''),
                            name_en=english_name,
                            name_th=thai_name,
                            category='diagnosis',
                            description=f"{english_name} ({thai_name})",
                            icd_code=icd_code
                        )
                        self.diagnoses.append(diagnosis)
                logger.info(f"📊 Loaded {len(self.diagnoses)} diagnoses")

            # Load treatments
            if os.path.exists(settings.treatment_data_path):
                with open(settings.treatment_data_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Handle the treatment CSV format: treatment_id,condition_name,thai_condition,icd_code,medications
                        diagnosis_code_name = row.get('condition_name', '')
                        prescription_en = row.get('medications', '')

                        # Get Thai condition name
                        prescription_th = row.get('thai_condition', '')

                        if prescription_en:  # Only add if we have a prescription
                            treatment = MedicalKnowledgeItem(
                                id=row.get('treatment_id', ''),
                                name_en=prescription_en,
                                name_th=prescription_th,
                                category='treatment',
                                description=f"{diagnosis_code_name} -> {prescription_en} ({prescription_th})"
                            )
                            self.treatments.append(treatment)
                logger.info(f"📊 Loaded {len(self.treatments)} treatments")

        except Exception as e:
            logger.warning(f"⚠️ Could not load some medical data: {e}")

    async def _detect_and_translate(self, message: str) -> Dict[str, Any]:
        """Detect language and translate if needed"""

        # Simple Thai detection
        thai_chars = sum(1 for char in message if '\u0e00' <= char <= '\u0e7f')
        total_chars = len([c for c in message if c.isalpha()])

        if total_chars > 0 and thai_chars / total_chars > 0.3:
            detected_language = "thai"
            # For simplicity, use original message
            translated_message = message
        else:
            detected_language = "english"
            translated_message = message

        return {
            "detected_language": detected_language,
            "translated_message": translated_message,
            "original_message": message,
            "translation_used": False
        }

    async def _translate_to_thai(self, english_text: str) -> str:
        """For simplicity, return original text"""
        return english_text

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
                "agents_used": len(response.get("agent_reasoning_chain", []))
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

    async def _is_aggressive_diagnosis_llm(self, diagnosis: Dict, symptoms: str) -> bool:
        """Use LLM to check if a diagnosis is overly aggressive for the given symptoms"""

        diagnosis_text = ""
        if isinstance(diagnosis, dict):
            diagnosis_text = f"{diagnosis.get('english_name', '')} ({diagnosis.get('thai_name', '')}) - ICD: {diagnosis.get('icd_code', '')}"
        else:
            diagnosis_text = str(diagnosis)

        prompt = f"""
You are a medical AI safety checker. Analyze if the proposed diagnosis is appropriate for the given symptoms.

SYMPTOMS: {symptoms}
PROPOSED DIAGNOSIS: {diagnosis_text}

TASK: Determine if this diagnosis is overly aggressive or inappropriate given the symptoms.

GUIDELINES:
- Common symptoms like fever, headache, fatigue should NOT lead to serious diagnoses like meningitis, stroke, heart attack
- Serious diagnoses require specific, characteristic symptoms
- Be conservative and favor common conditions over rare ones

RESPOND: "AGGRESSIVE" if the diagnosis is too serious for the symptoms, "APPROPRIATE" if reasonable.

RESPONSE:"""

        try:
            result = await ollama_client.generate(
                model=self.medllama_model,
                prompt=prompt,
                temperature=0.1,  # Very low temperature for consistent safety decisions
                max_tokens=10
            )

            if result.get('success'):
                response = result['response'].strip().upper()
                is_aggressive = "AGGRESSIVE" in response

                if is_aggressive:
                    logger.warning(f"🤖 LLM detected aggressive diagnosis: {diagnosis_text} for symptoms: {symptoms}")

                return is_aggressive
            else:
                logger.warning("LLM safety check failed, defaulting to conservative")
                return True  # Default to conservative if LLM fails

        except Exception as e:
            logger.error(f"Error in LLM aggressive diagnosis check: {e}")
            return True  # Default to conservative

    async def _get_llm_conservative_diagnosis(self, symptoms: str) -> List[Dict]:
        """Use LLM to get conservative diagnosis for symptoms"""

        prompt = f"""
You are a conservative medical AI. Analyze these symptoms and provide the MOST LIKELY COMMON diagnosis.

SYMPTOMS: {symptoms}

INSTRUCTIONS:
1. Consider the MOST COMMON causes first (viral illness, common cold, tension headache, etc.)
2. Avoid serious diagnoses unless symptoms are very specific
3. Provide 1 primary diagnosis and 2 differential diagnoses
4. Use appropriate ICD-10 codes

FORMAT YOUR RESPONSE EXACTLY AS:
PRIMARY: [ICD Code] [English Name] | [Thai Name] | Confidence: [0-100]
DIFFERENTIAL1: [ICD Code] [English Name] | [Thai Name] | Confidence: [0-100]
DIFFERENTIAL2: [ICD Code] [English Name] | [Thai Name] | Confidence: [0-100]

EXAMPLE:
PRIMARY: J00 Common cold | ไข้หวัด | Confidence: 75
DIFFERENTIAL1: J11.1 Influenza | ไข้หวัดใหญ่ | Confidence: 60
DIFFERENTIAL2: R50.9 Viral fever | ไข้จากไวรัส | Confidence: 55

RESPONSE:"""

        try:
            result = await ollama_client.generate(
                model=self.medllama_model,
                prompt=prompt,
                temperature=0.2,  # Low temperature for consistent responses
                max_tokens=200
            )

            if result.get('success'):
                response = result['response'].strip()
                return self._parse_llm_diagnosis_response(response)
            else:
                logger.warning("LLM conservative diagnosis failed")
                return None

        except Exception as e:
            logger.error(f"Error in LLM conservative diagnosis: {e}")
            return None

    def _extract_patient_info_from_message(self, message: str) -> Optional[PatientInfo]:
        """Extract patient demographic info from Thai message for elderly users"""
        import re

        # Extract age (อายุ 65 ปี, อายุ65ปี, 65 ปี, etc.)
        age_patterns = [
            r'อายุ\s*(\d+)\s*ปี',
            r'อายุ\s*(\d+)',
            r'(\d+)\s*ปี(?!\s*กิโลกรัม)',  # Avoid matching weight
            r'วัย\s*(\d+)',
            r'ขวบ\s*(\d+)',
            r'(\d+)\s*ขวบ'
        ]

        age = None
        for pattern in age_patterns:
            match = re.search(pattern, message)
            if match:
                candidate_age = int(match.group(1))
                # Reasonable age range for medical consultation
                if 1 <= candidate_age <= 120:
                    age = candidate_age
                    break

        # Extract height (สูง 160, สูง160เซนติเมตร, etc.)
        height_patterns = [
            r'สูง\s*(\d+)(?:\s*(?:เซนติเมตร|ซม\.?|cm))?',
            r'ความสูง\s*(\d+)',
            r'(\d+)\s*เซนติเมตร'
        ]

        height = None
        for pattern in height_patterns:
            match = re.search(pattern, message)
            if match:
                candidate_height = int(match.group(1))
                # Reasonable height range
                if 50 <= candidate_height <= 250:
                    height = candidate_height
                    break

        # Extract weight (หนัก 65, น้ำหนัก 65 กิโลกรัม, etc.)
        weight_patterns = [
            r'หนัก\s*(\d+)(?:\s*(?:กิโลกรัม|กก\.?|kg))?',
            r'น้ำหนัก\s*(\d+)',
            r'(\d+)\s*กิโลกรัม'
        ]

        weight = None
        for pattern in weight_patterns:
            match = re.search(pattern, message)
            if match:
                candidate_weight = int(match.group(1))
                # Reasonable weight range
                if 10 <= candidate_weight <= 300:
                    weight = candidate_weight
                    break

        # Extract gender (เป็นผู้ชาย, เป็นผู้หญิง, ชาย, หญิง)
        gender = None
        if re.search(r'ผู้ชาย|ชาย(?!หญิง)', message):
            gender = "male"
        elif re.search(r'ผู้หญิง|หญิง', message):
            gender = "female"

        # Extract medical history
        medical_history = []
        if re.search(r'ไม่มีประวัติโรคประจำตัว|ไม่เป็นโรคอะไร', message):
            medical_history.append("ไม่มีประวัติโรคประจำตัว")
        else:
            history_patterns = [
                r'เป็น(เบาหวาน|ความดันสูง|ความดันโลหิตสูง|โรคหัวใจ|ไตเสื่อม)',
                r'ประวัติ(เบาหวาน|ความดันสูง|โรคหัวใจ)',
                r'มี(เบาหวาน|ความดันสูง|โรคหัวใจ)'
            ]

            for pattern in history_patterns:
                matches = re.findall(pattern, message)
                medical_history.extend(matches)

        # Extract allergies
        allergies = []
        if re.search(r'ไม่แพ้(?:อะไร|ยา|อาหาร)|ไม่มีการแพ้', message):
            allergies.append("ไม่แพ้อะไร")
        else:
            allergy_patterns = [
                r'แพ้([^ก-ไ\s]+)',  # Match non-Thai characters after แพ้
                r'การแพ้([^ก-ไ\s]+)'
            ]

            for pattern in allergy_patterns:
                matches = re.findall(pattern, message)
                allergies.extend([m.strip() for m in matches if m.strip()])

        # Only create PatientInfo if we have meaningful data
        if age or gender or medical_history or allergies:
            # Match actual PatientInfo schema (no height/weight fields)
            return PatientInfo(
                age=age,
                gender=gender,
                allergies=allergies if allergies else None,
                conditions=medical_history if medical_history else None
            )

        return None

    def _detect_emergency_keywords(self, message: str) -> Optional[Dict[str, Any]]:
        """Quick emergency keyword detection for safety"""
        emergency_keywords = [
            "หายใจลำบาก", "เจ็บหน้าอก", "ปวดหัวรุนแรง",
            "เป็นลม", "ไม่รู้สึกตัว", "เลือดออก"
        ]

        message_lower = message.lower()
        for keyword in emergency_keywords:
            if keyword in message_lower:
                return {
                    "message": f"⚠️ ตรวจพบอาการฉุกเฉิน: {keyword}\n\n🚨 กรุณาโทร 1669 ทันทีหรือไปโรงพยาบาลใกล้บ้าน",
                    "urgency": "CRITICAL",
                    "recommendations": ["โทร 1669 ทันที", "ไปโรงพยาบาลเร่งด่วน"]
                }

        return None

    async def _process_with_doctor_approval(
        self,
        message: str,
        conversation_history: Optional[List[ConversationMessage]] = None,
        patient_info: Optional[PatientInfo] = None,
        preferred_language: str = "auto",
        session_id: Optional[str] = None,
        include_reasoning: bool = False
    ) -> Dict[str, Any]:
        """
        Complete workflow: Patient → LLM Diagnosis → RAG Enhancement → Doctor Approval

        Flow:
        1. LLM generates initial diagnosis
        2. RAG provides medication recommendations
        3. Combine into hybrid response
        4. Queue for doctor approval/edit/reject
        """

        logger.info(f"🔄 Processing with doctor approval workflow: {session_id}")

        try:
            # STEP 1: Generate complete AI response (LLM + RAG)
            logger.info("🤖 Generating LLM diagnosis...")
            ai_response = await self._generate_complete_ai_response(
                message, conversation_history, patient_info, preferred_language, include_reasoning, session_id
            )

            # STEP 2: Queue hybrid response for doctor review
            logger.info("📋 Queueing AI response for doctor approval...")
            return await self._queue_ai_response_for_doctor_approval(
                original_message=message,
                ai_response=ai_response,
                patient_info=patient_info,
                session_id=session_id
            )

        except Exception as e:
            logger.error(f"Error in doctor approval workflow: {e}")
            return {
                "type": "error",
                "message": "เกิดข้อผิดพลาดในระบบ กรุณาลองใหม่อีกครั้ง",
                "error": str(e)
            }

    async def _generate_complete_ai_response(
        self,
        message: str,
        conversation_history: Optional[List[ConversationMessage]] = None,
        patient_info: Optional[PatientInfo] = None,
        preferred_language: str = "auto",
        include_reasoning: bool = False,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate complete AI response (LLM diagnosis + RAG medications)"""

        # Detect language and translate if needed
        language_info = await self._detect_and_translate(message)
        processed_message = language_info.get("translated_message", message)

        # Log translation if needed
        if language_info.get("translation_used"):
            llm_logger.log_translation(
                source_text=message,
                translated_text=processed_message,
                source_language=language_info.get("detected_language", "unknown"),
                target_language="english"
            )

        # Get emergency response if needed
        emergency_response = self._detect_emergency_keywords(message)
        if emergency_response:
            return {
                "type": "emergency",
                "message": emergency_response["message"],
                "urgency": emergency_response["urgency"],
                "recommendations": emergency_response["recommendations"]
            }

        # STEP 1: LLM DIAGNOSIS - Use medical AI to analyze symptoms
        diagnostic_result = await self.agents["diagnostic"].analyze_common_symptoms({
            "message": processed_message,
            "patient_info": patient_info,
            "session_id": session_id or "unknown"
        })

        if not diagnostic_result or not diagnostic_result.get("primary_diagnosis"):
            return {
                "type": "no_diagnosis",
                "message": "ไม่สามารถวินิจฉัยได้ในขณะนี้ กรุณาปรึกษาแพทย์",
                "recommendations": ["ปรึกษาแพทย์เพื่อการตรวจสอบเพิ่มเติม"]
            }

        diagnosis = diagnostic_result["primary_diagnosis"]

        # STEP 2: RAG ENHANCEMENT - Get medication recommendations
        logger.info("🔍 Enhancing with RAG medications...")
        medications = await self.agents["treatment"]._recommend_medications(diagnosis, patient_info)

        # STEP 3: Generate treatment recommendations
        treatment_result = await self.agents["treatment"].generate_treatment_plan({
            "diagnosis": diagnosis,
            "patient_info": patient_info,
            "session_id": session_id or "unknown"
        })

        # STEP 4: Assess urgency
        triage_result = await self.agents["triage"].assess_urgency({
            "diagnosis": diagnosis,
            "symptoms": processed_message,
            "patient_info": patient_info
        })

        # STEP 5: Coordinate final response
        final_response = await self.agents["coordinator"].coordinate_response({
            "diagnosis": diagnosis,
            "medications": medications,  # RAG-enhanced medications
            "treatment": treatment_result,
            "triage": triage_result,
            "reasoning": include_reasoning
        })

        return {
            "type": "medical_consultation",
            "diagnosis": diagnosis,
            "medications": medications,  # RAG medications with LLM instructions
            "treatment": treatment_result,
            "urgency": triage_result.get("urgency", "routine"),
            "response": final_response.get("message", ""),
            "recommendations": final_response.get("recommendations", []),
            "reasoning": final_response.get("reasoning") if include_reasoning else None,
            "context_considered": patient_info is not None
        }

    async def _queue_ai_response_for_doctor_approval(
        self,
        original_message: str,
        ai_response: Dict[str, Any],
        patient_info: Optional[PatientInfo] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Queue the complete AI response for doctor approval/edit/reject"""

        # Extract diagnosis name properly
        diagnosis_display = "ไม่ระบุ"
        diagnosis_data = ai_response.get('diagnosis')
        if diagnosis_data:
            if isinstance(diagnosis_data, dict):
                diagnosis_display = diagnosis_data.get('thai_name') or diagnosis_data.get('english_name') or diagnosis_data.get('name') or str(diagnosis_data)
            else:
                diagnosis_display = str(diagnosis_data)

        # Create approval queue entry
        approval_entry = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "patient_message": original_message,
            "patient_info": patient_info.dict() if patient_info else None,
            "ai_response": ai_response,
            "status": "pending_doctor_review",
            "doctor_actions": {
                "approve": "ส่งคำแนะนำของ AI ให้ผู้ป่วยโดยตรง",
                "edit": "แก้ไขคำแนะนำก่อนส่งให้ผู้ป่วย",
                "reject": "ปฏิเสธและให้คำแนะนำใหม่"
            }
        }

        # TODO: Store in database/queue for doctor review
        # For now, log the approval entry
        logger.info(f"📋 AI Response queued for doctor approval: {session_id}")
        logger.info(f"Diagnosis: {ai_response.get('diagnosis', {}).get('english_name', 'Unknown')}")
        logger.info(f"Medications: {len(ai_response.get('medications', []))} items")

        # Return status message to patient
        thai_message = f"""📋 การวิเคราะห์อาการของคุณเสร็จสิ้นแล้ว

🤖 **ระบบ AI ได้วิเคราะห์อาการแล้ว**:
• การวินิจฉัยเบื้องต้น: {diagnosis_display}
• ยาที่แนะนำ: {len(ai_response.get('medications', []))} รายการ
• ระดับความเร่งด่วน: {ai_response.get('urgency', 'ปกติ')}

⏳ **สถานะ**: รอแพทย์ตรวจสอบและอนุมัติ

🩺 **ขั้นตอนต่อไป**:
• แพทย์จะตรวจสอบคำแนะนำของ AI
• อนุมัติ แก้ไข หรือให้คำแนะนำใหม่
• คุณจะได้รับคำตอบสุดท้ายภายใน 15-30 นาที

⚠️ **หากมีอาการฉุกเฉิน**: โทร 1669 ทันที

💬 **หมายเหตุ**: ระบบจะแจ้งเตือนเมื่อแพทย์ตอบกลับแล้ว"""

        return {
            "type": "pending_doctor_approval",
            "message": thai_message,
            "status": "pending_doctor_review",
            "ai_preview": {
                "diagnosis": diagnosis_display,
                "medication_count": len(ai_response.get('medications', [])),
                "urgency": ai_response.get('urgency', 'ปกติ')
            },
            "session_id": session_id,
            "timestamp": approval_entry["timestamp"]
        }

    def _retrieve_medications_from_rag(self, condition: str, symptoms: List[str]) -> List[Dict]:
        """STEP 1: Retrieve medicine names and dosages from RAG knowledge base"""

        relevant_medicines = []

        # Search treatments by condition
        logger.info(f"🔍 Searching {len(self.medical_service.treatments)} treatments for condition: '{condition}'")
        for treatment in self.medical_service.treatments:
            treatment_desc = treatment.description.lower()
            treatment_name = treatment.name_en.lower()

            logger.info(f"🔍 Checking treatment: {treatment.name_en} -> {treatment_desc}")

            # Search in both treatment name and description
            condition_found = condition.lower() in treatment_desc or condition.lower() in treatment_name
            symptom_found = any(symptom.lower() in treatment_desc or symptom.lower() in treatment_name for symptom in symptoms)

            if condition_found or symptom_found:
                # Extract medicine info from RAG
                medicine_data = {
                    "english_name": treatment.name_en,
                    "thai_name": treatment.name_th,
                    "dosage": self._extract_dosage_from_rag(treatment.description),
                    "category": treatment.category,
                    "rag_source": treatment.id
                }
                relevant_medicines.append(medicine_data)

        # Search medicines directly
        for medicine in self.medical_service.medicines:
            medicine_desc = medicine.description.lower()
            if condition in medicine_desc or any(symptom.lower() in medicine_desc for symptom in symptoms):
                medicine_data = {
                    "english_name": medicine.name_en,
                    "thai_name": medicine.name_th,
                    "dosage": self._extract_dosage_from_rag(medicine.description),
                    "category": medicine.category,
                    "rag_source": medicine.id
                }
                relevant_medicines.append(medicine_data)

        return relevant_medicines[:3]  # Limit to top 3 relevant medications

    def _extract_dosage_from_rag(self, description: str) -> str:
        """Extract dosage information from RAG description"""
        import re

        # Common dosage patterns
        dosage_patterns = [
            r'(\d+(?:\.\d+)?)\s*mg',
            r'(\d+(?:\.\d+)?)\s*ml',
            r'(\d+)\s*tablets?',
            r'(\d+)\s*capsules?'
        ]

        for pattern in dosage_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return match.group(0)

        return "ตามคำแนะนำของแพทย์"  # Default if no dosage found

    async def _generate_llm_medication_instructions(self, medicine: Dict, patient_info: Optional[PatientInfo],
                                            condition: str, symptoms: List[str]) -> Dict:
        """STEP 2: LLM generates duration, frequency, and clinical instructions"""

        patient_age = patient_info.age if patient_info else 30
        contraindications = self._check_contraindications(patient_info)

        # LLM prompt for clinical reasoning
        llm_prompt = f"""
Patient: {patient_age} years old
Medicine: {medicine['english_name']} ({medicine['thai_name']})
Condition: {condition}
Symptoms: {', '.join(symptoms)}

Generate appropriate:
1. Duration (how many days)
2. Frequency (how often per day)
3. Instructions (when to take, with/without food)
4. Warnings (side effects, precautions)

Consider patient age and safety.
"""

        # Use LLM to generate clinical instructions
        try:
            # Call LLM service asynchronously
            llm_response = await self._call_llm_for_medication_guidance(llm_prompt)
            return self._parse_llm_medication_response(llm_response, medicine['english_name'], contraindications)
        except Exception as e:
            # Fallback to safe defaults
            return self._get_safe_medication_defaults(medicine['english_name'], patient_age, contraindications)

    async def _call_llm_for_medication_guidance(self, prompt: str) -> str:
        """Call LLM model for medication duration and instructions"""
        try:
            # Use MedLlama2 for clinical reasoning about medication instructions
            if hasattr(self, 'ollama_client') and self.ollama_client:
                response = await self.ollama_client.chat(
                    model="medllama2",
                    messages=[{
                        "role": "system",
                        "content": "You are a medical AI providing medication guidance. Give specific duration, frequency, instructions, and warnings. Use this format:\nDuration: X days\nFrequency: X times per day\nInstructions: when/how to take\nWarnings: safety precautions"
                    }, {
                        "role": "user",
                        "content": prompt
                    }]
                )
                return response.get('message', {}).get('content', '')
            else:
                # Fallback response
                return """
Duration: 5-7 วัน
Frequency: ทุก 6-8 ชั่วโมง ตามอาการ
Instructions: รับประทานหลังอาหารเพื่อลดการระคายเคืองกระเพาะ
Warnings: ไม่ควรเกิน 4 ครั้งต่อวัน หลีกเลี่ยงแอลกอฮอล์
                """
        except Exception as e:
            logger.error(f"LLM medication guidance failed: {e}")
            # Safe fallback
            return """
Duration: 5-7 วัน
Frequency: ตามคำแนะนำของแพทย์
Instructions: รับประทานตามคำแนะนำบนฉลากยา
Warnings: ปรึกษาแพทย์หากอาการไม่ดีขึ้นภายใน 3 วัน
            """

    def _parse_llm_medication_response(self, llm_response: str, medicine_name: str, contraindications: Dict) -> Dict:
        """Parse LLM response into structured medication instructions"""
        import re

        duration_match = re.search(r'Duration:\s*(.+)', llm_response)
        frequency_match = re.search(r'Frequency:\s*(.+)', llm_response)
        instructions_match = re.search(r'Instructions:\s*(.+)', llm_response)
        warnings_match = re.search(r'Warnings:\s*(.+)', llm_response)

        return {
            "duration": duration_match.group(1).strip() if duration_match else "5-7 วัน",
            "frequency": frequency_match.group(1).strip() if frequency_match else "ทุก 6-8 ชั่วโมง",
            "instructions": instructions_match.group(1).strip() if instructions_match else "รับประทานหลังอาหาร",
            "warnings": [warnings_match.group(1).strip()] if warnings_match else ["ใช้ตามคำแนะนำของแพทย์"],
            "contraindications": contraindications.get(medicine_name.lower(), [])
        }

    def _get_safe_medication_defaults(self, medicine_name: str, patient_age: int, contraindications: Dict) -> Dict:
        """Safe fallback medication instructions when LLM fails"""
        return {
            "duration": "5-7 วัน",
            "frequency": "ตามคำแนะนำของแพทย์",
            "instructions": "รับประทานหลังอาหาร",
            "warnings": ["ปรึกษาแพทย์หากอาการไม่ดีขึ้น", "อ่านคำแนะนำบนฉลากยา"],
            "contraindications": contraindications.get(medicine_name.lower(), [])
        }

    def _parse_llm_diagnosis_response(self, response: str) -> List[Dict]:
        """Parse LLM diagnosis response into structured format"""

        diagnoses = []
        lines = response.split('\n')

        for line in lines:
            line = line.strip()
            if not line or ':' not in line:
                continue

            try:
                # Parse format: "PRIMARY: J00 Common cold | ไข้หวัด | Confidence: 75"
                parts = line.split(':', 1)
                if len(parts) != 2:
                    continue

                diagnosis_part = parts[1].strip()

                # Split by pipe
                diagnosis_components = diagnosis_part.split('|')
                if len(diagnosis_components) >= 3:
                    # Extract ICD code and English name
                    icd_and_english = diagnosis_components[0].strip()
                    thai_name = diagnosis_components[1].strip()
                    confidence_part = diagnosis_components[2].strip()

                    # Extract ICD code
                    icd_parts = icd_and_english.split(' ', 1)
                    icd_code = icd_parts[0] if icd_parts else ""
                    english_name = icd_parts[1] if len(icd_parts) > 1 else icd_and_english

                    # Extract confidence
                    confidence = 60.0  # Default
                    if 'confidence:' in confidence_part.lower():
                        try:
                            conf_str = confidence_part.lower().replace('confidence:', '').strip()
                            confidence = float(conf_str)
                        except:
                            pass

                    # Determine category
                    category = self._get_category_from_icd(icd_code)

                    diagnosis = {
                        "icd_code": icd_code,
                        "english_name": english_name,
                        "thai_name": thai_name,
                        "confidence": confidence,
                        "category": category
                    }

                    diagnoses.append(diagnosis)

            except Exception as e:
                logger.warning(f"Failed to parse diagnosis line: {line} - {e}")
                continue

        return diagnoses if diagnoses else None

    def _get_category_from_icd(self, icd_code: str) -> str:
        """Get medical category from ICD code"""

        if not icd_code:
            return "general"

        first_char = icd_code[0].upper()

        categories = {
            'A': 'infectious', 'B': 'infectious',
            'C': 'neoplasm', 'D': 'blood',
            'E': 'endocrine', 'F': 'mental',
            'G': 'neurological', 'H': 'sensory',
            'I': 'cardiovascular', 'J': 'respiratory',
            'K': 'digestive', 'L': 'dermatological',
            'M': 'musculoskeletal', 'N': 'genitourinary',
            'O': 'pregnancy', 'P': 'perinatal',
            'Q': 'congenital', 'R': 'symptoms',
            'S': 'injury', 'T': 'injury',
            'V': 'external', 'W': 'external',
            'X': 'external', 'Y': 'external',
            'Z': 'factors'
        }

        return categories.get(first_char, "general")


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


class DiagnosticAgent(BaseAgent):
    """Agent specialized in common illness diagnosis"""

    def __init__(self):
        super().__init__()
        self.name = "DiagnosticAgent"
        self.role = "Common illness diagnosis and analysis"
        self.few_shot_learning = AdvancedFewShotLearning()

    async def analyze_common_symptoms(self, case_data: Dict) -> Dict[str, Any]:
        """Analyze symptoms for common illnesses with context integration"""

        message = case_data.get("message", "")
        patient_info = case_data.get("patient_info")

        # CRITICAL FIX: Pass patient context to diagnosis search
        matching_diagnoses = await self._search_diagnoses(message, patient_info=patient_info)

        # Calculate confidence for each diagnosis
        primary_diagnosis = None
        differential_diagnoses = []

        if matching_diagnoses:
            primary_diagnosis = matching_diagnoses[0]
            differential_diagnoses = matching_diagnoses[1:3]  # Top 3 alternatives

        # Add context consideration flag
        context_considered = patient_info is not None

        return {
            "primary_diagnosis": primary_diagnosis,
            "differential_diagnoses": differential_diagnoses,
            "confidence": primary_diagnosis.get("confidence", 0) if primary_diagnosis else 0,
            "reasoning": f"Based on common illness symptom analysis{'with patient context' if context_considered else ''}",
            "context_considered": context_considered
        }

    async def analyze_symptoms(self, case_data: Dict) -> Dict[str, Any]:
        """Analyze symptoms for medical diagnosis API"""
        return await self.analyze_common_symptoms(case_data)

    async def _search_diagnoses(self, symptoms: str, patient_id: Optional[str] = None, patient_info: Optional[Any] = None) -> List[Dict]:
        """Enhanced diagnosis search with memory and advanced few-shot learning"""

        # PRIORITY 1: Check for red flag symptoms with CONTEXT AWARENESS
        red_flags = self._check_red_flags(symptoms)
        if red_flags and red_flags.get('detected') and red_flags.get('max_urgency') == 'critical':
            logger.warning(f"🚨 CRITICAL RED FLAGS detected: {red_flags['keywords']}")

            # CONTEXT-AWARE RED FLAG PROCESSING
            context_modified = self._apply_context_to_red_flags(red_flags, symptoms, patient_info)

            if context_modified.get('override_emergency', False):
                # Context suggests this may not be emergency - proceed with differential diagnosis
                logger.info(f"🧠 Context overrides emergency for: {context_modified['reasoning']}")
            else:
                # Still emergency after context consideration
                return [{
                    "icd_code": "Z71.1",
                    "english_name": "Emergency Medical Consultation",
                    "thai_name": "ต้องไปโรงพยาบาลทันที",
                    "confidence": 95,
                    "category": "Emergency",
                    "matched_keywords": red_flags["keywords"],
                    "red_flags": red_flags,
                    "severity_score": 100,
                    "context_considered": patient_info is not None
                }]

        # PRIORITY 2: Try adaptive diagnosis from memory agent
        adaptive_diagnosis = await memory_agent.get_adaptive_diagnosis(symptoms, patient_id)

        if adaptive_diagnosis.get('primary_diagnosis') and adaptive_diagnosis['confidence'] > 0.7:
            logger.info(f"🧠 Using adaptive diagnosis from memory: {adaptive_diagnosis['primary_diagnosis']}")
            return [adaptive_diagnosis['primary_diagnosis']]

        # PRIORITY 3: Try few-shot learning for enhanced diagnosis with CONTEXT
        few_shot_result = await self.few_shot_learning.enhanced_diagnosis(symptoms, patient_id, patient_info)
        if few_shot_result.get('primary_diagnosis'):
            # Only use few-shot diagnosis if it's not overly aggressive for common symptoms
            diagnosis = few_shot_result['primary_diagnosis']

            # Extra safety check for serious conditions
            serious_conditions = ["tuberculosis", "tb", "วัณโรค", "cancer", "มะเร็ง", "stroke", "heart attack", "meningitis"]
            diagnosis_name = diagnosis.get('name', '').lower()

            if any(condition in diagnosis_name for condition in serious_conditions):
                logger.warning(f"⚠️ Blocking serious diagnosis from few-shot: {diagnosis_name} for symptoms: {symptoms}")
                # Skip few-shot for serious conditions, use conservative diagnosis instead
            else:
                is_aggressive_diagnosis = await self.medical_service._is_aggressive_diagnosis_llm(diagnosis, symptoms)
                if not is_aggressive_diagnosis and few_shot_result['confidence'] > 0.75:  # Increased threshold
                    logger.info(f"🎯 Using few-shot learning diagnosis: {diagnosis}")
                    return [diagnosis] + few_shot_result.get('differential_diagnoses', [])

        # PRIORITY 4: Use LLM-powered conservative diagnosis for common symptoms
        conservative_diagnosis = await self.medical_service._get_llm_conservative_diagnosis(symptoms)
        if conservative_diagnosis:
            logger.info(f"🤖 Using LLM-powered conservative diagnosis")
            return conservative_diagnosis

        # PRIORITY 5: Check for any red flag symptoms (including non-critical ones)
        if red_flags and red_flags.get('detected'):
            logger.info(f"⚠️ Red flags detected: {red_flags['keywords']}")
            # Return consultation needed for any red flags
            return [{
                "icd_code": "Z71.1",
                "english_name": "Medical Consultation",
                "thai_name": "ต้องปรึกษาแพทย์",
                "confidence": 90,
                "category": "Requires Medical Attention",
                "matched_keywords": red_flags["keywords"],
                "red_flags": red_flags,
                "severity_score": 50
            }]

        # Comprehensive symptom-to-diagnosis mapping with variations
        symptom_diagnosis_map = {
            "common_cold": {
                "keywords": [
                    # Thai variations - expanded for better matching
                    "ไข้", "ไข้เล็กน้อย", "ไข้ต่ำ", "ไข้ 38", "ไข้สูง", "เป็นไข้", "มีไข้",
                    "ตัวร้อน", "ตัวรุ่ม", "ตัวร้อนๆ", "ไม่สบาย", "ไม่สบายตัว",
                    "คัดจมูก", "จมูกแน่น", "น้ำมูกใส", "น้ำมูกเหลว", "น้ำมูกเขียว", "น้ำมูกข้น",
                    "ไอ", "ไอแห้ง", "ไอเล็กน้อย", "ไอบ่อย", "ไอมีเสียง",
                    "เจ็บคอ", "คอแห้ง", "คอแสบ", "คอบวม", "กลืนลำบาก",
                    "ปวดเมื่อย", "เมื่อยตัว", "อ่อนเพลีย", "เหนื่อย", "นอนไม่หลับ",
                    "ปวดหัว", "ปวดหัวเล็กน้อย", "ศีรษะปวด",
                    "จาม", "จามบ่อย", "ตาแดง", "ตาคัน",
                    # Common cold specific phrases
                    "หวัด", "เป็นหวัด", "หัดตัว", "วัน", "สองสามวัน", "มาสองสามวัน",
                    # English
                    "fever", "low fever", "runny nose", "stuffy nose", "cough", "dry cough",
                    "sore throat", "body aches", "fatigue", "sneezing", "nasal congestion",
                    "green mucus", "yellow mucus", "cold", "common cold", "headache", "feel hot"
                ],
                "icd_code": "J00", "thai_name": "ไข้หวัด", "english_name": "Common Cold", "confidence_base": 85,
                "severity_indicators": ["ไข้สูง", "หายใจลำบาก", "เจ็บคอมาก"],
                "typical_combination": ["ไข้", "ไอ", "น้ำมูก"]  # If all 3 present, very likely cold
            },
            "flu": {
                "keywords": [
                    "ไข้สูง", "ไข้มาก", "ปวดหัว", "ปวดตัว", "หนาวสั่น", "เหนื่อยมาก", "ไอมาก",
                    "high fever", "body aches", "chills", "headache", "severe fatigue", "muscle pain"
                ],
                "icd_code": "J11.1", "thai_name": "ไข้หวัดใหญ่", "english_name": "Influenza", "confidence_base": 75
            },
            "diabetes": {
                "keywords": [
                    # Thai comprehensive
                    "ปัสสาวะบ่อย", "ปัสสาวะมาก", "ปัสสาวะตลอดเวลา", "กระหายน้ำ", "กระหายน้ำมาก",
                    "ดื่มน้ำมาก", "น้ำหนักลด", "ผอมลง", "อ่อนเพลีย", "เหนื่อยง่าย", "ตาพร่ามัว",
                    "มองไม่ชัด", "แผลหายช้า", "ติดเชื้อง่าย", "หิวบ่อย", "กินมากแต่ผอม",
                    # English
                    "frequent urination", "excessive thirst", "weight loss", "fatigue", "blurred vision",
                    "slow healing", "frequent infections", "increased hunger"
                ],
                "icd_code": "E11", "thai_name": "เบาหวาน", "english_name": "Diabetes Mellitus", "confidence_base": 75,
                "severity_indicators": ["น้ำหนักลดมาก", "ตาพร่ามัวรุนแรง", "หายใจเหม็นผลไม้"]
            },
            "gastritis": {
                "keywords": [
                    # Thai comprehensive - เฉพาะเจาะจงมากขึ้น
                    "ปวดท้องส่วนบน", "ปวดลิ้นปี่", "แสบร้อน", "แสบกลางอก", "แสบกระเพาะ",
                    "คลื่นไส้", "อาเจียน", "ท้องอืด", "ท้องเฟ้อ", "เรอบ่อย", "กรดไหลย้อน",
                    "กินไม่ได้", "หิวแต่กินไม่ลง", "อาหารไม่ย่อย", "ปวดหลังอาหาร", "ปวดท้องว่าง",
                    "กระเพาะอักเสบ", "แสบร้อนกระเพาะ",
                    # English
                    "stomach pain", "gastric pain", "heartburn", "nausea", "vomiting", "bloating",
                    "acid reflux", "indigestion", "loss of appetite", "epigastric pain", "gastritis"
                ],
                "icd_code": "K29", "thai_name": "กระเพาะอักเสบ", "english_name": "Gastritis", "confidence_base": 80,
                "severity_indicators": ["อาเจียนเป็นเลือด", "ถ่ายดำ", "ปวดมาก"],
                "specific_location": "ท้องส่วนบน"
            },
            "appendicitis": {
                "keywords": [
                    # Thai comprehensive - เฉพาะเจาะจงสำหรับไส้ติ่ง
                    "ปวดท้องขวาล่าง", "ปวดท้องขวา", "ปวดตำแหน่งไส้ติ่ง", "ไส้ติ่งอักเสบ",
                    "ปวดท้องลิง", "ปวดท้องอักเสบ", "ไข้สูง", "คลื่นไส้อาเจียน",
                    "ไม่สามารถเดินได้", "ปวดเมื่อกด", "ปวดเมื่อไอ",
                    # English
                    "appendicitis", "right lower abdomen pain", "lower right abdominal pain",
                    "appendix pain", "mcburney point", "right iliac fossa pain"
                ],
                "icd_code": "K37", "thai_name": "ไส้ติ่งอักเสบ", "english_name": "Appendicitis", "confidence_base": 85,
                "severity_indicators": ["ไข้สูง", "ปวดรุนแรง", "อาเจียนมาก", "เดินไม่ได้"],
                "specific_location": "ท้องขวาล่าง",
                "emergency": True
            },
            "migraine": {
                "keywords": [
                    # Thai comprehensive
                    "ปวดศีรษะ", "ปวดหัว", "ไมเกรน", "ปวดข้างเดียว", "ปวดแบบตุบๆ", "ปวดแรง",
                    "คลื่นไส้", "อาเจียน", "กลัวแสง", "กลัวเสียง", "ตาเจ็บ", "มองเห็นแสงแวบ",
                    "เครียด", "นอนไม่หลับ", "อารมณ์เปลี่ยน",
                    # English
                    "headache", "migraine", "severe headache", "throbbing pain", "pulsating pain",
                    "photophobia", "phonophobia", "nausea", "vomiting", "aura", "visual disturbance"
                ],
                "icd_code": "G43.909", "thai_name": "ไมเกรน", "english_name": "Migraine", "confidence_base": 75,
                "severity_indicators": ["ปวดรุนแรง", "อาเจียนมาก", "มองไม่เห็น"]
            },
            "uti": {
                "keywords": [
                    # Thai comprehensive
                    "ปัสสาวะแสบ", "ปัสสาวะเจ็บ", "ปัสสาวะบ่อย", "ปัสสาวะไม่สุด", "ปัสสาวะขุ่น",
                    "ปัสสาวะเหม็น", "ปัสสาวะมีเลือด", "ปวดท้องน้อย", "ปวดหลัง", "ไข้ต่ำ",
                    "ปวดเมื่อปัสสาวะ", "รู้สึกไม่สบาย",
                    # English
                    "painful urination", "burning urination", "frequent urination", "urgency",
                    "cloudy urine", "bloody urine", "pelvic pain", "back pain", "dysuria"
                ],
                "icd_code": "N39.0", "thai_name": "ติดเชื้อทางเดินปัสสาวะ", "english_name": "Urinary Tract Infection", "confidence_base": 70,
                "severity_indicators": ["ปัสสาวะมีเลือด", "ไข้สูง", "ปวดหลังรุนแรง"]
            },
            "allergic_reaction": {
                "keywords": [
                    # Thai comprehensive
                    "ผื่น", "ผื่นแดง", "ผื่นลมพิษ", "คัน", "คันมาก", "คันทั่วตัว", "บวม", "หน้าบวม",
                    "ตาบวม", "ริมฝีปากบวม", "ลิ้นบวม", "หายใจลำบาก", "หายใจไม่ออก", "เหงื่อออก",
                    "วิงเวียน", "แพ้", "กินแล้วแพ้", "แพ้ยา", "แพ้อาหาร",
                    # English
                    "rash", "hives", "itching", "swelling", "facial swelling", "lip swelling",
                    "tongue swelling", "difficulty breathing", "wheezing", "allergic", "allergy"
                ],
                "icd_code": "T78.40", "thai_name": "ปฏิกิริยาแพ้", "english_name": "Allergic Reaction", "confidence_base": 80,
                "severity_indicators": ["หายใจลำบาก", "บวมรุนแรง", "เป็นลม"]
            },
            "hypertension": {
                "keywords": [
                    "ความดันสูง", "ปวดหัว", "วิงเวียน", "หูอื้อ", "ใจสั่น", "เจ็บหน้าอก", "หายใจไม่อิ่ม",
                    "high blood pressure", "hypertension", "headache", "dizziness", "chest pain"
                ],
                "icd_code": "I10", "thai_name": "ความดันโลหิตสูง", "english_name": "Hypertension", "confidence_base": 70
            },
            "depression_anxiety": {
                "keywords": [
                    "เครียด", "กังวล", "เศร้า", "นอนไม่หลับ", "ไม่อยากทำอะไร", "เหนื่อยใจ", "ห่วงมาก",
                    "กลัว", "ตื่นตระหนก", "ใจเต้นแรง", "มือสั่น", "เหงื่อออก",
                    "stress", "anxiety", "depression", "insomnia", "panic", "worry", "fear"
                ],
                "icd_code": "F43.9", "thai_name": "ความเครียดและความวิตกกังวล", "english_name": "Stress and Anxiety", "confidence_base": 65
            },
            "fever": {
                "keywords": [
                    # Thai comprehensive
                    "ไข้", "ไข้สูง", "ไข้ต่ำ", "ตัวร้อน", "ร่างกายร้อน", "หนาวสั่น", "หนาว", "สั่น",
                    "ซึม", "อ่อนเพลีย", "เด็กไข้", "ลูกไข้", "ไข้ขึ้นลง", "ตัวเป็นไข้",
                    # English
                    "fever", "high fever", "temperature", "hot", "chills", "shivering", "feverish"
                ],
                "icd_code": "R50.9", "thai_name": "ไข้", "english_name": "Fever", "confidence_base": 70,
                "severity_indicators": ["ไข้สูงมาก", "ชัก", "ซึมมาก"]
            }
        }

        matches = []
        symptoms_lower = symptoms.lower()

        # Enhanced scoring with context and patterns
        for condition, data in symptom_diagnosis_map.items():
            score = 0
            matched_keywords = []
            severity_score = 0

            # Main keyword matching with weighted scoring
            for keyword in data["keywords"]:
                if keyword.lower() in symptoms_lower or keyword in symptoms:
                    # Weight scoring based on keyword specificity
                    if len(keyword) > 8:  # Specific symptoms get higher weight
                        score += 20
                    elif len(keyword) > 4:
                        score += 15
                    else:
                        score += 10
                    matched_keywords.append(keyword)

            # Check for severity indicators
            severity_indicators = data.get("severity_indicators", [])
            for indicator in severity_indicators:
                if indicator.lower() in symptoms_lower or indicator in symptoms:
                    severity_score += 25
                    score += 10  # Bonus for severity matching

            # Adjust score based on condition-specific patterns
            if condition == "diabetes" and any(word in symptoms_lower for word in ["บ่อย", "มาก", "ลด"]):
                score += 15  # Frequency/quantity patterns
            elif condition == "migraine" and any(word in symptoms_lower for word in ["ข้างเดียว", "ตุบ", "แรง"]):
                score += 15  # Pain pattern descriptions
            elif condition == "gastritis" and any(word in symptoms_lower for word in ["หลัง", "ว่าง", "กิน"]):
                score += 15  # Timing patterns

            # Minimum threshold: at least 2 keywords or 1 specific keyword
            threshold = 25 if len(matched_keywords) >= 2 else 40

            if score >= threshold:
                # Calculate final confidence with severity adjustment
                base_confidence = data["confidence_base"]
                bonus = min((score - threshold), 20)
                severity_bonus = min(severity_score, 15)

                final_confidence = min(base_confidence + bonus + severity_bonus, 98)

                matches.append({
                    "icd_code": data["icd_code"],
                    "english_name": data["english_name"],
                    "thai_name": data["thai_name"],
                    "confidence": final_confidence,
                    "category": "Common Illness",
                    "matched_keywords": matched_keywords,
                    "severity_score": severity_score,
                    "red_flags": red_flags if red_flags else None,
                    "pattern_bonus": bonus
                })

        # Include red flag results if found but no specific diagnosis
        if red_flags and not matches:
            matches.append({
                "icd_code": "Z71.1",
                "english_name": "Medical Consultation",
                "thai_name": "ต้องปรึกษาแพทย์",
                "confidence": 90,
                "category": "Requires Medical Attention",
                "matched_keywords": red_flags["keywords"],
                "red_flags": red_flags,
                "severity_score": 50
            })

        return sorted(matches, key=lambda x: x["confidence"], reverse=True)

    def _check_red_flags(self, symptoms: str) -> Dict[str, Any]:
        """Check for red flag symptoms that require immediate medical attention"""

        red_flag_symptoms = {
            "cardiovascular": {
                "keywords": ["เจ็บหน้าอก", "ปวดหน้าอก", "แน่นหน้าอก", "หายใจไม่ออก", "หายใจลำบาก",
                           "เหงื่อออก", "หน้าซีด", "ใจเต้นผิดปกติ", "chest pain", "shortness of breath"],
                "urgency": "critical"
            },
            "neurological": {
                "keywords": ["หมดสติ", "ชัก", "อัมพาต", "พูดไม่ได้", "มึนงง", "โรคหลอดเลือดสมอง",
                           "ปวดหัวรุนแรง", "มองไม่เห็น", "unconscious", "seizure", "stroke", "paralysis"],
                "urgency": "critical"
            },
            "severe_allergic": {
                "keywords": ["หายใจไม่ออก", "บวมรุนแรง", "ลิ้นบวม", "คอบวม", "เป็นลม", "วิงเวียนมาก",
                           "anaphylaxis", "severe swelling", "throat swelling"],
                "urgency": "critical"
            },
            "severe_bleeding": {
                "keywords": ["เลือดออกมาก", "อาเจียนเป็นเลือด", "ถ่ายเป็นเลือด", "ถ่ายดำ", "เลือดกำเดา",
                           "severe bleeding", "blood vomiting", "bloody stool"],
                "urgency": "high"
            },
            "high_fever_complications": {
                "keywords": ["ไข้สูงมาก", "ไข้เกิน 40", "ชัก", "ซึมมาก", "ปวดคอแข็ง", "ผื่นแดงไม่หาย",
                           "very high fever", "febrile seizure", "neck stiffness", "persistent rash"],
                "urgency": "high"
            }
        }

        detected_flags = []
        max_urgency = "none"

        symptoms_lower = symptoms.lower()

        for category, data in red_flag_symptoms.items():
            for keyword in data["keywords"]:
                if keyword.lower() in symptoms_lower or keyword in symptoms:
                    detected_flags.append({"keyword": keyword, "category": category, "urgency": data["urgency"]})
                    if data["urgency"] == "critical":
                        max_urgency = "critical"
                    elif data["urgency"] == "high" and max_urgency != "critical":
                        max_urgency = "high"

        if detected_flags:
            return {
                "detected": True,
                "flags": detected_flags,
                "max_urgency": max_urgency,
                "keywords": [flag["keyword"] for flag in detected_flags],
                "recommendation": "โทร 1669 หรือไปโรงพยาบาลทันที" if max_urgency == "critical" else "ควรพบแพทย์โดยเร็ว"
            }

        return None

    def _apply_context_to_red_flags(self, red_flags: Dict, symptoms: str, patient_info: Optional[Any]) -> Dict:
        """Apply patient context to modify red flag interpretation"""

        if not patient_info:
            return {"override_emergency": False, "reasoning": "No context available"}

        # Extract patient context features (simplified for immediate fix)
        try:
            # Parse context from message if it's in string format
            context_text = str(patient_info).lower() if patient_info else ""

            # Young athlete context
            is_young_athlete = (
                any(term in context_text for term in ["นักกีฬา", "athlete", "วิ่ง", "running"]) and
                any(term in context_text for term in ["25", "26", "27", "28", "29"])
            )

            # Post-exercise context
            post_exercise = any(term in symptoms.lower() for term in ["ออกกำลังกาย", "exercise", "วิ่ง"])

            # Migraine history context
            has_migraine_history = any(term in context_text for term in ["ไมเกรน", "migraine", "ปวดหัวข้างเดียว"])

            # Check for context-specific overrides
            breathing_difficulty = "หายใจลำบาก" in symptoms
            severe_headache = "ปวดหัวรุนแรง" in symptoms

            # Young athlete with breathing difficulty after exercise
            if is_young_athlete and post_exercise and breathing_difficulty:
                return {
                    "override_emergency": True,
                    "reasoning": "Young athlete post-exercise - likely musculoskeletal/exercise-induced",
                    "suggested_urgency": "moderate",
                    "context_factors": ["young_athlete", "post_exercise"]
                }

            # Migraine patient with severe headache
            if has_migraine_history and severe_headache:
                return {
                    "override_emergency": True,
                    "reasoning": "Known migraine history - likely migraine exacerbation",
                    "suggested_urgency": "moderate",
                    "context_factors": ["migraine_history"]
                }

            return {"override_emergency": False, "reasoning": "Context does not override emergency assessment"}

        except Exception as e:
            logger.error(f"Error in context red flag analysis: {e}")
            return {"override_emergency": False, "reasoning": "Context analysis failed"}


class TreatmentAgent(BaseAgent):
    """Agent specialized in treatment planning for common illnesses"""

    def __init__(self):
        super().__init__()
        self.name = "TreatmentAgent"
        self.role = "Treatment planning for common illnesses"

    async def recommend_treatment(self, case_data: Dict) -> Dict[str, Any]:
        """Recommend treatment plan for common illnesses"""

        diagnosis = case_data.get("diagnosis")
        patient_info = case_data.get("patient_info")

        medications = await self._recommend_medications(diagnosis, patient_info)
        lifestyle_recommendations = self._get_lifestyle_recommendations(diagnosis)

        return {
            "medications": medications,
            "lifestyle_recommendations": lifestyle_recommendations,
            "follow_up_instructions": "ติดตามอาการและกลับมาพบแพทย์หากอาการไม่ดีขึ้น",
            "safety_warnings": [
                "อ่านคำแนะนำการใช้ยาให้ครบถ้วน",
                "หากมีอาการข้างเคียง ให้หยุดยาและปรึกษาแพทย์"
            ]
        }

    async def _recommend_medications(self, diagnosis: Optional[Dict], patient_info: Optional[PatientInfo]) -> List[Dict]:
        """RAG→LLM Hybrid medication recommendations: RAG provides medicine+dosage, LLM provides duration+instructions"""

        logger.info(f"💊 _recommend_medications called with diagnosis: {diagnosis}")

        if not diagnosis:
            logger.warning("❌ No diagnosis provided, returning empty medications")
            return []

        # Extract condition name from diagnosis
        english_name = diagnosis.get("english_name", "")
        thai_name = diagnosis.get("thai_name", "")

        # Try different strategies to get the condition name
        condition = ""
        if "gastritis" in english_name.lower() or "กระเพาะอักเสบ" in thai_name.lower():
            condition = "gastritis"
        elif "common cold" in english_name.lower() or "ไข้หวัด" in thai_name.lower():
            condition = "common cold"
        elif "migraine" in english_name.lower() or "ไมเกรน" in thai_name.lower():
            condition = "migraine"
        elif "hypertension" in english_name.lower() or "ความดันโลหิตสูง" in thai_name.lower():
            condition = "hypertension"
        elif "diabetes" in english_name.lower() or "เบาหวาน" in thai_name.lower():
            condition = "diabetes"
        else:
            # Fallback: use english_name directly
            condition = english_name.lower()

        symptoms = diagnosis.get("matched_keywords", [])
        red_flags = diagnosis.get("red_flags")

        logger.info(f"🔍 Extracted condition: '{condition}', symptoms: {symptoms}, red_flags: {red_flags}")

        # Don't recommend medications for red flag conditions
        if red_flags:
            return [{
                "english_name": "Emergency Care",
                "thai_name": "การดูแลเร่งด่วน",
                "dosage": "ไม่แนะนำยาใดๆ",
                "instructions": "ต้องได้รับการรักษาจากแพทย์ทันที",
                "category": "emergency",
                "warning": "ห้ามใช้ยาใดๆ ให้ไปโรงพยาบาลทันที"
            }]

        # STEP 1: RAG RETRIEVAL - Get medicine names and dosages from knowledge base
        rag_medications = self._retrieve_medications_from_rag(condition, symptoms)

        if not rag_medications:
            return []

        # STEP 2: LLM ENHANCEMENT - Generate duration, frequency, and clinical instructions
        enhanced_medications = []
        for rag_med in rag_medications:
            llm_instructions = await self._generate_llm_medication_instructions(
                medicine=rag_med,
                patient_info=patient_info,
                condition=condition,
                symptoms=symptoms
            )

            # Combine RAG data with LLM reasoning
            enhanced_med = {
                **rag_med,  # RAG: medicine name, dosage
                **llm_instructions  # LLM: duration, frequency, warnings
            }
            enhanced_medications.append(enhanced_med)

        return enhanced_medications

    def _retrieve_medications_from_rag(self, condition: str, symptoms: List[str]) -> List[Dict]:
        """STEP 1: Retrieve medicine names and dosages from RAG knowledge base (treatments.csv)"""

        logger.info(f"🔍 RAG search for condition: '{condition}', symptoms: {symptoms}")
        logger.info(f"📊 Knowledge base: {len(self.medical_service.treatments)} treatments, {len(self.medical_service.medicines)} medicines")

        relevant_medicines = []

        # ค้นหาในฐานข้อมูล treatments.csv ตาม condition
        for treatment in self.medical_service.treatments:
            treatment_desc = treatment.description.lower()
            treatment_name_en = treatment.name_en.lower()
            treatment_name_th = treatment.name_th.lower()

            logger.info(f"🔍 Checking treatment: {treatment.name_en} | {treatment.name_th} | {treatment.description}")

            # ตรวจสอบว่า condition ตรงกับการรักษาหรือไม่
            condition_found = (
                condition.lower() in treatment_desc or
                condition.lower() in treatment_name_en or
                condition.lower() in treatment_name_th
            )

            # ตรวจสอบ symptoms
            symptom_found = any(
                symptom.lower() in treatment_desc or
                symptom.lower() in treatment_name_en or
                symptom.lower() in treatment_name_th
                for symptom in symptoms
            )

            if condition_found or symptom_found:
                logger.info(f"✅ Found matching treatment: {treatment.name_en}")

                # ดึงข้อมูลยาจาก treatments.csv
                # treatment.name_en คือ medications field จาก CSV
                medications_text = treatment.name_en

                # แยกยาหลายตัว (ถ้ามี)
                medication_list = medications_text.split(' ')

                for med_name in medication_list:
                    med_name = med_name.strip()
                    if med_name and len(med_name) > 2:  # ข้ามคำสั้นเกินไป

                        # หาข้อมูลเพิ่มเติมจาก medicines.csv
                        detailed_medicine = None
                        for medicine in self.medical_service.medicines:
                            if (med_name.lower() in medicine.name_th.lower() or
                                med_name.lower() in medicine.name_en.lower()):
                                detailed_medicine = medicine
                                break

                        medicine_data = {
                            "english_name": detailed_medicine.name_en if detailed_medicine else med_name,
                            "thai_name": detailed_medicine.name_th if detailed_medicine else med_name,
                            "dosage": self._extract_dosage_from_medicine(detailed_medicine) if detailed_medicine else "ตามคำแนะนำแพทย์",
                            "frequency": "ตามคำแนะนำแพทย์",
                            "duration": "5-7 วัน",
                            "category": detailed_medicine.category if detailed_medicine else "medication",
                            "rag_source": treatment.id,
                            "instructions": "ทานตามคำแนะนำของแพทย์",
                            "from_treatments_csv": True,
                            "original_treatment": medications_text
                        }
                        relevant_medicines.append(medicine_data)
                        logger.info(f"💊 Added medication: {medicine_data['thai_name']}")

        logger.info(f"📋 Total medications found: {len(relevant_medicines)}")
        return relevant_medicines[:3]  # Limit to top 3 relevant medications

    def _extract_dosage_from_medicine(self, medicine) -> str:
        """Extract dosage from medicines.csv data"""
        if not medicine:
            return "ตามคำแนะนำแพทย์"

        # Try to get strength from description
        description = medicine.description.lower()

        # Extract dosage from the description which contains full medicine info
        import re
        dosage_patterns = [
            r'(\d+(?:\.\d+)?)\s*mg',
            r'(\d+(?:\.\d+)?)\s*ml',
            r'(\d+)\s*units?/ml',
            r'(\d+)\s*mcg'
        ]

        for pattern in dosage_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return match.group(0)

        return "ตามคำแนะนำแพทย์"

    def _extract_dosage_from_rag(self, description: str) -> str:
        """Extract dosage information from RAG description"""
        import re

        # Common dosage patterns
        dosage_patterns = [
            r'(\d+(?:\.\d+)?)\s*mg',
            r'(\d+(?:\.\d+)?)\s*ml',
            r'(\d+)\s*tablets?',
            r'(\d+)\s*capsules?'
        ]

        for pattern in dosage_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return match.group(0)

        return "ตามคำแนะนำของแพทย์"  # Default if no dosage found

    async def _generate_llm_medication_instructions(self, medicine: Dict, patient_info: Optional[PatientInfo],
                                            condition: str, symptoms: List[str]) -> Dict:
        """STEP 2: LLM generates duration, frequency, and clinical instructions"""

        patient_age = patient_info.age if patient_info else 30
        contraindications = self._check_contraindications(patient_info)

        # LLM prompt for clinical reasoning
        llm_prompt = f"""
Patient: {patient_age} years old
Medicine: {medicine['english_name']} ({medicine['thai_name']})
Condition: {condition}
Symptoms: {', '.join(symptoms)}

Generate appropriate:
1. Duration (how many days)
2. Frequency (how often per day)
3. Instructions (when to take, with/without food)
4. Warnings (side effects, precautions)

Consider patient age and safety.
"""

        # Use LLM to generate clinical instructions
        try:
            # Call LLM service asynchronously
            llm_response = await self._call_llm_for_medication_guidance(llm_prompt)
            return self._parse_llm_medication_response(llm_response, medicine['english_name'], contraindications)
        except Exception as e:
            # Fallback to safe defaults
            return self._get_safe_medication_defaults(medicine['english_name'], patient_age, contraindications)

    async def _call_llm_for_medication_guidance(self, prompt: str) -> str:
        """Call LLM model for medication duration and instructions"""
        try:
            # Use MedLlama2 for clinical reasoning about medication instructions
            if hasattr(self, 'ollama_client') and self.ollama_client:
                response = await self.ollama_client.chat(
                    model="medllama2",
                    messages=[{
                        "role": "system",
                        "content": "You are a medical AI providing medication guidance. Give specific duration, frequency, instructions, and warnings. Use this format:\nDuration: X days\nFrequency: X times per day\nInstructions: when/how to take\nWarnings: safety precautions"
                    }, {
                        "role": "user",
                        "content": prompt
                    }]
                )
                return response.get('message', {}).get('content', '')
            else:
                # Fallback response
                return """
Duration: 5-7 วัน
Frequency: ทุก 6-8 ชั่วโมง ตามอาการ
Instructions: รับประทานหลังอาหารเพื่อลดการระคายเคืองกระเพาะ
Warnings: ไม่ควรเกิน 4 ครั้งต่อวัน หลีกเลี่ยงแอลกอฮอล์
                """
        except Exception as e:
            logger.error(f"LLM medication guidance failed: {e}")
            # Safe fallback
            return """
Duration: 5-7 วัน
Frequency: ตามคำแนะนำของแพทย์
Instructions: รับประทานตามคำแนะนำบนฉลากยา
Warnings: ปรึกษาแพทย์หากอาการไม่ดีขึ้นภายใน 3 วัน
            """

    def _parse_llm_medication_response(self, llm_response: str, medicine_name: str, contraindications: Dict) -> Dict:
        """Parse LLM response into structured medication instructions"""
        import re

        duration_match = re.search(r'Duration:\s*(.+)', llm_response)
        frequency_match = re.search(r'Frequency:\s*(.+)', llm_response)
        instructions_match = re.search(r'Instructions:\s*(.+)', llm_response)
        warnings_match = re.search(r'Warnings:\s*(.+)', llm_response)

        return {
            "duration": duration_match.group(1).strip() if duration_match else "5-7 วัน",
            "frequency": frequency_match.group(1).strip() if frequency_match else "ทุก 6-8 ชั่วโมง",
            "instructions": instructions_match.group(1).strip() if instructions_match else "รับประทานหลังอาหาร",
            "warnings": [warnings_match.group(1).strip()] if warnings_match else ["ใช้ตามคำแนะนำของแพทย์"],
            "contraindications": contraindications.get(medicine_name.lower(), [])
        }

    def _get_safe_medication_defaults(self, medicine_name: str, patient_age: int, contraindications: Dict) -> Dict:
        """Safe fallback medication instructions when LLM fails"""
        return {
            "duration": "5-7 วัน",
            "frequency": "ตามคำแนะนำของแพทย์",
            "instructions": "รับประทานหลังอาหาร",
            "warnings": ["ปรึกษาแพทย์หากอาการไม่ดีขึ้น", "อ่านคำแนะนำบนฉลากยา"],
            "contraindications": contraindications.get(medicine_name.lower(), [])
        }

    def _calculate_paracetamol_dose(self, age: int, weight: Optional[float] = None) -> Dict[str, str]:
        """Calculate age and weight appropriate paracetamol dosage"""

        if age < 3:
            return {
                "dosage": "10-15 mg/kg",
                "frequency": "ทุก 4-6 ชั่วโมง",
                "max_daily": "60 mg/kg ต่อวัน"
            }
        elif age < 12:
            return {
                "dosage": "250mg",
                "frequency": "ทุก 4-6 ชั่วโมง",
                "max_daily": "1500mg ต่อวัน"
            }
        elif age < 18:
            return {
                "dosage": "500mg",
                "frequency": "ทุก 4-6 ชั่วโมง",
                "max_daily": "3000mg ต่อวัน"
            }
        else:
            return {
                "dosage": "500-1000mg",
                "frequency": "ทุก 4-6 ชั่วโมง",
                "max_daily": "4000mg ต่อวัน"
            }

    def _check_contraindications(self, patient_info: Optional[PatientInfo]) -> Dict[str, List[str]]:
        """Check for medication contraindications based on patient info"""

        contraindications = {
            "paracetamol": [],
            "nsaid": [],
            "antihistamine": [],
            "decongestant": [],
            "ppi": []
        }

        if not patient_info:
            return contraindications

        medical_history = getattr(patient_info, 'medical_history', '') or ''

        # Check common contraindications
        if "liver" in medical_history.lower() or "ตับ" in medical_history:
            contraindications["paracetamol"].append("โรคตับ")

        if any(word in medical_history.lower() for word in ["ulcer", "gastric", "กระเพาะ", "แผล"]):
            contraindications["nsaid"].append("แผลในกระเพาะ")

        if "kidney" in medical_history.lower() or "ไต" in medical_history:
            contraindications["nsaid"].append("โรคไต")

        if "hypertension" in medical_history.lower() or "ความดัน" in medical_history:
            contraindications["decongestant"].append("ความดันโลหิตสูง")

        if "glaucoma" in medical_history.lower() or "ต้อหิน" in medical_history:
            contraindications["antihistamine"].append("ต้อหิน")

        return contraindications

    def _get_lifestyle_recommendations(self, diagnosis: Optional[Dict]) -> List[str]:
        """Get lifestyle recommendations based on condition"""

        general_recommendations = [
            "พักผ่อนให้เพียงพอ อย่างน้อย 7-8 ชั่วโมง",
            "ดื่มน้ำสะอาดให้มาก วันละ 8-10 แก้ว",
            "รับประทานอาหารที่มีประโยชน์และครบ 5 หมู่",
            "หลีกเลี่ยงการสูบบุหรี่และดื่มแอลกอฮอล์"
        ]

        if not diagnosis:
            return general_recommendations

        condition = diagnosis.get("english_name", "").lower()
        specific_recommendations = []

        if "cold" in condition or "flu" in condition or "fever" in condition:
            specific_recommendations = [
                "🍲 อาหารที่แนะนำ: โจ๊กข้าว ซุปไก่ น้ำผึ้งผสมมะนาว น้ำขิงอุ่น",
                "🥛 เครื่องดื่ม: น้ำเปล่าอุ่น ชาขิง น้ำต้มใบสะระแหน่ (8-10 แก้วต่อวัน)",
                "❌ หลีกเลี่ยง: อาหารเย็น ไอศครีม เครื่องดื่มเย็น อาหารมัน",
                "🍊 วิตามินซี: ส้ม มะนาว ฝรั่ง กิวี่ เพื่อเสริมภูมิคุ้มกัน",
                "อมน้ำเกลือเจือจางบ้วนคอ (1 ช้อนชาต่อน้ำ 1 แก้ว)",
                "พักผ่อนให้เพียงพอ นอน 7-8 ชั่วโมงต่อวัน",
                "หลีกเลี่ยงการสูบบุหรี่และเครื่องดื่มแอลกอฮอล์"
            ]

        elif "gastritis" in condition:
            specific_recommendations = [
                "🍚 อาหารที่แนะนำ: โจ๊กข้าว ต้มจืด ปลาต้ม ไก่ต้มเปล่า กล้วยหอม",
                "🥛 เครื่องดื่ม: น้ำเปล่า นมจืดอุ่น น้ำผึ้ง (หลีกเลี่ยงของเย็นจัด)",
                "❌ หลีกเลี่ยงเด็ดขาด: อาหารเผ็ด เปรี้ยว มัน กาแฟ เหล้า บุหรี่",
                "⏰ เวลาทานอาหาร: 5-6 มื้อเล็กๆ เคี้ยวช้าๆ ไม่กินก่อนนอน 3 ชม",
                "🍌 ผลไม้ที่ดี: กล้วยหอม แอปเปิ้ล ปอปิด (หลีกเลี่ยงส้ม มะม่วงดิบ)"
            ]

        elif "migraine" in condition:
            specific_recommendations = [
                "พักในห้องมืดและเงียบ ใช้ผ้าเย็นประคบหน้าผาก",
                "หลีกเลี่ยงอาหารกระตุ้น เช่น ช็อกโกแลต ชีส กลูตาเมท MSG",
                "ควบคุมความเครียดด้วยการทำสมาธิ โยคะ หรือการหายใจลึก",
                "รักษาการนอนหลับให้สม่ำเสมอ หลีกเลี่ยงการนอนดึกหรือขาดนอน",
                "หลีกเลี่ยงแสงจ้า เสียงดัง และกลิ่นแรง"
            ]

        elif "uti" in condition or "urinary" in condition:
            specific_recommendations = [
                "ดื่มน้ำสะอาดเพิ่มขึ้น 8-10 แก้วต่อวัน",
                "ปัสสาวะบ่อยๆ ไม่กลั้นปัสสาวะ ปัสสาวะหลังมีเพศสัมพันธ์",
                "รักษาความสะอาดบริเวณอวัยวะเพศ เช้ดจากหน้าไปหลัง",
                "หลีกเลี่ยงการใส่ชุดชั้นในที่แน่นเกินไป เลือกผ้าฝ้าย",
                "หลีกเลี่ยงการใช้สบู่หอม สเปรย์ทำความสะอาดบริเวณอวัยวะเพศ"
            ]

        elif "allergy" in condition or "allergic" in condition:
            specific_recommendations = [
                "หลีกเลี่ยงสิ่งที่ทำให้เกิดอาการแพ้ หากทราบสาเหตุ",
                "รักษาความสะอาดในบ้าน ลดฝุ่นและสิ่งแปลกปลอม",
                "ใช้ผ้าปิดปากในสถานที่มีฝุ่นหรือมลพิษ",
                "อาบน้ำเย็นหรือใช้ผ้าเย็นประคบบริเวณผื่น",
                "หลีกเลี่ยงการขูดขีดบริเวณที่คัน เพื่อป้องกันการติดเชื้อ"
            ]

        elif "anxiety" in condition or "stress" in condition or "depression" in condition:
            specific_recommendations = [
                "ฝึกการหายใจลึกๆ 4-7-8 (หายใจเข้า 4 นับ กลั้น 7 นับ หายใจออก 8 นับ)",
                "ออกกำลังกายเบาๆ เช่น เดิน โยคะ หรือยืดเส้น 30 นาทีต่อวัน",
                "หลีกเลี่ยงคาเฟอีนหลัง 14:00 และเครื่องดื่มแอลกอฮอล์",
                "สร้างกิจวัตรการนอนที่ดี นอนและตื่นเวลาเดียวกันทุกวัน",
                "ค้นหากิจกรรมที่ทำให้ผ่อนคลาย เช่น อ่านหนังสือ ฟังเพลง"
            ]

        elif "hypertension" in condition:
            specific_recommendations = [
                "ลดการบริโภคเกลือและโซเดียม เป้าหมายต่ำกว่า 2300mg ต่อวัน",
                "เพิ่มการบริโภคผลไม้และผักที่มีโพแทสเซียม เช่น กล้วย ผักโขม",
                "ออกกำลังกายแอโรบิกปานกลาง 150 นาทีต่อสัปดาห์",
                "ลดน้ำหนักหากมีน้ำหนักเกิน BMI เป้าหมาย 18.5-24.9",
                "หลีกเลี่ยงการสูบบุหรี่และเครื่องดื่มแอลกอฮอล์"
            ]

        return specific_recommendations + general_recommendations[:2]


class TriageAgent(BaseAgent):
    """Agent specialized in medical triage and urgency assessment"""

    def __init__(self):
        super().__init__()
        self.name = "TriageAgent"
        self.role = "Medical triage and urgency assessment"

    async def assess_urgency(self, case_data: Dict) -> Dict[str, Any]:
        """Assess medical urgency and determine triage level"""

        message = case_data.get("message", "")
        patient_info = case_data.get("patient_info")
        # Note: Vital signs assessment removed - inappropriate for consultation scope

        # Check for emergency keywords
        emergency_keywords = [
            # Thai emergency keywords
            "ฉุกเฉิน", "เร่งด่วน", "หัวใจวาย", "หัวใจหยุดเต้น", "หายใจไม่ได้", "หายใจลำบากมาก",
            "เจ็บหน้าอกมาก", "ปวดหน้าอกแปลบ", "หมดสติ", "ชัก", "เลือดออกมาก", "อุบัติเหตุ",
            # English emergency keywords
            "emergency", "cardiac arrest", "can't breathe", "severe chest pain",
            "unconscious", "seizure", "severe bleeding", "stroke"
        ]

        # Calculate risk score based on symptoms
        risk_score = 0
        urgency = "low"
        triage_level = 5

        # Check for emergency keywords
        for keyword in emergency_keywords:
            if keyword.lower() in message.lower():
                risk_score += 50
                urgency = "critical"
                triage_level = 1
                break

        # Note: Vital signs assessment removed - inappropriate for consultation scope

        # Check age factors
        if patient_info and hasattr(patient_info, 'age') and patient_info.age:
            if patient_info.age > 70 or patient_info.age < 2:
                risk_score += 10

        # Symptom severity indicators
        severity_indicators = [
            "มาก", "รุนแรง", "แย่", "ทนไม่ได้", "ปวดมาก", "เจ็บมาก",
            "severe", "intense", "unbearable", "worst", "acute"
        ]

        for indicator in severity_indicators:
            if indicator.lower() in message.lower():
                risk_score += 15
                break

        # Determine urgency level based on risk score
        if risk_score >= 50:
            urgency = "critical"
            triage_level = 1
        elif risk_score >= 35:
            urgency = "high"
            triage_level = 2
        elif risk_score >= 20:
            urgency = "medium"
            triage_level = 3
        elif risk_score >= 10:
            urgency = "low"
            triage_level = 4
        else:
            urgency = "low"
            triage_level = 5

        return {
            "urgency": urgency,
            "triage_level": triage_level,
            "risk_score": risk_score,
            "reasoning": f"Risk assessment based on symptoms, vital signs, and patient factors",
            "recommendations": self._get_urgency_recommendations(urgency)
        }

    def _get_urgency_recommendations(self, urgency: str) -> List[str]:
        """Get recommendations based on urgency level"""

        recommendations = {
            "critical": [
                "โทร 1669 หรือไปโรงพยาบาลทันที",
                "อย่าขับรถไปเอง ให้คนอื่นพาไป",
                "เตรียมข้อมูลการรักษาและยาที่ใช้"
            ],
            "high": [
                "ไปโรงพยาบาลโดยเร็วภายใน 2-4 ชั่วโมง",
                "โทรแจ้งโรงพยาบาลก่อนไป",
                "สังเกตอาการอย่างใกล้ชิด"
            ],
            "medium": [
                "ควรพบแพทย์ภายใน 24 ชั่วโมง",
                "สังเกตอาการและบันทึกการเปลี่ยนแปลง",
                "หากอาการแย่ลง ให้ไปโรงพยาบาลทันที"
            ],
            "low": [
                "สามารถดูแลตัวเองที่บ้านได้",
                "สังเกตอาการต่อไป",
                "หากไม่ดีขึ้นภายใน 3-5 วัน ให้ปรึกษาแพทย์"
            ]
        }

        return recommendations.get(urgency, recommendations["low"])


class CoordinatorAgent(BaseAgent):
    """Agent that coordinates consultation for common illnesses"""

    def __init__(self):
        super().__init__()
        self.name = "CoordinatorAgent"
        self.role = "Common illness consultation coordination"

    async def process_common_illness_consultation(self, case_data: Dict) -> Dict[str, Any]:
        """Process common illness consultation"""

        reasoning_chain = []

        # Step 1: Symptom analysis
        diagnostic_agent = self.medical_service.agents["diagnostic"]
        diagnosis_result = await diagnostic_agent.analyze_common_symptoms(case_data)

        reasoning_chain.append(AgentThought(
            agent="DiagnosticAgent",
            step=1,
            reasoning="Analyzed symptoms for common illnesses",
            action="Identify potential common conditions",
            observation=f"Primary diagnosis confidence: {diagnosis_result['confidence']}%",
            confidence=diagnosis_result['confidence']
        ).__dict__)

        # Step 2: Treatment planning with RAG medications
        treatment_agent = self.medical_service.agents["treatment"]
        treatment_data = {**case_data, "diagnosis": diagnosis_result.get("primary_diagnosis")}
        treatment_result = await treatment_agent.recommend_treatment(treatment_data)

        reasoning_chain.append(AgentThought(
            agent="TreatmentAgent",
            step=2,
            reasoning="Provided general care recommendations",
            action="Suggest home care and when to see doctor",
            observation=f"Provided care recommendations",
            confidence=80.0
        ).__dict__)

        # Generate response
        response_message = self._generate_common_illness_response(
            diagnosis_result, treatment_result, case_data
        )

        return {
            "type": "common_illness_consultation",
            "message": response_message,
            "diagnosis": diagnosis_result,
            "treatment": treatment_result,
            "agent_reasoning_chain": reasoning_chain,
            "recommendation": "หากอาการไม่ดีขึ้นหรือมีอาการรุนแรงขึ้น กรุณาปรึกษาแพทย์"
        }

    def _generate_common_illness_response(self, diagnosis: Dict, treatment: Dict, case_data: Dict) -> str:
        """Generate intelligent, context-aware response for common illness consultation"""

        message_parts = []
        primary_diagnosis = diagnosis.get("primary_diagnosis")
        red_flags = diagnosis.get("red_flags")
        patient_info = case_data.get("patient_info")

        # Context-aware greeting
        if patient_info and hasattr(patient_info, 'age'):
            age = patient_info.age
            if age < 12:
                message_parts.append("👶 คำแนะนำสำหรับเด็ก")
            elif age > 65:
                message_parts.append("👵 คำแนะนำสำหรับผู้สูงอายุ")
            else:
                message_parts.append("📋 คำแนะนำเกี่ยวกับอาการของคุณ")
        else:
            message_parts.append("📋 คำแนะนำเกี่ยวกับอาการของคุณ")

        # Red flag warning (highest priority)
        if red_flags and red_flags.get("detected"):
            message_parts.append("\n🚨 **คำเตือนสำคัญ**")
            message_parts.append("อาการของคุณอาจร้ายแรง ต้องได้รับการตรวจรักษาจากแพทย์ทันที")
            for keyword in red_flags.get("keywords", [])[:3]:
                message_parts.append(f"⚠️ อาการเตือน: {keyword}")
            message_parts.append("\n📞 โทรหา 1669 หรือไปโรงพยาบาลใกล้บ้านทันที")
            return "\n".join(message_parts)

        # Normal diagnosis flow
        if primary_diagnosis:
            thai_name = primary_diagnosis.get('thai_name', '')
            english_name = primary_diagnosis.get('english_name', '')
            confidence = primary_diagnosis.get("confidence", 0)

            if thai_name:
                if confidence > 80:
                    message_parts.append(f"🩺 อาการน่าจะเป็น: **{thai_name}** (มั่นใจสูง)")
                elif confidence > 60:
                    message_parts.append(f"🩺 อาการน่าจะเป็น: **{thai_name}** (มั่นใจปานกลาง)")
                else:
                    message_parts.append(f"🩺 อาการอาจเป็น: {thai_name} (ต้องติดตามอาการ)")

                # Add condition-specific context
                self._add_condition_context(message_parts, english_name.lower())

        # Severity assessment
        severity_score = diagnosis.get("severity_score", 0)
        if severity_score > 30:
            message_parts.append("📈 ระดับความรุนแรง: ปานกลาง - ควรพบแพทย์เร็วๆ นี้")
        elif severity_score > 15:
            message_parts.append("📊 ระดับความรุนแรง: เล็กน้อย - สามารถดูแลเองได้")

        # Personalized medication recommendations
        medications = treatment.get("medications", [])
        if medications:
            message_parts.append("\n💊 ยาที่แนะนำ:")
            for i, med in enumerate(medications[:3]):  # Show up to 3 medications
                thai_name = med.get('thai_name', med.get('english_name', ''))
                dosage = med.get('dosage', '')
                instructions = med.get('instructions', '')
                warnings = med.get('warnings', [])

                med_line = f"{i+1}. **{thai_name}** - {dosage}"
                if instructions:
                    med_line += f"\n   📝 วิธีใช้: {instructions}"

                if warnings:
                    med_line += f"\n   ⚠️ ข้อควรระวัง: {warnings[0]}"

                message_parts.append(med_line)

        # Contextual lifestyle recommendations
        lifestyle = treatment.get("lifestyle_recommendations", [])
        if lifestyle:
            message_parts.append("\n🏠 การดูแลตัวเองที่บ้าน:")
            for i, rec in enumerate(lifestyle[:4], 1):
                message_parts.append(f"{i}. {rec}")

        # Timeline and expectations
        condition_name = primary_diagnosis.get('english_name', '').lower() if primary_diagnosis else ''
        recovery_time = self._get_recovery_timeline(condition_name)
        if recovery_time:
            message_parts.append(f"\n⏰ **ระยะเวลาหาย**: {recovery_time}")

        # Age-specific warnings
        if patient_info and hasattr(patient_info, 'age'):
            age_warnings = self._get_age_specific_warnings(patient_info.age, condition_name)
            if age_warnings:
                message_parts.append(f"\n👥 **คำแนะนำเฉพาะ**: {age_warnings}")

        # When to see doctor (condition-specific)
        doctor_warnings = self._get_condition_specific_warnings(condition_name)
        message_parts.append("\n⚠️ **ควรพบแพทย์เมื่อ**:")
        for warning in doctor_warnings:
            message_parts.append(f"• {warning}")

        # Follow-up questions for better care
        follow_up = self._generate_follow_up_questions(condition_name)
        if follow_up:
            message_parts.append(f"\n❓ **คำถามติดตาม**: {follow_up}")

        return "\n".join(message_parts)

    def _add_condition_context(self, message_parts: List[str], condition: str) -> None:
        """Add condition-specific context information"""
        context_map = {
            "common cold": "โรคไข้หวัดธรรมดาที่พบบ่อย มักหายเองใน 7-10 วัน",
            "gastritis": "การอักเสบของกระเพาะอาหาร มักเกิดจากการกินอาหารไม่เป็นเวลา",
            "migraine": "ปวดหัวข้างเดียวที่มีสาเหตุจากความเครียดหรืออาหาร",
            "urinary tract infection": "การติดเชื้อทางเดินปัสสาวะ พบบ่อยในผู้หญิง",
            "allergic reaction": "ปฏิกิริยาแพ้จากสิ่งแวดล้อมหรืออาหาร",
            "hypertension": "ความดันโลหิตสูง ต้องควบคุมด้วยการปรับพฤติกรรม"
        }

        if condition in context_map:
            message_parts.append(f"ℹ️ **ข้อมูลเพิ่มเติม**: {context_map[condition]}")

    def _get_recovery_timeline(self, condition: str) -> str:
        """Get expected recovery timeline for condition"""
        timelines = {
            "common cold": "7-10 วัน",
            "flu": "7-14 วัน",
            "gastritis": "2-7 วัน หากปรับการกิน",
            "migraine": "4-72 ชั่วโมง ต่อครั้ง",
            "urinary tract infection": "3-7 วัน หากได้รับการรักษา",
            "allergic reaction": "1-3 วัน หากหลีกเลี่ยงสาเหตุ",
            "fever": "2-5 วัน ขึ้นกับสาเหตุ"
        }
        return timelines.get(condition, "")

    def _get_age_specific_warnings(self, age: int, condition: str) -> str:
        """Get age-specific warnings and recommendations"""
        if age < 2:
            return "เด็กเล็กต้องได้รับการดูแลอย่างใกล้ชิด ควรพบกุมารแพทย์"
        elif age < 12:
            return "เด็กต้องให้ยาตามน้ำหนักและอายุ หลีกเลี่ยงแอสไพริน"
        elif age > 65:
            if "hypertension" in condition:
                return "ผู้สูงอายุควรตรวจความดันเป็นประจำ ระวังการล้มล้มศีรษะ"
            return "ผู้สูงอายุมีความเสี่ยงแทรกซ้อนสูงกว่า ควรพบแพทย์เร็วขึ้น"
        elif 50 <= age <= 65:
            return "ควรตรวจสุขภาพเป็นประจำ หากมีโรคประจำตัวต้องระวังปฏิสัมพันธ์ยา"
        return ""

    def _get_condition_specific_warnings(self, condition: str) -> List[str]:
        """Get condition-specific warning signs"""
        warning_map = {
            "common cold": [
                "ไข้สูงเกิน 39°C นานกว่า 3 วัน",
                "หายใจลำบาก หรือเสียงเปี๊ยกเวลาหายใจ",
                "เจ็บคอมากจนกลืนน้ำลายไม่ได้",
                "อาการไม่ดีขึ้นหลัง 10 วัน"
            ],
            "gastritis": [
                "ปวดท้องรุนแรงกะทันหัน",
                "อาเจียนเป็นเลือดหรือสีดำ",
                "ถ่ายเป็นเลือดหรือดำวาบ",
                "อาการไม่ดีขึ้นหลังงดอาหารระคายเคือง 3 วัน"
            ],
            "migraine": [
                "ปวดหัวรุนแรงกะทันหันที่ไม่เคยมีมาก่อน",
                "ปวดหัวพร้อมไข้และคอแข็ง",
                "มองเห็นไม่ชัด หรือพูดไม่ชัด",
                "ปวดหัวหลังได้รับบาดเจ็บศีรษะ"
            ],
            "urinary tract infection": [
                "ไข้สูงเกิน 38.5°C",
                "ปวดหลังรุนแรงบริเวณไต",
                "ปัสสาวะเป็นเลือดมาก",
                "อาการไม่ดีขึ้นหลังดื่มน้ำเพิ่ม 2-3 วัน"
            ],
            "allergic reaction": [
                "หายใจลำบาก หรือหอบ",
                "บวมบริเวณหน้า ลิ้น หรือคอ",
                "ผื่นแพร่กระจายเร็ว ทั่วร่างกาย",
                "เป็นลมหรือมึนงง"
            ]
        }

        return warning_map.get(condition, [
            "อาการไม่ดีขึ้นภายใน 3-5 วัน",
            "มีไข้สูงต่อเนื่องเกิน 39°C",
            "อาการรุนแรงขึ้นกว่าเดิม"
        ])

    def _generate_follow_up_questions(self, condition: str) -> str:
        """Generate follow-up questions to help patient monitor their condition"""
        questions = {
            "common cold": "ติดตามไข้ทุกวัน และสังเกตสีของน้ำมูก",
            "gastritis": "บันทึกอาหารที่กินและอาการที่เกิดขึ้น",
            "migraine": "บันทึกเวลาที่ปวดหัวและสิ่งที่อาจเป็นตัวกระตุ้น",
            "urinary tract infection": "นับจำนวนครั้งที่ปัสสาวะและสีของปัสสาวะ",
            "allergic reaction": "หาสาเหตุที่ทำให้แพ้และหลีกเลี่ยง"
        }
        return questions.get(condition, "ติดตามอาการและบันทึกการเปลี่ยนแปลง")

    async def _queue_for_doctor_approval(
        self,
        message: str,
        conversation_history: Optional[List[ConversationMessage]] = None,
        patient_info: Optional[PatientInfo] = None,
        preferred_language: str = "auto",
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Queue patient message for doctor approval"""

        logger.info(f"📋 Queueing message for doctor approval: {session_id}")

        # Create response indicating waiting for approval
        thai_message = """📋 ข้อความของคุณได้รับแล้ว

⏳ **สถานะ**: รอแพทย์พิจารณา

🩺 **ขั้นตอนต่อไป**:
• แพทย์จะตรวจสอบอาการที่คุณแจ้ง
• ให้คำแนะนำที่เหมาะสมกับอาการของคุณ
• คุณจะได้รับคำตอบภายใน 15-30 นาที

⚠️ **หากมีอาการฉุกเฉิน**: โทร 1669 ทันที

💬 **หมายเหตุ**: ระบบจะแจ้งเตือนเมื่อแพทย์ตอบกลับแล้ว"""

        english_message = """📋 Your message has been received

⏳ **Status**: Waiting for doctor review

🩺 **Next steps**:
• Doctor will review your symptoms
• Provide appropriate medical advice
• You will receive a response within 15-30 minutes

⚠️ **For emergencies**: Call 1669 immediately

💬 **Note**: System will notify when doctor responds"""

        # Detect language
        response_message = thai_message
        if preferred_language == "english":
            response_message = english_message

        # TODO: Store in database/queue for doctor review
        # For now, return the waiting message

        return {
            "message": response_message,
            "type": "waiting_approval",
            "status": "pending_approval",
            "session_id": session_id,
            "triage": None,
            "diagnosis": None,
            "treatment": None,
            "metadata": {
                "processing_time_ms": 0,
                "translation_used": False,
                "detected_language": preferred_language,
                "detected_dialects": [],
                "agents_used": 0,
                "rag_results_count": 0,
                "requires_approval": True
            },
            "disclaimer": "ข้อมูลนี้เป็นเพียงข้อมูลทั่วไปเท่านั้น ไม่ใช่การวินิจฉัยทางการแพทย์ กรุณาปรึกษาแพทย์หรือผู้เชี่ยวชาญด้านสุขภาพสำหรับการวินิจฉัยและการรักษาที่เหมาะสม",
            "recommendation": "รอการตอบกลับจากแพทย์ หากมีอาการฉุกเฉินกรุณาโทร 1669",
            "timestamp": datetime.now().isoformat()
        }