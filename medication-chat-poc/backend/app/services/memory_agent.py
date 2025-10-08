"""
AI Agent Memory and Adaptability System
========================================
Implements learning, memory, and adaptation capabilities for medical AI agents
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import hashlib
import logging

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Types of memory storage"""
    EPISODIC = "episodic"       # Specific patient interactions
    SEMANTIC = "semantic"       # Medical knowledge and patterns
    PROCEDURAL = "procedural"   # How to diagnose/treat
    WORKING = "working"         # Current conversation context

class LearningType(Enum):
    """Types of learning mechanisms"""
    SUPERVISED = "supervised"       # From doctor feedback
    REINFORCEMENT = "reinforcement" # From outcome tracking
    TRANSFER = "transfer"          # From similar cases
    CONTINUOUS = "continuous"      # Ongoing adaptation

@dataclass
class MemoryUnit:
    """Individual memory unit"""
    id: str
    type: MemoryType
    content: Dict[str, Any]
    timestamp: datetime
    confidence: float
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    decay_rate: float = 0.1
    importance: float = 1.0
    embeddings: Optional[List[float]] = None

    def calculate_relevance(self, current_time: datetime) -> float:
        """Calculate memory relevance based on recency and importance"""
        if not self.last_accessed:
            self.last_accessed = self.timestamp

        time_diff = (current_time - self.last_accessed).total_seconds() / 3600  # hours
        recency_score = np.exp(-self.decay_rate * time_diff)
        frequency_score = min(1.0, self.access_count / 10)

        return (self.importance * 0.4 +
                recency_score * 0.3 +
                frequency_score * 0.2 +
                self.confidence * 0.1)

@dataclass
class PatientMemory:
    """Patient-specific memory tracking"""
    patient_id: str
    medical_history: List[Dict[str, Any]]
    interaction_history: List[Dict[str, Any]]
    preferences: Dict[str, Any]
    risk_factors: List[str]
    successful_treatments: List[Dict[str, Any]]
    allergies: List[str]
    last_visit: datetime
    total_interactions: int = 0

    def get_relevant_history(self, current_symptoms: str) -> List[Dict[str, Any]]:
        """Get relevant medical history for current symptoms"""
        relevant = []
        for history in self.medical_history:
            if any(symptom in current_symptoms.lower()
                   for symptom in history.get('symptoms', '').lower().split()):
                relevant.append(history)
        return relevant[-5:]  # Return last 5 relevant cases

class AdaptiveMemoryAgent:
    """Main memory and adaptation agent for medical AI"""

    def __init__(self,
                 max_short_term: int = 100,
                 max_long_term: int = 10000,
                 learning_rate: float = 0.1):

        # Memory stores
        self.short_term_memory = deque(maxlen=max_short_term)
        self.long_term_memory: Dict[str, MemoryUnit] = {}
        self.patient_memories: Dict[str, PatientMemory] = {}

        # Learning components
        self.learning_rate = learning_rate
        self.diagnosis_patterns: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.treatment_effectiveness: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.doctor_corrections: List[Dict[str, Any]] = []

        # Adaptation mechanisms
        self.confidence_thresholds = {
            'high': 0.85,
            'medium': 0.60,
            'low': 0.40
        }
        self.adaptation_weights = {
            'symptom_match': 0.3,
            'historical_success': 0.25,
            'doctor_feedback': 0.25,
            'pattern_recognition': 0.2
        }

        # Case-based reasoning
        self.case_library: List[Dict[str, Any]] = []
        self.max_cases = 1000

        # Few-shot examples for better diagnosis
        self.few_shot_examples = self._initialize_few_shot_examples()

        logger.info("🧠 Adaptive Memory Agent initialized")

    def _initialize_few_shot_examples(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize few-shot learning examples for common conditions"""
        return {
            'arthritis': [
                {
                    'symptoms': 'ปวดข้อ บวม แดง ร้อน ที่ข้อเข่า ข้อติดตอนเช้า',
                    'diagnosis': {
                        'icd_code': 'M19.90',
                        'name': 'ข้ออักเสบ (Osteoarthritis)',
                        'confidence': 0.85
                    },
                    'key_indicators': ['ปวดข้อ', 'บวม', 'แดง', 'ข้อติด'],
                    'treatment': {
                        'medications': ['NSAIDs', 'พาราเซตามอล'],
                        'lifestyle': ['ลดน้ำหนัก', 'กายภาพบำบัด', 'ประคบร้อนเย็น']
                    }
                },
                {
                    'symptoms': 'ปวดข้อหลายข้อ มีไข้ ข้อบวมสลับกัน',
                    'diagnosis': {
                        'icd_code': 'M06.9',
                        'name': 'รูมาตอยด์ (Rheumatoid Arthritis)',
                        'confidence': 0.80
                    },
                    'key_indicators': ['หลายข้อ', 'มีไข้', 'สลับกัน'],
                    'treatment': {
                        'medications': ['DMARDs', 'Corticosteroids'],
                        'lifestyle': ['พักผ่อน', 'ออกกำลังกายเบาๆ']
                    }
                }
            ],
            'diabetes': [
                {
                    'symptoms': 'เบาหวาน น้ำตาลสูง 300 ปัสสาวะบ่อย กระหายน้ำ',
                    'diagnosis': {
                        'icd_code': 'E11.9',
                        'name': 'เบาหวานชนิดที่ 2 (Type 2 Diabetes)',
                        'confidence': 0.90
                    },
                    'key_indicators': ['น้ำตาลสูง', 'ปัสสาวะบ่อย', 'กระหายน้ำ', 'เบาหวาน'],
                    'treatment': {
                        'medications': ['Metformin', 'Insulin'],
                        'lifestyle': ['ควบคุมอาหาร', 'ออกกำลังกาย', 'ตรวจน้ำตาลสม่ำเสมอ']
                    }
                },
                {
                    'symptoms': 'น้ำหนักลด ปัสสาวะบ่อย มองเห็นไม่ชัด',
                    'diagnosis': {
                        'icd_code': 'E10.9',
                        'name': 'เบาหวานชนิดที่ 1 (Type 1 Diabetes)',
                        'confidence': 0.75
                    },
                    'key_indicators': ['น้ำหนักลด', 'มองเห็นไม่ชัด'],
                    'treatment': {
                        'medications': ['Insulin therapy'],
                        'lifestyle': ['นับคาร์โบไฮเดรต', 'ตรวจน้ำตาลก่อนมื้ออาหาร']
                    }
                }
            ],
            'hypertension': [
                {
                    'symptoms': 'ปวดหัว ตาพร่า คลื่นไส้ ความดันสูง 180/110',
                    'diagnosis': {
                        'icd_code': 'I10',
                        'name': 'ความดันโลหิตสูง (Hypertension)',
                        'confidence': 0.95
                    },
                    'key_indicators': ['ความดันสูง', 'ปวดหัว', 'ตาพร่า'],
                    'treatment': {
                        'medications': ['ACE inhibitors', 'Beta blockers'],
                        'lifestyle': ['ลดเกลือ', 'ออกกำลังกาย', 'ลดความเครียด']
                    }
                }
            ]
        }

    async def store_interaction(self,
                                session_id: str,
                                patient_id: Optional[str],
                                symptoms: str,
                                diagnosis: Dict[str, Any],
                                treatment: Dict[str, Any],
                                doctor_feedback: Optional[Dict[str, Any]] = None) -> None:
        """Store interaction in memory for learning"""

        # Create memory unit
        memory_id = hashlib.md5(f"{session_id}_{datetime.now().isoformat()}".encode()).hexdigest()

        memory = MemoryUnit(
            id=memory_id,
            type=MemoryType.EPISODIC,
            content={
                'session_id': session_id,
                'patient_id': patient_id,
                'symptoms': symptoms,
                'diagnosis': diagnosis,
                'treatment': treatment,
                'doctor_feedback': doctor_feedback,
                'timestamp': datetime.now().isoformat()
            },
            timestamp=datetime.now(),
            confidence=diagnosis.get('confidence', 0.5),
            importance=self._calculate_importance(diagnosis, doctor_feedback)
        )

        # Store in short-term memory
        self.short_term_memory.append(memory)

        # Move to long-term if important
        if memory.importance > 0.7:
            self.long_term_memory[memory_id] = memory

        # Update patient memory
        if patient_id:
            await self._update_patient_memory(patient_id, symptoms, diagnosis, treatment)

        # Learn from interaction
        await self._learn_from_interaction(symptoms, diagnosis, treatment, doctor_feedback)

        logger.info(f"💾 Stored interaction {memory_id} with importance {memory.importance:.2f}")

    async def _update_patient_memory(self,
                                     patient_id: str,
                                     symptoms: str,
                                     diagnosis: Dict[str, Any],
                                     treatment: Dict[str, Any]) -> None:
        """Update patient-specific memory"""

        if patient_id not in self.patient_memories:
            self.patient_memories[patient_id] = PatientMemory(
                patient_id=patient_id,
                medical_history=[],
                interaction_history=[],
                preferences={},
                risk_factors=[],
                successful_treatments=[],
                allergies=[],
                last_visit=datetime.now()
            )

        patient_memory = self.patient_memories[patient_id]

        # Add to medical history
        patient_memory.medical_history.append({
            'date': datetime.now().isoformat(),
            'symptoms': symptoms,
            'diagnosis': diagnosis,
            'treatment': treatment
        })

        # Update interaction count
        patient_memory.total_interactions += 1
        patient_memory.last_visit = datetime.now()

        # Extract risk factors
        if 'เบาหวาน' in symptoms or 'diabetes' in str(diagnosis).lower():
            if 'diabetes' not in patient_memory.risk_factors:
                patient_memory.risk_factors.append('diabetes')

        if 'ความดัน' in symptoms or 'hypertension' in str(diagnosis).lower():
            if 'hypertension' not in patient_memory.risk_factors:
                patient_memory.risk_factors.append('hypertension')

    async def _learn_from_interaction(self,
                                      symptoms: str,
                                      diagnosis: Dict[str, Any],
                                      treatment: Dict[str, Any],
                                      doctor_feedback: Optional[Dict[str, Any]]) -> None:
        """Learn patterns from interactions"""

        # Update diagnosis patterns
        symptom_tokens = symptoms.lower().split()
        diagnosis_name = diagnosis.get('name', diagnosis.get('primary_diagnosis', 'unknown'))

        for token in symptom_tokens:
            self.diagnosis_patterns[token][diagnosis_name] += self.learning_rate

        # Learn from doctor feedback
        if doctor_feedback:
            if doctor_feedback.get('correction'):
                # Doctor corrected the diagnosis
                correct_diagnosis = doctor_feedback['correct_diagnosis']
                self.doctor_corrections.append({
                    'original': diagnosis_name,
                    'corrected': correct_diagnosis,
                    'symptoms': symptoms,
                    'timestamp': datetime.now().isoformat()
                })

                # Boost learning for corrections
                for token in symptom_tokens:
                    self.diagnosis_patterns[token][correct_diagnosis] += self.learning_rate * 2
                    self.diagnosis_patterns[token][diagnosis_name] -= self.learning_rate * 0.5

    def _calculate_importance(self,
                             diagnosis: Dict[str, Any],
                             doctor_feedback: Optional[Dict[str, Any]]) -> float:
        """Calculate importance score for memory"""

        importance = 0.5  # Base importance

        # High confidence diagnosis
        if diagnosis.get('confidence', 0) > 0.8:
            importance += 0.2

        # Doctor feedback provided
        if doctor_feedback:
            importance += 0.3
            if doctor_feedback.get('correction'):
                importance = 1.0  # Maximum importance for corrections

        # Critical diagnosis
        if diagnosis.get('urgency') == 'high' or diagnosis.get('risk_level') == 'high':
            importance += 0.2

        return min(1.0, importance)

    async def get_adaptive_diagnosis(self,
                                     symptoms: str,
                                     patient_id: Optional[str] = None) -> Dict[str, Any]:
        """Get diagnosis with adaptive learning"""

        # Start with few-shot examples
        best_match = await self._match_few_shot_examples(symptoms)

        # Get patient history if available
        patient_context = None
        if patient_id and patient_id in self.patient_memories:
            patient_memory = self.patient_memories[patient_id]
            patient_context = {
                'risk_factors': patient_memory.risk_factors,
                'allergies': patient_memory.allergies,
                'relevant_history': patient_memory.get_relevant_history(symptoms),
                'successful_treatments': patient_memory.successful_treatments[-3:]
            }

        # Find similar cases from memory
        similar_cases = await self._find_similar_cases(symptoms, limit=5)

        # Calculate adaptive diagnosis
        diagnosis = await self._calculate_adaptive_diagnosis(
            symptoms=symptoms,
            few_shot_match=best_match,
            similar_cases=similar_cases,
            patient_context=patient_context
        )

        # Learn from patterns
        diagnosis['confidence'] = self._adjust_confidence_from_patterns(symptoms, diagnosis)

        return diagnosis

    async def _match_few_shot_examples(self, symptoms: str) -> Optional[Dict[str, Any]]:
        """Match symptoms to few-shot examples"""

        symptoms_lower = symptoms.lower()
        best_match = None
        best_score = 0

        for condition, examples in self.few_shot_examples.items():
            for example in examples:
                # Calculate similarity score
                score = 0
                for indicator in example['key_indicators']:
                    if indicator in symptoms_lower:
                        score += 1

                # Normalize score
                if len(example['key_indicators']) > 0:
                    score = score / len(example['key_indicators'])

                if score > best_score:
                    best_score = score
                    best_match = example

        if best_score > 0.5:  # Threshold for match
            return best_match

        return None

    async def _find_similar_cases(self,
                                  symptoms: str,
                                  limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar cases from memory"""

        similar_cases = []
        symptoms_tokens = set(symptoms.lower().split())

        # Search in long-term memory
        for memory_id, memory in self.long_term_memory.items():
            if memory.type == MemoryType.EPISODIC:
                case_symptoms = memory.content.get('symptoms', '').lower()
                case_tokens = set(case_symptoms.split())

                # Calculate Jaccard similarity
                intersection = symptoms_tokens.intersection(case_tokens)
                union = symptoms_tokens.union(case_tokens)

                if union:
                    similarity = len(intersection) / len(union)
                    if similarity > 0.3:  # Threshold
                        similar_cases.append({
                            'similarity': similarity,
                            'case': memory.content,
                            'relevance': memory.calculate_relevance(datetime.now())
                        })

        # Sort by similarity and relevance
        similar_cases.sort(key=lambda x: x['similarity'] * 0.6 + x['relevance'] * 0.4, reverse=True)

        return similar_cases[:limit]

    async def _calculate_adaptive_diagnosis(self,
                                           symptoms: str,
                                           few_shot_match: Optional[Dict[str, Any]],
                                           similar_cases: List[Dict[str, Any]],
                                           patient_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate diagnosis using adaptive learning"""

        diagnosis_scores = defaultdict(float)

        # Weight from few-shot examples
        if few_shot_match:
            diagnosis_name = few_shot_match['diagnosis']['name']
            diagnosis_scores[diagnosis_name] += self.adaptation_weights['symptom_match']

        # Weight from similar cases
        for case_info in similar_cases:
            case = case_info['case']
            if 'diagnosis' in case:
                diagnosis_name = case['diagnosis'].get('name', case['diagnosis'].get('primary_diagnosis'))
                if diagnosis_name:
                    weight = case_info['similarity'] * self.adaptation_weights['pattern_recognition']
                    diagnosis_scores[diagnosis_name] += weight

        # Weight from patient history
        if patient_context and patient_context.get('relevant_history'):
            for history in patient_context['relevant_history']:
                if 'diagnosis' in history:
                    diagnosis_name = history['diagnosis'].get('name')
                    if diagnosis_name:
                        diagnosis_scores[diagnosis_name] += self.adaptation_weights['historical_success'] * 0.5

        # Get best diagnosis
        if diagnosis_scores:
            best_diagnosis = max(diagnosis_scores.items(), key=lambda x: x[1])

            # Find matching few-shot or case for full details
            if few_shot_match and few_shot_match['diagnosis']['name'] == best_diagnosis[0]:
                return {
                    'primary_diagnosis': few_shot_match['diagnosis'],
                    'confidence': min(0.95, best_diagnosis[1] / sum(self.adaptation_weights.values())),
                    'treatment': few_shot_match['treatment'],
                    'source': 'few_shot_learning'
                }

            # Look in similar cases
            for case_info in similar_cases:
                case = case_info['case']
                if case.get('diagnosis', {}).get('name') == best_diagnosis[0]:
                    return {
                        'primary_diagnosis': case['diagnosis'],
                        'confidence': min(0.90, best_diagnosis[1] / sum(self.adaptation_weights.values())),
                        'treatment': case.get('treatment', {}),
                        'source': 'case_based_reasoning'
                    }

        return {
            'primary_diagnosis': None,
            'confidence': 0.0,
            'treatment': {},
            'source': 'no_match'
        }

    def _adjust_confidence_from_patterns(self,
                                        symptoms: str,
                                        diagnosis: Dict[str, Any]) -> float:
        """Adjust confidence based on learned patterns"""

        if not diagnosis.get('primary_diagnosis'):
            return 0.0

        base_confidence = diagnosis.get('confidence', 0.5)
        diagnosis_name = diagnosis['primary_diagnosis'].get('name')

        # Check pattern strength
        symptom_tokens = symptoms.lower().split()
        pattern_scores = []

        for token in symptom_tokens:
            if token in self.diagnosis_patterns:
                if diagnosis_name in self.diagnosis_patterns[token]:
                    pattern_scores.append(self.diagnosis_patterns[token][diagnosis_name])

        if pattern_scores:
            avg_pattern_score = sum(pattern_scores) / len(pattern_scores)
            # Adjust confidence based on pattern strength
            adjusted_confidence = base_confidence * (1 + avg_pattern_score * 0.2)
            return min(0.99, adjusted_confidence)

        return base_confidence

    async def learn_from_doctor_feedback(self,
                                        session_id: str,
                                        original_diagnosis: Dict[str, Any],
                                        doctor_feedback: Dict[str, Any]) -> None:
        """Learn from doctor corrections and feedback"""

        # Find the original memory
        original_memory = None
        for memory in self.short_term_memory:
            if memory.content.get('session_id') == session_id:
                original_memory = memory
                break

        if not original_memory:
            for memory_id, memory in self.long_term_memory.items():
                if memory.content.get('session_id') == session_id:
                    original_memory = memory
                    break

        if original_memory:
            # Update memory with feedback
            original_memory.content['doctor_feedback'] = doctor_feedback
            original_memory.importance = 1.0  # Maximum importance

            # Store in long-term memory if not already there
            if original_memory.id not in self.long_term_memory:
                self.long_term_memory[original_memory.id] = original_memory

            # Learn from correction
            symptoms = original_memory.content.get('symptoms', '')
            await self._learn_from_interaction(
                symptoms=symptoms,
                diagnosis=original_diagnosis,
                treatment=original_memory.content.get('treatment', {}),
                doctor_feedback=doctor_feedback
            )

            logger.info(f"📚 Learned from doctor feedback for session {session_id}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""

        return {
            'short_term_count': len(self.short_term_memory),
            'long_term_count': len(self.long_term_memory),
            'patient_count': len(self.patient_memories),
            'learned_patterns': len(self.diagnosis_patterns),
            'doctor_corrections': len(self.doctor_corrections),
            'case_library_size': len(self.case_library),
            'confidence_thresholds': self.confidence_thresholds,
            'adaptation_weights': self.adaptation_weights
        }

    async def consolidate_memory(self) -> None:
        """Consolidate and optimize memory (run periodically)"""

        logger.info("🧹 Starting memory consolidation...")

        # Remove old low-importance memories
        current_time = datetime.now()
        memories_to_remove = []

        for memory_id, memory in self.long_term_memory.items():
            relevance = memory.calculate_relevance(current_time)
            if relevance < 0.1 and memory.importance < 0.5:
                memories_to_remove.append(memory_id)

        for memory_id in memories_to_remove:
            del self.long_term_memory[memory_id]

        # Update case library
        self._update_case_library()

        # Decay old patterns
        for symptom in self.diagnosis_patterns:
            for diagnosis in self.diagnosis_patterns[symptom]:
                self.diagnosis_patterns[symptom][diagnosis] *= 0.95  # Decay factor

        logger.info(f"✅ Memory consolidation complete. Removed {len(memories_to_remove)} low-relevance memories")

    def _update_case_library(self) -> None:
        """Update case library from successful cases"""

        # Extract successful cases from long-term memory
        successful_cases = []

        for memory in self.long_term_memory.values():
            if memory.type == MemoryType.EPISODIC:
                # Check if case was successful (high confidence or doctor approved)
                if memory.confidence > 0.8 or memory.content.get('doctor_feedback', {}).get('approved'):
                    successful_cases.append({
                        'symptoms': memory.content.get('symptoms'),
                        'diagnosis': memory.content.get('diagnosis'),
                        'treatment': memory.content.get('treatment'),
                        'confidence': memory.confidence,
                        'timestamp': memory.timestamp.isoformat()
                    })

        # Keep only the most recent and relevant cases
        successful_cases.sort(key=lambda x: x['timestamp'], reverse=True)
        self.case_library = successful_cases[:self.max_cases]

# Singleton instance
memory_agent = AdaptiveMemoryAgent()