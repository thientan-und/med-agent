#!/usr/bin/env python3
"""
RAG-Enhanced Few-Shot Learning Service

This service combines Retrieval-Augmented Generation (RAG) with few-shot learning
to dynamically generate contextually relevant medical examples from the knowledge base.

Key Features:
1. Knowledge Base Retrieval: Searches medical data for relevant cases
2. Dynamic Few-Shot Generation: Creates examples on-the-fly from retrieved knowledge
3. Context-Aware Matching: Uses symptoms, patient data, and medical history
4. Safety-First Approach: Prioritizes common conditions over rare/serious ones
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import csv
import os
from pathlib import Path

from app.util.config import get_settings
from app.services.memory_agent import memory_agent

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class KnowledgeItem:
    """Structured medical knowledge item from CSV data"""
    id: str
    name_en: str
    name_th: str
    icd_code: Optional[str]
    category: str
    keywords: List[str]
    frequency: str  # common, uncommon, rare
    severity: str   # mild, moderate, severe
    urgency: str    # low, medium, high, emergency
    symptoms: List[str]
    treatments: List[str]
    confidence_score: float = 0.0


@dataclass
class RAGFewShotExample:
    """Dynamically generated few-shot example from RAG"""
    id: str
    source: str  # "knowledge_base", "doctor_feedback", "training_data"
    symptoms_thai: str
    symptoms_english: str
    diagnosis: Dict[str, Any]
    treatment: Dict[str, Any]
    key_indicators: List[str]
    safety_notes: List[str]
    confidence_level: float
    retrieval_score: float
    patient_context: Dict[str, Any] = None


class RAGFewShotService:
    """RAG-Enhanced Few-Shot Learning for Medical AI"""

    def __init__(self):
        self.knowledge_base: List[KnowledgeItem] = []
        self.doctor_feedback: List[Dict] = []
        self.training_examples: List[Dict] = []
        self.symptom_embeddings: Dict[str, List[float]] = {}
        self.initialized = False

        # Safety configuration
        self.safety_config = {
            "common_conditions_boost": 2.0,  # Boost score for common conditions
            "serious_conditions_penalty": 0.5,  # Reduce score for serious conditions
            "min_confidence_threshold": 0.6,
            "max_examples_per_query": 5,
            "require_symptom_overlap": True
        }

        logger.info("🧠 RAG Few-Shot Service initialized")

    async def initialize(self):
        """Initialize RAG knowledge base and embeddings"""
        if self.initialized:
            return

        logger.info("🔄 Initializing RAG Few-Shot Learning...")

        try:
            # Load knowledge bases
            await self._load_medical_knowledge()
            await self._load_doctor_feedback()
            await self._load_training_data()

            # Build symptom embeddings (simple keyword-based for now)
            await self._build_symptom_embeddings()

            self.initialized = True
            logger.info(f"✅ RAG Few-Shot initialized with {len(self.knowledge_base)} knowledge items")

        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG Few-Shot: {e}")
            raise

    async def get_relevant_examples(self,
                                  symptoms: str,
                                  patient_data: Dict = None,
                                  max_examples: int = 3) -> List[RAGFewShotExample]:
        """
        Retrieve and generate contextually relevant few-shot examples
        using RAG approach from knowledge base
        """
        if not self.initialized:
            await self.initialize()

        logger.info(f"🔍 RAG retrieval for symptoms: {symptoms[:100]}...")

        # 1. Extract key symptoms and context
        extracted_symptoms = await self._extract_symptoms(symptoms)
        patient_context = patient_data or {}

        # 2. Retrieve relevant knowledge items
        retrieved_items = await self._retrieve_knowledge(
            extracted_symptoms,
            patient_context,
            max_items=max_examples * 3  # Retrieve more, then filter
        )

        # 3. Generate few-shot examples from retrieved items
        examples = []
        for item in retrieved_items[:max_examples]:
            example = await self._generate_few_shot_example(item, symptoms, patient_context)
            if example:
                examples.append(example)

        # 4. Apply safety filtering and ranking
        safe_examples = await self._apply_safety_filtering(examples, symptoms)

        logger.info(f"✅ Generated {len(safe_examples)} RAG few-shot examples")
        return safe_examples

    async def _load_medical_knowledge(self):
        """Load medical knowledge from CSV files and convert to structured format"""
        logger.info("📚 Loading medical knowledge base...")

        # Load diagnoses with enhanced metadata
        if os.path.exists(settings.diagnosis_data_path):
            logger.info(f"📋 Loading diagnoses from: {settings.diagnosis_data_path}")
            with open(settings.diagnosis_data_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Use our CSV column names
                    diagnosis_id = row.get('diagnosis_id', '')
                    condition_name = row.get('condition_name', '')
                    thai_name = row.get('thai_name', '')
                    icd_code = row.get('icd_code', '')
                    category = row.get('category', '')
                    common_symptoms = row.get('common_symptoms', '')

                    # Parse symptoms from CSV
                    symptoms = [s.strip() for s in common_symptoms.split(',') if s.strip()] if common_symptoms else []

                    # Categorize condition by frequency and severity
                    frequency, severity, urgency = self._categorize_condition(condition_name, icd_code)

                    # Generate keywords from condition name and symptoms
                    keywords = [condition_name.lower(), thai_name] + symptoms[:3]
                    keywords = [k for k in keywords if k]  # Remove empty strings

                    knowledge_item = KnowledgeItem(
                        id=diagnosis_id,
                        name_en=condition_name,
                        name_th=thai_name,
                        icd_code=icd_code,
                        category=category or 'diagnosis',
                        keywords=keywords,
                        frequency=frequency,
                        severity=severity,
                        urgency=urgency,
                        symptoms=symptoms,
                        treatments=[]  # Will be populated from treatment data
                    )

                    self.knowledge_base.append(knowledge_item)

        # Load treatments and associate with diagnoses
        if os.path.exists(settings.treatment_data_path):
            logger.info(f"💊 Loading treatments from: {settings.treatment_data_path}")
            with open(settings.treatment_data_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Use our CSV column names
                    condition_name = row.get('condition_name', '')
                    medications = row.get('medications', '')
                    treatment_approach = row.get('treatment_approach', '')

                    # Find matching diagnosis in knowledge base
                    for item in self.knowledge_base:
                        if condition_name and condition_name.lower() == item.name_en.lower():
                            if medications and medications not in item.treatments:
                                item.treatments.append(medications)
                            if treatment_approach and treatment_approach not in item.treatments:
                                item.treatments.append(treatment_approach)

        # Load medicines as additional knowledge items
        if os.path.exists(settings.medicine_data_path):
            logger.info(f"💉 Loading medicines from: {settings.medicine_data_path}")
            with open(settings.medicine_data_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    medicine_name = row.get('medicine_name', '')
                    thai_name = row.get('thai_name', '')
                    indications = row.get('indications', '')
                    category = row.get('category', '')

                    # Parse indications as symptoms/keywords
                    symptoms = [s.strip() for s in indications.split(',') if s.strip()] if indications else []
                    keywords = [medicine_name.lower(), thai_name, category] + symptoms[:2]
                    keywords = [k for k in keywords if k]

                    knowledge_item = KnowledgeItem(
                        id=row.get('medicine_id', ''),
                        name_en=medicine_name,
                        name_th=thai_name,
                        icd_code='',
                        category='medicine',
                        keywords=keywords,
                        frequency='common',  # Medicines are generally common
                        severity='mild',
                        urgency='low',
                        symptoms=symptoms,
                        treatments=[f"{medicine_name} - {row.get('adult_dosage', '')}"]
                    )

                    self.knowledge_base.append(knowledge_item)

        logger.info(f"📊 Loaded {len(self.knowledge_base)} knowledge items")

    async def _load_doctor_feedback(self):
        """Load doctor feedback for learning from corrections"""
        if os.path.exists(settings.feedback_data_path):
            with open(settings.feedback_data_path, 'r', encoding='utf-8') as f:
                self.doctor_feedback = json.load(f)
        logger.info(f"👩‍⚕️ Loaded {len(self.doctor_feedback)} doctor feedback entries")

    async def _load_training_data(self):
        """Load enhanced training data"""
        if os.path.exists(settings.training_data_path):
            with open(settings.training_data_path, 'r', encoding='utf-8') as f:
                self.training_examples = json.load(f)
        logger.info(f"📖 Loaded {len(self.training_examples)} training examples")

    async def _build_symptom_embeddings(self):
        """Build simple keyword-based embeddings for symptom matching"""
        logger.info("🔤 Building symptom embeddings...")

        all_symptoms = set()

        # Collect all symptoms from knowledge base
        for item in self.knowledge_base:
            all_symptoms.update(item.symptoms)
            all_symptoms.update(item.keywords)

        # Create simple keyword-based embeddings
        for symptom in all_symptoms:
            # For now, use simple keyword matching
            # In production, you'd use actual embeddings (e.g., sentence-transformers)
            self.symptom_embeddings[symptom.lower()] = [1.0] * len(symptom.split())

        logger.info(f"🔤 Built embeddings for {len(self.symptom_embeddings)} symptoms")

    def _categorize_condition(self, english_name: str, icd_code: str) -> Tuple[str, str, str]:
        """Categorize medical condition by frequency, severity, and urgency"""
        english_lower = english_name.lower()

        # Frequency classification
        if any(term in english_lower for term in ['common', 'cold', 'flu', 'cough', 'fever', 'headache']):
            frequency = 'common'
        elif any(term in english_lower for term in ['infection', 'inflammation', 'acute']):
            frequency = 'uncommon'
        else:
            frequency = 'rare'

        # Severity classification
        if any(term in english_lower for term in ['cancer', 'tumor', 'stroke', 'heart attack', 'sepsis']):
            severity = 'severe'
        elif any(term in english_lower for term in ['pneumonia', 'diabetes', 'hypertension']):
            severity = 'moderate'
        else:
            severity = 'mild'

        # Urgency classification
        if any(term in english_lower for term in ['emergency', 'acute', 'severe', 'critical']):
            urgency = 'emergency'
        elif any(term in english_lower for term in ['urgent', 'moderate']):
            urgency = 'high'
        elif severity == 'moderate':
            urgency = 'medium'
        else:
            urgency = 'low'

        return frequency, severity, urgency

    def _generate_symptoms_keywords(self, english_name: str, thai_name: str) -> Tuple[List[str], List[str]]:
        """Generate symptoms and keywords from condition names"""

        # Common symptom mappings
        symptom_mappings = {
            'cold': ['ไข้', 'ไอ', 'น้ำมูก', 'fever', 'cough', 'runny nose'],
            'flu': ['ไข้', 'ไอ', 'เมื่อยตัว', 'fever', 'cough', 'body aches'],
            'pneumonia': ['ไข้', 'ไอ', 'หายใจลำบาก', 'fever', 'cough', 'shortness of breath'],
            'diabetes': ['กระหายน้ำ', 'ปัสสาวะบ่อย', 'น้ำหนักลด', 'thirst', 'frequent urination', 'weight loss'],
            'hypertension': ['ปวดหัว', 'วิงเวียน', 'headache', 'dizziness'],
            'asthma': ['หอบหืด', 'หายใจลำบาก', 'เสียงหวีด', 'wheezing', 'shortness of breath']
        }

        symptoms = []
        keywords = []

        english_lower = english_name.lower()

        # Find matching symptoms
        for condition, condition_symptoms in symptom_mappings.items():
            if condition in english_lower:
                symptoms.extend(condition_symptoms)
                keywords.extend([condition])

        # Add condition names as keywords
        keywords.extend([english_name.lower(), thai_name])

        # Remove duplicates and empty strings
        symptoms = list(set([s for s in symptoms if s]))
        keywords = list(set([k for k in keywords if k]))

        return symptoms, keywords

    async def _extract_symptoms(self, symptoms_text: str) -> List[str]:
        """Extract key symptoms from input text"""

        # Common Thai-English symptom mappings
        symptom_mappings = {
            'ไข้': 'fever',
            'ไอ': 'cough',
            'น้ำมูก': 'runny nose',
            'ปวดหัว': 'headache',
            'เจ็บคอ': 'sore throat',
            'หายใจลำบาก': 'shortness of breath',
            'ปวดหน้าอก': 'chest pain',
            'วิงเวียน': 'dizziness',
            'เมื่อยตัว': 'body aches'
        }

        extracted = []
        symptoms_lower = symptoms_text.lower()

        # Extract Thai symptoms
        for thai, english in symptom_mappings.items():
            if thai in symptoms_text:
                extracted.extend([thai, english])

        # Extract English symptoms (simple keyword matching)
        for english in symptom_mappings.values():
            if english in symptoms_lower:
                extracted.append(english)

        # Add original words as potential symptoms
        words = symptoms_text.split()
        for word in words:
            if len(word) > 2:  # Only meaningful words
                extracted.append(word.lower())

        return list(set(extracted))

    async def _retrieve_knowledge(self,
                                symptoms: List[str],
                                patient_context: Dict,
                                max_items: int = 10) -> List[KnowledgeItem]:
        """Retrieve relevant knowledge items using symptom matching"""

        scored_items = []

        for item in self.knowledge_base:
            score = await self._calculate_relevance_score(item, symptoms, patient_context)
            if score > 0:
                item.confidence_score = score
                scored_items.append((score, item))

        # Sort by score and apply safety boost/penalty
        scored_items.sort(key=lambda x: x[0], reverse=True)

        # Apply safety adjustments
        adjusted_items = []
        for score, item in scored_items:
            adjusted_score = self._apply_safety_adjustment(score, item)
            adjusted_items.append((adjusted_score, item))

        # Re-sort with safety adjustments
        adjusted_items.sort(key=lambda x: x[0], reverse=True)

        return [item for score, item in adjusted_items[:max_items]]

    async def _calculate_relevance_score(self,
                                       item: KnowledgeItem,
                                       symptoms: List[str],
                                       patient_context: Dict) -> float:
        """Calculate relevance score for knowledge item"""

        score = 0.0

        # Symptom matching score
        item_symptoms = item.symptoms + item.keywords
        for symptom in symptoms:
            for item_symptom in item_symptoms:
                if symptom.lower() in item_symptom.lower() or item_symptom.lower() in symptom.lower():
                    score += 1.0

        # Normalize by number of symptoms
        if len(symptoms) > 0:
            score = score / len(symptoms)

        # Patient context matching (age, gender specific conditions)
        if patient_context:
            age = patient_context.get('age', 0)
            gender = patient_context.get('gender', '')

            # Age-specific adjustments
            if age < 18 and 'pediatric' in item.name_en.lower():
                score += 0.5
            elif age > 65 and 'geriatric' in item.name_en.lower():
                score += 0.5

            # Gender-specific adjustments
            if gender == 'female' and any(term in item.name_en.lower() for term in ['pregnancy', 'menstrual']):
                score += 0.3

        return score

    def _apply_safety_adjustment(self, score: float, item: KnowledgeItem) -> float:
        """Apply safety-first adjustments to relevance scores"""

        adjusted_score = score

        # Boost common conditions
        if item.frequency == 'common':
            adjusted_score *= self.safety_config["common_conditions_boost"]

        # Penalize serious/rare conditions unless high symptom match
        if item.severity == 'severe' or item.frequency == 'rare':
            if score < 0.8:  # Low symptom match
                adjusted_score *= self.safety_config["serious_conditions_penalty"]

        # Boost low urgency conditions for safety
        if item.urgency == 'low':
            adjusted_score *= 1.2

        return adjusted_score

    async def _generate_few_shot_example(self,
                                       item: KnowledgeItem,
                                       original_symptoms: str,
                                       patient_context: Dict) -> Optional[RAGFewShotExample]:
        """Generate a few-shot example from knowledge item"""

        try:
            # Generate symptom description
            symptoms_thai = self._generate_symptom_description(item, 'thai')
            symptoms_english = self._generate_symptom_description(item, 'english')

            # Create diagnosis structure
            diagnosis = {
                "icd_code": item.icd_code or "unknown",
                "name": f"{item.name_th} ({item.name_en})",
                "confidence": max(0.6, min(0.95, item.confidence_score)),
                "urgency": item.urgency
            }

            # Create treatment structure
            treatment = {
                "medications": item.treatments[:3] if item.treatments else ["รักษาตามอาการ"],
                "recommendations": self._generate_recommendations(item),
                "follow_up": self._generate_follow_up(item)
            }

            # Safety notes
            safety_notes = self._generate_safety_notes(item)

            example = RAGFewShotExample(
                id=f"rag_{item.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                source="knowledge_base",
                symptoms_thai=symptoms_thai,
                symptoms_english=symptoms_english,
                diagnosis=diagnosis,
                treatment=treatment,
                key_indicators=item.keywords[:5],
                safety_notes=safety_notes,
                confidence_level=item.confidence_score,
                retrieval_score=item.confidence_score,
                patient_context=patient_context
            )

            return example

        except Exception as e:
            logger.error(f"Error generating few-shot example: {e}")
            return None

    def _generate_symptom_description(self, item: KnowledgeItem, language: str) -> str:
        """Generate natural symptom description"""

        if language == 'thai':
            if item.symptoms:
                thai_symptoms = [s for s in item.symptoms if any(ord(c) > 127 for c in s)]  # Thai characters
                if thai_symptoms:
                    return " ".join(thai_symptoms[:4])
            return item.name_th or "อาการทั่วไป"

        else:  # English
            if item.symptoms:
                english_symptoms = [s for s in item.symptoms if all(ord(c) <= 127 for c in s)]  # ASCII only
                if english_symptoms:
                    return " ".join(english_symptoms[:4])
            return item.name_en or "general symptoms"

    def _generate_recommendations(self, item: KnowledgeItem) -> List[str]:
        """Generate treatment recommendations"""

        recommendations = ["เพียงพอนอนพัก", "ดื่มน้ำให้เพียงพอ"]  # Rest, hydration

        if item.urgency == 'emergency':
            recommendations = ["ไปโรงพยาบาลทันที", "โทร 1669"]
        elif item.urgency == 'high':
            recommendations.append("ปรึกษาแพทย์")
        elif item.severity == 'mild':
            recommendations.extend(["รักษาอาการ", "สังเกตอาการ"])

        return recommendations

    def _generate_follow_up(self, item: KnowledgeItem) -> str:
        """Generate follow-up instructions"""

        if item.urgency == 'emergency':
            return "ติดตามอาการทันที"
        elif item.urgency == 'high':
            return "ติดตามอาการใน 24-48 ชั่วโมง"
        elif item.frequency == 'common':
            return "ติดตามอาการใน 3-7 วัน"
        else:
            return "ติดตามอาการตามแพทย์แนะนำ"

    def _generate_safety_notes(self, item: KnowledgeItem) -> List[str]:
        """Generate safety notes for the condition"""

        notes = []

        if item.severity == 'severe':
            notes.append("⚠️ อาการร้อนแรง ต้องระวัง")

        if item.urgency == 'emergency':
            notes.append("🚨 ฉุกเฉิน ต้องรักษาทันที")

        if item.frequency == 'rare':
            notes.append("📋 ต้องตรวจเพิ่มเติม")

        if not notes:
            notes.append("💡 ติดตามอาการสม่ำเสมอ")

        return notes

    async def _apply_safety_filtering(self,
                                    examples: List[RAGFewShotExample],
                                    original_symptoms: str) -> List[RAGFewShotExample]:
        """Apply final safety filtering to generated examples"""

        safe_examples = []

        for example in examples:
            # Check minimum confidence
            if example.confidence_level < self.safety_config["min_confidence_threshold"]:
                continue

            # Check for dangerous mismatches (e.g., serious diagnosis for mild symptoms)
            if await self._is_safe_example(example, original_symptoms):
                safe_examples.append(example)

        # Limit number of examples
        return safe_examples[:self.safety_config["max_examples_per_query"]]

    async def _is_safe_example(self, example: RAGFewShotExample, original_symptoms: str) -> bool:
        """Enhanced safety check for examples with stronger guardrails"""

        diagnosis_name = example.diagnosis.get('name', '').lower()
        symptoms_lower = original_symptoms.lower()

        # Enhanced serious condition detection
        serious_conditions = [
            'cancer', 'tumor', 'stroke', 'heart attack', 'วัณโรค', 'มะเร็ง',
            'tuberculosis', 'tb', 'meningitis', 'เยื่อหุ้มสมอง', 'sepsis',
            'brain tumor', 'lung cancer', 'liver cancer'
        ]

        # Enhanced mild symptom detection
        mild_indicators = [
            'เล็กน้อย', 'mild', '38 องศา', 'สองสามวัน', 'วันเดียว',
            'น้ำมูกเขียว', 'น้ำมูกใส', 'เมื่อย', 'minor'
        ]

        # Enhanced serious symptom requirements for serious diagnoses
        serious_symptom_requirements = {
            'tuberculosis': ['ไอเป็นเลือด', 'hemoptysis', 'blood', 'เลือด'],
            'วัณโรค': ['ไอเป็นเลือด', 'hemoptysis', 'blood', 'เลือด'],
            'meningitis': ['ชัก', 'seizure', 'แข็งทื่อ', 'stiff neck', 'ไข้สูง', 'high fever'],
            'เยื่อหุ้มสมอง': ['ชัก', 'seizure', 'แข็งทื่อ', 'stiff neck', 'ไข้สูง', 'high fever'],
            'cancer': ['น้ำหนักลดมาก', 'severe weight loss', 'มวล', 'mass', 'ก้อน'],
            'มะเร็ง': ['น้ำหนักลดมาก', 'severe weight loss', 'มวล', 'mass', 'ก้อน']
        }

        # Check 1: Block serious diagnoses for clearly mild symptoms
        has_mild_symptoms = any(mild in symptoms_lower for mild in mild_indicators)
        has_serious_diagnosis = any(serious in diagnosis_name for serious in serious_conditions)

        if has_mild_symptoms and has_serious_diagnosis:
            logger.warning(f"🚫 GUARDRAIL: Blocking serious diagnosis '{diagnosis_name}' for mild symptoms '{original_symptoms}'")
            return False

        # Check 2: Require specific serious symptoms for serious diagnoses
        for serious_condition, required_symptoms in serious_symptom_requirements.items():
            if serious_condition in diagnosis_name:
                has_required_symptom = any(req in symptoms_lower for req in required_symptoms)
                if not has_required_symptom:
                    logger.warning(f"🚫 GUARDRAIL: Blocking '{serious_condition}' diagnosis - missing required symptoms {required_symptoms}")
                    return False

        # Check 3: Confidence threshold for serious conditions
        if has_serious_diagnosis and example.confidence_level < 0.8:
            logger.warning(f"🚫 GUARDRAIL: Blocking low-confidence serious diagnosis '{diagnosis_name}' (confidence: {example.confidence_level:.2f})")
            return False

        # Check 4: Special headache safety check
        if 'ปวดหัว' in symptoms_lower and 'headache' in symptoms_lower:
            if any(dangerous in diagnosis_name for dangerous in ['meningitis', 'เยื่อหุ้มสมอง', 'brain tumor', 'stroke']):
                # Require specific danger signs for serious headache diagnoses
                danger_signs = ['ชัก', 'seizure', 'แข็งทื่อ', 'stiff neck', 'ไข้สูงมาก', 'severe fever', 'อาเจียน', 'vomiting']
                has_danger_signs = any(sign in symptoms_lower for sign in danger_signs)
                if not has_danger_signs:
                    logger.warning(f"🚫 GUARDRAIL: Blocking serious headache diagnosis '{diagnosis_name}' without danger signs")
                    return False

        return True

    async def enhance_few_shot_learning(self,
                                      base_examples: List[Dict],
                                      symptoms: str,
                                      patient_data: Dict = None) -> List[Dict]:
        """Enhance existing few-shot examples with RAG-retrieved knowledge"""

        # Get RAG examples
        rag_examples = await self.get_relevant_examples(symptoms, patient_data, max_examples=2)

        # Convert RAG examples to standard format
        enhanced_examples = []

        # Add original examples
        enhanced_examples.extend(base_examples)

        # Add RAG examples
        for rag_example in rag_examples:
            standard_example = {
                "id": rag_example.id,
                "symptoms_thai": rag_example.symptoms_thai,
                "symptoms_english": rag_example.symptoms_english,
                "diagnosis": rag_example.diagnosis,
                "treatment": rag_example.treatment,
                "key_indicators": rag_example.key_indicators,
                "safety_notes": rag_example.safety_notes,
                "confidence": rag_example.confidence_level,
                "source": "RAG_knowledge_base"
            }
            enhanced_examples.append(standard_example)

        logger.info(f"🔗 Enhanced few-shot learning: {len(base_examples)} base + {len(rag_examples)} RAG examples")
        return enhanced_examples


# Global RAG few-shot service instance
rag_few_shot_service = RAGFewShotService()