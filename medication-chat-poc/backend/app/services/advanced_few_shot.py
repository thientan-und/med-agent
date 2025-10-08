"""
Advanced Few-Shot Learning System
=================================
Comprehensive few-shot examples for medical AI improvement
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Import RAG few-shot service for dynamic knowledge retrieval
try:
    from app.services.rag_few_shot_service import rag_few_shot_service
    RAG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RAG few-shot service not available: {e}")
    RAG_AVAILABLE = False

class MedicalDomain(Enum):
    """Medical domains for specialized learning"""
    CARDIOVASCULAR = "cardiovascular"
    RESPIRATORY = "respiratory"
    GASTROINTESTINAL = "gastrointestinal"
    NEUROLOGICAL = "neurological"
    MUSCULOSKELETAL = "musculoskeletal"
    ENDOCRINE = "endocrine"
    INFECTIOUS = "infectious"
    DERMATOLOGICAL = "dermatological"
    PSYCHIATRIC = "psychiatric"
    EMERGENCY = "emergency"
    PEDIATRIC = "pediatric"
    GYNECOLOGICAL = "gynecological"

@dataclass
class FewShotExample:
    """Enhanced few-shot learning example"""
    id: str
    domain: MedicalDomain
    symptoms_thai: str
    symptoms_english: str
    diagnosis: Dict[str, Any]
    treatment: Dict[str, Any]
    key_indicators: List[str]
    differential_diagnosis: List[Dict[str, Any]]
    red_flags: List[str]
    confidence_level: float
    complexity: str  # simple, moderate, complex
    learning_notes: str

class AdvancedFewShotLearning:
    """Advanced few-shot learning system with comprehensive medical examples"""

    def __init__(self):
        self.examples = self._initialize_comprehensive_examples()
        self.domain_templates = self._create_domain_templates()
        self.mistake_patterns = self._initialize_mistake_patterns()

        logger.info(f"🧠 Advanced Few-Shot Learning initialized with {len(self.examples)} examples")

    def _initialize_comprehensive_examples(self) -> Dict[str, List[FewShotExample]]:
        """Initialize comprehensive few-shot examples across all medical domains"""

        return {
            # CARDIOVASCULAR DOMAIN
            "cardiovascular": [
                FewShotExample(
                    id="cv_001",
                    domain=MedicalDomain.CARDIOVASCULAR,
                    symptoms_thai="เจ็บหน้าอกกลางๆ ปวดแน่น กดได้ ปวดร้าวไปแขนซ้าย ขากรรไกร หายใจลำบาก เหงื่อแตก",
                    symptoms_english="Central chest pain, pressure, radiating to left arm, jaw, shortness of breath, sweating",
                    diagnosis={
                        "icd_code": "I21.9",
                        "name": "กล้ามเนื้อหัวใจขาดเลือด (STEMI)",
                        "confidence": 0.95,
                        "urgency": "critical"
                    },
                    treatment={
                        "immediate": ["Call 1669", "Aspirin 300mg", "Oxygen", "IV access", "ECG"],
                        "hospital": ["PCI within 90min", "Thrombolysis", "Cardiac monitoring"],
                        "medications": ["Dual antiplatelet", "Beta blocker", "ACE inhibitor", "Statin"]
                    },
                    key_indicators=["เจ็บหน้าอกกลาง", "ปวดร้าวไปแขน", "หายใจลำบาก", "เหงื่อแตก"],
                    differential_diagnosis=[
                        {"name": "Unstable Angina", "icd": "I20.0"},
                        {"name": "Aortic Dissection", "icd": "I71.00"},
                        {"name": "Pulmonary Embolism", "icd": "I26.9"}
                    ],
                    red_flags=["เจ็บหน้าอกรุนแรง", "เหงื่อแตกเย็น", "ปวดร้าวไปแขน"],
                    confidence_level=0.95,
                    complexity="complex",
                    learning_notes="ไม่ผิดพลาดได้ - ต้องส่งโรงพยาบาลทันที"
                ),

                FewShotExample(
                    id="cv_002",
                    domain=MedicalDomain.CARDIOVASCULAR,
                    symptoms_thai="ปวดหัว ตาพร่า คลื่นไส้ ความดัน 180/110 ใจสั่น",
                    symptoms_english="Headache, blurred vision, nausea, BP 180/110, palpitations",
                    diagnosis={
                        "icd_code": "I16.9",
                        "name": "ความดันโลหิตสูงวิกฤต (Hypertensive Crisis)",
                        "confidence": 0.90,
                        "urgency": "high"
                    },
                    treatment={
                        "immediate": ["Emergency room", "BP monitoring", "IV antihypertensives"],
                        "medications": ["Nifedipine", "Labetalol", "Hydralazine"],
                        "target": "Reduce BP by 10-20% in first hour"
                    },
                    key_indicators=["ปวดหัว", "ตาพร่า", "ความดันสูงมาก"],
                    differential_diagnosis=[
                        {"name": "Stroke", "icd": "I64"},
                        {"name": "Kidney disease", "icd": "N18.9"}
                    ],
                    red_flags=["ความดัน > 180/120", "อาการทางสมอง"],
                    confidence_level=0.90,
                    complexity="moderate",
                    learning_notes="ความดันสูงมากกับอาการ = วิกฤต"
                ),

                FewShotExample(
                    id="cv_003",
                    domain=MedicalDomain.CARDIOVASCULAR,
                    symptoms_thai="ขาบวม หายใจลำบาก เมื่อนอน ไอเป็นเลือด อ่อนเพลีย",
                    symptoms_english="Leg swelling, orthopnea, hemoptysis, fatigue",
                    diagnosis={
                        "icd_code": "I50.9",
                        "name": "หัวใจล้มเหลว (Heart Failure)",
                        "confidence": 0.85,
                        "urgency": "high"
                    },
                    treatment={
                        "medications": ["ACE inhibitor", "Diuretics", "Beta blocker"],
                        "lifestyle": ["Salt restriction", "Fluid restriction", "Weight monitoring"]
                    },
                    key_indicators=["ขาบวม", "หายใจลำบากเมื่อนอน", "อ่อนเพลีย"],
                    differential_diagnosis=[
                        {"name": "Kidney disease", "icd": "N18.9"},
                        {"name": "Liver disease", "icd": "K72.9"}
                    ],
                    red_flags=["หายใจลำบากรุนแรง", "ไอเป็นเลือด"],
                    confidence_level=0.85,
                    complexity="moderate",
                    learning_notes="ขาบวม + หายใจลำบาก = หัวใจล้มเหลว"
                )
            ],

            # RESPIRATORY DOMAIN
            "respiratory": [
                # COMMON CONDITIONS FIRST - Critical to prevent serious mismatches
                FewShotExample(
                    id="resp_common_001",
                    domain=MedicalDomain.RESPIRATORY,
                    symptoms_thai="เป็นไข้เล็กน้อย 38 องศา ไอแห้ง น้ำมูกเขียว มาสองสามวัน",
                    symptoms_english="Mild fever 38°C, dry cough, green mucus, for 2-3 days",
                    diagnosis={
                        "icd_code": "J00",
                        "name": "หวัดธรรมดา (Common Cold)",
                        "confidence": 0.95,
                        "urgency": "low"
                    },
                    treatment={
                        "medications": ["Paracetamol", "Throat lozenges", "Nasal decongestant"],
                        "supportive": ["Rest", "Fluids", "Warm compress"],
                        "duration": "7-10 days self-limiting"
                    },
                    key_indicators=["ไข้เล็กน้อย", "ไข้ 38", "ไอแห้ง", "น้ำมูกเขียว", "สองสามวัน", "มาสองสามวัน"],
                    differential_diagnosis=[
                        {"name": "Viral rhinitis", "icd": "J00"},
                        {"name": "Allergic rhinitis", "icd": "J30.9"}
                    ],
                    red_flags=["ไข้สูงเกิน 39", "หายใจลำบาก", "เจ็บหน้าอกมาก"],
                    confidence_level=0.95,
                    complexity="simple",
                    learning_notes="ไข้เล็กน้อย + ไอแห้ง + น้ำมูกเขียว + สองสามวัน = หวัดธรรมดา"
                ),

                FewShotExample(
                    id="resp_common_002",
                    domain=MedicalDomain.RESPIRATORY,
                    symptoms_thai="ไข้ ไอ มีเสมหะเหลือง เมื่อยตัว จมูกไม่ได้กลิ่น",
                    symptoms_english="Fever, cough with yellow sputum, body aches, loss of smell",
                    diagnosis={
                        "icd_code": "J11.1",
                        "name": "ไข้หวัดใหญ่ (Influenza)",
                        "confidence": 0.90,
                        "urgency": "low"
                    },
                    treatment={
                        "medications": ["Oseltamivir (if within 48h)", "Paracetamol", "Cough syrup"],
                        "supportive": ["Bed rest", "Hydration", "Isolation"],
                        "complications": "Monitor for pneumonia"
                    },
                    key_indicators=["ไข้", "ไอ", "เสมหะเหลือง", "เมื่อยตัว", "ไม่ได้กลิ่น"],
                    differential_diagnosis=[
                        {"name": "Common cold", "icd": "J00"},
                        {"name": "COVID-19", "icd": "U07.1"}
                    ],
                    red_flags=["หายใจลำบาก", "ไข้สูงติดต่อ", "ปวดหน้าอกมาก"],
                    confidence_level=0.90,
                    complexity="simple",
                    learning_notes="ไข้ + ไอ + เสมหะ + เมื่อย + ไม่ได้กลิ่น = ไข้หวัดใหญ่"
                ),

                FewShotExample(
                    id="resp_001",
                    domain=MedicalDomain.RESPIRATORY,
                    symptoms_thai="หายใจลำบากฉับพลัน เจ็บหน้าอกข้างเดียว ไอแห้ง",
                    symptoms_english="Sudden dyspnea, unilateral chest pain, dry cough",
                    diagnosis={
                        "icd_code": "J93.9",
                        "name": "ปอดแฟบ (Pneumothorax)",
                        "confidence": 0.85,
                        "urgency": "high"
                    },
                    treatment={
                        "immediate": ["Oxygen", "Chest X-ray", "Needle decompression if tension"],
                        "definitive": ["Chest tube insertion", "Monitor"]
                    },
                    key_indicators=["หายใจลำบากฉับพลัน", "เจ็บหน้าอกข้างเดียว"],
                    differential_diagnosis=[
                        {"name": "Pulmonary embolism", "icd": "I26.9"},
                        {"name": "MI", "icd": "I21.9"}
                    ],
                    red_flags=["หายใจลำบากรุนแรง", "ความดันตก"],
                    confidence_level=0.85,
                    complexity="moderate",
                    learning_notes="หายใจลำบากฉับพลัน + เจ็บข้างเดียว = ปอดแฟบ"
                ),

                FewShotExample(
                    id="resp_002",
                    domain=MedicalDomain.RESPIRATORY,
                    symptoms_thai="ไอเป็นเลือด มีไข้ ปวดหน้าอก น้ำหนักลด เหงื่อออกกลางคืน",
                    symptoms_english="Hemoptysis, fever, chest pain, weight loss, night sweats",
                    diagnosis={
                        "icd_code": "A15.9",
                        "name": "วัณโรคปอด (Pulmonary TB)",
                        "confidence": 0.70,
                        "urgency": "high"
                    },
                    treatment={
                        "investigations": ["Chest X-ray", "Sputum AFB", "GeneXpert"],
                        "medications": ["RIPE therapy 6 months", "DOT"],
                        "isolation": "Respiratory precautions"
                    },
                    key_indicators=["ไอเป็นเลือด", "ไข้", "น้ำหนักลด", "เหงื่อกลางคืน"],
                    differential_diagnosis=[
                        {"name": "Lung cancer", "icd": "C78.00"},
                        {"name": "Pneumonia", "icd": "J18.9"}
                    ],
                    red_flags=["ไอเป็นเลือดต่อเนื่อง", "น้ำหนักลดมาก"],
                    confidence_level=0.70,
                    complexity="complex",
                    learning_notes="ไอเป็นเลือด + ไข้นาน + น้ำหนักลดมาก + เหงื่อกลางคืน = วัณโรค (ต้องมีไอเลือด!)"
                ),

                FewShotExample(
                    id="resp_003",
                    domain=MedicalDomain.RESPIRATORY,
                    symptoms_thai="หอบหืด หายใจขาด เสียงหวีด ไอกลางคืน",
                    symptoms_english="Wheezing, dyspnea, cough at night, chest tightness",
                    diagnosis={
                        "icd_code": "J45.9",
                        "name": "โรคหืด (Asthma)",
                        "confidence": 0.85,
                        "urgency": "moderate"
                    },
                    treatment={
                        "acute": ["Salbutamol inhaler", "Prednisolone", "Oxygen"],
                        "maintenance": ["ICS", "LABA", "Trigger avoidance"],
                        "education": "Inhaler technique, Action plan"
                    },
                    key_indicators=["หอบหืด", "เสียงหวีด", "ไอกลางคืน"],
                    differential_diagnosis=[
                        {"name": "COPD", "icd": "J44.9"},
                        {"name": "Heart failure", "icd": "I50.9"}
                    ],
                    red_flags=["หายใจลำบากรุนแรง", "พูดไม่ได้"],
                    confidence_level=0.85,
                    complexity="moderate",
                    learning_notes="หวีด + ไอกลางคืน + ทำให้หายใจขาด = หืด"
                )
            ],

            # GASTROINTESTINAL DOMAIN
            "gastrointestinal": [
                FewShotExample(
                    id="gi_001",
                    domain=MedicalDomain.GASTROINTESTINAL,
                    symptoms_thai="ปวดท้องน้อยด้านขวา มีไข้ คลื่นไส้ อาเจียน ปวดเมื่อกด",
                    symptoms_english="Right lower abdominal pain, fever, nausea, vomiting, tender on palpation",
                    diagnosis={
                        "icd_code": "K37",
                        "name": "ไส้ติ่งอักเสบ (Appendicitis)",
                        "confidence": 0.90,
                        "urgency": "high"
                    },
                    treatment={
                        "immediate": ["NPO", "IV fluids", "Pain control", "Antibiotics"],
                        "definitive": ["Appendectomy", "Laparoscopic preferred"]
                    },
                    key_indicators=["ปวดท้องน้อยขวา", "มีไข้", "ปวดเมื่อกด"],
                    differential_diagnosis=[
                        {"name": "Ovarian cyst", "icd": "N83.2"},
                        {"name": "UTI", "icd": "N39.0"}
                    ],
                    red_flags=["ปวดรุนแรงฉับพลัน", "ไข้สูง", "ท้องแข็ง"],
                    confidence_level=0.90,
                    complexity="moderate",
                    learning_notes="ปวดท้องขวาล่าง + ไข้ + กด = ไส้ติ่ง"
                ),

                FewShotExample(
                    id="gi_002",
                    domain=MedicalDomain.GASTROINTESTINAL,
                    symptoms_thai="ถ่ายเป็นเลือด ท้องเสีย ปวดท้อง มีไข้ ถ่ายบ่อย",
                    symptoms_english="Bloody diarrhea, abdominal pain, fever, frequent stools",
                    diagnosis={
                        "icd_code": "K59.1",
                        "name": "ท้องร่วงเป็นเลือด (Dysentery)",
                        "confidence": 0.85,
                        "urgency": "moderate"
                    },
                    treatment={
                        "investigations": ["Stool culture", "Blood culture", "CBC"],
                        "medications": ["Antibiotics", "ORS", "Probiotics"],
                        "monitoring": "Hydration status"
                    },
                    key_indicators=["ถ่ายเป็นเลือด", "ท้องเสีย", "ไข้"],
                    differential_diagnosis=[
                        {"name": "IBD", "icd": "K51.9"},
                        {"name": "Colon cancer", "icd": "C18.9"}
                    ],
                    red_flags=["เลือดมาก", "ขาดน้ำ", "ไข้สูง"],
                    confidence_level=0.85,
                    complexity="moderate",
                    learning_notes="ถ่ายเลือด + ไข้ + ท้องเสีย = ติดเชื้อลำไส้"
                ),

                FewShotExample(
                    id="gi_003",
                    domain=MedicalDomain.GASTROINTESTINAL,
                    symptoms_thai="ปวดท้องบน แสบกลางอก อาหารไม่ย่อย ท้องอืด",
                    symptoms_english="Epigastric pain, heartburn, dyspepsia, bloating",
                    diagnosis={
                        "icd_code": "K29.70",
                        "name": "กระเพาะอักเสบ (Gastritis)",
                        "confidence": 0.80,
                        "urgency": "low"
                    },
                    treatment={
                        "medications": ["PPI", "H2 blocker", "Antacids"],
                        "lifestyle": ["Avoid spicy food", "Small meals", "No alcohol"],
                        "follow_up": "2 weeks if no improvement"
                    },
                    key_indicators=["ปวดท้องบน", "แสบกลางอก", "อาหารไม่ย่อย"],
                    differential_diagnosis=[
                        {"name": "Peptic ulcer", "icd": "K27.9"},
                        {"name": "GERD", "icd": "K21.9"}
                    ],
                    red_flags=["อาเจียนเป็นเลือด", "ถ่ายดำ", "น้ำหนักลด"],
                    confidence_level=0.80,
                    complexity="simple",
                    learning_notes="ปวดท้องบน + แสบ + ท้องอืด = กระเพาะ"
                )
            ],

            # NEUROLOGICAL DOMAIN
            "neurological": [
                FewShotExample(
                    id="neuro_001",
                    domain=MedicalDomain.NEUROLOGICAL,
                    symptoms_thai="อ่อนแรงครึ่งซีก พูดไม่ชัด หน้าเบี้ยว เดินเซ",
                    symptoms_english="Hemiparesis, dysarthria, facial droop, ataxia",
                    diagnosis={
                        "icd_code": "I64",
                        "name": "โรคหลอดเลือดสมอง (Stroke)",
                        "confidence": 0.95,
                        "urgency": "critical"
                    },
                    treatment={
                        "immediate": ["Call 1669", "FAST assessment", "Blood glucose", "CT brain"],
                        "acute": ["Thrombolysis if <4.5hr", "Aspirin", "Monitor"],
                        "rehab": "Physical therapy, Speech therapy"
                    },
                    key_indicators=["อ่อนแรงครึ่งซีก", "พูดไม่ชัด", "หน้าเบี้ยว"],
                    differential_diagnosis=[
                        {"name": "TIA", "icd": "G93.1"},
                        {"name": "Brain tumor", "icd": "C71.9"}
                    ],
                    red_flags=["FAST positive", "อาการฉับพลัน"],
                    confidence_level=0.95,
                    complexity="complex",
                    learning_notes="FAST + อาการฉับพลัน = stroke ส่งทันที"
                ),

                FewShotExample(
                    id="neuro_002",
                    domain=MedicalDomain.NEUROLOGICAL,
                    symptoms_thai="ปวดหัวรุนแรงที่สุดในชีวิต คลื่นไส้ อาเจียน กลัวแสง",
                    symptoms_english="Worst headache ever, nausea, vomiting, photophobia",
                    diagnosis={
                        "icd_code": "I60.9",
                        "name": "เลือดออกในสมอง (Subarachnoid Hemorrhage)",
                        "confidence": 0.90,
                        "urgency": "critical"
                    },
                    treatment={
                        "immediate": ["Call 1669", "CT brain", "Lumbar puncture if CT negative"],
                        "management": ["ICU", "Nimodipine", "Aneurysm clipping/coiling"]
                    },
                    key_indicators=["ปวดหัวรุนแรงสุด", "ฉับพลัน", "กลัวแสง"],
                    differential_diagnosis=[
                        {"name": "Migraine", "icd": "G43.9"},
                        {"name": "Meningitis", "icd": "G03.9"}
                    ],
                    red_flags=["thunderclap headache", "ปวดหัวรุนแรงสุด"],
                    confidence_level=0.90,
                    complexity="complex",
                    learning_notes="ปวดหัวรุนแรงสุดในชีวิต = เลือดออกในสมอง"
                ),

                FewShotExample(
                    id="neuro_003",
                    domain=MedicalDomain.NEUROLOGICAL,
                    symptoms_thai="ชัก กระตุก สติเสื่อม ปวดหัว มีไข้",
                    symptoms_english="Seizure, convulsion, altered consciousness, headache, fever",
                    diagnosis={
                        "icd_code": "G03.9",
                        "name": "เยื่อหุ้มสมองอักเสบ (Meningitis)",
                        "confidence": 0.85,
                        "urgency": "critical"
                    },
                    treatment={
                        "immediate": ["Antibiotics IV", "Dexamethasone", "Seizure control"],
                        "investigations": ["Lumbar puncture", "Blood culture", "CT brain"]
                    },
                    key_indicators=["ชัก", "ไข้", "ปวดหัว", "คอแข็ง"],
                    differential_diagnosis=[
                        {"name": "Encephalitis", "icd": "G04.9"},
                        {"name": "Brain abscess", "icd": "G06.0"}
                    ],
                    red_flags=["ชัก + ไข้", "คอแข็ง", "ผื่นไม่หาย"],
                    confidence_level=0.85,
                    complexity="complex",
                    learning_notes="ชัก + ไข้ + ปวดหัว = เยื่อหุ้มสมอง"
                )
            ],

            # MUSCULOSKELETAL DOMAIN
            "musculoskeletal": [
                FewShotExample(
                    id="msk_001",
                    domain=MedicalDomain.MUSCULOSKELETAL,
                    symptoms_thai="ปวดข้อเข่า บวม แดง ร้อน ข้อติดตอนเช้า ปวดมากขึ้นเมื่อขยับ",
                    symptoms_english="Knee pain, swelling, redness, warmth, morning stiffness, worse with movement",
                    diagnosis={
                        "icd_code": "M19.90",
                        "name": "ข้ออักเสบเสื่อม (Osteoarthritis)",
                        "confidence": 0.85,
                        "urgency": "low"
                    },
                    treatment={
                        "medications": ["NSAIDs", "Paracetamol", "Topical analgesics"],
                        "non_pharmacological": ["Physio", "Weight loss", "Heat/cold therapy"],
                        "advanced": ["Intra-articular injection", "Joint replacement"]
                    },
                    key_indicators=["ปวดข้อ", "บวม", "แดง", "ร้อน", "ข้อติด"],
                    differential_diagnosis=[
                        {"name": "Rheumatoid arthritis", "icd": "M06.9"},
                        {"name": "Gout", "icd": "M10.9"}
                    ],
                    red_flags=["ข้อหลายข้อ", "ไข้", "น้ำหนักลด"],
                    confidence_level=0.85,
                    complexity="moderate",
                    learning_notes="ข้อเดียว + บวมแดงร้อน + ติดเช้า = ข้ออักเสบ"
                ),

                FewShotExample(
                    id="msk_002",
                    domain=MedicalDomain.MUSCULOSKELETAL,
                    symptoms_thai="ปวดข้อหลายข้อ มีไข้ ข้อบวมสลับกัน ปวดมากตอนเช้า เมื่อยล้า",
                    symptoms_english="Multiple joint pain, fever, migratory joint swelling, morning stiffness, fatigue",
                    diagnosis={
                        "icd_code": "M06.9",
                        "name": "ข้ออักเสบรูมาตอยด์ (Rheumatoid Arthritis)",
                        "confidence": 0.80,
                        "urgency": "moderate"
                    },
                    treatment={
                        "medications": ["DMARDs", "Methotrexate", "Corticosteroids", "Biologics"],
                        "monitoring": ["Liver function", "Blood count", "CRP/ESR"],
                        "lifestyle": ["Joint protection", "Exercise", "Rest during flares"]
                    },
                    key_indicators=["หลายข้อ", "มีไข้", "สลับกัน", "เช้ามาก"],
                    differential_diagnosis=[
                        {"name": "SLE", "icd": "M32.9"},
                        {"name": "Psoriatic arthritis", "icd": "M07.3"}
                    ],
                    red_flags=["ข้อพังทลาย", "อวัยวะอื่นเกี่ยว"],
                    confidence_level=0.80,
                    complexity="complex",
                    learning_notes="หลายข้อ + ไข้ + สลับ + เช้า = รูมาตอยด์"
                ),

                FewShotExample(
                    id="msk_003",
                    domain=MedicalDomain.MUSCULOSKELETAL,
                    symptoms_thai="ปวดข้อนิ้วเท้า บวมแดงมาก ปวดรุนแรงฉับพลัน กลางคืน",
                    symptoms_english="Severe toe joint pain, very swollen and red, sudden onset, nocturnal",
                    diagnosis={
                        "icd_code": "M10.9",
                        "name": "โรคเกาต์ (Gout)",
                        "confidence": 0.90,
                        "urgency": "moderate"
                    },
                    treatment={
                        "acute": ["Colchicine", "NSAIDs", "Corticosteroids"],
                        "chronic": ["Allopurinol", "Lifestyle modification"],
                        "lifestyle": ["Low purine diet", "Alcohol reduction", "Weight loss"]
                    },
                    key_indicators=["นิ้วเท้า", "บวมแดงมาก", "ฉับพลัน", "กลางคืน"],
                    differential_diagnosis=[
                        {"name": "Septic arthritis", "icd": "M00.9"},
                        {"name": "Pseudogout", "icd": "M11.9"}
                    ],
                    red_flags=["ข้อติดเชื้อ", "ไข้สูง"],
                    confidence_level=0.90,
                    complexity="moderate",
                    learning_notes="นิ้วเท้า + บวมแดงมาก + ฉับพลัน = เกาต์"
                )
            ],

            # ENDOCRINE DOMAIN
            "endocrine": [
                FewShotExample(
                    id="endo_001",
                    domain=MedicalDomain.ENDOCRINE,
                    symptoms_thai="เบาหวาน น้ำตาลในเลือดสูง 300 mg/dL ปัสสาวะบ่อย กระหายน้ำ น้ำหนักลด",
                    symptoms_english="Diabetes, blood sugar 300 mg/dL, polyuria, polydipsia, weight loss",
                    diagnosis={
                        "icd_code": "E11.9",
                        "name": "เบาหวานชนิดที่ 2 (Type 2 Diabetes)",
                        "confidence": 0.95,
                        "urgency": "high"
                    },
                    treatment={
                        "immediate": ["Hydration", "Insulin if severe", "Electrolyte monitoring"],
                        "long_term": ["Metformin", "Lifestyle modification", "HbA1c monitoring"],
                        "complications": ["Eye, Kidney, Foot screening"]
                    },
                    key_indicators=["น้ำตาลสูง", "ปัสสาวะบ่อย", "กระหายน้ำ", "น้ำหนักลด"],
                    differential_diagnosis=[
                        {"name": "Type 1 DM", "icd": "E10.9"},
                        {"name": "MODY", "icd": "E13.9"}
                    ],
                    red_flags=["DKA", "HHS", "น้ำตาล > 400"],
                    confidence_level=0.95,
                    complexity="moderate",
                    learning_notes="3P + น้ำตาลสูง = เบาหวาน"
                ),

                FewShotExample(
                    id="endo_002",
                    domain=MedicalDomain.ENDOCRINE,
                    symptoms_thai="น้ำหนักลดฉับพลัน ปัสสาวะบ่อย หิวน้ำมาก มองเห็นไม่ชัด อ่อนเพลีย",
                    symptoms_english="Rapid weight loss, polyuria, excessive thirst, blurred vision, fatigue",
                    diagnosis={
                        "icd_code": "E10.9",
                        "name": "เบาหวานชนิดที่ 1 (Type 1 Diabetes)",
                        "confidence": 0.85,
                        "urgency": "high"
                    },
                    treatment={
                        "immediate": ["Insulin therapy", "Check for DKA", "Hydration"],
                        "long_term": ["Multiple insulin regimen", "Carb counting", "CGM"],
                        "education": "Insulin technique, Hypoglycemia recognition"
                    },
                    key_indicators=["น้ำหนักลดเร็ว", "อายุน้อย", "ผอมบาง"],
                    differential_diagnosis=[
                        {"name": "LADA", "icd": "E10.9"},
                        {"name": "Hyperthyroid", "icd": "E05.9"}
                    ],
                    red_flags=["DKA", "ketones", "อาเจียน"],
                    confidence_level=0.85,
                    complexity="complex",
                    learning_notes="น้ำหนักลดเร็ว + อายุน้อย = เบาหวาน type 1"
                ),

                FewShotExample(
                    id="endo_003",
                    domain=MedicalDomain.ENDOCRINE,
                    symptoms_thai="ใจสั่น น้ำหนักลด ร้อน เหงื่อออก นอนไม่หลับ มือสั่น",
                    symptoms_english="Palpitations, weight loss, heat intolerance, sweating, insomnia, tremor",
                    diagnosis={
                        "icd_code": "E05.9",
                        "name": "ไทรอยด์เป็นพิษ (Hyperthyroidism)",
                        "confidence": 0.85,
                        "urgency": "moderate"
                    },
                    treatment={
                        "investigations": ["TSH", "Free T4", "T3", "TPO antibody"],
                        "medications": ["Methimazole", "Propranolol", "RAI"],
                        "monitoring": "Thyroid function, Liver function"
                    },
                    key_indicators=["ใจสั่น", "น้ำหนักลด", "ร้อน", "มือสั่น"],
                    differential_diagnosis=[
                        {"name": "Anxiety disorder", "icd": "F41.9"},
                        {"name": "Pheo", "icd": "E27.5"}
                    ],
                    red_flags=["thyroid storm", "ไข้สูง", "สับสน"],
                    confidence_level=0.85,
                    complexity="moderate",
                    learning_notes="ใจสั่น + ลดน้ำหนัก + ร้อน = ไทรอยด์เป็นพิษ"
                )
            ],

            # INFECTIOUS DISEASE DOMAIN
            "infectious": [
                FewShotExample(
                    id="inf_001",
                    domain=MedicalDomain.INFECTIOUS,
                    symptoms_thai="มีไข้สูง หนาวสั่น ปัสสาวะขุ่น ปัสสาวะเน่า ปัสสาวะบ่อย",
                    symptoms_english="High fever, chills, cloudy urine, foul-smelling urine, urinary frequency",
                    diagnosis={
                        "icd_code": "N39.0",
                        "name": "ติดเชื้อทางเดินปัสสาวะ (UTI)",
                        "confidence": 0.90,
                        "urgency": "moderate"
                    },
                    treatment={
                        "investigations": ["Urine analysis", "Urine culture", "Blood culture if severe"],
                        "medications": ["Empirical antibiotics", "Based on culture sensitivity"],
                        "supportive": "Hydration, Pain control"
                    },
                    key_indicators=["ไข้", "ปัสสาวะขุ่น", "กลิ่นเน่า", "ปัสสาวะบ่อย"],
                    differential_diagnosis=[
                        {"name": "Pyelonephritis", "icd": "N10"},
                        {"name": "Urethritis", "icd": "N34.1"}
                    ],
                    red_flags=["ไข้สูง", "ปวดข้าง", "คลื่นไส้"],
                    confidence_level=0.90,
                    complexity="simple",
                    learning_notes="ไข้ + ปัสสาวะผิดปกติ = UTI"
                ),

                FewShotExample(
                    id="inf_002",
                    domain=MedicalDomain.INFECTIOUS,
                    symptoms_thai="ไข้สูง มีผื่นแดง ไม่หายเมื่อกด ปวดหัว คอแข็ง",
                    symptoms_english="High fever, non-blanching rash, headache, neck stiffness",
                    diagnosis={
                        "icd_code": "A39.9",
                        "name": "เยื่อหุ้มสมองอักเสบจากเชื้อ (Bacterial Meningitis)",
                        "confidence": 0.95,
                        "urgency": "critical"
                    },
                    treatment={
                        "immediate": ["IV antibiotics", "Dexamethasone", "Contact precautions"],
                        "investigations": ["Lumbar puncture", "Blood culture", "CT brain"],
                        "prophylaxis": "Close contacts"
                    },
                    key_indicators=["ไข้สูง", "ผื่นไม่หาย", "คอแข็ง"],
                    differential_diagnosis=[
                        {"name": "Viral meningitis", "icd": "A87.9"},
                        {"name": "Sepsis", "icd": "A41.9"}
                    ],
                    red_flags=["ผื่นไม่หาย", "ชัก", "สติเสื่อม"],
                    confidence_level=0.95,
                    complexity="complex",
                    learning_notes="ไข้ + ผื่นไม่หาย + คอแข็ง = เยื่อหุ้มสมอง"
                )
            ],

            # PSYCHIATRIC DOMAIN
            "psychiatric": [
                FewShotExample(
                    id="psych_001",
                    domain=MedicalDomain.PSYCHIATRIC,
                    symptoms_thai="เศร้า ไม่มีแรง นอนไม่หลับ ไม่อยากทำอะไร สิ้นหวัง คิดทำร้ายตัวเอง",
                    symptoms_english="Depression, fatigue, insomnia, anhedonia, hopelessness, suicidal ideation",
                    diagnosis={
                        "icd_code": "F32.9",
                        "name": "ภาวะซึมเศร้า (Major Depression)",
                        "confidence": 0.85,
                        "urgency": "high"
                    },
                    treatment={
                        "immediate": ["Safety assessment", "Crisis intervention", "Hotline 1323"],
                        "medications": ["SSRIs", "SNRIs", "Psychotherapy"],
                        "follow_up": "Close monitoring, Psychiatrist referral"
                    },
                    key_indicators=["เศร้า", "ไม่มีแรง", "สิ้นหวัง", "คิดทำร้าย"],
                    differential_diagnosis=[
                        {"name": "Bipolar disorder", "icd": "F31.9"},
                        {"name": "Adjustment disorder", "icd": "F43.2"}
                    ],
                    red_flags=["คิดฆ่าตัวตาย", "แผนการทำร้าย"],
                    confidence_level=0.85,
                    complexity="moderate",
                    learning_notes="เศร้า + สิ้นหวัง + คิดทำร้าย = ซึมเศร้าร้ายแรง"
                ),

                FewShotExample(
                    id="psych_002",
                    domain=MedicalDomain.PSYCHIATRIC,
                    symptoms_thai="วิตกกังวล ใจเต้นเร็ว เหงื่อแตก สั่น หายใจเร็ว กลัวความตาย",
                    symptoms_english="Anxiety, palpitations, sweating, trembling, hyperventilation, fear of dying",
                    diagnosis={
                        "icd_code": "F41.0",
                        "name": "โรคแพนิค (Panic Disorder)",
                        "confidence": 0.80,
                        "urgency": "moderate"
                    },
                    treatment={
                        "acute": ["Breathing techniques", "Reassurance", "Benzodiazepines if severe"],
                        "long_term": ["CBT", "SSRIs", "Exposure therapy"],
                        "education": "Panic attack education, Trigger identification"
                    },
                    key_indicators=["วิตกรุนแรง", "ใจเต้นเร็ว", "หายใจเร็ว", "กลัวตาย"],
                    differential_diagnosis=[
                        {"name": "GAD", "icd": "F41.1"},
                        {"name": "Cardiac arrhythmia", "icd": "I49.9"}
                    ],
                    red_flags=["อาการหัวใจ", "หายใจไม่ออก"],
                    confidence_level=0.80,
                    complexity="moderate",
                    learning_notes="วิตก + ใจเต้น + หายใจเร็ว + กลัวตาย = แพนิค"
                )
            ]
        }

    def _create_domain_templates(self) -> Dict[str, str]:
        """Create specialized templates for each medical domain"""

        return {
            "cardiovascular": """
🫀 CARDIOVASCULAR DOMAIN TEMPLATE:
When analyzing cardiovascular symptoms, consider:
1. ACUTE CORONARY SYNDROME: Chest pain + radiation + sweating + dyspnea
2. HEART FAILURE: Edema + orthopnea + fatigue + JVD
3. HYPERTENSIVE CRISIS: BP >180/120 + end-organ damage
4. ARRHYTHMIAS: Palpitations + dizziness + syncope

RED FLAGS: Chest pain, severe dyspnea, syncope, severe hypertension
IMMEDIATE ACTION: ECG, cardiac enzymes, chest X-ray
""",

            "respiratory": """
🫁 RESPIRATORY DOMAIN TEMPLATE:
When analyzing respiratory symptoms, consider:
1. PNEUMONIA: Fever + cough + sputum + chest pain
2. ASTHMA: Wheezing + dyspnea + triggers + nocturnal symptoms
3. PNEUMOTHORAX: Sudden dyspnea + unilateral chest pain
4. PULMONARY EMBOLISM: Sudden dyspnea + chest pain + risk factors

RED FLAGS: Severe dyspnea, hemoptysis, sudden onset, hypoxia
IMMEDIATE ACTION: Oxygen saturation, chest X-ray, ABG
""",

            "gastrointestinal": """
🫃 GASTROINTESTINAL DOMAIN TEMPLATE:
When analyzing GI symptoms, consider:
1. APPENDICITIS: RLQ pain + fever + McBurney's point
2. CHOLECYSTITIS: RUQ pain + Murphy's sign + fever
3. BOWEL OBSTRUCTION: Crampy pain + vomiting + distension
4. GI BLEEDING: Hematemesis + melena + anemia

RED FLAGS: Severe abdominal pain, rigidity, hematemesis, melena
IMMEDIATE ACTION: Vitals, CBC, amylase/lipase, imaging
""",

            "neurological": """
🧠 NEUROLOGICAL DOMAIN TEMPLATE:
When analyzing neurological symptoms, consider:
1. STROKE: FAST positive + sudden onset + focal deficit
2. MENINGITIS: Fever + headache + neck stiffness + altered mental status
3. SEIZURE: Convulsion + altered consciousness + post-ictal state
4. MIGRAINE: Throbbing headache + photophobia + aura

RED FLAGS: Sudden severe headache, focal deficit, altered consciousness
IMMEDIATE ACTION: Neurological exam, glucose, CT brain
""",

            "musculoskeletal": """
🦴 MUSCULOSKELETAL DOMAIN TEMPLATE:
When analyzing MSK symptoms, consider:
1. OSTEOARTHRITIS: Single joint + morning stiffness + age-related
2. RHEUMATOID ARTHRITIS: Multiple joints + symmetrical + morning stiffness >1hr
3. GOUT: Sudden severe joint pain + usually big toe + nocturnal
4. SEPTIC ARTHRITIS: Hot joint + fever + restricted movement

RED FLAGS: Hot swollen joint + fever, multiple joint involvement
IMMEDIATE ACTION: Joint examination, ESR/CRP, joint aspiration if indicated
""",

            "endocrine": """
🔥 ENDOCRINE DOMAIN TEMPLATE:
When analyzing endocrine symptoms, consider:
1. DIABETES: Polyuria + polydipsia + polyphagia + hyperglycemia
2. HYPERTHYROIDISM: Weight loss + palpitations + heat intolerance + tremor
3. HYPOTHYROIDISM: Weight gain + fatigue + cold intolerance + bradycardia
4. ADRENAL CRISIS: Hypotension + electrolyte imbalance + altered mental status

RED FLAGS: DKA, thyroid storm, adrenal crisis, severe electrolyte imbalance
IMMEDIATE ACTION: Blood glucose, electrolytes, thyroid function
"""
        }

    def _initialize_mistake_patterns(self) -> Dict[str, List[str]]:
        """Common mistake patterns to learn from"""

        return {
            "missed_emergency": [
                "ไม่สังเกตอาการฉุกเฉิน",
                "ประเมิน urgency ต่ำเกินไป",
                "ไม่ส่งโรงพยาบาลเมื่อควรส่ง"
            ],
            "wrong_category": [
                "วินิจฉัยผิดหมวดโรค",
                "คิดเป็นโรคทั่วไปแต่เป็นโรคเฉพาะทาง",
                "ไม่คิดถึง differential diagnosis"
            ],
            "missed_red_flags": [
                "ไม่สังเกต red flag symptoms",
                "ไม่ถามอาการเพิ่มเติม",
                "ไม่ประเมินความรุนแรง"
            ],
            "confidence_issues": [
                "มั่นใจเกินไปในการวินิจฉัยที่ไม่แน่นอน",
                "มั่นใจต่ำเกินไปในการวินิจฉัยที่ชัดเจน",
                "ไม่พิจารณาความน่าจะเป็น"
            ]
        }

    def get_domain_specific_examples(self, domain: MedicalDomain, n_examples: int = 3) -> List[FewShotExample]:
        """Get examples for specific medical domain"""

        domain_key = domain.value
        if domain_key in self.examples:
            return self.examples[domain_key][:n_examples]
        return []

    def get_learning_prompt(self, domain: MedicalDomain, mistake_type: Optional[str] = None) -> str:
        """Generate learning prompt for specific domain and mistake type"""

        domain_template = self.domain_templates.get(domain.value, "")

        if mistake_type and mistake_type in self.mistake_patterns:
            mistake_info = "\n".join(self.mistake_patterns[mistake_type])
            return f"""
{domain_template}

🚨 COMMON MISTAKES TO AVOID:
{mistake_info}

Remember: Always consider red flags and differential diagnosis!
"""

        return domain_template

    def create_few_shot_prompt(self, symptoms: str, n_examples: int = 3) -> str:
        """Create comprehensive few-shot prompt based on symptoms"""

        # Determine most relevant domain
        relevant_domain = self._classify_domain(symptoms)

        # Get domain-specific examples
        examples = self.get_domain_specific_examples(relevant_domain, n_examples)

        # Build prompt
        prompt = f"""
Medical AI Diagnostic Assistant - Few-Shot Learning

DOMAIN: {relevant_domain.value.upper()}

{self.domain_templates.get(relevant_domain.value, "")}

EXAMPLES TO LEARN FROM:

"""

        for i, example in enumerate(examples, 1):
            prompt += f"""
Example {i}:
Symptoms (Thai): {example.symptoms_thai}
Symptoms (English): {example.symptoms_english}

✅ CORRECT DIAGNOSIS:
- ICD Code: {example.diagnosis['icd_code']}
- Name: {example.diagnosis['name']}
- Confidence: {example.confidence_level:.0%}
- Urgency: {example.diagnosis.get('urgency', 'moderate')}

🎯 KEY INDICATORS: {', '.join(example.key_indicators)}

💊 TREATMENT:
{json.dumps(example.treatment, indent=2, ensure_ascii=False)}

🚨 RED FLAGS: {', '.join(example.red_flags)}

📝 LEARNING NOTE: {example.learning_notes}

---
"""

        prompt += f"""
NOW ANALYZE THESE SYMPTOMS:
{symptoms}

Apply the patterns and knowledge from the examples above.
Consider the key indicators, red flags, and differential diagnoses.
Provide confidence level and urgency assessment.
"""

        return prompt

    def create_specialized_complex_prompt(self, symptoms: str, complexity_level: str = "complex") -> str:
        """Create specialized prompt for complex diagnostic scenarios"""

        domain = self._classify_domain(symptoms)

        if complexity_level == "complex":
            return self._create_complex_diagnostic_prompt(symptoms, domain)
        elif complexity_level == "emergency":
            return self._create_emergency_prompt(symptoms, domain)
        elif complexity_level == "differential":
            return self._create_differential_prompt(symptoms, domain)
        else:
            return self.create_few_shot_prompt(symptoms)

    def _create_complex_diagnostic_prompt(self, symptoms: str, domain: MedicalDomain) -> str:
        """Create prompt for complex diagnostic scenarios with multiple possibilities"""

        return f"""
🧠 COMPLEX DIAGNOSTIC ANALYSIS - {domain.value.upper()} DOMAIN

ADVANCED DIAGNOSTIC FRAMEWORK:

1️⃣ PATTERN RECOGNITION:
- Primary symptom cluster analysis
- Timeline and progression assessment
- Associated symptoms mapping
- Risk factor evaluation

2️⃣ DIFFERENTIAL DIAGNOSIS TREE:
- Most likely diagnosis (>70% confidence)
- Alternative diagnoses (30-70% confidence)
- Rare but critical diagnoses (<30% but high risk)

3️⃣ RED FLAG ASSESSMENT:
- Emergency indicators requiring immediate action
- Concerning patterns needing urgent evaluation
- Stable presentations for outpatient management

4️⃣ EVIDENCE-BASED REASONING:
- Clinical probability scoring
- Supporting evidence strength
- Contradictory evidence analysis
- Uncertainty acknowledgment

PATIENT PRESENTATION:
{symptoms}

DIAGNOSTIC APPROACH:
1. Systematically analyze each symptom cluster
2. Consider temporal relationships and triggers
3. Apply domain-specific diagnostic criteria
4. Weight differential diagnoses by probability
5. Identify any red flags requiring immediate action
6. Provide confidence intervals for each diagnosis
7. Recommend next steps based on uncertainty level

CRITICAL THINKING REQUIREMENTS:
- Question initial impressions
- Consider multiple diagnostic pathways
- Acknowledge diagnostic uncertainty
- Prioritize patient safety over diagnostic confidence
"""

    def _create_emergency_prompt(self, symptoms: str, domain: MedicalDomain) -> str:
        """Create prompt specifically for emergency presentations"""

        return f"""
🚨 EMERGENCY DIAGNOSTIC PROTOCOL - {domain.value.upper()}

EMERGENCY ASSESSMENT FRAMEWORK:

⚡ IMMEDIATE TRIAGE (First 30 seconds):
- Life-threatening conditions (ABCs)
- Time-critical diagnoses
- Immediate intervention needs

🎯 RAPID DIFFERENTIAL (Next 2 minutes):
- Most likely emergency diagnosis
- Critical alternative diagnoses
- Benign mimics to exclude

⏰ TIME-SENSITIVE ACTIONS:
- Immediate interventions required
- Diagnostic tests needed urgently
- Specialist consultation triggers

🔴 RED FLAG IDENTIFICATION:
- Cardiovascular: Chest pain + radiation + hemodynamic instability
- Neurological: Focal deficits + altered consciousness + sudden onset
- Respiratory: Severe dyspnea + hypoxia + asymmetric findings
- GI: Severe pain + hematemesis/melena + hemodynamic compromise

PATIENT PRESENTATION:
{symptoms}

EMERGENCY ANALYSIS PROTOCOL:
1. IMMEDIATE THREAT ASSESSMENT: Life/limb/organ threatening?
2. RAPID PATTERN RECOGNITION: Classic emergency presentations?
3. CRITICAL DECISION POINTS: Admit/discharge/urgent referral?
4. TIME-SENSITIVE INTERVENTIONS: What cannot wait?
5. DIFFERENTIAL PRIORITIES: Most dangerous diagnosis first
6. SAFETY NET: What could we be missing?

RESPONSE FORMAT:
- Emergency Level: CRITICAL/HIGH/MODERATE/LOW
- Immediate Actions: [List 3 most urgent steps]
- Primary Diagnosis: [Most likely with confidence %]
- Cannot Miss: [Dangerous alternatives to exclude]
- Timeline: [How quickly must this be addressed?]
"""

    def _create_differential_prompt(self, symptoms: str, domain: MedicalDomain) -> str:
        """Create prompt focused on differential diagnosis generation"""

        return f"""
🎯 DIFFERENTIAL DIAGNOSIS GENERATOR - {domain.value.upper()}

SYSTEMATIC DIFFERENTIAL APPROACH:

📊 SYMPTOM CLUSTER ANALYSIS:
Primary Symptoms: [Extract key symptoms]
Secondary Symptoms: [Supporting symptoms]
Timeline: [Acute/subacute/chronic]
Context: [Triggers, precipitants, associations]

🔄 DIFFERENTIAL CATEGORIES:

1️⃣ MOST LIKELY (Confidence >70%):
- Common presentations in this domain
- Classic symptom patterns
- Epidemiologically probable

2️⃣ POSSIBLE (Confidence 30-70%):
- Atypical presentations of common conditions
- Less common but plausible diagnoses
- Symptom overlap scenarios

3️⃣ CANNOT MISS (Confidence <30% but critical):
- Life-threatening conditions
- Progressive/irreversible conditions
- Conditions requiring immediate intervention

4️⃣ RARE BUT RELEVANT:
- Zebra diagnoses worth considering
- Condition-specific risk factors present
- Unusual presentations of serious conditions

PATIENT SYMPTOMS:
{symptoms}

DIFFERENTIAL GENERATION PROCESS:
1. Identify dominant symptom pattern
2. List all conditions that could cause this pattern
3. Rank by probability in this patient population
4. Separate "common" from "cannot miss" diagnoses
5. Consider atypical presentations
6. Factor in patient demographics and risk factors
7. Acknowledge diagnostic uncertainty

OUTPUT STRUCTURE:
- Primary Diagnosis: [Most likely with reasoning]
- Active Differentials: [2-3 strong alternatives]
- Cannot Miss: [Critical conditions to exclude]
- Working Diagnosis Confidence: [Percentage]
- Next Diagnostic Steps: [Tests/examinations needed]
- Red Flags to Monitor: [Warning signs to watch for]
"""

    def _classify_domain(self, symptoms: str) -> MedicalDomain:
        """Classify symptoms into medical domain"""

        symptoms_lower = symptoms.lower()

        # Domain classification rules
        if any(keyword in symptoms_lower for keyword in ['เจ็บหน้าอก', 'ใจเต้น', 'ความดัน', 'ขาบวม']):
            return MedicalDomain.CARDIOVASCULAR
        elif any(keyword in symptoms_lower for keyword in ['หายใจ', 'ไอ', 'เสียงหวีด', 'ปอด']):
            return MedicalDomain.RESPIRATORY
        elif any(keyword in symptoms_lower for keyword in ['ปวดท้อง', 'อาเจียน', 'ท้องเสีย', 'ถ่าย']):
            return MedicalDomain.GASTROINTESTINAL
        elif any(keyword in symptoms_lower for keyword in ['ปวดหัว', 'ชัก', 'เดิน', 'พูด', 'สติ']):
            return MedicalDomain.NEUROLOGICAL
        elif any(keyword in symptoms_lower for keyword in ['ปวดข้อ', 'บวม', 'แดง', 'ร้อน']):
            return MedicalDomain.MUSCULOSKELETAL
        elif any(keyword in symptoms_lower for keyword in ['เบาหวาน', 'น้ำตาล', 'ปัสสาวะบ่อย', 'กระหาย']):
            return MedicalDomain.ENDOCRINE
        elif any(keyword in symptoms_lower for keyword in ['ไข้', 'หนาวสั่น', 'ติดเชื้อ']):
            return MedicalDomain.INFECTIOUS
        elif any(keyword in symptoms_lower for keyword in ['เศร้า', 'วิตก', 'กังวล', 'นอนไม่หลับ']):
            return MedicalDomain.PSYCHIATRIC
        elif any(keyword in symptoms_lower for keyword in ['ผื่น', 'คัน', 'แสง']):
            return MedicalDomain.DERMATOLOGICAL
        else:
            return MedicalDomain.EMERGENCY  # Default to emergency for unknown

    def update_examples_from_feedback(self,
                                    symptoms: str,
                                    wrong_diagnosis: Dict[str, Any],
                                    correct_diagnosis: Dict[str, Any],
                                    feedback: str) -> None:
        """Update few-shot examples based on doctor feedback"""

        domain = self._classify_domain(symptoms)

        # Create new learning example from mistake
        new_example = FewShotExample(
            id=f"feedback_{len(self.examples.get(domain.value, []))}",
            domain=domain,
            symptoms_thai=symptoms,
            symptoms_english="",  # Would need translation
            diagnosis=correct_diagnosis,
            treatment={},  # Would be filled in later
            key_indicators=self._extract_key_indicators(symptoms),
            differential_diagnosis=[wrong_diagnosis],
            red_flags=[],
            confidence_level=correct_diagnosis.get('confidence', 0.8),
            complexity="moderate",
            learning_notes=f"Learned from feedback: {feedback}"
        )

        # Add to examples
        if domain.value not in self.examples:
            self.examples[domain.value] = []

        self.examples[domain.value].append(new_example)

        logger.info(f"🎓 Added new few-shot example for {domain.value} domain")

    def _extract_key_indicators(self, symptoms: str) -> List[str]:
        """Extract key symptom indicators"""

        # Simple keyword extraction - could be enhanced with NLP
        keywords = []
        symptom_words = symptoms.split()

        medical_terms = [
            'ปวด', 'เจ็บ', 'บวม', 'แดง', 'ร้อน', 'ไข้', 'ไอ', 'หายใจ',
            'ใจเต้น', 'เหงื่อ', 'คลื่นไส้', 'อาเจียน', 'ท้องเสีย',
            'ปัสสาวะ', 'เศร้า', 'วิตก', 'นอนไม่หลับ', 'เมื่อย'
        ]

        for word in symptom_words:
            if any(term in word for term in medical_terms):
                keywords.append(word)

        return keywords[:5]  # Return top 5 key indicators

    def get_statistics(self) -> Dict[str, Any]:
        """Get few-shot learning statistics"""

        total_examples = sum(len(examples) for examples in self.examples.values())
        domain_counts = {domain: len(examples) for domain, examples in self.examples.items()}

        return {
            "total_examples": total_examples,
            "domain_distribution": domain_counts,
            "domains_covered": len(self.examples),
            "average_examples_per_domain": total_examples / len(self.examples) if self.examples else 0
        }

    async def enhanced_diagnosis(self, symptoms: str, patient_id: Optional[str] = None, patient_info: Optional[Any] = None) -> Dict[str, Any]:
        """Enhanced diagnosis using comprehensive few-shot examples + RAG knowledge base"""

        # STEP 1: Get traditional few-shot examples
        relevant_examples = self._find_relevant_examples(symptoms)

        # STEP 2: Enhance with RAG-retrieved examples from knowledge base
        if RAG_AVAILABLE:
            try:
                logger.info("🔍 Enhancing few-shot with RAG knowledge retrieval...")
                # Include patient context in RAG retrieval
                patient_data = {"patient_id": patient_id} if patient_id else {}
                if patient_info:
                    patient_data["patient_context"] = str(patient_info)

                rag_examples = await rag_few_shot_service.get_relevant_examples(
                    symptoms=symptoms,
                    patient_data=patient_data if patient_data else None,
                    max_examples=3
                )

                # Convert RAG examples to FewShotExample format for compatibility
                for rag_example in rag_examples:
                    converted_example = self._convert_rag_to_few_shot(rag_example)
                    if converted_example:
                        relevant_examples.insert(0, converted_example)  # Prioritize RAG examples

                logger.info(f"✅ Enhanced with {len(rag_examples)} RAG examples")

            except Exception as e:
                logger.error(f"❌ RAG enhancement failed: {e}")

        if not relevant_examples:
            return {"confidence": 0, "primary_diagnosis": None}

        # Calculate confidence based on pattern matching
        best_match = relevant_examples[0]
        confidence = self._calculate_confidence(symptoms, best_match)

        # Apply domain-specific enhancements
        enhanced_confidence = self._apply_domain_enhancements(symptoms, best_match, confidence)

        # Apply safety adjustments based on source
        if hasattr(best_match, 'rag_source') and best_match.rag_source:
            enhanced_confidence = self._apply_rag_safety_adjustment(enhanced_confidence, best_match, symptoms)
        else:
            # Apply static safety check for non-RAG examples
            enhanced_confidence = self._apply_static_safety_check(enhanced_confidence, best_match, symptoms)

        return {
            "confidence": enhanced_confidence,
            "primary_diagnosis": {
                "icd_code": best_match.diagnosis["icd_code"],
                "english_name": best_match.diagnosis["name"],
                "thai_name": best_match.diagnosis["name"],  # Could be enhanced with proper Thai names
                "confidence": enhanced_confidence,
                "category": best_match.domain.value,
                "matched_keywords": best_match.key_indicators[:3],
                "few_shot_source": True,
                "pattern_analysis": self._get_pattern_analysis(symptoms, best_match)
            },
            "differential_diagnoses": [
                {
                    "icd_code": ex.diagnosis["icd_code"],
                    "english_name": ex.diagnosis["name"],
                    "thai_name": ex.diagnosis["name"],
                    "confidence": max(50, enhanced_confidence - 20),
                    "category": ex.domain.value
                } for ex in relevant_examples[1:3]
            ]
        }

    def _find_relevant_examples(self, symptoms: str) -> List[FewShotExample]:
        """Find most relevant few-shot examples for given symptoms"""

        # Classify domain first
        domain = self._classify_domain(symptoms)
        domain_examples = self.examples.get(domain.value, [])

        # Score examples based on symptom similarity
        scored_examples = []
        symptoms_lower = symptoms.lower()

        for example in domain_examples:
            score = 0
            # Score based on key indicators match
            for indicator in example.key_indicators:
                if indicator.lower() in symptoms_lower:
                    score += 2

            # Score based on symptoms overlap
            example_symptoms = example.symptoms_thai.lower()
            for word in symptoms_lower.split():
                if len(word) > 3 and word in example_symptoms:
                    score += 1

            if score > 0:
                scored_examples.append((score, example))

        # Sort by score and return top examples
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [example for score, example in scored_examples[:5]]

    def _calculate_confidence(self, symptoms: str, example: FewShotExample) -> float:
        """Calculate confidence based on symptom pattern matching"""

        base_confidence = example.confidence_level
        symptoms_lower = symptoms.lower()

        # Count matching key indicators
        matching_indicators = 0
        for indicator in example.key_indicators:
            if indicator.lower() in symptoms_lower:
                matching_indicators += 1

        # Adjust confidence based on matches
        indicator_bonus = (matching_indicators / len(example.key_indicators)) * 0.15

        # Check for red flags
        red_flag_penalty = 0
        for red_flag in example.red_flags:
            if red_flag.lower() in symptoms_lower:
                red_flag_penalty = 0.1  # Increase confidence if red flags present
                break

        final_confidence = min(0.95, base_confidence + indicator_bonus + red_flag_penalty)
        return final_confidence

    def _apply_domain_enhancements(self, symptoms: str, example: FewShotExample, base_confidence: float) -> float:
        """Apply domain-specific confidence enhancements"""

        enhanced_confidence = base_confidence
        domain = example.domain

        # Domain-specific enhancement rules
        if domain == MedicalDomain.CARDIOVASCULAR:
            # Higher confidence for classic presentations
            if all(keyword in symptoms.lower() for keyword in ['เจ็บหน้าอก', 'ปวดร้าว']):
                enhanced_confidence += 0.1

        elif domain == MedicalDomain.MUSCULOSKELETAL:
            # Arthritis pattern enhancement
            if 'ปวดข้อ' in symptoms.lower() and any(word in symptoms.lower() for word in ['บวม', 'แดง', 'ร้อน']):
                enhanced_confidence += 0.15

        elif domain == MedicalDomain.ENDOCRINE:
            # Diabetes pattern enhancement
            diabetes_keywords = ['ปัสสาวะบ่อย', 'กระหายน้ำ', 'น้ำหนักลด']
            if sum(1 for keyword in diabetes_keywords if keyword in symptoms.lower()) >= 2:
                enhanced_confidence += 0.2

        elif domain == MedicalDomain.EMERGENCY:
            # Emergency presentations should have high confidence
            enhanced_confidence += 0.1

        return min(0.98, enhanced_confidence)

    def _get_pattern_analysis(self, symptoms: str, example: FewShotExample) -> Dict[str, Any]:
        """Get detailed pattern analysis for the diagnosis"""

        return {
            "matched_patterns": [indicator for indicator in example.key_indicators
                               if indicator.lower() in symptoms.lower()],
            "domain_classification": example.domain.value,
            "complexity_level": example.complexity,
            "urgency_assessment": example.diagnosis.get("urgency", "moderate"),
            "learning_source": "few_shot_examples",
            "red_flags_detected": [flag for flag in example.red_flags
                                 if flag.lower() in symptoms.lower()]
        }

    def _convert_rag_to_few_shot(self, rag_example) -> Optional[FewShotExample]:
        """Convert RAG example to FewShotExample format for compatibility"""
        try:
            # Determine domain from diagnosis category or symptoms
            domain = self._determine_domain_from_rag(rag_example)

            few_shot_example = FewShotExample(
                id=rag_example.id,
                domain=domain,
                symptoms_thai=rag_example.symptoms_thai,
                symptoms_english=rag_example.symptoms_english,
                diagnosis=rag_example.diagnosis,
                treatment=rag_example.treatment,
                key_indicators=rag_example.key_indicators,
                differential_diagnosis=[],  # Could be enhanced
                red_flags=rag_example.safety_notes,
                confidence_level=rag_example.confidence_level,
                complexity="dynamic",  # Mark as RAG-generated
                learning_notes=f"RAG-retrieved from knowledge base (score: {rag_example.retrieval_score:.2f})"
            )

            # Mark as RAG source for special handling
            few_shot_example.rag_source = True
            few_shot_example.rag_retrieval_score = rag_example.retrieval_score

            return few_shot_example

        except Exception as e:
            logger.error(f"Failed to convert RAG example: {e}")
            return None

    def _determine_domain_from_rag(self, rag_example) -> MedicalDomain:
        """Determine medical domain from RAG example"""
        diagnosis_name = rag_example.diagnosis.get('name', '').lower()
        symptoms = (rag_example.symptoms_english + " " + rag_example.symptoms_thai).lower()

        # Domain classification based on diagnosis and symptoms
        if any(term in diagnosis_name or term in symptoms for term in ['heart', 'cardiac', 'หัวใจ', 'หน้าอก']):
            return MedicalDomain.CARDIOVASCULAR
        elif any(term in diagnosis_name or term in symptoms for term in ['lung', 'respiratory', 'ปอด', 'หายใจ', 'ไอ']):
            return MedicalDomain.RESPIRATORY
        elif any(term in diagnosis_name or term in symptoms for term in ['stomach', 'ท้อง', 'อาหาร', 'gastro']):
            return MedicalDomain.GASTROINTESTINAL
        elif any(term in diagnosis_name or term in symptoms for term in ['brain', 'neuro', 'สมอง', 'ประสาท']):
            return MedicalDomain.NEUROLOGICAL
        elif any(term in diagnosis_name or term in symptoms for term in ['bone', 'joint', 'กระดูก', 'ข้อ']):
            return MedicalDomain.MUSCULOSKELETAL
        elif any(term in diagnosis_name or term in symptoms for term in ['diabetes', 'thyroid', 'เบาหวาน', 'ไทรอยด์']):
            return MedicalDomain.ENDOCRINE
        elif any(term in diagnosis_name or term in symptoms for term in ['infection', 'fever', 'ติดเชื้อ', 'ไข้']):
            return MedicalDomain.INFECTIOUS
        else:
            return MedicalDomain.RESPIRATORY  # Default fallback

    def _apply_rag_safety_adjustment(self, confidence: float, rag_example, symptoms: str) -> float:
        """Apply safety adjustments for RAG-retrieved examples"""

        adjusted_confidence = confidence

        # Conservative adjustment for RAG examples
        adjusted_confidence *= 0.9  # Slight reduction for dynamic examples

        # Check retrieval score quality
        if hasattr(rag_example, 'rag_retrieval_score'):
            if rag_example.rag_retrieval_score < 0.7:
                adjusted_confidence *= 0.8  # Reduce confidence for low retrieval scores

        # Apply symptom-diagnosis safety check
        diagnosis_name = rag_example.diagnosis.get('name', '').lower()
        symptoms_lower = symptoms.lower()

        # Safety check: Don't allow serious diagnoses for mild symptoms
        serious_indicators = ['วัณโรค', 'tuberculosis', 'cancer', 'มะเร็ง', 'stroke', 'heart attack']
        mild_indicators = ['เล็กน้อย', 'mild', '38 องศา', 'สองสามวัน', 'น้ำมูกเขียว']

        if any(serious in diagnosis_name for serious in serious_indicators):
            if any(mild in symptoms_lower for mild in mild_indicators):
                logger.warning(f"🚫 RAG safety: Reducing confidence for serious diagnosis {diagnosis_name} with mild symptoms")
                adjusted_confidence *= 0.3  # Significant reduction

        # Ensure confidence stays within bounds
        return max(0.1, min(0.95, adjusted_confidence))

    def _apply_static_safety_check(self, confidence: float, example: FewShotExample, symptoms: str) -> float:
        """Apply safety checks for static few-shot examples"""

        diagnosis_name = example.diagnosis.get('name', '').lower()
        symptoms_lower = symptoms.lower()

        # Block dangerous static examples for mild symptoms
        serious_indicators = [
            'วัณโรค', 'tuberculosis', 'cancer', 'มะเร็ง', 'stroke', 'heart attack',
            'meningitis', 'เยื่อหุ้มสมอง', 'sepsis', 'brain tumor'
        ]
        mild_indicators = ['เล็กน้อย', 'mild', '38 องศา', 'สองสามวัน', 'น้ำมูกเขียว', 'เมื่อย', 'วันเดียว']

        has_mild_symptoms = any(mild in symptoms_lower for mild in mild_indicators)
        has_serious_diagnosis = any(serious in diagnosis_name for serious in serious_indicators)

        if has_mild_symptoms and has_serious_diagnosis:
            logger.warning(f"🚫 STATIC safety: Blocking serious diagnosis {diagnosis_name} for mild symptoms")
            return 0.1  # Minimal confidence to effectively block

        # Special case: meningitis requires specific symptoms
        if 'meningitis' in diagnosis_name or 'เยื่อหุ้มสมอง' in diagnosis_name:
            required_symptoms = ['ชัก', 'seizure', 'แข็งทื่อ', 'stiff neck', 'ไข้สูงมาก', 'severe fever']
            has_required = any(req in symptoms_lower for req in required_symptoms)
            if not has_required:
                logger.warning(f"🚫 STATIC safety: Blocking meningitis diagnosis without required symptoms")
                return 0.1

        return confidence

# Singleton instance
advanced_few_shot = AdvancedFewShotLearning()