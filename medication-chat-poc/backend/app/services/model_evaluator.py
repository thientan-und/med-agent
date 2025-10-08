"""
Model Evaluation and Self-Learning System
==========================================
Zero-shot evaluation with RAG knowledge and few-shot learning from mistakes
"""

import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict

from app.services.memory_agent import memory_agent

logger = logging.getLogger(__name__)

@dataclass
class TestCase:
    """Medical test case for evaluation"""
    id: str
    symptoms: str
    expected_diagnosis: Dict[str, Any]
    expected_treatment: Dict[str, Any]
    severity: str
    category: str
    ground_truth_source: str

@dataclass
class EvaluationResult:
    """Result of model evaluation"""
    test_case_id: str
    predicted_diagnosis: Dict[str, Any]
    expected_diagnosis: Dict[str, Any]
    diagnosis_match: bool
    confidence_score: float
    treatment_accuracy: float
    response_time_ms: float
    error_analysis: Optional[Dict[str, Any]] = None

class ModelEvaluator:
    """Evaluates and improves medical AI model through testing and learning"""

    def __init__(self):
        # Test cases based on RAG knowledge
        self.test_cases = self._initialize_test_cases()

        # Mistake tracking for few-shot learning
        self.mistake_examples: List[Dict[str, Any]] = []
        self.correct_examples: List[Dict[str, Any]] = []

        # Performance metrics
        self.evaluation_history: List[EvaluationResult] = []
        self.accuracy_over_time: List[float] = []

        # Few-shot learning templates
        self.learning_templates = {
            'mistake_correction': self._create_mistake_template,
            'success_reinforcement': self._create_success_template,
            'edge_case': self._create_edge_case_template
        }

        logger.info("🔬 Model Evaluator initialized with RAG test cases")

    def _initialize_test_cases(self) -> List[TestCase]:
        """Initialize comprehensive test cases from medical knowledge"""
        return [
            # Arthritis cases (previously failing)
            TestCase(
                id="arthritis_001",
                symptoms="ปวดข้อเข่า บวม แดง ร้อน ข้อติดตอนเช้า ปวดมากขึ้นเมื่อขยับ",
                expected_diagnosis={
                    "icd_code": "M19.90",
                    "name": "ข้ออักเสบ (Osteoarthritis)",
                    "confidence": 0.85,
                    "category": "Musculoskeletal"
                },
                expected_treatment={
                    "medications": ["NSAIDs", "Paracetamol", "Topical analgesics"],
                    "lifestyle": ["Weight reduction", "Physical therapy", "Hot/cold compress"],
                    "follow_up": "2-4 weeks"
                },
                severity="moderate",
                category="chronic",
                ground_truth_source="Clinical Guidelines 2024"
            ),

            TestCase(
                id="arthritis_002",
                symptoms="ปวดข้อหลายข้อ มีไข้ ข้อบวมสลับกัน ปวดมากตอนเช้า",
                expected_diagnosis={
                    "icd_code": "M06.9",
                    "name": "รูมาตอยด์ (Rheumatoid Arthritis)",
                    "confidence": 0.80,
                    "category": "Autoimmune"
                },
                expected_treatment={
                    "medications": ["DMARDs", "Corticosteroids", "NSAIDs"],
                    "lifestyle": ["Rest", "Gentle exercise", "Joint protection"],
                    "follow_up": "Regular monitoring"
                },
                severity="high",
                category="chronic",
                ground_truth_source="Rheumatology Association"
            ),

            # Diabetes cases (previously failing)
            TestCase(
                id="diabetes_001",
                symptoms="เบาหวาน น้ำตาลในเลือดสูง 300 mg/dL ปัสสาวะบ่อย กระหายน้ำ น้ำหนักลด",
                expected_diagnosis={
                    "icd_code": "E11.9",
                    "name": "เบาหวานชนิดที่ 2 (Type 2 Diabetes)",
                    "confidence": 0.95,
                    "category": "Endocrine"
                },
                expected_treatment={
                    "medications": ["Metformin", "Insulin (if severe)", "DPP-4 inhibitors"],
                    "lifestyle": ["Diet control", "Regular exercise", "Blood sugar monitoring"],
                    "follow_up": "Every 3 months"
                },
                severity="high",
                category="chronic",
                ground_truth_source="Diabetes Association Guidelines"
            ),

            TestCase(
                id="diabetes_002",
                symptoms="น้ำหนักลดฉับพลัน ปัสสาวะบ่อย หิวน้ำมาก มองเห็นไม่ชัด อ่อนเพลีย",
                expected_diagnosis={
                    "icd_code": "E10.9",
                    "name": "เบาหวานชนิดที่ 1 (Type 1 Diabetes)",
                    "confidence": 0.85,
                    "category": "Endocrine"
                },
                expected_treatment={
                    "medications": ["Insulin therapy (mandatory)", "Glucagon kit"],
                    "lifestyle": ["Carb counting", "Regular monitoring", "Exercise planning"],
                    "follow_up": "Endocrinologist referral"
                },
                severity="critical",
                category="chronic",
                ground_truth_source="Pediatric Diabetes Guidelines"
            ),

            # Emergency cases
            TestCase(
                id="emergency_001",
                symptoms="เจ็บหน้าอกรุนแรง หายใจลำบาก เหงื่อแตก คลื่นไส้ ปวดร้าวไปแขนซ้าย",
                expected_diagnosis={
                    "icd_code": "I21.9",
                    "name": "กล้ามเนื้อหัวใจขาดเลือด (Myocardial Infarction)",
                    "confidence": 0.95,
                    "category": "Cardiovascular Emergency"
                },
                expected_treatment={
                    "immediate": ["Call 1669", "Aspirin 300mg", "Oxygen", "ECG"],
                    "hospital": ["PCI", "Thrombolysis", "Cardiac monitoring"],
                    "follow_up": "Cardiac rehabilitation"
                },
                severity="critical",
                category="emergency",
                ground_truth_source="AHA Guidelines"
            ),

            # Common illnesses
            TestCase(
                id="common_001",
                symptoms="ไข้ ไอ เจ็บคอ น้ำมูกไหล ปวดเมื่อยตัว",
                expected_diagnosis={
                    "icd_code": "J00",
                    "name": "ไข้หวัด (Common Cold)",
                    "confidence": 0.90,
                    "category": "Respiratory"
                },
                expected_treatment={
                    "medications": ["Paracetamol", "Antihistamines", "Throat lozenges"],
                    "lifestyle": ["Rest", "Hydration", "Avoid cold"],
                    "follow_up": "If symptoms persist > 7 days"
                },
                severity="low",
                category="acute",
                ground_truth_source="Primary Care Guidelines"
            ),

            # Gastric cases
            TestCase(
                id="gastric_001",
                symptoms="ปวดท้อง แสบร้อนกลางอก อาหารไม่ย่อย ท้องอืด",
                expected_diagnosis={
                    "icd_code": "K29.70",
                    "name": "กระเพาะอักเสบ (Gastritis)",
                    "confidence": 0.85,
                    "category": "Gastrointestinal"
                },
                expected_treatment={
                    "medications": ["PPIs", "Antacids", "H2 blockers"],
                    "lifestyle": ["Avoid spicy food", "Small frequent meals", "No alcohol"],
                    "follow_up": "If symptoms persist > 2 weeks"
                },
                severity="moderate",
                category="acute",
                ground_truth_source="GI Society Guidelines"
            ),

            # Allergy cases
            TestCase(
                id="allergy_001",
                symptoms="ผื่นแดง คัน ทั่วตัว หลังกินกุ้ง หายใจลำบาก ปากบวม",
                expected_diagnosis={
                    "icd_code": "T78.40",
                    "name": "ภูมิแพ้รุนแรง (Anaphylaxis)",
                    "confidence": 0.95,
                    "category": "Allergic Emergency"
                },
                expected_treatment={
                    "immediate": ["Epinephrine", "Call 1669", "Antihistamines", "Corticosteroids"],
                    "hospital": ["Observation", "IV fluids", "Monitoring"],
                    "follow_up": "Allergy testing"
                },
                severity="critical",
                category="emergency",
                ground_truth_source="Allergy Foundation"
            ),

            # Hypertension cases
            TestCase(
                id="hypertension_001",
                symptoms="ปวดหัว ตาพร่า คลื่นไส้ ความดัน 180/110 เวียนหัว",
                expected_diagnosis={
                    "icd_code": "I10",
                    "name": "ความดันโลหิตสูง (Hypertensive Crisis)",
                    "confidence": 0.95,
                    "category": "Cardiovascular"
                },
                expected_treatment={
                    "immediate": ["Emergency room", "BP monitoring", "IV antihypertensives"],
                    "medications": ["ACE inhibitors", "Beta blockers", "Diuretics"],
                    "lifestyle": ["Low sodium diet", "Weight loss", "Exercise"],
                    "follow_up": "Weekly until controlled"
                },
                severity="high",
                category="chronic",
                ground_truth_source="Hypertension Guidelines 2024"
            ),

            # Mental health cases
            TestCase(
                id="mental_001",
                symptoms="นอนไม่หลับ เบื่ออาหาร ไม่มีแรง สิ้นหวัง คิดทำร้ายตัวเอง",
                expected_diagnosis={
                    "icd_code": "F32.9",
                    "name": "ภาวะซึมเศร้า (Major Depression)",
                    "confidence": 0.85,
                    "category": "Mental Health"
                },
                expected_treatment={
                    "immediate": ["Crisis hotline 1323", "Safety assessment"],
                    "medications": ["SSRIs", "Psychotherapy referral"],
                    "lifestyle": ["Sleep hygiene", "Exercise", "Social support"],
                    "follow_up": "Psychiatrist referral"
                },
                severity="high",
                category="mental",
                ground_truth_source="Mental Health Guidelines"
            )
        ]

    async def run_zero_shot_evaluation(self, medical_ai_service) -> Dict[str, Any]:
        """Run zero-shot evaluation on all test cases"""

        logger.info("🔬 Starting zero-shot evaluation...")
        results = []

        for test_case in self.test_cases:
            start_time = datetime.now()

            # Get AI prediction (zero-shot)
            ai_response = await medical_ai_service.process_common_illness_consultation(
                message=test_case.symptoms,
                conversation_history=[],
                patient_info=None,
                vital_signs=None,
                preferred_language="thai",
                session_id=f"eval_{test_case.id}"
            )

            # Calculate response time
            response_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Extract predicted diagnosis
            predicted_diagnosis = ai_response.get("diagnosis", {})

            # Evaluate accuracy
            evaluation = self._evaluate_prediction(
                test_case=test_case,
                predicted_diagnosis=predicted_diagnosis,
                ai_response=ai_response,
                response_time_ms=response_time_ms
            )

            results.append(evaluation)

            # Learn from mistakes
            if not evaluation.diagnosis_match:
                await self._learn_from_mistake(test_case, evaluation, ai_response)
            else:
                await self._reinforce_success(test_case, evaluation, ai_response)

        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(results)

        # Store evaluation results
        self.evaluation_history.extend(results)
        self.accuracy_over_time.append(overall_metrics['overall_accuracy'])

        logger.info(f"✅ Evaluation complete: {overall_metrics['overall_accuracy']:.2%} accuracy")

        return {
            "timestamp": datetime.now().isoformat(),
            "test_cases_count": len(self.test_cases),
            "results": [self._result_to_dict(r) for r in results],
            "overall_metrics": overall_metrics,
            "learning_summary": {
                "mistakes_learned": len(self.mistake_examples),
                "successes_reinforced": len(self.correct_examples),
                "accuracy_trend": self.accuracy_over_time[-5:] if self.accuracy_over_time else []
            }
        }

    def _evaluate_prediction(self,
                            test_case: TestCase,
                            predicted_diagnosis: Dict[str, Any],
                            ai_response: Dict[str, Any],
                            response_time_ms: float) -> EvaluationResult:
        """Evaluate a single prediction against expected results"""

        # Check diagnosis match
        diagnosis_match = False
        if predicted_diagnosis:
            pred_icd = predicted_diagnosis.get("primary_diagnosis", {}).get("icd_code", "")
            exp_icd = test_case.expected_diagnosis["icd_code"]
            diagnosis_match = pred_icd == exp_icd

        # Calculate confidence
        confidence_score = predicted_diagnosis.get("confidence", 0.0)

        # Evaluate treatment accuracy
        treatment_accuracy = self._calculate_treatment_accuracy(
            predicted_treatment=ai_response.get("treatment", {}),
            expected_treatment=test_case.expected_treatment
        )

        # Error analysis if wrong
        error_analysis = None
        if not diagnosis_match:
            error_analysis = self._analyze_error(
                test_case=test_case,
                predicted_diagnosis=predicted_diagnosis,
                ai_response=ai_response
            )

        return EvaluationResult(
            test_case_id=test_case.id,
            predicted_diagnosis=predicted_diagnosis,
            expected_diagnosis=test_case.expected_diagnosis,
            diagnosis_match=diagnosis_match,
            confidence_score=confidence_score,
            treatment_accuracy=treatment_accuracy,
            response_time_ms=response_time_ms,
            error_analysis=error_analysis
        )

    def _calculate_treatment_accuracy(self,
                                     predicted_treatment: Dict[str, Any],
                                     expected_treatment: Dict[str, Any]) -> float:
        """Calculate how accurate the treatment recommendations are"""

        score = 0.0
        total = 0

        # Check medications
        if "medications" in expected_treatment:
            expected_meds = set(expected_treatment["medications"])
            predicted_meds = set(predicted_treatment.get("medications", []))
            if expected_meds:
                score += len(expected_meds.intersection(predicted_meds)) / len(expected_meds)
                total += 1

        # Check lifestyle recommendations
        if "lifestyle" in expected_treatment:
            expected_lifestyle = set(expected_treatment["lifestyle"])
            predicted_lifestyle = set(predicted_treatment.get("lifestyle_recommendations", []))
            if expected_lifestyle:
                score += len(expected_lifestyle.intersection(predicted_lifestyle)) / len(expected_lifestyle)
                total += 1

        return score / total if total > 0 else 0.0

    def _analyze_error(self,
                      test_case: TestCase,
                      predicted_diagnosis: Dict[str, Any],
                      ai_response: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze why the prediction was wrong"""

        return {
            "error_type": self._classify_error_type(test_case, predicted_diagnosis),
            "missed_keywords": self._find_missed_keywords(test_case.symptoms),
            "confidence_issue": "overconfident" if predicted_diagnosis.get("confidence", 0) > 0.7 else "underconfident",
            "severity_mismatch": test_case.severity != ai_response.get("triage", {}).get("priority", ""),
            "suggested_improvements": self._suggest_improvements(test_case, predicted_diagnosis)
        }

    def _classify_error_type(self, test_case: TestCase, predicted_diagnosis: Dict[str, Any]) -> str:
        """Classify the type of error"""

        if not predicted_diagnosis or not predicted_diagnosis.get("primary_diagnosis"):
            return "no_diagnosis"

        pred_category = predicted_diagnosis.get("primary_diagnosis", {}).get("category", "")
        exp_category = test_case.expected_diagnosis.get("category", "")

        if pred_category != exp_category:
            return "wrong_category"

        return "wrong_specific_diagnosis"

    def _find_missed_keywords(self, symptoms: str) -> List[str]:
        """Find important keywords that might have been missed"""

        critical_keywords = {
            "arthritis": ["ข้อ", "บวม", "แดง", "ร้อน", "ข้อติด"],
            "diabetes": ["เบาหวาน", "น้ำตาลสูง", "ปัสสาวะบ่อย", "กระหายน้ำ"],
            "cardiac": ["เจ็บหน้าอก", "หายใจลำบาก", "ปวดร้าวไปแขน"],
            "emergency": ["รุนแรง", "ฉับพลัน", "ทันที", "เร่งด่วน"]
        }

        missed = []
        for category, keywords in critical_keywords.items():
            for keyword in keywords:
                if keyword in symptoms and keyword not in missed:
                    # This keyword should have triggered better diagnosis
                    missed.append(keyword)

        return missed

    def _suggest_improvements(self, test_case: TestCase, predicted_diagnosis: Dict[str, Any]) -> List[str]:
        """Suggest how to improve the diagnosis"""

        suggestions = []

        if test_case.severity == "critical" and predicted_diagnosis.get("confidence", 0) < 0.9:
            suggestions.append("Increase confidence for critical symptoms")

        if test_case.category == "emergency":
            suggestions.append("Add emergency detection for these symptoms")

        if not predicted_diagnosis.get("primary_diagnosis"):
            suggestions.append(f"Add pattern recognition for {test_case.expected_diagnosis['name']}")

        return suggestions

    async def _learn_from_mistake(self,
                                 test_case: TestCase,
                                 evaluation: EvaluationResult,
                                 ai_response: Dict[str, Any]) -> None:
        """Learn from incorrect predictions"""

        # Create mistake example for few-shot learning
        mistake_example = {
            "symptoms": test_case.symptoms,
            "wrong_diagnosis": evaluation.predicted_diagnosis,
            "correct_diagnosis": test_case.expected_diagnosis,
            "correct_treatment": test_case.expected_treatment,
            "error_analysis": evaluation.error_analysis,
            "learning_prompt": self._create_mistake_template(test_case, evaluation)
        }

        self.mistake_examples.append(mistake_example)

        # Store in memory agent for future reference
        await memory_agent.store_interaction(
            session_id=f"eval_mistake_{test_case.id}",
            patient_id=None,
            symptoms=test_case.symptoms,
            diagnosis=test_case.expected_diagnosis,  # Store correct diagnosis
            treatment=test_case.expected_treatment,
            doctor_feedback={
                "correction": True,
                "correct_diagnosis": test_case.expected_diagnosis["name"],
                "original_was_wrong": True,
                "error_type": evaluation.error_analysis["error_type"]
            }
        )

        logger.info(f"📚 Learned from mistake: {test_case.id} - {evaluation.error_analysis['error_type']}")

    async def _reinforce_success(self,
                                test_case: TestCase,
                                evaluation: EvaluationResult,
                                ai_response: Dict[str, Any]) -> None:
        """Reinforce correct predictions"""

        # Create success example
        success_example = {
            "symptoms": test_case.symptoms,
            "correct_diagnosis": evaluation.predicted_diagnosis,
            "confidence": evaluation.confidence_score,
            "treatment_accuracy": evaluation.treatment_accuracy,
            "reinforcement_prompt": self._create_success_template(test_case, evaluation)
        }

        self.correct_examples.append(success_example)

        # Store successful case in memory
        await memory_agent.store_interaction(
            session_id=f"eval_success_{test_case.id}",
            patient_id=None,
            symptoms=test_case.symptoms,
            diagnosis=evaluation.predicted_diagnosis,
            treatment=ai_response.get("treatment", {}),
            doctor_feedback={
                "approved": True,
                "confidence_boost": 0.1
            }
        )

        logger.info(f"✅ Reinforced success: {test_case.id}")

    def _create_mistake_template(self, test_case: TestCase, evaluation: EvaluationResult) -> str:
        """Create few-shot learning template from mistake"""

        return f"""
LEARNING FROM MISTAKE:
Symptoms: {test_case.symptoms}

❌ WRONG Answer: {evaluation.predicted_diagnosis.get('primary_diagnosis', {}).get('name', 'No diagnosis')}
✅ CORRECT Answer: {test_case.expected_diagnosis['name']} (ICD: {test_case.expected_diagnosis['icd_code']})

KEY LEARNING POINTS:
1. When you see symptoms like "{', '.join(evaluation.error_analysis['missed_keywords'][:3])}",
   consider {test_case.expected_diagnosis['category']} conditions
2. {test_case.expected_diagnosis['name']} typically presents with these symptoms
3. Confidence should be {test_case.expected_diagnosis['confidence']} for this diagnosis

CORRECT TREATMENT:
- Medications: {', '.join(test_case.expected_treatment.get('medications', []))}
- Lifestyle: {', '.join(test_case.expected_treatment.get('lifestyle', []))}

Remember: {' '.join(evaluation.error_analysis['suggested_improvements'])}
"""

    def _create_success_template(self, test_case: TestCase, evaluation: EvaluationResult) -> str:
        """Create reinforcement template from success"""

        return f"""
SUCCESSFUL DIAGNOSIS:
Symptoms: {test_case.symptoms}
✅ Correct Diagnosis: {evaluation.predicted_diagnosis['primary_diagnosis']['name']}
Confidence: {evaluation.confidence_score:.2%}

PATTERN TO REMEMBER:
- These symptoms strongly indicate {test_case.expected_diagnosis['category']} condition
- High confidence ({evaluation.confidence_score:.2%}) was appropriate
- Treatment recommendations were {evaluation.treatment_accuracy:.0%} accurate

Continue using this pattern for similar cases.
"""

    def _create_edge_case_template(self, test_case: TestCase, evaluation: EvaluationResult) -> str:
        """Create template for edge cases"""

        return f"""
EDGE CASE LEARNING:
This is an unusual presentation that requires special attention.

Symptoms: {test_case.symptoms}
Diagnosis: {test_case.expected_diagnosis['name']}

SPECIAL CONSIDERATIONS:
- Severity: {test_case.severity}
- Category: {test_case.category}
- Ground Truth Source: {test_case.ground_truth_source}

Always consider rare conditions when common diagnoses don't fit perfectly.
"""

    def _calculate_overall_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate overall evaluation metrics"""

        total = len(results)
        correct = sum(1 for r in results if r.diagnosis_match)

        # Group by category
        category_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})
        for i, result in enumerate(results):
            category = self.test_cases[i].category
            category_accuracy[category]["total"] += 1
            if result.diagnosis_match:
                category_accuracy[category]["correct"] += 1

        # Calculate category accuracies
        for category in category_accuracy:
            cat_data = category_accuracy[category]
            cat_data["accuracy"] = cat_data["correct"] / cat_data["total"] if cat_data["total"] > 0 else 0

        # Average metrics
        avg_confidence = np.mean([r.confidence_score for r in results])
        avg_treatment_accuracy = np.mean([r.treatment_accuracy for r in results])
        avg_response_time = np.mean([r.response_time_ms for r in results])

        return {
            "overall_accuracy": correct / total if total > 0 else 0,
            "total_cases": total,
            "correct_diagnoses": correct,
            "category_accuracy": dict(category_accuracy),
            "average_confidence": avg_confidence,
            "average_treatment_accuracy": avg_treatment_accuracy,
            "average_response_time_ms": avg_response_time,
            "critical_cases_accuracy": sum(1 for i, r in enumerate(results)
                                          if r.diagnosis_match and self.test_cases[i].severity == "critical") /
                                      sum(1 for tc in self.test_cases if tc.severity == "critical")
        }

    def _result_to_dict(self, result: EvaluationResult) -> Dict[str, Any]:
        """Convert evaluation result to dictionary"""

        return {
            "test_case_id": result.test_case_id,
            "diagnosis_match": result.diagnosis_match,
            "confidence_score": result.confidence_score,
            "treatment_accuracy": result.treatment_accuracy,
            "response_time_ms": result.response_time_ms,
            "predicted": result.predicted_diagnosis.get("primary_diagnosis", {}).get("name", "None"),
            "expected": result.expected_diagnosis["name"],
            "error_analysis": result.error_analysis
        }

    def get_few_shot_examples(self, n_examples: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Get few-shot examples for model improvement"""

        return {
            "mistake_corrections": self.mistake_examples[-n_examples:],
            "success_patterns": self.correct_examples[-n_examples:],
            "total_mistakes": len(self.mistake_examples),
            "total_successes": len(self.correct_examples),
            "learning_rate": len(self.correct_examples) / (len(self.correct_examples) + len(self.mistake_examples))
                           if (self.correct_examples or self.mistake_examples) else 0
        }

    def get_improvement_suggestions(self) -> List[str]:
        """Get suggestions for model improvement based on evaluation"""

        suggestions = []

        # Analyze common error patterns
        if self.mistake_examples:
            error_types = defaultdict(int)
            for mistake in self.mistake_examples:
                if mistake.get("error_analysis"):
                    error_types[mistake["error_analysis"]["error_type"]] += 1

            most_common_error = max(error_types.items(), key=lambda x: x[1])[0]
            suggestions.append(f"Focus on improving {most_common_error} errors (most common)")

        # Check category-specific issues
        if self.evaluation_history:
            latest_eval = self.evaluation_history[-len(self.test_cases):]
            for i, result in enumerate(latest_eval):
                if not result.diagnosis_match and self.test_cases[i].category == "chronic":
                    suggestions.append("Improve chronic disease detection (diabetes, arthritis)")
                    break

        # Check emergency detection
        emergency_accuracy = sum(1 for i, r in enumerate(self.evaluation_history[-len(self.test_cases):])
                                if r.diagnosis_match and self.test_cases[i].severity == "critical")
        if emergency_accuracy < len([tc for tc in self.test_cases if tc.severity == "critical"]):
            suggestions.append("Enhance emergency condition detection")

        return suggestions

# Singleton instance
model_evaluator = ModelEvaluator()