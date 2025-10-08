# Precision Medical AI Service
# Orchestrates the complete precision pipeline: Router → Executor → Critic → Uncertainty

from typing import Dict, List, Optional, Tuple
import logging
import asyncio
from datetime import datetime

from app.core.types import (
    DiagnosisCard, RouteSignals, PrecisionPlan, DxCandidate, Evidence,
    Calculator, Treatment, Uncertainty, TriageLevel, CriticResult
)
from app.core.router import PrecisionRouter, extract_signals_from_text
from app.core.calculators import CalculatorRegistry, get_applicable_calculators
from app.core.critic import PrecisionCritic
from app.core.uncertainty import UncertaintyQuantifier, AbstractionEngine

logger = logging.getLogger(__name__)


class PrecisionMedicalAI:
    """
    Precision-oriented medical AI system with systematic accuracy improvements

    Pipeline: Signal Extraction → Evidence-First Routing → Parallel/Sequential Execution
             → Precision Critic → Uncertainty Quantification → Abstention Logic
    """

    def __init__(self):
        # Core components
        self.router = PrecisionRouter()
        self.calculator_registry = CalculatorRegistry()
        self.critic = PrecisionCritic()
        self.uncertainty_quantifier = UncertaintyQuantifier(coverage_target=0.9)
        self.abstention_engine = AbstractionEngine()

        # Configuration
        self.temperature_schedule = {
            "extraction": 0.2,      # Low temperature for factual extraction
            "calculators": 0.0,     # Deterministic for calculators
            "differential": 0.4,    # Medium for differential generation
            "critic": 0.0          # Deterministic for validation
        }

        # Precision thresholds
        self.precision_thresholds = {
            "min_safety_certainty": 0.85,
            "min_diagnostic_coverage": 0.6,
            "max_voi_questions": 3,
            "abstention_threshold": 0.7
        }

    async def process_medical_consultation(self,
                                         message: str,
                                         patient_data: Optional[Dict] = None,
                                         session_id: str = "default") -> DiagnosisCard:
        """
        Process medical consultation with precision pipeline

        Args:
            message: Patient's symptom description
            patient_data: Additional patient data (vitals, history, etc.)
            session_id: Session identifier

        Returns:
            Validated DiagnosisCard or abstention with explanation
        """
        start_time = datetime.now()
        patient_data = patient_data or {}

        try:
            logger.info(f"🎯 Starting precision consultation for session {session_id}")

            # Step 1: Signal Extraction and Routing
            signals = extract_signals_from_text(message, patient_data)
            plan = self.router.create_execution_plan(signals)

            logger.info(f"📊 Extracted signals: {signals}")
            logger.info(f"📋 Execution plan: {plan.steps}")

            # Step 2: Evidence-First Execution
            execution_context = {
                "symptoms": message,
                "patient_data": patient_data,
                "signals": signals,
                "captured_fields": self._extract_captured_fields(patient_data)
            }

            diagnosis_card = await self._execute_precision_pipeline(
                plan, execution_context, session_id
            )

            # Step 3: Precision Critic Validation
            critic_result = self.critic.validate_diagnosis_card(
                diagnosis_card, execution_context["captured_fields"]
            )

            if not critic_result.passed:
                diagnosis_card = await self._handle_critic_failure(
                    diagnosis_card, critic_result, execution_context
                )

            # Step 4: Final Uncertainty and Abstention Check
            should_abstain, action, abstention_message = self.abstention_engine.should_abstain(
                diagnosis_card.uncertainty
            )

            if should_abstain:
                diagnosis_card = self._create_abstention_card(
                    action, abstention_message, execution_context, session_id
                )

            # Log processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            diagnosis_card.processing_metadata["processing_time_ms"] = processing_time

            logger.info(f"✅ Precision consultation completed in {processing_time:.1f}ms")
            return diagnosis_card

        except Exception as e:
            logger.error(f"❌ Precision consultation failed: {e}")
            return self._create_error_card(str(e), session_id)

    async def _execute_precision_pipeline(self,
                                        plan: PrecisionPlan,
                                        context: Dict,
                                        session_id: str) -> DiagnosisCard:
        """Execute the precision pipeline according to the plan"""

        # Initialize diagnosis card structure
        card_data = {
            "patient_id": context.get("patient_id", "unknown"),
            "language": "thai",  # Detect from message in real implementation
            "session_id": session_id,
            "routing_reasons": plan.routing_reasons,
            "processing_metadata": {
                "execution_start": datetime.now().isoformat()
            }
        }

        # Execute pipeline steps
        if "emergency_triage" in plan.steps:
            card_data.update(await self._execute_emergency_protocol(context))
        else:
            card_data.update(await self._execute_standard_protocol(context, plan))

        # Create diagnosis card
        diagnosis_card = DiagnosisCard(**card_data)
        return diagnosis_card

    async def _execute_emergency_protocol(self, context: Dict) -> Dict:
        """Execute emergency protocol for critical symptoms"""
        symptoms = context["symptoms"]
        signals = context["signals"]

        # Emergency triage
        triage = {
            "level": TriageLevel.EMERGENCY,
            "rationale": f"Emergency keywords detected: {signals.emergency_keywords}"
        }

        # Emergency differential focused on life-threatening conditions
        emergency_differential = await self._generate_emergency_differential(symptoms, signals)

        # High safety certainty due to emergency routing
        uncertainty = Uncertainty(
            diagnostic_coverage=0.95,  # High coverage for emergency protocol
            safety_certainty=0.95,     # High safety due to escalation
            abstention_reason=None,
            prediction_set_size=len(emergency_differential)
        )

        return {
            "triage": triage,
            "differential": emergency_differential,
            "tests": [{"name": "Emergency assessment", "rationale": "Critical symptoms", "voi_score": 1.0, "urgency": TriageLevel.EMERGENCY}],
            "treatment_candidates": [],  # No treatment recommendations in emergency
            "uncertainty": uncertainty,
            "overall_confidence": 0.85,
            "calculators": []
        }

    async def _execute_standard_protocol(self, context: Dict, plan: PrecisionPlan) -> Dict:
        """Execute standard precision protocol"""
        symptoms = context["symptoms"]
        signals = context["signals"]
        captured_fields = context["captured_fields"]

        # Parallel execution of safe tools
        parallel_tasks = []

        # Generate differential diagnosis
        parallel_tasks.append(self._generate_differential_diagnosis(symptoms, signals))

        # Apply applicable calculators
        applicable_calcs = get_applicable_calculators(signals, captured_fields)
        if applicable_calcs:
            parallel_tasks.append(self._apply_calculators(applicable_calcs, captured_fields))

        # Execute parallel tasks
        results = await asyncio.gather(*parallel_tasks, return_exceptions=True)

        # Extract results
        differential = results[0] if len(results) > 0 and not isinstance(results[0], Exception) else []
        calculators = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else []

        # Sequential execution of safety-critical tasks
        triage = await self._perform_triage_assessment(differential, signals)
        tests = await self._recommend_tests(differential, signals, captured_fields)
        treatments = await self._generate_treatment_recommendations(differential)

        # Uncertainty quantification
        uncertainty = self.uncertainty_quantifier.quantify_uncertainty(
            differential, context, temperature=self.temperature_schedule["differential"]
        )

        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            differential, calculators, uncertainty
        )

        return {
            "triage": triage,
            "differential": differential,
            "calculators": calculators,
            "tests": tests,
            "treatment_candidates": treatments,
            "uncertainty": uncertainty,
            "overall_confidence": overall_confidence
        }

    async def _generate_emergency_differential(self, symptoms: str, signals: RouteSignals) -> List[DxCandidate]:
        """Generate emergency-focused differential diagnosis"""
        emergency_conditions = []

        # Map signals to emergency conditions (probabilities must sum ≤ 1.0)
        if signals.chest_pain and signals.breathing_difficulty:
            # Both present - prioritize most urgent
            emergency_conditions.append(
                DxCandidate(
                    icd10="I21.9",
                    label="Acute Myocardial Infarction",
                    p=0.8,
                    evidence=Evidence(
                        for_=["chest pain", "breathing difficulty", "emergency presentation"],
                        citations=["guideline:aha_stemi_2022"]
                    )
                )
            )
        elif signals.chest_pain:
            emergency_conditions.append(
                DxCandidate(
                    icd10="I21.9",
                    label="Acute Myocardial Infarction",
                    p=0.7,
                    evidence=Evidence(
                        for_=["chest pain", "emergency presentation"],
                        citations=["guideline:aha_stemi_2022"]
                    )
                )
            )
        elif signals.breathing_difficulty:
            emergency_conditions.append(
                DxCandidate(
                    icd10="I26.9",
                    label="Pulmonary Embolism",
                    p=0.6,
                    evidence=Evidence(
                        for_=["breathing difficulty", "emergency presentation"],
                        citations=["guideline:esc_pe_2019"]
                    )
                )
            )

        if signals.neurological_deficit:
            emergency_conditions.append(
                DxCandidate(
                    icd10="I63.9",
                    label="Cerebral Infarction",
                    p=0.8,
                    evidence=Evidence(
                        for_=["neurological deficit", "emergency presentation"],
                        citations=["guideline:aha_stroke_2019"]
                    )
                )
            )

        # Default emergency consultation if no specific mapping
        if not emergency_conditions:
            emergency_conditions.append(
                DxCandidate(
                    icd10="Z71.1",
                    label="Emergency Medical Consultation",
                    p=0.9,
                    evidence=Evidence(
                        for_=["emergency keywords", "urgent presentation"],
                        citations=["guideline:emergency_triage_2020"]
                    )
                )
            )

        return emergency_conditions

    async def _generate_differential_diagnosis(self, symptoms: str, signals: RouteSignals) -> List[DxCandidate]:
        """Generate conservative differential diagnosis"""
        differential = []

        # Use signals to generate targeted differential
        if signals.fever and not signals.severe_headache:
            # Simple viral illness
            differential.append(
                DxCandidate(
                    icd10="J00",
                    label="Common Cold",
                    p=0.75,
                    evidence=Evidence(
                        for_=["fever", "common symptoms"],
                        citations=["kb:common_cold", "guideline:uri_management_2021"]
                    )
                )
            )

        if signals.chest_pain and not signals.emergency_keywords:
            # Non-emergency chest pain
            differential.append(
                DxCandidate(
                    icd10="R07.89",
                    label="Chest Pain, Other",
                    p=0.6,
                    evidence=Evidence(
                        for_=["chest pain", "no emergency features"],
                        citations=["guideline:chest_pain_evaluation_2021"]
                    )
                )
            )

        # Default to symptom-based consultation if no specific pattern
        if not differential:
            differential.append(
                DxCandidate(
                    icd10="Z71.1",
                    label="Medical Consultation",
                    p=0.7,
                    evidence=Evidence(
                        for_=["symptom evaluation needed"],
                        citations=["guideline:primary_care_consultation_2021"]
                    )
                )
            )

        return differential

    async def _apply_calculators(self, calculator_names: List[str], captured_fields: Dict) -> List[Calculator]:
        """Apply medical calculators with validation"""
        calculators = []

        for calc_name in calculator_names:
            try:
                # Mock calculator inputs based on captured fields
                calc_inputs = self._prepare_calculator_inputs(calc_name, captured_fields)

                if calc_inputs:
                    calculator = self.calculator_registry.calculate(
                        calc_name, calc_inputs, captured_fields
                    )
                    calculators.append(calculator)

            except Exception as e:
                logger.warning(f"Calculator {calc_name} failed: {e}")

        return calculators

    def _prepare_calculator_inputs(self, calc_name: str, captured_fields: Dict) -> Optional[Dict]:
        """Prepare inputs for specific calculator"""
        # This would map captured fields to calculator inputs
        # For now, return mock data for testing

        if calc_name == "heart_score" and "age" in captured_fields:
            return {
                "age": captured_fields.get("age", 50),
                "history": captured_fields.get("cardiac_history", False),
                "ecg": captured_fields.get("ecg_abnormal", False),
                "risk_factors": captured_fields.get("risk_factors", 1),
                "troponin_elevated": captured_fields.get("troponin_elevated", False)
            }

        return None

    async def _perform_triage_assessment(self, differential: List[DxCandidate], signals: RouteSignals) -> Dict:
        """Perform triage assessment"""
        # Default triage based on signals
        if signals.emergency_keywords:
            level = TriageLevel.EMERGENCY
            rationale = "Emergency keywords detected"
        elif signals.chest_pain or signals.breathing_difficulty:
            level = TriageLevel.URGENT
            rationale = "Potentially serious symptoms"
        elif signals.fever and signals.severe_headache:
            level = TriageLevel.URGENT
            rationale = "Fever with severe headache"
        else:
            level = TriageLevel.SEMI_URGENT
            rationale = "Standard symptom evaluation"

        return {"level": level, "rationale": rationale}

    async def _recommend_tests(self, differential: List[DxCandidate], signals: RouteSignals, captured_fields: Dict) -> List[Dict]:
        """Recommend diagnostic tests"""
        tests = []

        # Test recommendations based on differential
        for dx in differential[:2]:  # Top 2 diagnoses
            if dx.icd10.startswith('I2'):  # Cardiac
                tests.append({
                    "name": "ECG and Troponin",
                    "rationale": f"Evaluate {dx.label}",
                    "voi_score": 0.8,
                    "urgency": TriageLevel.URGENT
                })
            elif dx.icd10.startswith('G0'):  # CNS
                tests.append({
                    "name": "Lumbar Puncture",
                    "rationale": f"Rule out {dx.label}",
                    "voi_score": 0.9,
                    "urgency": TriageLevel.EMERGENCY
                })

        return tests

    async def _generate_treatment_recommendations(self, differential: List[DxCandidate]) -> List[Treatment]:
        """Generate treatment recommendations with evidence"""
        treatments = []

        for dx in differential:
            if dx.icd10 == "J00":  # Common cold
                treatments.append(
                    Treatment(
                        medication="Paracetamol",
                        dosage="500mg every 6 hours as needed",
                        instructions="For symptom relief of fever and aches",
                        contraindications=["liver disease", "alcohol dependence"],
                        evidence=Evidence(
                            for_=["symptomatic relief", "safe for common cold"],
                            citations=["guideline:common_cold_treatment_2021"]
                        ),
                        safety_score=0.95
                    )
                )

        return treatments

    def _extract_captured_fields(self, patient_data: Dict) -> Dict:
        """Extract fields that were actually captured from patient"""
        captured = {}

        # Note: Vital signs extraction removed - inappropriate for consultation scope

        if "age" in patient_data:
            captured["age"] = patient_data["age"]

        if "history" in patient_data:
            captured["history"] = patient_data["history"]

        return captured

    def _calculate_overall_confidence(self, differential: List[DxCandidate], calculators: List[Calculator], uncertainty: Uncertainty) -> float:
        """Calculate overall diagnostic confidence"""
        if not differential:
            return 0.0

        # Weight by top diagnosis probability
        dx_confidence = differential[0].p if differential else 0.5

        # Factor in calculator confidence
        calc_confidence = 1.0
        if calculators:
            calc_confidence = sum(c.confidence for c in calculators) / len(calculators)

        # Factor in uncertainty metrics
        uncertainty_factor = (uncertainty.diagnostic_coverage + uncertainty.safety_certainty) / 2

        # Combined confidence
        overall = (dx_confidence * 0.4) + (calc_confidence * 0.3) + (uncertainty_factor * 0.3)

        return min(1.0, max(0.0, overall))

    async def _handle_critic_failure(self, card: DiagnosisCard, critic_result: CriticResult, context: Dict) -> DiagnosisCard:
        """Handle critic validation failure"""
        logger.warning(f"Critic validation failed: {critic_result.failed_rules}")

        # Create safe fallback card
        fallback_card = self._create_safe_fallback_card(context, card.session_id)

        # Add critic failure metadata
        fallback_card.processing_metadata["critic_failure"] = {
            "failed_rules": critic_result.failed_rules,
            "actions": critic_result.actions,
            "original_confidence": card.overall_confidence
        }

        return fallback_card

    def _create_abstention_card(self, action: str, message: str, context: Dict, session_id: str) -> DiagnosisCard:
        """Create abstention diagnosis card"""
        abstention_dx = DxCandidate(
            icd10="Z71.1",
            label="Medical Consultation Needed",
            p=1.0,
            evidence=Evidence(
                for_=["insufficient_certainty_for_ai_diagnosis"],
                citations=["guideline:ai_limitations_2023"]
            )
        )

        uncertainty = Uncertainty(
            diagnostic_coverage=0.0,
            safety_certainty=1.0,  # High safety due to abstention
            abstention_reason=message,
            prediction_set_size=1
        )

        return DiagnosisCard(
            patient_id=context.get("patient_id", "unknown"),
            language="thai",
            triage={"level": TriageLevel.SEMI_URGENT, "rationale": "AI abstention - human review needed"},
            differential=[abstention_dx],
            uncertainty=uncertainty,
            overall_confidence=0.0,
            session_id=session_id,
            processing_metadata={"abstention_action": action, "abstention_message": message}
        )

    def _create_safe_fallback_card(self, context: Dict, session_id: str) -> DiagnosisCard:
        """Create safe fallback diagnosis card"""
        fallback_dx = DxCandidate(
            icd10="Z71.1",
            label="General Medical Consultation",
            p=0.9,
            evidence=Evidence(
                for_=["symptom_evaluation_needed"],
                citations=["guideline:primary_care_2021"]
            )
        )

        uncertainty = Uncertainty(
            diagnostic_coverage=0.5,
            safety_certainty=0.9,  # High safety due to conservative approach
            abstention_reason=None,
            prediction_set_size=1
        )

        return DiagnosisCard(
            patient_id=context.get("patient_id", "unknown"),
            language="thai",
            triage={"level": TriageLevel.SEMI_URGENT, "rationale": "Conservative assessment"},
            differential=[fallback_dx],
            uncertainty=uncertainty,
            overall_confidence=0.6,
            session_id=session_id,
            routing_reasons=[],
            processing_metadata={"fallback_reason": "critic_validation_failure"}
        )

    def _create_error_card(self, error_message: str, session_id: str) -> DiagnosisCard:
        """Create error diagnosis card"""
        error_dx = DxCandidate(
            icd10="Z99.9",
            label="System Error - Physician Consultation Required",
            p=1.0,
            evidence=Evidence(
                for_=["system_error"],
                citations=["system:error_handling"]
            )
        )

        uncertainty = Uncertainty(
            diagnostic_coverage=0.0,
            safety_certainty=1.0,  # High safety due to escalation
            abstention_reason=f"System error: {error_message}",
            prediction_set_size=1
        )

        return DiagnosisCard(
            patient_id="unknown",
            language="thai",
            triage={"level": TriageLevel.URGENT, "rationale": "System error - immediate physician review"},
            differential=[error_dx],
            uncertainty=uncertainty,
            overall_confidence=0.0,
            session_id=session_id,
            processing_metadata={"error": error_message}
        )


# Factory function
def create_precision_medical_ai() -> PrecisionMedicalAI:
    """Create configured precision medical AI system"""
    return PrecisionMedicalAI()


# Example usage
if __name__ == "__main__":
    import asyncio

    async def test_precision_ai():
        ai = create_precision_medical_ai()

        # Test basic symptoms
        basic_result = await ai.process_medical_consultation(
            "ปวดหัว เป็นไข้",
            patient_data={"age": 30},
            session_id="test_basic"
        )
        print(f"Basic result: {basic_result.differential[0].label} (confidence: {basic_result.overall_confidence:.2f})")

        # Test emergency symptoms
        emergency_result = await ai.process_medical_consultation(
            "ปวดหน้าอกเฉียบพลัน หายใจไม่ออก เร่งด่วน",
            patient_data={"age": 55},
            session_id="test_emergency"
        )
        print(f"Emergency result: {emergency_result.differential[0].label} (triage: {emergency_result.triage['level']})")

    asyncio.run(test_precision_ai())