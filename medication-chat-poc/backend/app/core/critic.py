# Precision Critic System with Blocking Rules
# Validates diagnosis cards and blocks unsafe/imprecise outputs

from typing import List, Dict, Optional, Tuple
import logging
from app.core.types import DiagnosisCard, CriticResult, DxCandidate, Calculator, Treatment, TriageLevel

logger = logging.getLogger(__name__)


class PrecisionCritic:
    """Blocking critic system for medical AI precision and safety"""

    def __init__(self):
        self.validation_rules = self._build_validation_rules()
        self.safety_thresholds = self._build_safety_thresholds()

    def _build_validation_rules(self) -> Dict:
        """Define comprehensive validation rules"""
        return {
            "treatment_guideline_citation": {
                "description": "Any treatment suggestion must include ≥1 guideline citation",
                "severity": "blocking",
                "action": "request_info"
            },
            "high_risk_diagnosis_evidence": {
                "description": "High-risk diagnoses require specific supporting evidence",
                "severity": "blocking",
                "action": "request_info"
            },
            "calculator_input_completeness": {
                "description": "Calculators may only use captured fields, no hallucinated values",
                "severity": "blocking",
                "action": "request_info"
            },
            "meningitis_without_redflags": {
                "description": "Meningitis without neck stiffness/AMS/photophobia must be downgraded",
                "severity": "blocking",
                "action": "downgrade_diagnosis"
            },
            "serious_diagnosis_without_specificity": {
                "description": "Serious diagnoses require characteristic symptoms",
                "severity": "blocking",
                "action": "downgrade_diagnosis"
            },
            "safety_certainty_threshold": {
                "description": "Safety certainty must meet minimum threshold",
                "severity": "blocking",
                "action": "escalate"
            },
            "triage_consistency": {
                "description": "Triage level must match diagnosis severity",
                "severity": "warning",
                "action": "review"
            },
            "differential_probability_coherence": {
                "description": "Differential probabilities must be coherent",
                "severity": "blocking",
                "action": "recalculate"
            }
        }

    def _build_safety_thresholds(self) -> Dict:
        """Define safety thresholds for validation"""
        return {
            "min_safety_certainty": 0.85,
            "min_calculator_confidence": 0.8,
            "max_high_risk_probability_without_evidence": 0.3,
            "min_evidence_items_for_treatment": 1,
            "max_differential_size": 5
        }

    def validate_diagnosis_card(self, card: DiagnosisCard, captured_fields: Dict = None) -> CriticResult:
        """
        Comprehensive validation of diagnosis card

        Args:
            card: DiagnosisCard to validate
            captured_fields: Fields actually captured from patient (for calculator validation)

        Returns:
            CriticResult with pass/fail and required actions
        """
        failed_rules = []
        actions = []
        warnings = []

        # Run all validation rules
        failed_rules.extend(self._validate_treatment_guidelines(card))
        failed_rules.extend(self._validate_high_risk_diagnoses(card))
        failed_rules.extend(self._validate_calculator_integrity(card, captured_fields or {}))
        failed_rules.extend(self._validate_meningitis_criteria(card))
        failed_rules.extend(self._validate_serious_diagnoses(card))
        failed_rules.extend(self._validate_safety_thresholds(card))
        warnings.extend(self._validate_triage_consistency(card))
        failed_rules.extend(self._validate_differential_coherence(card))

        # Determine actions based on failed rules
        for rule in failed_rules:
            rule_config = self.validation_rules.get(rule, {})
            action = rule_config.get("action", "review")
            if action not in actions:
                actions.append(action)

        passed = len(failed_rules) == 0

        # Generate rationale
        rationale = self._generate_validation_rationale(card, failed_rules, warnings)

        logger.info(f"Critic validation: {'PASSED' if passed else 'FAILED'} - {len(failed_rules)} rule violations")

        return CriticResult(
            passed=passed,
            failed_rules=failed_rules,
            actions=actions,
            rationale=rationale
        )

    def _validate_treatment_guidelines(self, card: DiagnosisCard) -> List[str]:
        """Validate that treatments have guideline citations"""
        failed_rules = []

        for treatment in card.treatment_candidates:
            has_guideline = any(
                citation.startswith('guideline:')
                for citation in treatment.evidence.citations
            )
            if not has_guideline:
                failed_rules.append("treatment_guideline_citation")
                logger.warning(f"Treatment '{treatment.instructions}' lacks guideline citation")

        return failed_rules

    def _validate_high_risk_diagnoses(self, card: DiagnosisCard) -> List[str]:
        """Validate high-risk diagnoses have supporting evidence"""
        failed_rules = []
        high_risk_icds = [
            'I2',  # IHD
            'I4',  # Heart failure
            'G0',  # CNS infections
            'G9',  # Nervous system disorders
            'R06', # Respiratory distress
            'R50', # Fever
            'I60', # Subarachnoid hemorrhage
            'I63', # Cerebral infarction
        ]

        for dx in card.differential[:3]:  # Check top 3
            is_high_risk = any(dx.icd10.startswith(prefix) for prefix in high_risk_icds)

            if is_high_risk:
                has_evidence = len(dx.evidence.for_) > 0
                high_probability = dx.p > self.safety_thresholds["max_high_risk_probability_without_evidence"]

                if high_probability and not has_evidence:
                    failed_rules.append("high_risk_diagnosis_evidence")
                    logger.warning(f"High-risk diagnosis {dx.icd10} ({dx.p:.2f}) lacks evidence")

        return failed_rules

    def _validate_calculator_integrity(self, card: DiagnosisCard, captured_fields: Dict) -> List[str]:
        """Validate calculators only use captured fields"""
        failed_rules = []

        for calc in card.calculators:
            # Check if inputs were actually captured
            calc_inputs = set(calc.inputs_used.keys())
            captured_field_names = set(captured_fields.keys())

            # Allow some flexibility for derived fields
            derived_fields = {'age_ge_50', 'hr_ge_100', 'heart_rate_gt_100'}  # Can be derived from age, heart_rate
            calc_inputs_basic = calc_inputs - derived_fields

            uncaptured_inputs = calc_inputs_basic - captured_field_names

            if uncaptured_inputs:
                failed_rules.append("calculator_input_completeness")
                logger.warning(f"Calculator {calc.name} uses uncaptured fields: {uncaptured_inputs}")

            # Check confidence threshold
            if calc.confidence < self.safety_thresholds["min_calculator_confidence"]:
                failed_rules.append("calculator_input_completeness")
                logger.warning(f"Calculator {calc.name} confidence too low: {calc.confidence}")

        return failed_rules

    def _validate_meningitis_criteria(self, card: DiagnosisCard) -> List[str]:
        """Validate meningitis diagnosis against red flag criteria"""
        failed_rules = []

        for dx in card.differential:
            if 'meningitis' in dx.label.lower() or dx.icd10.startswith('G0'):
                # Check for classic meningitis signs
                meningitis_signs = [
                    'neck stiffness', 'photophobia', 'altered mental status',
                    'คอแข็ง', 'เกลียดแสง', 'ซึม', 'สับสน'
                ]

                has_meningitis_signs = any(
                    any(sign.lower() in evidence.lower() for sign in meningitis_signs)
                    for evidence in dx.evidence.for_
                )

                if dx.p > 0.3 and not has_meningitis_signs:
                    failed_rules.append("meningitis_without_redflags")
                    logger.warning(f"Meningitis diagnosis {dx.p:.2f} without classic signs")

        return failed_rules

    def _validate_serious_diagnoses(self, card: DiagnosisCard) -> List[str]:
        """Validate serious diagnoses have characteristic symptoms"""
        failed_rules = []

        serious_conditions = {
            'I21': ['chest pain', 'troponin', 'ecg changes'],  # MI
            'I26': ['dyspnea', 'chest pain', 'd-dimer'],       # PE
            'G93': ['headache', 'vomiting', 'altered mental'], # Brain disorders
            'R06.02': ['dyspnea', 'hypoxia']                   # Respiratory failure
        }

        for dx in card.differential:
            for condition_icd, required_symptoms in serious_conditions.items():
                if dx.icd10.startswith(condition_icd[:3]):
                    has_characteristic_symptoms = any(
                        any(symptom.lower() in evidence.lower() for symptom in required_symptoms)
                        for evidence in dx.evidence.for_
                    )

                    if dx.p > 0.4 and not has_characteristic_symptoms:
                        failed_rules.append("serious_diagnosis_without_specificity")
                        logger.warning(f"Serious diagnosis {dx.icd10} lacks characteristic symptoms")

        return failed_rules

    def _validate_safety_thresholds(self, card: DiagnosisCard) -> List[str]:
        """Validate safety thresholds are met"""
        failed_rules = []

        if card.uncertainty.safety_certainty < self.safety_thresholds["min_safety_certainty"]:
            failed_rules.append("safety_certainty_threshold")
            logger.warning(f"Safety certainty too low: {card.uncertainty.safety_certainty}")

        return failed_rules

    def _validate_triage_consistency(self, card: DiagnosisCard) -> List[str]:
        """Validate triage level matches diagnosis severity (warning only)"""
        warnings = []

        triage_level = card.triage.get("level")
        top_diagnosis = card.differential[0] if card.differential else None

        if top_diagnosis and triage_level:
            # High-risk diagnoses should have urgent triage
            high_risk_icds = ['I2', 'I4', 'G0', 'R06']
            is_high_risk = any(top_diagnosis.icd10.startswith(prefix) for prefix in high_risk_icds)

            if is_high_risk and triage_level in [TriageLevel.NON_URGENT, TriageLevel.SEMI_URGENT]:
                warnings.append("triage_consistency")
                logger.warning(f"High-risk diagnosis {top_diagnosis.icd10} with low triage: {triage_level}")

        return warnings

    def _validate_differential_coherence(self, card: DiagnosisCard) -> List[str]:
        """Validate differential diagnosis probabilities make sense"""
        failed_rules = []

        if len(card.differential) > 1:
            # Check total probability doesn't exceed 1.0 (with small tolerance)
            total_p = sum(dx.p for dx in card.differential)
            if total_p > 1.1:
                failed_rules.append("differential_probability_coherence")
                logger.warning(f"Total differential probability exceeds 1.0: {total_p}")

            # Check probabilities are in descending order
            probabilities = [dx.p for dx in card.differential]
            if probabilities != sorted(probabilities, reverse=True):
                failed_rules.append("differential_probability_coherence")
                logger.warning("Differential probabilities not in descending order")

        return failed_rules

    def _generate_validation_rationale(self, card: DiagnosisCard, failed_rules: List[str], warnings: List[str]) -> str:
        """Generate human-readable validation rationale"""
        rationale_parts = []

        # Summary
        dx_count = len(card.differential)
        calc_count = len(card.calculators)
        treatment_count = len(card.treatment_candidates)

        rationale_parts.append(
            f"Validated {dx_count} diagnoses, {calc_count} calculators, {treatment_count} treatments"
        )

        # Failed rules
        if failed_rules:
            rule_descriptions = []
            for rule in failed_rules:
                rule_config = self.validation_rules.get(rule, {})
                desc = rule_config.get("description", rule)
                rule_descriptions.append(desc)

            rationale_parts.append(f"Failed rules: {'; '.join(rule_descriptions)}")

        # Warnings
        if warnings:
            rationale_parts.append(f"Warnings: {len(warnings)} consistency issues")

        # Safety assessment
        safety_score = card.uncertainty.safety_certainty
        rationale_parts.append(f"Safety certainty: {safety_score:.2f}")

        return ". ".join(rationale_parts)

    def suggest_improvements(self, card: DiagnosisCard, failed_rules: List[str]) -> List[str]:
        """Suggest specific improvements for failed validations"""
        suggestions = []

        if "treatment_guideline_citation" in failed_rules:
            suggestions.append("Add guideline citations (e.g., 'guideline:aha_chest_pain_2021') to treatment recommendations")

        if "high_risk_diagnosis_evidence" in failed_rules:
            suggestions.append("Provide specific evidence for high-risk diagnoses (symptoms, exam findings, test results)")

        if "calculator_input_completeness" in failed_rules:
            suggestions.append("Gather missing data for calculator inputs or use only calculators with complete data")

        if "meningitis_without_redflags" in failed_rules:
            suggestions.append("Assess for neck stiffness, photophobia, altered mental status before considering meningitis")

        if "safety_certainty_threshold" in failed_rules:
            suggestions.append("Escalate to physician due to low safety certainty; gather more information")

        return suggestions


# Blocking Guard Functions
def critic_guard_treatment(card: DiagnosisCard) -> None:
    """Blocking guard for treatment recommendations"""
    for treatment in card.treatment_candidates:
        has_guideline = any(
            citation.startswith('guideline:')
            for citation in treatment.evidence.citations
        )
        if not has_guideline:
            raise ValueError(f"Treatment '{treatment.instructions}' blocked: no guideline citation")


def critic_guard_calculator(calc: Calculator, captured_fields: Dict) -> None:
    """Blocking guard for calculator usage"""
    calc_inputs = set(calc.inputs_used.keys())
    captured_field_names = set(captured_fields.keys())

    # Allow derived fields
    derived_fields = {'age_ge_50', 'hr_ge_100', 'heart_rate_gt_100'}
    calc_inputs_basic = calc_inputs - derived_fields

    uncaptured_inputs = calc_inputs_basic - captured_field_names

    if uncaptured_inputs:
        raise ValueError(f"Calculator {calc.name} blocked: uses uncaptured fields {uncaptured_inputs}")


def critic_guard_safety(card: DiagnosisCard, min_safety: float = 0.85) -> None:
    """Blocking guard for safety thresholds"""
    if card.uncertainty.safety_certainty < min_safety:
        raise ValueError(f"Diagnosis blocked: safety certainty {card.uncertainty.safety_certainty} < {min_safety}")


# Factory function
def create_precision_critic() -> PrecisionCritic:
    """Create configured precision critic"""
    return PrecisionCritic()


# Example usage
if __name__ == "__main__":
    from app.core.types import DiagnosisCard, DxCandidate, Evidence, Uncertainty, TriageLevel
    from datetime import datetime

    # Test with a sample diagnosis card
    test_card = DiagnosisCard(
        patient_id="test_123",
        language="thai",
        triage={"level": TriageLevel.URGENT, "rationale": "Chest pain evaluation"},
        differential=[
            DxCandidate(
                icd10="I21.9",
                label="Acute MI",
                p=0.6,
                evidence=Evidence(
                    for_=["chest pain", "troponin elevated"],
                    citations=["guideline:aha_stemi_2022"]
                )
            )
        ],
        uncertainty=Uncertainty(
            diagnostic_coverage=0.8,
            safety_certainty=0.9,
            prediction_set_size=3
        ),
        overall_confidence=0.75,
        session_id="test_session"
    )

    critic = create_precision_critic()
    result = critic.validate_diagnosis_card(test_card)
    print(f"Validation result: {result}")