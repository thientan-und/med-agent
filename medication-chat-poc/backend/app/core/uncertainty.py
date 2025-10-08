# Uncertainty Quantification and Calibrated Abstention
# Implements prediction sets, calibration, and abstention logic

from typing import List, Dict, Tuple, Optional
import numpy as np
import logging
from scipy import stats
from app.core.types import DiagnosisCard, DxCandidate, Uncertainty, VOIQuestion

logger = logging.getLogger(__name__)


class UncertaintyQuantifier:
    """Calibrated uncertainty quantification for medical AI"""

    def __init__(self, coverage_target: float = 0.9):
        self.coverage_target = coverage_target
        self.calibration_data = {}  # Would be loaded from validation data
        self.abstention_threshold = 0.7

    def quantify_uncertainty(self,
                           differential: List[DxCandidate],
                           context: Dict,
                           temperature: float = 1.0) -> Uncertainty:
        """
        Quantify uncertainty for differential diagnosis

        Args:
            differential: List of diagnosis candidates
            context: Clinical context (symptoms, risk factors, etc.)
            temperature: Temperature scaling parameter

        Returns:
            Uncertainty object with calibrated metrics
        """
        if not differential:
            return Uncertainty(
                diagnostic_coverage=0.0,
                safety_certainty=0.0,
                abstention_reason="No differential diagnoses generated",
                prediction_set_size=0
            )

        # Temperature-scale probabilities
        scaled_probs = self._temperature_scale(differential, temperature)

        # Create prediction set for target coverage
        prediction_set_size, diagnostic_coverage = self._create_prediction_set(
            scaled_probs, self.coverage_target
        )

        # Calculate safety certainty
        safety_certainty = self._calculate_safety_certainty(
            differential, context
        )

        # Determine abstention
        abstention_reason = self._should_abstain(
            diagnostic_coverage, safety_certainty, context
        )

        logger.info(f"Uncertainty: coverage={diagnostic_coverage:.3f}, safety={safety_certainty:.3f}")

        return Uncertainty(
            diagnostic_coverage=diagnostic_coverage,
            safety_certainty=safety_certainty,
            abstention_reason=abstention_reason,
            prediction_set_size=prediction_set_size
        )

    def _temperature_scale(self, differential: List[DxCandidate], temperature: float) -> List[float]:
        """Apply temperature scaling to calibrate probabilities"""
        if temperature <= 0:
            temperature = 1.0

        # Extract probabilities and apply temperature scaling
        logits = [np.log(max(dx.p, 1e-8)) for dx in differential]
        scaled_logits = [logit / temperature for logit in logits]

        # Softmax normalization
        max_logit = max(scaled_logits)
        exp_logits = [np.exp(logit - max_logit) for logit in scaled_logits]
        sum_exp = sum(exp_logits)

        scaled_probs = [exp_logit / sum_exp for exp_logit in exp_logits]

        return scaled_probs

    def _create_prediction_set(self,
                             probabilities: List[float],
                             target_coverage: float) -> Tuple[int, float]:
        """
        Create prediction set for conformal prediction

        Args:
            probabilities: Calibrated probabilities
            target_coverage: Target coverage probability

        Returns:
            Tuple of (set_size, actual_coverage)
        """
        if not probabilities:
            return 0, 0.0

        # Sort probabilities in descending order
        sorted_probs = sorted(probabilities, reverse=True)

        # Build prediction set until target coverage
        cumulative_prob = 0.0
        set_size = 0

        for prob in sorted_probs:
            cumulative_prob += prob
            set_size += 1

            if cumulative_prob >= target_coverage:
                break

        # Actual coverage achieved
        actual_coverage = min(cumulative_prob, 1.0)

        return set_size, actual_coverage

    def _calculate_safety_certainty(self,
                                  differential: List[DxCandidate],
                                  context: Dict) -> float:
        """
        Calculate certainty that no critical conditions are missed

        Args:
            differential: Diagnosis candidates
            context: Clinical context

        Returns:
            Safety certainty score (0-1)
        """
        # Start with base safety score
        safety_score = 0.8

        # Check for emergency/critical conditions coverage
        critical_icds = [
            'I21',  # STEMI
            'I26',  # PE
            'G93',  # Brain disorders
            'R06.02',  # Respiratory failure
            'G00',  # Bacterial meningitis
        ]

        # Check if critical conditions are properly addressed
        has_critical_symptoms = self._has_critical_symptoms(context)
        critical_in_differential = any(
            any(dx.icd10.startswith(icd) for icd in critical_icds)
            for dx in differential
        )

        if has_critical_symptoms:
            if critical_in_differential:
                # Critical symptoms addressed in differential
                safety_score += 0.1
            else:
                # Critical symptoms not addressed - major safety concern
                safety_score -= 0.3

        # Adjust based on evidence quality
        evidence_quality = self._assess_evidence_quality(differential)
        safety_score += (evidence_quality - 0.5) * 0.2

        # Adjust based on differential completeness
        if len(differential) < 2:
            safety_score -= 0.1  # Single diagnosis risky

        # Adjust based on probability distribution
        if differential:
            top_prob = max(dx.p for dx in differential)
            if top_prob < 0.3:
                safety_score -= 0.15  # Very uncertain top diagnosis

        return max(0.0, min(1.0, safety_score))

    def _has_critical_symptoms(self, context: Dict) -> bool:
        """Check if context contains critical symptoms"""
        symptoms = context.get('symptoms', '').lower()
        critical_patterns = [
            'chest pain', 'ปวดหน้าอก',
            'shortness of breath', 'หายใจไม่ออก',
            'severe headache', 'ปวดหัวรุนแรง',
            'altered mental status', 'สติเปลี่ยนแปลง',
            'unconscious', 'หมดสติ'
        ]

        return any(pattern in symptoms for pattern in critical_patterns)

    def _assess_evidence_quality(self, differential: List[DxCandidate]) -> float:
        """Assess quality of evidence supporting diagnoses"""
        if not differential:
            return 0.0

        total_evidence_score = 0.0
        for dx in differential:
            # Score based on evidence items
            evidence_count = len(dx.evidence.for_) + len(dx.evidence.against)
            citation_count = len(dx.evidence.citations)

            dx_score = min(1.0, (evidence_count * 0.2) + (citation_count * 0.3))
            total_evidence_score += dx_score * dx.p  # Weight by probability

        return total_evidence_score

    def _should_abstain(self,
                       diagnostic_coverage: float,
                       safety_certainty: float,
                       context: Dict) -> Optional[str]:
        """
        Determine if system should abstain from diagnosis

        Args:
            diagnostic_coverage: Coverage probability
            safety_certainty: Safety certainty score
            context: Clinical context

        Returns:
            Abstention reason or None if should proceed
        """
        # Safety-based abstention
        if safety_certainty < 0.85:
            return f"Safety certainty too low ({safety_certainty:.2f} < 0.85)"

        # Coverage-based abstention
        if diagnostic_coverage < 0.6:
            return f"Diagnostic coverage too low ({diagnostic_coverage:.2f} < 0.6)"

        # Context-based abstention
        critical_symptoms = self._has_critical_symptoms(context)
        if critical_symptoms and diagnostic_coverage < 0.8:
            return "Critical symptoms with insufficient diagnostic certainty"

        # Evidence-based abstention
        if 'insufficient_evidence' in context.get('flags', []):
            return "Insufficient evidence for reliable diagnosis"

        return None

    def generate_voi_questions(self,
                             differential: List[DxCandidate],
                             context: Dict,
                             max_questions: int = 3) -> List[VOIQuestion]:
        """
        Generate value-of-information questions to reduce uncertainty

        Args:
            differential: Current differential diagnosis
            context: Clinical context
            max_questions: Maximum questions to generate

        Returns:
            List of high-value questions
        """
        questions = []

        if not differential:
            return questions

        # Calculate current top diagnosis uncertainty
        top_dx = differential[0]
        uncertainty_score = 1.0 - top_dx.p

        # Only generate questions if uncertainty is meaningful
        if uncertainty_score < 0.2:
            return questions

        # Generate questions based on top diagnoses
        for i, dx in enumerate(differential[:3]):
            question_candidates = self._generate_dx_questions(dx, context)
            questions.extend(question_candidates)

        # Score and rank questions by VOI
        scored_questions = []
        for q_text, category in question_candidates:
            voi_score = self._calculate_voi_score(
                q_text, category, differential, context
            )
            if voi_score > 0.15:  # Minimum VOI threshold
                question = VOIQuestion(
                    question=q_text,
                    voi_score=voi_score,
                    expected_delta_p=voi_score,  # Simplified
                    category=category
                )
                scored_questions.append(question)

        # Return top questions by VOI score
        scored_questions.sort(key=lambda q: q.voi_score, reverse=True)
        return scored_questions[:max_questions]

    def _generate_dx_questions(self,
                             dx: DxCandidate,
                             context: Dict) -> List[Tuple[str, str]]:
        """Generate questions specific to a diagnosis"""
        questions = []

        # Questions based on ICD category
        if dx.icd10.startswith('I2'):  # Ischemic heart disease
            questions.extend([
                ("Do you have crushing chest pain radiating to your arm or jaw?", "symptoms"),
                ("Have you had any previous heart problems?", "history"),
                ("Are you experiencing shortness of breath with the chest pain?", "symptoms")
            ])

        elif dx.icd10.startswith('G0'):  # CNS infections
            questions.extend([
                ("Do you have neck stiffness or pain when moving your neck?", "physical_exam"),
                ("Are you bothered by bright lights (photophobia)?", "symptoms"),
                ("Have you been confused or had changes in thinking?", "mental_status")
            ])

        elif dx.icd10.startswith('J'):  # Respiratory
            questions.extend([
                ("How many days have you had these symptoms?", "timeline"),
                ("Do you have a cough with colored sputum?", "symptoms"),
                ("Have you had fever with chills?", "symptoms")
            ])

        # Generic high-value questions
        if not questions:
            questions.extend([
                ("When did these symptoms first start?", "timeline"),
                ("Have you had similar symptoms before?", "history"),
                ("Are the symptoms getting better or worse?", "progression")
            ])

        return questions

    def _calculate_voi_score(self,
                           question: str,
                           category: str,
                           differential: List[DxCandidate],
                           context: Dict) -> float:
        """Calculate value of information for a question"""
        # Base VOI based on current uncertainty
        if not differential:
            return 0.0

        top_prob = differential[0].p
        uncertainty = 1.0 - top_prob

        # Higher VOI for more uncertain scenarios
        base_voi = uncertainty * 0.5

        # Category-specific adjustments
        category_weights = {
            "physical_exam": 0.8,  # High value
            "vitals": 0.9,         # Very high value
            "symptoms": 0.6,       # Medium value
            "history": 0.5,        # Lower value
            "timeline": 0.4        # Lowest value
        }

        category_weight = category_weights.get(category, 0.5)
        voi_score = base_voi * category_weight

        # Boost if question could distinguish between top diagnoses
        if len(differential) > 1:
            prob_gap = differential[0].p - differential[1].p
            if prob_gap < 0.3:  # Close competition
                voi_score *= 1.5

        return min(1.0, voi_score)

    def calibrate_from_validation_data(self, validation_results: List[Dict]):
        """
        Calibrate uncertainty from validation data

        Args:
            validation_results: List of {predicted_prob, actual_outcome, features}
        """
        # This would implement proper calibration using validation data
        # For now, store basic statistics
        if validation_results:
            probs = [r.get('predicted_prob', 0.5) for r in validation_results]
            outcomes = [r.get('actual_outcome', 0) for r in validation_results]

            # Simple reliability diagram data
            self.calibration_data = {
                'mean_predicted': np.mean(probs),
                'mean_actual': np.mean(outcomes),
                'brier_score': np.mean([(p - o)**2 for p, o in zip(probs, outcomes)])
            }

            logger.info(f"Calibration data updated: {self.calibration_data}")


class AbstractionEngine:
    """Engine for making abstention decisions"""

    def __init__(self):
        self.abstention_rules = self._build_abstention_rules()

    def _build_abstention_rules(self) -> Dict:
        """Build abstention rules"""
        return {
            "low_confidence": {
                "condition": lambda uncertainty: uncertainty.diagnostic_coverage < 0.6,
                "action": "request_more_info",
                "message": "ข้อมูลไม่เพียงพอสำหรับการวินิจฉัย กรุณาให้ข้อมูลเพิ่มเติม"
            },
            "safety_concern": {
                "condition": lambda uncertainty: uncertainty.safety_certainty < 0.85,
                "action": "escalate_to_physician",
                "message": "ควรปรึกษาแพทย์เพื่อการประเมินที่ละเอียดมากขึ้น"
            },
            "high_uncertainty": {
                "condition": lambda uncertainty: (
                    uncertainty.prediction_set_size > 4 and
                    uncertainty.diagnostic_coverage < 0.8
                ),
                "action": "request_additional_tests",
                "message": "จำเป็นต้องทำการตรวจเพิ่มเติมเพื่อการวินิจฉัยที่แม่นยำ"
            }
        }

    def should_abstain(self, uncertainty: Uncertainty) -> Tuple[bool, str, str]:
        """
        Determine if should abstain and what action to take

        Returns:
            Tuple of (should_abstain, action, message)
        """
        for rule_name, rule_config in self.abstention_rules.items():
            if rule_config["condition"](uncertainty):
                return True, rule_config["action"], rule_config["message"]

        return False, "proceed", "การวินิจฉัยพร้อมใช้งาน"


# Factory functions
def create_uncertainty_quantifier(coverage_target: float = 0.9) -> UncertaintyQuantifier:
    """Create configured uncertainty quantifier"""
    return UncertaintyQuantifier(coverage_target)


def create_abstention_engine() -> AbstractionEngine:
    """Create configured abstention engine"""
    return AbstractionEngine()


# Example usage
if __name__ == "__main__":
    from app.core.types import DxCandidate, Evidence

    # Test uncertainty quantification
    quantifier = create_uncertainty_quantifier()

    test_differential = [
        DxCandidate(
            icd10="J00",
            label="Common Cold",
            p=0.6,
            evidence=Evidence(for_=["runny nose", "mild fever"], citations=["kb:common_cold"])
        ),
        DxCandidate(
            icd10="J11.1",
            label="Influenza",
            p=0.3,
            evidence=Evidence(for_=["fever", "body aches"], citations=["kb:influenza"])
        )
    ]

    context = {"symptoms": "runny nose, mild fever"}

    uncertainty = quantifier.quantify_uncertainty(test_differential, context)
    print(f"Uncertainty result: {uncertainty}")

    # Test VOI questions
    questions = quantifier.generate_voi_questions(test_differential, context)
    print(f"VOI questions: {[q.question for q in questions]}")

    # Test abstention
    abstention_engine = create_abstention_engine()
    should_abstain, action, message = abstention_engine.should_abstain(uncertainty)
    print(f"Abstention: {should_abstain}, Action: {action}, Message: {message}")