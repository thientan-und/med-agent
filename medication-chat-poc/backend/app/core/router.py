# Evidence-First Precision Router
# Routes based on clinical signals, not generic "run everything"

from typing import List, Dict, Tuple, Set
import logging
from app.core.types import RouteSignals, RoutingReason, PrecisionPlan

logger = logging.getLogger(__name__)


class PrecisionRouter:
    """Evidence-first router for medical AI precision"""

    def __init__(self):
        self.routing_rules = self._build_routing_rules()

    def _build_routing_rules(self) -> Dict[RoutingReason, Dict]:
        """Define routing rules with clinical rationale"""
        return {
            RoutingReason.CHEST_PAIN_RISK: {
                "triggers": ["chest_pain"],
                "tools": ["heart_score", "perc_rule", "chest_pain_guidelines"],
                "parallel_safe": True,
                "rationale": "Chest pain requires cardiac risk stratification"
            },
            RoutingReason.FEVER_HEADACHE_REDFLAGS: {
                "triggers": ["fever", "severe_headache"],
                "tools": ["meningitis_redflags", "neuro_guidelines"],
                "parallel_safe": False,  # Sequential for safety
                "rationale": "Fever + headache requires meningitis screening"
            },
            RoutingReason.NEURO_DEFICIT: {
                "triggers": ["neurological_deficit"],
                "tools": ["stroke_scale", "neuro_emergency_protocol"],
                "parallel_safe": False,
                "rationale": "Neurological deficits require stroke evaluation"
            },
            RoutingReason.RESPIRATORY_DISTRESS: {
                "triggers": ["breathing_difficulty"],
                "tools": ["pe_wells_score", "respiratory_guidelines"],
                "parallel_safe": True,
                "rationale": "Breathing difficulty requires PE/respiratory assessment"
            },
            RoutingReason.EMERGENCY_KEYWORDS: {
                "triggers": ["emergency_keywords"],
                "tools": ["red_flag_detector", "emergency_protocols"],
                "parallel_safe": False,
                "rationale": "Emergency keywords trigger immediate assessment"
            },
            RoutingReason.BASIC_SYMPTOMS: {
                "triggers": [],  # Default fallback
                "tools": ["conservative_diagnosis", "common_illness_guidelines"],
                "parallel_safe": True,
                "rationale": "Standard symptom evaluation"
            }
        }

    def route_tools(self, signals: RouteSignals) -> Tuple[List[str], List[RoutingReason]]:
        """
        Route to appropriate tools based on clinical signals

        Returns:
            Tuple of (tool_names, routing_reasons)
        """
        tools = []
        reasons = []

        # Emergency keywords get highest priority
        if signals.emergency_keywords:
            tools.extend(self.routing_rules[RoutingReason.EMERGENCY_KEYWORDS]["tools"])
            reasons.append(RoutingReason.EMERGENCY_KEYWORDS)
            logger.warning(f"🚨 Emergency routing triggered: {signals.emergency_keywords}")
            return tools, reasons

        # Check specific clinical patterns
        if signals.chest_pain:
            tools.extend(self.routing_rules[RoutingReason.CHEST_PAIN_RISK]["tools"])
            reasons.append(RoutingReason.CHEST_PAIN_RISK)
            logger.info("🫀 Chest pain routing: cardiac risk stratification")

        if signals.fever and signals.severe_headache:
            tools.extend(self.routing_rules[RoutingReason.FEVER_HEADACHE_REDFLAGS]["tools"])
            reasons.append(RoutingReason.FEVER_HEADACHE_REDFLAGS)
            logger.warning("🧠 Fever + headache routing: meningitis screening")

        if signals.neurological_deficit:
            tools.extend(self.routing_rules[RoutingReason.NEURO_DEFICIT]["tools"])
            reasons.append(RoutingReason.NEURO_DEFICIT)
            logger.warning("🧠 Neurological deficit routing: stroke protocol")

        if signals.breathing_difficulty:
            tools.extend(self.routing_rules[RoutingReason.RESPIRATORY_DISTRESS]["tools"])
            reasons.append(RoutingReason.RESPIRATORY_DISTRESS)
            logger.info("🫁 Respiratory distress routing: PE assessment")

        # If no specific patterns, use basic symptom evaluation
        if not tools:
            tools.extend(self.routing_rules[RoutingReason.BASIC_SYMPTOMS]["tools"])
            reasons.append(RoutingReason.BASIC_SYMPTOMS)
            logger.info("📋 Basic symptom routing: conservative evaluation")

        return tools, reasons

    def create_execution_plan(self, signals: RouteSignals, max_questions: int = 3) -> PrecisionPlan:
        """Create precise execution plan with success criteria"""
        tools, reasons = self.route_tools(signals)

        # Define execution steps based on routing
        steps = []
        success_criteria = {}

        # Always start with signal validation
        steps.append("validate_signals")
        success_criteria["validate_signals"] = "RouteSignals object validates and contains expected fields"

        # Emergency routing overrides everything
        if RoutingReason.EMERGENCY_KEYWORDS in reasons:
            steps.extend(["emergency_triage", "red_flag_assessment", "escalation_protocol"])
            success_criteria.update({
                "emergency_triage": "TriageLevel.RESUSCITATION or TriageLevel.EMERGENCY assigned",
                "red_flag_assessment": "Red flags identified and documented",
                "escalation_protocol": "Emergency escalation triggered"
            })
        else:
            # Standard precision flow
            steps.extend([
                "extract_evidence",
                "apply_calculators",
                "guideline_rag",
                "differential_diagnosis",
                "precision_critic",
                "uncertainty_quantification"
            ])
            success_criteria.update({
                "extract_evidence": "Evidence object with ≥1 for/against items",
                "apply_calculators": "Calculator results with confidence ≥0.8",
                "guideline_rag": "≥1 guideline citation retrieved",
                "differential_diagnosis": "≥1 DxCandidate with probability ≥0.3",
                "precision_critic": "All critic rules pass",
                "uncertainty_quantification": "Safety_certainty ≥0.85"
            })

            # Add VOI questioning if needed
            if self._needs_voi_questions(signals):
                steps.insert(-2, "voi_questioning")  # Before critic
                success_criteria["voi_questioning"] = f"≤{max_questions} questions with VOI ≥0.15"

        return PrecisionPlan(
            steps=steps,
            success_criteria=success_criteria,
            routing_reasons=reasons,
            max_questions=max_questions,
            abstention_threshold=0.7
        )

    def _needs_voi_questions(self, signals: RouteSignals) -> bool:
        """Determine if VOI questioning is needed"""
        # Need VOI if we have ambiguous signals
        ambiguous_patterns = [
            signals.chest_pain,  # Chest pain (vitals removed)
            signals.fever and signals.severe_headache and not signals.neurological_deficit,  # Unclear neuro
            signals.abdominal_pain,  # Abdominal pain (vitals removed)
        ]
        return any(ambiguous_patterns)

    def get_parallel_tools(self, tools: List[str]) -> Tuple[List[str], List[str]]:
        """
        Separate tools into parallel-safe and sequential groups

        Returns:
            Tuple of (parallel_tools, sequential_tools)
        """
        parallel = []
        sequential = []

        for tool in tools:
            # Safe tools that can run in parallel
            if tool in [
                "heart_score", "perc_rule", "pe_wells_score",
                "conservative_diagnosis", "common_illness_guidelines",
                "chest_pain_guidelines", "respiratory_guidelines"
            ]:
                parallel.append(tool)
            else:
                # Safety-critical tools run sequentially
                sequential.append(tool)

        return parallel, sequential

    def validate_routing_decision(self, signals: RouteSignals, tools: List[str], reasons: List[RoutingReason]) -> bool:
        """Validate that routing decision makes clinical sense"""
        validation_rules = [
            # Emergency keywords must trigger emergency tools
            (signals.emergency_keywords and "red_flag_detector" not in tools,
             "Emergency keywords present but no red flag detection"),

            # Chest pain must trigger cardiac assessment
            (signals.chest_pain and not any(tool in tools for tool in ["heart_score", "chest_pain_guidelines"]),
             "Chest pain present but no cardiac assessment"),

            # Neurological symptoms must trigger neuro assessment
            (signals.neurological_deficit and "stroke_scale" not in tools,
             "Neurological deficit present but no stroke assessment"),
        ]

        for condition, error_msg in validation_rules:
            if condition:
                logger.error(f"Routing validation failed: {error_msg}")
                return False

        return True

    def get_routing_rationale(self, reasons: List[RoutingReason]) -> str:
        """Get human-readable rationale for routing decisions"""
        rationales = []
        for reason in reasons:
            if reason in self.routing_rules:
                rationales.append(self.routing_rules[reason]["rationale"])

        return "; ".join(rationales) if rationales else "Standard symptom evaluation"


# Routing utility functions
def extract_signals_from_text(symptoms: str, patient_data: Dict = None) -> RouteSignals:
    """Extract routing signals from symptom text and patient data"""
    signals = RouteSignals.from_symptoms(symptoms)

    # Enhance with patient data if available
    if patient_data:
        # Note: vital_signs enhancement removed - inappropriate for consultation scope

        # Add logic for other patient data enhancement
        history = patient_data.get("history", [])
        if any("cardiac" in h.lower() for h in history):
            signals.chest_pain = True  # Increase sensitivity for cardiac patients

    return signals


def create_router() -> PrecisionRouter:
    """Factory function to create configured router"""
    return PrecisionRouter()


# Example usage and testing
if __name__ == "__main__":
    # Test routing scenarios
    router = create_router()

    # Test emergency scenario
    emergency_signals = RouteSignals(
        chest_pain=True,
        breathing_difficulty=True,
        emergency_keywords=["urgent", "severe"]
    )
    tools, reasons = router.route_tools(emergency_signals)
    print(f"Emergency routing: {tools} | Reasons: {reasons}")

    # Test basic scenario
    basic_signals = RouteSignals(fever=True)
    tools, reasons = router.route_tools(basic_signals)
    print(f"Basic routing: {tools} | Reasons: {reasons}")

    # Test execution plan
    plan = router.create_execution_plan(emergency_signals)
    print(f"Execution plan: {plan.steps}")