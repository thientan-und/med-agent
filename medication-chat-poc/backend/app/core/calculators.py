# Strict Medical Calculators with Validation
# Schema-enforced calculators with proper I/O contracts

from pydantic import BaseModel, Field, validator, confloat
from typing import Dict, Union, Optional, List
from enum import Enum
import logging
from app.core.types import Calculator

logger = logging.getLogger(__name__)


# HEART Score Calculator
class HeartScoreInput(BaseModel):
    """HEART Score input validation"""
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    history: bool = Field(..., description="History of CAD, MI, or revascularization")
    ecg: bool = Field(..., description="ECG abnormalities")
    risk_factors: int = Field(..., ge=0, le=5, description="Number of risk factors")
    troponin_elevated: bool = Field(..., description="Troponin elevated (>99th percentile)")

    @validator('age')
    def validate_age(cls, v):
        if v < 16:
            logger.warning(f"HEART score not validated for age < 16: {v}")
        return v


class HeartScoreOutput(BaseModel):
    """HEART Score output"""
    score: int = Field(..., ge=0, le=10, description="HEART score (0-10)")
    risk_band: str = Field(..., description="Risk stratification")
    recommendation: str = Field(..., description="Clinical recommendation")
    inputs_used: HeartScoreInput = Field(..., description="Inputs used in calculation")
    reference: str = Field(default="guideline:esc_chest_pain_2020", description="Guideline reference")
    confidence: confloat(ge=0.0, le=1.0) = Field(..., description="Confidence in input completeness")


def heart_score(inputs: HeartScoreInput, input_confidence: float = 1.0) -> HeartScoreOutput:
    """
    Calculate HEART Score for chest pain risk stratification

    HEART Score Components:
    - History: 0=low suspicion, 1=moderate, 2=high suspicion
    - ECG: 0=normal, 1=non-specific, 2=significant ST depression
    - Age: 0=<45, 1=45-64, 2=≥65
    - Risk factors: 0=none, 1=1-2 factors, 2=≥3 factors
    - Troponin: 0=normal, 1=1-3x upper limit, 2=>3x upper limit
    """
    score = 0

    # Age scoring
    if inputs.age < 45:
        age_score = 0
    elif inputs.age < 65:
        age_score = 1
    else:
        age_score = 2
    score += age_score

    # History (simplified binary)
    history_score = 2 if inputs.history else 0
    score += history_score

    # ECG (simplified binary)
    ecg_score = 2 if inputs.ecg else 0
    score += ecg_score

    # Risk factors
    if inputs.risk_factors == 0:
        rf_score = 0
    elif inputs.risk_factors <= 2:
        rf_score = 1
    else:
        rf_score = 2
    score += rf_score

    # Troponin (simplified binary)
    troponin_score = 2 if inputs.troponin_elevated else 0
    score += troponin_score

    # Risk stratification
    if score <= 3:
        risk_band = "Low Risk"
        recommendation = "Discharge with outpatient follow-up"
    elif score <= 6:
        risk_band = "Moderate Risk"
        recommendation = "Observe 6-12 hours, serial troponins"
    else:
        risk_band = "High Risk"
        recommendation = "Urgent cardiology consultation, consider catheterization"

    logger.info(f"HEART Score calculated: {score} ({risk_band})")

    return HeartScoreOutput(
        score=score,
        risk_band=risk_band,
        recommendation=recommendation,
        inputs_used=inputs,
        confidence=input_confidence
    )


# PERC Rule Calculator
class PERCRuleInput(BaseModel):
    """PERC Rule input validation"""
    age_ge_50: bool = Field(..., description="Age ≥ 50 years")
    hr_ge_100: bool = Field(..., description="Heart rate ≥ 100 bpm")
    o2_sat_lt_95: bool = Field(..., description="O2 saturation < 95%")
    unilateral_leg_swelling: bool = Field(..., description="Unilateral leg swelling")
    hemoptysis: bool = Field(..., description="Hemoptysis")
    recent_surgery: bool = Field(..., description="Surgery or trauma in past 4 weeks")
    pe_dvt_history: bool = Field(..., description="Prior PE or DVT")
    estrogen_use: bool = Field(..., description="Estrogen use")


class PERCRuleOutput(BaseModel):
    """PERC Rule output"""
    perc_negative: bool = Field(..., description="PERC rule negative (all criteria absent)")
    criteria_positive: List[str] = Field(..., description="Positive criteria")
    recommendation: str = Field(..., description="Clinical recommendation")
    inputs_used: PERCRuleInput = Field(..., description="Inputs used")
    reference: str = Field(default="guideline:accp_pe_2012", description="Guideline reference")
    confidence: confloat(ge=0.0, le=1.0) = Field(..., description="Confidence in inputs")


def perc_rule(inputs: PERCRuleInput, input_confidence: float = 1.0) -> PERCRuleOutput:
    """
    Apply PERC Rule for pulmonary embolism

    PERC Rule: If all 8 criteria are absent AND clinical suspicion is low,
    PE can be ruled out without further testing
    """
    criteria = [
        (inputs.age_ge_50, "Age ≥ 50 years"),
        (inputs.hr_ge_100, "Heart rate ≥ 100 bpm"),
        (inputs.o2_sat_lt_95, "O2 saturation < 95%"),
        (inputs.unilateral_leg_swelling, "Unilateral leg swelling"),
        (inputs.hemoptysis, "Hemoptysis"),
        (inputs.recent_surgery, "Recent surgery/trauma"),
        (inputs.pe_dvt_history, "Prior PE/DVT"),
        (inputs.estrogen_use, "Estrogen use")
    ]

    positive_criteria = [desc for present, desc in criteria if present]
    perc_negative = len(positive_criteria) == 0

    if perc_negative:
        recommendation = "PERC negative: PE ruled out, no further testing needed"
    else:
        recommendation = f"PERC positive ({len(positive_criteria)} criteria): Consider D-dimer or imaging"

    logger.info(f"PERC Rule: {'Negative' if perc_negative else 'Positive'} - {len(positive_criteria)} criteria")

    return PERCRuleOutput(
        perc_negative=perc_negative,
        criteria_positive=positive_criteria,
        recommendation=recommendation,
        inputs_used=inputs,
        confidence=input_confidence
    )


# Wells Score for PE
class WellsPEInput(BaseModel):
    """Wells Score for PE input validation"""
    clinical_signs_dvt: bool = Field(..., description="Clinical signs of DVT")
    pe_likely_as_alternative: bool = Field(..., description="PE as likely as alternative diagnosis")
    heart_rate_gt_100: bool = Field(..., description="Heart rate > 100 bpm")
    immobilization_surgery: bool = Field(..., description="Immobilization ≥3 days or surgery in past 4 weeks")
    previous_pe_dvt: bool = Field(..., description="Previous PE or DVT")
    hemoptysis: bool = Field(..., description="Hemoptysis")
    malignancy: bool = Field(..., description="Active malignancy")


class WellsPEOutput(BaseModel):
    """Wells Score for PE output"""
    score: float = Field(..., description="Wells score")
    probability: str = Field(..., description="PE probability")
    recommendation: str = Field(..., description="Clinical recommendation")
    inputs_used: WellsPEInput = Field(..., description="Inputs used")
    reference: str = Field(default="guideline:wells_pe_2000", description="Guideline reference")
    confidence: confloat(ge=0.0, le=1.0) = Field(..., description="Confidence in inputs")


def wells_pe_score(inputs: WellsPEInput, input_confidence: float = 1.0) -> WellsPEOutput:
    """Calculate Wells Score for PE probability"""
    score = 0.0

    # Score components
    if inputs.clinical_signs_dvt:
        score += 3.0
    if inputs.pe_likely_as_alternative:
        score += 3.0
    if inputs.heart_rate_gt_100:
        score += 1.5
    if inputs.immobilization_surgery:
        score += 1.5
    if inputs.previous_pe_dvt:
        score += 1.5
    if inputs.hemoptysis:
        score += 1.0
    if inputs.malignancy:
        score += 1.0

    # Probability assessment
    if score <= 4:
        probability = "Low Probability"
        recommendation = "D-dimer; if negative, PE ruled out"
    elif score <= 6:
        probability = "Moderate Probability"
        recommendation = "D-dimer or CTPA; if D-dimer positive, proceed to imaging"
    else:
        probability = "High Probability"
        recommendation = "CTPA or V/Q scan recommended"

    logger.info(f"Wells PE Score: {score} ({probability})")

    return WellsPEOutput(
        score=score,
        probability=probability,
        recommendation=recommendation,
        inputs_used=inputs,
        confidence=input_confidence
    )


# Calculator Registry and Validation
class CalculatorRegistry:
    """Registry for all medical calculators with validation"""

    def __init__(self):
        self.calculators = {
            "heart_score": {
                "function": heart_score,
                "input_model": HeartScoreInput,
                "output_model": HeartScoreOutput,
                "required_fields": ["age", "history", "ecg", "risk_factors", "troponin_elevated"],
                "description": "HEART Score for chest pain risk stratification"
            },
            "perc_rule": {
                "function": perc_rule,
                "input_model": PERCRuleInput,
                "output_model": PERCRuleOutput,
                "required_fields": [
                    "age_ge_50", "hr_ge_100", "o2_sat_lt_95", "unilateral_leg_swelling",
                    "hemoptysis", "recent_surgery", "pe_dvt_history", "estrogen_use"
                ],
                "description": "PERC Rule for PE exclusion"
            },
            "wells_pe": {
                "function": wells_pe_score,
                "input_model": WellsPEInput,
                "output_model": WellsPEOutput,
                "required_fields": [
                    "clinical_signs_dvt", "pe_likely_as_alternative", "heart_rate_gt_100",
                    "immobilization_surgery", "previous_pe_dvt", "hemoptysis", "malignancy"
                ],
                "description": "Wells Score for PE probability"
            }
        }

    def get_calculator(self, name: str):
        """Get calculator by name with validation"""
        if name not in self.calculators:
            raise ValueError(f"Calculator '{name}' not found. Available: {list(self.calculators.keys())}")
        return self.calculators[name]

    def calculate(self, name: str, inputs: Dict, captured_fields: Dict) -> Calculator:
        """
        Execute calculator with strict validation

        Args:
            name: Calculator name
            inputs: Input data dictionary
            captured_fields: Fields actually captured from patient

        Returns:
            Calculator result with confidence scoring
        """
        calc_config = self.get_calculator(name)

        # Validate inputs against schema
        try:
            validated_inputs = calc_config["input_model"](**inputs)
        except Exception as e:
            raise ValueError(f"Calculator {name} input validation failed: {e}")

        # Check if all required fields were captured
        required_fields = set(calc_config["required_fields"])
        captured_field_names = set(captured_fields.keys())
        missing_fields = required_fields - captured_field_names

        # Calculate confidence based on field completeness
        field_confidence = 1.0 - (len(missing_fields) / len(required_fields))

        if missing_fields:
            logger.warning(f"Calculator {name} missing fields: {missing_fields}")

        # Execute calculator
        result = calc_config["function"](validated_inputs, field_confidence)

        # Convert to standardized Calculator format
        return Calculator(
            name=name,
            inputs_used=validated_inputs.dict(),
            score=result.score if hasattr(result, 'score') else 0,
            risk_band=result.risk_band if hasattr(result, 'risk_band') else result.probability,
            reference=result.reference,
            confidence=result.confidence
        )

    def validate_calculator_call(self, name: str, captured_fields: Dict) -> bool:
        """Validate that a calculator can be called with available data"""
        calc_config = self.get_calculator(name)
        required_fields = set(calc_config["required_fields"])
        captured_field_names = set(captured_fields.keys())

        # Require at least 80% of fields to be captured
        completeness = len(captured_field_names & required_fields) / len(required_fields)
        return completeness >= 0.8


# Global calculator registry
calculator_registry = CalculatorRegistry()


# Utility functions
def get_applicable_calculators(signals: "RouteSignals", captured_fields: Dict) -> List[str]:
    """Get list of calculators applicable to current signals and data"""
    applicable = []

    if signals.chest_pain:
        if calculator_registry.validate_calculator_call("heart_score", captured_fields):
            applicable.append("heart_score")

    if signals.breathing_difficulty:
        if calculator_registry.validate_calculator_call("perc_rule", captured_fields):
            applicable.append("perc_rule")
        if calculator_registry.validate_calculator_call("wells_pe", captured_fields):
            applicable.append("wells_pe")

    return applicable


def create_mock_patient_data() -> Dict:
    """Create mock patient data for testing"""
    return {
        "age": 55,
        "history": False,
        "ecg": True,
        "risk_factors": 2,
        "troponin_elevated": False,
        "heart_rate": 110,
        "oxygen_saturation": 96
    }


# Example usage
if __name__ == "__main__":
    # Test HEART Score
    registry = CalculatorRegistry()

    test_inputs = {
        "age": 55,
        "history": False,
        "ecg": True,
        "risk_factors": 2,
        "troponin_elevated": False
    }

    captured_fields = test_inputs  # In real use, this would be from patient data

    try:
        result = registry.calculate("heart_score", test_inputs, captured_fields)
        print(f"HEART Score Result: {result}")
    except Exception as e:
        print(f"Calculator error: {e}")