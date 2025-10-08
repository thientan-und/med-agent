#!/usr/bin/env python3
"""
RAG-Driven Scenario Generation for Few-Shot Learning
===================================================

This service creates dynamic few-shot learning scenarios by leveraging
RAG knowledge base to generate contextually relevant training examples
for the AI model, making it more adaptive and knowledge-driven.

Key Features:
1. Dynamic Scenario Creation: Builds realistic medical scenarios from knowledge base
2. Contextual Prompt Generation: Creates AI training prompts with proper examples
3. Safety-Validated Scenarios: Ensures all generated scenarios follow safety guidelines
4. Multi-Modal Learning: Supports different learning patterns and complexity levels
"""

import asyncio
import json
import logging
import random
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from app.services.rag_few_shot_service import rag_few_shot_service, KnowledgeItem
from app.services.advanced_few_shot import MedicalDomain
from app.util.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ScenarioComplexity(Enum):
    """Complexity levels for generated scenarios"""
    SIMPLE = "simple"          # Single symptom, clear diagnosis
    MODERATE = "moderate"      # Multiple symptoms, differential diagnosis
    COMPLEX = "complex"        # Complex presentation, multiple systems
    EMERGENCY = "emergency"    # Time-critical scenarios


class ScenarioType(Enum):
    """Types of medical scenarios"""
    DIAGNOSTIC = "diagnostic"              # Focus on diagnosis
    TREATMENT = "treatment"                # Focus on treatment planning
    TRIAGE = "triage"                     # Focus on urgency assessment
    SAFETY_CHECK = "safety_check"         # Focus on safety validation
    DIFFERENTIAL = "differential"          # Focus on differential diagnosis


@dataclass
class PatientProfile:
    """Generated patient profile for scenario"""
    age: int
    gender: str
    occupation: Optional[str]
    medical_history: List[str]
    risk_factors: List[str]
    current_medications: List[str]


@dataclass
class GeneratedScenario:
    """Complete medical scenario generated from RAG knowledge"""
    id: str
    scenario_type: ScenarioType
    complexity: ScenarioComplexity
    patient_profile: PatientProfile
    presenting_symptoms: Dict[str, str]  # Thai and English
    clinical_context: str
    expected_diagnosis: Dict[str, Any]
    expected_treatment: Dict[str, Any]
    learning_objectives: List[str]
    safety_considerations: List[str]
    few_shot_prompt: str
    knowledge_sources: List[str]
    confidence_target: float


class RAGScenarioGenerator:
    """Generate dynamic few-shot learning scenarios from RAG knowledge"""

    def __init__(self):
        self.initialized = False
        self.knowledge_base: List[KnowledgeItem] = []
        self.scenario_templates = {}
        self.patient_profiles = []

        # Scenario generation parameters
        self.generation_config = {
            "max_scenarios_per_condition": 3,
            "symptom_variation_rate": 0.3,
            "complexity_distribution": {
                ScenarioComplexity.SIMPLE: 0.4,
                ScenarioComplexity.MODERATE: 0.3,
                ScenarioComplexity.COMPLEX: 0.2,
                ScenarioComplexity.EMERGENCY: 0.1
            },
            "safety_validation_threshold": 0.8
        }

        logger.info("🎭 RAG Scenario Generator initialized")

    async def initialize(self):
        """Initialize scenario generator with RAG knowledge base"""
        if self.initialized:
            return

        logger.info("🔄 Initializing RAG Scenario Generator...")

        try:
            # Initialize RAG service
            await rag_few_shot_service.initialize()
            self.knowledge_base = rag_few_shot_service.knowledge_base

            # Build scenario templates
            await self._build_scenario_templates()

            # Generate patient profiles
            await self._generate_patient_profiles()

            self.initialized = True
            logger.info(f"✅ RAG Scenario Generator initialized with {len(self.knowledge_base)} knowledge items")

        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG Scenario Generator: {e}")
            raise

    async def generate_few_shot_scenarios(self,
                                        target_condition: Optional[str] = None,
                                        scenario_type: ScenarioType = ScenarioType.DIAGNOSTIC,
                                        count: int = 5) -> List[GeneratedScenario]:
        """Generate few-shot learning scenarios from RAG knowledge"""

        if not self.initialized:
            await self.initialize()

        logger.info(f"🎭 Generating {count} few-shot scenarios for {target_condition or 'mixed conditions'}")

        scenarios = []

        # Select knowledge items to use as basis
        if target_condition:
            knowledge_items = [item for item in self.knowledge_base
                             if target_condition.lower() in item.name_en.lower() or
                                target_condition.lower() in item.name_th.lower()]
        else:
            # Select diverse conditions
            knowledge_items = random.sample(self.knowledge_base, min(count * 2, len(self.knowledge_base)))

        # Generate scenarios
        for i in range(count):
            if i < len(knowledge_items):
                knowledge_item = knowledge_items[i]
            else:
                knowledge_item = random.choice(self.knowledge_base)

            scenario = await self._generate_single_scenario(
                knowledge_item=knowledge_item,
                scenario_type=scenario_type,
                scenario_id=f"rag_scenario_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            if scenario:
                scenarios.append(scenario)

        logger.info(f"✅ Generated {len(scenarios)} scenarios successfully")
        return scenarios

    async def create_ai_training_prompt(self,
                                      scenarios: List[GeneratedScenario],
                                      target_learning: str = "diagnostic reasoning") -> str:
        """Create a complete AI training prompt using generated scenarios"""

        logger.info(f"📝 Creating AI training prompt for '{target_learning}' with {len(scenarios)} scenarios")

        # Build comprehensive prompt
        prompt_sections = []

        # Header
        prompt_sections.append(f"""
# Medical AI Training: {target_learning.title()}
## Few-Shot Learning with RAG-Generated Scenarios

You are a medical AI assistant trained to provide safe, evidence-based medical consultations.
Learn from these carefully curated scenarios generated from verified medical knowledge.

### Learning Objectives:
1. Pattern recognition from real medical knowledge base
2. Conservative diagnosis approach prioritizing patient safety
3. Proper confidence calibration and uncertainty quantification
4. Evidence-based reasoning with clear rationale

### Safety Guidelines:
- Always prioritize common conditions over rare diseases
- Require strong evidence for serious diagnoses
- Maintain appropriate confidence levels (avoid overconfidence)
- Escalate to healthcare professionals when uncertain
""")

        # Add scenarios as few-shot examples
        for i, scenario in enumerate(scenarios, 1):
            prompt_sections.append(f"""
## Example {i}: {scenario.expected_diagnosis.get('name', 'Medical Case')}
**Complexity**: {scenario.complexity.value.title()}
**Learning Focus**: {scenario.scenario_type.value.replace('_', ' ').title()}

### Patient Profile:
- Age: {scenario.patient_profile.age}, Gender: {scenario.patient_profile.gender}
- Medical History: {', '.join(scenario.patient_profile.medical_history) if scenario.patient_profile.medical_history else 'None significant'}
- Risk Factors: {', '.join(scenario.patient_profile.risk_factors) if scenario.patient_profile.risk_factors else 'None identified'}

### Presenting Symptoms:
**Thai**: {scenario.presenting_symptoms.get('thai', 'ไม่ระบุ')}
**English**: {scenario.presenting_symptoms.get('english', 'Not specified')}

### Clinical Context:
{scenario.clinical_context}

### Diagnostic Reasoning:
1. **Primary Assessment**: {scenario.expected_diagnosis.get('name', 'Unknown')}
2. **Confidence Level**: {scenario.confidence_target:.0%}
3. **ICD Code**: {scenario.expected_diagnosis.get('icd_code', 'Not specified')}
4. **Urgency**: {scenario.expected_diagnosis.get('urgency', 'Standard')}

### Treatment Plan:
{self._format_treatment_plan(scenario.expected_treatment)}

### Learning Points:
{chr(10).join(f"- {obj}" for obj in scenario.learning_objectives)}

### Safety Considerations:
{chr(10).join(f"⚠️ {safety}" for safety in scenario.safety_considerations)}

### Knowledge Sources:
Generated from: {', '.join(scenario.knowledge_sources)}

---
""")

        # Add application template
        prompt_sections.append("""
## Now Apply This Learning:

When presented with a new medical case, follow this pattern:

1. **Patient Assessment**: Analyze age, gender, history, and risk factors
2. **Symptom Analysis**: Identify key clinical indicators in both languages
3. **Differential Consideration**: Consider multiple possibilities, starting with common conditions
4. **Evidence Evaluation**: Assess strength of evidence for each possibility
5. **Conservative Diagnosis**: Select most likely diagnosis with appropriate confidence
6. **Safety Check**: Verify no red flags are missed, escalate if uncertain
7. **Treatment Planning**: Provide evidence-based recommendations
8. **Follow-up**: Specify monitoring and next steps

### Response Format:
```
## Assessment
[Patient profile analysis]

## Primary Diagnosis
- Condition: [Thai name] ([English name])
- ICD Code: [Code]
- Confidence: [XX%]
- Urgency: [Low/Medium/High/Emergency]

## Reasoning
[Clinical reasoning with evidence]

## Treatment Plan
[Medications, recommendations, follow-up]

## Safety Notes
[Important warnings or escalation criteria]
```

Remember: When in doubt, prioritize patient safety and recommend professional medical consultation.
""")

        final_prompt = "\n".join(prompt_sections)

        logger.info(f"📝 Created comprehensive training prompt ({len(final_prompt)} characters)")
        return final_prompt

    async def _generate_single_scenario(self,
                                      knowledge_item: KnowledgeItem,
                                      scenario_type: ScenarioType,
                                      scenario_id: str) -> Optional[GeneratedScenario]:
        """Generate a single medical scenario from knowledge item"""

        try:
            # Determine complexity based on condition
            complexity = self._determine_scenario_complexity(knowledge_item)

            # Generate patient profile
            patient_profile = await self._generate_patient_profile(knowledge_item, complexity)

            # Generate presenting symptoms
            presenting_symptoms = await self._generate_presenting_symptoms(
                knowledge_item, complexity, patient_profile
            )

            # Create clinical context
            clinical_context = await self._generate_clinical_context(
                knowledge_item, patient_profile, presenting_symptoms
            )

            # Build expected diagnosis
            expected_diagnosis = {
                "icd_code": knowledge_item.icd_code,
                "name": f"{knowledge_item.name_th} ({knowledge_item.name_en})",
                "urgency": knowledge_item.urgency,
                "category": knowledge_item.category,
                "evidence_level": "knowledge_base_derived"
            }

            # Build expected treatment
            expected_treatment = await self._generate_treatment_plan(knowledge_item, complexity)

            # Generate learning objectives
            learning_objectives = await self._generate_learning_objectives(
                knowledge_item, scenario_type, complexity
            )

            # Generate safety considerations
            safety_considerations = await self._generate_safety_considerations(
                knowledge_item, presenting_symptoms
            )

            # Calculate confidence target
            confidence_target = self._calculate_confidence_target(knowledge_item, complexity)

            # Generate few-shot prompt
            few_shot_prompt = await self._generate_few_shot_prompt(
                knowledge_item, presenting_symptoms, expected_diagnosis
            )

            scenario = GeneratedScenario(
                id=scenario_id,
                scenario_type=scenario_type,
                complexity=complexity,
                patient_profile=patient_profile,
                presenting_symptoms=presenting_symptoms,
                clinical_context=clinical_context,
                expected_diagnosis=expected_diagnosis,
                expected_treatment=expected_treatment,
                learning_objectives=learning_objectives,
                safety_considerations=safety_considerations,
                few_shot_prompt=few_shot_prompt,
                knowledge_sources=[f"Knowledge Item {knowledge_item.id}: {knowledge_item.name_en}"],
                confidence_target=confidence_target
            )

            # Validate scenario safety
            if await self._validate_scenario_safety(scenario):
                return scenario
            else:
                logger.warning(f"❌ Scenario {scenario_id} failed safety validation")
                return None

        except Exception as e:
            logger.error(f"❌ Failed to generate scenario {scenario_id}: {e}")
            return None

    def _determine_scenario_complexity(self, knowledge_item: KnowledgeItem) -> ScenarioComplexity:
        """Determine appropriate complexity level for scenario"""

        # Base on condition characteristics
        if knowledge_item.urgency == "emergency":
            return ScenarioComplexity.EMERGENCY
        elif knowledge_item.severity == "severe":
            return ScenarioComplexity.COMPLEX
        elif knowledge_item.frequency == "rare":
            return ScenarioComplexity.COMPLEX
        elif knowledge_item.frequency == "common":
            return ScenarioComplexity.SIMPLE
        else:
            return ScenarioComplexity.MODERATE

    async def _generate_patient_profile(self,
                                      knowledge_item: KnowledgeItem,
                                      complexity: ScenarioComplexity) -> PatientProfile:
        """Generate realistic patient profile"""

        # Age distribution based on condition
        if 'pediatric' in knowledge_item.name_en.lower():
            age = random.randint(1, 17)
        elif 'geriatric' in knowledge_item.name_en.lower() or 'elderly' in knowledge_item.name_en.lower():
            age = random.randint(65, 85)
        else:
            age = random.randint(18, 75)

        # Gender distribution
        gender_weights = {"male": 0.5, "female": 0.5}
        if 'pregnancy' in knowledge_item.name_en.lower() or 'menstrual' in knowledge_item.name_en.lower():
            gender = "female"
        else:
            gender = random.choices(list(gender_weights.keys()), weights=list(gender_weights.values()))[0]

        # Medical history based on complexity
        medical_history = []
        if complexity in [ScenarioComplexity.MODERATE, ScenarioComplexity.COMPLEX]:
            potential_history = [
                "Hypertension", "Diabetes Type 2", "Hyperlipidemia",
                "Previous MI", "Asthma", "COPD", "Depression"
            ]
            medical_history = random.sample(potential_history, random.randint(0, 2))

        # Risk factors
        risk_factors = []
        if age > 50:
            risk_factors.append("Advanced age")
        if random.random() < 0.3:
            risk_factors.extend(["Smoking history", "Family history of disease"])

        # Current medications
        current_medications = []
        if medical_history:
            medication_map = {
                "Hypertension": "Lisinopril",
                "Diabetes Type 2": "Metformin",
                "Hyperlipidemia": "Atorvastatin"
            }
            current_medications = [medication_map.get(condition, "As needed")
                                 for condition in medical_history if condition in medication_map]

        return PatientProfile(
            age=age,
            gender=gender,
            occupation=random.choice(["Office worker", "Teacher", "Retired", "Student", "Farmer"]),
            medical_history=medical_history,
            risk_factors=risk_factors,
            current_medications=current_medications
        )

    async def _generate_presenting_symptoms(self,
                                          knowledge_item: KnowledgeItem,
                                          complexity: ScenarioComplexity,
                                          patient_profile: PatientProfile) -> Dict[str, str]:
        """Generate realistic presenting symptoms"""

        # Base symptoms from knowledge item
        base_symptoms = knowledge_item.symptoms[:3]  # Take first 3 symptoms

        # Add complexity-based variations
        if complexity == ScenarioComplexity.SIMPLE:
            symptom_count = 2
        elif complexity == ScenarioComplexity.MODERATE:
            symptom_count = 3
        else:
            symptom_count = 4

        # Select and format symptoms
        selected_symptoms = base_symptoms[:symptom_count]

        # Create realistic symptom descriptions
        thai_symptoms = []
        english_symptoms = []

        for symptom in selected_symptoms:
            if any(thai_char for thai_char in symptom if ord(thai_char) > 127):
                thai_symptoms.append(symptom)
            else:
                english_symptoms.append(symptom)

        # Add common modifiers
        thai_description = " ".join(thai_symptoms)
        if complexity == ScenarioComplexity.SIMPLE:
            thai_description += " เล็กน้อย"

        english_description = " ".join(english_symptoms)
        if complexity == ScenarioComplexity.SIMPLE:
            english_description += " mild"

        # Add duration
        duration_thai = random.choice(["มา 2-3 วัน", "มา 1 สัปดาห์", "เป็นมาหลายวัน"])
        duration_english = random.choice(["for 2-3 days", "for 1 week", "for several days"])

        return {
            "thai": f"{thai_description} {duration_thai}",
            "english": f"{english_description} {duration_english}"
        }

    async def _generate_clinical_context(self,
                                       knowledge_item: KnowledgeItem,
                                       patient_profile: PatientProfile,
                                       presenting_symptoms: Dict[str, str]) -> str:
        """Generate clinical context narrative"""

        context_parts = []

        # Patient presentation
        context_parts.append(
            f"A {patient_profile.age}-year-old {patient_profile.gender} {patient_profile.occupation} "
            f"presents with {presenting_symptoms['english']}."
        )

        # Medical history context
        if patient_profile.medical_history:
            context_parts.append(
                f"Past medical history includes {', '.join(patient_profile.medical_history)}."
            )

        # Current medications
        if patient_profile.current_medications:
            context_parts.append(
                f"Current medications: {', '.join(patient_profile.current_medications)}."
            )

        # Risk factors
        if patient_profile.risk_factors:
            context_parts.append(
                f"Risk factors include {', '.join(patient_profile.risk_factors)}."
            )

        # Clinical correlation
        if knowledge_item.frequency == "common":
            context_parts.append(
                "This presentation is consistent with common patterns seen in primary care."
            )
        elif knowledge_item.urgency == "emergency":
            context_parts.append(
                "The clinical presentation raises concerns requiring immediate evaluation."
            )

        return " ".join(context_parts)

    async def _generate_treatment_plan(self,
                                     knowledge_item: KnowledgeItem,
                                     complexity: ScenarioComplexity) -> Dict[str, Any]:
        """Generate appropriate treatment plan"""

        treatment_plan = {
            "immediate": [],
            "medications": knowledge_item.treatments[:3] if knowledge_item.treatments else ["Symptomatic treatment"],
            "monitoring": [],
            "follow_up": "As clinically indicated",
            "patient_education": []
        }

        # Add immediate actions based on urgency
        if knowledge_item.urgency == "emergency":
            treatment_plan["immediate"] = ["Emergency evaluation", "Vital signs monitoring"]
        elif knowledge_item.urgency == "high":
            treatment_plan["immediate"] = ["Urgent clinical assessment"]

        # Add monitoring based on complexity
        if complexity in [ScenarioComplexity.COMPLEX, ScenarioComplexity.EMERGENCY]:
            treatment_plan["monitoring"] = ["Symptom progression", "Treatment response"]

        # Patient education
        treatment_plan["patient_education"] = [
            "Return if symptoms worsen",
            "Follow medication instructions",
            "Rest and hydration as appropriate"
        ]

        return treatment_plan

    async def _generate_learning_objectives(self,
                                          knowledge_item: KnowledgeItem,
                                          scenario_type: ScenarioType,
                                          complexity: ScenarioComplexity) -> List[str]:
        """Generate learning objectives for the scenario"""

        objectives = []

        # Base objectives by scenario type
        if scenario_type == ScenarioType.DIAGNOSTIC:
            objectives.append(f"Recognize key features of {knowledge_item.name_en}")
            objectives.append("Apply systematic diagnostic reasoning")

        elif scenario_type == ScenarioType.TREATMENT:
            objectives.append("Select appropriate treatment based on evidence")
            objectives.append("Consider patient-specific factors in treatment planning")

        elif scenario_type == ScenarioType.TRIAGE:
            objectives.append("Assess urgency level appropriately")
            objectives.append("Identify red flag symptoms requiring escalation")

        # Complexity-specific objectives
        if complexity == ScenarioComplexity.SIMPLE:
            objectives.append("Demonstrate confidence in common presentations")
        elif complexity == ScenarioComplexity.EMERGENCY:
            objectives.append("Recognize time-critical clinical situations")

        # Safety objectives
        objectives.append("Maintain appropriate diagnostic confidence levels")
        objectives.append("Apply safety-first approach to patient care")

        return objectives

    async def _generate_safety_considerations(self,
                                            knowledge_item: KnowledgeItem,
                                            presenting_symptoms: Dict[str, str]) -> List[str]:
        """Generate safety considerations for the scenario"""

        safety_considerations = []

        # Urgency-based safety
        if knowledge_item.urgency == "emergency":
            safety_considerations.append("Time-critical condition requiring immediate intervention")
        elif knowledge_item.urgency == "high":
            safety_considerations.append("Requires prompt medical evaluation within 24 hours")

        # Severity-based safety
        if knowledge_item.severity == "severe":
            safety_considerations.append("Monitor for complications and disease progression")

        # Symptom-specific safety
        symptoms_text = presenting_symptoms.get('english', '').lower()
        if 'chest pain' in symptoms_text:
            safety_considerations.append("Rule out cardiac emergency")
        if 'shortness of breath' in symptoms_text:
            safety_considerations.append("Assess respiratory status and oxygen saturation")
        if 'headache' in symptoms_text:
            safety_considerations.append("Screen for neurological emergency signs")

        # General safety
        safety_considerations.append("Ensure patient understanding of when to seek urgent care")

        if not safety_considerations:
            safety_considerations.append("Monitor symptom progression and treatment response")

        return safety_considerations

    def _calculate_confidence_target(self,
                                   knowledge_item: KnowledgeItem,
                                   complexity: ScenarioComplexity) -> float:
        """Calculate appropriate confidence target for scenario"""

        base_confidence = {
            ScenarioComplexity.SIMPLE: 0.85,
            ScenarioComplexity.MODERATE: 0.75,
            ScenarioComplexity.COMPLEX: 0.65,
            ScenarioComplexity.EMERGENCY: 0.70
        }

        confidence = base_confidence[complexity]

        # Adjust based on frequency
        if knowledge_item.frequency == "common":
            confidence += 0.05
        elif knowledge_item.frequency == "rare":
            confidence -= 0.10

        # Ensure bounds
        return max(0.5, min(0.95, confidence))

    async def _generate_few_shot_prompt(self,
                                      knowledge_item: KnowledgeItem,
                                      presenting_symptoms: Dict[str, str],
                                      expected_diagnosis: Dict[str, Any]) -> str:
        """Generate few-shot prompt segment"""

        prompt = f"""
Patient Case: {presenting_symptoms['thai']}

Analysis:
- Presenting symptoms: {presenting_symptoms['english']}
- Primary diagnosis: {expected_diagnosis['name']}
- ICD Code: {expected_diagnosis['icd_code']}
- Confidence: {int(self._calculate_confidence_target(knowledge_item, ScenarioComplexity.MODERATE) * 100)}%
- Urgency: {expected_diagnosis['urgency']}

Clinical Reasoning:
The presentation is consistent with {knowledge_item.name_en}, characterized by {', '.join(knowledge_item.keywords[:3])}.
This diagnosis is supported by the patient's symptom pattern and clinical context.
"""

        return prompt.strip()

    async def _validate_scenario_safety(self, scenario: GeneratedScenario) -> bool:
        """Validate that generated scenario meets safety requirements"""

        # Check confidence target is reasonable
        if scenario.confidence_target > 0.95:
            logger.warning(f"Scenario {scenario.id} has overconfident target: {scenario.confidence_target}")
            return False

        # Check for dangerous combinations
        symptoms_text = scenario.presenting_symptoms.get('english', '').lower()
        diagnosis_name = scenario.expected_diagnosis.get('name', '').lower()

        # Don't create scenarios that suggest serious diseases for mild symptoms
        mild_indicators = ['mild', 'minor', 'slight']
        serious_diseases = ['cancer', 'tumor', 'stroke', 'heart attack', 'tuberculosis']

        has_mild_symptoms = any(mild in symptoms_text for mild in mild_indicators)
        has_serious_diagnosis = any(serious in diagnosis_name for serious in serious_diseases)

        if has_mild_symptoms and has_serious_diagnosis:
            logger.warning(f"Scenario {scenario.id} has dangerous mild→serious pattern")
            return False

        return True

    async def _build_scenario_templates(self):
        """Build scenario templates for different types"""
        # Implementation for building reusable scenario templates
        pass

    async def _generate_patient_profiles(self):
        """Generate a pool of realistic patient profiles"""
        # Implementation for generating diverse patient profiles
        pass

    def _format_treatment_plan(self, treatment_plan: Dict[str, Any]) -> str:
        """Format treatment plan for display"""
        formatted_parts = []

        if treatment_plan.get('immediate'):
            formatted_parts.append(f"**Immediate**: {', '.join(treatment_plan['immediate'])}")

        if treatment_plan.get('medications'):
            formatted_parts.append(f"**Medications**: {', '.join(treatment_plan['medications'])}")

        if treatment_plan.get('monitoring'):
            formatted_parts.append(f"**Monitoring**: {', '.join(treatment_plan['monitoring'])}")

        if treatment_plan.get('follow_up'):
            formatted_parts.append(f"**Follow-up**: {treatment_plan['follow_up']}")

        return "\n".join(formatted_parts)


# Global instance
rag_scenario_generator = RAGScenarioGenerator()