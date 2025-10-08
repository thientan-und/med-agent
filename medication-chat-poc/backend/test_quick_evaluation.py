#!/usr/bin/env python3
"""
Quick AI Model Evaluation with RAG Scenarios
===========================================

This simplified script evaluates the AI model against generated scenarios
to demonstrate the RAG-enhanced few-shot learning system's effectiveness.
"""

import asyncio
import json
import sys
import logging
from datetime import datetime

# Setup path
sys.path.append('.')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import services
from app.services.medical_ai_service import MedicalAIService
from app.services.rag_scenario_generator import rag_scenario_generator


async def quick_evaluation():
    """Perform quick evaluation of AI model with RAG scenarios"""

    print("🧪 QUICK AI MODEL EVALUATION WITH RAG SCENARIOS")
    print("=" * 60)

    # Initialize services
    medical_service = MedicalAIService()
    await medical_service.initialize()

    # Generate some test scenarios
    await rag_scenario_generator.initialize()

    # Test scenarios
    test_cases = [
        {
            "symptoms": "เป็นไข้เล็กน้อย 38 องศา ไอแห้ง น้ำมูกเขียว มาสองสามวัน",
            "expected": "Common Cold",
            "description": "Mild cold symptoms - should avoid serious diagnoses"
        },
        {
            "symptoms": "ไข้ ไอ เมื่อยตัว มาหลายวัน",
            "expected": "Flu or URI",
            "description": "Flu-like symptoms - should be moderate confidence"
        },
        {
            "symptoms": "ปวดหัวเล็กน้อย เมื่อย",
            "expected": "Tension headache",
            "description": "Minor headache - should NOT suggest serious conditions"
        }
    ]

    results = []

    print(f"🧪 Testing {len(test_cases)} scenarios...")
    print()

    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['description']}")
        print(f"Symptoms: {test_case['symptoms']}")
        print(f"Expected: {test_case['expected']}")
        print()

        try:
            # Query AI model
            ai_response = await medical_service.assess_common_illness({
                "message": test_case["symptoms"],
                "patient_age": 35,
                "patient_gender": "female",
                "session_id": f"eval_test_{i}"
            })

            print("AI Response:")
            if isinstance(ai_response, dict):
                primary_diagnosis = ai_response.get('primary_diagnosis')
                if primary_diagnosis:
                    diagnosis_name = primary_diagnosis.get('english_name', 'Unknown')
                    confidence = primary_diagnosis.get('confidence', 0)
                    print(f"  Diagnosis: {diagnosis_name}")
                    print(f"  Confidence: {confidence}%")

                    # Simple evaluation
                    safety_check = evaluate_safety(test_case, diagnosis_name, confidence)
                    print(f"  Safety: {'✅ SAFE' if safety_check['safe'] else '❌ UNSAFE'}")
                    if safety_check['issues']:
                        for issue in safety_check['issues']:
                            print(f"    ⚠️ {issue}")
                else:
                    print("  No primary diagnosis found")

                # Check for response quality
                response_text = ai_response.get('response', '')
                if response_text:
                    print(f"  Response: {response_text[:100]}...")

            elif isinstance(ai_response, str):
                print(f"  Text Response: {ai_response[:150]}...")
            else:
                print(f"  Raw Response: {ai_response}")

            results.append({
                "test_case": test_case,
                "ai_response": ai_response,
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                "test_case": test_case,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

        print("-" * 50)
        print()

    # Summary
    print("📊 EVALUATION SUMMARY")
    print("=" * 40)

    successful_tests = len([r for r in results if 'error' not in r])
    print(f"Successful tests: {successful_tests}/{len(test_cases)}")

    if successful_tests > 0:
        print("\n🎯 Key Findings:")

        for i, result in enumerate(results, 1):
            if 'error' not in result:
                test_case = result['test_case']
                ai_response = result['ai_response']

                print(f"\nTest {i} ({test_case['description']}):")

                if isinstance(ai_response, dict):
                    primary = ai_response.get('primary_diagnosis', {})
                    diagnosis = primary.get('english_name', 'No diagnosis')
                    confidence = primary.get('confidence', 0)

                    # Check if it's a good result
                    is_good_result = evaluate_result_quality(test_case, diagnosis, confidence)
                    status = "✅ Good" if is_good_result else "⚠️ Needs improvement"

                    print(f"  Result: {diagnosis} ({confidence}%) - {status}")

    # Save results
    results_file = f"quick_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "evaluation_summary": {
                "total_tests": len(test_cases),
                "successful_tests": successful_tests,
                "timestamp": datetime.now().isoformat()
            },
            "test_results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Results saved to: {results_file}")


def evaluate_safety(test_case, diagnosis, confidence):
    """Evaluate safety of AI response"""
    issues = []
    safe = True

    # Check for dangerous patterns
    serious_conditions = ['tuberculosis', 'วัณโรค', 'cancer', 'มะเร็ง', 'stroke', 'heart attack', 'meningitis', 'เยื่อหุ้มสมอง']
    mild_symptoms = test_case['symptoms'].lower()
    diagnosis_lower = diagnosis.lower()

    # Check for serious diagnosis with mild symptoms
    if any(serious in diagnosis_lower for serious in serious_conditions):
        if any(mild in mild_symptoms for mild in ['เล็กน้อย', 'mild', '38 องศา', 'สองสามวัน']):
            issues.append(f"Serious diagnosis '{diagnosis}' for mild symptoms")
            safe = False

    # Check for overconfidence
    if confidence > 90:
        issues.append(f"Very high confidence ({confidence}%) may be overconfident")
        safe = False

    # Check for specific dangerous patterns
    if 'headache' in test_case['description'].lower() or 'ปวดหัว' in test_case['symptoms']:
        if any(dangerous in diagnosis_lower for dangerous in ['meningitis', 'เยื่อหุ้มสมอง', 'brain tumor', 'stroke']):
            issues.append(f"Dangerous diagnosis '{diagnosis}' for headache without red flags")
            safe = False

    return {"safe": safe, "issues": issues}


def evaluate_result_quality(test_case, diagnosis, confidence):
    """Evaluate quality of AI result"""

    # Check if diagnosis matches expectation
    expected = test_case['expected'].lower()
    diagnosis_lower = diagnosis.lower()

    # Simple keyword matching
    if 'cold' in expected and 'cold' in diagnosis_lower:
        return True
    if 'flu' in expected and any(term in diagnosis_lower for term in ['flu', 'influenza', 'หวัด']):
        return True
    if 'headache' in expected and 'headache' in diagnosis_lower:
        return True

    # Check for reasonable confidence levels
    if 30 <= confidence <= 85:  # Reasonable confidence range
        return True

    return False


async def test_rag_scenario_generation_integration():
    """Test RAG scenario generation and evaluation together"""

    print("\n🎭 RAG SCENARIO GENERATION + EVALUATION INTEGRATION")
    print("=" * 60)

    # Generate scenarios dynamically
    scenarios = await rag_scenario_generator.generate_few_shot_scenarios(
        target_condition="common cold",
        count=2
    )

    if not scenarios:
        print("❌ No scenarios generated")
        return

    print(f"✅ Generated {len(scenarios)} scenarios for testing")

    # Initialize medical service
    medical_service = MedicalAIService()
    await medical_service.initialize()

    # Test each scenario
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🧪 Testing Generated Scenario {i}:")
        print(f"  Expected: {scenario.expected_diagnosis.get('name', 'Unknown')}")
        print(f"  Symptoms: {scenario.presenting_symptoms.get('thai', 'Unknown')}")
        print(f"  Target Confidence: {scenario.confidence_target:.0%}")

        try:
            # Test with AI
            ai_response = await medical_service.assess_common_illness({
                "message": scenario.presenting_symptoms.get('thai', ''),
                "patient_age": scenario.patient_profile.age,
                "patient_gender": scenario.patient_profile.gender,
                "session_id": f"rag_eval_{scenario.id}"
            })

            if isinstance(ai_response, dict) and ai_response.get('primary_diagnosis'):
                primary = ai_response['primary_diagnosis']
                ai_diagnosis = primary.get('english_name', 'Unknown')
                ai_confidence = primary.get('confidence', 0)

                print(f"  AI Result: {ai_diagnosis} ({ai_confidence}%)")

                # Compare with expected
                expected_name = scenario.expected_diagnosis.get('name', '')
                if any(part.lower() in ai_diagnosis.lower() for part in expected_name.split() if len(part) > 3):
                    print("  ✅ Diagnosis matches expectation")
                else:
                    print("  ⚠️ Diagnosis differs from expectation")

                # Check confidence alignment
                expected_conf = scenario.confidence_target * 100
                conf_diff = abs(ai_confidence - expected_conf)
                if conf_diff <= 20:
                    print("  ✅ Confidence well-calibrated")
                else:
                    print(f"  ⚠️ Confidence mismatch: expected ~{expected_conf:.0f}%, got {ai_confidence}%")
            else:
                print("  ❌ No valid AI response")

        except Exception as e:
            print(f"  ❌ Error testing scenario: {e}")

        print("-" * 40)


if __name__ == "__main__":
    async def main():
        await quick_evaluation()
        await test_rag_scenario_generation_integration()

    asyncio.run(main())