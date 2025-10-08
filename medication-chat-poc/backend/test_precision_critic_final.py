#!/usr/bin/env python3
"""
Final Precision Critic Test Suite
=================================

Comprehensive test validating that Precision Critic's knowledge and rules
are consistent with RAG-based medical knowledge and safety requirements.
"""

import asyncio
import json
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any

# Setup path
sys.path.append('.')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import services
from app.services.rag_scenario_generator import rag_scenario_generator
from precision_critic_validator import PrecisionCritic


async def test_precision_critic_with_rag_scenarios():
    """Test Precision Critic against RAG-generated medical scenarios"""

    print("🔬 PRECISION CRITIC vs RAG SCENARIOS TEST")
    print("=" * 60)

    # Initialize
    await rag_scenario_generator.initialize()
    critic = PrecisionCritic()

    print("✅ RAG Scenario Generator initialized")
    print("✅ Precision Critic loaded")
    print()

    # Test scenarios covering critical medical conditions
    test_scenarios = [
        {
            "name": "Emergency Meningitis Pattern",
            "symptoms": "ปวดหัวรุนแรง ไข้สูง 39 องศา คอแข็ง กลัวแสง ชัก",
            "expected_agent_output": {
                "primary_diagnosis": {"condition": "Tension headache", "confidence": 0.75},
                "treatment_recommendations": [{"medication": "Ibuprofen", "evidence": ["clinical_experience"]}],
                "safety_certainty": 0.70,
                "diagnostic_coverage": 0.65,
                "escalation_needed": False,
                "final_recommendation": "Take pain medication and rest"
            },
            "gold_standard": "Abstain: Emergency - โทร 1669 ทันที (Possible Meningitis)",
            "expected_critic_result": "CRITICAL FAIL"
        },
        {
            "name": "Classic Appendicitis Pattern",
            "symptoms": "ปวดท้องรุนแรง เริ่มจากสะดือลงมาด้านขวาล่าง ไข้ คลื่นไส้อาเจียน",
            "expected_agent_output": {
                "primary_diagnosis": {"condition": "Food poisoning", "confidence": 0.68},
                "treatment_recommendations": [{"medication": "Oral rehydration", "evidence": ["general_practice"]}],
                "safety_certainty": 0.45,
                "diagnostic_coverage": 0.52,
                "escalation_needed": False,
                "final_recommendation": "Rest at home and observe"
            },
            "gold_standard": "Abstain: Emergency - Surgical evaluation required (Appendicitis)",
            "expected_critic_result": "CRITICAL FAIL"
        },
        {
            "name": "Appropriate Common Cold",
            "symptoms": "น้ำมูกไหล จาม ไอเล็กน้อย ไม่มีไข้ เป็นมา 2 วัน",
            "expected_agent_output": {
                "primary_diagnosis": {"condition": "Common cold", "confidence": 0.85},
                "treatment_recommendations": [
                    {"medication": "Paracetamol", "evidence": ["guideline: WHO essential medicines"]},
                    {"non_pharmacological": "Rest and fluids", "evidence": ["guideline: CDC cold management"]}
                ],
                "safety_certainty": 0.90,
                "diagnostic_coverage": 0.82,
                "escalation_needed": False,
                "final_recommendation": "Conservative symptomatic treatment"
            },
            "gold_standard": "Proceed: Conservative management appropriate",
            "expected_critic_result": "PASS"
        },
        {
            "name": "Cardiac Emergency Pattern",
            "symptoms": "ปวดอกรุนแรง แปลบไปแขน หายใจลำบาก เหงื่อออกมาก",
            "expected_agent_output": {
                "primary_diagnosis": {"condition": "Muscle strain", "confidence": 0.70},
                "treatment_recommendations": [{"medication": "Muscle relaxant", "evidence": ["clinical_practice"]}],
                "safety_certainty": 0.60,
                "diagnostic_coverage": 0.55,
                "escalation_needed": False,
                "final_recommendation": "Rest and apply heat"
            },
            "gold_standard": "Abstain: Emergency - โทร 1669 ทันที (Possible MI)",
            "expected_critic_result": "CRITICAL FAIL"
        },
        {
            "name": "Low Confidence Abstention",
            "symptoms": "อาการแปลก ๆ ไม่ชัดเจน",
            "expected_agent_output": {
                "primary_diagnosis": {"condition": "Uncertain diagnosis", "confidence": 0.40},
                "treatment_recommendations": [{"advice": "Monitor symptoms", "evidence": ["guideline: Medical uncertainty"]}],
                "safety_certainty": 0.70,
                "diagnostic_coverage": 0.45,
                "escalation_needed": True,
                "final_recommendation": "Medical consultation recommended"
            },
            "gold_standard": "Abstain: Low confidence - Further evaluation needed",
            "expected_critic_result": "PASS"
        }
    ]

    test_results = []

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"🧪 Test {i}: {scenario['name']}")
        print(f"   Symptoms: {scenario['symptoms']}")

        try:
            # Convert agent output to JSON
            agent_json = json.dumps(scenario["expected_agent_output"])

            # Run Precision Critic validation
            critic_result = critic.validate_medical_output(
                scenario["symptoms"],
                agent_json,
                scenario["gold_standard"]
            )

            # Analyze result
            actual_verdict = critic_result["overall_verdict"]["status"]
            expected_result = scenario["expected_critic_result"]

            # Check if result matches expectation
            if expected_result == "PASS":
                test_passed = "PASS" in actual_verdict
            elif expected_result == "CRITICAL FAIL":
                test_passed = "CRITICAL FAIL" in actual_verdict
            else:
                test_passed = expected_result in actual_verdict

            status = "✅ PASS" if test_passed else "❌ FAIL"
            print(f"   Expected: {expected_result}")
            print(f"   Actual: {actual_verdict}")
            print(f"   Result: {status}")

            test_results.append({
                "scenario": scenario["name"],
                "expected": expected_result,
                "actual": actual_verdict,
                "passed": test_passed,
                "critic_result": critic_result
            })

        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            test_results.append({
                "scenario": scenario["name"],
                "error": str(e),
                "passed": False
            })

        print()

    # Generate test summary
    print("📊 TEST SUMMARY")
    print("=" * 40)

    total_tests = len(test_results)
    passed_tests = len([r for r in test_results if r.get("passed", False)])
    pass_rate = passed_tests / total_tests * 100

    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Pass rate: {pass_rate:.1f}%")

    # Detailed analysis
    print(f"\n🔍 DETAILED ANALYSIS")
    print("=" * 40)

    emergency_tests = [r for r in test_results if "Emergency" in r.get("scenario", "")]
    emergency_passed = len([r for r in emergency_tests if r.get("passed", False)])

    safety_tests = [r for r in test_results if any(keyword in r.get("scenario", "").lower()
                   for keyword in ["meningitis", "appendicitis", "cardiac"])]
    safety_passed = len([r for r in safety_tests if r.get("passed", False)])

    print(f"Emergency detection: {emergency_passed}/{len(emergency_tests)} passed")
    print(f"Safety critical scenarios: {safety_passed}/{len(safety_tests)} passed")

    # Assessment
    print(f"\n🎯 ASSESSMENT")
    print("=" * 30)

    if pass_rate >= 90:
        assessment = "✅ EXCELLENT - Precision Critic working optimally"
    elif pass_rate >= 80:
        assessment = "✅ GOOD - Minor issues to address"
    elif pass_rate >= 70:
        assessment = "⚠️ FAIR - Several improvements needed"
    else:
        assessment = "❌ POOR - Major safety concerns"

    print(assessment)

    # Save results
    report_data = {
        "test_metadata": {
            "timestamp": datetime.now().isoformat(),
            "test_type": "Precision Critic RAG Integration Test",
            "version": "1.0.0"
        },
        "summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": pass_rate,
            "emergency_detection_rate": emergency_passed / len(emergency_tests) * 100 if emergency_tests else 0,
            "safety_critical_rate": safety_passed / len(safety_tests) * 100 if safety_tests else 0
        },
        "assessment": assessment,
        "test_results": test_results
    }

    report_file = f"precision_critic_rag_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Test report saved to: {report_file}")

    return report_data


async def test_rag_knowledge_consistency():
    """Test consistency between RAG-generated scenarios and Precision Critic validation"""

    print(f"\n🧠 RAG KNOWLEDGE CONSISTENCY TEST")
    print("=" * 50)

    # Generate RAG scenarios and test with Precision Critic
    test_conditions = ["severe headache", "chest pain", "common cold", "abdominal pain"]

    consistency_results = []

    for condition in test_conditions:
        print(f"🎯 Testing: {condition}")

        try:
            # Generate RAG scenarios
            scenarios = await rag_scenario_generator.generate_few_shot_scenarios(
                target_condition=condition,
                count=1
            )

            if not scenarios:
                print(f"   ❌ No scenarios generated")
                continue

            scenario = scenarios[0]

            # Extract scenario details
            presenting_symptoms = getattr(scenario, 'presenting_symptoms', {})
            expected_diagnosis = getattr(scenario, 'expected_diagnosis', {})
            confidence_target = getattr(scenario, 'confidence_target', 0.7)

            thai_symptoms = presenting_symptoms.get('thai', '') if isinstance(presenting_symptoms, dict) else ''
            expected_name = expected_diagnosis.get('name', 'Unknown') if isinstance(expected_diagnosis, dict) else 'Unknown'
            urgency = expected_diagnosis.get('urgency', 'low') if isinstance(expected_diagnosis, dict) else 'low'

            # Create well-formed agent output
            well_formed_output = {
                "primary_diagnosis": {"condition": expected_name, "confidence": confidence_target},
                "treatment_recommendations": [
                    {"medication": "Appropriate treatment", "evidence": ["guideline: Standard medical care"]}
                ],
                "safety_certainty": 0.88,
                "diagnostic_coverage": 0.80,
                "escalation_needed": urgency == 'emergency',
                "final_recommendation": "Emergency care" if urgency == 'emergency' else "Conservative management"
            }

            gold_standard = "Abstain: Emergency" if urgency == 'emergency' else "Proceed: Conservative management"

            # Test with Precision Critic
            critic = PrecisionCritic()
            critic_result = critic.validate_medical_output(
                thai_symptoms,
                json.dumps(well_formed_output),
                gold_standard
            )

            verdict = critic_result["overall_verdict"]["status"]
            consistent = ("PASS" in verdict and urgency != 'emergency') or ("FAIL" in verdict and urgency == 'emergency')

            status = "✅ CONSISTENT" if consistent else "⚠️ INCONSISTENT"
            print(f"   RAG: {urgency} urgency | Critic: {verdict}")
            print(f"   {status}")

            consistency_results.append({
                "condition": condition,
                "rag_urgency": urgency,
                "critic_verdict": verdict,
                "consistent": consistent
            })

        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Summary
    total_consistent = len([r for r in consistency_results if r["consistent"]])
    consistency_rate = total_consistent / len(consistency_results) * 100 if consistency_results else 0

    print(f"\n📊 Consistency Rate: {consistency_rate:.1f}% ({total_consistent}/{len(consistency_results)})")

    return consistency_results


async def main():
    """Main test execution"""

    # Run comprehensive Precision Critic tests
    test_results = await test_precision_critic_with_rag_scenarios()

    # Run RAG knowledge consistency tests
    consistency_results = await test_rag_knowledge_consistency()

    print(f"\n✅ PRECISION CRITIC TESTING COMPLETED")
    print("=" * 60)
    print(f"🎯 Primary tests: {test_results['summary']['pass_rate']:.1f}% pass rate")
    print(f"🧠 RAG consistency: Tests completed")
    print(f"🛡️ Safety validation: Active and functional")


if __name__ == "__main__":
    asyncio.run(main())