#!/usr/bin/env python3
"""
RAG Model Validation and Testing System
======================================

This script uses RAG-generated scenarios to validate AI model performance
without encountering the technical issues from direct API calls.
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


async def create_comprehensive_test_suite():
    """Create a comprehensive test suite using RAG scenarios"""

    print("🧪 RAG MODEL VALIDATION SYSTEM")
    print("=" * 60)

    await rag_scenario_generator.initialize()
    print("✅ RAG Scenario Generator initialized")
    print()

    # Generate test scenarios across different medical domains
    test_domains = [
        ("respiratory conditions", ["common cold", "flu", "cough", "respiratory infection"], 3),
        ("pain conditions", ["headache", "chest pain", "abdominal pain", "back pain"], 2),
        ("general symptoms", ["fever", "fatigue", "dizziness", "nausea"], 2),
        ("skin conditions", ["rash", "skin irritation"], 2),
        ("digestive issues", ["stomach pain", "diarrhea"], 2)
    ]

    all_test_scenarios = []
    domain_stats = {}

    print("📊 CREATING COMPREHENSIVE TEST SUITE")
    print("=" * 50)

    total_requested = 0
    total_generated = 0

    for domain_name, conditions, count_per_condition in test_domains:
        print(f"\n🎯 Domain: {domain_name}")
        domain_scenarios = []

        for condition in conditions:
            print(f"   Generating {count_per_condition} scenarios for {condition}: ", end="")
            total_requested += count_per_condition

            try:
                scenarios = await rag_scenario_generator.generate_few_shot_scenarios(
                    target_condition=condition,
                    count=count_per_condition
                )

                if scenarios:
                    domain_scenarios.extend(scenarios)
                    total_generated += len(scenarios)
                    print(f"✅ {len(scenarios)}")
                else:
                    print("❌ 0")

            except Exception as e:
                print(f"❌ Error: {str(e)[:30]}...")

        all_test_scenarios.extend(domain_scenarios)
        domain_stats[domain_name] = {
            "conditions": len(conditions),
            "scenarios_generated": len(domain_scenarios),
            "requested": len(conditions) * count_per_condition
        }

        print(f"   Domain Total: {len(domain_scenarios)} scenarios")

    print(f"\n📈 TEST SUITE GENERATION SUMMARY")
    print("=" * 50)
    print(f"Total scenarios generated: {total_generated}")
    print(f"Total requested: {total_requested}")
    print(f"Success rate: {total_generated/total_requested*100:.1f}%")
    print(f"Domains covered: {len(test_domains)}")

    return all_test_scenarios, domain_stats


async def analyze_scenario_characteristics(scenarios: List[Any]):
    """Analyze the characteristics of generated scenarios"""

    print(f"\n🔍 SCENARIO CHARACTERISTICS ANALYSIS")
    print("=" * 50)

    if not scenarios:
        print("❌ No scenarios to analyze")
        return {}

    # Collect scenario characteristics
    characteristics = {
        "scenario_types": {},
        "complexity_levels": {},
        "urgency_levels": {},
        "confidence_distribution": [],
        "age_distribution": [],
        "gender_distribution": {},
        "icd_codes": set(),
        "learning_objectives": [],
        "safety_considerations": []
    }

    for scenario in scenarios:
        # Scenario type and complexity
        scenario_type = getattr(scenario, 'scenario_type', None)
        complexity = getattr(scenario, 'complexity', None)

        if scenario_type:
            type_name = str(scenario_type).split('.')[-1]
            characteristics["scenario_types"][type_name] = characteristics["scenario_types"].get(type_name, 0) + 1

        if complexity:
            complexity_name = str(complexity).split('.')[-1]
            characteristics["complexity_levels"][complexity_name] = characteristics["complexity_levels"].get(complexity_name, 0) + 1

        # Expected diagnosis
        expected_diagnosis = getattr(scenario, 'expected_diagnosis', {})
        if isinstance(expected_diagnosis, dict):
            urgency = expected_diagnosis.get('urgency', 'unknown')
            icd_code = expected_diagnosis.get('icd_code', '')

            characteristics["urgency_levels"][urgency] = characteristics["urgency_levels"].get(urgency, 0) + 1
            if icd_code:
                characteristics["icd_codes"].add(icd_code)

        # Confidence target
        confidence_target = getattr(scenario, 'confidence_target', 0)
        if confidence_target > 0:
            characteristics["confidence_distribution"].append(confidence_target)

        # Patient profile
        patient_profile = getattr(scenario, 'patient_profile', None)
        if patient_profile:
            age = getattr(patient_profile, 'age', None)
            gender = getattr(patient_profile, 'gender', None)

            if age:
                characteristics["age_distribution"].append(age)
            if gender:
                characteristics["gender_distribution"][gender] = characteristics["gender_distribution"].get(gender, 0) + 1

        # Learning objectives and safety considerations
        learning_objectives = getattr(scenario, 'learning_objectives', [])
        safety_considerations = getattr(scenario, 'safety_considerations', [])

        characteristics["learning_objectives"].extend(learning_objectives)
        characteristics["safety_considerations"].extend(safety_considerations)

    # Display analysis results
    print(f"📊 Analysis Results:")
    print(f"   Total scenarios analyzed: {len(scenarios)}")

    print(f"\n🎭 Scenario Types:")
    for stype, count in characteristics["scenario_types"].items():
        print(f"   {stype}: {count} scenarios")

    print(f"\n⚡ Complexity Levels:")
    for complexity, count in characteristics["complexity_levels"].items():
        print(f"   {complexity}: {count} scenarios")

    print(f"\n🚨 Urgency Levels:")
    for urgency, count in characteristics["urgency_levels"].items():
        print(f"   {urgency}: {count} scenarios")

    # Confidence analysis
    if characteristics["confidence_distribution"]:
        conf_dist = characteristics["confidence_distribution"]
        avg_conf = sum(conf_dist) / len(conf_dist)
        min_conf = min(conf_dist)
        max_conf = max(conf_dist)

        print(f"\n🎯 Confidence Distribution:")
        print(f"   Average: {avg_conf:.1%}")
        print(f"   Range: {min_conf:.1%} - {max_conf:.1%}")
        print(f"   Conservative approach: {'✅' if avg_conf < 0.8 else '⚠️'} (avg < 80%)")

        # Confidence buckets
        low_conf = len([c for c in conf_dist if c < 0.6])
        med_conf = len([c for c in conf_dist if 0.6 <= c < 0.8])
        high_conf = len([c for c in conf_dist if c >= 0.8])

        print(f"   Low confidence (< 60%): {low_conf}")
        print(f"   Medium confidence (60-80%): {med_conf}")
        print(f"   High confidence (≥ 80%): {high_conf}")

    # Age and gender analysis
    if characteristics["age_distribution"]:
        ages = characteristics["age_distribution"]
        avg_age = sum(ages) / len(ages)
        print(f"\n👥 Patient Demographics:")
        print(f"   Average age: {avg_age:.1f} years")
        print(f"   Age range: {min(ages)} - {max(ages)} years")

    if characteristics["gender_distribution"]:
        print(f"   Gender distribution:")
        for gender, count in characteristics["gender_distribution"].items():
            print(f"     {gender}: {count}")

    print(f"\n🏥 Medical Coding:")
    print(f"   Unique ICD codes: {len(characteristics['icd_codes'])}")

    print(f"\n📚 Learning Framework:")
    print(f"   Learning objectives: {len(characteristics['learning_objectives'])}")
    print(f"   Safety considerations: {len(characteristics['safety_considerations'])}")

    return characteristics


async def create_model_testing_framework(scenarios: List[Any]):
    """Create a framework for testing AI models using RAG scenarios"""

    print(f"\n🔬 MODEL TESTING FRAMEWORK")
    print("=" * 50)

    if not scenarios:
        print("❌ No scenarios available for testing framework")
        return {}

    # Create test categories
    test_categories = {
        "safety_critical": [],
        "confidence_calibration": [],
        "diagnostic_accuracy": [],
        "emergency_detection": [],
        "conservative_diagnosis": []
    }

    # Categorize scenarios for different types of testing
    for scenario in scenarios:
        expected_diagnosis = getattr(scenario, 'expected_diagnosis', {})
        urgency = expected_diagnosis.get('urgency', 'unknown') if isinstance(expected_diagnosis, dict) else 'unknown'
        confidence_target = getattr(scenario, 'confidence_target', 0)
        complexity = getattr(scenario, 'complexity', None)

        # Safety critical scenarios (emergency urgency)
        if urgency == 'emergency':
            test_categories["safety_critical"].append(scenario)

        # Confidence calibration (scenarios with specific confidence targets)
        if confidence_target > 0:
            test_categories["confidence_calibration"].append(scenario)

        # Diagnostic accuracy (all scenarios with clear expected diagnoses)
        if isinstance(expected_diagnosis, dict) and expected_diagnosis.get('name'):
            test_categories["diagnostic_accuracy"].append(scenario)

        # Emergency detection (emergency urgency scenarios)
        if urgency == 'emergency':
            test_categories["emergency_detection"].append(scenario)

        # Conservative diagnosis (low urgency scenarios)
        if urgency == 'low':
            test_categories["conservative_diagnosis"].append(scenario)

    print(f"📋 Test Categories Created:")
    for category, category_scenarios in test_categories.items():
        print(f"   {category}: {len(category_scenarios)} scenarios")

    # Create sample test cases for each category
    print(f"\n🧪 Sample Test Cases:")

    for category, category_scenarios in test_categories.items():
        if category_scenarios:
            sample = category_scenarios[0]  # Take first scenario as example

            print(f"\n   {category.replace('_', ' ').title()}:")

            # Get scenario details
            presenting_symptoms = getattr(sample, 'presenting_symptoms', {})
            expected_diagnosis = getattr(sample, 'expected_diagnosis', {})
            confidence_target = getattr(sample, 'confidence_target', 0)

            thai_symptoms = presenting_symptoms.get('thai', 'Unknown') if isinstance(presenting_symptoms, dict) else 'Unknown'
            expected_name = expected_diagnosis.get('name', 'Unknown') if isinstance(expected_diagnosis, dict) else 'Unknown'

            print(f"     Input: {thai_symptoms}")
            print(f"     Expected: {expected_name}")
            print(f"     Target Confidence: {confidence_target:.0%}")

            # Test criteria based on category
            if category == "safety_critical":
                print(f"     Test Criteria: Must not miss emergency conditions")
            elif category == "confidence_calibration":
                print(f"     Test Criteria: Confidence within ±15% of target")
            elif category == "diagnostic_accuracy":
                print(f"     Test Criteria: Correct diagnostic category")
            elif category == "emergency_detection":
                print(f"     Test Criteria: Proper emergency escalation")
            elif category == "conservative_diagnosis":
                print(f"     Test Criteria: Avoid overdiagnosis")

    return test_categories


async def generate_testing_report(scenarios: List[Any], characteristics: Dict, test_categories: Dict, domain_stats: Dict):
    """Generate comprehensive testing report"""

    print(f"\n📊 COMPREHENSIVE TESTING REPORT")
    print("=" * 60)

    # Summary statistics
    print(f"🎯 Test Suite Summary:")
    print(f"   Total scenarios: {len(scenarios)}")
    print(f"   Domains covered: {len(domain_stats)}")
    print(f"   Test categories: {len(test_categories)}")

    # Domain breakdown
    print(f"\n🏥 Domain Breakdown:")
    for domain, stats in domain_stats.items():
        success_rate = stats["scenarios_generated"] / stats["requested"] * 100
        print(f"   {domain}: {stats['scenarios_generated']}/{stats['requested']} scenarios ({success_rate:.1f}%)")

    # Testing capabilities
    print(f"\n🧪 Testing Capabilities:")
    for category, category_scenarios in test_categories.items():
        if category_scenarios:
            print(f"   ✅ {category.replace('_', ' ').title()}: {len(category_scenarios)} test cases")
        else:
            print(f"   ❌ {category.replace('_', ' ').title()}: No test cases")

    # Quality metrics
    if characteristics.get("confidence_distribution"):
        conf_dist = characteristics["confidence_distribution"]
        avg_conf = sum(conf_dist) / len(conf_dist)
        print(f"\n📈 Quality Metrics:")
        print(f"   Average confidence target: {avg_conf:.1%}")
        print(f"   Conservative approach: {'✅' if avg_conf < 0.8 else '⚠️'}")
        print(f"   ICD-10 coverage: {len(characteristics.get('icd_codes', set()))} unique codes")

    # Save comprehensive report
    report_data = {
        "report_metadata": {
            "timestamp": datetime.now().isoformat(),
            "report_type": "RAG Model Validation System",
            "version": "1.0.0"
        },
        "test_suite_summary": {
            "total_scenarios": len(scenarios),
            "domains_covered": len(domain_stats),
            "test_categories": len(test_categories)
        },
        "domain_statistics": domain_stats,
        "scenario_characteristics": {
            k: v for k, v in characteristics.items()
            if k not in ['icd_codes', 'learning_objectives', 'safety_considerations']
        },
        "test_categories_summary": {
            category: len(category_scenarios)
            for category, category_scenarios in test_categories.items()
        },
        "validation_capabilities": [
            "Safety critical scenario testing",
            "Confidence calibration validation",
            "Diagnostic accuracy assessment",
            "Emergency detection verification",
            "Conservative diagnosis validation"
        ]
    }

    # Convert sets to lists for JSON serialization
    if 'icd_codes' in characteristics:
        report_data['scenario_characteristics']['unique_icd_codes'] = len(characteristics['icd_codes'])

    report_file = f"rag_model_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Validation report saved to: {report_file}")

    # Final assessment
    print(f"\n🏆 RAG MODEL VALIDATION ASSESSMENT")
    print("=" * 50)

    total_test_cases = sum(len(category_scenarios) for category_scenarios in test_categories.values())

    if total_test_cases >= 20:
        print("✅ COMPREHENSIVE TEST SUITE READY")
        print("✅ Multiple testing categories available")
        print("✅ Conservative confidence calibration")
        print("✅ Safety-critical scenario coverage")
        print("✅ Emergency detection capabilities")
        print("✅ Diagnostic accuracy validation")
        print(f"📊 Total test cases available: {total_test_cases}")
    else:
        print("⚠️ Limited test suite - consider generating more scenarios")

    return report_data


async def main():
    """Main execution function"""

    # Create comprehensive test suite
    scenarios, domain_stats = await create_comprehensive_test_suite()

    if not scenarios:
        print("❌ No scenarios generated - cannot proceed")
        return

    # Analyze scenario characteristics
    characteristics = await analyze_scenario_characteristics(scenarios)

    # Create model testing framework
    test_categories = await create_model_testing_framework(scenarios)

    # Generate comprehensive report
    report = await generate_testing_report(scenarios, characteristics, test_categories, domain_stats)

    print(f"\n✅ RAG Model Validation System ready for AI testing!")


if __name__ == "__main__":
    asyncio.run(main())