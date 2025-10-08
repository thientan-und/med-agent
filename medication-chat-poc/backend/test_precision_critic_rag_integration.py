#!/usr/bin/env python3
"""
Precision Critic RAG Integration Test
===================================

This script validates that the Precision Critic's knowledge and validation rules
are consistent with the RAG knowledge base, ensuring coherent medical AI validation.
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
from app.services.rag_few_shot_service import rag_few_shot_service
from app.services.rag_scenario_generator import rag_scenario_generator
from precision_critic_validator import PrecisionCritic


class RAGPrecisionCriticIntegrationTest:
    """Test integration between RAG knowledge base and Precision Critic validation"""

    def __init__(self):
        self.precision_critic = PrecisionCritic()
        self.test_results = []
        self.rag_knowledge = []

    async def initialize(self):
        """Initialize RAG services and load knowledge base"""

        print("🔄 RAG-PRECISION CRITIC INTEGRATION TEST")
        print("=" * 60)

        await rag_few_shot_service.initialize()
        await rag_scenario_generator.initialize()

        print("✅ RAG Few-Shot Service initialized")
        print("✅ RAG Scenario Generator initialized")
        print("✅ Precision Critic loaded")
        print()

        # Load RAG knowledge for comparison
        self.rag_knowledge = rag_few_shot_service.knowledge_items

    async def test_rag_scenario_vs_critic_validation(self):
        """Test RAG-generated scenarios against Precision Critic validation"""

        print("🧪 TESTING RAG SCENARIOS VS PRECISION CRITIC")
        print("=" * 50)

        # Generate scenarios from RAG for critical conditions
        test_conditions = [
            "common cold",
            "severe headache",
            "chest pain",
            "abdominal pain",
            "respiratory infection"
        ]

        validation_results = []

        for condition in test_conditions:
            print(f"\n🎯 Testing condition: {condition}")

            try:
                # Generate RAG scenarios
                scenarios = await rag_scenario_generator.generate_few_shot_scenarios(
                    target_condition=condition,
                    count=2
                )

                if not scenarios:
                    print(f"   ❌ No scenarios generated for {condition}")
                    continue

                for i, scenario in enumerate(scenarios, 1):
                    print(f"   📋 Testing scenario {i}...")

                    # Create test case from RAG scenario
                    test_case = self.create_test_case_from_rag_scenario(scenario)

                    if not test_case:
                        print(f"      ⚠️ Could not create test case from scenario")
                        continue

                    # Validate with Precision Critic
                    critic_result = self.precision_critic.validate_medical_output(
                        test_case["symptoms"],
                        test_case["agent_output"],
                        test_case["gold_standard"]
                    )

                    # Analyze consistency
                    consistency_analysis = self.analyze_rag_critic_consistency(
                        scenario, test_case, critic_result)

                    validation_results.append({
                        "condition": condition,
                        "scenario_id": getattr(scenario, 'id', f'{condition}_{i}'),
                        "test_case": test_case,
                        "critic_result": critic_result,
                        "consistency_analysis": consistency_analysis
                    })

                    status = "✅ CONSISTENT" if consistency_analysis["consistent"] else "⚠️ INCONSISTENT"
                    print(f"      {status} - {consistency_analysis['summary']}")

            except Exception as e:
                print(f"   ❌ Error testing {condition}: {e}")

        return validation_results

    def create_test_case_from_rag_scenario(self, scenario) -> Dict[str, Any]:
        """Create a test case from a RAG-generated scenario"""

        try:
            # Extract scenario details
            presenting_symptoms = getattr(scenario, 'presenting_symptoms', {})
            expected_diagnosis = getattr(scenario, 'expected_diagnosis', {})
            patient_profile = getattr(scenario, 'patient_profile', None)
            confidence_target = getattr(scenario, 'confidence_target', 0.7)

            # Get symptoms
            thai_symptoms = presenting_symptoms.get('thai', '') if isinstance(presenting_symptoms, dict) else ''
            if not thai_symptoms:
                return None

            # Get expected diagnosis details
            expected_name = expected_diagnosis.get('name', 'Unknown') if isinstance(expected_diagnosis, dict) else 'Unknown'
            urgency = expected_diagnosis.get('urgency', 'low') if isinstance(expected_diagnosis, dict) else 'low'
            icd_code = expected_diagnosis.get('icd_code', 'Unknown') if isinstance(expected_diagnosis, dict) else 'Unknown'

            # Create mock agent output based on scenario
            agent_output = {
                "primary_diagnosis": {
                    "condition": expected_name,
                    "icd_code": icd_code,
                    "confidence": confidence_target
                },
                "treatment_recommendations": [
                    {
                        "medication": "Conservative management",
                        "evidence": ["guideline: WHO treatment guidelines", "guideline: Medical standard care"]
                    }
                ],
                "safety_certainty": confidence_target + 0.1,  # Slightly higher than confidence
                "diagnostic_coverage": confidence_target + 0.05,
                "overall_confidence": confidence_target,
                "escalation_needed": urgency == 'emergency',
                "final_recommendation": "Emergency consultation" if urgency == 'emergency' else "Conservative management"
            }

            # Create gold standard based on urgency
            if urgency == 'emergency':
                gold_standard = "Abstain: Emergency - Immediate medical evaluation required"
            elif urgency == 'low' and confidence_target > 0.8:
                gold_standard = "Proceed: Conservative management appropriate"
            else:
                gold_standard = "Abstain: Low confidence - Medical consultation recommended"

            return {
                "symptoms": thai_symptoms,
                "agent_output": json.dumps(agent_output),
                "gold_standard": gold_standard,
                "scenario_metadata": {
                    "condition": expected_name,
                    "urgency": urgency,
                    "confidence_target": confidence_target
                }
            }

        except Exception as e:
            logger.error(f"Error creating test case: {e}")
            return None

    def analyze_rag_critic_consistency(self, scenario, test_case: Dict, critic_result: Dict) -> Dict[str, Any]:
        """Analyze consistency between RAG scenario and Precision Critic validation"""

        consistency_issues = []
        consistent = True

        # Get scenario details
        expected_diagnosis = getattr(scenario, 'expected_diagnosis', {})
        urgency = expected_diagnosis.get('urgency', 'low') if isinstance(expected_diagnosis, dict) else 'low'
        confidence_target = getattr(scenario, 'confidence_target', 0.7)

        # Check if critic verdict aligns with RAG scenario expectations
        critic_verdict = critic_result["overall_verdict"]["status"]

        # Expected behavior based on RAG scenario
        if urgency == 'emergency':
            # Emergency scenarios should either pass (if properly escalated) or fail (if not escalated)
            if "PASS" in critic_verdict:
                # Check if escalation was properly handled
                escalation_needed = json.loads(test_case["agent_output"]).get("escalation_needed", False)
                if not escalation_needed:
                    consistency_issues.append("Emergency scenario passed validation without escalation")
                    consistent = False
        elif urgency == 'low' and confidence_target > 0.8:
            # High confidence, low urgency should generally pass
            if "FAIL" in critic_verdict:
                # Check if failure is due to missing guidelines (acceptable) or other issues
                rule_results = critic_result.get("rule_results", {})
                guideline_fail = rule_results.get("A2_GUIDELINE_ADHERENCE", {}).get("status", "").value == "CRITICAL_FAIL"
                if not guideline_fail:
                    consistency_issues.append("High confidence low urgency scenario failed for non-guideline reasons")
                    consistent = False

        # Check knowledge base alignment
        knowledge_alignment = self.check_knowledge_base_alignment(scenario, critic_result)
        if not knowledge_alignment["aligned"]:
            consistency_issues.extend(knowledge_alignment["issues"])
            consistent = False

        # Check safety threshold consistency
        safety_consistency = self.check_safety_threshold_consistency(test_case, critic_result)
        if not safety_consistency["consistent"]:
            consistency_issues.extend(safety_consistency["issues"])
            consistent = False

        return {
            "consistent": consistent,
            "issues": consistency_issues,
            "summary": "RAG scenario and Precision Critic validation are consistent" if consistent else f"Inconsistencies found: {len(consistency_issues)} issues",
            "details": {
                "knowledge_alignment": knowledge_alignment,
                "safety_consistency": safety_consistency
            }
        }

    def check_knowledge_base_alignment(self, scenario, critic_result: Dict) -> Dict[str, Any]:
        """Check if Precision Critic validation aligns with RAG knowledge base"""

        aligned = True
        issues = []

        # Get expected diagnosis from scenario
        expected_diagnosis = getattr(scenario, 'expected_diagnosis', {})
        expected_name = expected_diagnosis.get('name', '') if isinstance(expected_diagnosis, dict) else ''

        # Find matching knowledge in RAG
        matching_knowledge = None
        for knowledge_item in self.rag_knowledge:
            if expected_name.lower() in knowledge_item.get('condition', '').lower():
                matching_knowledge = knowledge_item
                break

        if not matching_knowledge:
            # No direct match found - check if condition is in knowledge base
            issues.append(f"Condition '{expected_name}' not found in RAG knowledge base")
            aligned = False
        else:
            # Check if critic's emergency detection aligns with knowledge base urgency
            kb_urgency = matching_knowledge.get('urgency', 'unknown')
            critic_emergency_detected = any(
                result.get("status", "").value == "CRITICAL_FAIL" and "emergency" in result.get("details", "").lower()
                for result in critic_result.get("rule_results", {}).values()
            )

            if kb_urgency == 'emergency' and not critic_emergency_detected:
                issues.append(f"RAG knowledge indicates emergency for '{expected_name}' but critic didn't detect emergency")
                aligned = False

        return {
            "aligned": aligned,
            "issues": issues,
            "matching_knowledge": matching_knowledge
        }

    def check_safety_threshold_consistency(self, test_case: Dict, critic_result: Dict) -> Dict[str, Any]:
        """Check if safety threshold application is consistent"""

        consistent = True
        issues = []

        try:
            agent_data = json.loads(test_case["agent_output"])
            safety_certainty = agent_data.get("safety_certainty", 0)
            diagnostic_coverage = agent_data.get("diagnostic_coverage", 0)

            # Get critic's abstention logic result
            abstention_result = critic_result.get("rule_results", {}).get("B4_ABSTENTION_LOGIC", {})
            abstention_status = abstention_result.get("status", "").value

            # Check consistency with thresholds
            should_abstain = (safety_certainty < 0.85 or diagnostic_coverage < 0.6)

            if should_abstain and abstention_status == "PASS":
                issues.append(f"Should abstain (safety={safety_certainty:.2f}, coverage={diagnostic_coverage:.2f}) but critic approved")
                consistent = False
            elif not should_abstain and abstention_status == "CRITICAL_FAIL":
                issues.append(f"Should proceed (safety={safety_certainty:.2f}, coverage={diagnostic_coverage:.2f}) but critic failed abstention logic")
                consistent = False

        except Exception as e:
            issues.append(f"Error checking safety thresholds: {e}")
            consistent = False

        return {
            "consistent": consistent,
            "issues": issues
        }

    async def test_knowledge_base_coverage(self):
        """Test how well Precision Critic covers RAG knowledge base scenarios"""

        print(f"\n📚 TESTING KNOWLEDGE BASE COVERAGE")
        print("=" * 50)

        coverage_results = {
            "total_conditions": len(self.rag_knowledge),
            "covered_conditions": 0,
            "emergency_conditions": 0,
            "emergency_covered": 0,
            "coverage_details": []
        }

        for knowledge_item in self.rag_knowledge:
            condition_name = knowledge_item.get('condition', 'Unknown')
            urgency = knowledge_item.get('urgency', 'unknown')

            print(f"📋 Testing: {condition_name} (urgency: {urgency})")

            # Check if Precision Critic has appropriate handling for this condition
            coverage_analysis = self.analyze_condition_coverage(knowledge_item)

            coverage_results["coverage_details"].append({
                "condition": condition_name,
                "urgency": urgency,
                "coverage": coverage_analysis
            })

            if coverage_analysis["covered"]:
                coverage_results["covered_conditions"] += 1

            if urgency == 'emergency':
                coverage_results["emergency_conditions"] += 1
                if coverage_analysis["emergency_handled"]:
                    coverage_results["emergency_covered"] += 1

            status = "✅ COVERED" if coverage_analysis["covered"] else "⚠️ LIMITED"
            print(f"   {status} - {coverage_analysis['summary']}")

        # Calculate coverage rates
        overall_coverage = coverage_results["covered_conditions"] / coverage_results["total_conditions"] * 100
        emergency_coverage = (coverage_results["emergency_covered"] / coverage_results["emergency_conditions"] * 100
                            if coverage_results["emergency_conditions"] > 0 else 100)

        print(f"\n📊 COVERAGE ANALYSIS:")
        print(f"   Overall coverage: {overall_coverage:.1f}% ({coverage_results['covered_conditions']}/{coverage_results['total_conditions']})")
        print(f"   Emergency coverage: {emergency_coverage:.1f}% ({coverage_results['emergency_covered']}/{coverage_results['emergency_conditions']})")

        return coverage_results

    def analyze_condition_coverage(self, knowledge_item: Dict) -> Dict[str, Any]:
        """Analyze how well Precision Critic covers a specific condition from RAG knowledge"""

        condition_name = knowledge_item.get('condition', '').lower()
        urgency = knowledge_item.get('urgency', 'unknown')
        symptoms = knowledge_item.get('symptoms', [])

        covered = True
        emergency_handled = False
        coverage_issues = []

        # Check if condition is in critic's emergency patterns
        if urgency == 'emergency':
            pattern_found = False
            for pattern in self.precision_critic.emergency_patterns:
                if any(symptom.lower() in condition_name for symptom in pattern["symptoms"]):
                    pattern_found = True
                    emergency_handled = True
                    break

            if not pattern_found:
                # Check if symptoms are in emergency keywords
                keyword_found = any(
                    keyword in ' '.join(symptoms).lower()
                    for keyword in self.precision_critic.thai_emergency_keywords
                )
                if keyword_found:
                    emergency_handled = True
                else:
                    coverage_issues.append(f"Emergency condition '{condition_name}' not in critic's emergency detection patterns")
                    covered = False

        # Check for dangerous misdiagnosis patterns
        if 'appendicitis' in condition_name or 'ไส้ติ่ง' in condition_name:
            # Should be covered by appendicitis pattern
            emergency_handled = True
        elif 'meningitis' in condition_name or 'เยื่อหุ้มสมอง' in condition_name:
            # Should be covered by meningitis pattern
            emergency_handled = True

        return {
            "covered": covered,
            "emergency_handled": emergency_handled,
            "issues": coverage_issues,
            "summary": "Well covered" if covered else f"Coverage gaps: {len(coverage_issues)} issues"
        }

    async def generate_comprehensive_report(self, validation_results: List, coverage_results: Dict):
        """Generate comprehensive integration test report"""

        print(f"\n📊 COMPREHENSIVE INTEGRATION REPORT")
        print("=" * 60)

        # Analyze validation results
        total_validations = len(validation_results)
        consistent_validations = len([r for r in validation_results if r["consistency_analysis"]["consistent"]])
        consistency_rate = consistent_validations / total_validations * 100 if total_validations > 0 else 0

        print(f"🔄 Validation Consistency:")
        print(f"   Total tests: {total_validations}")
        print(f"   Consistent: {consistent_validations}/{total_validations} ({consistency_rate:.1f}%)")

        # Analyze coverage
        overall_coverage = coverage_results["covered_conditions"] / coverage_results["total_conditions"] * 100
        emergency_coverage = (coverage_results["emergency_covered"] / coverage_results["emergency_conditions"] * 100
                            if coverage_results["emergency_conditions"] > 0 else 100)

        print(f"\n📚 Knowledge Base Coverage:")
        print(f"   Overall coverage: {overall_coverage:.1f}%")
        print(f"   Emergency coverage: {emergency_coverage:.1f}%")

        # Integration assessment
        print(f"\n🎯 Integration Assessment:")
        if consistency_rate >= 90 and overall_coverage >= 80:
            integration_status = "✅ EXCELLENT - High consistency and coverage"
        elif consistency_rate >= 80 and overall_coverage >= 70:
            integration_status = "✅ GOOD - Adequate consistency and coverage"
        elif consistency_rate >= 70 and overall_coverage >= 60:
            integration_status = "⚠️ FAIR - Some gaps in consistency or coverage"
        else:
            integration_status = "❌ POOR - Significant gaps requiring attention"

        print(f"   {integration_status}")

        # Save comprehensive report
        report_data = {
            "report_metadata": {
                "timestamp": datetime.now().isoformat(),
                "report_type": "RAG-Precision Critic Integration Test",
                "version": "1.0.0"
            },
            "summary": {
                "total_validations": total_validations,
                "consistent_validations": consistent_validations,
                "consistency_rate": consistency_rate,
                "overall_coverage": overall_coverage,
                "emergency_coverage": emergency_coverage,
                "integration_status": integration_status
            },
            "validation_results": validation_results,
            "coverage_results": coverage_results
        }

        report_file = f"rag_precision_critic_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"\n📁 Integration report saved to: {report_file}")

        return report_data

    async def run_complete_integration_test(self):
        """Run the complete RAG-Precision Critic integration test"""

        try:
            await self.initialize()

            # Test RAG scenarios vs Critic validation
            validation_results = await self.test_rag_scenario_vs_critic_validation()

            # Test knowledge base coverage
            coverage_results = await self.test_knowledge_base_coverage()

            # Generate comprehensive report
            report = await self.generate_comprehensive_report(validation_results, coverage_results)

            print(f"\n✅ RAG-PRECISION CRITIC INTEGRATION TEST COMPLETED")
            print("=" * 60)
            print(f"🎯 {report['summary']['integration_status']}")

        except Exception as e:
            print(f"❌ Integration test failed: {e}")


async def main():
    """Main execution function"""

    integration_test = RAGPrecisionCriticIntegrationTest()
    await integration_test.run_complete_integration_test()


if __name__ == "__main__":
    asyncio.run(main())