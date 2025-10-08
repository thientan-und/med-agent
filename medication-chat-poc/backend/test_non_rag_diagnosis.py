#!/usr/bin/env python3
"""
Test Non-RAG Diagnosis Capability
=================================

Tests the agentic AI's ability to diagnose conditions that are NOT included
in the RAG knowledge base, using only the underlying LLM's medical knowledge.
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import services
from app.services.rag_scenario_generator import rag_scenario_generator
from app.services.medical_ai_service import MedicalAIService
from app.services.rag_few_shot_service import RAGFewShotService
from precision_critic_validator import PrecisionCritic


class NonRAGDiagnosisTest:
    """Test AI's capability to diagnose conditions not in RAG knowledge base"""

    def __init__(self):
        self.medical_ai_service = None
        self.rag_service = None
        self.precision_critic = None
        self.rag_conditions = set()
        self.test_scenarios = []

    async def initialize(self):
        """Initialize services and analyze RAG knowledge base"""
        print("🧪 NON-RAG DIAGNOSIS CAPABILITY TEST")
        print("=" * 50)

        # Initialize services
        await rag_scenario_generator.initialize()

        self.medical_ai_service = MedicalAIService()
        await self.medical_ai_service.initialize()
        print("✅ Medical AI Service initialized")

        self.rag_service = RAGFewShotService()
        await self.rag_service.initialize()
        print("✅ RAG Service initialized")

        self.precision_critic = PrecisionCritic()
        print("✅ Precision Critic loaded")

        # Analyze RAG knowledge base
        await self._analyze_rag_knowledge()

        # Create test scenarios
        self._create_non_rag_test_scenarios()
        print(f"✅ Created {len(self.test_scenarios)} non-RAG test scenarios")

    async def _analyze_rag_knowledge(self):
        """Analyze what conditions are included in RAG knowledge base"""
        print("\n📊 ANALYZING RAG KNOWLEDGE BASE")
        print("-" * 40)

        # Get RAG knowledge items
        knowledge_items = self.rag_service.knowledge_base
        print(f"📚 Total RAG knowledge items: {len(knowledge_items)}")

        # Extract conditions from RAG
        rag_conditions = set()
        for item in knowledge_items:
            # Get both English and Thai names
            name_en = item.name_en.lower() if item.name_en else ''
            name_th = item.name_th.lower() if item.name_th else ''
            if name_en:
                rag_conditions.add(name_en)
            if name_th:
                rag_conditions.add(name_th)

        self.rag_conditions = rag_conditions

        print(f"🎯 Conditions in RAG knowledge base:")
        for i, condition in enumerate(sorted(rag_conditions), 1):
            print(f"   {i:2d}. {condition}")

        print(f"\n📈 RAG Knowledge Summary:")
        print(f"   Total unique conditions: {len(rag_conditions)}")

    def _create_non_rag_test_scenarios(self):
        """Create test scenarios with conditions NOT in RAG knowledge base"""

        # Test conditions that are likely NOT in the current RAG knowledge
        non_rag_scenarios = [
            # RARE GENETIC CONDITIONS
            {
                "id": "NON_RAG_001",
                "condition": "Ehlers-Danlos Syndrome",
                "thai_symptoms": "ข้อต่อหลวมผิดปกติ ผิวหนังยืดหยุ่นมาก ฟกช้ำง่าย ปวดข้อเรื้อรัง",
                "category": "rare_genetic",
                "expected_in_rag": False,
                "description": "Rare connective tissue disorder"
            },
            {
                "id": "NON_RAG_002",
                "condition": "Marfan Syndrome",
                "thai_symptoms": "ตัวสูงผอม นิ้วยาวผิดปกติ ปัญหาสายตา หัวใจเต้นผิดจังหวะ",
                "category": "rare_genetic",
                "expected_in_rag": False,
                "description": "Genetic disorder affecting connective tissue"
            },

            # RARE NEUROLOGICAL CONDITIONS
            {
                "id": "NON_RAG_003",
                "condition": "Narcolepsy",
                "thai_symptoms": "หลับในเวลาที่ไม่เหมาะสม ล้มลงเมื่อมีอารมณ์รุนแรง เห็นภาพหลอน",
                "category": "rare_neurological",
                "expected_in_rag": False,
                "description": "Neurological disorder affecting sleep-wake cycles"
            },
            {
                "id": "NON_RAG_004",
                "condition": "Trigeminal Neuralgia",
                "thai_symptoms": "ปวดใบหน้ารุนแรงมากเป็นชู่ ๆ เหมือนไฟฟ้าช็อต กระตุ้นด้วยการสัมผัสเบา ๆ",
                "category": "rare_neurological",
                "expected_in_rag": False,
                "description": "Severe facial nerve pain"
            },

            # RARE ENDOCRINE CONDITIONS
            {
                "id": "NON_RAG_005",
                "condition": "Addison's Disease",
                "thai_symptoms": "เหนื่อยล้าอย่างรุนแรง ผิวดำขึ้น น้ำหนักลด อยากเค็ม",
                "category": "rare_endocrine",
                "expected_in_rag": False,
                "description": "Adrenal insufficiency"
            },
            {
                "id": "NON_RAG_006",
                "condition": "Cushing's Syndrome",
                "thai_symptoms": "หน้าบวมกลม ท้องใหญ่ แต่แขนขาผอม รอยแตกลายสีม่วง น้ำตาลสูง",
                "category": "rare_endocrine",
                "expected_in_rag": False,
                "description": "Excess cortisol production"
            },

            # RARE AUTOIMMUNE CONDITIONS
            {
                "id": "NON_RAG_007",
                "condition": "Sjögren's Syndrome",
                "thai_symptoms": "ตาแห้งมาก ปากแห้ง กลืนลำบาก ข้อบวมปวด",
                "category": "rare_autoimmune",
                "expected_in_rag": False,
                "description": "Autoimmune disorder affecting moisture-producing glands"
            },
            {
                "id": "NON_RAG_008",
                "condition": "Myasthenia Gravis",
                "thai_symptoms": "กล้ามเนื้ออ่อนแรงเมื่อใช้นาน เปลือกตาตก เคี้ยวลำบาก พูดไม่ชัด",
                "category": "rare_autoimmune",
                "expected_in_rag": False,
                "description": "Neuromuscular autoimmune disorder"
            },

            # TROPICAL/REGIONAL DISEASES (may or may not be in RAG)
            {
                "id": "NON_RAG_009",
                "condition": "Melioidosis",
                "thai_symptoms": "ไข้สูงขึ้นลง ไอเลือด ปวดหน้าอก โรคในดินภาคตะวันออกเฉียงเหนือ",
                "category": "tropical_regional",
                "expected_in_rag": False,
                "description": "Bacterial infection from soil"
            },
            {
                "id": "NON_RAG_010",
                "condition": "Glanders",
                "thai_symptoms": "ไข้ ไอเลือด แผลหนอง ติดจากม้า โรคหายาก",
                "category": "tropical_regional",
                "expected_in_rag": False,
                "description": "Rare bacterial infection from horses"
            },

            # OCCUPATIONAL/ENVIRONMENTAL DISEASES
            {
                "id": "NON_RAG_011",
                "condition": "Silicosis",
                "thai_symptoms": "ไอเรื้อรัง หายใจลำบาก ทำงานเหมือง สัมผัสฝุ่นหิน",
                "category": "occupational",
                "expected_in_rag": False,
                "description": "Lung disease from silica dust exposure"
            },
            {
                "id": "NON_RAG_012",
                "condition": "Berylliosis",
                "thai_symptoms": "หายใจลำบากค่อยเป็นค่อยไป ไอแห้ง ทำงานโรงงาน สัมผัสโลหะ",
                "category": "occupational",
                "expected_in_rag": False,
                "description": "Chronic lung disease from beryllium exposure"
            }
        ]

        self.test_scenarios = non_rag_scenarios

    async def test_non_rag_diagnosis_capability(self) -> List[Dict[str, Any]]:
        """Test AI's ability to diagnose conditions not in RAG"""

        print(f"\n🧬 TESTING NON-RAG DIAGNOSIS CAPABILITY")
        print("=" * 60)

        results = []

        for i, scenario in enumerate(self.test_scenarios, 1):
            print(f"\n" + "="*70)
            print(f"🔬 TESTING SCENARIO {i}/{len(self.test_scenarios)}: {scenario['id']}")
            print(f"🎯 Target Condition: {scenario['condition']}")
            print(f"📂 Category: {scenario['category']}")
            print(f"💬 Thai Symptoms: {scenario['thai_symptoms']}")
            print(f"❓ Expected in RAG: {scenario['expected_in_rag']}")
            print("="*70)

            try:
                # Check if condition is actually in RAG knowledge base
                condition_in_rag = any(scenario['condition'].lower() in rag_condition
                                     for rag_condition in self.rag_conditions)

                print(f"📊 Condition '{scenario['condition']}' found in RAG: {condition_in_rag}")

                # Call Medical AI Service
                print(f"⏱️  Calling Medical AI Service...")

                message = f"{scenario['thai_symptoms']} (อาการผิดปกติที่อาจเป็นโรคหายาก)"

                api_response = await self.medical_ai_service.assess_common_illness(
                    message=message
                )

                print(f"✅ Response received")
                print(f"\n📤 AI RESPONSE:")
                print("-" * 40)
                print(json.dumps(api_response, indent=2, ensure_ascii=False))
                print("-" * 40)

                # Analyze response
                primary_diagnosis = api_response.get('primary_diagnosis', {})
                diagnosed_condition = primary_diagnosis.get('english_name', 'Unknown')
                thai_name = primary_diagnosis.get('thai_name', 'Unknown')
                confidence = primary_diagnosis.get('confidence', 0)

                # Check if AI diagnosed the correct condition
                correct_diagnosis = (scenario['condition'].lower() in diagnosed_condition.lower() or
                                   diagnosed_condition.lower() in scenario['condition'].lower())

                print(f"\n🎯 DIAGNOSIS ANALYSIS:")
                print(f"   Target Condition: {scenario['condition']}")
                print(f"   AI Diagnosed: {diagnosed_condition}")
                print(f"   Thai Name: {thai_name}")
                print(f"   Confidence: {confidence}")
                print(f"   Correct Match: {'✅' if correct_diagnosis else '❌'}")
                print(f"   Condition in RAG: {'✅' if condition_in_rag else '❌'}")

                # Test with Precision Critic
                critic_result = self.precision_critic.validate_medical_output(
                    scenario['thai_symptoms'],
                    json.dumps(api_response),
                    f"Professional assessment needed for {scenario['condition']}"
                )

                critic_verdict = critic_result["overall_verdict"]["status"]
                print(f"   Critic Verdict: {critic_verdict}")

                result = {
                    "scenario_id": scenario['id'],
                    "target_condition": scenario['condition'],
                    "category": scenario['category'],
                    "thai_symptoms": scenario['thai_symptoms'],
                    "expected_in_rag": scenario['expected_in_rag'],
                    "actually_in_rag": condition_in_rag,
                    "ai_response": api_response,
                    "diagnosed_condition": diagnosed_condition,
                    "thai_diagnosis": thai_name,
                    "confidence": confidence,
                    "correct_diagnosis": correct_diagnosis,
                    "critic_verdict": critic_verdict,
                    "timestamp": datetime.now().isoformat()
                }

                results.append(result)

                # Summary for this test
                capability_icon = "🎯" if correct_diagnosis else "❌"
                rag_icon = "📚" if condition_in_rag else "🧬"

                print(f"\n{capability_icon} TEST SUMMARY:")
                print(f"   {rag_icon} RAG Status: {'In RAG' if condition_in_rag else 'NOT in RAG'}")
                print(f"   🎯 Diagnosis Accuracy: {'CORRECT' if correct_diagnosis else 'INCORRECT'}")
                print(f"   🛡️  Safety Assessment: {critic_verdict}")

            except Exception as e:
                print(f"❌ ERROR testing {scenario['id']}: {e}")
                logger.error(f"Non-RAG test error for {scenario['id']}: {e}")

                results.append({
                    "scenario_id": scenario['id'],
                    "target_condition": scenario['condition'],
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })

        return results

    async def analyze_non_rag_capability(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the AI's capability to diagnose non-RAG conditions"""

        print(f"\n📊 NON-RAG DIAGNOSIS CAPABILITY ANALYSIS")
        print("=" * 60)

        successful_tests = [r for r in results if 'error' not in r]
        failed_tests = [r for r in results if 'error' in r]

        print(f"📈 Test Execution:")
        print(f"   Total scenarios: {len(results)}")
        print(f"   Successful: {len(successful_tests)}")
        print(f"   Failed: {len(failed_tests)}")

        if not successful_tests:
            print("❌ No successful tests to analyze")
            return {"error": "No successful tests"}

        # Categorize results
        truly_non_rag = [r for r in successful_tests if not r['actually_in_rag']]
        unexpectedly_in_rag = [r for r in successful_tests if r['actually_in_rag']]

        print(f"\n🧬 RAG KNOWLEDGE STATUS:")
        print(f"   Truly NOT in RAG: {len(truly_non_rag)}")
        print(f"   Unexpectedly IN RAG: {len(unexpectedly_in_rag)}")

        # Analyze diagnosis accuracy for non-RAG conditions
        if truly_non_rag:
            non_rag_correct = len([r for r in truly_non_rag if r['correct_diagnosis']])
            non_rag_accuracy = non_rag_correct / len(truly_non_rag) * 100

            print(f"\n🎯 NON-RAG DIAGNOSIS CAPABILITY:")
            print(f"   Correct diagnoses: {non_rag_correct}/{len(truly_non_rag)} ({non_rag_accuracy:.1f}%)")

            print(f"\n📋 DETAILED NON-RAG RESULTS:")
            for result in truly_non_rag:
                status = "✅ CORRECT" if result['correct_diagnosis'] else "❌ INCORRECT"
                print(f"   {result['scenario_id']}: {result['target_condition']}")
                print(f"      AI Diagnosed: {result['diagnosed_condition']} ({result['confidence']}% confidence)")
                print(f"      Status: {status}")
        else:
            print(f"⚠️  All test conditions were found in RAG knowledge base!")
            non_rag_accuracy = 0

        # Analyze by category
        categories = {}
        for result in successful_tests:
            category = result['category']
            if category not in categories:
                categories[category] = {'total': 0, 'correct': 0, 'non_rag': 0}
            categories[category]['total'] += 1
            if result['correct_diagnosis']:
                categories[category]['correct'] += 1
            if not result['actually_in_rag']:
                categories[category]['non_rag'] += 1

        print(f"\n📊 CATEGORY ANALYSIS:")
        for category, stats in categories.items():
            accuracy = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"   {category}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%) - {stats['non_rag']} truly non-RAG")

        # Overall capability assessment
        if non_rag_accuracy >= 70:
            capability_assessment = "✅ STRONG - Good capability to diagnose conditions outside RAG knowledge"
        elif non_rag_accuracy >= 50:
            capability_assessment = "⚠️ MODERATE - Limited capability for non-RAG conditions"
        elif non_rag_accuracy >= 30:
            capability_assessment = "❌ WEAK - Poor performance on non-RAG conditions"
        else:
            capability_assessment = "❌ VERY WEAK - Relies heavily on RAG knowledge"

        print(f"\n🏆 NON-RAG CAPABILITY ASSESSMENT:")
        print(f"   {capability_assessment}")

        # Save report
        report_data = {
            "report_metadata": {
                "timestamp": datetime.now().isoformat(),
                "report_type": "Non-RAG Diagnosis Capability Test",
                "version": "1.0.0"
            },
            "rag_analysis": {
                "total_rag_conditions": len(self.rag_conditions),
                "rag_conditions": list(self.rag_conditions)
            },
            "test_summary": {
                "total_scenarios": len(results),
                "successful_tests": len(successful_tests),
                "truly_non_rag_count": len(truly_non_rag),
                "non_rag_accuracy": non_rag_accuracy
            },
            "capability_assessment": capability_assessment,
            "category_analysis": categories,
            "detailed_results": results
        }

        report_file = f"non_rag_diagnosis_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"\n📁 Detailed report saved to: {report_file}")

        return report_data

    async def run_complete_non_rag_test(self):
        """Run complete non-RAG diagnosis capability test"""

        try:
            print("🚀 STARTING NON-RAG DIAGNOSIS CAPABILITY TEST")
            print("=" * 70)

            # Initialize and analyze
            await self.initialize()

            # Test non-RAG diagnosis capability
            results = await self.test_non_rag_diagnosis_capability()

            # Analyze capability
            analysis = await self.analyze_non_rag_capability(results)

            print(f"\n✅ NON-RAG DIAGNOSIS TEST COMPLETED")
            print("=" * 50)
            if 'error' not in analysis:
                print(f"🧬 Non-RAG accuracy: {analysis['test_summary']['non_rag_accuracy']:.1f}%")
                print(f"📚 Total RAG conditions: {analysis['rag_analysis']['total_rag_conditions']}")
                print(f"🎯 Capability: {analysis['capability_assessment']}")

            return analysis

        except Exception as e:
            logger.error(f"Non-RAG test error: {e}")
            print(f"❌ Non-RAG test failed: {e}")
            return None


async def main():
    """Main execution function"""
    tester = NonRAGDiagnosisTest()
    await tester.run_complete_non_rag_test()


if __name__ == "__main__":
    asyncio.run(main())