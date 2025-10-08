#!/usr/bin/env python3
"""
Test Context Integration Fix
============================
Quick test to verify context-aware diagnosis improvements
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add project root to path
sys.path.append('/home/naiplawan/Desktop/Unixdev/medical-chat-app/backend')

from app.services.medical_ai_service import MedicalAIService

class ContextIntegrationTester:
    def __init__(self):
        self.medical_ai_service = None
        self.test_results = []

    async def initialize(self):
        """Initialize medical AI service"""
        print("🚀 Initializing Context Integration Test...")
        self.medical_ai_service = MedicalAIService()
        await self.medical_ai_service.initialize()
        print("✅ Medical AI Service initialized")

    async def test_context_integration(self):
        """Test the context integration fix"""

        print("\n🧪 TESTING CONTEXT INTEGRATION FIX")
        print("=" * 60)

        # Test Case 1: Young Athlete vs Elderly Diabetic (Chest Pain)
        await self._test_chest_pain_scenarios()

        # Test Case 2: Migraine Patient vs Hypertensive (Headache)
        await self._test_headache_scenarios()

        # Generate test report
        self._generate_test_report()

    async def _test_chest_pain_scenarios(self):
        """Test chest pain with different patient contexts"""

        print("\n📋 TEST 1: Chest Pain - Context Differentiation")
        print("-" * 50)

        symptoms = "ปวดอกหลังออกกำลังกาย หายใจลำบาก เหนื่อย"

        # Scenario A: Young Athlete
        context_a = "male อายุ 25 ปี | อาชีพ: นักกีฬาวิ่งมาราธอน | ประวัติ: สุขภาพดี"
        message_a = f"{symptoms} | ผู้ป่วย: {context_a}"

        print(f"🔬 Testing Young Athlete Context:")
        print(f"   Message: {message_a}")

        result_a = await self.medical_ai_service.assess_common_illness(
            message=symptoms,
            patient_info=context_a  # Pass context as separate parameter
        )

        print(f"   Diagnosis: {result_a['primary_diagnosis']['english_name']}")
        print(f"   Context Considered: {result_a.get('context_considered', False)}")

        # Scenario B: Elderly Diabetic
        context_b = "male อายุ 65 ปี | อาชีพ: ข้าราชการบำนาญ | ประวัติ: เบาหวาน 10 ปี, ความดันสูง"
        message_b = f"{symptoms} | ผู้ป่วย: {context_b}"

        print(f"\n🔬 Testing Elderly Diabetic Context:")
        print(f"   Message: {message_b}")

        result_b = await self.medical_ai_service.assess_common_illness(
            message=symptoms,
            patient_info=context_b  # Pass context as separate parameter
        )

        print(f"   Diagnosis: {result_b['primary_diagnosis']['english_name']}")
        print(f"   Context Considered: {result_b.get('context_considered', False)}")

        # Compare results
        same_diagnosis = (
            result_a['primary_diagnosis']['english_name'] ==
            result_b['primary_diagnosis']['english_name']
        )

        print(f"\n📊 COMPARISON:")
        print(f"   Same Diagnosis: {'❌ FAILED' if same_diagnosis else '✅ IMPROVED'}")
        print(f"   Context Integration: {'✅ WORKING' if result_a.get('context_considered') else '❌ NOT WORKING'}")

        self.test_results.append({
            "test": "chest_pain_context",
            "young_athlete": result_a,
            "elderly_diabetic": result_b,
            "same_diagnosis": same_diagnosis,
            "context_working": result_a.get('context_considered', False)
        })

    async def _test_headache_scenarios(self):
        """Test headache with migraine vs hypertensive context"""

        print("\n📋 TEST 2: Headache - Context Differentiation")
        print("-" * 50)

        symptoms = "ปวดหัวรุนแรง ตาพร่า คลื่นไส้"

        # Scenario A: Migraine Patient
        context_a = "female อายุ 30 ปี | ประวัติ: ไมเกรน, ปวดหัวข้างเดียวบ่อย"
        message_a = f"{symptoms} | ผู้ป่วย: {context_a}"

        print(f"🔬 Testing Migraine Patient Context:")
        print(f"   Message: {message_a}")

        result_a = await self.medical_ai_service.assess_common_illness(
            message=symptoms,
            patient_info=context_a  # Pass context as separate parameter
        )

        print(f"   Diagnosis: {result_a['primary_diagnosis']['english_name']}")
        print(f"   Context Considered: {result_a.get('context_considered', False)}")

        # Scenario B: Hypertensive Patient
        context_b = "female อายุ 55 ปี | ประวัติ: ความดันสูง | ยาที่ใช้: หยุดยาความดัน 3 วัน"
        message_b = f"{symptoms} | ผู้ป่วย: {context_b}"

        print(f"\n🔬 Testing Hypertensive Patient Context:")
        print(f"   Message: {message_b}")

        result_b = await self.medical_ai_service.assess_common_illness(
            message=symptoms,
            patient_info=context_b  # Pass context as separate parameter
        )

        print(f"   Diagnosis: {result_b['primary_diagnosis']['english_name']}")
        print(f"   Context Considered: {result_b.get('context_considered', False)}")

        # Compare results
        same_diagnosis = (
            result_a['primary_diagnosis']['english_name'] ==
            result_b['primary_diagnosis']['english_name']
        )

        print(f"\n📊 COMPARISON:")
        print(f"   Same Diagnosis: {'❌ STILL SAME' if same_diagnosis else '✅ DIFFERENTIATED'}")
        print(f"   Context Integration: {'✅ WORKING' if result_a.get('context_considered') else '❌ NOT WORKING'}")

        self.test_results.append({
            "test": "headache_context",
            "migraine_patient": result_a,
            "hypertensive_patient": result_b,
            "same_diagnosis": same_diagnosis,
            "context_working": result_a.get('context_considered', False)
        })

    def _generate_test_report(self):
        """Generate comprehensive test report"""

        print("\n" + "=" * 60)
        print("🏆 CONTEXT INTEGRATION TEST RESULTS")
        print("=" * 60)

        total_tests = len(self.test_results)
        context_working_count = sum(1 for r in self.test_results if r.get('context_working', False))
        differentiated_count = sum(1 for r in self.test_results if not r.get('same_diagnosis', True))

        print(f"📊 Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Context Integration Working: {context_working_count}/{total_tests} ({context_working_count/total_tests*100:.1f}%)")
        print(f"   Diagnosis Differentiation: {differentiated_count}/{total_tests} ({differentiated_count/total_tests*100:.1f}%)")

        if context_working_count == total_tests:
            print(f"\n✅ SUCCESS: Context integration is working!")
        else:
            print(f"\n❌ ISSUES: Context integration needs more work")

        if differentiated_count > 0:
            print(f"✅ IMPROVEMENT: Some diagnosis differentiation achieved")
        else:
            print(f"❌ NO IMPROVEMENT: Still same diagnoses regardless of context")

        # Save detailed results
        report_file = f"context_integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_tests": total_tests,
                    "context_working_rate": context_working_count/total_tests*100,
                    "differentiation_rate": differentiated_count/total_tests*100
                },
                "detailed_results": self.test_results
            }, f, ensure_ascii=False, indent=2)

        print(f"\n📁 Detailed results saved to: {report_file}")

async def main():
    """Run context integration test"""
    tester = ContextIntegrationTester()

    try:
        await tester.initialize()
        await tester.test_context_integration()

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())