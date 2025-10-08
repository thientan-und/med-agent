#!/usr/bin/env python3
"""
Test Real API Chat with Context
================================
Test the context-aware diagnosis with real Thai patient message
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add project root to path
sys.path.append('/home/naiplawan/Desktop/Unixdev/medical-chat-app/backend')

from app.services.medical_ai_service import MedicalAIService

class RealAPIChatTester:
    def __init__(self):
        self.medical_ai_service = None

    async def initialize(self):
        """Initialize medical AI service"""
        print("🚀 Testing Real API Chat with Context Integration...")
        self.medical_ai_service = MedicalAIService()
        await self.medical_ai_service.initialize()
        print("✅ Medical AI Service initialized")

    async def test_real_chat_message(self):
        """Test with the real Thai patient message"""

        print("\n🧪 TESTING REAL API CHAT MESSAGE")
        print("=" * 60)

        # Real user message
        full_message = "ฉันอายุ 28 สูง 170 หนัก 65 ไม่มีประวัติโรคประจำตัว การแพ้ยา แพ้อาหาร วันนี้ทำงานหนัก แล้วปวดหัว ปวดตา ตอนนี้ฉันเป็นโรคอะไร"

        print(f"📝 Full Message: {full_message}")
        print()

        # Parse context from message (simulate what frontend would do)
        symptoms = "ปวดหัว ปวดตา"
        patient_context = "อายุ 28 สูง 170 หนัก 65 ไม่มีประวัติโรคประจำตัว ไม่แพ้ยา ไม่แพ้อาหาร ทำงานหนัก"

        print(f"🔍 Extracted Symptoms: {symptoms}")
        print(f"👤 Patient Context: {patient_context}")
        print()

        # Test 1: Without context (old way)
        print("📋 TEST 1: Without Patient Context (Old Way)")
        print("-" * 40)

        result_without_context = await self.medical_ai_service.assess_common_illness(
            message=symptoms
            # No patient_info parameter
        )

        if result_without_context.get('primary_diagnosis'):
            print(f"   Diagnosis: {result_without_context['primary_diagnosis']['english_name']}")
            print(f"   Thai Name: {result_without_context['primary_diagnosis']['thai_name']}")
            print(f"   Confidence: {result_without_context['primary_diagnosis']['confidence']}")
        else:
            print(f"   Diagnosis: ❌ No diagnosis found")
        print(f"   Context Considered: {result_without_context.get('context_considered', False)}")

        # Test 2: With context (new way)
        print("\n📋 TEST 2: With Patient Context (New Way)")
        print("-" * 40)

        result_with_context = await self.medical_ai_service.assess_common_illness(
            message=symptoms,
            patient_info=patient_context
        )

        if result_with_context.get('primary_diagnosis'):
            print(f"   Diagnosis: {result_with_context['primary_diagnosis']['english_name']}")
            print(f"   Thai Name: {result_with_context['primary_diagnosis']['thai_name']}")
            print(f"   Confidence: {result_with_context['primary_diagnosis']['confidence']}")
        else:
            print(f"   Diagnosis: ❌ No diagnosis found")
        print(f"   Context Considered: {result_with_context.get('context_considered', False)}")

        # Test 3: Full message analysis (most realistic)
        print("\n📋 TEST 3: Full Message Analysis (Most Realistic)")
        print("-" * 40)

        result_full_message = await self.medical_ai_service.assess_common_illness(
            message=full_message,
            patient_info=patient_context
        )

        if result_full_message.get('primary_diagnosis'):
            print(f"   Diagnosis: {result_full_message['primary_diagnosis']['english_name']}")
            print(f"   Thai Name: {result_full_message['primary_diagnosis']['thai_name']}")
            print(f"   Confidence: {result_full_message['primary_diagnosis']['confidence']}")
        else:
            print(f"   Diagnosis: ❌ No diagnosis found")
        print(f"   Context Considered: {result_full_message.get('context_considered', False)}")

        # Show differential diagnoses if available
        if result_full_message.get('differential_diagnoses'):
            print(f"   Differential Diagnoses:")
            for i, diff in enumerate(result_full_message['differential_diagnoses'][:3], 1):
                print(f"      {i}. {diff['english_name']} ({diff.get('confidence', 'N/A')}%)")

        # Analysis
        print("\n📊 CONTEXT INTEGRATION ANALYSIS")
        print("=" * 60)

        # Check if diagnoses are different (handle None cases)
        diagnosis_1 = (result_without_context or {}).get('primary_diagnosis', {})
        diagnosis_1 = (diagnosis_1 or {}).get('english_name', 'None')

        diagnosis_2 = (result_with_context or {}).get('primary_diagnosis', {})
        diagnosis_2 = (diagnosis_2 or {}).get('english_name', 'None')

        diagnosis_3 = (result_full_message or {}).get('primary_diagnosis', {})
        diagnosis_3 = (diagnosis_3 or {}).get('english_name', 'None')

        same_diagnosis = (diagnosis_1 == diagnosis_2 == diagnosis_3)

        context_working = result_with_context.get('context_considered', False)

        print(f"🔍 Context Integration Status:")
        print(f"   Working: {'✅ YES' if context_working else '❌ NO'}")
        print(f"   Same diagnosis across all tests: {'❌ NO DIFFERENTIATION' if same_diagnosis else '✅ CONTEXT EFFECT'}")

        # Context-specific analysis for this case
        print(f"\n🧠 Clinical Context Analysis:")
        print(f"   Patient: 28-year-old, healthy, no medical history")
        print(f"   Symptoms: Headache + eye pain after hard work")
        print(f"   Expected: Tension headache, eye strain, dehydration")
        print(f"   Emergency Risk: LOW (young, healthy, work-related)")

        # Check if emergency escalation is appropriate
        primary_diagnosis = result_full_message.get('primary_diagnosis', {})
        diagnosis_name = primary_diagnosis.get('english_name', 'None')
        is_emergency = "Emergency" in diagnosis_name
        print(f"   Emergency Escalation: {'⚠️ OVER-CONSERVATIVE' if is_emergency else '✅ APPROPRIATE'}")

        # Context consideration for young healthy worker
        appropriate_for_context = any(term in diagnosis_name.lower()
                                    for term in ['tension', 'headache', 'strain', 'stress', 'fatigue'])
        print(f"   Context Appropriateness: {'✅ APPROPRIATE' if appropriate_for_context else '❌ NEEDS IMPROVEMENT'}")

        # Save detailed results
        report = {
            "timestamp": datetime.now().isoformat(),
            "test_message": full_message,
            "extracted_symptoms": symptoms,
            "patient_context": patient_context,
            "results": {
                "without_context": result_without_context,
                "with_context": result_with_context,
                "full_message": result_full_message
            },
            "analysis": {
                "context_working": context_working,
                "same_diagnosis": same_diagnosis,
                "is_emergency": is_emergency,
                "appropriate_for_context": appropriate_for_context
            }
        }

        report_file = f"real_api_chat_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n📁 Detailed results saved to: {report_file}")

        return report

async def main():
    """Run real API chat test"""
    tester = RealAPIChatTester()

    try:
        await tester.initialize()
        await tester.test_real_chat_message()

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())