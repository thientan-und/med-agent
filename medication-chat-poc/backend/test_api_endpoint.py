#!/usr/bin/env python3
"""
Test API Endpoint Directly
===========================
Test the message through the actual API endpoint
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add project root to path
sys.path.append('/home/naiplawan/Desktop/Unixdev/medical-chat-app/backend')

from app.services.medical_ai_service import MedicalAIService

async def test_api_endpoint():
    """Test the message through the medical AI service directly"""

    print("🚀 Testing API Endpoint with Real Message...")

    # Initialize service
    medical_ai_service = MedicalAIService()
    await medical_ai_service.initialize()
    print("✅ Medical AI Service initialized")

    # Your exact message
    message = "ฉันอายุ 28 สูง 170 หนัก 65 ไม่มีประวัติโรคประจำตัว การแพ้ยา แพ้อาหาร วันนี้ทำงานหนัก แล้วปวดหัว ปวดตา ตอนนี้ฉันเป็นโรคอะไร"

    print(f"\n📝 Testing Message:")
    print(f"   {message}")
    print()

    # Call the API
    result = await medical_ai_service.assess_common_illness(message=message)

    print("🎯 DIAGNOSIS RESULT:")
    print("=" * 50)

    if result and result.get('primary_diagnosis'):
        diagnosis = result['primary_diagnosis']
        print(f"🏥 Primary Diagnosis:")
        print(f"   English: {diagnosis['english_name']}")
        print(f"   Thai: {diagnosis['thai_name']}")
        print(f"   ICD Code: {diagnosis.get('icd_code', 'N/A')}")
        print(f"   Confidence: {diagnosis['confidence']}%")
        print(f"   Category: {diagnosis.get('category', 'N/A')}")

        if diagnosis.get('matched_keywords'):
            print(f"   Matched Keywords: {', '.join(diagnosis['matched_keywords'])}")

        print(f"\n🧠 Analysis:")
        print(f"   Context Considered: {result.get('context_considered', False)}")
        print(f"   Reasoning: {result.get('reasoning', 'N/A')}")

        # Show differential diagnoses
        if result.get('differential_diagnoses'):
            print(f"\n🔍 Differential Diagnoses:")
            for i, diff in enumerate(result['differential_diagnoses'][:3], 1):
                print(f"   {i}. {diff['english_name']} ({diff.get('confidence', 'N/A')}%)")

    else:
        print("❌ No diagnosis found")
        print(f"Context Considered: {result.get('context_considered', False) if result else False}")

    print("\n" + "=" * 50)

    # Clinical assessment
    print("\n🩺 CLINICAL ASSESSMENT:")
    print("For a 28-year-old healthy patient with headache and eye pain after hard work:")
    print("📍 Most likely causes:")
    print("   1. Tension headache (from work stress)")
    print("   2. Eye strain (from computer work)")
    print("   3. Dehydration (from physical work)")
    print("   4. Muscle tension (neck/shoulder strain)")

    if result and result.get('primary_diagnosis'):
        ai_diagnosis = result['primary_diagnosis']['english_name']
        print(f"\n🤖 AI Diagnosis: {ai_diagnosis}")

        # Check appropriateness
        appropriate_terms = ['tension', 'headache', 'strain', 'stress', 'fatigue', 'dehydration']
        is_appropriate = any(term in ai_diagnosis.lower() for term in appropriate_terms)

        print(f"✅ Clinical Appropriateness: {'GOOD' if is_appropriate else 'NEEDS IMPROVEMENT'}")

        if ai_diagnosis == "Allergic Reaction":
            print("💡 Note: AI focused on allergy keywords in message.")
            print("   For better results, emphasize work-related symptoms.")

    # Save result
    report = {
        "timestamp": datetime.now().isoformat(),
        "message": message,
        "result": result,
        "clinical_notes": {
            "patient_age": 28,
            "context": "healthy, work-related symptoms",
            "expected_diagnoses": ["tension headache", "eye strain", "dehydration"],
            "ai_diagnosis": result.get('primary_diagnosis', {}).get('english_name') if result else None
        }
    }

    filename = f"api_endpoint_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n📁 Results saved to: {filename}")

if __name__ == "__main__":
    asyncio.run(test_api_endpoint())