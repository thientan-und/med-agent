#!/usr/bin/env python3
"""
Test Script for RAG-Enhanced Few-Shot Learning
==============================================

This script tests the new RAG (Retrieval-Augmented Generation) enhanced
few-shot learning system that combines static examples with dynamic
knowledge base retrieval.

Key Features Being Tested:
1. Knowledge base loading from CSV files
2. Dynamic few-shot example generation
3. Safety-first symptom matching
4. Integration with existing few-shot system
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

# Import the services
from app.services.rag_few_shot_service import rag_few_shot_service
from app.services.advanced_few_shot import advanced_few_shot

async def test_rag_few_shot_system():
    """Test the complete RAG-enhanced few-shot learning system"""

    print("🧠 RAG-ENHANCED FEW-SHOT LEARNING TEST")
    print("=" * 60)

    # Test cases that should benefit from RAG
    test_cases = [
        {
            "id": "RAG_001",
            "symptoms": "เป็นไข้เล็กน้อย 38 องศา ไอแห้ง น้ำมูกเขียว มาสองสามวัน",
            "description": "Common cold - should prefer RAG common examples over serious ones",
            "expected_priority": "common_cold_over_serious"
        },
        {
            "id": "RAG_002",
            "symptoms": "ปวดหน้าอก หายใจลำบาก เหงื่อออก",
            "description": "Chest pain - should retrieve cardiac knowledge",
            "expected_priority": "cardiac_evaluation"
        },
        {
            "id": "RAG_003",
            "symptoms": "ไข้ ไอ มีเสมหะเหลือง เมื่อยตัว",
            "description": "Flu symptoms - should match influenza from knowledge base",
            "expected_priority": "influenza_diagnosis"
        },
        {
            "id": "RAG_004",
            "symptoms": "กระหายน้ำ ปัสสาวะบ่อย น้ำหนักลด",
            "description": "Diabetes symptoms - should retrieve endocrine knowledge",
            "expected_priority": "diabetes_screening"
        }
    ]

    print(f"🔬 Testing {len(test_cases)} RAG scenarios...")
    print()

    # Initialize services
    try:
        print("📚 Initializing RAG few-shot service...")
        await rag_few_shot_service.initialize()
        print("✅ RAG service initialized")

        print("🧠 Initializing advanced few-shot learning...")
        # Advanced few-shot should automatically load examples
        print("✅ Advanced few-shot initialized")
        print()

    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return

    # Test each scenario
    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"🧪 Test {i}: {test_case['id']}")
        print(f"   Description: {test_case['description']}")
        print(f"   Symptoms: {test_case['symptoms']}")
        print()

        try:
            # Test RAG-only retrieval
            print("   🔍 Testing RAG knowledge retrieval...")
            rag_examples = await rag_few_shot_service.get_relevant_examples(
                symptoms=test_case['symptoms'],
                max_examples=3
            )

            print(f"   📋 RAG retrieved {len(rag_examples)} examples:")
            for j, example in enumerate(rag_examples):
                confidence = example.confidence_level
                diagnosis = example.diagnosis.get('name', 'Unknown')
                print(f"      {j+1}. {diagnosis} (confidence: {confidence:.2f})")
            print()

            # Test integrated few-shot learning
            print("   🔗 Testing integrated few-shot learning...")
            enhanced_result = await advanced_few_shot.enhanced_diagnosis(
                symptoms=test_case['symptoms'],
                patient_id=f"test_patient_{test_case['id']}"
            )

            if enhanced_result and enhanced_result.get('primary_diagnosis'):
                primary = enhanced_result['primary_diagnosis']
                print(f"   🎯 Primary Diagnosis: {primary.get('english_name', 'Unknown')}")
                print(f"   📊 Confidence: {primary.get('confidence', 0):.2f}")
                print(f"   🏷️  Category: {primary.get('category', 'Unknown')}")
                print(f"   🔑 Keywords: {primary.get('matched_keywords', [])}")

                # Check if RAG was used
                if primary.get('few_shot_source'):
                    print("   ✅ Enhanced with few-shot learning")

            else:
                print("   ⚠️ No diagnosis returned")

            # Store results
            results.append({
                "test_case": test_case,
                "rag_examples": len(rag_examples),
                "enhanced_result": enhanced_result,
                "success": enhanced_result is not None
            })

        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            results.append({
                "test_case": test_case,
                "error": str(e),
                "success": False
            })

        print("-" * 60)
        print()

    # Summary
    print("📈 RAG FEW-SHOT LEARNING TEST SUMMARY")
    print("=" * 60)

    successful_tests = sum(1 for r in results if r.get('success', False))
    print(f"✅ Successful tests: {successful_tests}/{len(test_cases)}")

    # Analyze RAG effectiveness
    total_rag_examples = sum(r.get('rag_examples', 0) for r in results if 'rag_examples' in r)
    print(f"🔍 Total RAG examples retrieved: {total_rag_examples}")

    # Check safety improvements
    print("\n🛡️ SAFETY ANALYSIS:")

    for result in results:
        if result.get('success') and result.get('enhanced_result'):
            test_id = result['test_case']['id']
            primary = result['enhanced_result'].get('primary_diagnosis', {})
            confidence = primary.get('confidence', 0)
            diagnosis = primary.get('english_name', '')

            # Check for dangerous patterns
            if 'RAG_001' in test_id:  # Common cold test
                if any(serious in diagnosis.lower() for serious in ['tuberculosis', 'วัณโรค', 'cancer']):
                    print(f"   ⚠️ {test_id}: Still showing serious diagnosis for cold symptoms")
                else:
                    print(f"   ✅ {test_id}: Correctly avoided serious diagnosis for mild symptoms")

            # Check confidence levels
            if confidence > 0.9:
                print(f"   ⚠️ {test_id}: High confidence ({confidence:.2f}) - check if appropriate")
            elif confidence < 0.3:
                print(f"   ⚠️ {test_id}: Very low confidence ({confidence:.2f}) - may need tuning")
            else:
                print(f"   ✅ {test_id}: Reasonable confidence level ({confidence:.2f})")

    print(f"\n⏰ Test completed at: {datetime.now()}")

    # Save detailed results
    results_file = f"rag_few_shot_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_summary": {
                "total_tests": len(test_cases),
                "successful_tests": successful_tests,
                "total_rag_examples": total_rag_examples,
                "timestamp": datetime.now().isoformat()
            },
            "detailed_results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"📁 Detailed results saved to: {results_file}")

async def test_knowledge_base_loading():
    """Test knowledge base loading specifically"""
    print("\n🔬 KNOWLEDGE BASE LOADING TEST")
    print("=" * 40)

    try:
        await rag_few_shot_service.initialize()

        kb_size = len(rag_few_shot_service.knowledge_base)
        feedback_size = len(rag_few_shot_service.doctor_feedback)
        training_size = len(rag_few_shot_service.training_examples)

        print(f"📚 Knowledge base items: {kb_size}")
        print(f"👩‍⚕️ Doctor feedback entries: {feedback_size}")
        print(f"📖 Training examples: {training_size}")

        if kb_size > 0:
            print("\n📋 Sample knowledge items:")
            for i, item in enumerate(rag_few_shot_service.knowledge_base[:3]):
                print(f"   {i+1}. {item.name_en} | {item.name_th}")
                print(f"      Keywords: {item.keywords[:3]}")
                print(f"      Frequency: {item.frequency}, Severity: {item.severity}")

        return kb_size > 0

    except Exception as e:
        print(f"❌ Knowledge base loading failed: {e}")
        return False

if __name__ == "__main__":
    async def main():
        # Test knowledge base loading first
        kb_success = await test_knowledge_base_loading()

        if kb_success:
            # Run full RAG few-shot tests
            await test_rag_few_shot_system()
        else:
            print("⚠️ Skipping RAG tests due to knowledge base loading failure")
            print("💡 Check CSV file paths in app/util/config.py")

    # Run the tests
    asyncio.run(main())