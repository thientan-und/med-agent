#!/usr/bin/env python3
"""
Precision Architecture Test with Vector Database
Tests the precision medical AI system after VitalSigns removal
"""

import asyncio
import json
import sys
sys.path.append('.')

from app.core.precision_service import create_precision_medical_ai
from app.core.types import DiagnosisCard

async def test_precision_with_vector_db():
    """Test precision architecture with vector database integration"""
    print("🔬 PRECISION ARCHITECTURE + VECTOR DB TEST")
    print("=" * 60)

    # Initialize precision AI
    precision_ai = create_precision_medical_ai()
    print("✅ Precision AI initialized")

    # Test cases designed for vector database retrieval
    test_cases = [
        {
            "id": "VDB001",
            "symptoms": "chest pain shortness of breath sweating",
            "patient_data": {"age": 55, "gender": "male"},
            "expected": "cardiac emergency",
            "description": "Classic MI symptoms - should retrieve cardiac guidelines"
        },
        {
            "id": "VDB002",
            "symptoms": "ไข้ ปวดหัว เจ็บคอ",
            "patient_data": {"age": 25, "gender": "female"},
            "expected": "upper respiratory infection",
            "description": "Thai common cold - should use local knowledge base"
        },
        {
            "id": "VDB003",
            "symptoms": "frequent urination excessive thirst weight loss",
            "patient_data": {"age": 45, "gender": "male"},
            "expected": "diabetes mellitus",
            "description": "Diabetes symptoms - should retrieve endocrine knowledge"
        },
        {
            "id": "VDB004",
            "symptoms": "severe abdominal pain nausea vomiting",
            "patient_data": {"age": 35, "gender": "female"},
            "expected": "acute abdomen",
            "description": "Emergency abdominal - should trigger emergency protocols"
        },
        {
            "id": "VDB005",
            "symptoms": "rash itching swelling difficulty breathing",
            "patient_data": {"age": 20, "gender": "male"},
            "expected": "anaphylaxis",
            "description": "Allergic emergency - should retrieve allergy protocols"
        }
    ]

    results = []

    print(f"\n🧪 Testing {len(test_cases)} cases...")

    for case in test_cases:
        print(f"\n📋 Testing {case['id']}: {case['description']}")
        print(f"   Symptoms: {case['symptoms']}")

        try:
            # Process with precision AI (no vital_signs)
            result = await precision_ai.process_medical_consultation(
                message=case["symptoms"],
                patient_data=case["patient_data"],
                session_id=f"test_{case['id']}"
            )

            # Analyze result
            if isinstance(result, DiagnosisCard):
                print(f"   ✅ DiagnosisCard generated")
                print(f"   📊 Overall confidence: {result.overall_confidence:.2%}")

                if result.differential:
                    top_diagnosis = result.differential[0]
                    print(f"   🩺 Top diagnosis: {top_diagnosis.label}")
                    print(f"   📈 Confidence: {top_diagnosis.confidence:.2%}")

                    if top_diagnosis.evidence:
                        print(f"   📚 Evidence items: {len(top_diagnosis.evidence)}")
                        for i, evidence in enumerate(top_diagnosis.evidence[:2]):
                            print(f"      {i+1}. {evidence.source}: {evidence.content[:50]}...")

                if result.uncertainty:
                    print(f"   🤔 Uncertainty: {1-result.uncertainty.safety_certainty:.1%}")
                    print(f"   🛡️ Should abstain: {result.uncertainty.should_abstain}")

                if result.triage:
                    print(f"   ⚖️ Triage level: {result.triage.get('level', 'unknown')}")

                # Check for VitalSigns contamination
                result_str = str(result)
                has_vitalsigns = "vital" in result_str.lower()
                print(f"   🚫 VitalSigns refs: {has_vitalsigns}")

                results.append({
                    "case_id": case["id"],
                    "success": True,
                    "diagnosis": top_diagnosis.label if result.differential else "None",
                    "confidence": result.overall_confidence,
                    "evidence_count": len(top_diagnosis.evidence) if result.differential and top_diagnosis.evidence else 0,
                    "abstention": result.uncertainty.should_abstain if result.uncertainty else False,
                    "vitalsigns_clean": not has_vitalsigns
                })

            else:
                print(f"   ❌ Unexpected result type: {type(result)}")
                results.append({
                    "case_id": case["id"],
                    "success": False,
                    "error": f"Wrong result type: {type(result)}"
                })

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            results.append({
                "case_id": case["id"],
                "success": False,
                "error": str(e)
            })

    # Generate summary
    print("\n" + "=" * 60)
    print("📊 PRECISION + VECTOR DB EVALUATION SUMMARY")
    print("=" * 60)

    total_cases = len(results)
    successful_cases = sum(1 for r in results if r.get("success", False))
    vitalsigns_clean = sum(1 for r in results if r.get("vitalsigns_clean", False))
    avg_confidence = sum(r.get("confidence", 0) for r in results if r.get("success")) / max(successful_cases, 1)
    total_evidence = sum(r.get("evidence_count", 0) for r in results if r.get("success"))
    abstention_rate = sum(1 for r in results if r.get("abstention", False)) / total_cases

    print(f"\n📈 METRICS")
    print(f"  Total Cases: {total_cases}")
    print(f"  Success Rate: {successful_cases}/{total_cases} ({successful_cases/total_cases:.1%})")
    print(f"  VitalSigns Clean: {vitalsigns_clean}/{total_cases} ({vitalsigns_clean/total_cases:.1%})")
    print(f"  Average Confidence: {avg_confidence:.1%}")
    print(f"  Total Evidence Retrieved: {total_evidence}")
    print(f"  Abstention Rate: {abstention_rate:.1%}")

    print(f"\n📋 DETAILED RESULTS")
    print("Case   | Success | Diagnosis                | Conf   | Evidence | Clean")
    print("-" * 75)
    for result in results:
        success = "✅" if result.get("success") else "❌"
        clean = "✅" if result.get("vitalsigns_clean") else "❌"
        diagnosis = result.get("diagnosis", "Error")[:20]
        confidence = f"{result.get('confidence', 0):.1%}" if result.get("success") else "N/A"
        evidence = str(result.get("evidence_count", 0)) if result.get("success") else "N/A"

        print(f"{result['case_id']:6} | {success:7} | {diagnosis:24} | {confidence:6} | {evidence:8} | {clean}")

    # Errors
    errors = [r for r in results if not r.get("success")]
    if errors:
        print(f"\n❌ ERRORS ({len(errors)} cases)")
        for result in errors:
            print(f"  {result['case_id']}: {result.get('error', 'Unknown error')}")

    print(f"\n🎉 EVALUATION COMPLETE")
    if vitalsigns_clean == total_cases:
        print("✅ System is completely free of VitalSigns references")
    else:
        print("❌ VitalSigns contamination detected")

    if successful_cases >= total_cases * 0.8:
        print("✅ Precision architecture performing well")
    else:
        print("⚠️ Precision architecture needs attention")

    # Save results
    with open("precision_vector_db_results.json", "w") as f:
        json.dump({
            "summary": {
                "total_cases": total_cases,
                "success_rate": successful_cases/total_cases,
                "vitalsigns_clean_rate": vitalsigns_clean/total_cases,
                "avg_confidence": avg_confidence,
                "abstention_rate": abstention_rate
            },
            "results": results
        }, f, indent=2)

    print("💾 Results saved to precision_vector_db_results.json")

if __name__ == "__main__":
    asyncio.run(test_precision_with_vector_db())