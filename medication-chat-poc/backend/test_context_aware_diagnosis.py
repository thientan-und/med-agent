#!/usr/bin/env python3
"""
Context-Aware Diagnosis Test
============================

Tests the Agentic AI's ability to combine patient context with RAG knowledge
for accurate, context-aware medical diagnosis.
"""

import asyncio
import json
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Setup path
sys.path.append('.')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import services
from app.services.medical_ai_service import MedicalAIService
from app.services.rag_few_shot_service import RAGFewShotService
from precision_critic_validator import PrecisionCritic


@dataclass
class PatientContext:
    """Patient context information for diagnosis"""
    age: int
    gender: str
    occupation: Optional[str] = None
    location: Optional[str] = None
    medical_history: Optional[List[str]] = None
    current_medications: Optional[List[str]] = None
    allergies: Optional[List[str]] = None
    lifestyle: Optional[List[str]] = None
    recent_travel: Optional[str] = None
    family_history: Optional[List[str]] = None


class ContextAwareDiagnosisTest:
    """Test context-aware diagnosis capability"""

    def __init__(self):
        self.medical_ai_service = None
        self.rag_service = None
        self.precision_critic = None
        self.test_scenarios = []

    async def initialize(self):
        """Initialize services"""
        print("🧠 CONTEXT-AWARE DIAGNOSIS TEST")
        print("=" * 60)
        print("Testing how AI combines patient context with RAG knowledge")
        print()

        self.medical_ai_service = MedicalAIService()
        await self.medical_ai_service.initialize()
        print("✅ Medical AI Service initialized")

        self.rag_service = RAGFewShotService()
        await self.rag_service.initialize()
        print("✅ RAG Service initialized")

        self.precision_critic = PrecisionCritic()
        print("✅ Precision Critic loaded")

        # Create context-aware test scenarios
        self._create_context_aware_scenarios()
        print(f"✅ Created {len(self.test_scenarios)} context-aware test scenarios")

    def _create_context_aware_scenarios(self):
        """Create test scenarios that require context awareness"""

        self.test_scenarios = [
            # SCENARIO 1: Same symptoms, different contexts → Different diagnoses
            {
                "id": "CONTEXT_001A",
                "name": "Chest Pain - Young Athlete",
                "symptoms": "ปวดอกหลังออกกำลังกาย หายใจลำบาก เหนื่อย",
                "context": PatientContext(
                    age=25,
                    gender="male",
                    occupation="นักกีฬาวิ่งมาราธอน",
                    lifestyle=["ออกกำลังกายหนัก", "วิ่งวันละ 20 กม."],
                    medical_history=["สุขภาพดี", "ไม่มีโรคประจำตัว"]
                ),
                "expected_diagnosis": "Muscle strain / Exercise-induced pain",
                "expected_urgency": "low",
                "rationale": "Young athlete with pain after exercise → likely musculoskeletal"
            },
            {
                "id": "CONTEXT_001B",
                "name": "Chest Pain - Elderly Diabetic",
                "symptoms": "ปวดอกหลังออกกำลังกาย หายใจลำบาก เหนื่อย",  # SAME symptoms
                "context": PatientContext(
                    age=65,
                    gender="male",
                    occupation="ข้าราชการบำนาญ",
                    medical_history=["เบาหวาน 10 ปี", "ความดันสูง", "ไขมันสูง"],
                    current_medications=["Metformin", "Amlodipine"],
                    lifestyle=["สูบบุหรี่ 30 ปี", "ไม่ค่อยออกกำลังกาย"]
                ),
                "expected_diagnosis": "Possible cardiac event / Angina",
                "expected_urgency": "critical",
                "rationale": "Elderly with multiple cardiac risk factors → urgent cardiac evaluation"
            },

            # SCENARIO 2: Fever pattern with occupational context
            {
                "id": "CONTEXT_002A",
                "name": "Fever - Construction Worker",
                "symptoms": "ไข้สูง ปวดหัว ปวดกล้ามเนื้อ อ่อนเพลีย",
                "context": PatientContext(
                    age=35,
                    gender="male",
                    occupation="คนงานก่อสร้าง",
                    location="ภาคตะวันออกเฉียงเหนือ",
                    lifestyle=["ทำงานกลางแจ้ง", "สัมผัสดิน", "ดื่มน้ำจากบ่อ"],
                    recent_travel=None
                ),
                "expected_diagnosis": "Leptospirosis / Melioidosis consideration",
                "expected_urgency": "high",
                "rationale": "Construction worker in endemic area with soil/water exposure"
            },
            {
                "id": "CONTEXT_002B",
                "name": "Fever - Office Worker",
                "symptoms": "ไข้สูง ปวดหัว ปวดกล้ามเนื้อ อ่อนเพลีย",  # SAME symptoms
                "context": PatientContext(
                    age=35,
                    gender="male",
                    occupation="พนักงานออฟฟิศ",
                    location="กรุงเทพมหานคร",
                    lifestyle=["ทำงานในห้องแอร์", "นั่งทำงาน 8 ชม./วัน"],
                    recent_travel=None
                ),
                "expected_diagnosis": "Common viral infection / Influenza",
                "expected_urgency": "low",
                "rationale": "Office worker with no special exposure → common viral illness"
            },

            # SCENARIO 3: Abdominal pain with gender/age context
            {
                "id": "CONTEXT_003A",
                "name": "Abdominal Pain - Young Woman",
                "symptoms": "ปวดท้องน้อยข้างขวา ไข้เล็กน้อย คลื่นไส้",
                "context": PatientContext(
                    age=28,
                    gender="female",
                    medical_history=["ประจำเดือนไม่ปกติ", "ขาดประจำเดือน 6 สัปดาห์"],
                    lifestyle=["แต่งงานแล้ว", "ไม่ได้คุมกำเนิด"]
                ),
                "expected_diagnosis": "Consider ectopic pregnancy vs appendicitis",
                "expected_urgency": "critical",
                "rationale": "Reproductive age woman with missed period → must rule out ectopic pregnancy"
            },
            {
                "id": "CONTEXT_003B",
                "name": "Abdominal Pain - Elderly Man",
                "symptoms": "ปวดท้องน้อยข้างขวา ไข้เล็กน้อย คลื่นไส้",  # SAME symptoms
                "context": PatientContext(
                    age=70,
                    gender="male",
                    medical_history=["ท้องผูกเรื้อรัง", "ริดสีดวง"],
                    current_medications=["ยาระบาย"],
                    lifestyle=["กินผักน้อย", "ดื่มน้ำน้อย"]
                ),
                "expected_diagnosis": "Diverticulitis / Bowel obstruction consideration",
                "expected_urgency": "moderate",
                "rationale": "Elderly with constipation history → consider diverticular disease"
            },

            # SCENARIO 4: Headache with medication history
            {
                "id": "CONTEXT_004A",
                "name": "Headache - Hypertensive Patient",
                "symptoms": "ปวดหัวรุนแรง ตาพร่า คลื่นไส้",
                "context": PatientContext(
                    age=55,
                    gender="female",
                    medical_history=["ความดันสูง"],
                    current_medications=["หยุดยาความดัน 3 วัน"],
                    lifestyle=["เครียดงาน", "นอนดึก"]
                ),
                "expected_diagnosis": "Hypertensive crisis / Uncontrolled hypertension",
                "expected_urgency": "critical",
                "rationale": "Stopped antihypertensive medication → risk of hypertensive crisis"
            },
            {
                "id": "CONTEXT_004B",
                "name": "Headache - Migraine Patient",
                "symptoms": "ปวดหัวรุนแรง ตาพร่า คลื่นไส้",  # SAME symptoms
                "context": PatientContext(
                    age=30,
                    gender="female",
                    medical_history=["ไมเกรน", "ปวดหัวข้างเดียวบ่อย"],
                    family_history=["แม่เป็นไมเกรน"],
                    lifestyle=["ดื่มกาแฟวันละ 4 แก้ว", "นอนน้อย"]
                ),
                "expected_diagnosis": "Migraine attack",
                "expected_urgency": "low",
                "rationale": "Known migraine with typical pattern → likely migraine episode"
            },

            # SCENARIO 5: Rash with travel/exposure context
            {
                "id": "CONTEXT_005A",
                "name": "Rash - Recent Forest Travel",
                "symptoms": "ผื่นแดง ไข้ ปวดข้อ ปวดกล้ามเนื้อ",
                "context": PatientContext(
                    age=40,
                    gender="male",
                    occupation="นักท่องเที่ยว",
                    recent_travel="เดินป่าที่เขาใหญ่ 1 สัปดาห์ก่อน",
                    lifestyle=["ชอบเดินป่า", "นอนแคมป์"]
                ),
                "expected_diagnosis": "Scrub typhus / Tick-borne illness",
                "expected_urgency": "high",
                "rationale": "Forest exposure → consider rickettsial infections"
            },
            {
                "id": "CONTEXT_005B",
                "name": "Rash - Allergic History",
                "symptoms": "ผื่นแดง ไข้ ปวดข้อ ปวดกล้ามเนื้อ",  # SAME symptoms
                "context": PatientContext(
                    age=40,
                    gender="male",
                    medical_history=["แพ้อาหารทะเล", "ลมพิษเรื้อรัง"],
                    recent_travel=None,
                    allergies=["กุ้ง", "ปู", "หอย"],
                    lifestyle=["กินอาหารทะเลเมื่อวาน"]
                ),
                "expected_diagnosis": "Allergic reaction with secondary symptoms",
                "expected_urgency": "moderate",
                "rationale": "Known seafood allergy + recent exposure → allergic reaction"
            },

            # SCENARIO 6: Cough with environmental context
            {
                "id": "CONTEXT_006A",
                "name": "Cough - Smoker",
                "symptoms": "ไอเรื้อรัง ไอมีเสมหะ หายใจลำบากเล็กน้อย",
                "context": PatientContext(
                    age=60,
                    gender="male",
                    lifestyle=["สูบบุหรี่ 40 ปี", "วันละ 2 ซอง"],
                    medical_history=["ไอเรื้อรังทุกเช้า"],
                    occupation="คนขับรถแท็กซี่"
                ),
                "expected_diagnosis": "COPD / Chronic bronchitis",
                "expected_urgency": "moderate",
                "rationale": "Heavy smoker with chronic cough → likely COPD"
            },
            {
                "id": "CONTEXT_006B",
                "name": "Cough - Teacher",
                "symptoms": "ไอเรื้อรัง ไอมีเสมหะ หายใจลำบากเล็กน้อย",  # SAME symptoms
                "context": PatientContext(
                    age=35,
                    gender="female",
                    occupation="ครูประถม",
                    lifestyle=["ไม่สูบบุหรี่", "ไม่ดื่มเหล้า"],
                    medical_history=["มีนักเรียนหลายคนป่วยไข้หวัด"],
                    location="โรงเรียน"
                ),
                "expected_diagnosis": "Upper respiratory infection / Viral bronchitis",
                "expected_urgency": "low",
                "rationale": "Teacher exposed to sick students → likely viral infection"
            },

            # SCENARIO 7: Diarrhea with dietary context
            {
                "id": "CONTEXT_007A",
                "name": "Diarrhea - Street Food",
                "symptoms": "ท้องเสีย ปวดท้อง ไข้ อาเจียน",
                "context": PatientContext(
                    age=25,
                    gender="male",
                    lifestyle=["กินอาหารริมทาง", "กินส้มตำเมื่อวาน"],
                    location="กรุงเทพฯ",
                    medical_history=["สุขภาพดี"]
                ),
                "expected_diagnosis": "Food poisoning / Bacterial gastroenteritis",
                "expected_urgency": "low",
                "rationale": "Street food exposure → likely food poisoning"
            },
            {
                "id": "CONTEXT_007B",
                "name": "Diarrhea - Recent Antibiotics",
                "symptoms": "ท้องเสีย ปวดท้อง ไข้ อาเจียน",  # SAME symptoms
                "context": PatientContext(
                    age=45,
                    gender="female",
                    medical_history=["เพิ่งรักษาติดเชื้อทางเดินปัสสาวะ"],
                    current_medications=["Augmentin กิน 5 วันแล้ว"],
                    lifestyle=["กินอาหารปกติ"]
                ),
                "expected_diagnosis": "Antibiotic-associated diarrhea / C. difficile consideration",
                "expected_urgency": "moderate",
                "rationale": "Recent antibiotic use → consider antibiotic-associated diarrhea"
            }
        ]

    def _format_context_message(self, symptoms: str, context: PatientContext) -> str:
        """Format symptoms with full patient context for AI"""

        # Build comprehensive context message
        message_parts = [symptoms]

        # Add demographic context
        message_parts.append(f"ผู้ป่วย: {context.gender} อายุ {context.age} ปี")

        if context.occupation:
            message_parts.append(f"อาชีพ: {context.occupation}")

        if context.location:
            message_parts.append(f"พื้นที่: {context.location}")

        # Add medical history
        if context.medical_history:
            message_parts.append(f"ประวัติ: {', '.join(context.medical_history)}")

        if context.current_medications:
            message_parts.append(f"ยาที่ใช้: {', '.join(context.current_medications)}")

        if context.allergies:
            message_parts.append(f"แพ้: {', '.join(context.allergies)}")

        # Add lifestyle factors
        if context.lifestyle:
            message_parts.append(f"พฤติกรรม: {', '.join(context.lifestyle)}")

        if context.recent_travel:
            message_parts.append(f"การเดินทาง: {context.recent_travel}")

        if context.family_history:
            message_parts.append(f"ประวัติครอบครัว: {', '.join(context.family_history)}")

        return " | ".join(message_parts)

    async def test_context_aware_diagnosis(self) -> List[Dict[str, Any]]:
        """Test context-aware diagnosis capability"""

        print(f"\n🧬 TESTING CONTEXT-AWARE DIAGNOSIS")
        print("=" * 60)

        results = []

        # Group scenarios by symptom set to compare different contexts
        symptom_groups = {}
        for scenario in self.test_scenarios:
            symptoms = scenario['symptoms']
            if symptoms not in symptom_groups:
                symptom_groups[symptoms] = []
            symptom_groups[symptoms].append(scenario)

        for symptoms, scenarios in symptom_groups.items():
            print(f"\n{'='*70}")
            print(f"📋 TESTING SYMPTOM SET: {symptoms[:50]}...")
            print(f"   Comparing {len(scenarios)} different patient contexts")
            print("="*70)

            for scenario in scenarios:
                print(f"\n🔬 Scenario {scenario['id']}: {scenario['name']}")
                print(f"   Expected: {scenario['expected_diagnosis']}")
                print(f"   Urgency: {scenario['expected_urgency']}")
                print(f"   Rationale: {scenario['rationale']}")

                try:
                    # Format message with full context
                    context_message = self._format_context_message(
                        scenario['symptoms'],
                        scenario['context']
                    )

                    print(f"\n📝 Full Context Message:")
                    print(f"   {context_message}")

                    # Get AI diagnosis
                    print(f"⏱️  Calling AI with context...")
                    api_response = await self.medical_ai_service.assess_common_illness(
                        message=context_message
                    )

                    print(f"✅ Response received")

                    # Extract diagnosis
                    primary_diagnosis = api_response.get('primary_diagnosis', {})
                    diagnosed_condition = primary_diagnosis.get('english_name', 'Unknown')
                    thai_name = primary_diagnosis.get('thai_name', 'Unknown')
                    confidence = primary_diagnosis.get('confidence', 0)
                    urgency = primary_diagnosis.get('urgency', 'unknown')
                    category = primary_diagnosis.get('category', 'unknown')

                    # Check for red flags
                    red_flags = primary_diagnosis.get('red_flags', {})
                    has_red_flags = red_flags.get('detected', False)

                    print(f"\n🎯 AI DIAGNOSIS:")
                    print(f"   Condition: {diagnosed_condition}")
                    print(f"   Thai: {thai_name}")
                    print(f"   Confidence: {confidence}")
                    print(f"   Category: {category}")
                    print(f"   Urgency: {urgency}")
                    print(f"   Red Flags: {'YES' if has_red_flags else 'NO'}")

                    # Check if context influenced diagnosis
                    context_considered = self._check_context_influence(
                        scenario,
                        api_response,
                        diagnosed_condition
                    )

                    print(f"\n📊 CONTEXT ANALYSIS:")
                    print(f"   Context Considered: {'✅' if context_considered else '❌'}")
                    print(f"   Expected Urgency: {scenario['expected_urgency']}")
                    print(f"   Actual Urgency: {urgency if urgency else 'not specified'}")

                    # Store result
                    result = {
                        "scenario_id": scenario['id'],
                        "scenario_name": scenario['name'],
                        "symptoms": scenario['symptoms'],
                        "context": {
                            "age": scenario['context'].age,
                            "gender": scenario['context'].gender,
                            "occupation": scenario['context'].occupation,
                            "medical_history": scenario['context'].medical_history
                        },
                        "context_message": context_message,
                        "expected_diagnosis": scenario['expected_diagnosis'],
                        "expected_urgency": scenario['expected_urgency'],
                        "ai_diagnosis": diagnosed_condition,
                        "ai_urgency": urgency,
                        "confidence": confidence,
                        "red_flags_detected": has_red_flags,
                        "context_considered": context_considered,
                        "api_response": api_response,
                        "timestamp": datetime.now().isoformat()
                    }

                    results.append(result)

                except Exception as e:
                    print(f"❌ ERROR testing {scenario['id']}: {e}")
                    logger.error(f"Context test error for {scenario['id']}: {e}")

                    results.append({
                        "scenario_id": scenario['id'],
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })

        return results

    def _check_context_influence(self, scenario, api_response, diagnosed_condition) -> bool:
        """Check if context influenced the diagnosis appropriately"""

        context = scenario['context']
        expected = scenario['expected_diagnosis'].lower()
        actual = diagnosed_condition.lower()

        # Check for age-appropriate diagnosis
        age_appropriate = False
        if context.age > 60:
            # Elderly context
            if any(term in actual for term in ['cardiac', 'heart', 'หัวใจ', 'stroke']):
                age_appropriate = True
        elif context.age < 30:
            # Young context
            if any(term in actual for term in ['strain', 'viral', 'common', 'ทั่วไป']):
                age_appropriate = True

        # Check for occupation-related diagnosis
        occupation_appropriate = False
        if context.occupation:
            if 'construction' in str(context.occupation).lower() or 'ก่อสร้าง' in str(context.occupation):
                if any(term in actual for term in ['leptospirosis', 'melioid']):
                    occupation_appropriate = True

        # Check for medical history influence
        history_appropriate = False
        if context.medical_history:
            history_str = ' '.join(context.medical_history).lower()
            if 'เบาหวาน' in history_str or 'diabetes' in history_str:
                if any(term in actual for term in ['cardiac', 'emergency']):
                    history_appropriate = True

        # Overall context consideration
        return age_appropriate or occupation_appropriate or history_appropriate

    async def analyze_context_awareness(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how well AI uses context for diagnosis"""

        print(f"\n📊 CONTEXT-AWARENESS ANALYSIS")
        print("=" * 60)

        successful_tests = [r for r in results if 'error' not in r]
        failed_tests = [r for r in results if 'error' in r]

        print(f"📈 Test Execution:")
        print(f"   Total scenarios: {len(results)}")
        print(f"   Successful: {len(successful_tests)}")
        print(f"   Failed: {len(failed_tests)}")

        if not successful_tests:
            return {"error": "No successful tests"}

        # Analyze context consideration
        context_considered_count = len([r for r in successful_tests if r['context_considered']])
        context_rate = context_considered_count / len(successful_tests) * 100 if successful_tests else 0

        print(f"\n🧠 CONTEXT AWARENESS:")
        print(f"   Context Considered: {context_considered_count}/{len(successful_tests)} ({context_rate:.1f}%)")

        # Compare same symptoms with different contexts
        symptom_comparison = {}
        for result in successful_tests:
            symptoms = result['symptoms']
            if symptoms not in symptom_comparison:
                symptom_comparison[symptoms] = []
            symptom_comparison[symptoms].append(result)

        print(f"\n🔄 DIFFERENTIAL DIAGNOSIS BY CONTEXT:")
        for symptoms, scenarios in symptom_comparison.items():
            if len(scenarios) > 1:
                print(f"\n   Symptoms: {symptoms[:50]}...")
                for scenario in scenarios:
                    context_str = f"Age {scenario['context']['age']}, {scenario['context']['gender']}"
                    if scenario['context']['occupation']:
                        context_str += f", {scenario['context']['occupation']}"
                    print(f"      Context: {context_str}")
                    print(f"      → Diagnosis: {scenario['ai_diagnosis']}")
                    print(f"      → Urgency: {scenario['ai_urgency']}")

        # Calculate accuracy metrics
        appropriate_urgency = 0
        for result in successful_tests:
            if result['expected_urgency'] == 'critical' and result.get('red_flags_detected'):
                appropriate_urgency += 1
            elif result['expected_urgency'] == 'low' and not result.get('red_flags_detected'):
                appropriate_urgency += 1

        urgency_accuracy = appropriate_urgency / len(successful_tests) * 100 if successful_tests else 0

        print(f"\n⚠️  URGENCY ASSESSMENT:")
        print(f"   Appropriate Urgency: {appropriate_urgency}/{len(successful_tests)} ({urgency_accuracy:.1f}%)")

        # Overall assessment
        if context_rate >= 70 and urgency_accuracy >= 70:
            assessment = "✅ EXCELLENT - Strong context-aware diagnosis"
        elif context_rate >= 50 and urgency_accuracy >= 50:
            assessment = "⚠️ MODERATE - Some context awareness"
        else:
            assessment = "❌ WEAK - Limited context utilization"

        print(f"\n🏆 CONTEXT-AWARENESS ASSESSMENT:")
        print(f"   {assessment}")

        # Save report
        report_data = {
            "report_metadata": {
                "timestamp": datetime.now().isoformat(),
                "report_type": "Context-Aware Diagnosis Test",
                "version": "1.0.0"
            },
            "test_summary": {
                "total_scenarios": len(results),
                "successful_tests": len(successful_tests),
                "context_consideration_rate": context_rate,
                "urgency_accuracy": urgency_accuracy
            },
            "symptom_comparison": {
                symptoms: [
                    {
                        "scenario_id": s['scenario_id'],
                        "context": s['context'],
                        "diagnosis": s['ai_diagnosis'],
                        "urgency": s['ai_urgency']
                    }
                    for s in scenarios
                ]
                for symptoms, scenarios in symptom_comparison.items()
                if len(scenarios) > 1
            },
            "assessment": assessment,
            "detailed_results": results
        }

        report_file = f"context_aware_diagnosis_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"\n📁 Report saved to: {report_file}")

        return report_data

    async def run_complete_context_test(self):
        """Run complete context-aware diagnosis test"""

        try:
            print("🚀 STARTING CONTEXT-AWARE DIAGNOSIS TEST")
            print("=" * 70)
            print("Testing: Patient Context + RAG Knowledge = Better Diagnosis")
            print("=" * 70)

            # Initialize
            await self.initialize()

            # Test context-aware diagnosis
            results = await self.test_context_aware_diagnosis()

            # Analyze context awareness
            analysis = await self.analyze_context_awareness(results)

            print(f"\n✅ CONTEXT-AWARE DIAGNOSIS TEST COMPLETED")
            print("=" * 60)
            if 'error' not in analysis:
                print(f"🧠 Context Consideration: {analysis['test_summary']['context_consideration_rate']:.1f}%")
                print(f"⚠️  Urgency Accuracy: {analysis['test_summary']['urgency_accuracy']:.1f}%")
                print(f"🏆 Assessment: {analysis['assessment']}")

            return analysis

        except Exception as e:
            logger.error(f"Context test error: {e}")
            print(f"❌ Context test failed: {e}")
            return None


async def main():
    """Main execution function"""
    tester = ContextAwareDiagnosisTest()
    await tester.run_complete_context_test()


if __name__ == "__main__":
    asyncio.run(main())