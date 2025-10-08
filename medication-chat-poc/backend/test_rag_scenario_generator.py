#!/usr/bin/env python3
"""
Test RAG-Driven Scenario Generation for AI Model Training
=========================================================

This script demonstrates how to use RAG knowledge base to dynamically
generate few-shot learning scenarios that can train AI models to be
more accurate, safe, and knowledge-driven.

Testing Areas:
1. Dynamic scenario generation from medical knowledge
2. Contextual prompt creation for AI training
3. Safety-validated learning examples
4. Multi-complexity scenario generation
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

# Import services
from app.services.rag_scenario_generator import (
    rag_scenario_generator, ScenarioType, ScenarioComplexity
)
from app.services.ollama_client import ollama_client

async def test_rag_scenario_generation():
    """Test complete RAG-driven scenario generation system"""

    print("🎭 RAG-DRIVEN SCENARIO GENERATION FOR AI TRAINING")
    print("=" * 60)

    # Test different scenario types
    test_scenarios = [
        {
            "target_condition": "common cold",
            "scenario_type": ScenarioType.DIAGNOSTIC,
            "count": 3,
            "description": "Common Cold Diagnostic Scenarios"
        },
        {
            "target_condition": "diabetes",
            "scenario_type": ScenarioType.TREATMENT,
            "count": 2,
            "description": "Diabetes Treatment Planning Scenarios"
        },
        {
            "target_condition": None,  # Mixed conditions
            "scenario_type": ScenarioType.SAFETY_CHECK,
            "count": 3,
            "description": "Mixed Safety Validation Scenarios"
        },
        {
            "target_condition": "pneumonia",
            "scenario_type": ScenarioType.TRIAGE,
            "count": 2,
            "description": "Pneumonia Triage Scenarios"
        }
    ]

    print(f"🔬 Testing {len(test_scenarios)} scenario generation types...")
    print()

    # Initialize generator
    try:
        print("📚 Initializing RAG Scenario Generator...")
        await rag_scenario_generator.initialize()
        print("✅ RAG Scenario Generator initialized")
        print()

    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return

    # Test each scenario type
    all_scenarios = []

    for i, test_config in enumerate(test_scenarios, 1):
        print(f"🎭 Test {i}: {test_config['description']}")
        print(f"   Target: {test_config['target_condition'] or 'Mixed conditions'}")
        print(f"   Type: {test_config['scenario_type'].value}")
        print(f"   Count: {test_config['count']}")
        print()

        try:
            # Generate scenarios
            scenarios = await rag_scenario_generator.generate_few_shot_scenarios(
                target_condition=test_config['target_condition'],
                scenario_type=test_config['scenario_type'],
                count=test_config['count']
            )

            print(f"   ✅ Generated {len(scenarios)} scenarios:")

            for j, scenario in enumerate(scenarios, 1):
                print(f"      {j}. {scenario.expected_diagnosis['name']}")
                print(f"         Complexity: {scenario.complexity.value}")
                print(f"         Patient: {scenario.patient_profile.age}y {scenario.patient_profile.gender}")
                print(f"         Confidence Target: {scenario.confidence_target:.0%}")
                print(f"         Symptoms (Thai): {scenario.presenting_symptoms['thai'][:50]}...")
                print(f"         Learning Objectives: {len(scenario.learning_objectives)} points")
                print(f"         Safety Considerations: {len(scenario.safety_considerations)} items")
                print()

            all_scenarios.extend(scenarios)

        except Exception as e:
            print(f"   ❌ Failed to generate scenarios: {e}")

        print("-" * 60)
        print()

    # Test AI training prompt generation
    print("📝 AI TRAINING PROMPT GENERATION")
    print("=" * 60)

    if all_scenarios:
        try:
            # Create comprehensive training prompt
            print("📝 Creating AI training prompt from generated scenarios...")

            training_prompt = await rag_scenario_generator.create_ai_training_prompt(
                scenarios=all_scenarios[:5],  # Use first 5 scenarios
                target_learning="diagnostic reasoning and safety"
            )

            print(f"✅ Generated training prompt: {len(training_prompt)} characters")
            print()

            # Show prompt structure
            print("📋 Training Prompt Structure:")
            lines = training_prompt.split('\n')
            structure_lines = [line for line in lines[:50] if line.strip() and (line.startswith('#') or line.startswith('**'))]
            for line in structure_lines[:10]:
                print(f"   {line}")
            print(f"   ... (showing first 10 structural elements of {len(structure_lines)} total)")
            print()

            # Save full prompt for review
            prompt_file = f"rag_generated_training_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(training_prompt)
            print(f"📁 Full training prompt saved to: {prompt_file}")
            print()

        except Exception as e:
            print(f"❌ Failed to create training prompt: {e}")

    # Test AI model with generated scenarios
    print("🤖 AI MODEL TESTING WITH GENERATED SCENARIOS")
    print("=" * 60)

    if all_scenarios:
        await test_ai_with_generated_scenarios(all_scenarios[:3])

    # Comprehensive analysis
    print("📊 SCENARIO GENERATION ANALYSIS")
    print("=" * 60)

    if all_scenarios:
        analyze_generated_scenarios(all_scenarios)

    # Save results
    results_file = f"rag_scenario_generation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        # Convert scenarios to serializable format
        serializable_scenarios = []
        for scenario in all_scenarios:
            scenario_dict = {
                "id": scenario.id,
                "scenario_type": scenario.scenario_type.value,
                "complexity": scenario.complexity.value,
                "patient_profile": {
                    "age": scenario.patient_profile.age,
                    "gender": scenario.patient_profile.gender,
                    "medical_history": scenario.patient_profile.medical_history,
                    "risk_factors": scenario.patient_profile.risk_factors
                },
                "presenting_symptoms": scenario.presenting_symptoms,
                "expected_diagnosis": scenario.expected_diagnosis,
                "learning_objectives": scenario.learning_objectives,
                "safety_considerations": scenario.safety_considerations,
                "confidence_target": scenario.confidence_target,
                "knowledge_sources": scenario.knowledge_sources
            }
            serializable_scenarios.append(scenario_dict)

        json.dump({
            "generation_summary": {
                "total_scenarios": len(all_scenarios),
                "scenario_types": list(set(s.scenario_type.value for s in all_scenarios)),
                "complexity_levels": list(set(s.complexity.value for s in all_scenarios)),
                "average_confidence": sum(s.confidence_target for s in all_scenarios) / len(all_scenarios),
                "timestamp": datetime.now().isoformat()
            },
            "scenarios": serializable_scenarios
        }, f, indent=2, ensure_ascii=False)

    print(f"📁 Detailed results saved to: {results_file}")
    print()

    print(f"⏰ Testing completed at: {datetime.now()}")

async def test_ai_with_generated_scenarios(scenarios):
    """Test AI model using generated scenarios"""

    print("🤖 Testing AI model with RAG-generated scenarios...")
    print()

    # Check if Ollama is available
    try:
        await ollama_client.initialize()
        if not await ollama_client.check_connection():
            print("⚠️ Ollama not available - skipping AI model testing")
            return
    except Exception as e:
        print(f"⚠️ Ollama connection failed: {e} - skipping AI model testing")
        return

    for i, scenario in enumerate(scenarios, 1):
        print(f"🧪 AI Test {i}: {scenario.expected_diagnosis['name']}")
        print(f"   Input Symptoms: {scenario.presenting_symptoms['thai']}")
        print(f"   Expected: {scenario.expected_diagnosis['name']} ({scenario.confidence_target:.0%})")
        print()

        try:
            # Create test prompt for AI
            test_prompt = f"""
You are a medical AI assistant. Analyze these symptoms and provide a diagnosis.

Patient: {scenario.patient_profile.age}-year-old {scenario.patient_profile.gender}
Symptoms: {scenario.presenting_symptoms['thai']}
Context: {scenario.clinical_context}

Provide:
1. Primary diagnosis (Thai and English)
2. Confidence level (%)
3. Urgency level
4. Key reasoning

Keep response concise and professional.
"""

            # Test with AI model
            print("   🔄 Querying AI model...")
            response = await ollama_client.chat(
                model="medllama2:latest",
                prompt=test_prompt,
                context={"test_scenario": True}
            )

            if response.get("success"):
                ai_response = response.get("response", "No response")
                print(f"   🤖 AI Response:")
                print(f"      {ai_response[:200]}...")
                print()

                # Basic analysis of response
                response_lower = ai_response.lower()
                expected_condition = scenario.expected_diagnosis['name'].lower()

                if any(part in response_lower for part in expected_condition.split()):
                    print("   ✅ AI correctly identified condition")
                else:
                    print("   ⚠️ AI response differs from expected")

            else:
                print(f"   ❌ AI query failed: {response.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"   ❌ AI testing error: {e}")

        print("-" * 40)
        print()

def analyze_generated_scenarios(scenarios):
    """Analyze the quality and characteristics of generated scenarios"""

    print("📈 Scenario Generation Quality Analysis:")
    print()

    # Complexity distribution
    complexity_counts = {}
    for scenario in scenarios:
        complexity = scenario.complexity.value
        complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1

    print("🎯 Complexity Distribution:")
    for complexity, count in complexity_counts.items():
        percentage = (count / len(scenarios)) * 100
        print(f"   {complexity.title()}: {count} scenarios ({percentage:.1f}%)")
    print()

    # Confidence target analysis
    confidence_levels = [s.confidence_target for s in scenarios]
    avg_confidence = sum(confidence_levels) / len(confidence_levels)
    min_confidence = min(confidence_levels)
    max_confidence = max(confidence_levels)

    print("📊 Confidence Target Analysis:")
    print(f"   Average: {avg_confidence:.2f}")
    print(f"   Range: {min_confidence:.2f} - {max_confidence:.2f}")
    if max_confidence > 0.9:
        print("   ⚠️ Some scenarios have very high confidence targets")
    if min_confidence < 0.5:
        print("   ⚠️ Some scenarios have very low confidence targets")
    print()

    # Safety considerations analysis
    total_safety_items = sum(len(s.safety_considerations) for s in scenarios)
    avg_safety_items = total_safety_items / len(scenarios)

    print("🛡️ Safety Analysis:")
    print(f"   Total safety considerations: {total_safety_items}")
    print(f"   Average per scenario: {avg_safety_items:.1f}")

    # Learning objectives analysis
    total_objectives = sum(len(s.learning_objectives) for s in scenarios)
    avg_objectives = total_objectives / len(scenarios)

    print("🎯 Learning Objectives Analysis:")
    print(f"   Total learning objectives: {total_objectives}")
    print(f"   Average per scenario: {avg_objectives:.1f}")
    print()

    # Knowledge source diversity
    all_sources = set()
    for scenario in scenarios:
        all_sources.update(scenario.knowledge_sources)

    print("📚 Knowledge Source Diversity:")
    print(f"   Unique knowledge sources: {len(all_sources)}")
    print("   Sample sources:")
    for source in list(all_sources)[:3]:
        print(f"      - {source}")
    print()

    # Scenario type distribution
    type_counts = {}
    for scenario in scenarios:
        scenario_type = scenario.scenario_type.value
        type_counts[scenario_type] = type_counts.get(scenario_type, 0) + 1

    print("🏷️ Scenario Type Distribution:")
    for scenario_type, count in type_counts.items():
        percentage = (count / len(scenarios)) * 100
        print(f"   {scenario_type.replace('_', ' ').title()}: {count} scenarios ({percentage:.1f}%)")

if __name__ == "__main__":
    asyncio.run(test_rag_scenario_generation())