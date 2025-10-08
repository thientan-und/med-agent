#!/bin/bash

echo "🎯 PRECISION MEDICAL AI EVALUATION SUITE"
echo "========================================"
echo ""

# Check if we're in the right directory
if [ ! -d "backend" ]; then
    echo "❌ Please run this script from the medical-chat-app root directory"
    exit 1
fi

# Check if Python dependencies are installed
echo "🔍 Checking Python dependencies..."
cd backend

if ! python3 -c "import app.core.precision_service" 2>/dev/null; then
    echo "⚠️ Precision service not found. Installing dependencies..."
    pip install -r requirements.txt
fi

echo "✅ Dependencies checked"
echo ""

# Check if Ollama is running
echo "🤖 Checking Ollama connection..."
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "✅ Ollama is running on localhost:11434"

    # Check for required models
    MODELS=$(curl -s http://localhost:11434/api/tags | python3 -c "
import json, sys
data = json.load(sys.stdin)
models = [m['name'] for m in data.get('models', [])]
print(','.join(models))
")

    if [[ $MODELS == *"medllama2"* ]]; then
        echo "✅ MedLlama2 model found"
    else
        echo "⚠️ MedLlama2 model not found. Download with: ollama pull medllama2"
    fi

    if [[ $MODELS == *"seallm"* ]]; then
        echo "✅ SeaLLM model found"
    else
        echo "⚠️ SeaLLM model not found. Download with: ollama pull nxphi47/seallm-7b-v2-q4_0"
    fi
else
    echo "⚠️ Ollama not running. Start with: ollama serve"
    echo "   Models will be tested in fallback mode"
fi

echo ""

# Check if Qdrant is running (optional)
echo "🗄️ Checking Qdrant connection..."
if curl -s http://localhost:6333/collections >/dev/null 2>&1; then
    echo "✅ Qdrant is running on localhost:6333"
    QDRANT_AVAILABLE=true
else
    echo "⚠️ Qdrant not running. Start with: ./setup-qdrant.sh"
    echo "   RAG evaluation will use simulated data"
    QDRANT_AVAILABLE=false
fi

echo ""
echo "🚀 Starting evaluation suite..."
echo ""

# Menu for evaluation options
echo "Select evaluation to run:"
echo "1) Precision Architecture Evaluation (tests precision critic, uncertainty, abstention)"
echo "2) Qdrant RAG Integration Evaluation (tests knowledge retrieval and integration)"
echo "3) Comprehensive Evaluation (runs both + combined analysis)"
echo "4) Quick Patient Scenario Test (existing test-patient-scenarios.js)"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "🎯 Running Precision Architecture Evaluation..."
        python3 test_precision_evaluation.py
        ;;
    2)
        echo "🔍 Running Qdrant RAG Integration Evaluation..."
        python3 test_qdrant_rag_evaluation.py
        ;;
    3)
        echo "🏆 Running Comprehensive Evaluation Suite..."
        python3 run_comprehensive_evaluation.py
        ;;
    4)
        echo "👥 Running Patient Scenario Tests..."
        cd ..
        node test-patient-scenarios.js
        ;;
    *)
        echo "❌ Invalid choice. Please select 1-4."
        exit 1
        ;;
esac

echo ""
echo "✨ Evaluation completed!"
echo ""
echo "📊 Results have been saved to JSON files in the backend/ directory"
echo "📄 Check the generated summary files for detailed analysis"
echo ""
echo "🔧 Next steps based on results:"
echo "   - If precision scores < 80%: Review and tune precision critic rules"
echo "   - If RAG scores < 80%: Improve Qdrant knowledge base and retrieval"
echo "   - If safety compliance < 85%: Strengthen safety thresholds"
echo "   - If ready for pilot: Consider controlled deployment testing"