#!/bin/bash
# Test script to verify both AI models are working correctly

echo "🧪 Testing Medical AI Models"
echo "============================"

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "❌ Ollama is not running. Please start it first:"
    echo "   ollama serve"
    exit 1
fi

echo "✅ Ollama service is running"
echo ""

# List available models
echo "📋 Available models:"
ollama list
echo ""

# Test MedLlama2
echo "🩺 Testing MedLlama2 (Medical Analysis)..."
medllama_response=$(curl -s -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "medllama2:latest",
    "prompt": "What are the symptoms of common cold?",
    "stream": false
  }' | jq -r '.response' 2>/dev/null)

if [ $? -eq 0 ] && [ ! -z "$medllama_response" ]; then
    echo "✅ MedLlama2 is responding"
    echo "   Sample response: ${medllama_response:0:100}..."
else
    echo "❌ MedLlama2 is not responding properly"
fi

echo ""

# Test SeaLLM
echo "🇹🇭 Testing SeaLLM (Thai Translation)..."
seallm_response=$(curl -s -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nxphi47/seallm-7b-v2-q4_0:latest",
    "prompt": "Translate to Thai: Hello, how are you feeling today?",
    "stream": false
  }' | jq -r '.response' 2>/dev/null)

if [ $? -eq 0 ] && [ ! -z "$seallm_response" ]; then
    echo "✅ SeaLLM is responding"
    echo "   Sample response: ${seallm_response:0:100}..."
else
    echo "❌ SeaLLM is not responding properly"
fi

echo ""
echo "🎯 Model Testing Complete!"
echo ""
echo "If both models are working, your Medical Chat Application is ready!"
echo "Run the app with: make dev"