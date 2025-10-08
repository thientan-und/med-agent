#!/bin/bash

echo "Setting up Ollama for Medical Chat App"
echo "======================================="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✓ Ollama installed"
else
    echo "✓ Ollama is already installed"
fi

# Start Ollama service
echo "Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!
sleep 5

# Pull required models
echo "Pulling required models..."

echo "1. Pulling MedLlama2 (medical model)..."
ollama pull medllama2

echo "2. Pulling SeaLLM (translation model)..."
ollama pull nxphi47/seallm-7b-v2-q4_0

# List available models
echo ""
echo "Available models:"
ollama list

echo ""
echo "Setup complete! Ollama is running with PID: $OLLAMA_PID"
echo "To stop Ollama: kill $OLLAMA_PID"
echo ""
echo "You can now run: npm run dev"