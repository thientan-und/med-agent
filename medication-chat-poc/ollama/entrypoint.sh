#!/bin/bash
set -e

echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is ready!"
        break
    fi
    echo "Waiting for Ollama... ($i/30)"
    sleep 2
done

# Pull required models
echo "Pulling MedLlama2 model..."
ollama pull medllama2:latest

echo "Pulling SEA-LLM model..."
ollama pull nxphi47/seallm-7b-v2-q4_0:latest

echo "All models loaded successfully!"
echo "Ollama is ready to serve requests"

# Keep the container running
wait $OLLAMA_PID
