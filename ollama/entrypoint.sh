#!/bin/bash
set -e

echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

echo "Waiting for Ollama to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is ready!"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

echo "Pulling MedLlama2 model..."
ollama pull medllama2:latest || echo "⚠️ Failed to pull medllama2, will retry later"

echo "Pulling SEA-LLM model..."
ollama pull nxphi47/seallm-7b-v2-q4_0:latest || echo "⚠️ Failed to pull seallm, will retry later"

echo "✅ Ollama ready with models!"
wait $OLLAMA_PID
