#!/bin/bash

echo "Setting up Ollama Models for Medical Chat App"
echo "=============================================="
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed!"
    echo ""
    echo "Please install Ollama first by running:"
    echo "  curl -fsSL https://ollama.com/install.sh | sh"
    echo ""
    exit 1
fi

echo "✓ Ollama is installed"
echo ""

# Check if Ollama service is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama service..."
    ollama serve &
    sleep 5
    echo "✓ Ollama service started"
else
    echo "✓ Ollama service is already running"
fi

echo ""
echo "Pulling required models (this may take a while)..."
echo ""

# Pull MedLlama2 model (medical AI model)
echo "1. Pulling MedLlama2 (medical AI model)..."
echo "   Size: ~3.8GB"
ollama pull medllama2
if [ $? -eq 0 ]; then
    echo "   ✓ MedLlama2 installed successfully"
else
    echo "   ❌ Failed to pull MedLlama2"
    echo "   Trying alternative medical model..."
    ollama pull llama3.2:3b
    if [ $? -eq 0 ]; then
        echo "   ✓ Llama3.2:3b installed (can be used for medical tasks)"
        echo "   Note: Update config to use llama3.2:3b instead of medllama2"
    else
        echo "   ❌ Failed to pull alternative model"
        echo "   You can try manually: ollama pull medllama2"
    fi
fi

echo ""

# Pull SeaLLM model for Thai translation
echo "2. Pulling SeaLLM-7B-v2 (Thai translation model)..."
echo "   Size: ~4GB"
ollama pull nxphi47/seallm-7b-v2-q4_0:latest
if [ $? -eq 0 ]; then
    echo "   ✓ SeaLLM-7B-v2 installed successfully (nxphi47/seallm-7b-v2-q4_0:latest)"
else
    echo "   ❌ Failed to pull nxphi47/seallm-7b-v2-q4_0:latest"
    echo "   Trying alternative SeaLLM model names..."
    ollama pull nxphi47/seallm-7b-v2-q4_0
    if [ $? -eq 0 ]; then
        echo "   ✓ SeaLLM installed successfully (nxphi47/seallm-7b-v2-q4_0)"
    else
        # Try simpler variants
        ollama pull seallm:7b
        if [ $? -eq 0 ]; then
            echo "   ✓ SeaLLM installed successfully (seallm:7b)"
            echo "   Note: Update config to use seallm:7b instead of nxphi47/seallm-7b-v2-q4_0:latest"
        else
            echo "   ❌ All SeaLLM attempts failed"
            echo "   Falling back to general model that supports Thai..."
            ollama pull llama3.2:3b
            if [ $? -eq 0 ]; then
                echo "   ✓ Llama3.2:3b can handle Thai translation"
                echo "   Note: Update config to use llama3.2:3b for both models"
            fi
        fi
    fi
fi

echo ""
echo "Checking installed models..."
ollama list

echo ""
echo "================================"
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Update .env.local if needed (currently using localhost:11434)"
echo "2. Start the app: npm run dev"
echo "3. Test the models: node test-llm.js"
echo ""