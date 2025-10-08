#!/bin/bash

# Medical Chat AI Backend Setup Script
# Sets up Python virtual environment and dependencies

set -e

echo "🏥 Setting up Medical Chat AI Backend..."

# Check if Python 3.9+ is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $PYTHON_VERSION found, but Python $REQUIRED_VERSION or higher is required."
    exit 1
fi

echo "✅ Python $PYTHON_VERSION found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p training-data
mkdir -p temp

# Copy environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env.example .env
    echo "📝 Please edit .env with your configuration"
else
    echo "✅ .env file already exists"
fi

# Check dependencies
echo "🔍 Checking backend dependencies..."
python start.py --check-only

echo ""
echo "✅ Backend setup completed successfully!"
echo ""
echo "📚 Next steps:"
echo "   1. Edit backend/.env with your configuration"
echo "   2. Ensure Ollama is running: ../setup-ollama.sh"
echo "   3. Download AI models: ../setup-models.sh"
echo "   4. Start development: pnpm dev (from project root)"
echo ""
echo "🚀 FastAPI docs will be available at:"
echo "   - Swagger UI: http://localhost:8000/docs"
echo "   - ReDoc: http://localhost:8000/redoc"