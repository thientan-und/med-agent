#!/bin/bash
# Dokploy Deployment Script for Medical Chat Application

set -e

echo "🚀 Starting Dokploy deployment for Medical Chat Application"
echo "============================================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install docker-compose."
    exit 1
fi

# Check if required files exist
required_files=(
    "dokploy-compose.yml"
    "backend/Dockerfile"
    "frontend/Dockerfile"
    "ollama/Dockerfile"
    "ollama/download-models.sh"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Required file not found: $file"
        exit 1
    fi
done

echo "✅ All required files found"

# Make scripts executable
chmod +x ollama/download-models.sh
chmod +x frontend/start-sync.sh 2>/dev/null || echo "⚠️ start-sync.sh not found in frontend"

echo "✅ Scripts made executable"

# Set up environment variables
if [ ! -f ".env.production" ]; then
    echo "⚠️ .env.production not found, using defaults"
fi

# Build and start services
echo "🔧 Building and starting services..."
echo "This may take 10-15 minutes for first-time setup (downloading models)"

# Pull base images first
echo "📦 Pulling base images..."
docker pull node:22-alpine
docker pull python:3.11-slim
docker pull ollama/ollama:latest

# Build and start with compose
echo "🏗️ Building services..."
docker-compose -f dokploy-compose.yml build --no-cache

echo "🚀 Starting services..."
docker-compose -f dokploy-compose.yml up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."

# Check Ollama health
echo "🧠 Checking Ollama service..."
timeout=300  # 5 minutes
counter=0
while [ $counter -lt $timeout ]; do
    if docker-compose -f dokploy-compose.yml exec -T ollama curl -f http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is healthy"
        break
    fi
    echo "⏳ Waiting for Ollama... ($counter/$timeout)"
    sleep 10
    counter=$((counter + 10))
done

if [ $counter -ge $timeout ]; then
    echo "❌ Ollama failed to start within timeout"
    exit 1
fi

# Check Backend health
echo "🔧 Checking Backend service..."
counter=0
while [ $counter -lt 120 ]; do
    if docker-compose -f dokploy-compose.yml exec -T backend curl -f http://localhost:8000/api/v1/health/ > /dev/null 2>&1; then
        echo "✅ Backend is healthy"
        break
    fi
    echo "⏳ Waiting for Backend... ($counter/120)"
    sleep 5
    counter=$((counter + 5))
done

if [ $counter -ge 120 ]; then
    echo "❌ Backend failed to start within timeout"
    exit 1
fi

# Check Frontend health
echo "🌐 Checking Frontend service..."
counter=0
while [ $counter -lt 60 ]; do
    if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
        echo "✅ Frontend is healthy"
        break
    fi
    echo "⏳ Waiting for Frontend... ($counter/60)"
    sleep 5
    counter=$((counter + 5))
done

if [ $counter -ge 60 ]; then
    echo "❌ Frontend failed to start within timeout"
    exit 1
fi

echo ""
echo "🎉 DEPLOYMENT SUCCESSFUL!"
echo "============================================================"
echo "📋 Service Status:"
docker-compose -f dokploy-compose.yml ps

echo ""
echo "🌐 Access URLs:"
echo "  Frontend:  http://localhost:3000"
echo "  Backend:   http://localhost:8000"
echo "  Ollama:    http://localhost:11434"
echo ""
echo "📚 API Documentation:"
echo "  Swagger UI: http://localhost:8000/docs"
echo "  ReDoc:      http://localhost:8000/redoc"
echo ""
echo "🔍 Health Checks:"
echo "  Frontend:  http://localhost:3000/api/health"
echo "  Backend:   http://localhost:8000/api/v1/health/"
echo "  Ollama:    http://localhost:11434/api/tags"
echo ""
echo "📊 Monitoring:"
echo "  docker-compose -f dokploy-compose.yml logs -f"
echo "  docker-compose -f dokploy-compose.yml ps"
echo ""
echo "🛑 To stop:"
echo "  docker-compose -f dokploy-compose.yml down"

# Test basic functionality
echo "🧪 Running basic health tests..."
sleep 5

# Test Ollama models
echo "Testing Ollama models..."
if curl -s http://localhost:11434/api/tags | grep -q "medllama2"; then
    echo "✅ MedLlama2 model is available"
else
    echo "⚠️ MedLlama2 model not found"
fi

if curl -s http://localhost:11434/api/tags | grep -q "seallm"; then
    echo "✅ SeaLLM model is available"
else
    echo "⚠️ SeaLLM model not found"
fi

# Test backend API
echo "Testing backend API..."
if curl -s http://localhost:8000/api/v1/health/ | grep -q "healthy"; then
    echo "✅ Backend API is responding"
else
    echo "⚠️ Backend API may have issues"
fi

echo ""
echo "🎯 DEPLOYMENT COMPLETE!"
echo "Your Medical Chat Application is now running on Dokploy-compatible setup"