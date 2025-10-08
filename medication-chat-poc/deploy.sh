#!/bin/bash
# Dokploy Deployment Script for Medical Chat Application

set -e

echo "🚀 Starting Dokploy Deployment for Medical Chat Application"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: docker-compose is not installed${NC}"
    exit 1
fi

# Check if docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    exit 1
fi

# Build and start services
echo -e "${YELLOW}Building Docker images...${NC}"
docker-compose -f dokploy-compose.yml build

echo -e "${YELLOW}Starting services...${NC}"
docker-compose -f dokploy-compose.yml up -d

# Wait for services to be healthy
echo -e "${YELLOW}Waiting for services to be healthy...${NC}"

# Wait for Ollama
echo "Checking Ollama service..."
for i in {1..60}; do
    if docker-compose -f dokploy-compose.yml exec -T ollama curl -f http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama is ready${NC}"
        break
    fi
    echo "Waiting for Ollama... ($i/60)"
    sleep 2
done

# Wait for Backend
echo "Checking Backend service..."
for i in {1..30}; do
    if docker-compose -f dokploy-compose.yml exec -T backend curl -f http://localhost:8000/api/v1/health/ > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Backend is ready${NC}"
        break
    fi
    echo "Waiting for Backend... ($i/30)"
    sleep 2
done

# Wait for Frontend
echo "Checking Frontend service..."
for i in {1..30}; do
    if curl -f http://localhost:3000 > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Frontend is ready${NC}"
        break
    fi
    echo "Waiting for Frontend... ($i/30)"
    sleep 2
done

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}🎉 Deployment Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Services are running:"
echo "  Frontend:  http://localhost:3000"
echo "  Backend:   http://localhost:8000"
echo "  Ollama:    http://localhost:11434"
echo ""
echo "To view logs:"
echo "  docker-compose -f dokploy-compose.yml logs -f"
echo ""
echo "To stop services:"
echo "  docker-compose -f dokploy-compose.yml down"
echo ""
echo "To stop and remove volumes:"
echo "  docker-compose -f dokploy-compose.yml down -v"
echo ""
