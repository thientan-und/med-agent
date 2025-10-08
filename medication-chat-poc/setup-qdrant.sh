#!/bin/bash

echo "=== QDRANT SETUP SCRIPT ==="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "✅ Docker is installed"
echo ""

# Check if Qdrant is already running
if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    echo "✅ Qdrant is already running on port 6333"
else
    echo "Starting Qdrant vector database..."

    # Pull and run Qdrant
    docker pull qdrant/qdrant

    # Run Qdrant with persistent storage
    docker run -d \
        --name qdrant \
        -p 6333:6333 \
        -p 6334:6334 \
        -v $(pwd)/qdrant_storage:/qdrant/storage:z \
        qdrant/qdrant

    echo "⏳ Waiting for Qdrant to start..."
    sleep 5

    # Check if Qdrant started successfully
    if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
        echo "✅ Qdrant started successfully!"
        echo "   - REST API: http://localhost:6333"
        echo "   - Web UI: http://localhost:6333/dashboard"
        echo "   - Data stored in: $(pwd)/qdrant_storage"
    else
        echo "❌ Failed to start Qdrant. Check Docker logs:"
        echo "   docker logs qdrant"
        exit 1
    fi
fi

echo ""
echo "=== QDRANT READY ==="
echo "You can now run: node index-to-qdrant.js"