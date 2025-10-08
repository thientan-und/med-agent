#!/bin/bash

echo "🚀 Starting Medical Chat Application in Production Mode"
echo "=================================================="

# Set environment variables
export NODE_ENV=production
export PORT=${PORT:-3000}
export WEBSOCKET_PORT=${WEBSOCKET_PORT:-3003}

echo "📊 Environment Configuration:"
echo "NODE_ENV: $NODE_ENV"
echo "PORT: $PORT"
echo "WEBSOCKET_PORT: $WEBSOCKET_PORT"
echo ""

# Function to check if port is available
check_port() {
    local port=$1
    if command -v lsof >/dev/null 2>&1; then
        ! lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1
    else
        ! netstat -tuln 2>/dev/null | grep -q ":$port "
    fi
}

# Start WebSocket server
echo "🔌 Starting WebSocket server on port $WEBSOCKET_PORT..."
if check_port $WEBSOCKET_PORT; then
    node websocket-server.js &
    WEBSOCKET_PID=$!
    echo "✅ WebSocket server started (PID: $WEBSOCKET_PID)"
    sleep 2
else
    echo "⚠️  Port $WEBSOCKET_PORT already in use, WebSocket server may already be running"
fi

# Start the main application server
echo "🌐 Starting main application server on port $PORT..."
if check_port $PORT; then
    if [ -f "server.js" ]; then
        # Use standalone server if available
        node server.js &
    else
        # Use Next.js start command
        npm start &
    fi
    APP_PID=$!
    echo "✅ Application server started (PID: $APP_PID)"
else
    echo "⚠️  Port $PORT already in use, application may already be running"
fi

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 5

echo ""
echo "✅ Medical Chat Application is ready!"
echo "=================================================="
echo "👤 Patient Interface:  http://0.0.0.0:$PORT"
echo "👨‍⚕️  Doctor Interface:   http://0.0.0.0:$PORT/doctor"
echo "🔌 WebSocket Server:    http://0.0.0.0:$WEBSOCKET_PORT"
echo "💊 Health Check:       http://0.0.0.0:$PORT/api/health"
echo ""
echo "🔄 Real-time synchronization is active!"
echo "💬 Messages will sync between patient and doctor interfaces"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."

    if [ ! -z "$APP_PID" ]; then
        echo "Stopping application server (PID: $APP_PID)..."
        kill $APP_PID 2>/dev/null
    fi

    if [ ! -z "$WEBSOCKET_PID" ]; then
        echo "Stopping WebSocket server (PID: $WEBSOCKET_PID)..."
        kill $WEBSOCKET_PID 2>/dev/null
    fi

    # Kill any remaining processes
    pkill -f "node.*server" 2>/dev/null
    pkill -f "websocket-server.js" 2>/dev/null

    echo "✅ All services stopped"
    exit 0
}

# Set up trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Keep the script running
echo "🔄 Services running... Press Ctrl+C to stop"
wait