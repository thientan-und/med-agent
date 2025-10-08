#!/bin/bash

echo "🚀 Starting Medical Chat Application with Real-time Synchronization"
echo "=================================================="

# Check if ports are available
echo "📡 Checking available ports..."

# Start WebSocket server if not running
if ! lsof -Pi :3003 -sTCP:LISTEN -t >/dev/null; then
    echo "🔌 Starting WebSocket server on port 3003..."
    node websocket-server.js &
    WEBSOCKET_PID=$!
    sleep 2
else
    echo "✅ WebSocket server already running on port 3003"
fi

# Start the main development server
echo "🌐 Starting main development server on port 3000..."
pnpm dev &
DEV_PID=$!

# Wait for dev server to start
sleep 5

echo ""
echo "✅ Medical Chat Application is ready!"
echo "=================================================="
echo "👤 Patient Interface:  http://localhost:3000"
echo "👨‍⚕️  Doctor Interface:   http://localhost:3000/doctor"
echo "🔌 WebSocket Server:    http://localhost:3003"
echo ""
echo "🔄 Real-time synchronization is active!"
echo "💬 Messages will sync between patient and doctor interfaces"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    if [ ! -z "$DEV_PID" ]; then
        kill $DEV_PID 2>/dev/null
    fi
    if [ ! -z "$WEBSOCKET_PID" ]; then
        kill $WEBSOCKET_PID 2>/dev/null
    fi
    # Kill any remaining processes
    pkill -f "pnpm dev" 2>/dev/null
    pkill -f "websocket-server.js" 2>/dev/null
    echo "✅ All services stopped"
    exit 0
}

# Set up trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for background processes
wait