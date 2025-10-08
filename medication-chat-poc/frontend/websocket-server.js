import { Server } from 'socket.io';
import { createServer } from 'http';

// Environment configuration
const PORT = process.env.WEBSOCKET_PORT || 3001;
const NODE_ENV = process.env.NODE_ENV || 'development';
const CORS_ORIGIN = process.env.WEBSOCKET_CORS_ORIGIN || "*";

// Create HTTP server
const server = createServer();

// Configure CORS origins based on environment
let corsOrigins;
if (NODE_ENV === 'production') {
  corsOrigins = CORS_ORIGIN === "*" ? "*" : CORS_ORIGIN.split(',');
} else {
  corsOrigins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3002"
  ];
}

// Create Socket.IO server with production-ready configuration
const io = new Server(server, {
  cors: {
    origin: corsOrigins,
    methods: ["GET", "POST"],
    credentials: true
  },
  pingTimeout: 60000,
  pingInterval: 25000,
  transports: ['websocket', 'polling'],
  allowEIO3: true
});

console.log(`WebSocket server starting on port ${PORT}...`);
console.log(`Environment: ${NODE_ENV}`);
console.log(`CORS Origins:`, corsOrigins);

// Handle socket connections
io.on('connection', (socket) => {
  console.log(`Client connected: ${socket.id}`);

  // Join chat rooms
  socket.on('join-chat', (chatId) => {
    socket.join(`chat-${chatId}`);
    console.log(`Socket ${socket.id} joined chat-${chatId}`);
  });

  // Handle new messages
  socket.on('new-message', (data) => {
    const { chatId, message } = data;
    console.log(`New message in chat-${chatId}:`, message.content.substring(0, 50) + '...');
    // Broadcast message to all clients in the chat room
    io.to(`chat-${chatId}`).emit('message-update', data);
  });

  // Handle message status updates
  socket.on('message-status-update', (data) => {
    const { chatId, messageId, status } = data;
    console.log(`Status update in chat-${chatId}: message ${messageId} -> ${status}`);
    io.to(`chat-${chatId}`).emit('status-update', data);
  });

  // Handle AI analysis updates
  socket.on('ai-analysis-update', (data) => {
    const { chatId, messageId } = data;
    console.log(`AI analysis update in chat-${chatId} for message ${messageId}`);
    io.to(`chat-${chatId}`).emit('analysis-update', data);
  });

  // Handle chat list updates
  socket.on('chat-list-update', (data) => {
    console.log('Chat list update received');
    socket.broadcast.emit('chat-list-refresh', data);
  });

  socket.on('disconnect', () => {
    console.log(`Client disconnected: ${socket.id}`);
  });
});

// Start the server
server.listen(PORT, '0.0.0.0', () => {
  console.log(`✅ WebSocket server running on http://0.0.0.0:${PORT}`);
  console.log('🔄 Ready for real-time chat synchronization between patient and doctor interfaces');
});

// Handle server shutdown gracefully
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down WebSocket server...');
  server.close(() => {
    console.log('✅ WebSocket server closed');
    process.exit(0);
  });
});