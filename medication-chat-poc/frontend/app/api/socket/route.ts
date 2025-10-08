import { Server as SocketIOServer } from 'socket.io';
import { Server as HTTPServer } from 'http';

// Store socket server instance
let io: SocketIOServer | null = null;

export async function GET() {
  if (!io) {
    // Initialize Socket.IO server
    const httpServer = new HTTPServer();
    io = new SocketIOServer(httpServer, {
      cors: {
        origin: ["http://localhost:3000", "http://localhost:3000"],
        methods: ["GET", "POST"]
      }
    });

    // Handle socket connections
    io.on('connection', (socket) => {
      console.log('Client connected:', socket.id);

      // Join chat rooms
      socket.on('join-chat', (chatId: string) => {
        socket.join(`chat-${chatId}`);
        console.log(`Socket ${socket.id} joined chat-${chatId}`);
      });

      // Handle new messages
      socket.on('new-message', (data: {
        chatId: string;
        message: {
          id: string;
          role: 'patient' | 'ai' | 'doctor';
          content: string;
          timestamp: Date;
          status?: 'pending' | 'approved' | 'rejected' | 'sent';
          aiAnalysis?: {
            riskLevel?: string;
            recommendations?: string[];
            urgency?: string;
          };
        };
      }) => {
        // Broadcast message to all clients in the chat room
        io?.to(`chat-${data.chatId}`).emit('message-update', data);
        console.log(`Message broadcasted to chat-${data.chatId}`);
      });

      // Handle message status updates
      socket.on('message-status-update', (data: {
        chatId: string;
        messageId: string;
        status: 'pending' | 'approved' | 'rejected' | 'sent';
        modifications?: string;
      }) => {
        io?.to(`chat-${data.chatId}`).emit('status-update', data);
        console.log(`Status update broadcasted to chat-${data.chatId}`);
      });

      // Handle AI analysis updates
      socket.on('ai-analysis-update', (data: {
        chatId: string;
        messageId: string;
        analysis: {
          riskLevel?: string;
          recommendations?: string[];
          urgency?: string;
        };
      }) => {
        io?.to(`chat-${data.chatId}`).emit('analysis-update', data);
        console.log(`AI analysis update broadcasted to chat-${data.chatId}`);
      });

      // Handle chat list updates
      socket.on('chat-list-update', (data: {
        chatId?: string;
        action?: string;
        timestamp?: Date;
      }) => {
        socket.broadcast.emit('chat-list-refresh', data);
      });

      socket.on('disconnect', () => {
        console.log('Client disconnected:', socket.id);
      });
    });

    // Start the server on port 3000 for WebSocket
    httpServer.listen(3000, () => {
      console.log('Socket.IO server running on port 3000');
    });
  }

  return new Response(JSON.stringify({ status: 'Socket server initialized' }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' }
  });
}