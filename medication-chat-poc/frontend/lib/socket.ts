import { io, Socket } from 'socket.io-client';

class SocketManager {
  private socket: Socket | null = null;
  private static instance: SocketManager;

  private constructor() {}

  static getInstance(): SocketManager {
    if (!SocketManager.instance) {
      SocketManager.instance = new SocketManager();
    }
    return SocketManager.instance;
  }

  connect(): Socket {
    if (!this.socket) {
      this.socket = io('http://localhost:3001', {
        transports: ['websocket'],
        autoConnect: true
      });

      this.socket.on('connect', () => {
        console.log('Connected to WebSocket server');
      });

      this.socket.on('disconnect', () => {
        console.log('Disconnected from WebSocket server');
      });
    }
    return this.socket;
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  getSocket(): Socket | null {
    return this.socket;
  }

  // Chat synchronization methods
  joinChat(chatId: string) {
    if (this.socket) {
      this.socket.emit('join-chat', chatId);
    }
  }

  sendMessage(chatId: string, message: {
    id: string;
    role: 'user' | 'assistant' | 'patient' | 'ai' | 'doctor';
    content: string;
    timestamp: Date;
    status?: 'pending' | 'approved' | 'rejected' | 'sent';
    aiAnalysis?: {
      riskLevel?: string;
      recommendations?: string[];
      urgency?: string;
    };
  }) {
    if (this.socket) {
      this.socket.emit('new-message', { chatId, message });
    }
  }

  updateMessageStatus(chatId: string, messageId: string, status: string, modifications?: string) {
    if (this.socket) {
      this.socket.emit('message-status-update', { chatId, messageId, status, modifications });
    }
  }

  updateAIAnalysis(chatId: string, messageId: string, analysis: {
    riskLevel?: string;
    recommendations?: string[];
    urgency?: string;
  }) {
    if (this.socket) {
      this.socket.emit('ai-analysis-update', { chatId, messageId, analysis });
    }
  }

  // Event listeners
  onMessageUpdate(callback: (data: {
    chatId: string;
    message: {
      id: string;
      role: 'user' | 'assistant' | 'patient' | 'ai' | 'doctor';
      content: string;
      timestamp: Date;
      status?: 'pending' | 'approved' | 'rejected' | 'sent';
      aiAnalysis?: {
        riskLevel?: string;
        recommendations?: string[];
        urgency?: string;
      };
    };
  }) => void) {
    if (this.socket) {
      this.socket.on('message-update', callback);
    }
  }

  onStatusUpdate(callback: (data: {
    chatId: string;
    messageId: string;
    status: 'pending' | 'approved' | 'rejected' | 'sent';
    modifications?: string;
  }) => void) {
    if (this.socket) {
      this.socket.on('status-update', callback);
    }
  }

  onAnalysisUpdate(callback: (data: {
    chatId: string;
    messageId: string;
    analysis: {
      riskLevel?: string;
      recommendations?: string[];
    };
  }) => void) {
    if (this.socket) {
      this.socket.on('analysis-update', callback);
    }
  }

  onChatListRefresh(callback: (data: {
    chatId?: string;
    action?: string;
    timestamp?: Date;
  }) => void) {
    if (this.socket) {
      this.socket.on('chat-list-refresh', callback);
    }
  }
}

export default SocketManager;