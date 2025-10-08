'use client';

import { useParams, useRouter } from 'next/navigation';
import { useState, useEffect } from 'react';
import ChatInterface from '@/components/ChatInterface';

export default function ChatRoom() {
  const params = useParams();
  const router = useRouter();
  const roomId = params.roomId as string;
  const [conversationHistory, setConversationHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  // Load existing conversation when component mounts
  useEffect(() => {
    if (roomId) {
      loadConversationHistory();
    }
  }, [roomId]);

  const loadConversationHistory = async () => {
    try {
      setLoading(true);
      // Get the chat history for this room from the doctor's active chats
      const response = await fetch('/api/doctor/chats');
      if (response.ok) {
        const data = await response.json();
        const chat = data.chats.find((c: any) => c.id === `chat-${roomId}`);

        if (chat) {
          // Convert messages to conversation history format
          const history = chat.messages.map((msg: any) => ({
            role: msg.role === 'patient' ? 'user' : msg.role,
            content: msg.content,
            timestamp: msg.timestamp
          }));
          setConversationHistory(history);
        }
      }
    } catch (error) {
      console.error('Error loading conversation history:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSendMessage = async (message: string): Promise<string> => {
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          conversationHistory,
          sessionId: roomId
        }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();

      // Add the user message and response to conversation history
      const newHistory = [
        ...conversationHistory,
        { role: 'user', content: message, timestamp: new Date().toISOString() },
        { role: 'assistant', content: data.response, timestamp: new Date().toISOString() }
      ];
      setConversationHistory(newHistory);

      return data.response;
    } catch (error) {
      console.error('Error sending message:', error);
      throw error;
    }
  };

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">กำลังโหลดห้องสนทนา...</p>
          <p className="text-sm text-gray-500">Room ID: {roomId}</p>
        </div>
      </div>
    );
  }

  return (
    <main className="h-screen">
      <div className="bg-blue-50 p-4 border-b">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-gray-800">ระบบปรึกษาแพทย์ออนไลน์</h1>
            <p className="text-sm text-gray-600">Room ID: <span className="font-mono bg-white px-2 py-1 rounded">{roomId}</span></p>
          </div>
          <div className="text-sm text-gray-600">
            <button
              onClick={() => {
                navigator.clipboard.writeText(`${window.location.origin}/chat/${roomId}`);
                alert('คัดลอก URL สำหรับกลับมาใช้งานแล้ว');
              }}
              className="bg-blue-100 hover:bg-blue-200 px-3 py-1 rounded-lg transition-colors"
            >
              📋 คัดลอก URL
            </button>
          </div>
        </div>
      </div>

      <div className="h-[calc(100vh-80px)]">
        <ChatInterface
          onSendMessage={handleSendMessage}
          initialMessages={conversationHistory}
          roomId={roomId}
        />
      </div>
    </main>
  );
}