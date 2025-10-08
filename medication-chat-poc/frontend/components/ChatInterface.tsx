'use client';

import { useState, useEffect, useRef } from 'react';
import { Send, User, Bot, AlertTriangle } from 'lucide-react';
import { Message } from '@/types/chat';

interface ChatInterfaceProps {
  onSendMessage: (message: string) => Promise<string>;
  initialMessages?: any[];
  roomId?: string;
}

export default function ChatInterface({ onSendMessage, initialMessages = [], roomId }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [chatId] = useState(() => roomId ? `chat-${roomId}` : 'chat_001'); // Use room-based chat ID
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load initial messages if provided
  useEffect(() => {
    if (initialMessages.length > 0) {
      const formattedMessages: Message[] = initialMessages.map((msg, index) => ({
        id: `msg-${index}`,
        content: msg.content,
        role: msg.role === 'user' ? 'user' : msg.role === 'doctor' ? 'assistant' : msg.role,
        timestamp: new Date(msg.timestamp || Date.now())
      }));
      setMessages(formattedMessages);
    }
  }, [initialMessages]);

  // Disabled polling to prevent conflicts with direct message handling
  // Real-time updates will be handled by direct API responses

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputMessage,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    // TODO: Send message via WebSocket for real-time sync
    // socketManager.current?.sendMessage(chatId, userMessage);

    try {
      const response = await onSendMessage(inputMessage);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response,
        timestamp: new Date(),
        status: 'pending'
      };

      setMessages(prev => [...prev, assistantMessage]);

      // TODO: Send AI response via WebSocket for real-time sync
      // socketManager.current?.sendMessage(chatId, assistantMessage);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'ขออภัย เกิดข้อผิดพลาดในระบบ กรุณาลองใหม่อีกครั้ง หรือปรึกษาแพทย์โดยตรงหากเป็นเรื่องเร่งด่วน',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };


  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Chat Section */}
      <div className="flex-1 flex flex-col">
        {/* Medical Disclaimer */}
        <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4">
          <div className="flex items-center">
            <AlertTriangle className="h-5 w-5 text-yellow-400 mr-2" />
            <p className="text-sm text-yellow-800">
              <strong>ข้อจำกัดความรับผิดชอบทางการแพทย์:</strong> ผู้ช่วย AI นี้ให้ข้อมูลสุขภาพทั่วไปเท่านั้น
              กรุณาปรึกษาผู้เชี่ยวชาญด้านสุขภาพที่มีคุณสมบัติเหมาะสมสำหรับคำแนะนำ การวินิจฉัย หรือการรักษาทางการแพทย์เสมอ
            </p>
          </div>
        </div>

        {/* Header */}
        <div className="bg-white border-b border-gray-200 p-4">
          <h1 className="text-xl font-semibold text-gray-800">ผู้ช่วยปรึกษาทางการแพทย์</h1>
          <p className="text-sm text-gray-600">อธิบายอาการและความกังวลของคุณ</p>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 && (
            <div className="text-center text-gray-500 mt-8">
              <Bot className="h-12 w-12 mx-auto mb-4 text-gray-300" />
              <p>สวัสดีครับ/ค่ะ! ผมอยู่ที่นี่เพื่อช่วยให้คุณเข้าใจอาการและให้คำแนะนำด้านสุขภาพทั่วไป</p>
              <p className="text-sm mt-2">กรุณาอธิบายว่าคุณรู้สึกอย่างไรหรืออาการที่คุณกำลังประสบอยู่</p>
            </div>
          )}

          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                  message.role === 'user'
                    ? 'bg-blue-500 text-white'
                    : 'bg-white border border-gray-200 text-gray-800'
                }`}
              >
                <div className="flex items-start space-x-2">
                  {message.role === 'assistant' && (
                    <Bot className="h-4 w-4 mt-1 text-gray-500" />
                  )}
                  {message.role === 'user' && (
                    <User className="h-4 w-4 mt-1 text-white" />
                  )}
                  <div className="flex-1">
                    <p className="whitespace-pre-wrap">{message.content}</p>
                    <p className={`text-xs mt-1 ${
                      message.role === 'user' ? 'text-blue-100' : 'text-gray-500'
                    }`}>
                      {new Date(message.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-white border border-gray-200 rounded-lg px-4 py-2">
                <div className="flex items-center space-x-2">
                  <Bot className="h-4 w-4 text-gray-500" />
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="bg-white border-t border-gray-200 p-4">
          <div className="flex space-x-2">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="อธิบายอาการของคุณ..."
              className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={isLoading}
            />
            <button
              onClick={handleSendMessage}
              disabled={!inputMessage.trim() || isLoading}
              className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>

    </div>
  );
}