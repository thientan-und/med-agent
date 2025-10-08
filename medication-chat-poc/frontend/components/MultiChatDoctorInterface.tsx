'use client';

import { useState, useEffect, useRef } from 'react';
import {
  MessageSquare,
  User,
  Bot,
  Stethoscope,
  Edit3,
  Save,
  X,
  Send,
  CheckCircle,
  XCircle,
  RefreshCw
} from 'lucide-react';
import SocketManager from '@/lib/socket';

// shadcn/ui components
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'patient' | 'ai' | 'doctor';
  content: string;
  timestamp: Date;
  status?: 'pending' | 'approved' | 'rejected' | 'sent';
  aiAnalysis?: {
    confidence?: number;
    detectedConditions?: string[];
    suggestedMedications?: string[];
    riskLevel?: 'low' | 'medium' | 'high' | string;
    ragScore?: number;
    recommendations?: string[];
    urgency?: string;
  };
  doctorNotes?: string;
}

interface PatientChat {
  id: string;
  patientId: string;
  patientName: string;
  status: 'active' | 'waiting_approval' | 'completed' | 'doctor_takeover';
  messages: ChatMessage[];
  lastActivity: Date;
  unreadCount: number;
}

export default function MultiChatDoctorInterface() {
  const [chats, setChats] = useState<PatientChat[]>([]);
  const [selectedChatId, setSelectedChatId] = useState<string | null>(null);
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(null);
  const [editingAnalysis, setEditingAnalysis] = useState(false);
  const [analysisNotes, setAnalysisNotes] = useState('');
  const [newMessage, setNewMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const socketManager = useRef<SocketManager | null>(null);

  // Load real chat data from API
  useEffect(() => {
    const loadChats = async () => {
      try {
        // Try to fetch real chats from API
        const response = await fetch('/api/doctor/chats');
        if (response.ok) {
          const data = await response.json();
          setChats(data.chats || []);
          if (data.chats && data.chats.length > 0) {
            setSelectedChatId(data.chats[0].id);
          }
        } else {
          // If no API data, start with empty state
          setChats([]);
        }
      } catch (error) {
        console.error('Error loading chats:', error);
        setChats([]);
      }
    };

    loadChats();
  }, []);

  // Initialize WebSocket connection for real-time sync
  useEffect(() => {
    socketManager.current = SocketManager.getInstance();
    socketManager.current.connect();

    // Join all chat rooms for monitoring
    chats.forEach(chat => {
      socketManager.current?.joinChat(chat.id);
    });

    // Listen for message updates from patient interface
    socketManager.current.onMessageUpdate((data) => {
      const { chatId, message } = data;
      setChats(prevChats =>
        prevChats.map(chat => {
          if (chat.id === chatId) {
            const messageExists = chat.messages.find(m => m.id === message.id);
            if (!messageExists) {
              return {
                ...chat,
                messages: [...chat.messages, message],
                unreadCount: chat.unreadCount + 1,
                lastActivity: new Date()
              };
            }
            return {
              ...chat,
              messages: chat.messages.map(m => m.id === message.id ? message : m)
            };
          }
          return chat;
        })
      );
    });

    // Listen for status updates
    socketManager.current.onStatusUpdate((data) => {
      const { chatId, messageId, status } = data;
      setChats(prevChats =>
        prevChats.map(chat => {
          if (chat.id === chatId) {
            return {
              ...chat,
              messages: chat.messages.map(m =>
                m.id === messageId ? { ...m, status } : m
              )
            };
          }
          return chat;
        })
      );
    });

    return () => {
      socketManager.current?.disconnect();
    };
  }, [chats]);

  const selectedChat = chats.find(chat => chat.id === selectedChatId);
  const selectedMessage = selectedChat?.messages.find(msg => msg.id === selectedMessageId);

  const handleApproveMessage = async (messageId: string) => {
    setIsLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));

      setChats(prev => prev.map(chat => {
        const updatedChat = {
          ...chat,
          messages: chat.messages.map(msg =>
            msg.id === messageId ? { ...msg, status: 'approved' as const } : msg
          )
        };

        // Broadcast status update via WebSocket
        if (chat.messages.some(m => m.id === messageId)) {
          socketManager.current?.updateMessageStatus(chat.id, messageId, 'approved');
        }

        return updatedChat;
      }));
    } finally {
      setIsLoading(false);
    }
  };

  const handleRejectMessage = async (messageId: string) => {
    setIsLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));

      setChats(prev => prev.map(chat => {
        const updatedChat = {
          ...chat,
          messages: chat.messages.map(msg =>
            msg.id === messageId ? { ...msg, status: 'rejected' as const } : msg
          )
        };

        // Broadcast status update via WebSocket
        if (chat.messages.some(m => m.id === messageId)) {
          socketManager.current?.updateMessageStatus(chat.id, messageId, 'rejected');
        }

        return updatedChat;
      }));
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async () => {
    if (!newMessage.trim() || !selectedChatId) return;

    const message: ChatMessage = {
      id: `msg_${Date.now()}`,
      role: 'doctor',
      content: newMessage,
      timestamp: new Date(),
      status: 'sent'
    };

    setChats(prev => prev.map(chat =>
      chat.id === selectedChatId
        ? {
            ...chat,
            messages: [...chat.messages, message],
            status: 'doctor_takeover' as const,
            lastActivity: new Date()
          }
        : chat
    ));

    setNewMessage('');
  };


  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'default';
      case 'waiting_approval': return 'secondary';
      case 'doctor_takeover': return 'outline';
      case 'completed': return 'secondary';
      default: return 'outline';
    }
  };

  return (
    <div className="h-screen flex bg-background">
      {/* Chat List Sidebar */}
      <div className="w-80 border-r flex flex-col">
        <div className="p-4 border-b">
          <h2 className="text-lg font-semibold mb-2">Patient Conversations</h2>
          <div className="flex space-x-2 text-sm">
            <Badge variant="default">{chats.length} Active</Badge>
            <Badge variant="secondary">{chats.filter(c => c.status === 'waiting_approval').length} Pending</Badge>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto">
          {chats.map((chat) => (
            <div
              key={chat.id}
              onClick={() => setSelectedChatId(chat.id)}
              className={`p-4 border-b cursor-pointer hover:bg-muted/50 transition-colors ${
                selectedChatId === chat.id ? 'bg-muted' : ''
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                  <User className="w-4 h-4 text-muted-foreground" />
                  <span className="font-medium text-sm">{chat.patientName}</span>
                  {chat.unreadCount > 0 && (
                    <Badge variant="destructive" className="text-xs">
                      {chat.unreadCount}
                    </Badge>
                  )}
                </div>
              </div>

              <div className="flex items-center justify-between mb-1">
                <Badge variant={getStatusColor(chat.status)} className="text-xs">
                  {chat.status.replace('_', ' ')}
                </Badge>
                <span className="text-xs text-muted-foreground">
                  {new Date(chat.lastActivity).toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit' })}
                </span>
              </div>

              <p className="text-sm text-muted-foreground truncate">
                {chat.messages[chat.messages.length - 1]?.content || 'No messages'}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {selectedChat ? (
          <>
            {/* Chat Header */}
            <div className="p-4 border-b">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold">{selectedChat.patientName}</h3>
                  <p className="text-sm text-muted-foreground">Patient ID: {selectedChat.patientId}</p>
                </div>
                <div className="flex items-center space-x-2">
                  <Badge variant={getStatusColor(selectedChat.status)}>
                    {selectedChat.status.replace('_', ' ')}
                  </Badge>
                  <Button variant="outline" size="sm">
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Refresh
                  </Button>
                </div>
              </div>
            </div>

            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {selectedChat.messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.role === 'patient' ? 'justify-start' : 'justify-end'}`}
                >
                  <div
                    className={`max-w-2xl p-3 rounded-lg cursor-pointer transition-all ${
                      message.role === 'patient'
                        ? 'bg-muted'
                        : message.role === 'ai'
                        ? 'bg-blue-50 border border-blue-200'
                        : 'bg-green-50 border border-green-200'
                    } ${
                      selectedMessageId === message.id ? 'ring-2 ring-blue-500' : ''
                    }`}
                    onClick={() => setSelectedMessageId(message.id)}
                  >
                    <div className="flex items-center space-x-2 mb-2">
                      {message.role === 'patient' ? (
                        <User className="w-4 h-4" />
                      ) : message.role === 'ai' ? (
                        <Bot className="w-4 h-4" />
                      ) : (
                        <Stethoscope className="w-4 h-4" />
                      )}
                      <span className="text-sm font-medium capitalize">{message.role}</span>
                      <span className="text-xs text-muted-foreground">
                        {new Date(message.timestamp).toLocaleTimeString('th-TH')}
                      </span>
                      {message.status && (
                        <Badge variant="outline" className="text-xs">
                          {message.status}
                        </Badge>
                      )}
                    </div>

                    <div className="whitespace-pre-wrap text-sm">
                      {message.content}
                    </div>

                    {message.role === 'ai' && message.status === 'pending' && (
                      <div className="flex space-x-2 mt-3">
                        <Button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleApproveMessage(message.id);
                          }}
                          disabled={isLoading}
                          size="sm"
                          variant="default"
                        >
                          <CheckCircle className="w-3 h-3 mr-1" />
                          Approve
                        </Button>
                        <Button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleRejectMessage(message.id);
                          }}
                          disabled={isLoading}
                          size="sm"
                          variant="destructive"
                        >
                          <XCircle className="w-3 h-3 mr-1" />
                          Reject
                        </Button>
                        <Button
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedMessageId(message.id);
                          }}
                          size="sm"
                          variant="outline"
                        >
                          <Edit3 className="w-3 h-3 mr-1" />
                          Edit
                        </Button>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* Message Input */}
            <div className="p-4 border-t">
              <div className="flex space-x-2">
                <Textarea
                  value={newMessage}
                  onChange={(e) => setNewMessage(e.target.value)}
                  placeholder="Type your message to the patient..."
                  className="flex-1"
                  rows={2}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSendMessage();
                    }
                  }}
                />
                <Button
                  onClick={handleSendMessage}
                  disabled={!newMessage.trim()}
                  className="self-end"
                >
                  <Send className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <MessageSquare className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-medium mb-2">Select a conversation</h3>
              <p className="text-muted-foreground">Choose a patient conversation to start monitoring</p>
            </div>
          </div>
        )}
      </div>

      {/* AI Analysis Sidebar */}
      {selectedMessage?.role === 'ai' && (
        <div className="w-96 border-l flex flex-col">
          <div className="p-4 border-b">
            <h3 className="text-lg font-semibold mb-2">AI Response Analysis</h3>
            <div className="flex space-x-2">
              <Badge variant={selectedMessage.aiAnalysis?.riskLevel === 'high' ? 'destructive' : 'default'}>
                {selectedMessage.aiAnalysis?.riskLevel} Risk
              </Badge>
              <Badge variant="outline">
                {selectedMessage.aiAnalysis?.confidence}% Confidence
              </Badge>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {selectedMessage.aiAnalysis && (
              <>
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Analysis Metrics</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Confidence:</span>
                      <span>{selectedMessage.aiAnalysis.confidence || 0}%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>RAG Score:</span>
                      <span>{selectedMessage.aiAnalysis.ragScore || 0}%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Risk Level:</span>
                      <Badge variant={selectedMessage.aiAnalysis.riskLevel === 'high' ? 'destructive' : 'default'}>
                        {selectedMessage.aiAnalysis.riskLevel || 'unknown'}
                      </Badge>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Detected Conditions</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-1">
                      {selectedMessage.aiAnalysis.detectedConditions?.map((condition, index) => (
                        <Badge key={index} variant="outline" className="mr-1 mb-1">
                          {condition}
                        </Badge>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Suggested Medications</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-1">
                      {selectedMessage.aiAnalysis.suggestedMedications?.map((medication, index) => (
                        <Badge key={index} variant="secondary" className="mr-1 mb-1">
                          {medication}
                        </Badge>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </>
            )}

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center justify-between">
                  Doctor Notes
                  <Button
                    onClick={() => setEditingAnalysis(!editingAnalysis)}
                    size="sm"
                    variant="outline"
                  >
                    <Edit3 className="w-3 h-3" />
                  </Button>
                </CardTitle>
              </CardHeader>
              <CardContent>
                {editingAnalysis ? (
                  <div className="space-y-2">
                    <Textarea
                      value={analysisNotes}
                      onChange={(e) => setAnalysisNotes(e.target.value)}
                      placeholder="Add your analysis notes..."
                      rows={4}
                    />
                    <div className="flex space-x-2">
                      <Button size="sm" onClick={() => setEditingAnalysis(false)}>
                        <Save className="w-3 h-3 mr-1" />
                        Save
                      </Button>
                      <Button size="sm" variant="outline" onClick={() => setEditingAnalysis(false)}>
                        <X className="w-3 h-3 mr-1" />
                        Cancel
                      </Button>
                    </div>
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    {selectedMessage.doctorNotes || analysisNotes || 'No notes added yet. Click edit to add your analysis.'}
                  </p>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}