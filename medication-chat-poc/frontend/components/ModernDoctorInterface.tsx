'use client';

import { useState, useEffect, useRef } from 'react';
import {
  MessageSquare,
  FileText,
  Settings,
  User,
  Search,
  Send,
  MoreVertical,
  Phone,
  Video,
  UserCircle,
  Clock,
  Check,
  X,
  Edit
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { PatientChat, PendingResponse } from '@/types/doctor';
import SocketManager from '@/lib/socket';

export default function ModernDoctorInterface() {
  const [activeSection, setActiveSection] = useState('chat');
  const [selectedChat, setSelectedChat] = useState<PatientChat | null>(null);
  const [pendingResponses, setPendingResponses] = useState<PendingResponse[]>([]);
  const [activeChats, setActiveChats] = useState<PatientChat[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null);
  const [editingContent, setEditingContent] = useState('');
  const socketManager = useRef<SocketManager | null>(null);

  // Initialize WebSocket connection
  useEffect(() => {
    socketManager.current = SocketManager.getInstance();
    socketManager.current.connect();

    // Listen for real-time updates
    socketManager.current.onMessageUpdate(() => {
      fetchActiveChats();
      fetchPendingResponses();
    });

    socketManager.current.onStatusUpdate(() => {
      fetchActiveChats();
      fetchPendingResponses();
    });

    return () => {
      socketManager.current?.disconnect();
    };
  }, []);

  // Fetch pending responses that need doctor approval
  const fetchPendingResponses = async () => {
    try {
      const response = await fetch('/api/doctor/pending');
      if (response.ok) {
        const data = await response.json();
        setPendingResponses(data.pendingResponses || []);
      }
    } catch (error) {
      console.error('Error fetching pending responses:', error);
    }
  };

  // Fetch active patient chats
  const fetchActiveChats = async () => {
    try {
      const response = await fetch('/api/doctor/chats');
      if (response.ok) {
        const data = await response.json();
        setActiveChats(data.chats || []);
      }
    } catch (error) {
      console.error('Error fetching active chats:', error);
    }
  };

  // Load data on component mount
  useEffect(() => {
    fetchPendingResponses();
    fetchActiveChats();

    // Set up polling for real-time updates
    const interval = setInterval(() => {
      fetchPendingResponses();
      fetchActiveChats();
    }, 5000); // Poll every 5 seconds

    return () => clearInterval(interval);
  }, []);

  // Handle approving AI response
  const handleApproveResponse = async (responseId: string, modifications?: string) => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/doctor/approve', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          responseId,
          action: 'approve',
          modifications
        }),
      });

      if (response.ok) {
        await fetchPendingResponses();
        await fetchActiveChats();
      }
    } catch (error) {
      console.error('Error approving response:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle rejecting AI response
  const handleRejectResponse = async (responseId: string, reason: string, newResponse?: string) => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/doctor/approve', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          responseId,
          action: 'reject',
          reason,
          newResponse
        }),
      });

      if (response.ok) {
        await fetchPendingResponses();
        await fetchActiveChats();
      }
    } catch (error) {
      console.error('Error rejecting response:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Send direct message to patient
  const handleSendDirectMessage = async () => {
    if (!newMessage.trim() || !selectedChat) return;

    setIsLoading(true);
    try {
      const response = await fetch('/api/doctor/direct-message', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          chatId: selectedChat.id,
          message: newMessage,
          doctorId: 'doctor-1' // In real app, get from auth
        }),
      });

      if (response.ok) {
        setNewMessage('');
        await fetchActiveChats();

        // Update selected chat with new message
        const updatedChat = activeChats.find(chat => chat.id === selectedChat.id);
        if (updatedChat) {
          setSelectedChat(updatedChat);
        }
      }
    } catch (error) {
      console.error('Error sending direct message:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle editing AI responses
  const handleEditMessage = (messageId: string, content: string) => {
    setEditingMessageId(messageId);
    setEditingContent(content);
  };

  const handleSaveEdit = async () => {
    if (!editingMessageId || !selectedChat) return;

    await handleApproveResponse(editingMessageId, editingContent);
    setEditingMessageId(null);
    setEditingContent('');
  };

  const sidebarSections = [
    { id: 'chat', label: 'Chat', icon: MessageSquare },
    { id: 'reports', label: 'รายงาน', icon: FileText },
    { id: 'settings', label: 'ตั้งค่า', icon: Settings }
  ];

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <div className="w-64 bg-white border-r border-gray-200 flex flex-col">
        {/* Logo/Brand */}
        <div className="p-6 border-b border-gray-200">
          <h1 className="text-xl font-bold text-gray-800">แพทย์ AI</h1>
          <p className="text-sm text-gray-500">ระบบช่วยเหลือการรักษา</p>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4">
          <div className="space-y-2">
            {sidebarSections.map((section) => {
              const Icon = section.icon;
              return (
                <button
                  key={section.id}
                  onClick={() => setActiveSection(section.id)}
                  className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg text-left transition-colors ${
                    activeSection === section.id
                      ? 'bg-blue-50 text-blue-600 border border-blue-200'
                      : 'text-gray-600 hover:bg-gray-50'
                  }`}
                >
                  <Icon className="h-5 w-5" />
                  <span className="font-medium">{section.label}</span>
                </button>
              );
            })}
          </div>
        </nav>

        {/* User Profile */}
        <div className="p-4 border-t border-gray-200">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center">
              <User className="h-6 w-6 text-white" />
            </div>
            <div>
              <p className="font-medium text-gray-800">นพ.สมเกียรติ</p>
              <p className="text-sm text-gray-500">แพทย์ประจำ</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex">
        {/* Medical Summaries Pending Approval */}
        <div className="w-80 bg-white border-r border-gray-200">
          {/* Header */}
          <div className="p-4 border-b border-gray-200">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-800">รอการอนุมัติ</h2>
              <Badge variant="secondary">
                {pendingResponses.length}
              </Badge>
            </div>

            {/* Search */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
              <input
                type="text"
                placeholder="ค้นหาผู้ป่วย..."
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>

          {/* Pending Medical Summaries */}
          <div className="overflow-y-auto h-full">
            {pendingResponses.map((response) => (
              <div
                key={response.id}
                onClick={() => {
                  const relatedChat = activeChats.find(chat => chat.id === response.chatId);
                  if (relatedChat) setSelectedChat(relatedChat);
                }}
                className={`p-4 border-b border-gray-100 cursor-pointer hover:bg-gray-50 transition-colors ${
                  selectedChat?.id === response.chatId ? 'bg-blue-50 border-blue-200' : ''
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-medium text-gray-800">ผู้ป่วย #{response.patientId}</h3>
                  <Badge
                    variant={
                      response.riskLevel === 'high' ? 'destructive' :
                      response.riskLevel === 'medium' ? 'default' : 'secondary'
                    }
                  >
                    {response.riskLevel === 'high' ? 'สูง' :
                     response.riskLevel === 'medium' ? 'ปานกลาง' : 'ต่ำ'}
                  </Badge>
                </div>
                <p className="text-sm text-gray-600 mb-2 line-clamp-2">{response.originalMessage}</p>
                <div className="flex items-center justify-between text-xs text-gray-500">
                  <span>ความมั่นใจ: {Math.round(response.confidence * 100)}%</span>
                  <span>{new Date(response.timestamp).toLocaleTimeString('th-TH')}</span>
                </div>
                {response.isUrgent && (
                  <div className="mt-2">
                    <Badge variant="destructive" className="text-xs">
                      เร่งด่วน
                    </Badge>
                  </div>
                )}
              </div>
            ))}

            {pendingResponses.length === 0 && (
              <div className="p-8 text-center text-gray-500">
                <Clock className="h-8 w-8 mx-auto mb-2 text-gray-400" />
                <p>ไม่มีรายการรอการอนุมัติ</p>
              </div>
            )}
          </div>
        </div>

        {/* Chat Area */}
        <div className="flex-1 flex flex-col">
          {selectedChat ? (
            <>
              {/* Chat Header */}
              <div className="p-4 border-b border-gray-200 bg-white">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-gray-300 rounded-full flex items-center justify-center">
                      <UserCircle className="h-6 w-6 text-gray-600" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-800">
                        {selectedChat.patientName || `ผู้ป่วย #${selectedChat.patientId}`}
                      </h3>
                      <p className="text-sm text-gray-500">
                        สถานะ: {selectedChat.status === 'active' ? 'ใช้งาน' :
                                selectedChat.status === 'waiting_approval' ? 'รอการอนุมัติ' :
                                selectedChat.status === 'completed' ? 'เสร็จสิ้น' : 'แพทย์ดูแล'}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Button variant="outline" size="sm">
                      <Phone className="h-4 w-4" />
                    </Button>
                    <Button variant="outline" size="sm">
                      <Video className="h-4 w-4" />
                    </Button>
                    <Button variant="outline" size="sm">
                      <MoreVertical className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>

              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {selectedChat.messages.map((message) => (
                  <div key={message.id}>
                    <div
                      className={`flex ${message.role === 'doctor' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div
                        className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                          message.role === 'patient'
                            ? 'bg-gray-200 text-gray-800'
                            : message.role === 'doctor'
                            ? 'bg-blue-500 text-white'
                            : 'bg-yellow-100 text-gray-800 border border-yellow-300'
                        }`}
                      >
                        {editingMessageId === message.id ? (
                          <div className="space-y-2">
                            <textarea
                              value={editingContent}
                              onChange={(e) => setEditingContent(e.target.value)}
                              className="w-full p-2 border rounded text-gray-800 text-sm"
                              rows={3}
                            />
                            <div className="flex space-x-2">
                              <Button size="sm" onClick={handleSaveEdit}>
                                <Check className="h-3 w-3 mr-1" />
                                บันทึก
                              </Button>
                              <Button size="sm" variant="outline" onClick={() => setEditingMessageId(null)}>
                                <X className="h-3 w-3 mr-1" />
                                ยกเลิก
                              </Button>
                            </div>
                          </div>
                        ) : (
                          <>
                            <p className="text-sm">{message.content}</p>
                            <div className="flex items-center justify-between mt-1">
                              <span className="text-xs opacity-70">
                                {new Date(message.timestamp).toLocaleTimeString('th-TH', {
                                  hour: '2-digit',
                                  minute: '2-digit'
                                })}
                              </span>
                              {message.status === 'pending' && message.role === 'ai' && (
                                <div className="flex items-center space-x-1">
                                  <Badge variant="secondary" className="text-xs">
                                    รอการอนุมัติ
                                  </Badge>
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    onClick={() => handleEditMessage(message.id, message.content)}
                                  >
                                    <Edit className="h-3 w-3" />
                                  </Button>
                                </div>
                              )}
                            </div>
                          </>
                        )}
                      </div>
                    </div>

                    {/* Action buttons for pending AI messages */}
                    {message.status === 'pending' && message.role === 'ai' && editingMessageId !== message.id && (
                      <div className="flex justify-center mt-2 space-x-2">
                        <Button
                          size="sm"
                          onClick={() => handleApproveResponse(message.id)}
                          disabled={isLoading}
                        >
                          <Check className="h-3 w-3 mr-1" />
                          อนุมัติ
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleEditMessage(message.id, message.content)}
                        >
                          <Edit className="h-3 w-3 mr-1" />
                          แก้ไข
                        </Button>
                        <Button
                          size="sm"
                          variant="destructive"
                          onClick={() => handleRejectResponse(message.id, 'ไม่เหมาะสม')}
                          disabled={isLoading}
                        >
                          <X className="h-3 w-3 mr-1" />
                          ปฏิเสธ
                        </Button>
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {/* Message Input */}
              <div className="p-4 border-t border-gray-200 bg-white">
                <div className="flex space-x-2">
                  <input
                    type="text"
                    value={newMessage}
                    onChange={(e) => setNewMessage(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSendDirectMessage()}
                    placeholder="พิมพ์ข้อความถึงผู้ป่วย..."
                    className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    disabled={isLoading}
                  />
                  <Button onClick={handleSendDirectMessage} disabled={isLoading}>
                    <Send className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center bg-gray-50">
              <div className="text-center">
                <MessageSquare className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-600 mb-2">เลือกผู้ป่วย</h3>
                <p className="text-gray-500">เลือกผู้ป่วยจากรายชื่อเพื่อเริ่มการสนทนา</p>
              </div>
            </div>
          )}
        </div>

        {/* Medical Summary Panel */}
        {selectedChat && (
          <div className="w-96 bg-white border-l border-gray-200 p-4 overflow-y-auto">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">สรุปทางการแพทย์</h3>

            {/* Find the pending AI response for this chat */}
            {(() => {
              const pendingResponse = pendingResponses.find(r => r.chatId === selectedChat.id);

              if (!pendingResponse && selectedChat.summary) {
                return (
                  <Card className="mb-4">
                    <CardContent className="p-4">
                      <div className="space-y-4">
                        {/* Assessment Results */}
                        <div>
                          <h4 className="font-medium text-gray-700 mb-2 border-b pb-1">ผลการประเมิน</h4>
                          <p className="text-sm text-gray-600">
                            การประเมินเบื้องต้นจากอาการที่ผู้ป่วยรายงาน
                          </p>
                        </div>

                        {/* Symptoms */}
                        <div>
                          <h4 className="font-medium text-gray-700 mb-2">อาการ</h4>
                          <ul className="space-y-1">
                            {selectedChat.summary.symptoms.map((symptom, index) => (
                              <li key={index} className="flex items-start text-sm text-gray-600">
                                <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                                {symptom}
                              </li>
                            ))}
                          </ul>
                        </div>

                        {/* Possible Conditions */}
                        <div>
                          <h4 className="font-medium text-gray-700 mb-2">โรคที่เป็นไปได้</h4>
                          <ul className="space-y-1">
                            {selectedChat.summary.conditions.map((condition, index) => (
                              <li key={index} className="flex items-start text-sm text-gray-600">
                                <span className="w-2 h-2 bg-yellow-500 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                                {condition}
                              </li>
                            ))}
                          </ul>
                        </div>

                        {/* Recommendations */}
                        <div>
                          <h4 className="font-medium text-gray-700 mb-2">คำแนะนำ</h4>
                          <ul className="space-y-1">
                            {selectedChat.summary.medications.map((medication, index) => (
                              <li key={index} className="flex items-start text-sm text-gray-600">
                                <span className="w-2 h-2 bg-green-500 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                                {medication}
                              </li>
                            ))}
                          </ul>
                        </div>

                        {/* Urgency Level */}
                        <div className={`p-3 rounded-lg ${
                          selectedChat.status === 'doctor_takeover' ? 'bg-red-50 border border-red-200' :
                          selectedChat.status === 'waiting_approval' ? 'bg-yellow-50 border border-yellow-200' :
                          'bg-green-50 border border-green-200'
                        }`}>
                          <h4 className="font-medium text-gray-700 mb-1">ระดับความเร่งด่วน</h4>
                          <span className={`text-sm font-medium ${
                            selectedChat.status === 'doctor_takeover' ? 'text-red-600' :
                            selectedChat.status === 'waiting_approval' ? 'text-yellow-600' :
                            'text-green-600'
                          }`}>
                            {selectedChat.status === 'doctor_takeover' ? 'เร่งด่วน' :
                             selectedChat.status === 'waiting_approval' ? 'รอการอนุมัติ' : 'ปกติ'}
                          </span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                );
              }

              if (pendingResponse) {
                return (
                  <Card className="mb-4">
                    <CardContent className="p-4">
                      <div className="space-y-4">
                        {/* Assessment Results */}
                        <div>
                          <h4 className="font-medium text-gray-700 mb-2 border-b pb-1">ผลการประเมิน</h4>
                          <p className="text-sm text-gray-600 mb-2">
                            การประเมินโดย AI จากข้อมูลที่มีอยู่
                          </p>
                          <div className="flex items-center justify-between text-xs">
                            <span>ความมั่นใจ: {Math.round(pendingResponse.confidence * 100)}%</span>
                            <Badge variant={
                              pendingResponse.riskLevel === 'high' ? 'destructive' :
                              pendingResponse.riskLevel === 'medium' ? 'default' : 'secondary'
                            }>
                              ความเสี่ยง: {pendingResponse.riskLevel === 'high' ? 'สูง' :
                                         pendingResponse.riskLevel === 'medium' ? 'ปานกลาง' : 'ต่ำ'}
                            </Badge>
                          </div>
                        </div>

                        {/* Patient Query */}
                        <div>
                          <h4 className="font-medium text-gray-700 mb-2">คำถามของผู้ป่วย</h4>
                          <p className="text-sm text-gray-600 bg-gray-50 p-2 rounded">
                            {pendingResponse.originalMessage}
                          </p>
                        </div>

                        {/* AI Response */}
                        <div>
                          <h4 className="font-medium text-gray-700 mb-2">คำตอบของ AI</h4>
                          <div className="text-sm text-gray-600 bg-yellow-50 p-3 rounded border">
                            {pendingResponse.aiResponse}
                          </div>
                        </div>

                        {/* Detected Conditions */}
                        {pendingResponse.aiAnalysis.detectedConditions.length > 0 && (
                          <div>
                            <h4 className="font-medium text-gray-700 mb-2">โรคที่ตรวจพบ</h4>
                            <ul className="space-y-1">
                              {pendingResponse.aiAnalysis.detectedConditions.map((condition, index) => (
                                <li key={index} className="flex items-start text-sm text-gray-600">
                                  <span className="w-2 h-2 bg-orange-500 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                                  <div>
                                    {typeof condition === 'string' ? condition : (
                                      <>
                                        <div className="font-medium">{condition.thai_name || condition.english_name}</div>
                                        {condition.icd_code && (
                                          <div className="text-xs text-gray-500">ICD: {condition.icd_code} ({condition.confidence}%)</div>
                                        )}
                                      </>
                                    )}
                                  </div>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {/* Suggested Medications */}
                        {pendingResponse.aiAnalysis.suggestedMedications.length > 0 && (
                          <div>
                            <h4 className="font-medium text-gray-700 mb-2">ยาที่แนะนำ</h4>
                            <ul className="space-y-1">
                              {pendingResponse.aiAnalysis.suggestedMedications.map((medication, index) => (
                                <li key={index} className="flex items-start text-sm text-gray-600">
                                  <span className="w-2 h-2 bg-purple-500 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                                  {medication.drugName} - {medication.frequency}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {/* RAG Evidence */}
                        <div className="bg-blue-50 p-3 rounded border">
                          <h4 className="font-medium text-gray-700 mb-2">หลักฐานจากฐานข้อมูล</h4>
                          <div className="text-xs text-gray-600 space-y-1">
                            <div>คะแนนความเชื่อถือ: {pendingResponse.aiAnalysis.ragScore}%</div>
                            {pendingResponse.aiAnalysis.suggestedMedications.length > 0 && (
                              <div>ยาที่พบ: {pendingResponse.aiAnalysis.suggestedMedications.map(m => m.drugName).join(', ')}</div>
                            )}
                            <div>รหัส ICD: มี</div>
                            <div>ขนาดยา: ระบุ</div>
                          </div>
                        </div>

                        {/* Action Buttons */}
                        <div className="flex space-x-2 pt-4 border-t">
                          <Button
                            className="flex-1"
                            onClick={() => handleApproveResponse(pendingResponse.id)}
                            disabled={isLoading}
                          >
                            <Check className="h-4 w-4 mr-2" />
                            อนุมัติ
                          </Button>
                          <Button
                            variant="destructive"
                            className="flex-1"
                            onClick={() => handleRejectResponse(pendingResponse.id, 'ข้อมูลไม่ถูกต้อง')}
                            disabled={isLoading}
                          >
                            <X className="h-4 w-4 mr-2" />
                            ปฏิเสธ
                          </Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                );
              }

              return (
                <div className="p-8 text-center text-gray-500">
                  <FileText className="h-8 w-8 mx-auto mb-2 text-gray-400" />
                  <p>ไม่มีข้อมูลการประเมิน</p>
                </div>
              );
            })()}

            {/* Patient Info */}
            <Card>
              <CardContent className="p-4">
                <h4 className="font-medium text-gray-700 mb-3">ข้อมูลผู้ป่วย</h4>
                <div className="space-y-2 text-sm">
                  <div>
                    <span className="text-gray-600">รหัสผู้ป่วย:</span>
                    <span className="ml-2 text-gray-800">{selectedChat.patientId}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">ชื่อ:</span>
                    <span className="ml-2 text-gray-800">{selectedChat.patientName || 'ไม่ระบุ'}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">กิจกรรมล่าสุด:</span>
                    <span className="ml-2 text-gray-800">
                      {new Date(selectedChat.lastActivity).toLocaleString('th-TH')}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-600">สถานะ:</span>
                    <span className="ml-2">
                      <Badge variant={
                        selectedChat.status === 'active' ? 'default' :
                        selectedChat.status === 'waiting_approval' ? 'secondary' : 'outline'
                      }>
                        {selectedChat.status === 'active' ? 'ใช้งาน' :
                         selectedChat.status === 'waiting_approval' ? 'รอการอนุมัติ' :
                         selectedChat.status === 'completed' ? 'เสร็จสิ้น' : 'แพทย์ดูแล'}
                      </Badge>
                    </span>
                  </div>
                </div>

                {selectedChat.doctorNotes && (
                  <div className="mt-4 pt-4 border-t">
                    <span className="text-gray-600 text-sm">บันทึกของแพทย์:</span>
                    <p className="mt-1 text-sm text-gray-800 bg-blue-50 p-2 rounded">
                      {selectedChat.doctorNotes}
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}