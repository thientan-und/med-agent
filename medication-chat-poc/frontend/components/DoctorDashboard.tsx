'use client';

import { useState } from 'react';
import {
  CheckCircle,
  XCircle,
  Clock,
  AlertTriangle,
  MessageSquare,
  User,
  Bot,
  Edit3,
  Send,
  Eye,
  Stethoscope
} from 'lucide-react';
import { PendingResponse, PatientChat } from '@/types/doctor';

// shadcn/ui components
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Textarea } from '@/components/ui/textarea';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog';

interface DoctorDashboardProps {
  pendingResponses: PendingResponse[];
  activeChats: PatientChat[];
  onApproveResponse: (responseId: string, modifications?: string) => Promise<void>;
  onRejectResponse: (responseId: string, reason: string, newResponse?: string) => Promise<void>;
  onSendDirectMessage: (chatId: string, message: string) => Promise<void>;
  isLoading: boolean;
}

export default function DoctorDashboard({
  pendingResponses,
  activeChats,
  onApproveResponse,
  onRejectResponse,
  onSendDirectMessage,
  isLoading
}: DoctorDashboardProps) {
  // Removed unused state variables per ESLint warnings
  const [modifications, setModifications] = useState('');
  const [rejectionReason, setRejectionReason] = useState('');
  const [newResponse, setNewResponse] = useState('');
  const [directMessage, setDirectMessage] = useState('');

  const getRiskVariant = (risk: string) => {
    switch (risk) {
      case 'high': return 'destructive';
      case 'medium': return 'default';
      case 'low': return 'secondary';
      default: return 'outline';
    }
  };

  const getStatusVariant = (status: string) => {
    switch (status) {
      case 'active': return 'default';
      case 'waiting_approval': return 'secondary';
      case 'doctor_takeover': return 'outline';
      case 'completed': return 'secondary';
      default: return 'outline';
    }
  };

  const formatTimestamp = (timestamp: Date) => {
    return new Date(timestamp).toLocaleString('th-TH');
  };

  const handleApprove = async (responseId: string) => {
    await onApproveResponse(responseId, modifications);
    // Removed setSelectedResponse call - state variable removed
    setModifications('');
  };

  const handleReject = async (responseId: string) => {
    if (!rejectionReason.trim()) {
      alert('กรุณาระบุเหตุผลในการปฏิเสธ');
      return;
    }
    await onRejectResponse(responseId, rejectionReason, newResponse);
    // Removed setSelectedResponse call - state variable removed
    setRejectionReason('');
    setNewResponse('');
  };

  const handleSendMessage = async (chatId: string) => {
    if (!directMessage.trim()) return;
    await onSendDirectMessage(chatId, directMessage);
    setDirectMessage('');
    // Removed setSelectedChat call - state variable removed
  };

  return (
    <div className="space-y-6">
      {/* Header Stats */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">รอการอนุมัติ</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{pendingResponses.length}</div>
            <p className="text-xs text-muted-foreground">
              {pendingResponses.filter(r => r.isUrgent).length} เร่งด่วน
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">การสนทนาที่ใช้งานอยู่</CardTitle>
            <MessageSquare className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{activeChats.length}</div>
            <p className="text-xs text-muted-foreground">
              {activeChats.filter(c => c.status === 'waiting_approval').length} รอการตรวจสอบ
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">ความแม่นยำ AI</CardTitle>
            <Stethoscope className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">87%</div>
            <p className="text-xs text-muted-foreground">24 ชั่วโมงที่ผ่านมา</p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="pending" className="space-y-4">
        <TabsList>
          <TabsTrigger value="pending" className="flex items-center space-x-2">
            <Clock className="w-4 h-4" />
            <span>Pending Approval ({pendingResponses.length})</span>
          </TabsTrigger>
          <TabsTrigger value="chats" className="flex items-center space-x-2">
            <MessageSquare className="w-4 h-4" />
            <span>Active Chats ({activeChats.length})</span>
          </TabsTrigger>
        </TabsList>

        {/* Pending Responses Tab */}
        <TabsContent value="pending" className="space-y-4">
          {pendingResponses.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <CheckCircle className="h-12 w-12 text-green-500 mb-4" />
                <CardTitle className="text-lg mb-2">No pending responses</CardTitle>
                <CardDescription>All AI responses have been reviewed.</CardDescription>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-4">
              {pendingResponses.map((response) => (
                <Card key={response.id} className="relative">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <Badge variant={getRiskVariant(response.riskLevel)}>
                          {response.riskLevel.toUpperCase()} RISK
                        </Badge>
                        <span className="text-sm text-muted-foreground">
                          {formatTimestamp(response.timestamp)}
                        </span>
                        {response.isUrgent && (
                          <Badge variant="destructive" className="animate-pulse">
                            <AlertTriangle className="w-3 h-3 mr-1" />
                            URGENT
                          </Badge>
                        )}
                      </div>
                    </div>
                  </CardHeader>

                  <CardContent className="space-y-4">
                    <div>
                      <h4 className="font-medium mb-2">Patient Query:</h4>
                      <div className="bg-muted p-3 rounded text-sm">
                        {response.originalMessage}
                      </div>
                    </div>

                    <div>
                      <h4 className="font-medium mb-2">AI Response:</h4>
                      <div className="bg-blue-50 border border-blue-200 p-3 rounded text-sm">
                        <pre className="whitespace-pre-wrap">{response.aiResponse}</pre>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <span className="font-medium">RAG Score:</span>
                        <p className="text-muted-foreground">{response.aiAnalysis.ragScore}%</p>
                      </div>
                      <div>
                        <span className="font-medium">Confidence:</span>
                        <p className="text-muted-foreground">{response.confidence}%</p>
                      </div>
                      <div>
                        <span className="font-medium">Conditions:</span>
                        <p className="text-muted-foreground">
                          {response.aiAnalysis.detectedConditions.join(', ') || 'None'}
                        </p>
                      </div>
                      <div>
                        <span className="font-medium">Medications:</span>
                        <p className="text-muted-foreground">
                          {response.aiAnalysis.suggestedMedications.map(med => med.drugName).join(', ') || 'None'}
                        </p>
                      </div>
                    </div>

                    <div className="flex space-x-2">
                      <Button
                        onClick={() => handleApprove(response.id)}
                        disabled={isLoading}
                        variant="default"
                        size="sm"
                      >
                        <CheckCircle className="w-4 h-4 mr-2" />
                        Quick Approve
                      </Button>

                      <Dialog>
                        <DialogTrigger asChild>
                          <Button variant="outline" size="sm">
                            <Edit3 className="w-4 h-4 mr-2" />
                            Review & Edit
                          </Button>
                        </DialogTrigger>
                        <DialogContent className="max-w-2xl">
                          <DialogHeader>
                            <DialogTitle>Review AI Response</DialogTitle>
                            <DialogDescription>
                              Review and optionally modify the AI response before approval
                            </DialogDescription>
                          </DialogHeader>
                          <div className="space-y-4">
                            <div>
                              <label className="text-sm font-medium">Modifications (optional):</label>
                              <Textarea
                                value={modifications}
                                onChange={(e) => setModifications(e.target.value)}
                                placeholder="Any modifications to the AI response..."
                                className="mt-1"
                              />
                            </div>
                            <div>
                              <label className="text-sm font-medium">Rejection Reason (if rejecting):</label>
                              <Textarea
                                value={rejectionReason}
                                onChange={(e) => setRejectionReason(e.target.value)}
                                placeholder="Reason for rejection..."
                                className="mt-1"
                              />
                            </div>
                            <div>
                              <label className="text-sm font-medium">Alternative Response (if rejecting):</label>
                              <Textarea
                                value={newResponse}
                                onChange={(e) => setNewResponse(e.target.value)}
                                placeholder="Your alternative response..."
                                className="mt-1"
                                rows={4}
                              />
                            </div>
                          </div>
                          <DialogFooter className="space-x-2">
                            <Button
                              onClick={() => handleApprove(response.id)}
                              disabled={isLoading}
                              variant="default"
                            >
                              <CheckCircle className="w-4 h-4 mr-2" />
                              Approve
                            </Button>
                            <Button
                              onClick={() => handleReject(response.id)}
                              disabled={isLoading}
                              variant="destructive"
                            >
                              <XCircle className="w-4 h-4 mr-2" />
                              Reject
                            </Button>
                          </DialogFooter>
                        </DialogContent>
                      </Dialog>

                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button variant="destructive" size="sm">
                            <XCircle className="w-4 h-4 mr-2" />
                            Quick Reject
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent>
                          <AlertDialogHeader>
                            <AlertDialogTitle>Reject AI Response</AlertDialogTitle>
                            <AlertDialogDescription>
                              Are you sure you want to reject this AI response? This will prevent it from being sent to the patient.
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel>Cancel</AlertDialogCancel>
                            <AlertDialogAction
                              onClick={() => handleReject(response.id)}
                              className="bg-red-600 hover:bg-red-700"
                            >
                              Reject
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>

        {/* Active Chats Tab */}
        <TabsContent value="chats" className="space-y-4">
          {activeChats.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <MessageSquare className="h-12 w-12 text-muted-foreground mb-4" />
                <CardTitle className="text-lg mb-2">No active chats</CardTitle>
                <CardDescription>No patients are currently chatting.</CardDescription>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-4">
              {activeChats.map((chat) => (
                <Card key={chat.id}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <User className="w-5 h-5 text-muted-foreground" />
                        <CardTitle className="text-base">
                          {chat.patientName || `Patient ${chat.patientId.slice(-4)}`}
                        </CardTitle>
                        <Badge variant={getStatusVariant(chat.status)}>
                          {chat.status.replace('_', ' ').toUpperCase()}
                        </Badge>
                      </div>
                      <span className="text-sm text-muted-foreground">
                        Last: {formatTimestamp(chat.lastActivity)}
                      </span>
                    </div>
                  </CardHeader>

                  <CardContent className="space-y-4">
                    {chat.summary && (
                      <div className="bg-muted p-3 rounded">
                        <h5 className="font-medium mb-2">AI Summary:</h5>
                        <div className="text-sm space-y-1">
                          <p><strong>Symptoms:</strong> {chat.summary.symptoms.join(', ')}</p>
                          <p><strong>Conditions:</strong> {chat.summary.conditions.join(', ')}</p>
                          <p><strong>Status:</strong>
                            <Badge variant={chat.status === 'doctor_takeover' ? 'destructive' : 'secondary'} className="ml-1">
                              {chat.status}
                            </Badge>
                          </p>
                        </div>
                      </div>
                    )}

                    <div className="space-y-2 max-h-48 overflow-y-auto">
                      {chat.messages.slice(-5).map((message) => (
                        <div key={message.id} className={`flex ${message.role === 'patient' ? 'justify-start' : 'justify-end'}`}>
                          <div className={`max-w-xs px-3 py-2 rounded-lg text-sm ${
                            message.role === 'patient'
                              ? 'bg-muted'
                              : message.role === 'ai'
                              ? 'bg-blue-100'
                              : 'bg-green-100'
                          }`}>
                            <div className="flex items-center space-x-1 mb-1">
                              {message.role === 'patient' ? (
                                <User className="w-3 h-3" />
                              ) : message.role === 'ai' ? (
                                <Bot className="w-3 h-3" />
                              ) : (
                                <Stethoscope className="w-3 h-3" />
                              )}
                              <span className="text-xs font-medium capitalize">{message.role}</span>
                              {message.status && (
                                <Badge variant="outline" className="text-xs px-1 py-0">
                                  {message.status}
                                </Badge>
                              )}
                            </div>
                            <p>{message.content}</p>
                          </div>
                        </div>
                      ))}
                    </div>

                    <div className="flex space-x-2">
                      <Dialog>
                        <DialogTrigger asChild>
                          <Button variant="outline" size="sm">
                            <MessageSquare className="w-4 h-4 mr-2" />
                            Message Patient
                          </Button>
                        </DialogTrigger>
                        <DialogContent>
                          <DialogHeader>
                            <DialogTitle>Send Direct Message</DialogTitle>
                            <DialogDescription>
                              Send a direct message to {chat.patientName || 'the patient'}
                            </DialogDescription>
                          </DialogHeader>
                          <div className="space-y-4">
                            <Textarea
                              value={directMessage}
                              onChange={(e) => setDirectMessage(e.target.value)}
                              placeholder="Type your message to the patient..."
                              rows={4}
                            />
                          </div>
                          <DialogFooter>
                            <Button
                              onClick={() => handleSendMessage(chat.id)}
                              disabled={isLoading || !directMessage.trim()}
                            >
                              <Send className="w-4 h-4 mr-2" />
                              Send Message
                            </Button>
                          </DialogFooter>
                        </DialogContent>
                      </Dialog>

                      <Button variant="secondary" size="sm">
                        <Eye className="w-4 h-4 mr-2" />
                        Take Over Chat
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}