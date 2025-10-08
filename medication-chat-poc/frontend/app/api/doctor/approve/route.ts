import { NextRequest, NextResponse } from 'next/server';

// Import the same stores from other API routes (in production, use a database)
// This would be better handled with a shared state management solution

export async function POST(request: NextRequest) {
  try {
    const { responseId, action, modifications, reason, newResponse } = await request.json();

    if (!responseId || !action) {
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400 }
      );
    }

    // Get current pending approvals and active chats
    const pendingRes = await fetch(`${request.nextUrl.origin}/api/doctor/pending`);
    const pendingData = await pendingRes.json();
    const pendingApprovals = pendingData.pendingResponses || [];

    const chatsRes = await fetch(`${request.nextUrl.origin}/api/doctor/chats`);
    const chatsData = await chatsRes.json();
    const activeChats = chatsData.chats || [];

    // Find the pending approval
    const approval = pendingApprovals.find((p: any) => p.id === responseId);
    if (!approval) {
      return NextResponse.json(
        { error: 'Approval not found' },
        { status: 404 }
      );
    }

    // Find the corresponding chat
    const chat = activeChats.find((c: any) => c.id === approval.chatId);
    if (!chat) {
      return NextResponse.json(
        { error: 'Associated chat not found' },
        { status: 404 }
      );
    }

    if (action === 'approve') {
      // Create AI response message
      const aiResponse = modifications || approval.aiResponse;
      const newMessage = {
        id: `msg-${Date.now()}`,
        role: 'ai',
        content: aiResponse,
        timestamp: new Date().toISOString(),
        status: 'approved'
      };

      // Add AI response to chat
      chat.messages.push(newMessage);
      chat.status = 'active';
      chat.lastActivity = new Date().toISOString();

      // Update the chat via API
      await fetch(`${request.nextUrl.origin}/api/doctor/chats`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(chat)
      });

      // Remove from pending approvals
      await fetch(`${request.nextUrl.origin}/api/doctor/pending?id=${responseId}`, {
        method: 'DELETE'
      });

      return NextResponse.json({
        success: true,
        action: 'approved',
        message: 'Response approved and sent to patient'
      });

    } else if (action === 'reject') {
      // Create doctor response if provided
      if (newResponse) {
        const doctorMessage = {
          id: `msg-${Date.now()}`,
          role: 'doctor',
          content: newResponse,
          timestamp: new Date().toISOString(),
          status: 'sent',
          doctorNotes: reason
        };

        // Add doctor response to chat
        chat.messages.push(doctorMessage);
        chat.status = 'doctor_takeover';
        chat.lastActivity = new Date().toISOString();
        chat.doctorNotes = reason;

        // Update the chat via API
        await fetch(`${request.nextUrl.origin}/api/doctor/chats`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(chat)
        });
      } else {
        // Just mark as rejected, doctor will provide response later
        chat.status = 'doctor_takeover';
        chat.doctorNotes = reason;
        chat.lastActivity = new Date().toISOString();

        // Update the chat via API
        await fetch(`${request.nextUrl.origin}/api/doctor/chats`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(chat)
        });
      }

      // Remove from pending approvals
      await fetch(`${request.nextUrl.origin}/api/doctor/pending?id=${responseId}`, {
        method: 'DELETE'
      });

      return NextResponse.json({
        success: true,
        action: 'rejected',
        message: 'Response rejected, doctor will handle manually'
      });
    }

    return NextResponse.json(
      { error: 'Invalid action' },
      { status: 400 }
    );

  } catch (error) {
    console.error('Error processing approval:', error);
    return NextResponse.json(
      { error: 'Failed to process approval' },
      { status: 500 }
    );
  }
}