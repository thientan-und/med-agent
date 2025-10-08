import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const { chatId, message, doctorId } = await request.json();

    if (!chatId || !message) {
      return NextResponse.json(
        { error: 'Chat ID and message are required' },
        { status: 400 }
      );
    }

    // Get current active chats
    const chatsRes = await fetch(`${request.nextUrl.origin}/api/doctor/chats`);
    const chatsData = await chatsRes.json();
    const activeChats = chatsData.chats || [];

    // Find the target chat
    const chat = activeChats.find((c: any) => c.id === chatId);
    if (!chat) {
      return NextResponse.json(
        { error: 'Chat not found' },
        { status: 404 }
      );
    }

    // Create doctor message
    const doctorMessage = {
      id: `msg-${Date.now()}`,
      role: 'doctor',
      content: message,
      timestamp: new Date().toISOString(),
      status: 'sent',
      doctorId: doctorId || 'doctor-1'
    };

    // Add doctor message to chat
    chat.messages.push(doctorMessage);
    chat.lastActivity = new Date().toISOString();
    chat.unreadCount = (chat.unreadCount || 0) + 1;

    // Update the chat
    await fetch(`${request.nextUrl.origin}/api/doctor/chats`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(chat)
    });

    return NextResponse.json({
      success: true,
      message: 'Direct message sent to patient',
      messageId: doctorMessage.id
    });

  } catch (error) {
    console.error('Error sending direct message:', error);
    return NextResponse.json(
      { error: 'Failed to send direct message' },
      { status: 500 }
    );
  }
}