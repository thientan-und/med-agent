import { NextRequest, NextResponse } from 'next/server';

// In-memory store for active chats (in production, use a database)
let activeChats: any[] = [];

export async function GET(request: NextRequest) {
  try {
    // Return active doctor-patient chats
    return NextResponse.json({
      chats: activeChats
    });
  } catch (error) {
    console.error('Error fetching active chats:', error);
    return NextResponse.json(
      { error: 'Failed to fetch active chats' },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const chatData = await request.json();

    // Check if chat already exists
    const existingChatIndex = activeChats.findIndex(chat => chat.id === chatData.id);

    if (existingChatIndex >= 0) {
      // Update existing chat
      activeChats[existingChatIndex] = chatData;
    } else {
      // Add new chat
      activeChats.push(chatData);
    }

    return NextResponse.json({
      success: true,
      chatId: chatData.id
    });
  } catch (error) {
    console.error('Error adding active chat:', error);
    return NextResponse.json(
      { error: 'Failed to add active chat' },
      { status: 500 }
    );
  }
}