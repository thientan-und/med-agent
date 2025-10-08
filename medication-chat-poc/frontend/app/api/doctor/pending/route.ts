import { NextRequest, NextResponse } from 'next/server';

// In-memory store for pending approvals (in production, use a database)
let pendingApprovals: any[] = [];

export async function GET(request: NextRequest) {
  try {
    // Return pending approvals waiting for doctor review
    return NextResponse.json({
      pendingResponses: pendingApprovals
    });
  } catch (error) {
    console.error('Error fetching pending approvals:', error);
    return NextResponse.json(
      { error: 'Failed to fetch pending approvals' },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const approvalData = await request.json();

    // Add to pending approvals queue
    const approval = {
      id: `approval-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString(),
      status: 'pending',
      ...approvalData
    };

    pendingApprovals.push(approval);

    return NextResponse.json({
      success: true,
      approvalId: approval.id
    });
  } catch (error) {
    console.error('Error adding pending approval:', error);
    return NextResponse.json(
      { error: 'Failed to add pending approval' },
      { status: 500 }
    );
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const url = new URL(request.url);
    const approvalId = url.searchParams.get('id');

    if (!approvalId) {
      return NextResponse.json(
        { error: 'Approval ID is required' },
        { status: 400 }
      );
    }

    // Remove from pending approvals
    const initialLength = pendingApprovals.length;
    pendingApprovals = pendingApprovals.filter(approval => approval.id !== approvalId);

    if (pendingApprovals.length === initialLength) {
      return NextResponse.json(
        { error: 'Approval not found' },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      message: 'Approval removed from pending queue'
    });
  } catch (error) {
    console.error('Error removing pending approval:', error);
    return NextResponse.json(
      { error: 'Failed to remove pending approval' },
      { status: 500 }
    );
  }
}