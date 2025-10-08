import { NextRequest, NextResponse } from 'next/server';

// Proxy to FastAPI backend
export async function POST(request: NextRequest) {
  try {
    const requestBody = await request.json();
    const { message, conversationHistory, sessionId, patientInfo, vitalSigns } = requestBody;

    if (!message || typeof message !== 'string') {
      return NextResponse.json(
        { error: 'Message is required' },
        { status: 400 }
      );
    }

    // Check for greetings (Thai and English)
    const greetingPatterns = [
      /^(hi|hello|hey|good morning|good afternoon|good evening)$/i,
      /^(สวัสดี|หวัดดี|ดีจ้า|ดีค่ะ|ดีครับ|อรุณสวัสดิ์|ราตรีสวัสดิ์|สายธาร)$/i,
      /^(ฮัลโหล|ฮาย|เฮ้|เฮลโล่)$/i
    ];

    const isGreeting = greetingPatterns.some(pattern => pattern.test(message.trim()));

    if (isGreeting) {
      return NextResponse.json({
        response: `สวัสดีค่ะ/ครับ! ยินดีต้อนรับสู่ระบบให้คำปรึกษาด้านสุขภาพ 🏥

ฉันสามารถช่วยตอบคำถามเกี่ยวกับ:
• อาการเจ็บป่วยทั่วไป
• คำแนะนำการดูแลสุขภาพเบื้องต้น
• ข้อมูลเกี่ยวกับยาที่ใช้ทั่วไป

กรุณาอธิบายอาการหรือปัญหาสุขภาพที่คุณต้องการปรึกษา

⚠️ หมายเหตุ: ข้อมูลนี้เป็นเพียงคำแนะนำทั่วไป ไม่ใช่การวินิจฉัยโรค หากมีอาการรุนแรงกรุณาพบแพทย์`
      });
    }

    // Check if doctor approval is required (MVP requirement) - Temporarily disabled for medication testing
    const requireDoctorApproval = false; // process.env.REQUIRE_DOCTOR_APPROVAL === 'true'; // Default to false

    if (requireDoctorApproval) {
      // First, get AI analysis from backend for doctor review
      try {
        const currentSessionId = sessionId || `session-${Date.now()}`;
        const chatId = `chat-${currentSessionId}`;

        // Call backend AI service to get real medical analysis
        const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
        const aiResponse = await fetch(`${backendUrl}/api/v1/medical/chat/`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message,
            conversation_history: conversationHistory || [],
            patient_info: patientInfo || null,
            vital_signs: vitalSigns || null,
            preferred_language: 'auto',
            session_id: currentSessionId,
            include_reasoning: true
          })
        });

        let aiData: any = {};
        let aiResponseText = 'ขออภัยค่ะ/ครับ ไม่สามารถประมวลผล AI ได้ในขณะนี้';
        let detectedConditions: string[] = ['รอการประเมินโดยแพทย์'];
        let suggestedMedications: any[] = [];
        let riskLevel: 'low' | 'medium' | 'high' = 'medium';
        let confidence = 0.5;
        let isUrgent = false;

        if (aiResponse.ok) {
          aiData = await aiResponse.json();
          aiResponseText = aiData.message || aiResponseText;

          // Extract diagnosis information
          if (aiData.diagnosis?.primary_diagnosis) {
            detectedConditions = [aiData.diagnosis.primary_diagnosis];
            if (aiData.diagnosis.differential_diagnoses?.length > 0) {
              detectedConditions = [...detectedConditions, ...aiData.diagnosis.differential_diagnoses];
            }
          }

          // Extract medication information
          if (aiData.treatment?.medications?.length > 0) {
            suggestedMedications = aiData.treatment.medications.map((med: any) => ({
              diagnosisCode: '',
              diagnosisName: '',
              drugName: med.name || med.drugName || 'ไม่ระบุ',
              quantity: med.quantity || med.dosage || '',
              frequency: med.frequency || '',
              duration: med.duration || '',
              recommendations: med.instructions || med.recommendations || ''
            }));
          }

          // Determine risk level and urgency
          confidence = aiData.diagnosis?.confidence || 0.5;
          if (aiData.triage?.priority === 'high' || aiData.triage?.urgency === 'urgent') {
            riskLevel = 'high';
            isUrgent = true;
          } else if (aiData.triage?.priority === 'low') {
            riskLevel = 'low';
          }
        } else {
          console.error('Backend AI API error:', aiResponse.status, aiResponse.statusText);
        }

        // Create comprehensive approval data with real AI analysis
        const approvalData = {
          chatId: chatId,
          patientId: currentSessionId,
          patientName: `ผู้ป่วย ${currentSessionId}`,
          patientContext: {
            medicalHistory: patientInfo?.medicalHistory || '',
            currentMedications: patientInfo?.currentMedications || '',
            drugAllergies: patientInfo?.drugAllergies || '',
            foodAllergies: patientInfo?.foodAllergies || '',
            height: patientInfo?.height || 0,
            weight: patientInfo?.weight || 0,
            age: patientInfo?.age || 0,
            gender: patientInfo?.gender || 'male'
          },
          aiResponse: aiResponseText,
          originalMessage: message,
          timestamp: new Date().toISOString(),
          riskLevel: riskLevel,
          confidence: confidence,
          isUrgent: isUrgent,
          aiAnalysis: {
            detectedConditions: detectedConditions,
            suggestedMedications: suggestedMedications,
            riskLevel: riskLevel,
            ragScore: aiData.metadata?.rag_results_count || 0
          }
        };

        // Create corresponding active chat
        const chatData = {
          id: chatId,
          patientId: currentSessionId,
          patientName: `ผู้ป่วย ${currentSessionId}`,
          status: 'waiting_approval',
          messages: [
            {
              id: `msg-${Date.now()}`,
              role: 'patient',
              content: message,
              timestamp: new Date().toISOString(),
              status: 'sent'
            }
          ],
          lastActivity: new Date().toISOString(),
          unreadCount: 1
        };

        // Add to doctor's pending queue
        const pendingUrl = `${request.nextUrl.origin}/api/doctor/pending`;
        await fetch(pendingUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(approvalData)
        });

        // Add to active chats
        const chatsUrl = `${request.nextUrl.origin}/api/doctor/chats`;
        await fetch(chatsUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(chatData)
        });
      } catch (error) {
        console.error('Error adding to pending queue:', error);
      }

      // Return waiting for approval message instead of processing immediately
      return NextResponse.json({
        response: `📋 ข้อความของคุณได้รับแล้ว

⏳ **สถานะ**: รอแพทย์พิจารณา

🩺 **ขั้นตอนต่อไป**:
• แพทย์จะตรวจสอบอาการที่คุณแจ้ง
• ให้คำแนะนำที่เหมาะสมกับอาการของคุณ
• คุณจะได้รับคำตอบภายใน 15-30 นาที

⚠️ **หากมีอาการฉุกเฉิน**: โทร 1669 ทันที

💬 **หมายเหตุ**: ระบบจะแจ้งเตือนเมื่อแพทย์ตอบกลับแล้ว`,
        type: 'waiting_approval',
        status: 'pending_approval',
        metadata: {
          processing_time_ms: 0,
          requires_approval: true,
          session_id: sessionId
        }
      });
    }

    // Proxy to FastAPI backend (only when approval not required)
    try {
      const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';

      const response = await fetch(`${backendUrl}/api/v1/medical/chat/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          conversation_history: conversationHistory || [],
          patient_info: patientInfo || null,
          vital_signs: vitalSigns || null,
          preferred_language: 'auto',
          session_id: sessionId || null,
          include_reasoning: false
        })
      });

      if (!response.ok) {
        console.error('Backend API error:', response.status, response.statusText);

        // Fallback response
        return NextResponse.json({
          response: `ขออภัยค่ะ/ครับ ระบบกำลังมีปัญหาชั่วคราว

กรุณาลองใหม่อีกครั้งในอีกสักครู่ หากมีอาการฉุกเฉินกรุณาโทร 1669

⚠️ หมายเหตุ: ข้อมูลนี้เป็นเพียงคำแนะนำทั่วไป ไม่ใช่การวินิจฉัยโรค หากมีอาการรุนแรงกรุณาพบแพทย์`
        });
      }

      const result = await response.json();

      // Return in the format expected by the frontend
      return NextResponse.json({
        response: result.message || 'ขออภัย ไม่สามารถประมวลผลได้',
        type: result.type || 'general',
        metadata: result.metadata || {},
        triage: result.triage || null,
        diagnosis: result.diagnosis || null,
        treatment: result.treatment || null
      });

    } catch (backendError) {
      console.error('Backend connection error:', backendError);

      // Fallback response when backend is unavailable
      return NextResponse.json({
        response: `ขออภัยค่ะ/ครับ ไม่สามารถเชื่อมต่อกับระบบได้ในขณะนี้

กรุณาลองใหม่อีกครั้งในอีกสักครู่ หากมีอาการฉุกเฉินกรุณาโทร 1669

⚠️ หมายเหตุ: ข้อมูลนี้เป็นเพียงคำแนะนำทั่วไป ไม่ใช่การวินิจฉัยโรค หากมีอาการรุนแรงกรุณาพบแพทย์`
      });
    }

  } catch (error) {
    console.error('Chat API error:', error);
    return NextResponse.json(
      {
        error: 'Internal server error',
        response: 'ขออภัย เกิดข้อผิดพลาดในระบบ กรุณาลองใหม่อีกครั้ง'
      },
      { status: 500 }
    );
  }
}