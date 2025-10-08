export interface PatientContext {
  medicalHistory: string; // โรคประจำตัว
  currentMedications: string; // ยาที่ใช้ประจำ
  drugAllergies: string; // แพ้ยา
  foodAllergies: string; // แพ้อาหาร
  height: number; // ส่วนสูง
  weight: number; // น้ำหนัก
  age: number; // อายุ
  gender: 'male' | 'female'; // เพศ
}

export interface DrugPrescription {
  diagnosisCode: string; // Diagnosis Code
  diagnosisName: string; // Diagnosis Name
  drugName: string; // Prescription of Drug
  quantity: string; // Quantity of Drug
  frequency: string; // Frequency of Drug use
  duration: string; // Duration
  recommendations: string; // Recommend of drug use
}

export interface PendingResponse {
  id: string;
  chatId: string;
  patientId: string;
  patientName: string;
  patientContext: PatientContext;
  aiResponse: string;
  originalMessage: string;
  timestamp: Date;
  riskLevel: 'low' | 'medium' | 'high';
  confidence: number;
  isUrgent: boolean;
  aiAnalysis: {
    detectedConditions: string[];
    suggestedMedications: DrugPrescription[];
    riskLevel: 'low' | 'medium' | 'high';
    ragScore: number;
  };
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'patient' | 'ai' | 'doctor';
  content: string;
  timestamp: Date;
  status?: 'pending' | 'approved' | 'rejected' | 'sent';
  aiAnalysis?: {
    confidence: number;
    detectedConditions: string[];
    suggestedMedications: string[];
    riskLevel: 'low' | 'medium' | 'high';
    ragScore: number;
  };
  doctorNotes?: string;
}

export interface PatientChat {
  id: string;
  patientId: string;
  patientName: string;
  status: 'active' | 'waiting_approval' | 'completed' | 'doctor_takeover';
  messages: ChatMessage[];
  lastActivity: Date;
  unreadCount: number;
  doctorNotes?: string;
  summary?: {
    symptoms: string[];
    conditions: string[];
    medications: string[];
  };
}

export interface DiagnosisFeedback {
  id: string;
  chatId: string;
  patientSymptoms: string;
  aiDiagnosis: {
    icdCode: string;
    englishName: string;
    thaiName: string;
    confidence: number;
    medications: Array<{
      englishName: string;
      thaiName: string;
      dosage: string;
    }>;
  };
  timestamp: Date;
  status: 'pending' | 'approved' | 'rejected';
}

export interface DoctorStats {
  totalConsultations: number;
  pendingReviews: number;
  approvalRate: number;
  averageResponseTime: number;
  modelAccuracy: number;
}