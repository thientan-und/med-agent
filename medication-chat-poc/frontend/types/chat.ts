export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'patient' | 'ai' | 'doctor';
  content: string;
  timestamp: Date;
  status?: 'pending' | 'approved' | 'rejected' | 'sent';
  aiAnalysis?: {
    riskLevel?: string;
    recommendations?: string[];
    urgency?: string;
  };
}

export interface DiagnosisSummary {
  symptoms: string[];
  possibleConditions: string[];
  recommendations: string[];
  medications: string[];
  followUpNeeded: boolean;
  urgencyLevel: 'low' | 'medium' | 'high';
}

export interface ChatSession {
  id: string;
  messages: Message[];
  diagnosisSummary?: DiagnosisSummary;
  createdAt: Date;
  updatedAt: Date;
}