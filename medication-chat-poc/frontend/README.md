# 🖥️ Frontend - Medical AI Chat Interface

Next.js 13+ frontend application for the RAG-Enhanced Medical AI system with Thai language support and real-time chat interface.

## 🌟 Features

### 💬 Chat Interface
- **Real-time messaging** with WebSocket support
- **Thai language input** with dialect normalization
- **Patient context forms** for demographic information
- **Doctor approval status** display and notifications
- **Emergency escalation** UI with prominent warnings

### 🔄 Multi-Model Integration
- **Ollama client integration** for local LLM communication
- **SeaLLM translation** for Thai↔English processing
- **Medical prompts** optimized for elderly user interactions
- **Response streaming** for improved user experience

### 🎨 UI/UX Design
- **Tailwind CSS** for responsive design
- **Radix UI components** for accessibility
- **Thai font support** with proper character rendering
- **Mobile-first design** for elderly users
- **High contrast mode** for better visibility

## 🏗️ Architecture

### App Router Structure
```
frontend/
├── app/                    # Next.js 13+ App Router
│   ├── api/               # API routes
│   │   ├── chat/          # Main chat endpoint
│   │   └── summary/       # Medical summary generation
│   ├── globals.css        # Global styles
│   ├── layout.tsx         # Root layout
│   └── page.tsx           # Home page
├── components/            # React components
│   ├── ui/               # Reusable UI components
│   ├── chat/             # Chat-specific components
│   └── medical/          # Medical interface components
├── lib/                   # Utility functions
│   ├── ollama-client.ts   # Ollama integration
│   ├── seallm-translator.ts # Thai translation service
│   ├── medical-prompts.ts # Medical AI prompts
│   └── thai-dialect-helper.ts # Thai dialect normalization
└── public/               # Static assets
```

### Key Components

#### Chat System
- `ChatInterface.tsx` - Main chat component with message handling
- `MessageBubble.tsx` - Individual message rendering with Thai support
- `PatientForm.tsx` - Demographic information collection
- `DoctorApprovalStatus.tsx` - Real-time approval status display

#### Medical Features
- `EmergencyAlert.tsx` - Critical symptom warnings
- `MedicationDisplay.tsx` - RAG-LLM medication recommendations
- `DiagnosisCard.tsx` - Structured diagnosis presentation
- `ContextExtractor.tsx` - Patient info extraction interface

#### Translation & Language
- `ThaiDialectNormalizer.ts` - Convert regional dialects to standard Thai
- `LanguageDetector.ts` - Automatic language detection
- `TranslationPipeline.ts` - SeaLLM translation workflow

## 🚀 Development

### Prerequisites
- Node.js 18+ and pnpm
- Backend API running on port 8000
- Ollama server with required models

### Setup
```bash
cd frontend
pnpm install
pnpm dev
```

### Environment Variables
```bash
# .env.local
OLLAMA_URL=http://localhost:11434
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

### Available Scripts
```bash
pnpm dev          # Start development server (port 3000)
pnpm build        # Build for production
pnpm start        # Start production server
pnpm lint         # Run ESLint checks
pnpm websocket    # Run WebSocket server for real-time features
```

## 🔧 API Integration

### Chat API
```typescript
// Main chat endpoint
POST /api/chat
{
  message: string,
  patient_info?: PatientInfo,
  session_id?: string
}

// Response
{
  type: "pending_doctor_approval" | "medical_consultation" | "emergency",
  message: string,
  ai_preview?: {
    diagnosis: string,
    medication_count: number,
    urgency: string
  }
}
```

### WebSocket Events
```typescript
// Real-time doctor approval updates
ws.on('doctor_approval', (data) => {
  status: 'approved' | 'edited' | 'rejected',
  final_response: string,
  doctor_notes?: string
})

// Emergency escalation
ws.on('emergency_alert', (data) => {
  urgency: 'critical',
  message: string,
  contact_info: string
})
```

## 🎯 Thai Language Support

### Dialect Handling
```typescript
// Regional dialect normalization
const dialectMap = {
  // Northern Thai
  'จุก': 'ปวด',
  'แหง': 'ปวด',

  // Isan
  'แล้ง': 'ปวด',
  'บักแล้วโพด': 'มากจริงๆ',

  // Southern Thai
  'ปวดหัง': 'ปวดหัว',
  'โพดหัง': 'มาก'
};
```

### Typography
- **Thai font stack**: Noto Sans Thai, Sarabun, sans-serif
- **Line height**: Optimized for Thai characters (1.6)
- **Character spacing**: Proper rendering for Thai diacritics
- **Text input**: Thai keyboard support and IME handling

## 🛡️ Safety Features

### Emergency Detection
- **Prominent red alerts** for critical symptoms
- **1669 emergency contact** integration
- **Automatic escalation** UI workflow
- **Clear call-to-action** buttons

### Doctor Approval Flow
- **Status indicators** for approval process
- **Real-time updates** via WebSocket
- **Fallback polling** if WebSocket fails
- **User-friendly waiting** screens

### Accessibility
- **ARIA labels** in Thai and English
- **Keyboard navigation** support
- **Screen reader** compatibility
- **High contrast** mode option

## 📱 Mobile Optimization

### Responsive Design
- **Mobile-first** approach for elderly users
- **Large touch targets** (minimum 44px)
- **Simple navigation** with clear hierarchy
- **Readable font sizes** (minimum 16px)

### Performance
- **Code splitting** for faster initial load
- **Image optimization** with Next.js Image
- **Lazy loading** for non-critical components
- **Service worker** for offline capability

## 🔄 Development Workflow

### Code Style
- **TypeScript** for type safety
- **ESLint + Prettier** for consistent formatting
- **Two-space indentation**
- **camelCase** for variables and functions
- **PascalCase** for components

### Testing
```bash
pnpm lint                 # Check code style
node test-llm.js         # Test LLM integrations
pnpm build               # Verify production build
```

### Deployment
```bash
pnpm build               # Production build
pnpm start               # Start production server
# Or deploy to Vercel/Netlify
```

---

**The frontend provides a user-friendly interface for elderly patients to interact with the RAG-enhanced medical AI system, with comprehensive Thai language support and real-time doctor approval workflow.**