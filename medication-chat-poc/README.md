# 🏥 RAG-Enhanced Medical AI with Doctor Approval (ระบบ AI ทางการแพทย์ที่ปรับปรุงด้วย RAG และการอนุมัติจากแพทย์)

แอปพลิเคชันทางการแพทย์สำหรับผู้สูงอายุที่ใช้สถาปัตยกรรม RAG-LLM Hybrid ขั้นสูง พร้อมระบบ Doctor Approval Workflow, Context-Aware Diagnosis และ Multi-Model AI Coordination รองรับภาษาไทยและภาษาถิ่นไทย

## 🌟 คุณสมบัติหลัก (Key Features)

### 👵 RAG-Enhanced Medical AI for Elderly Users
- **📚 Knowledge-Based Recommendations**: ใช้ฐานข้อมูล 55 การรักษา, 19 ยา และ 42 การวินิจฉัยที่คัดสรรแล้ว
- **🤖 Hybrid RAG-LLM System**: RAG ให้ชื่อยาและปริมาณ, LLM ให้ระยะเวลาและคำแนะนำทางคลินิก
- **👨‍⚕️ Doctor Approval Required**: แพทย์ตรวจสอบและอนุมัติคำแนะนำของ AI ก่อนส่งให้ผู้ป่วย
- **📋 Auto Context Extraction**: ดึงข้อมูลผู้ป่วย (อายุ, เพศ, ประวัติ) จากข้อความภาษาไทยอัตโนมัติ
- **🔄 Complete Workflow**: ผู้ป่วย → การวินิจฉัย LLM → การปรับปรุงด้วย RAG → การอนุมัติของแพทย์

### 🤖 Multi-Model AI Architecture
- **🗣️ รองรับภาษาถิ่นไทย**: เข้าใจภาษาถิ่นเหนือ อีสาน ใต้ และภาคกลาง
- **🔄 Translation Pipeline 4 ขั้นตอน**: Thai → SeaLLM → MedLlama2 → SeaLLM → Thai
- **🏥 Specialized Medical AI Models**:
  - **MedLlama2** (`medllama2:latest`) สำหรับวิเคราะห์ทางการแพทย์
  - **SeaLLM-7B-v2** (`nxphi47/seallm-7b-v2-q4_0:latest`) สำหรับแปลภาษาไทย-อังกฤษ
- **🎛️ Model Routing Logic**: เลือกโมเดลตามประเภทงาน (การแปล vs การวิเคราะห์ทางการแพทย์)

### 📋 Structured Medical Output
- **DiagnosisCard Schema**: โครงสร้างผลลัพธ์แบบมาตรฐาน
- **Medical Calculators**: HEART Score, PERC Rule, Wells PE Score พร้อมการตรวจสอบ Input
- **Treatment Guidelines**: คำแนะนำการรักษาต้องมี Guideline Citations
- **Uncertainty Metrics**: แสดงระดับความแน่นอนและเหตุผลในการ Abstain

### 🛡️ Advanced Safety Systems
- **⚕️ Medical Diagnostic Principles**: ใช้หลัก OPQRST ในการซักประวัติ
- **🚨 Emergency Detection**: ตรวจจับคำหลักฉุกเฉินหลายภาษาถิ่นและ Escalate ทันที
- **🔍 Red Flag Detection**: ตรวจจับอาการเตือนภัยและส่งต่อแพทย์
- **🎯 Conservative Abstention**: หลีกเลี่ยงการวินิจฉัยเมื่อข้อมูลไม่เพียงพอ

## 🗣️ ภาษาถิ่นที่รองรับ (Supported Thai Dialects)

### ภาคเหนือ (Northern)
- จุก, จุกแล้ว → ปวด, ปวดมาก
- แหง, แหงแล้ว → ปวด, ปวดมาก

### ภาคอีสาน (Isan)
- แล้ง, แล้งโพด → ปวด, ปวดมาก
- บักแล้วโพด → มากจริงๆ
- หื่อ → อะไร

### ภาคใต้ (Southern)
- ปวดหัง, เจ็บหัง → ปวดหัว, เจ็บหัว
- โพดหัง → มาก
- แดก → กิน

## 🔄 RAG-LLM Hybrid Medical Pipeline

```
1. Thai Input + Auto Context Extraction
   ↓
2. Patient Demographics → Extract age, gender, medical history from Thai text
   ↓
3. SeaLLM-7B-v2 Translation → English symptoms
   ↓
4. Emergency Detection → Check for critical symptoms ('มึนงง', etc.)
   ↓
5. LLM Medical Analysis (MedLlama2) → Generate primary diagnosis
   ↓
6. RAG Knowledge Retrieval → Find medicines and dosages from 55 treatments/19 medicines
   ↓
7. LLM Enhancement → Generate duration, frequency, clinical instructions
   ↓
8. Hybrid Response Creation → Combine RAG medicines + LLM clinical reasoning
   ↓
9. Doctor Approval Queue → Complete AI response awaits physician review
   ↓
10. Doctor Actions → Approve, Edit, or Reject AI recommendations
   ↓
11. Final Response to Patient:
    • Complete medication guidance (RAG + LLM)
    • Doctor-approved recommendations
    • Emergency escalation when needed
    • Clear status updates
```

## 🛡️ Multi-Layer Safety Architecture

### RAG-LLM Safety Systems
- **👨‍⚕️ Doctor Approval Required**:
  - All AI responses must be approved by qualified physicians
  - Complete audit trail of AI decisions and doctor modifications
  - Three-step workflow: Approve, Edit, or Reject AI recommendations
- **📚 Evidence-Based RAG**:
  - Medicine recommendations from curated knowledge base (55 treatments, 19 medicines)
  - Contraindication checking based on patient demographics
  - Safety guidelines integrated into medication retrieval
- **🤖 LLM Clinical Reasoning**:
  - Duration and instructions generated by MedLlama2 with safety considerations
  - Patient context integration (age, gender, medical history)
  - Conservative escalation for emergency symptoms

### Traditional Safety Features
- **การตรวจจับคำสนทนาทั่วไป**: แยก greeting และคำถามทั่วไปออกจากการวิเคราะห์ทางการแพทย์
- **การขอข้อมูลเพิ่มเติม**: หากอาการไม่ชัดเจน (เช่น "ไม่สบาย") จะขอข้อมูลเฉพาะเจาะจง
- **หลัก OPQRST**: ใช้หลักการซักประวัติทางการแพทย์มาตรฐาน
- **การตรวจจับเหตุฉุกเฉิน**: รองรับคำฉุกเฉินหลายภาษาถิ่น
- **ไม่มีขนาดยาเฉพาะเจาะจง**: ระบุเฉพาะประเภทยาทั่วไป
- **ข้อความปฏิเสธความรับผิดชอบ**: แสดงชัดเจนในทุกคำตอบ

## Setup

### 1. **Install Dependencies**:
```bash
cd medical-chat-app
pnpm install
```

### 2. **Install and Setup Ollama**:
```bash
# Install Ollama
./setup-ollama.sh

# Download required models (~8GB total)
./setup-models.sh
```

### 3. **Environment Setup**:
```bash
# Create .env.local
echo "OLLAMA_URL=http://localhost:11434" > .env.local
```

### 4. **Test AI Models**:
```bash
# Test all integrations
node test-llm.js
```

### 5. **Run Development Server**:
```bash
pnpm dev
```

### 6. **Open Application**:
Open [http://localhost:3000](http://localhost:3000) in your browser

## Project Structure

```
medical-chat-app/
├── frontend/                      # Next.js frontend application
│   ├── app/
│   │   ├── api/
│   │   │   ├── chat/route.ts      # Chat API with dialect normalization
│   │   │   └── summary/route.ts   # Summary generation API
│   │   └── page.tsx               # Main application page
│   ├── components/
│   │   └── ChatInterface.tsx      # Main chat component
│   ├── lib/
│   │   ├── ollama-client.ts       # Ollama/MedLlama2 integration
│   │   ├── seallm-translator.ts   # Thai-English translation service
│   │   ├── thai-dialect-helper.ts # Thai dialect normalization
│   │   └── medical-prompts.ts     # Medical prompts and safety
│   └── types/
│       └── chat.ts                # TypeScript types
└── backend/                       # FastAPI backend with precision architecture
    ├── app/
    │   ├── core/                  # Precision-oriented components
    │   │   ├── types.py           # Pydantic schemas (DiagnosisCard, etc.)
    │   │   ├── router.py          # Evidence-first routing system
    │   │   ├── calculators.py     # Medical calculators (HEART, PERC, Wells)
    │   │   ├── critic.py          # Blocking precision critic
    │   │   ├── uncertainty.py     # Uncertainty quantification
    │   │   └── precision_service.py # Main precision orchestrator
    │   ├── services/
    │   │   ├── ollama_client.py   # Real Ollama client for multi-model
    │   │   └── medical_ai_service.py # Medical AI service
    │   └── api/
    │       └── v1/
    │           └── medical/       # Precision medical APIs
    └── requirements.txt           # Python dependencies
```

## AI Models

### Local Models via Ollama

1. **MedLlama2** (3.8GB)
   - Medical-specialized LLM
   - Provides structured medical assessments
   - Follows diagnostic principles

2. **SeaLLM-7B-v2-Q4** (4.2GB)
   - Southeast Asian language model
   - Thai-English bidirectional translation
   - Preserves medical terminology accuracy

## Medical Diagnostic Principles

### OPQRST Method for Pain Assessment
- **O**nset: When did it start?
- **P**rovocation/Palliation: What makes it better/worse?
- **Q**uality: What does it feel like?
- **R**egion/Radiation: Where is it located?
- **S**everity: Scale 1-10
- **T**ime: How long has it been?

### Information Gathering
- Vague symptoms ("ไม่สบาย", "feeling unwell") trigger specific questions
- Follows proper medical history-taking principles
- Never diagnoses based on insufficient information

## ข้อความปฏิเสธความรับผิดชอบที่สำคัญ (Important Disclaimers)

⚠️ **ข้อจำกัดความรับผิดชอบทางการแพทย์**: ผู้ช่วย AI นี้ให้ข้อมูลสุขภาพทั่วไปเท่านั้น กรุณาปรึกษาผู้เชี่ยวชาญด้านสุขภาพที่มีคุณสมบัติเหมาะสมสำหรับคำแนะนำ การวินิจฉัย หรือการรักษาทางการแพทย์เสมอ

⚠️ **เหตุฉุกเฉิน**: หากคุณประสบเหตุฉุกเฉินทางการแพทย์ โทรหาบริการฉุกเฉิน (1669) ทันที

⚠️ **ไม่ใช่แพทย์**: แอปพลิเคชันนี้ไม่ใช่สิ่งทดแทนคำแนะนำ การวินิจฉัย หรือการรักษาทางการแพทย์จากผู้เชี่ยวชาญ

⚠️ **เฉพาะการศึกษา**: โครงการนี้เป็นเพียงการพิสูจน์แนวความคิด (POC) เพื่อการศึกษาและการสาธิตเท่านั้น

## Development Commands

```bash
# Development
pnpm dev          # Start dev server (port 3000)
pnpm build        # Production build
pnpm start        # Start production server
pnpm lint         # Run ESLint checks

# Testing
node test-llm.js  # Test AI model integrations

# Model Management
ollama list       # List loaded models
ollama serve      # Start Ollama server
```

## Technology Stack

- **Frontend**: Next.js 15, TypeScript, Tailwind CSS
- **AI Runtime**: Ollama (local model hosting)
- **Medical Model**: MedLlama2 (quantized GGUF)
- **Translation**: SeaLLM-7B-v2 (Thai-English)
- **Icons**: Lucide React

## License

This project is for educational and demonstration purposes only.
