# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure

```
medical-chat-app/
├── frontend/          # Next.js frontend application
│   ├── app/          # Next.js 13+ App Router
│   ├── components/   # React components
│   ├── lib/          # Utility functions and clients
│   ├── public/       # Static assets
│   └── package.json  # Frontend dependencies
├── backend/          # FastAPI backend with precision architecture
│   ├── app/
│   │   ├── core/     # Precision-oriented medical AI components
│   │   │   ├── types.py           # Pydantic schemas (DiagnosisCard, etc.)
│   │   │   ├── router.py          # Evidence-first routing system
│   │   │   ├── calculators.py     # Medical calculators (HEART, PERC, Wells)
│   │   │   ├── critic.py          # Blocking precision critic
│   │   │   ├── uncertainty.py     # Uncertainty quantification & abstention
│   │   │   └── precision_service.py # Main precision orchestrator
│   │   ├── services/ # AI services and clients
│   │   │   ├── ollama_client.py   # Real Ollama client for multi-model coordination
│   │   │   └── medical_ai_service.py # Medical AI service
│   │   └── api/      # API endpoints
│   │       └── v1/
│   │           └── medical/ # Precision medical APIs
│   ├── logs/         # Application logs
│   ├── Makefile      # Backend commands
│   └── requirements.txt # Python dependencies (includes scipy for uncertainty)
├── Makefile          # Root project commands
├── AGENTIC_AI_FLOW.md # Complete architecture documentation
└── package.json      # Monorepo configuration
```

## Commands

### Quick Start (from root directory)
```bash
make dev               # Start both frontend (port 3000) and backend (port 8000)
make install           # Install all dependencies
make setup             # Complete setup with Ollama and models
make test              # Run tests
make help              # Show all available commands
```

### Development Commands
```bash
make dev               # Run both frontend and backend
make dev-frontend      # Run only Next.js frontend (port 3000)
make dev-backend       # Run only FastAPI backend (port 8000)
make build             # Build production version
make start             # Start production servers
make stop              # Stop all running services
make clean             # Clean build artifacts and caches
make status            # Check system status
```

### Frontend Commands (from frontend/ directory)
```bash
pnpm dev               # Start Next.js development server
pnpm build             # Build for production
pnpm start             # Start production server
pnpm lint              # Run ESLint checks
pnpm websocket         # Run WebSocket server
```

### Backend Commands (from backend/ directory)
```bash
make run               # Start FastAPI development server with auto-reload
make dev               # Same as 'make run' (alias)
make prod              # Start production server
make check             # Check configuration without starting
make install           # Install Python dependencies
make setup             # Setup backend environment
make status            # Show backend status and info
make clean             # Clean cache and temporary files
make logs              # View recent API logs
make test              # Run API tests with patient scenarios
make help              # Show all available commands
```

### Setup & Testing
```bash
./setup-ollama.sh         # Install and configure Ollama server
./setup-models.sh         # Download required AI models (medllama2, seallm-7b-v2)
backend/setup_backend.sh  # Setup Python backend environment
node test-llm.js          # Test all AI integrations (Ollama, MedLlama2, SeaLLM, chat API)
```

## Architecture Overview

### RAG-Enhanced Medical AI with Doctor Approval Workflow
This app implements a comprehensive elderly-focused medical AI system with hybrid RAG-LLM medication recommendations:

```
Thai Input → Context Extraction → LLM Diagnosis → RAG Enhancement
→ Hybrid Response → Doctor Approval Queue → Final Response
```

### Multi-Model AI Coordination
- **SeaLLM-7B-v2** (`nxphi47/seallm-7b-v2-q4_0:latest`) - Thai↔English translation with medical context preservation
- **MedLlama2** (`medllama2:latest`) - Medical analysis, differential diagnosis, and clinical instructions
- **RAG Knowledge Base** - 55 treatments, 19 medicines, 42 diagnoses for medication recommendations
- **Ollama Client** - Real model coordination and fallback handling
- **Model Routing** - Intelligent routing based on task type (translation vs medical analysis vs medication guidance)

### Precision Architecture Components

**Evidence-First Routing (`app/core/router.py`):**
- RouteSignals extraction from symptoms
- Clinical evidence-based tool selection (not "run everything")
- Emergency vs standard pathway routing
- Parallel safe tools, sequential critical tools

**Precision Critic (`app/core/critic.py`):**
- Blocking validation rules for safety
- Treatment must have guideline citations
- High-risk diagnoses require supporting evidence
- Calculator inputs must use captured fields only
- Meningitis diagnosis requires red flags

**Uncertainty Quantification (`app/core/uncertainty.py`):**
- Prediction sets for 90% coverage targets
- Temperature scaling for calibration
- Safety certainty ≥ 85% threshold
- Principled abstention when uncertain

**Medical Calculators (`app/core/calculators.py`):**
- HEART Score for chest pain risk
- PERC Rule for pulmonary embolism
- Wells PE Score for PE probability
- Strict input validation and confidence scoring

### Key Components

**Frontend (Next.js):**
- `/api/chat` - Main chat endpoint with multi-model translation pipeline
- `/api/summary` - Medical summary generation

**Backend (FastAPI with RAG-LLM Hybrid Architecture):**
- `/api/v1/medical/chat/` - Complete elderly-focused medical consultation workflow
- `/api/v1/medical/diagnosis/` - Context-aware diagnosis with patient demographics
- `/api/v1/medical/feedback/` - Doctor feedback system for continuous improvement
- `/api/v1/health/` - Health monitoring and system metrics

**Core Libraries:**
- `backend/app/services/medical_ai_service.py` - Complete RAG-LLM medical workflow
- `backend/app/services/ollama_client.py` - Real multi-model Ollama integration
- `backend/app/core/types.py` - Pydantic schemas for medical contracts
- `lib/ollama-client.ts` - Frontend Ollama integration
- `lib/seallm-translator.ts` - Thai↔English translation service
- `lib/medical-prompts.ts` - Medical prompts and emergency detection
- `lib/thai-dialect-helper.ts` - Regional Thai dialect normalization

**RAG Knowledge Base:**
- `backend/data/medicines.csv` - 19 medicines with dosages and safety information
- `backend/data/treatments.csv` - 55 treatments with clinical guidelines
- `backend/data/diagnoses.csv` - 42 common diagnoses for elderly patients

### Multi-Layer Safety Systems

**Doctor Approval Workflow:**
- All AI responses queued for physician review before reaching patients
- Doctor can approve, edit, or reject AI recommendations
- Complete audit trail of AI decisions and doctor modifications
- Conservative approach: when uncertain, escalate to human oversight

**RAG-LLM Safety:**
- RAG provides evidence-based medicine names and dosages from curated knowledge base
- LLM generates clinical instructions with safety considerations
- Contraindication checking based on patient demographics and medical history
- Emergency detection with automatic red flag escalation

**Traditional Safety:**
- Emergency keyword detection across Thai dialects and English
- Automatic escalation for urgent symptoms (contact 1669)
- Prominent medical disclaimers on every interaction
- Red flag detection and physician escalation for critical symptoms like 'มึนงง' (confusion)

## Environment Configuration

```bash
# Frontend (.env.local)
OLLAMA_URL=http://localhost:11434  # Local development
# OLLAMA_URL=https://[ngrok-url]   # Production via ngrok

# Backend (backend/.env)
ENVIRONMENT=development
HOST=0.0.0.0
PORT=8000
OLLAMA_URL=http://localhost:11434
SEALLM_MODEL=seallm-7b-v2
MEDLLAMA_MODEL=medllama2
```

## Thai Language Support

The app supports multiple Thai dialects:
- Northern Thai (ล้านนา)
- Isan (อีสาน)
- Southern Thai (ใต้)
- Elderly/Rural expressions

Dialect normalization happens automatically in `thai-dialect-helper.ts`.

## Development Guidelines

### Code Style
- TypeScript with React functional components
- Two-space indentation
- camelCase for functions/variables
- PascalCase for components and types
- Hooks prefixed with `use`
- Tailwind CSS for styling (avoid inline styles)

### Testing Approach
- No formal test suite yet
- Always run `pnpm lint` before commits
- Test AI integrations with `node test-llm.js`
- Manual validation for Thai translations

### Adding Features

**Frontend:**
- Co-locate domain logic in `frontend/src/lib`
- Keep UI state localized in components
- Add new API routes under `frontend/src/app/api`
- Update medical prompts in `frontend/lib/medical-prompts.ts`

**Backend (RAG-LLM Architecture):**
- Add new patient context extraction patterns in `medical_ai_service.py`
- Extend RAG knowledge base with new medicines/treatments in `backend/data/`
- Add new agent capabilities to DiagnosticAgent, TreatmentAgent, TriageAgent
- Implement new LLM prompts for clinical instruction generation
- Update doctor approval workflow in `_queue_ai_response_for_doctor_approval`
- Always follow machine-checkable contracts (Pydantic validation)
- Test RAG-LLM integration with evaluation scripts

## Critical Safety Notes

### RAG-LLM Safety
1. **Doctor Approval Required**: All AI responses must be approved by qualified physicians before reaching patients
2. **Evidence-Based RAG**: Medicine recommendations sourced from curated knowledge base with 55 treatments and 19 medicines
3. **LLM Clinical Reasoning**: Duration and instructions generated by MedLlama2 with safety considerations
4. **Patient Context Integration**: Age, gender, medical history automatically extracted and considered
5. **Conservative Escalation**: Emergency symptoms like 'มึนงง' trigger immediate physician escalation

### Traditional Safety
6. **Emergency Detection**: The system automatically detects emergency keywords in multiple Thai dialects
7. **Medical Ethics**: Never provide specific diagnoses or medication prescriptions
8. **Translation Fallbacks**: If SeaLLM fails, responses fall back to English with warnings
9. **Model Requirements**: Both `medllama2` (~3.8GB) and `seallm-7b-v2` (~4GB) must be running via Ollama

### Development Safety
10. **Test RAG-LLM Integration**: Use evaluation scripts to test complete workflow end-to-end
11. **Validate Medication Recommendations**: Ensure RAG retrieval and LLM enhancement work correctly
12. **Check Doctor Approval Flow**: Verify all AI responses are properly queued for physician review
13. **Monitor Context Extraction**: Ensure patient demographics are correctly parsed from Thai messages
14. **Test Emergency Detection**: Verify critical symptoms trigger appropriate escalation