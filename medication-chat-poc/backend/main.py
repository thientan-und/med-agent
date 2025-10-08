# Medical Chat AI Backend - FastAPI Application
# Agentic AI Medical System with Thai Language Support

import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.api.v1 import medical_chat, medical_diagnosis, medical_feedback, health, websocket
from app.api.v1.llm_logs import router as llm_logs_router
from app.services.medical_ai_service import MedicalAIService
from app.util.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global services
medical_ai_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global medical_ai_service

    logger.info("🚀 Starting Medical AI Backend...")

    # Initialize services
    try:
        medical_ai_service = MedicalAIService()
        await medical_ai_service.initialize()

        # Store in app state for access in routes
        app.state.medical_ai_service = medical_ai_service

        logger.info("✅ Medical AI services initialized successfully")

    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise

    yield

    # Cleanup
    logger.info("🛑 Shutting down Medical AI Backend...")
    if medical_ai_service:
        await medical_ai_service.cleanup()

# Create FastAPI app
settings = get_settings()

app = FastAPI(
    title="Medical Chat AI API",
    description="""
    🏥 **Agentic AI Medical Consultation System**

    An advanced medical AI system with multi-agent architecture for Thai medical consultations.

    ## Features

    * 🤖 **Agentic AI Architecture**: Autonomous medical reasoning with specialized agents
    * 🗣️ **Thai Language Support**: Comprehensive Thai dialect recognition and processing
    * 🩺 **Medical RAG System**: Knowledge-enhanced diagnosis with vector search
    * 🚨 **Emergency Detection**: Multilingual emergency keyword detection
    * 📊 **Risk Assessment**: Clinical decision support with risk scoring
    * 💊 **Treatment Planning**: Medication recommendations with safety checks
    * 🔄 **Real-time Sync**: WebSocket support for live consultations
    * 📝 **Doctor Feedback**: Learning from medical professional feedback

    ## Safety Features

    * Emergency escalation protocols
    * Medical disclaimer enforcement
    * Drug interaction checking
    * Regional dialect emergency detection
    * Professional oversight integration

    ## Usage

    This API provides endpoints for medical consultations, diagnosis assistance,
    treatment recommendations, and doctor workflow management.

    **⚠️ Medical Disclaimer**: This system provides general health information only.
    Always consult qualified healthcare professionals for medical advice.
    """,
    version="2.0.0",
    contact={
        "name": "Medical AI Team",
        "email": "medical-ai@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Custom exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": "2025-09-28T00:00:00Z"
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": 500,
            "timestamp": "2025-09-28T00:00:00Z"
        }
    )

# Include routers
app.include_router(
    health.router,
    prefix="/api/v1/health",
    tags=["Health Check"]
)

app.include_router(
    medical_chat.router,
    prefix="/api/v1/medical/chat",
    tags=["Medical Chat"]
)

app.include_router(
    medical_diagnosis.router,
    prefix="/api/v1/medical/diagnosis",
    tags=["Medical Diagnosis"]
)

app.include_router(
    medical_feedback.router,
    prefix="/api/v1/medical/feedback",
    tags=["Medical Feedback"]
)

app.include_router(
    llm_logs_router,
    prefix="/api/v1",
    tags=["LLM Logs"]
)

app.include_router(
    websocket.router,
    prefix="/api/v1/ws",
    tags=["WebSocket"]
)

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    🏥 Welcome to Medical Chat AI API

    A sophisticated agentic AI system for medical consultations with Thai language support.
    """
    return {
        "message": "🏥 Medical Chat AI API",
        "version": "2.0.0",
        "description": "Agentic AI Medical Consultation System",
        "features": [
            "Multi-agent medical reasoning",
            "Thai language & dialect support",
            "Emergency detection",
            "RAG-enhanced diagnosis",
            "Treatment planning",
            "Doctor feedback integration"
        ],
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/api/v1/health",
            "chat": "/api/v1/medical/chat",
            "diagnosis": "/api/v1/medical/diagnosis",
            "feedback": "/api/v1/medical/feedback"
        },
        "disclaimer": "⚠️ For informational purposes only. Consult healthcare professionals for medical advice."
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug"
    )