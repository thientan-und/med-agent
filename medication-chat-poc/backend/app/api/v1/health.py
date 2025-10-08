# Health Check API - System status and monitoring endpoints

import logging
import time
from datetime import datetime
from fastapi import APIRouter, Depends, Request
from typing import Dict, Any

from app.schemas.medical_chat import HealthCheckResponse
from app.services.medical_ai_service import MedicalAIService
from app.util.config import get_settings
from app.util.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

# Track service start time
start_time = time.time()


def get_medical_ai_service(request: Request) -> MedicalAIService:
    """Get medical AI service from app state"""
    return request.app.state.medical_ai_service


@router.get(
    "/",
    response_model=HealthCheckResponse,
    summary="🏥 Health Check",
    description="""
    **System Health Check**

    Check the overall health and status of the Medical AI system.

    ### Checks Include:
    - API server status
    - Medical AI service status
    - Ollama model connectivity
    - Rate limiter status
    - System uptime

    ### Response Codes:
    - **200**: All systems healthy
    - **503**: Some services unavailable
    """
)
async def health_check(
    ai_service: MedicalAIService = Depends(get_medical_ai_service)
) -> HealthCheckResponse:
    """
    Comprehensive health check for all system components
    """

    try:
        # Check individual services
        services_status = {}

        # 1. API Server (if we're here, it's working)
        services_status["api_server"] = "healthy"

        # 2. Medical AI Service
        try:
            if ai_service:
                # Try to get stats to verify service is working
                stats = await ai_service.get_system_statistics()
                services_status["medical_ai_service"] = "healthy"
            else:
                services_status["medical_ai_service"] = "unavailable"
        except Exception as e:
            logger.error(f"Medical AI service health check failed: {e}")
            services_status["medical_ai_service"] = "unhealthy"

        # 3. Ollama connectivity
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{settings.ollama_url}/api/tags", timeout=5) as response:
                    if response.status == 200:
                        services_status["ollama_server"] = "healthy"
                    else:
                        services_status["ollama_server"] = "unhealthy"
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            services_status["ollama_server"] = "unavailable"

        # 4. Rate limiter
        try:
            rate_limiter = get_rate_limiter()
            rate_stats = await rate_limiter.get_system_stats()
            services_status["rate_limiter"] = "healthy"
        except Exception as e:
            logger.error(f"Rate limiter health check failed: {e}")
            services_status["rate_limiter"] = "unhealthy"

        # 5. Knowledge base
        try:
            if ai_service and (ai_service.medicines or ai_service.diagnoses or ai_service.treatments):
                services_status["knowledge_base"] = "healthy"
            else:
                services_status["knowledge_base"] = "limited"
        except Exception:
            services_status["knowledge_base"] = "unavailable"

        # Calculate uptime
        uptime_seconds = time.time() - start_time

        # Determine overall status
        unhealthy_services = [svc for svc, status in services_status.items() if status in ["unhealthy", "unavailable"]]
        overall_status = "healthy" if not unhealthy_services else "degraded"

        return HealthCheckResponse(
            status=overall_status,
            version=settings.version,
            services=services_status,
            uptime_seconds=uptime_seconds
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            version=settings.version,
            services={"api_server": "healthy", "error": str(e)},
            uptime_seconds=time.time() - start_time
        )


@router.get(
    "/detailed",
    response_model=Dict[str, Any],
    summary="🔍 Detailed Health Check",
    description="""
    **Detailed System Health Report**

    Comprehensive health check with detailed metrics and diagnostics.

    ### Additional Information:
    - Memory usage statistics
    - Request processing metrics
    - AI model status
    - Rate limiting statistics
    - Knowledge base metrics
    """
)
async def detailed_health_check(
    ai_service: MedicalAIService = Depends(get_medical_ai_service)
) -> Dict[str, Any]:
    """
    Detailed health check with comprehensive system metrics
    """

    try:
        # Basic health info
        uptime_seconds = time.time() - start_time

        # System metrics
        system_info = {
            "status": "healthy",
            "version": settings.version,
            "uptime_seconds": uptime_seconds,
            "uptime_human": _format_uptime(uptime_seconds),
            "timestamp": datetime.now().isoformat()
        }

        # Medical AI Service metrics
        ai_metrics = {}
        if ai_service:
            try:
                ai_stats = await ai_service.get_system_statistics()
                ai_metrics = {
                    "total_consultations": ai_stats.get("total_consultations", 0),
                    "emergency_cases": ai_stats.get("emergency_cases_detected", 0),
                    "avg_response_time_ms": ai_stats.get("average_response_time_ms", 0),
                    "languages_supported": list(ai_stats.get("languages_detected", {}).keys()),
                    "knowledge_base": ai_stats.get("knowledge_base", {}),
                    "agents_status": ai_stats.get("agents_status", {})
                }
            except Exception as e:
                ai_metrics = {"error": str(e), "status": "unavailable"}

        # Rate limiter metrics
        rate_limiter_metrics = {}
        try:
            rate_limiter = get_rate_limiter()
            rate_stats = await rate_limiter.get_system_stats()
            rate_limiter_metrics = rate_stats
        except Exception as e:
            rate_limiter_metrics = {"error": str(e), "status": "unavailable"}

        # Ollama connectivity
        ollama_info = {}
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{settings.ollama_url}/api/tags", timeout=5) as response:
                    if response.status == 200:
                        models_data = await response.json()
                        ollama_info = {
                            "status": "connected",
                            "url": settings.ollama_url,
                            "models_available": len(models_data.get("models", [])),
                            "models": [model.get("name", "") for model in models_data.get("models", [])]
                        }
                    else:
                        ollama_info = {"status": "error", "code": response.status}
        except Exception as e:
            ollama_info = {"status": "unavailable", "error": str(e)}

        # Configuration info (safe subset)
        config_info = {
            "debug_mode": settings.debug,
            "max_requests_per_minute": settings.max_requests_per_minute,
            "max_requests_per_hour": settings.max_requests_per_hour,
            "emergency_confidence_threshold": settings.emergency_confidence_threshold,
            "diagnosis_confidence_threshold": settings.diagnosis_confidence_threshold,
            "enable_dialect_detection": settings.enable_dialect_detection
        }

        return {
            "system": system_info,
            "medical_ai": ai_metrics,
            "rate_limiter": rate_limiter_metrics,
            "ollama": ollama_info,
            "configuration": config_info
        }

    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return {
            "system": {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        }


@router.get(
    "/readiness",
    summary="⚡ Readiness Check",
    description="""
    **Service Readiness Check**

    Quick check to determine if the service is ready to handle requests.
    Useful for load balancers and orchestration systems.

    ### Response:
    - **200**: Service ready
    - **503**: Service not ready
    """
)
async def readiness_check(
    ai_service: MedicalAIService = Depends(get_medical_ai_service)
) -> Dict[str, str]:
    """
    Quick readiness check for load balancers
    """

    try:
        # Check if critical services are available
        if not ai_service:
            return {"status": "not_ready", "reason": "Medical AI service unavailable"}

        # Quick Ollama check
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{settings.ollama_url}/api/tags", timeout=2) as response:
                    if response.status != 200:
                        return {"status": "not_ready", "reason": "Ollama server unavailable"}
        except Exception:
            return {"status": "not_ready", "reason": "Cannot connect to Ollama"}

        return {"status": "ready"}

    except Exception as e:
        return {"status": "not_ready", "reason": str(e)}


@router.get(
    "/liveness",
    summary="💓 Liveness Check",
    description="""
    **Service Liveness Check**

    Simple check to verify the service is alive and responding.
    Used by container orchestrators to detect hung processes.
    """
)
async def liveness_check() -> Dict[str, str]:
    """
    Simple liveness check - if this responds, the service is alive
    """
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": str(int(time.time() - start_time))
    }


@router.get(
    "/metrics",
    summary="📊 System Metrics",
    description="""
    **System Performance Metrics**

    Get performance metrics for monitoring and alerting.

    ### Metrics Include:
    - Request counts and rates
    - Response times
    - Error rates
    - Resource utilization
    """
)
async def get_metrics(
    ai_service: MedicalAIService = Depends(get_medical_ai_service)
) -> Dict[str, Any]:
    """
    Get system performance metrics
    """

    try:
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - start_time
        }

        # AI service metrics
        if ai_service:
            try:
                ai_stats = await ai_service.get_system_statistics()
                metrics.update({
                    "total_consultations": ai_stats.get("total_consultations", 0),
                    "emergency_cases": ai_stats.get("emergency_cases_detected", 0),
                    "avg_response_time_ms": ai_stats.get("average_response_time_ms", 0)
                })
            except Exception:
                pass

        # Rate limiter metrics
        try:
            rate_limiter = get_rate_limiter()
            rate_stats = await rate_limiter.get_system_stats()
            metrics.update({
                "rate_limiter_clients": rate_stats.get("total_clients_tracked", 0),
                "rate_limited_clients": rate_stats.get("rate_limited_clients", 0),
                "requests_last_minute": rate_stats.get("total_requests_last_minute", 0),
                "requests_last_hour": rate_stats.get("total_requests_last_hour", 0)
            })
        except Exception:
            pass

        return metrics

    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Utility functions

def _format_uptime(seconds: float) -> str:
    """Format uptime in human-readable format"""

    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if days > 0:
        return f"{days}d {hours}h {minutes}m {secs}s"
    elif hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"