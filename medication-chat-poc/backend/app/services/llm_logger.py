"""
LLM Response Logger
==================
Comprehensive logging system for tracking LLM model interactions, responses, and performance
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import uuid

@dataclass
class LLMRequest:
    """Structure for LLM request logging"""
    request_id: str
    timestamp: str
    model_name: str
    prompt: str
    parameters: Dict[str, Any]
    context: Dict[str, Any]

@dataclass
class LLMResponse:
    """Structure for LLM response logging"""
    request_id: str
    response_timestamp: str
    response_text: str
    tokens_used: Optional[int]
    response_time_ms: float
    confidence_score: Optional[float]
    error: Optional[str]

@dataclass
class LLMInteraction:
    """Complete LLM interaction log"""
    request: LLMRequest
    response: LLMResponse
    success: bool
    metadata: Dict[str, Any]

class LLMLogger:
    """Specialized logger for LLM interactions"""

    def __init__(self, log_dir: str = "backend/logs"):
        self.log_dir = log_dir
        self.ensure_log_directory()

        # Setup structured logging
        self.logger = logging.getLogger("llm_interactions")
        self.logger.setLevel(logging.INFO)

        # Create file handler for LLM logs
        log_file = os.path.join(log_dir, "llm_interactions.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create JSON formatter for structured logging
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(file_handler)

        # Separate detailed JSON logs
        self.json_log_file = os.path.join(log_dir, "llm_detailed.jsonl")

        self.logger.info("LLM Logger initialized")

    def ensure_log_directory(self):
        """Ensure log directory exists"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def log_request(self,
                   model_name: str,
                   prompt: str,
                   parameters: Dict[str, Any] = None,
                   context: Dict[str, Any] = None) -> str:
        """Log LLM request and return request ID"""

        request_id = str(uuid.uuid4())

        request = LLMRequest(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            prompt=prompt[:1000] + "..." if len(prompt) > 1000 else prompt,  # Truncate long prompts
            parameters=parameters or {},
            context=context or {}
        )

        self.logger.info(f"LLM Request - ID: {request_id}, Model: {model_name}, Prompt Length: {len(prompt)}")

        # Log detailed request to JSON
        self._log_to_json({
            "type": "request",
            "data": asdict(request)
        })

        return request_id

    def log_response(self,
                    request_id: str,
                    response_text: str,
                    response_time_ms: float,
                    tokens_used: Optional[int] = None,
                    confidence_score: Optional[float] = None,
                    error: Optional[str] = None) -> None:
        """Log LLM response"""

        response = LLMResponse(
            request_id=request_id,
            response_timestamp=datetime.now().isoformat(),
            response_text=response_text[:2000] + "..." if len(response_text) > 2000 else response_text,
            tokens_used=tokens_used,
            response_time_ms=response_time_ms,
            confidence_score=confidence_score,
            error=error
        )

        success = error is None

        self.logger.info(
            f"LLM Response - ID: {request_id}, Success: {success}, "
            f"Time: {response_time_ms:.0f}ms, Tokens: {tokens_used or 'N/A'}, "
            f"Response Length: {len(response_text)}"
        )

        if error:
            self.logger.error(f"LLM Error - ID: {request_id}, Error: {error}")

        # Log detailed response to JSON
        self._log_to_json({
            "type": "response",
            "data": asdict(response),
            "success": success
        })

    def log_interaction(self,
                       model_name: str,
                       prompt: str,
                       response_text: str,
                       response_time_ms: float,
                       parameters: Dict[str, Any] = None,
                       context: Dict[str, Any] = None,
                       tokens_used: Optional[int] = None,
                       confidence_score: Optional[float] = None,
                       error: Optional[str] = None,
                       metadata: Dict[str, Any] = None) -> str:
        """Log complete LLM interaction in one call"""

        request_id = str(uuid.uuid4())

        request = LLMRequest(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            prompt=prompt[:1000] + "..." if len(prompt) > 1000 else prompt,
            parameters=parameters or {},
            context=context or {}
        )

        response = LLMResponse(
            request_id=request_id,
            response_timestamp=datetime.now().isoformat(),
            response_text=response_text[:2000] + "..." if len(response_text) > 2000 else response_text,
            tokens_used=tokens_used,
            response_time_ms=response_time_ms,
            confidence_score=confidence_score,
            error=error
        )

        success = error is None

        interaction = LLMInteraction(
            request=request,
            response=response,
            success=success,
            metadata=metadata or {}
        )

        # Log summary
        self.logger.info(
            f"LLM Interaction - ID: {request_id}, Model: {model_name}, "
            f"Success: {success}, Time: {response_time_ms:.0f}ms, "
            f"Prompt: {len(prompt)} chars, Response: {len(response_text)} chars"
        )

        if error:
            self.logger.error(f"LLM Interaction Error - ID: {request_id}, Error: {error}")

        # Log complete interaction to JSON
        self._log_to_json({
            "type": "complete_interaction",
            "data": asdict(interaction)
        })

        return request_id

    def _log_to_json(self, data: Dict[str, Any]) -> None:
        """Log structured data to JSON lines file"""
        try:
            with open(self.json_log_file, 'a', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            self.logger.error(f"Failed to write JSON log: {e}")

    def log_medical_diagnosis(self,
                            patient_symptoms: str,
                            model_response: str,
                            diagnosis_confidence: float,
                            processing_time_ms: float,
                            model_name: str = "medllama2",
                            additional_context: Dict[str, Any] = None) -> str:
        """Specialized logging for medical diagnosis interactions"""

        context = {
            "interaction_type": "medical_diagnosis",
            "symptoms_length": len(patient_symptoms),
            "diagnosis_confidence": diagnosis_confidence,
            **(additional_context or {})
        }

        return self.log_interaction(
            model_name=model_name,
            prompt=patient_symptoms,
            response_text=model_response,
            response_time_ms=processing_time_ms,
            confidence_score=diagnosis_confidence,
            context=context,
            metadata={
                "domain": "medical",
                "task": "diagnosis",
                "language": "thai" if any('\u0e00' <= c <= '\u0e7f' for c in patient_symptoms) else "english"
            }
        )

    def log_translation(self,
                       source_text: str,
                       translated_text: str,
                       source_lang: str,
                       target_lang: str,
                       model_name: str,
                       processing_time_ms: float,
                       translation_confidence: Optional[float] = None) -> str:
        """Specialized logging for translation interactions"""

        context = {
            "interaction_type": "translation",
            "source_language": source_lang,
            "target_language": target_lang,
            "source_length": len(source_text),
            "translated_length": len(translated_text)
        }

        return self.log_interaction(
            model_name=model_name,
            prompt=f"[{source_lang}→{target_lang}] {source_text}",
            response_text=translated_text,
            response_time_ms=processing_time_ms,
            confidence_score=translation_confidence,
            context=context,
            metadata={
                "domain": "translation",
                "task": f"{source_lang}_to_{target_lang}"
            }
        )

    def get_recent_interactions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent LLM interactions from JSON log"""
        try:
            interactions = []
            if os.path.exists(self.json_log_file):
                with open(self.json_log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines[-limit:]:
                        try:
                            interactions.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
            return interactions
        except Exception as e:
            self.logger.error(f"Failed to read recent interactions: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get LLM interaction statistics"""
        try:
            interactions = self.get_recent_interactions(limit=1000)  # Last 1000

            total_interactions = len(interactions)
            successful_interactions = sum(1 for i in interactions if i.get('success', True))

            # Calculate average response times
            response_times = []
            for interaction in interactions:
                if interaction.get('type') == 'complete_interaction':
                    response_time = interaction.get('data', {}).get('response', {}).get('response_time_ms')
                    if response_time:
                        response_times.append(response_time)

            avg_response_time = sum(response_times) / len(response_times) if response_times else 0

            # Model usage statistics
            model_usage = {}
            for interaction in interactions:
                if interaction.get('type') == 'complete_interaction':
                    model = interaction.get('data', {}).get('request', {}).get('model_name')
                    if model:
                        model_usage[model] = model_usage.get(model, 0) + 1

            return {
                "total_interactions": total_interactions,
                "successful_interactions": successful_interactions,
                "success_rate": successful_interactions / total_interactions if total_interactions > 0 else 0,
                "average_response_time_ms": avg_response_time,
                "model_usage": model_usage,
                "recent_interactions_analyzed": len(interactions)
            }

        except Exception as e:
            self.logger.error(f"Failed to calculate statistics: {e}")
            return {"error": str(e)}

# Global LLM logger instance
llm_logger = LLMLogger()