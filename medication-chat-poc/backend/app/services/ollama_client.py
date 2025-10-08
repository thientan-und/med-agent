"""
Real Ollama Client for Medical AI
================================
Connects to actual Ollama server for LLM model interactions
"""

import aiohttp
import asyncio
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class OllamaClient:
    """Real Ollama API client for LLM interactions"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.session = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def initialize(self):
        """Initialize the client session"""
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300)
            )

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None

    async def check_connection(self) -> bool:
        """Check if Ollama server is accessible"""
        try:
            if not self.session:
                await self.initialize()

            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ Ollama server connected. Available models: {len(data.get('models', []))}")
                    return True
                else:
                    logger.error(f"❌ Ollama server returned status {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ Failed to connect to Ollama: {e}")
            return False

    async def list_models(self) -> list:
        """Get list of available models"""
        try:
            if not self.session:
                await self.initialize()

            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model['name'] for model in data.get('models', [])]
                    logger.info(f"Available models: {models}")
                    return models
                else:
                    logger.error(f"Failed to get models: HTTP {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    async def generate(self,
                      model: str,
                      prompt: str,
                      system_prompt: Optional[str] = None,
                      temperature: float = 0.7,
                      max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate response from Ollama model"""

        start_time = datetime.now()

        try:
            if not self.session:
                await self.initialize()

            # Check if model is available
            available_models = await self.list_models()
            if not available_models:
                raise Exception("No models available in Ollama")

            # Use first available model if requested model not found
            if model not in available_models:
                logger.warning(f"Model '{model}' not found. Available: {available_models}")
                model = available_models[0]
                logger.info(f"Using fallback model: {model}")

            # Prepare request payload
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }

            if system_prompt:
                payload["system"] = system_prompt

            logger.info(f"🤖 Calling Ollama model '{model}' with prompt length: {len(prompt)}")

            # Make request to Ollama
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:

                if response.status == 200:
                    data = await response.json()
                    response_text = data.get('response', '')

                    # Calculate metrics
                    processing_time = (datetime.now() - start_time).total_seconds() * 1000

                    # Estimate token usage (rough approximation)
                    prompt_tokens = len(prompt.split())
                    response_tokens = len(response_text.split())
                    total_tokens = prompt_tokens + response_tokens

                    logger.info(f"✅ Ollama response received. Length: {len(response_text)}, Time: {processing_time:.0f}ms")

                    return {
                        "success": True,
                        "response": response_text,
                        "model_used": model,
                        "processing_time_ms": processing_time,
                        "tokens_used": total_tokens,
                        "prompt_tokens": prompt_tokens,
                        "response_tokens": response_tokens,
                        "raw_data": data
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Ollama API error: HTTP {response.status} - {error_text}")
                    raise Exception(f"Ollama API returned {response.status}: {error_text}")

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"❌ Ollama generation failed: {e}")

            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": processing_time,
                "model_used": model
            }

    async def generate_medical_response(self,
                                      symptoms: str,
                                      model: str = "medllama2:latest",
                                      language: str = "thai") -> Dict[str, Any]:
        """Generate medical response with specialized prompting"""

        system_prompt = """You are a medical AI assistant. Provide accurate, helpful medical information while emphasizing that this is not a substitute for professional medical care.

Guidelines:
- Analyze symptoms carefully
- Provide differential diagnoses when appropriate
- Include confidence levels
- Suggest when to seek immediate medical attention
- Use clear, understandable language
- Include relevant ICD codes when possible
- Never provide specific medication dosages
- Always recommend consulting healthcare professionals"""

        if language == "thai":
            system_prompt += "\n\nRespond in Thai language. Use medical terminology that Thai patients can understand."

        medical_prompt = f"""
Medical Consultation Request:

Patient Symptoms: {symptoms}

Please provide:
1. Primary Assessment with confidence level
2. Possible differential diagnoses
3. Recommended immediate actions
4. When to seek medical attention
5. General care recommendations

Important: This is for informational purposes only and does not replace professional medical consultation.
"""

        return await self.generate(
            model=model,
            prompt=medical_prompt,
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for medical accuracy
            max_tokens=1500
        )

    async def generate_translation(self,
                                 text: str,
                                 source_lang: str = "thai",
                                 target_lang: str = "english",
                                 model: str = "seallm-7b-v2") -> Dict[str, Any]:
        """Generate translation using SeaLLM model"""

        system_prompt = f"""You are a professional translator specializing in {source_lang} to {target_lang} translation.
Provide accurate, natural translations while preserving the original meaning and context.

For medical content:
- Preserve medical terminology accuracy
- Maintain symptom descriptions precisely
- Keep cultural context when relevant"""

        translation_prompt = f"""
Translate the following text from {source_lang} to {target_lang}:

{text}

Provide only the translation without additional explanations.
"""

        return await self.generate(
            model=model,
            prompt=translation_prompt,
            system_prompt=system_prompt,
            temperature=0.2,  # Low temperature for accurate translation
            max_tokens=800
        )

# Global Ollama client instance
ollama_client = OllamaClient()