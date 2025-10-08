# Configuration management for Medical AI Backend

import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings"""

    # App settings
    app_name: str = "Medical Chat AI Backend"
    version: str = "2.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # Security
    secret_key: str = "your-secret-key-change-in-production"
    allowed_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:8080",
        "https://your-domain.com"
    ]

    # Database
    database_url: str = "sqlite:///./medical_chat.db"

    # Medical workflow settings
    require_doctor_approval: bool = False  # Enable doctor approval workflow
    auto_approve_low_risk: bool = False   # Auto-approve low-risk consultations

    # AI Model settings
    ollama_url: str = "http://localhost:11434"
    seallm_model: str = "nxphi47/seallm-7b-v2-q4_0:latest"
    medllama_model: str = "medllama2:latest"

    # External APIs
    qdrant_url: str = "https://5ab0afa9-4525-4842-86df-b7662668bf20.us-east4-0.gcp.cloud.qdrant.io:6333"
    qdrant_api_key: str = ""

    # Medical data paths
    medicine_data_path: str = "./data/medicines.csv"
    diagnosis_data_path: str = "./data/diagnoses.csv"
    treatment_data_path: str = "./data/treatments.csv"

    # Training data
    training_data_path: str = "./training-data/enhanced-training.json"
    feedback_data_path: str = "./training-data/doctor-feedback.json"

    # Rate limiting
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000

    # Logging
    log_level: str = "INFO"
    log_file: str = "medical_ai.log"

    # WebSocket
    websocket_port: int = 3001

    # Medical AI specific settings
    max_conversation_history: int = 10
    emergency_confidence_threshold: float = 0.8
    diagnosis_confidence_threshold: float = 0.6
    max_rag_results: int = 5

    # Thai language settings
    enable_dialect_detection: bool = True
    enable_translation_fallback: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables that don't match fields


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Environment-specific configurations
class DevelopmentSettings(Settings):
    """Development environment settings"""
    debug: bool = True
    log_level: str = "DEBUG"
    allowed_origins: List[str] = ["*"]  # Allow all origins in development


class ProductionSettings(Settings):
    """Production environment settings"""
    debug: bool = False
    log_level: str = "WARNING"
    # Strict CORS in production
    allowed_origins: List[str] = [
        "https://your-medical-app.com",
        "https://api.your-medical-app.com"
    ]


class TestingSettings(Settings):
    """Testing environment settings"""
    database_url: str = "sqlite:///./test_medical_chat.db"
    debug: bool = True
    log_level: str = "DEBUG"


def get_settings_for_environment() -> Settings:
    """Get settings based on environment"""
    environment = os.getenv("ENVIRONMENT", "development").lower()

    if environment == "production":
        return ProductionSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


# Medical AI specific constants
class MedicalConstants:
    """Medical AI system constants"""

    # ICD-10 Categories
    ICD_CATEGORIES = {
        "A00-B99": "Infectious and parasitic diseases",
        "C00-D49": "Neoplasms",
        "D50-D89": "Blood disorders",
        "E00-E89": "Endocrine disorders",
        "F01-F99": "Mental disorders",
        "G00-G99": "Nervous system",
        "H00-H59": "Eye disorders",
        "H60-H95": "Ear disorders",
        "I00-I99": "Circulatory system",
        "J00-J99": "Respiratory system",
        "K00-K95": "Digestive system",
        "L00-L99": "Skin disorders",
        "M00-M99": "Musculoskeletal system",
        "N00-N99": "Genitourinary system",
        "O00-O9A": "Pregnancy and childbirth",
        "P00-P96": "Perinatal conditions",
        "Q00-Q99": "Congenital malformations",
        "R00-R99": "Symptoms and signs",
        "S00-T88": "Injury and poisoning",
        "V00-Y99": "External causes",
        "Z00-Z99": "Health status factors"
    }

    # Emergency keywords by language
    EMERGENCY_KEYWORDS = {
        "english": [
            "emergency", "urgent", "severe", "chest pain", "can't breathe",
            "unconscious", "stroke", "heart attack", "bleeding", "overdose"
        ],
        "thai_standard": [
            "ฉุกเฉิน", "เร่งด่วน", "รุนแรง", "ปวดหน้าอก", "หายใจไม่ออก",
            "หมดสติ", "โรคหลอดเลือดสมอง", "หัวใจวาย", "เลือดออก"
        ],
        "thai_northern": [
            "จุกแล้ว", "จุกโพด", "เจ็บแล้ว", "หายใจไม่ออก", "จุกหน้าอก"
        ],
        "thai_isan": [
            "บักแล้วโพด", "แล้งโพด", "เจ็บบักแล้ว", "แล้งหน้าอก"
        ],
        "thai_southern": [
            "ปวดหัง", "เจ็บหัง", "ปวดโพดหัง", "หายใจไม่ออกหัง"
        ]
    }

    # Triage levels
    TRIAGE_LEVELS = {
        1: {"name": "Resuscitation", "color": "red", "time": "Immediate"},
        2: {"name": "Emergency", "color": "orange", "time": "Within 10 minutes"},
        3: {"name": "Urgent", "color": "yellow", "time": "Within 30 minutes"},
        4: {"name": "Semi-urgent", "color": "green", "time": "Within 60 minutes"},
        5: {"name": "Non-urgent", "color": "blue", "time": "Within 120 minutes"}
    }

    # Common medication categories
    MEDICATION_CATEGORIES = {
        "pain_relief": ["paracetamol", "ibuprofen", "aspirin", "พาราเซตามอล"],
        "antibiotics": ["amoxicillin", "ampicillin", "อะม็อกซิซิลิน"],
        "antacids": ["omeprazole", "antacid", "โอเมพราโซล"],
        "antihistamines": ["cetirizine", "loratadine", "เซทิริซีน"],
        "cardiovascular": ["atenolol", "amlodipine", "อะเทนอลอล"]
    }


# Export settings
__all__ = [
    "Settings",
    "get_settings",
    "get_settings_for_environment",
    "MedicalConstants"
]