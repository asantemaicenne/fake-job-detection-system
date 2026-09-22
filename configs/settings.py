# configs/settings.py
import os
from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central application configuration loaded from environment variables and .env.
    Validates types and fails fast on configuration errors at application startup.
    """

    # Core Application Settings
    PROJECT_NAME: str = "Fake Job Detection API"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: Literal["dev", "staging", "prod", "test"] = "dev"
    DEBUG: bool = False

    # MongoDB Datastore Settings
    MONGODB_URI: str = "mongodb://root:example@localhost:27017/"
    MONGODB_DB_NAME: str = "job_detection"
    MONGODB_MAX_POOL_SIZE: int = 50
    MONGODB_MIN_POOL_SIZE: int = 10

    # Security & JWT Authentication
    JWT_SECRET_KEY: str = "insecure_dev_secret_replace_in_production_with_32_byte_hex"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 Days

    # Machine Learning Engine Configuration
    MODEL_ARTIFACTS_DIR: str = "./src/models/artifacts"
    MODEL_VERSION: str = "v1.0.0"
    ENABLE_SHAP_EXPLAINABILITY: bool = True
    HITL_UNCERTAINTY_LOWER: float = 0.40
    HITL_UNCERTAINTY_UPPER: float = 0.65

    # Observability & Monitoring
    PROMETHEUS_METRICS_ENABLED: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Cached getter for application settings.
    Ensures .env is read once and reused across request lifecycles.
    """
    return Settings()


# Global settings singleton
settings = get_settings()
