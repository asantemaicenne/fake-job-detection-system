import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import joblib
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from configs.settings import settings
from src.api.endpoints import analysis
from src.api.middleware.auth import Token, create_access_token
from src.api.middleware.metrics import PrometheusMiddleware, metrics_endpoint
from src.data.repositories.database import DatabaseManager
from src.models.ethics.safeguards import HumanInTheLoop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan manager handling async startup and shutdown sequences.
    """
    # 1. Connect database pool
    logger.info("Connecting to MongoDB datastore...")
    await DatabaseManager.connect()

    # 2. Load serialized ML model pipeline and explainability engine
    artifact_path = os.path.join(
        settings.MODEL_ARTIFACTS_DIR,
        f"{settings.MODEL_VERSION}.pkl",
    )
    if os.path.exists(artifact_path):
        try:
            logger.info(f"Loading model artifact from {artifact_path}...")
            app.state.model_pipeline = joblib.load(artifact_path)
            logger.info("Model pipeline loaded successfully.")
        except Exception as e:
            logger.error(
                "Failed to load model pipeline: %s. Falling back to "
                "heuristics.",
                e,
            )
            app.state.model_pipeline = None
    else:
        logger.warning(
            f"No artifact found at '{artifact_path}'. Operating in heuristic "
            "detection mode."
        )
        app.state.model_pipeline = None

    app.state.explainer = None
    app.state.hitl = HumanInTheLoop(
        lower_threshold=settings.HITL_UNCERTAINTY_LOWER,
        upper_threshold=settings.HITL_UNCERTAINTY_UPPER,
    )

    yield

    # 3. Graceful shutdown
    logger.info("Disconnecting from MongoDB...")
    await DatabaseManager.disconnect()
    logger.info("Application shutdown completed.")


# Initialize core FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.MODEL_VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)

# Apply CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Apply Prometheus observability middleware
if settings.PROMETHEUS_METRICS_ENABLED:
    app.add_middleware(PrometheusMiddleware)
    app.add_api_route(
        "/metrics",
        metrics_endpoint,
        methods=["GET"],
        include_in_schema=False,
    )

# Register functional route groups
app.include_router(
    analysis.router,
    prefix=f"{settings.API_V1_STR}/analysis",
    tags=["Job Analysis & Predictions"],
)


@app.get("/health", tags=["System"], status_code=status.HTTP_200_OK)
async def health_check() -> dict:
    """Basic health check probe for container orchestrators."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.MODEL_VERSION,
        "environment": settings.ENVIRONMENT,
    }


@app.post(
    f"{settings.API_V1_STR}/auth/dev-token",
    response_model=Token,
    tags=["Authentication"],
    status_code=status.HTTP_200_OK,
)
async def generate_development_token(
    username: str = "dev_admin",
    role: str = "admin",
) -> Token:
    """
    Utility endpoint to issue development JWT tokens without
    external IdP setup.
    """
    access_token = create_access_token(subject=username, role=role)
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
