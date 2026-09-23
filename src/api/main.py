import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from configs.settings import settings
from src.api.endpoints import analysis
from src.api.middleware.auth import Token, create_access_token
from src.api.middleware.metrics import PrometheusMiddleware, metrics_endpoint
from src.data.repositories.database import DatabaseManager
from src.models.ethics.safeguards import ExplainabilityEngine, HumanInTheLoop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup, model loading, and shutdown for the API."""
    # 1. Connect MongoDB
    logger.info("Connecting to MongoDB datastore...")
    await DatabaseManager.connect()

    # 2. Load serialized XGBoost Pipeline artifact
    artifact_path = os.path.join(
        settings.MODEL_ARTIFACTS_DIR,
        f"{settings.MODEL_VERSION}.pkl",
    )
    if os.path.exists(artifact_path):
        try:
            logger.info("Loading model artifact: %s", artifact_path)
            app.state.model_pipeline = joblib.load(artifact_path)
            logger.info("XGBoost pipeline loaded into memory.")

            # 3. Initialize ExplainabilityEngine with background reference data
            if settings.ENABLE_SHAP_EXPLAINABILITY:
                logger.info("Initializing SHAP Explainability Engine...")
                background_baseline = pd.DataFrame(
                    [
                        {
                            "description": (
                                "Standard software engineer with python and "
                                "cloud background"
                            ),
                            "has_payment_request": False,
                            "has_pii_request": False,
                            "salary_anomaly_score": 0.0,
                            "urgency_score": 0.0,
                            "grammar_anomaly_score": 0.0,
                            "is_generic_email": False,
                            "missing_company_url": False,
                            "poster_reputation_score": 1.0,
                            "duplicate_count": 0,
                        },
                        {
                            "description": (
                                "Urgent hiring wire transfer payment upfront "
                                "needed immediately"
                            ),
                            "has_payment_request": True,
                            "has_pii_request": True,
                            "salary_anomaly_score": 1.0,
                            "urgency_score": 0.8,
                            "grammar_anomaly_score": 0.5,
                            "is_generic_email": True,
                            "missing_company_url": True,
                            "poster_reputation_score": 0.1,
                            "duplicate_count": 5,
                        },
                    ]
                )
                app.state.explainer = ExplainabilityEngine(
                    pipeline=app.state.model_pipeline,
                    background_sample=background_baseline,
                )
                logger.info("SHAP Explainer ready for live inference.")
            else:
                app.state.explainer = None
        except (
            AttributeError,
            FileNotFoundError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            logger.error(
                "Failed to initialize ML models: %s. Falling back to "
                "heuristics.",
                exc,
            )
            app.state.model_pipeline = None
            app.state.explainer = None
    else:
        logger.warning(
            "No artifact found at '%s'. Using heuristic fallback.",
            artifact_path,
        )
        app.state.model_pipeline = None
        app.state.explainer = None

    app.state.hitl = HumanInTheLoop(
        lower_threshold=settings.HITL_UNCERTAINTY_LOWER,
        upper_threshold=settings.HITL_UNCERTAINTY_UPPER,
    )
    logger.info("Application startup complete.")

    yield

    # 4. Graceful shutdown
    logger.info("Disconnecting from MongoDB...")
    await DatabaseManager.disconnect()
    logger.info("Application shutdown completed.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.MODEL_VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.PROMETHEUS_METRICS_ENABLED:
    app.add_middleware(PrometheusMiddleware)
    app.add_api_route(
        "/metrics",
        metrics_endpoint,
        methods=["GET"],
        include_in_schema=False,
    )

app.include_router(
    analysis.router,
    prefix=f"{settings.API_V1_STR}/analysis",
    tags=["Job Analysis & Predictions"],
)


@app.get("/health", tags=["System"], status_code=status.HTTP_200_OK)
async def health_check() -> dict:
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
    access_token = create_access_token(subject=username, role=role)
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
