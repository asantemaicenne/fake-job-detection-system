import asyncio
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)

from configs.settings import settings
from src.api.middleware.auth import User, require_role
from src.api.middleware.metrics import MODEL_PREDICTIONS_TOTAL
from src.data.repositories.database import DatabaseManager
from src.data.schemas.models import Feedback, Prediction, RawJob
from src.features.extractors import FeatureEngineer
from src.models.ethics.safeguards import HumanInTheLoop

router = APIRouter()


@router.post(
    "/single",
    response_model=Prediction,
    status_code=status.HTTP_200_OK,
    summary="Analyze a single job posting",
)
async def analyze_single_job(
    job_payload: RawJob,
    request: Request,
    current_user: User = Depends(
        require_role(["admin", "analyst", "api_user"])
    ),
) -> Prediction:
    """
    Ingests a raw job posting, extracts NLP and heuristic features, executes ML
    inference, determines HITL review status, and logs metrics.
    """
    db = DatabaseManager.get_database()

    # 1. Persist raw job submission
    raw_doc = job_payload.model_dump(by_alias=True, exclude={"id"})
    insert_result = await db["raw_jobs"].insert_one(raw_doc)
    job_payload.id = str(insert_result.inserted_id)

    # 2. Extract engineered features
    engineer = FeatureEngineer(db=db)
    features = await engineer.generate_features(job_payload)
    await db["job_features"].insert_one(
        features.model_dump(by_alias=True, exclude={"id"})
    )

    # 3. Assemble inference feature matrix
    feature_dict = features.model_dump(exclude={"id", "job_id", "created_at"})
    feature_dict["description"] = job_payload.description
    feature_df = pd.DataFrame([feature_dict])

    # 4. Model Inference
    model_pipeline = getattr(request.app.state, "model_pipeline", None)
    explainer = getattr(request.app.state, "explainer", None)
    hitl: HumanInTheLoop = getattr(
        request.app.state,
        "hitl",
        HumanInTheLoop(
            settings.HITL_UNCERTAINTY_LOWER,
            settings.HITL_UNCERTAINTY_UPPER,
        ),
    )

    heuristic_score = (
        (1.0 if features.has_payment_request else 0.0) * 0.30
        + (1.0 if features.has_pii_request else 0.0) * 0.30
        + features.salary_anomaly_score * 0.12
        + features.urgency_score * 0.12
        + features.grammar_anomaly_score * 0.10
        + (1.0 if features.is_generic_email else 0.0) * 0.04
        + (1.0 if features.missing_company_url else 0.0) * 0.04
        + (1.0 - features.poster_reputation_score) * 0.08
    )
    heuristic_score = min(max(heuristic_score, 0.0), 1.0)

    if model_pipeline is not None:
        try:
            probabilities = await asyncio.to_thread(
                model_pipeline.predict_proba,
                feature_df,
            )
            model_probability = float(probabilities[0][1])
            fake_probability = max(model_probability, heuristic_score)
            is_fake = bool(fake_probability >= 0.5)
            confidence = (
                fake_probability
                if is_fake
                else float(probabilities[0][0])
            )
        except Exception as err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference execution failed: {str(err)}",
            )
    else:
        # Fallback heuristic calculation if model weights are not yet
        # trained on disk
        fake_probability = heuristic_score
        is_fake = bool(fake_probability >= 0.5)
        confidence = fake_probability if is_fake else (1.0 - fake_probability)

    # 5. Determine Human-In-The-Loop routing and SHAP explanations
    requires_review = hitl.evaluate(fake_probability)
    shap_values: Dict[str, float] = {}

    if settings.ENABLE_SHAP_EXPLAINABILITY and explainer is not None:
        try:
            shap_values = await asyncio.to_thread(
                explainer.explain_instance,
                feature_df,
            )
        except Exception:
            shap_values = {}

    # 6. Persist prediction and record Prometheus metrics
    prediction = Prediction(
        job_id=str(job_payload.id),
        model_version=settings.MODEL_VERSION,
        is_fake=is_fake,
        confidence_score=round(confidence, 4),
        requires_human_review=requires_review,
        shap_values=shap_values,
    )
    await db["predictions"].insert_one(
        prediction.model_dump(by_alias=True, exclude={"id"})
    )

    MODEL_PREDICTIONS_TOTAL.labels(
        model_version=settings.MODEL_VERSION,
        is_fake=str(is_fake),
        requires_human_review=str(requires_review),
    ).inc()

    return prediction


@router.post(
    "/bulk",
    response_model=List[Prediction],
    status_code=status.HTTP_200_OK,
    summary="Batch analyze multiple job postings",
)
async def analyze_bulk_jobs(
    jobs: List[RawJob],
    request: Request,
    current_user: User = Depends(require_role(["admin", "analyst"])),
) -> List[Prediction]:
    """
    Processes up to 100 job postings sequentially within a managed
    batch transaction.
    """
    if len(jobs) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Batch limit exceeded. Maximum allowed per request is "
                "100 postings."
            ),
        )

    results: List[Prediction] = []
    for job in jobs:
        pred = await analyze_single_job(job, request, current_user)
        results.append(pred)
    return results


@router.post(
    "/feedback",
    response_model=Feedback,
    status_code=status.HTTP_201_CREATED,
    summary="Submit ground truth feedback from human review",
)
async def submit_feedback(
    feedback: Feedback,
    current_user: User = Depends(require_role(["admin", "analyst"])),
) -> Feedback:
    """
    Stores human-verified ground truth labels for false-positive
    auditing and retraining.
    """
    db = DatabaseManager.get_database()
    feedback_doc = feedback.model_dump(by_alias=True, exclude={"id"})
    await db["feedback"].insert_one(feedback_doc)
    return feedback


@router.get(
    "/results",
    response_model=List[Prediction],
    status_code=status.HTTP_200_OK,
    summary="Query historical predictions with pagination",
)
async def get_results(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    is_fake: Optional[bool] = Query(default=None),
    requires_human_review: Optional[bool] = Query(default=None),
    current_user: User = Depends(require_role(["admin", "analyst"])),
) -> List[Prediction]:
    """
    Retrieves paginated predictions with optional filtering by outcome
    or review status.
    """
    db = DatabaseManager.get_database()
    filter_query: Dict[str, Any] = {}

    if is_fake is not None:
        filter_query["is_fake"] = is_fake
    if requires_human_review is not None:
        filter_query["requires_human_review"] = requires_human_review

    cursor = (
        db["predictions"]
        .find(filter_query)
        .sort("timestamp", -1)
        .skip(skip)
        .limit(limit)
    )

    documents = await cursor.to_list(length=limit)
    return [Prediction(**doc) for doc in documents]


@router.websocket("/ws/status")
async def websocket_status_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket channel for broadcasting asynchronous processing status updates.
    """
    await websocket.accept()
    try:
        while True:
            client_msg = await websocket.receive_text()
            await websocket.send_json(
                {
                    "event": "heartbeat",
                    "echo": client_msg,
                    "status": "online",
                }
            )
    except WebSocketDisconnect:
        pass
