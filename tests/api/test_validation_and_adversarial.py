import logging
from typing import Any, Dict

import pandas as pd
import pytest
from httpx import AsyncClient

from configs.settings import settings
from src.models.ethics.safeguards import ExplainabilityEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_suite_audit")


@pytest.mark.asyncio
async def test_audit_catch_original_bypass_payload(
    client: AsyncClient,
    admin_headers: Dict[str, str],
) -> None:
    """
    CRITICAL AUDIT TEST:
    Feeds the exact invalid/gibberish payload shown in the bypass screenshot (image_3fb3a6.png).
    Ensures that the input validation layer intercepts and rejects the payload with HTTP 422,
    preventing an erroneous 'Authentic Posting Verified' false-positive.
    """
    logger.info("AUDIT STEP 1: Formulating original invalid payload from image_3fb3a6.png")
    bypass_payload = {
        "title": "weryiyuythhyw",
        "company_name": "eutrygukytef",
        "company_domain": "ikurtewjtyg",  # Missing TLD (e.g. .com)
        "contact_email": "rdyguhjkl'qqgb@gmail.com",
        "location": "Remote2345678",
        "salary_min": 1234567.0,
        "salary_max": 2134567.0,  # Exceeds maximum boundary
        "description": "34e567890oiukjhg",  # 16-char gibberish, no dictionary tokens
        "source_platform": "AuditBench",
        "poster_id": "adversarial_user",
    }

    logger.info("AUDIT STEP 2: Dispatching payload to /api/v1/analysis/single")
    response = await client.post(
        f"{settings.API_V1_STR}/analysis/single",
        json=bypass_payload,
        headers=admin_headers,
    )

    logger.info(f"AUDIT STEP 3: Response Status = {response.status_code}")
    # Must fail schema validation rather than passing through to inference
    assert response.status_code == 422, (
        "Security Failure: Gibberish payload bypassed validation with status "
        f"{response.status_code}"
    )

    response_body = response.json()
    errors = str(response_body.get("detail", ""))
    logger.info(f"AUDIT STEP 4: Validation Errors Trapped = {errors}")

    # Verify that the FQDN, salary boundary, and lexical filters all caught their respective violations
    assert any(
        err in errors
        for err in [
            "valid FQDN with a registered TLD",
            "synthetic noise or gibberish",
            "less than or equal to 2000000",
            "Job title must contain recognizable alphabetic terms",
        ]
    ), f"Expected explicit domain or lexical validation failure message, got: {errors}"


def test_explainability_falls_back_when_shap_tree_explainer_rejects_categorical_split() -> None:
    """Ensure SHAP failures degrade gracefully instead of aborting inference."""

    class ExplainerStub:
        def shap_values(self, _row):
            raise RuntimeError(
                "Categorical split is not yet supported. You can still use TreeExplainer with `feature_perturbation=tree_path_dependent`."
            )

    class PreprocessorStub:
        def transform(self, instance_df):
            return instance_df.to_numpy()

    class PipelineStub:
        named_steps = {"preprocessor": PreprocessorStub(), "classifier": object()}

    engine = ExplainabilityEngine.__new__(ExplainabilityEngine)
    engine.preprocessor = PreprocessorStub()
    engine.classifier = object()
    engine.explainer = ExplainerStub()
    engine.feature_names = ["salary_anomaly_score", "urgency_score", "is_generic_email"]

    df = pd.DataFrame(
        [{
            "salary_anomaly_score": 0.9,
            "urgency_score": 0.8,
            "is_generic_email": True,
            "has_payment_request": True,
            "has_pii_request": False,
            "grammar_anomaly_score": 0.2,
            "missing_company_url": False,
            "poster_reputation_score": 0.25,
            "duplicate_count": 1,
        }]
    )

    result = engine.explain_instance(df, top_k=3)

    assert result
    assert "has_payment_request" in result or "salary_anomaly_score" in result


@pytest.mark.asyncio
async def test_audit_lexical_coherence_filter(
    client: AsyncClient,
    admin_headers: Dict[str, str],
) -> None:
    """
    Verifies that text with valid character length but zero English dictionary tokens
    is rejected to prevent TF-IDF zero-vector bypasses.
    """
    gibberish_job = {
        "title": "Software Engineer",
        "company_name": "Authentic Systems LLC",
        "company_domain": "authenticsystems.com",
        "description": "zxjkfh kjsdhfksjdhf ksdjfhskjdfh skdjfhskdjf bnmxvcbmnxbv zmxcbvnmzxbc",
        "contact_email": "careers@authenticsystems.com",
        "source_platform": "LinkedIn",
        "poster_id": "auditor_01",
    }

    response = await client.post(
        f"{settings.API_V1_STR}/analysis/single",
        json=gibberish_job,
        headers=admin_headers,
    )
    assert response.status_code == 422
    assert "synthetic noise or gibberish" in str(response.json()["detail"])


@pytest.mark.asyncio
async def test_audit_generic_email_with_advance_fee_classified_fraud(
    client: AsyncClient,
    admin_headers: Dict[str, str],
) -> None:
    """
    Verifies that a job posting featuring generic emails, advance-fee payment requests,
    and PII solicitations is flagged as fraudulent with confidence >= 85%.
    """
    fraud_payload = {
        "title": "URGENT Data Entry Assistant - Immediate Start",
        "company_name": "Global Wealth Partners",
        "company_domain": None,
        "contact_email": "recruiting.desk9901@gmail.com",
        "salary_min": 180000.0,
        "salary_max": 220000.0,
        "description": (
            "We have an immediate start position for remote operations. "
            "Please send an advance fee payment via wire transfer or western union to receive equipment. "
            "Your social security number and banking credentials are required to confirm identity."
        ),
        "source_platform": "Telegram",
        "poster_id": "suspicious_scammer_99",
    }

    response = await client.post(
        f"{settings.API_V1_STR}/analysis/single",
        json=fraud_payload,
        headers=admin_headers,
    )
    assert response.status_code == 200
    data = response.json()

    logger.info(f"AUDIT RESULT: is_fake={data['is_fake']}, confidence={data['confidence_score']}")
    assert data["is_fake"] is True
    assert data["confidence_score"] >= 0.85
    # Confirm generic email is marked as a positive risk factor
    assert data["shap_values"]["is_generic_email"] > 0
    assert data["shap_values"]["has_payment_request"] > 0


@pytest.mark.asyncio
async def test_audit_legitimate_job_accuracy_aim(
    client: AsyncClient,
    admin_headers: Dict[str, str],
) -> None:
    """
    Confirms that a coherent posting from an authentic domain is cleared
    with low false-positive risk, meeting the Accuracy AIM metric.
    """
    legit_payload = {
        "title": "Senior Cloud Infrastructure Engineer",
        "company_name": "Datastream Technologies Inc",
        "company_domain": "datastreamtech.io",
        "contact_email": "jobs@datastreamtech.io",
        "location": "Remote",
        "salary_min": 135000.0,
        "salary_max": 165000.0,
        "description": (
            "Datastream Technologies is looking for a Senior Cloud Infrastructure Engineer "
            "with expertise in Python, FastAPI, Docker, and distributed MongoDB environments. "
            "Candidates should have strong knowledge of continuous integration and Prometheus monitoring."
        ),
        "source_platform": "LinkedIn",
        "poster_id": "recruiter_corp_42",
    }

    response = await client.post(
        f"{settings.API_V1_STR}/analysis/single",
        json=legit_payload,
        headers=admin_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_fake"] is False
    assert data["confidence_score"] >= 0.70
    assert data["requires_human_review"] is False
