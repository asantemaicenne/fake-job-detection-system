from typing import Dict

import pytest
from httpx import AsyncClient

from configs.settings import settings


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient) -> None:
    """Verifies the /health probe returns OK and expected metadata."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == settings.PROJECT_NAME
    assert data["version"] == settings.MODEL_VERSION


@pytest.mark.asyncio
async def test_auth_dev_token_generation(client: AsyncClient) -> None:
    """Verifies the dev token endpoint issues valid JWT tokens."""
    response = await client.post(
        f"{settings.API_V1_STR}/auth/dev-token",
        params={"username": "qa_tester", "role": "analyst"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "access_token" in payload
    assert payload["token_type"] == "bearer"
    assert payload["expires_in"] > 0


@pytest.mark.asyncio
async def test_single_analysis_unauthorized(client: AsyncClient) -> None:
    """Ensures unauthenticated requests receive a 401 response."""
    payload = {
        "title": "Software Engineer",
        "company_name": "Acme Tech",
        "description": (
            "Standard legitimate position for software development."
        ),
        "source_platform": "Indeed",
        "poster_id": "recruiter_01",
    }
    response = await client.post(
        f"{settings.API_V1_STR}/analysis/single",
        json=payload,
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_bulk_analysis_forbidden_role(
    client: AsyncClient,
    unauthorized_role_headers: Dict[str, str],
) -> None:
    """Verifies accounts lacking proper permissions receive 403."""
    jobs_batch = [
        {
            "title": "QA Engineer",
            "company_name": "Test Labs",
            "description": (
                "Full-time position writing end-to-end integration tests."
            ),
            "source_platform": "LinkedIn",
            "poster_id": "recruiter_02",
        }
    ]
    response = await client.post(
        f"{settings.API_V1_STR}/analysis/bulk",
        json=jobs_batch,
        headers=unauthorized_role_headers,
    )
    assert response.status_code == 403
    assert "Operation not permitted" in response.json()["detail"]


@pytest.mark.asyncio
async def test_single_analysis_legitimate_job(
    client: AsyncClient,
    api_user_headers: Dict[str, str],
) -> None:
    """Tests normal analysis pipeline on a benign job posting."""
    payload = {
        "title": "Senior Backend Developer",
        "company_name": "Enterprise Solutions LLC",
        "company_domain": "enterprisesolutions.com",
        "description": (
            "We are seeking a skilled Senior Backend Developer experienced "
            "in Python, FastAPI, and MongoDB to join our distributed "
            "infrastructure team. Must possess solid design patterns "
            "knowledge and 5+ years experience."
        ),
        "contact_email": "careers@enterprisesolutions.com",
        "location": "Remote",
        "salary_min": 110000.0,
        "salary_max": 140000.0,
        "source_platform": "LinkedIn",
        "poster_id": "verified_recruiter_101",
    }

    response = await client.post(
        f"{settings.API_V1_STR}/analysis/single",
        json=payload,
        headers=api_user_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["model_version"] == settings.MODEL_VERSION
    assert 0.0 <= data["confidence_score"] <= 1.0
    assert data["is_fake"] is False


@pytest.mark.asyncio
async def test_single_analysis_scam_detection(
    client: AsyncClient,
    analyst_headers: Dict[str, str],
) -> None:
    """Verifies scam indicators trigger a fake classification flag."""
    scam_payload = {
        "title": "URGENT Data Entry - Immediate Start",
        "company_name": "Global Wealth Partners",
        "company_domain": None,
        "description": (
            "URGENT HIRING!! Earn $5000 weekly from home! Immediate start "
            "required. Send wire transfer deposit of $150 via Western Union "
            "or Crypto for training starter kit! Please provide your Social "
            "Security number and date of birth immediately."
        ),
        "contact_email": "quickmoney991@gmail.com",
        "location": "Remote",
        "salary_min": 250000.0,
        "salary_max": 300000.0,
        "source_platform": "Telegram",
        "poster_id": "suspicious_account_99",
    }

    response = await client.post(
        f"{settings.API_V1_STR}/analysis/single",
        json=scam_payload,
        headers=analyst_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_fake"] is True
    assert data["confidence_score"] >= 0.5


@pytest.mark.asyncio
async def test_single_analysis_validation_failure(
    client: AsyncClient,
    api_user_headers: Dict[str, str],
) -> None:
    """Verifies schema validation rejects incomplete payloads with 422."""
    invalid_payload = {
        "title": "X",
        "company_name": "",
        "description": "Too short",
    }
    response = await client.post(
        f"{settings.API_V1_STR}/analysis/single",
        json=invalid_payload,
        headers=api_user_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_bulk_analysis_success(
    client: AsyncClient,
    analyst_headers: Dict[str, str],
) -> None:
    """Tests sequential batch ingestion and processing for multiple jobs."""
    jobs_batch = [
        {
            "title": f"Staff Software Engineer #{i}",
            "company_name": f"TechCorp {i}",
            "company_domain": "techcorp.io",
            "description": (
                "Full-time software engineering role focused on "
                "high-throughput backend microservices."
            ),
            "contact_email": f"hr{i}@techcorp.io",
            "location": "Remote",
            "source_platform": "Indeed",
            "poster_id": f"recruiter_batch_{i}",
        }
        for i in range(3)
    ]

    response = await client.post(
        f"{settings.API_V1_STR}/analysis/bulk",
        json=jobs_batch,
        headers=analyst_headers,
    )
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    assert len(results) == 3
    for pred in results:
        assert "job_id" in pred
        assert "is_fake" in pred


@pytest.mark.asyncio
async def test_bulk_analysis_limit_exceeded(
    client: AsyncClient,
    admin_headers: Dict[str, str],
) -> None:
    """Ensures batch submissions over 100 items are rejected with 400."""
    oversized_batch = [
        {
            "title": f"Job Posting #{i}",
            "company_name": "Bulk Corp",
            "description": (
                "Valid job description meeting standard character count "
                "constraints."
            ),
            "source_platform": "Monster",
            "poster_id": "spammer_account",
        }
        for i in range(101)
    ]

    response = await client.post(
        f"{settings.API_V1_STR}/analysis/bulk",
        json=oversized_batch,
        headers=admin_headers,
    )
    assert response.status_code == 400
    assert "Batch limit exceeded" in response.json()["detail"]


@pytest.mark.asyncio
async def test_submit_feedback_and_get_results(
    client: AsyncClient,
    admin_headers: Dict[str, str],
) -> None:
    """Tests feedback submission and paginated historical queries."""
    feedback_payload = {
        "prediction_id": "65f000000000000000000001",
        "job_id": "65f000000000000000000002",
        "ground_truth_is_fake": True,
        "reviewer_id": "auditor_smith",
        "comments": "Confirmed advance-fee scam asking for money upfront.",
    }
    feedback_res = await client.post(
        f"{settings.API_V1_STR}/analysis/feedback",
        json=feedback_payload,
        headers=admin_headers,
    )
    assert feedback_res.status_code == 201
    feedback_data = feedback_res.json()
    assert feedback_data["ground_truth_is_fake"] is True
    assert feedback_data["reviewer_id"] == "auditor_smith"

    results_res = await client.get(
        f"{settings.API_V1_STR}/analysis/results?skip=0&limit=10",
        headers=admin_headers,
    )
    assert results_res.status_code == 200
    assert isinstance(results_res.json(), list)
