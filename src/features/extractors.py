import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from src.data.schemas.models import JobFeatures, RawJob


class ExplicitIndicatorsExtractor:
    """Detects overt scam patterns: advance-fee fraud and PII harvesting."""

    PAYMENT_REGEX = re.compile(
        r"\b(wire transfer|western union|moneygram|crypto|cryptocurrency|bitcoin|usdt|"
        r"processing fee|registration fee|starter kit fee|training fee|cashier'?s check|"
        r"gift card|venmo|zelle|upfront deposit)\b",
        re.IGNORECASE,
    )

    PII_REGEX = re.compile(
        r"\b(social security|ssn|passport number|national id|driver'?s license|"
        r"bank account number|routing number|credit card|cvv|date of birth|dob)\b",
        re.IGNORECASE,
    )

    @classmethod
    def extract(cls, text: str, salary_min: Optional[float], salary_max: Optional[float]) -> Dict[str, Any]:
        """Scans job content for explicit red flags and salary incongruities."""
        has_payment = bool(cls.PAYMENT_REGEX.search(text))
        has_pii = bool(cls.PII_REGEX.search(text))

        salary_anomaly = 0.0
        effective_salary = salary_max if salary_max is not None else salary_min

        # Flag unusually high compensation paired with very brief descriptions
        word_count = len(text.split())
        if effective_salary is not None:
            if effective_salary >= 180000.0 and word_count < 80:
                salary_anomaly = 1.0
            elif effective_salary >= 120000.0 and word_count < 40:
                salary_anomaly = 0.75
            elif effective_salary <= 0:
                salary_anomaly = 0.50

        return {
            "has_payment_request": has_payment,
            "has_pii_request": has_pii,
            "salary_anomaly_score": salary_anomaly,
        }


class LinguisticPatternsExtractor:
    """Analyzes psychological urgency cues, capitalization, and punctuation anomalies."""

    URGENCY_REGEX = re.compile(
        r"\b(immediate start|start today|urgent hiring|apply immediately|"
        r"limited positions|act fast|offer expires|no experience necessary|"
        r"make \$[0-9]+ daily|guaranteed income)\b",
        re.IGNORECASE,
    )

    EXCLAMATION_REGEX = re.compile(r"!{2,}|\?{2,}")

    @classmethod
    def extract(cls, text: str) -> Dict[str, float]:
        """Calculates normalized linguistic pressure and structural anomaly scores."""
        words = text.split()
        if not words:
            return {"urgency_score": 0.0, "grammar_anomaly_score": 0.0}

        # Urgency scoring (saturation cap at 4 distinct triggers)
        urgency_matches = len(cls.URGENCY_REGEX.findall(text))
        urgency_score = min(urgency_matches * 0.25, 1.0)

        # Capitalization abuse scoring
        caps_words = sum(1 for w in words if w.isupper() and len(w) > 1 and w.isalpha())
        caps_ratio = caps_words / len(words)
        caps_penalty = min(caps_ratio / 0.12, 1.0)

        # Punctuation anomalies (multiple exclamation/question marks)
        punct_matches = len(cls.EXCLAMATION_REGEX.findall(text))
        punct_penalty = min(punct_matches * 0.20, 1.0)

        grammar_anomaly_score = max(caps_penalty, punct_penalty)

        return {
            "urgency_score": round(urgency_score, 4),
            "grammar_anomaly_score": round(grammar_anomaly_score, 4),
        }


class StructuralSignalsExtractor:
    """Examines domain credibility and infrastructure authenticity."""

    GENERIC_DOMAINS = {
        "gmail.com",
        "yahoo.com",
        "hotmail.com",
        "outlook.com",
        "aol.com",
        "protonmail.com",
        "proton.me",
        "icloud.com",
        "mail.com",
        "zoho.com",
        "yandex.com",
    }

    @classmethod
    def extract(cls, job: RawJob) -> Dict[str, bool]:
        """Checks for missing company presence and generic contact channels."""
        is_generic = False
        if job.contact_email and "@" in job.contact_email:
            email_domain = job.contact_email.split("@")[-1].strip().lower()
            is_generic = email_domain in cls.GENERIC_DOMAINS

        missing_url = (job.company_domain is None) or (len(job.company_domain.strip()) == 0)

        return {
            "is_generic_email": is_generic,
            "missing_company_url": missing_url,
        }


class NetworkFeaturesExtractor:
    """Calculates behavioral posting frequencies and cross-platform duplication via MongoDB lookups."""

    @classmethod
    async def extract(cls, db: AsyncIOMotorDatabase, job: RawJob) -> Dict[str, Any]:
        """Evaluates historical posting patterns to detect bot spam or syndication rings."""
        raw_jobs = db["raw_jobs"]

        # Calculate poster volume in the preceding 7 days
        seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
        recent_post_count = await raw_jobs.count_documents({
            "poster_id": job.poster_id,
            "posted_at": {"$gte": seven_days_ago},
        })

        # Calculate poster reputation (degrades if volume indicates bulk automated posting)
        reputation = 1.0
        if recent_post_count > 30:
            reputation = max(0.1, 1.0 - ((recent_post_count - 30) * 0.03))

        # Check for duplicate job postings across other platforms
        query: Dict[str, Any] = {
            "title": job.title,
            "company_name": job.company_name,
        }
        if job.id is not None:
            query["_id"] = {"$ne": job.id}

        duplicate_count = await raw_jobs.count_documents(query)

        return {
            "poster_reputation_score": round(reputation, 4),
            "duplicate_count": duplicate_count,
        }


class FeatureEngineer:
    """Facade orchestrating all extractors to generate a unified JobFeatures instance."""

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self.db = db

    async def generate_features(self, job: RawJob) -> JobFeatures:
        """Runs heuristic, NLP, structural, and network feature extractors."""
        if job.id is None:
            raise ValueError("RawJob must possess an assigned _id before feature extraction.")

        explicit_feats = ExplicitIndicatorsExtractor.extract(
            text=job.description,
            salary_min=job.salary_min,
            salary_max=job.salary_max,
        )
        linguistic_feats = LinguisticPatternsExtractor.extract(text=job.description)
        structural_feats = StructuralSignalsExtractor.extract(job=job)
        network_feats = await NetworkFeaturesExtractor.extract(db=self.db, job=job)

        features_payload = {
            "job_id": str(job.id),
            **explicit_feats,
            **linguistic_feats,
            **structural_feats,
            **network_feats,
        }

        return JobFeatures(**features_payload)
