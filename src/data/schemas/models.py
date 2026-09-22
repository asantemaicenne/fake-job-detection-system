from datetime import datetime, timezone
from typing import Annotated, Any, Dict, Optional

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field
from pydantic.functional_validators import BeforeValidator


def validate_object_id(v: Any) -> str:
    """Validates and converts BSON ObjectId or string representations."""
    if isinstance(v, ObjectId):
        return str(v)
    if isinstance(v, str) and ObjectId.is_valid(v):
        return v
    if v is None:
        return v
    raise ValueError(f"Invalid ObjectId: {v}")


PyObjectId = Annotated[str, BeforeValidator(validate_object_id)]


class MongoBaseModel(BaseModel):
    """Base model providing MongoDB primary key mapping and shared JSON serialization."""

    id: Optional[PyObjectId] = Field(default=None, alias="_id")

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat(),
            ObjectId: lambda oid: str(oid),
        },
    )


class RawJob(MongoBaseModel):
    """Ingested job advertisement payload before processing."""

    title: str = Field(..., min_length=2, max_length=255, description="Official job title")
    company_name: str = Field(..., min_length=1, max_length=255, description="Hiring organization")
    company_domain: Optional[str] = Field(default=None, description="Official corporate website domain")
    description: str = Field(..., min_length=10, description="Full job description body")
    contact_email: Optional[str] = Field(default=None, description="Contact/recruiter email address")
    location: str = Field(default="Remote", description="Job location or Remote designation")
    salary_min: Optional[float] = Field(default=None, ge=0.0, description="Minimum stated compensation")
    salary_max: Optional[float] = Field(default=None, ge=0.0, description="Maximum stated compensation")
    posted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the posting went live",
    )
    source_platform: str = Field(..., description="Platform identifier (e.g., LinkedIn, Indeed, Internal)")
    poster_id: str = Field(..., description="Unique account identifier of the job submitter")


class JobFeatures(MongoBaseModel):
    """Engineered feature vector passed to ML inference models."""

    job_id: str = Field(..., description="Foreign key reference to RawJob._id")

    # Explicit scam triggers
    has_payment_request: bool = Field(default=False, description="Flag for upfront fee/wire transfer solicitations")
    has_pii_request: bool = Field(default=False, description="Flag for sensitive identity requests (SSN/Banking)")
    salary_anomaly_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Outlier score for salary-to-detail ratio")

    # Linguistic analysis
    urgency_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Prevalence of artificial urgency terminology")
    grammar_anomaly_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Capitalization and punctuation abuse score")

    # Structural indicators
    is_generic_email: bool = Field(default=False, description="Contact email belongs to free/public mailbox provider")
    missing_company_url: bool = Field(default=False, description="Company website domain is omitted")

    # Network signals
    poster_reputation_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Calculated trustworthiness of poster account")
    duplicate_count: int = Field(default=0, ge=0, description="Number of identical postings across platforms")

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Feature extraction timestamp",
    )


class Prediction(MongoBaseModel):
    """Inference output produced by the ensemble classification engine."""

    job_id: str = Field(..., description="Foreign key reference to RawJob._id")
    model_version: str = Field(..., description="Semantic version of model used for prediction")
    is_fake: bool = Field(..., description="Binary classification output: True if fraudulent")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Posterior probability of being fake")
    requires_human_review: bool = Field(default=False, description="Flagged for HITL queue if confidence is ambiguous")
    shap_values: Dict[str, float] = Field(default_factory=dict, description="SHAP feature attribution weights")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when inference completed",
    )


class Feedback(MongoBaseModel):
    """Ground truth auditing labels supplied by human reviewers."""

    prediction_id: str = Field(..., description="Foreign key reference to Prediction._id")
    job_id: str = Field(..., description="Foreign key reference to RawJob._id")
    ground_truth_is_fake: bool = Field(..., description="Verified label: True if confirmed fake")
    reviewer_id: str = Field(..., description="Identifier of human analyst who performed review")
    comments: Optional[str] = Field(default=None, description="Optional qualitative audit notes")
    reviewed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Audit verification timestamp",
    )
