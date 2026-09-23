import math
import re
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, Optional

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.functional_validators import BeforeValidator


def validate_object_id(v: Any) -> str:
    if isinstance(v, ObjectId):
        return str(v)
    if isinstance(v, str) and ObjectId.is_valid(v):
        return v
    if v is None:
        return v
    raise ValueError(f"Invalid ObjectId: {v}")


PyObjectId = Annotated[str, BeforeValidator(validate_object_id)]

# Standard English dictionary check (common functional words)
COMMON_ENGLISH_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her",
    "she", "or", "an", "will", "my", "one", "all", "would", "there",
    "their", "what", "so", "up", "out", "if", "about", "who", "get",
    "which", "go", "me", "when", "make", "can", "like", "time", "no",
    "just", "him", "know", "take", "people", "into", "year", "your",
    "good", "some", "could", "them", "see", "other", "than", "then",
    "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first",
    "well", "way", "even", "new", "want", "because", "any", "these",
    "give", "day", "most", "us", "job", "position", "team", "skills",
    "experience", "required", "candidate", "role", "company", "duties",
    "responsibilities", "requirements", "apply", "developer", "engineer",
    "manager", "assistant", "salary", "benefits", "remote", "full-time"
}


def calculate_shannon_entropy(text: str) -> float:
    """Calculates Shannon entropy to detect random keyboard mashing."""
    if not text:
        return 0.0
    entropy = 0.0
    text_len = len(text)
    for count in [text.count(c) for c in set(text)]:
        p_x = count / text_len
        entropy -= p_x * math.log2(p_x)
    return entropy


class MongoBaseModel(BaseModel):
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
    title: str = Field(
        ..., min_length=3, max_length=255, description="Official job title"
    )
    company_name: str = Field(
        ..., min_length=2, max_length=255, description="Hiring organization"
    )
    company_domain: Optional[str] = Field(
        default=None, description="Corporate website domain"
    )
    description: str = Field(
        ..., min_length=40, description="Full job description body"
    )
    contact_email: Optional[str] = Field(
        default=None, description="Contact email address"
    )
    location: str = Field(default="Remote", description="Job location")
    salary_min: Optional[float] = Field(default=None, ge=0.0, le=1000000.0)
    salary_max: Optional[float] = Field(default=None, ge=0.0, le=2000000.0)
    posted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    source_platform: str = Field(..., description="Platform identifier")
    poster_id: str = Field(..., description="Poster ID")

    @field_validator("title")
    @classmethod
    def validate_title_entropy(cls, v: str) -> str:
        words = re.findall(r"\b[a-zA-Z]{2,}\b", v.lower())
        if not words:
            raise ValueError(
                "Job title must contain recognizable alphabetic terms."
            )
        entropy = calculate_shannon_entropy(v)
        if entropy < 1.5 and len(v) > 8:
            raise ValueError(
                f"Job title has unnaturally low character entropy "
                f"({entropy:.2f})."
            )
        return v

    @field_validator("company_domain")
    @classmethod
    def validate_fqdn(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not v.strip():
            return None
        cleaned = v.strip().lower()
        fqdn_pattern = re.compile(
            r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+"
            r"[a-zA-Z]{2,63}$"
        )
        if not fqdn_pattern.match(cleaned):
            raise ValueError(
                f"Invalid corporate domain '{v}'. Must be a valid FQDN with "
                "a registered TLD (e.g., 'company.com')."
            )
        return cleaned

    @field_validator("description")
    @classmethod
    def validate_description_lexical_coherence(cls, v: str) -> str:
        words = re.findall(r"\b[a-zA-Z]{2,}\b", v.lower())
        if len(words) < 5:
            raise ValueError(
                "Job description is too brief or contains insufficient "
                "textual tokens."
            )

        recognized_tokens = sum(1 for w in words if w in COMMON_ENGLISH_WORDS)
        dictionary_ratio = recognized_tokens / len(words)

        if dictionary_ratio < 0.25:
            raise ValueError(
                f"Input rejected as synthetic noise or gibberish. Only "
                f"{dictionary_ratio:.1%} of tokens match valid dictionary "
                "terms (minimum threshold is 25%)."
            )
        return v

    @field_validator("salary_max")
    @classmethod
    def validate_salary_bounds(cls, v: Optional[float], info) -> Optional[float]:
        s_min = info.data.get("salary_min")
        if v is not None and s_min is not None and v < s_min:
            raise ValueError(
                f"Maximum salary (${v:,.2f}) cannot be lower than minimum "
                f"salary (${s_min:,.2f})."
            )
        if v is not None and v > 500000.0:
            desc = info.data.get("description", "")
            word_count = len(desc.split())
            if word_count < 40:
                raise ValueError(
                    f"Salary of ${v:,.2f} requires a descriptive scope "
                    f"of at least 40 words. Found {word_count} words."
                )
        return v


class JobFeatures(MongoBaseModel):
    job_id: str = Field(...)
    has_payment_request: bool = Field(default=False)
    has_pii_request: bool = Field(default=False)
    salary_anomaly_score: float = Field(default=0.0, ge=0.0, le=1.0)
    urgency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    grammar_anomaly_score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_generic_email: bool = Field(default=False)
    missing_company_url: bool = Field(default=False)
    poster_reputation_score: float = Field(default=1.0, ge=0.0, le=1.0)
    duplicate_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class Prediction(MongoBaseModel):
    job_id: str = Field(...)
    model_version: str = Field(...)
    is_fake: bool = Field(...)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    requires_human_review: bool = Field(default=False)
    shap_values: Dict[str, float] = Field(default_factory=dict)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class Feedback(MongoBaseModel):
    prediction_id: str = Field(...)
    job_id: str = Field(...)
    ground_truth_is_fake: bool = Field(...)
    reviewer_id: str = Field(...)
    comments: Optional[str] = Field(default=None)
    reviewed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
