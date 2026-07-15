from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from app.schemas.models import AgentResponse


class EpisodeOutcome(StrEnum):
    RECOMMENDATION_READY = "recommendation_ready"
    MISSING_INFORMATION = "missing_information"
    PLAN_REJECTED = "plan_rejected"
    EXECUTION_FAILED = "execution_failed"
    NO_RECOMMENDATION = "no_recommendation"
    NO_ELIGIBLE_APPOINTMENTS = "no_eligible_appointments"
    RECOMMENDATION_REJECTED = "recommendation_rejected"
    UNKNOWN_FAILURE = "unknown_failure"


class AgentExecutionEpisode(BaseModel):
    """A historical record of one complete agent request."""

    episode_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    user_message: str
    outcome: EpisodeOutcome
    response: AgentResponse
    duration_ms: int = Field(ge=0)
