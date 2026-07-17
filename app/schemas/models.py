from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator



class UtilityWeights(BaseModel):
    availability: float = 0.35
    distance: float = 0.25
    provider_preference: float = 0.20
    time_preference: float = 0.15
    cost: float = 0.05


class PatientPreferences(BaseModel):
    preferred_provider: str | None = None
    preferred_specialty: str = "primary care"
    preferred_days: list[str] = Field(default_factory=list)
    preferred_time_of_day: Literal["morning", "afternoon", "evening"] | None = None
    #max_distance_miles: float = 25.0
    max_distance_miles: float | None = None
    earliest_date: str | None = None
    latest_date: str | None = None
    weights: UtilityWeights = Field(default_factory=UtilityWeights)


class AppointmentSlot(BaseModel):
    slot_id: str
    provider_id: str
    provider_name: str
    specialty: str
    start_time: datetime
    clinic_name: str
    distance_miles: float
    estimated_cost: float
    in_network: bool = True


class UtilityBreakdown(BaseModel):
    availability: float
    distance: float
    provider_preference: float
    time_preference: float
    cost: float


class ScoredOption(BaseModel):
    slot: AppointmentSlot
    utility_score: float
    breakdown: UtilityBreakdown
    explanation: str


class UtilityDecision(BaseModel):
    goal: str
    selected: ScoredOption | None
    alternatives: list[ScoredOption] = Field(default_factory=list)
    constraints_applied: list[str] = Field(default_factory=list)
    requires_confirmation: bool = True


class PlanAction(str, Enum):
    FIND_APPOINTMENTS = "find_appointments"
    EVALUATE_OPTIONS = "evaluate_options"
    RANK_OPTIONS = "rank_options"


class PlanStep(BaseModel):
    step_id: str
    action: PlanAction
    purpose: str
    depends_on: list[str] = Field(default_factory=list)


class PlanStepExecution(BaseModel):
    step_id: str
    action: str
    status: Literal["completed", "skipped", "failed"]
    detail: str
    source: Literal["planner", "platform"] = "planner"


class PlannerOutput(BaseModel):
    goal: str
    preferences: PatientPreferences
    steps: list[PlanStep]
    missing_required_information: list[str] = Field(default_factory=list)


class RankedCandidate(BaseModel):
    slot_id: str
    rank: int
    reasoning: str


class UtilityAgentOutput(BaseModel):
    recommended_slot_id: str | None
    ranked_candidates: list[RankedCandidate] = Field(default_factory=list)
    recommendation_summary: str


class PlanCriticOutput(BaseModel):
    approved: bool
    issues: list[str] = Field(default_factory=list)
    summary: str


class CriticOutput(BaseModel):
    approved: bool
    issues: list[str] = Field(default_factory=list)
    final_message: str


class AgentTrace(BaseModel):
    planner: PlannerOutput
    plan_critic: PlanCriticOutput
    deterministic_decision: UtilityDecision
    utility_agent: UtilityAgentOutput
    critic: CriticOutput
    plan_attempt: int = 1
    executed_steps: list[PlanStepExecution] = Field(default_factory=list)


class AgentResponse(BaseModel):
    success: bool
    message: str
    recommended_slot_id: str | None = None
    requires_confirmation: bool = True
    trace: AgentTrace | None = None


class ToolResult(BaseModel):
    success: bool
    data: Any = None
    error: str | None = None