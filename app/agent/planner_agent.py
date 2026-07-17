from __future__ import annotations

import json

from app.llm.client import StructuredLLM
from app.schemas.models import PlannerOutput, UtilityAgentOutput, UtilityDecision


class PlannerAgent:
    SYSTEM_PROMPT = """You are a healthcare appointment business-planning agent.

Create an executable business plan rather than merely extracting preferences.

Goal:

Generate the smallest valid business workflow needed to find and recommend an appointment.

Available business actions:

- find_appointments: retrieve appointment candidates for the requested specialty.
- evaluate_options: apply hard constraints and calculate deterministic utility scores.
- rank_options: explain and rank only the scored candidates; never invent a slot.

Planning rules:

- Use only the available business actions; the runtime maps them to registered handlers.
- Give every step a unique step_id and declare depends_on step IDs.
- Dependencies must form an acyclic executable graph.
- A consumer step must depend on the step that produces its input.
- Do not include critic_review, request_confirmation, booking, or other platform controls.
  Critic validation and human confirmation are mandatory guardrails enforced outside the plan.
- Distinguish hard constraints from soft wording such as "preferably".
- Extract only preferences actually stated or safely defaulted by the schema.
- Normalize specialty wording: "primary-care", "primary care doctor", and "PCP" become "primary care".
- Do not invent patient facts, appointment slots, or booking results.
- Do not book anything.
- Relevant past executions may be included as episodic-memory examples.
- Treat past executions only as reference examples; the current user request is always the source of truth.
- Do not copy patient-specific facts, dates, locations, providers, appointment slots, or preferences
  from memory unless they are also stated in the current request.
- Do not assume that a past plan is correct for the current request.

Return structured output only."""

    REVISION_SYSTEM_PROMPT = """You are revising a rejected healthcare appointment business plan.
Use the critic's concrete issues and any available execution evidence to produce a complete replacement plan.
Preserve the original goal and stated preferences. Correct ordering, dependencies, missing business
steps, or misunderstood constraints. Do not merely rewrite wording when the workflow is wrong.
Use only: find_appointments, evaluate_options, rank_options.
Do not add critic_review, request_confirmation, or booking; those are platform guardrails.
Return structured output only."""

    def __init__(self, llm: StructuredLLM) -> None:
        self.llm = llm

    def create_plan(self, 
                    user_message: str,
                    *,
                    memory_context: str | None = None,
                    ) -> PlannerOutput:
        user_prompt = self._build_user_prompt(
            user_message=user_message,
            memory_context=memory_context,
        )


        return self.llm.parse(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=PlannerOutput,
        )

    def revise_plan(
        self,
        *,
        user_message: str,
        previous_plan: PlannerOutput,
        critic_issues: list[str],
        decision: UtilityDecision | None = None,
        recommendation: UtilityAgentOutput | None = None,
        rejection_stage: str = "recommendation",
    ) -> PlannerOutput:
        payload = {
            "original_user_request": user_message,
            "previous_plan": previous_plan.model_dump(mode="json"),
            "rejection_stage": rejection_stage,
            "deterministic_decision": (
                decision.model_dump(mode="json") if decision is not None else None
            ),
            "rejected_recommendation": (
                recommendation.model_dump(mode="json")
                if recommendation is not None
                else None
            ),
            "critic_issues": critic_issues,
            "instruction": "Return a corrected executable business plan with explicit dependencies.",
        }
        return self.llm.parse(
            system_prompt=self.REVISION_SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, indent=2),
            response_model=PlannerOutput,
        )
    
    @staticmethod
    def _build_user_prompt(
        *,
        user_message: str,
        memory_context: str | None = None,
    ) -> str:
        sections = [
            "Current user request:",
            user_message,
        ]

        if memory_context:
            sections.extend(
                [
                    "",
                    "Relevant successful past executions:",
                    memory_context,
                ]
            )

        return "\n".join(sections)
