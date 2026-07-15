from __future__ import annotations

import json

from app.llm.client import StructuredLLM
from app.schemas.models import CriticOutput, PlannerOutput, UtilityAgentOutput, UtilityDecision


class CriticAgent:
    SYSTEM_PROMPT = """You are a cautious healthcare recommendation critic.
Verify that the recommended slot exists in the supplied valid candidates, matches the
highest deterministic utility score, and that the response asks for human confirmation.

Treat hard constraints and soft preferences differently:
- Hard constraints are the constraints listed in deterministic_decision.constraints_applied.
  Reject a recommendation that violates one of them.
- Preferred provider and preferred time of day are soft preferences used for utility
  scoring. Do NOT reject the best valid candidate merely because it misses one of these
  preferences. Instead, approve it, clearly disclose each mismatch in final_message, and
  ask the user whether they want to accept the best available alternative.

Do not add medical advice and do not book anything. If a real validation check fails,
reject the recommendation and list concrete issues. Return structured output only."""

    def __init__(self, llm: StructuredLLM) -> None:
        self.llm = llm

    def review(
        self,
        plan: PlannerOutput,
        decision: UtilityDecision,
        recommendation: UtilityAgentOutput,
    ) -> CriticOutput:
        payload = {
            "plan": plan.model_dump(mode="json"),
            "deterministic_decision": decision.model_dump(mode="json"),
            "utility_recommendation": recommendation.model_dump(mode="json"),
            "requires_confirmation": True,
        }
        return self.llm.parse(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, indent=2),
            response_model=CriticOutput,
        )
