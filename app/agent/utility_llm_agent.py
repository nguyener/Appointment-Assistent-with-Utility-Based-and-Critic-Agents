from __future__ import annotations

import json

from app.llm.client import StructuredLLM
from app.schemas.models import PatientPreferences, UtilityAgentOutput, UtilityDecision


class LLMUtilityAgent:
    SYSTEM_PROMPT = """You are a utility-based healthcare recommendation agent.
You receive appointment candidates that have already passed hard constraints and have
application-calculated utility scores. Rank only the supplied candidates.
The highest deterministic utility score must be the recommendation unless two scores
are tied. Never invent a slot, change a score, or book an appointment.
Explain the tradeoffs in plain language. Return structured output only."""

    def __init__(self, llm: StructuredLLM) -> None:
        self.llm = llm

    def rank(self, preferences: PatientPreferences, decision: UtilityDecision) -> UtilityAgentOutput:
        candidates = []
        if decision.selected:
            candidates.append(decision.selected)
        candidates.extend(decision.alternatives)
        payload = {
            "preferences": preferences.model_dump(mode="json"),
            "constraints_applied": decision.constraints_applied,
            "candidates": [c.model_dump(mode="json") for c in candidates],
        }
        return self.llm.parse(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, indent=2),
            response_model=UtilityAgentOutput,
        )
