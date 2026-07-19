from __future__ import annotations

import json

from app.llm.client import StructuredLLM
from app.schemas.episode import (
    AgentExecutionEpisode,
    EpisodeReflection,
)


class ReflectionAgent:
    """Extract reusable lessons from a completed agent execution."""

    SYSTEM_PROMPT = """
You are an episodic-memory reflection agent for a healthcare
appointment assistant.

Analyze one completed execution and extract concise, reusable
experience that may improve future planning.

Your job is not to repeat the full execution trace.

Identify:
- the user's goal
- important constraints
- what worked
- what failed
- reusable lessons
- an appropriate action for a similar future request

Rules:
- Use only evidence contained in the supplied episode.
- Do not invent patient facts.
- Do not invent appointment availability.
- Do not assume that a previously available appointment remains available.
- Do not treat a one-time preference as a permanent user preference.
- Do not claim that a provider should always be selected.
- Distinguish successful behavior from failed behavior.
- Keep every field concise.
- Return structured output only.
""".strip()

    def __init__(self, llm: StructuredLLM) -> None:
        self.llm = llm

    def reflect(
        self,
        episode: AgentExecutionEpisode,
    ) -> EpisodeReflection:
        payload = {
            "user_message": episode.user_message,
            "outcome": episode.outcome.value,
            "response": episode.response.model_dump(mode="json"),
            "duration_ms": episode.duration_ms,
        }

        return self.llm.parse(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, indent=2),
            response_model=EpisodeReflection,
        )