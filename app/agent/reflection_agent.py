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

Analyze one completed execution and produce a concise reflection
about that specific execution.

The reflection should help a future planner understand what happened,
but it must remain tied to this individual episode.

Extract:
- the user's goal in this episode
- constraints present in this episode
- actions that succeeded in this episode
- actions that failed or caused difficulty in this episode
- observations supported by this episode
- considerations that may be worth checking in a similar future case

Rules:
- Use only evidence contained in the supplied episode.
- Do not invent patient facts.
- Do not invent appointment availability.
- Do not create universal workflow rules from one episode.
- Do not use words such as "always", "never", or "must" unless they
  describe an explicit system rule visible in the episode.
- Do not assume that a previously available provider or appointment
  remains available.
- Do not treat a one-time user preference as a permanent preference.
- Do not claim that one provider should generally be preferred.
- Phrase observations as facts about this execution.
- Phrase future considerations as possibilities to check, not rules.
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