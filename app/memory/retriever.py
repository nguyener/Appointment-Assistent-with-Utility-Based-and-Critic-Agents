from __future__ import annotations

import re
from dataclasses import dataclass

from app.memory.repository import EpisodicMemoryRepository
from app.schemas.episode import AgentExecutionEpisode, EpisodeOutcome


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "at",
    "book",
    "can",
    "could",
    "find",
    "for",
    "i",
    "in",
    "is",
    "me",
    "my",
    "of",
    "on",
    "please",
    "the",
    "to",
    "want",
    "with",
    "would",
}

# we don't want the planner to from not useful outcomes like 
# EpisodeOutcome.EXECUTION_FAILED
# EpisodeOutcome.PLAN_REJECTED
# EpisodeOutcome.MISSING_INFORMATION
_USEFUL_OUTCOMES = {
    EpisodeOutcome.RECOMMENDATION_READY,
}


class EpisodicMemoryRetriever:

    def __init__(
        self,
        repository: EpisodicMemoryRepository,
        *,
        candidate_limit: int = 100,
        minimum_score: float = 0.10,
    ) -> None:
        if candidate_limit <= 0:
            raise ValueError(
                "candidate_limit must be greater than zero"
            )

        if not 0 <= minimum_score <= 1:
            raise ValueError(
                "minimum_score must be between zero and one"
            )

        self.repository = repository
        self.candidate_limit = candidate_limit
        self.minimum_score = minimum_score
    
    def retrieve(
        self,
        user_message: str,
        *,
        limit: int = 3,
    ) -> list[AgentExecutionEpisode]:

        if limit <= 0:
            return []

        query_terms = self._terms(user_message)

        if not query_terms:
            return []

        matches: list[tuple[float, AgentExecutionEpisode]] = []

        recent_episodes = self.repository.list_recent(
            limit=self.candidate_limit
        )

        for episode in recent_episodes:
            if episode.outcome not in _USEFUL_OUTCOMES:
                continue

            episode_terms = self._episode_terms(episode)

            shared_terms = query_terms & episode_terms

            if not shared_terms:
                continue

            score = (
                2 * len(shared_terms)
            ) / (
                len(query_terms) + len(episode_terms)
            )

            if score < self.minimum_score:
                continue

            matches.append(
                (
                    score,
                    episode,
                )  
            )

        matches.sort(
            key=lambda item: (
                item[0],
                item[1].created_at,
            ),
            reverse=True,
        )
        
        return [episode for score, episode in matches[:limit]]
    
    @classmethod
    def _episode_terms(
        cls,
        episode: AgentExecutionEpisode,
    ) -> set[str]:

        values = [
            episode.user_message,
        ]

        trace = episode.response.trace

        if trace is not None:
            preference_values = (
                trace.planner
                .preferences
                .model_dump(exclude_none=True)
                .values()
            )

            values.extend(
                str(value)
                for value in preference_values
            )

        return cls._terms(" ".join(values))
    
    @staticmethod
    def _terms(text: str) -> set[str]:
        return {
            token
            for token in _TOKEN_PATTERN.findall(
                text.lower()
            )
            if len(token) > 1
            and token not in _STOP_WORDS
        }
    

   
    def format_episodes_for_planner(self,
        episodes: list[AgentExecutionEpisode],
    ) -> str:

        if not episodes:
            return (
                "No relevant previous successful experiences "
                "were found. Create the plan from the current "
                "request only."
            )

        formatted_episodes: list[str] = []

        for episode in episodes:
            trace = episode.response.trace
            if trace is None:
                continue
            preferences = trace.planner.preferences.model_dump(exclude={"weights"})
            preferences = {
                name:value
                for name, value in preferences.items() if value not in ("", [], {})
            }

            workflow = [
                step.action.value
                for step in trace.planner.steps
            ]

            preference_lines = [
                f"- {name}: {value}"
                for name, value in preferences.items()
            ]

            workflow_lines = [
                f"{index}. {action}"
                for index, action in enumerate(
                    workflow,
                    start=1,
                )
            ]
            
            formatted_episodes.append(
                "\n".join(
                    [
                        "Previous successful execution:",
                        "",
                        "User request:",
                        episode.user_message,
                        "",
                        "Extracted preferences:",
                        *preference_lines,
                        "",
                        "Successful workflow:",
                        *workflow_lines,
                    ]
                )
            )
    
        return "\n\n---\n\n".join(formatted_episodes)