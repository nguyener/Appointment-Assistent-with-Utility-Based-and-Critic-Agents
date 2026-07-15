from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.episode import AgentExecutionEpisode


class EpisodicMemoryRepository(ABC):
    """Persistence contract for agent execution episodes."""

    @abstractmethod
    def save(self, episode: AgentExecutionEpisode) -> None:
        """Persist one execution episode."""

    @abstractmethod
    def get(self, episode_id: str) -> AgentExecutionEpisode | None:
        """Return an episode by ID, or None when it does not exist."""

    @abstractmethod
    def list_recent(self, limit: int = 20) -> list[AgentExecutionEpisode]:
        """Return the most recently recorded episodes."""
