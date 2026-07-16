from pathlib import Path

from app.memory.sqlite_repository import SQLiteEpisodicMemoryRepository
from app.schemas.episode import AgentExecutionEpisode, EpisodeOutcome
from app.schemas.models import AgentResponse


def create_episode(
    user_message: str = "Find me a primary care appointment",
) -> AgentExecutionEpisode:
    return AgentExecutionEpisode(
        user_message=user_message,
        outcome=EpisodeOutcome.RECOMMENDATION_READY,
        response=AgentResponse(
            success=True,
            message="Appointment recommendation prepared.",
            recommended_slot_id="slot-123",
            requires_confirmation=True,
        ),
        duration_ms=125,
    )


def test_save_and_get_episode(tmp_path: Path) -> None:
    repository = SQLiteEpisodicMemoryRepository(
        tmp_path / "episodic_memory.db"
    )
    episode = create_episode()

    repository.save(episode)

    loaded = repository.get(str(episode.episode_id))

    assert loaded is not None
    assert loaded.episode_id == episode.episode_id
    assert loaded.user_message == episode.user_message
    assert loaded.outcome == EpisodeOutcome.RECOMMENDATION_READY
    assert loaded.response.success is True
    assert loaded.response.recommended_slot_id == "slot-123"
    assert loaded.duration_ms == 125


def test_get_returns_none_for_unknown_episode(tmp_path: Path) -> None:
    repository = SQLiteEpisodicMemoryRepository(
        tmp_path / "episodic_memory.db"
    )

    loaded = repository.get("unknown-episode")

    assert loaded is None


def test_list_recent_returns_newest_first(tmp_path: Path) -> None:
    repository = SQLiteEpisodicMemoryRepository(
        tmp_path / "episodic_memory.db"
    )

    first = create_episode("First request")
    second = create_episode("Second request")

    repository.save(first)
    repository.save(second)

    episodes = repository.list_recent(limit=10)

    assert len(episodes) == 2
    assert episodes[0].episode_id == second.episode_id
    assert episodes[1].episode_id == first.episode_id


def test_list_recent_returns_empty_for_non_positive_limit(
    tmp_path: Path,
) -> None:
    repository = SQLiteEpisodicMemoryRepository(
        tmp_path / "episodic_memory.db"
    )
    repository.save(create_episode())

    assert repository.list_recent(limit=0) == []
    assert repository.list_recent(limit=-1) == []
