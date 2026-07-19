from pathlib import Path

from app.memory.retriever import EpisodicMemoryRetriever
from app.memory.sqlite_repository import SQLiteEpisodicMemoryRepository
from app.schemas.episode import (
    AgentExecutionEpisode,
    EpisodeOutcome,
)
from app.schemas.models import AgentResponse


def create_episode(
    user_message: str,
    outcome: EpisodeOutcome = EpisodeOutcome.RECOMMENDATION_READY,
) -> AgentExecutionEpisode:
    
    successful = (
        outcome == EpisodeOutcome.RECOMMENDATION_READY
    )

    return AgentExecutionEpisode(
        user_message=user_message,
        outcome=outcome,
        response=AgentResponse(
            success=successful,
            message="I found an appointment recommendation.",
            recommended_slot_id="slot-123",
            requires_confirmation=True,
        ),
        duration_ms=100,
    )


def test_retrieve_similar_episode(tmp_path: Path) -> None:
    repository = SQLiteEpisodicMemoryRepository(
        tmp_path / "episodic-memory.db"
    )

    repository.save(
        create_episode(
            "Find a dermatologist appointment in Bellevue"
        )
    )

    repository.save(
        create_episode(
            "Find a primary care doctor in Seattle"
        )
    )

    retriever = EpisodicMemoryRetriever(repository)

    results = retriever.retrieve(
        "I need a Bellevue dermatologist",
        limit=3,
    )

    assert len(results) == 1

    assert results[0].user_message == (
        "Find a dermatologist appointment in Bellevue"
    )

def test_retrieve_ignores_failed_episode(
    tmp_path: Path,
) -> None:
    repository = SQLiteEpisodicMemoryRepository(
        tmp_path / "episodic-memory.db"
    )

    failed_episode = create_episode(
        "Find a dermatologist in Bellevue",
        outcome=EpisodeOutcome.EXECUTION_FAILED,
    )

    repository.save(failed_episode)

    retriever = EpisodicMemoryRetriever(repository)

    results = retriever.retrieve(
        "I need a Bellevue dermatologist"
    )

    assert results == []


def test_retrieve_orders_by_similarity(
    tmp_path: Path,
) -> None:
    repository = SQLiteEpisodicMemoryRepository(
        tmp_path / "episodic-memory.db"
    )

    repository.save(
        create_episode(
            "Find a dermatologist in Bellevue"
        )
    )

    repository.save(
        create_episode(
            "Find a doctor in Bellevue"
        )
    )

    retriever = EpisodicMemoryRetriever(repository)

    results = retriever.retrieve(
        "Find a dermatologist in Bellevue",
        limit=2,
    )

    assert len(results) == 2

    assert results[0].user_message == (
        "Find a dermatologist in Bellevue"
    )

    assert results[1].user_message == (
        "Find a doctor in Bellevue"
    )

def test_retrieve_respects_limit(
    tmp_path: Path,
) -> None:
    repository = SQLiteEpisodicMemoryRepository(
        tmp_path / "episodic-memory.db"
    )

    repository.save(
        create_episode(
            "Find a dermatologist in Bellevue tomorrow"
        )
    )

    repository.save(
        create_episode(
            "Find a Bellevue dermatologist next week"
        )
    )

    repository.save(
        create_episode(
            "Bellevue dermatologist morning appointment"
        )
    )

    retriever = EpisodicMemoryRetriever(repository)

    results = retriever.retrieve(
        "Find a dermatologist in Bellevue",
        limit=2,
    )

    assert len(results) == 2
