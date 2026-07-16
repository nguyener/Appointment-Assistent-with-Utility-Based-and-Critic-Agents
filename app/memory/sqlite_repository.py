from __future__ import annotations

import sqlite3
from pathlib import Path

from app.memory.repository import EpisodicMemoryRepository
from app.schemas.episode import AgentExecutionEpisode


class SQLiteEpisodicMemoryRepository(EpisodicMemoryRepository):
    """SQLite-backed storage for agent execution episodes."""

    def __init__(self, database_path: str | Path = "data/episodic_memory.db") -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def save(self, episode: AgentExecutionEpisode) -> None:
        serialized_episode = episode.model_dump_json()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_execution_episodes (
                    episode_id,
                    created_at,
                    user_message,
                    outcome,
                    success,
                    duration_ms,
                    episode_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(episode.episode_id),
                    episode.created_at.isoformat(),
                    episode.user_message,
                    episode.outcome.value,
                    int(episode.response.success),
                    episode.duration_ms,
                    serialized_episode,
                ),
            )

    def get(self, episode_id: str) -> AgentExecutionEpisode | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT episode_json
                FROM agent_execution_episodes
                WHERE episode_id = ?
                """,
                (episode_id,),
            ).fetchone()

        if row is None:
            return None

        return AgentExecutionEpisode.model_validate_json(row["episode_json"])

    def list_recent(self, limit: int = 20) -> list[AgentExecutionEpisode]:
        if limit <= 0:
            return []

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT episode_json
                FROM agent_execution_episodes
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            AgentExecutionEpisode.model_validate_json(row["episode_json"])
            for row in rows
        ]

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_execution_episodes (
                    episode_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    episode_json TEXT NOT NULL
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_execution_episodes_created_at
                ON agent_execution_episodes(created_at DESC)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_execution_episodes_outcome
                ON agent_execution_episodes(outcome)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection
