from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from app.config import settings

T = TypeVar("T", bound=BaseModel)


class StructuredLLM(Protocol):
    def parse(self, *, system_prompt: str, user_prompt: str, response_model: type[T]) -> T:
        ...


class OpenAIStructuredLLM:
    """Small adapter that makes every model request visible and testable."""

    def __init__(self, client: Any | None = None, model: str | None = None) -> None:
        if client is None:
            if not settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is required to run the LLM-powered agent")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc
            client = OpenAI(api_key=settings.openai_api_key)
        self.client = client
        self.model = model or settings.openai_model

    def parse(self, *, system_prompt: str, user_prompt: str, response_model: type[T]) -> T:
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=response_model,
        )
        if response.output_parsed is None:
            raise RuntimeError("Model returned no structured output")
        return response.output_parsed
