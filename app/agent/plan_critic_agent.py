from __future__ import annotations

import json

from app.llm.client import StructuredLLM
from app.schemas.models import PlanCriticOutput, PlannerOutput


class PlanCriticAgent:
    SYSTEM_PROMPT = """You are a cautious healthcare workflow plan critic.
Review the proposed business plan before any action or tool is executed.

Verify that:
- the plan addresses the user's stated goal;
- the plan uses only the available business actions;
- the workflow is complete enough to produce an appointment recommendation;
- steps are ordered through valid dependencies;
- no consumer runs before the step that produces its input;
- hard constraints and soft preferences are interpreted correctly;
- the plan does not include booking, critic review, confirmation, or invented capabilities.

The runtime may supply deterministic static-validation issues. Any such issue is blocking and
must be included in your response. Do not execute the plan and do not evaluate appointment results,
because none exist yet. Return structured output only."""

    def __init__(self, llm: StructuredLLM) -> None:
        self.llm = llm

    def review(
        self,
        *,
        user_message: str,
        plan: PlannerOutput,
        static_issues: list[str],
    ) -> PlanCriticOutput:
        payload = {
            "original_user_request": user_message,
            "proposed_plan": plan.model_dump(mode="json"),
            "static_validation_issues": static_issues,
        }
        result = self.llm.parse(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, indent=2),
            response_model=PlanCriticOutput,
        )

        # Deterministic structural checks cannot be overridden by an LLM approval.
        if static_issues:
            combined = list(dict.fromkeys([*static_issues, *result.issues]))
            return PlanCriticOutput(
                approved=False,
                issues=combined,
                summary="Plan rejected before execution because it is not executable.",
            )
        return result
