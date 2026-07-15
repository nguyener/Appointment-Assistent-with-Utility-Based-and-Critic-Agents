from __future__ import annotations

from app.agent.orchestrator import UtilityHealthcareOrchestrator
from app.schemas.models import (
    CriticOutput,
    PatientPreferences,
    PlanCriticOutput,
    PlanStep,
    PlannerOutput,
    RankedCandidate,
    UtilityAgentOutput,
)


class FakeLLM:
    def parse(self, *, system_prompt: str, user_prompt: str, response_model):
        if response_model is PlannerOutput:
            return PlannerOutput(
                goal="Find the best primary care appointment",
                preferences=PatientPreferences(
                    preferred_provider="Lee",
                    preferred_time_of_day="evening",
                    max_distance_miles=15,
                ),
                steps=[
                    PlanStep(step_id="search", action="find_appointments", purpose="Get candidates", depends_on=[]),
                    PlanStep(step_id="score", action="evaluate_options", purpose="Score candidates", depends_on=["search"]),
                    PlanStep(step_id="rank", action="rank_options", purpose="Explain the deterministic ranking", depends_on=["score"]),
                ],
            )
        if response_model is PlanCriticOutput:
            return PlanCriticOutput(
                approved=True,
                issues=[],
                summary="Plan is complete and executable.",
            )
        if response_model is UtilityAgentOutput:
            import json
            payload = json.loads(user_prompt)
            candidates = sorted(payload["candidates"], key=lambda c: c["utility_score"], reverse=True)
            return UtilityAgentOutput(
                recommended_slot_id=candidates[0]["slot"]["slot_id"],
                ranked_candidates=[
                    RankedCandidate(
                        slot_id=c["slot"]["slot_id"],
                        rank=i + 1,
                        reasoning="Ranked by deterministic utility score",
                    )
                    for i, c in enumerate(candidates)
                ],
                recommendation_summary="Selected the highest deterministic utility score.",
            )
        if response_model is CriticOutput:
            return CriticOutput(
                approved=True,
                issues=[],
                final_message="I recommend the highest-utility slot. Please confirm before I book it.",
            )
        raise AssertionError(f"Unexpected response model: {response_model}")


def test_multi_agent_orchestrator_calls_planner_utility_and_critic():
    agent = UtilityHealthcareOrchestrator(llm=FakeLLM())
    response = agent.handle("Find me an evening appointment with Dr. Lee within 15 miles")

    assert response.success is True
    assert response.recommended_slot_id == response.trace.deterministic_decision.selected.slot.slot_id
    assert response.requires_confirmation is True
    assert response.trace is not None
    assert response.trace.plan_critic.approved is True
    assert response.trace.critic.approved is True


class RejectOnceThenApproveLLM(FakeLLM):
    def __init__(self):
        self.planner_calls = 0
        self.critic_calls = 0

    def parse(self, *, system_prompt: str, user_prompt: str, response_model):
        if response_model is PlannerOutput:
            self.planner_calls += 1
            return super().parse(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=response_model,
            )
        if response_model is CriticOutput:
            self.critic_calls += 1
            if self.critic_calls == 1:
                return CriticOutput(
                    approved=False,
                    issues=["The recommendation explanation does not clearly request confirmation."],
                    final_message="Rejected.",
                )
        return super().parse(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
        )


def test_replans_when_critic_rejects_then_succeeds():
    llm = RejectOnceThenApproveLLM()
    agent = UtilityHealthcareOrchestrator(llm=llm, max_plan_retries=2)

    response = agent.handle("Find me an evening appointment with Dr. Lee within 15 miles")

    assert response.success is True
    assert llm.planner_calls == 2
    assert llm.critic_calls == 2
    assert response.trace is not None
    assert response.trace.plan_attempt == 2


class AlwaysRejectLLM(FakeLLM):
    def __init__(self):
        self.planner_calls = 0
        self.critic_calls = 0

    def parse(self, *, system_prompt: str, user_prompt: str, response_model):
        if response_model is PlannerOutput:
            self.planner_calls += 1
            return super().parse(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=response_model,
            )
        if response_model is CriticOutput:
            self.critic_calls += 1
            return CriticOutput(
                approved=False,
                issues=["Still not acceptable."],
                final_message="Rejected.",
            )
        return super().parse(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
        )


def test_stops_after_configured_plan_retries():
    llm = AlwaysRejectLLM()
    agent = UtilityHealthcareOrchestrator(llm=llm, max_plan_retries=2)

    response = agent.handle("Find me an appointment")

    assert response.success is False
    assert "after 3 attempts" in response.message
    assert llm.planner_calls == 3
    assert llm.critic_calls == 3

class RejectSoftPreferenceMismatchLLM(FakeLLM):
    def __init__(self):
        self.planner_calls = 0
        self.critic_calls = 0

    def parse(self, *, system_prompt: str, user_prompt: str, response_model):
        if response_model is PlannerOutput:
            self.planner_calls += 1
            return super().parse(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=response_model,
            )
        if response_model is CriticOutput:
            self.critic_calls += 1
            return CriticOutput(
                approved=False,
                issues=["The selected slot is not in the preferred evening time."],
                final_message="Rejected.",
            )
        return super().parse(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
        )


def test_returns_best_available_slot_when_only_soft_preference_is_unmet():
    llm = RejectSoftPreferenceMismatchLLM()
    agent = UtilityHealthcareOrchestrator(llm=llm, max_plan_retries=2)

    response = agent.handle(
        "Find me a primary-care appointment with Dr. Lee, preferably in the evening and within 15 miles"
    )

    assert response.success is True
    assert response.recommended_slot_id == "S1"
    assert response.requires_confirmation is True
    assert "could not find" in response.message.lower()
    assert "evening" in response.message.lower()
    assert "would you like me to book" in response.message.lower()
    assert llm.planner_calls == 1
    assert llm.critic_calls == 1


class HyphenatedSpecialtyLLM(FakeLLM):
    def parse(self, *, system_prompt: str, user_prompt: str, response_model):
        result = super().parse(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
        )
        if response_model is PlannerOutput:
            result.preferences.preferred_specialty = "primary-care"
        return result


def test_hyphenated_primary_care_specialty_still_returns_candidates():
    agent = UtilityHealthcareOrchestrator(llm=HyphenatedSpecialtyLLM())

    response = agent.handle(
        "Find me a primary-care appointment with Dr. Lee, preferably in the evening and within 15 miles"
    )

    assert response.success is True
    assert response.recommended_slot_id is not None
    assert response.trace is not None
    assert response.trace.deterministic_decision.selected is not None

class InvalidDependencyPlanLLM(FakeLLM):
    def parse(self, *, system_prompt: str, user_prompt: str, response_model):
        result = super().parse(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
        )
        if response_model is PlannerOutput:
            result.steps[1].depends_on = ["missing-step"]
        return result


def test_rejects_non_executable_planner_dependencies():
    agent = UtilityHealthcareOrchestrator(llm=InvalidDependencyPlanLLM())

    response = agent.handle("Find me a primary care appointment")

    assert response.success is False
    assert "Plan failed validation" in response.message
    assert "unknown step" in response.message


def test_trace_records_execution_of_planner_generated_steps():
    agent = UtilityHealthcareOrchestrator(llm=FakeLLM())

    response = agent.handle("Find me a primary care appointment")

    assert response.success is True
    assert response.trace is not None
    assert [step.step_id for step in response.trace.executed_steps] == [
        "platform_plan_critic_review", "search", "score", "rank",
        "platform_recommendation_critic_review", "platform_request_confirmation"
    ]
    assert [step.source for step in response.trace.executed_steps] == [
        "platform", "planner", "planner", "planner", "platform", "platform"
    ]
    assert all(step.status == "completed" for step in response.trace.executed_steps)


def test_platform_guardrails_are_not_planner_actions():
    agent = UtilityHealthcareOrchestrator(llm=FakeLLM())
    response = agent.handle("Find me a primary care appointment")

    assert response.success is True
    assert response.trace is not None
    planned_actions = {step.action.value for step in response.trace.planner.steps}
    assert "critic_review" not in planned_actions
    assert "request_confirmation" not in planned_actions
    assert response.trace.executed_steps[0].action == "plan_critic_review"
    assert response.trace.executed_steps[0].source == "platform"
    assert response.trace.executed_steps[-2].action == "recommendation_critic_review"
    assert response.trace.executed_steps[-2].source == "platform"
    assert response.trace.executed_steps[-1].action == "request_confirmation"
    assert response.trace.executed_steps[-1].source == "platform"


class RejectPlanOnceThenApproveLLM(FakeLLM):
    def __init__(self):
        self.planner_calls = 0
        self.plan_critic_calls = 0
        self.utility_calls = 0

    def parse(self, *, system_prompt: str, user_prompt: str, response_model):
        if response_model is PlannerOutput:
            self.planner_calls += 1
        elif response_model is PlanCriticOutput:
            self.plan_critic_calls += 1
            if self.plan_critic_calls == 1:
                return PlanCriticOutput(
                    approved=False,
                    issues=["rank_options does not depend on evaluate_options."],
                    summary="Plan must be revised before execution.",
                )
        elif response_model is UtilityAgentOutput:
            self.utility_calls += 1
        return super().parse(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
        )


def test_plan_critic_rejects_before_any_business_step_executes_then_replans():
    llm = RejectPlanOnceThenApproveLLM()
    agent = UtilityHealthcareOrchestrator(llm=llm, max_plan_retries=2)

    response = agent.handle("Find me a primary care appointment")

    assert response.success is True
    assert llm.planner_calls == 2
    assert llm.plan_critic_calls == 2
    # The rejected first plan produced no utility recommendation.
    assert llm.utility_calls == 1
    assert response.trace is not None
    assert response.trace.plan_attempt == 2
    assert response.trace.plan_critic.approved is True


class AlwaysRejectPlanLLM(FakeLLM):
    def __init__(self):
        self.planner_calls = 0
        self.plan_critic_calls = 0
        self.utility_calls = 0
        self.recommendation_critic_calls = 0

    def parse(self, *, system_prompt: str, user_prompt: str, response_model):
        if response_model is PlannerOutput:
            self.planner_calls += 1
        elif response_model is PlanCriticOutput:
            self.plan_critic_calls += 1
            return PlanCriticOutput(
                approved=False,
                issues=["The workflow is incomplete."],
                summary="Rejected before execution.",
            )
        elif response_model is UtilityAgentOutput:
            self.utility_calls += 1
        elif response_model is CriticOutput:
            self.recommendation_critic_calls += 1
        return super().parse(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
        )


def test_plan_critic_failure_never_executes_or_calls_recommendation_critic():
    llm = AlwaysRejectPlanLLM()
    agent = UtilityHealthcareOrchestrator(llm=llm, max_plan_retries=2)

    response = agent.handle("Find me a primary care appointment")

    assert response.success is False
    assert "Plan failed validation after 3 attempts" in response.message
    assert llm.planner_calls == 3
    assert llm.plan_critic_calls == 3
    assert llm.utility_calls == 0
    assert llm.recommendation_critic_calls == 0
