from __future__ import annotations

from time import perf_counter
from collections.abc import Callable
from typing import Any

from app.agent.critic_agent import CriticAgent
from app.agent.plan_critic_agent import PlanCriticAgent
from app.agent.planner_agent import PlannerAgent
from app.agent.reflection_agent import ReflectionAgent
from app.agent.utility import UtilityEvaluator
from app.agent.utility_llm_agent import LLMUtilityAgent
from app.llm.client import OpenAIStructuredLLM, StructuredLLM
from app.schemas.models import (
    AgentResponse,
    AgentTrace,
    AppointmentSlot,
    PlanAction,
    PlanCriticOutput,
    PlanStep,
    PlanStepExecution,
    PlannerOutput,
)
from app.tools.appointment_tools import find_appointments, schedule_appointment
from app.memory.repository import EpisodicMemoryRepository
from app.memory.sqlite_repository import SQLiteEpisodicMemoryRepository
from app.memory.retriever import EpisodicMemoryRetriever
from app.schemas.episode import AgentExecutionEpisode, EpisodeOutcome

ExecutionContext = dict[str, Any]
ActionHandler = Callable[[PlanStep, PlannerOutput, ExecutionContext], PlanStepExecution]


class UtilityHealthcareOrchestrator:
    """Execute planner-selected business actions, then enforce platform guardrails."""

    def __init__(
        self,
        llm: StructuredLLM | None = None,
        max_plan_retries: int = 2,
        episodic_memory: EpisodicMemoryRepository | None = None,
    ) -> None:
        if max_plan_retries < 0:
            raise ValueError("max_plan_retries must be zero or greater")

        shared_llm = llm or OpenAIStructuredLLM()
        self.planner = PlannerAgent(shared_llm)
        self.utility_agent = LLMUtilityAgent(shared_llm)
        self.plan_critic = PlanCriticAgent(shared_llm)
        self.critic = CriticAgent(shared_llm)
        self.reflection_agent = ReflectionAgent(shared_llm)
        self.evaluator = UtilityEvaluator()
        self.max_plan_retries = max_plan_retries
        self.episodic_memory = episodic_memory or SQLiteEpisodicMemoryRepository()  
        self.memory_retriever = EpisodicMemoryRetriever(self.episodic_memory)  
        # The registry maps the planner's symbolic business actions to real code.
        # Adding an action requires registering a handler, not expanding a workflow if/elif chain.
        self.action_handlers: dict[PlanAction, ActionHandler] = {
            PlanAction.FIND_APPOINTMENTS: self._execute_find_appointments,
            PlanAction.EVALUATE_OPTIONS: self._execute_evaluate_options,
            PlanAction.RANK_OPTIONS: self._execute_rank_options,
        }

    def _handle_internal(self, user_message: str, include_trace: bool = True) -> AgentResponse:
        
        # Retrieve relevant past episodes from memory and format them for the planner.
        episodes = self.memory_retriever.retrieve(
            user_message,
            limit=3,
        )
       
        memory_context = self.memory_retriever.format_episodes_for_planner(episodes)

        print(
            "Retrieved reflected memory_context:", memory_context or None
        )
        
        plan = self.planner.create_plan(
            user_message,
            memory_context=memory_context or None
        )

        #print("Planner preferences:", plan.preferences.model_dump())
        last_issues: list[str] = []

        # One initial plan plus the configured number of revisions. A rejection can
        # happen either before execution (plan critic) or after execution
        # (recommendation critic).
        for attempt in range(self.max_plan_retries + 1):
            if plan.missing_required_information:
                return AgentResponse(
                    success=False,
                    message="Please provide: " + ", ".join(plan.missing_required_information),
                    requires_confirmation=False,
                )

            # Phase 1: review the proposed workflow before any action executes.
            static_plan_issues = self._validate_executable_plan(plan)
            plan_critic = self.plan_critic.review(
                user_message=user_message,
                plan=plan,
                static_issues=static_plan_issues,
            )
            if not plan_critic.approved:
                last_issues = list(plan_critic.issues)
                if attempt < self.max_plan_retries:
                    plan = self.planner.revise_plan(
                        user_message=user_message,
                        previous_plan=plan,
                        critic_issues=last_issues,
                        rejection_stage="plan",
                    )
                    continue
                return AgentResponse(
                    success=False,
                    message=(
                        f"Plan failed validation after {self.max_plan_retries + 1} attempts: "
                        + "; ".join(last_issues)
                    ),
                    requires_confirmation=False,
                )

            # Phase 2: only an approved plan may execute.
            context: ExecutionContext = {}
            executed_steps: list[PlanStepExecution] = [
                PlanStepExecution(
                    step_id="platform_plan_critic_review",
                    action="plan_critic_review",
                    status="completed",
                    detail="Approved the business plan before execution.",
                    source="platform",
                )
            ]

            try:
                for step in self._ordered_steps(plan):
                    handler = self.action_handlers.get(step.action)
                    if handler is None:
                        raise RuntimeError(f"Unsupported plan action: {step.action.value}")
                    executed_steps.append(handler(step, plan, context))
            except RuntimeError as exc:
                return AgentResponse(
                    success=False,
                    message=f"Plan execution failed: {exc}",
                    requires_confirmation=False,
                )

            decision = context.get("decision")
            if decision is None:
                return AgentResponse(
                    success=False,
                    message="Business plan finished without producing a utility decision.",
                    requires_confirmation=False,
                )
            
            if decision.selected is None:
                return AgentResponse(
                    success=False,
                    message="No appointments satisfy the required constraints.",
                    requires_confirmation=False,
                )
            recommendation = context.get("recommendation")
            if recommendation is None:
                return AgentResponse(
                    success=False,
                    message="Business plan finished without producing a recommendation.",
                    requires_confirmation=False,
                )

            if decision is None or recommendation is None:
                return AgentResponse(
                    success=False,
                    message="Business plan finished without producing a recommendation.",
                    requires_confirmation=False,
                )
            if decision.selected is None:
                return AgentResponse(
                    success=False,
                    message="No appointments satisfy the required constraints.",
                    requires_confirmation=False,
                )

            # Phase 3: validate the produced recommendation after execution.
            critic = self.critic.review(plan, decision, recommendation)
            executed_steps.append(
                PlanStepExecution(
                    step_id="platform_recommendation_critic_review",
                    action="recommendation_critic_review",
                    status="completed",
                    detail=(
                        "Approved recommendation."
                        if critic.approved
                        else "Rejected recommendation: " + "; ".join(critic.issues)
                    ),
                    source="platform",
                )
            )

            expected_slot_id = decision.selected.slot.slot_id
            model_selected_correctly = recommendation.recommended_slot_id == expected_slot_id
            preference_mismatches = self._soft_preference_mismatches(plan, decision)
            soft_only_rejection = self._critic_issues_are_soft_only(
                critic.issues, preference_mismatches
            )
            approved = model_selected_correctly and (critic.approved or soft_only_rejection)

            if approved:
                executed_steps.append(
                    PlanStepExecution(
                        step_id="platform_request_confirmation",
                        action="request_confirmation",
                        status="completed",
                        detail="Prepared a human-confirmation request; no booking was performed.",
                        source="platform",
                    )
                )
                trace = (
                    AgentTrace(
                        planner=plan,
                        plan_critic=plan_critic,
                        deterministic_decision=decision,
                        utility_agent=recommendation,
                        critic=critic,
                        plan_attempt=attempt + 1,
                        executed_steps=executed_steps,
                    )
                    if include_trace
                    else None
                )
                message = (
                    self._fallback_message(decision, preference_mismatches)
                    if preference_mismatches
                    else critic.final_message
                )
                return AgentResponse(
                    success=True,
                    message=message,
                    recommended_slot_id=expected_slot_id,
                    requires_confirmation=True,
                    trace=trace,
                )

            last_issues = list(critic.issues)
            if not model_selected_correctly:
                last_issues.append(
                    f"Utility agent selected {recommendation.recommended_slot_id}; "
                    f"deterministic winner is {expected_slot_id}."
                )

            if attempt < self.max_plan_retries:
                plan = self.planner.revise_plan(
                    user_message=user_message,
                    previous_plan=plan,
                    critic_issues=last_issues,
                    decision=decision,
                    recommendation=recommendation,
                    rejection_stage="recommendation",
                )

        return AgentResponse(
            success=False,
            message=(
                f"Recommendation failed validation after {self.max_plan_retries + 1} attempts: "
                + "; ".join(last_issues)
            ),
            requires_confirmation=False,
        )

    def handle(
        self,
        user_message: str,
        include_trace: bool = True,
)   -> AgentResponse:
        started_at = perf_counter()

        # Always build the trace for episodic memory. It can be removed from
        # the response returned to the API caller afterward.
        response = self._handle_internal(
            user_message=user_message,
            include_trace=True,
        )

        duration_ms = max(
            0,
            round((perf_counter() - started_at) * 1000),
        )

        episode = AgentExecutionEpisode(
            user_message=user_message,
            outcome=self._classify_episode_outcome(response),
            response=response,
            duration_ms=duration_ms,
        )

        try:
            reflection = self.reflection_agent.reflect(episode)
            episode = episode.model_copy(
                update={
                    "reflection": reflection,
                }
            )
        except Exception as exc:
            episode = episode.model_copy(
                update={
                    "reflection_error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            )

        self.episodic_memory.save(episode)

        if include_trace:
            return response

        return response.model_copy(update={"trace": None})
        
    def _execute_find_appointments(
        self, step: PlanStep, plan: PlannerOutput, context: ExecutionContext
    ) -> PlanStepExecution:
        tool_result = find_appointments(plan.preferences.preferred_specialty)
        if not tool_result.success:
            raise RuntimeError(tool_result.error or "Appointment search failed")
        slots = [AppointmentSlot.model_validate(item) for item in tool_result.data]
        context["slots"] = slots
        #print(f"Found {len(slots)} candidate appointments for specialty {plan.preferences.preferred_specialty}.")
        #print(f"Slots: {[slot.slot_id for slot in slots]}")
        return PlanStepExecution(
            step_id=step.step_id,
            action=step.action.value,
            status="completed",
            detail=f"Found {len(slots)} candidate appointments.",
            source="planner",
        )

    def _execute_evaluate_options(
        self, step: PlanStep, plan: PlannerOutput, context: ExecutionContext
    ) -> PlanStepExecution:
        slots = context.get("slots")
        if not isinstance(slots, list):
            raise RuntimeError("evaluate_options requires find_appointments output")
        #print("Planner preferences:", plan.preferences.model_dump())
        decision = self.evaluator.evaluate(slots, plan.preferences)
        #print(f"Evaluated {decision} candidate appointments.")
        #print("Selected:", decision.selected)

        context["decision"] = decision
        detail = (
            "No candidate satisfied the hard constraints."
            if decision.selected is None
            else "Applied hard constraints and selected deterministic winner "
            f"{decision.selected.slot.slot_id}."
        )
        return PlanStepExecution(
            step_id=step.step_id,
            action=step.action.value,
            status="completed",
            detail=detail,
            source="planner",
        )

    def _execute_rank_options(
        self, step: PlanStep, plan: PlannerOutput, context: ExecutionContext
    ) -> PlanStepExecution:
        decision = context.get("decision")
        if decision is None:
            raise RuntimeError("rank_options requires evaluate_options output")
        if decision.selected is None:
            # There is nothing valid for the utility LLM to rank.
            context["recommendation"] = None
            return PlanStepExecution(
                step_id=step.step_id,
                action=step.action.value,
                status="skipped",
                detail="Skipped ranking because no eligible appointment remained.",
                source="planner",
            )
        recommendation = self.utility_agent.rank(plan.preferences, decision)
        context["recommendation"] = recommendation
        return PlanStepExecution(
            step_id=step.step_id,
            action=step.action.value,
            status="completed",
            detail=f"Produced recommendation {recommendation.recommended_slot_id}.",
            source="planner",
        )

    def _validate_executable_plan(self, plan: PlannerOutput) -> list[str]:
        """Validate registered actions, unique IDs, dependencies, and required data flow."""
        issues: list[str] = []
        steps = plan.steps
        if not steps:
            return ["Plan contains no business steps."]

        ids = [step.step_id for step in steps]
        if len(ids) != len(set(ids)):
            issues.append("Every step_id must be unique.")

        id_set = set(ids)
        position = {step_id: index for index, step_id in enumerate(ids)}
        for step in steps:
            if step.action not in self.action_handlers:
                issues.append(f"Action {step.action.value} has no registered handler.")
            for dependency in step.depends_on:
                if dependency not in id_set:
                    issues.append(f"Step {step.step_id} depends on unknown step {dependency}.")
                elif position[dependency] >= position[step.step_id]:
                    issues.append(f"Step {step.step_id} must appear after dependency {dependency}.")

        action_ids: dict[PlanAction, list[str]] = {}
        for step in steps:
            action_ids.setdefault(step.action, []).append(step.step_id)

        # For the current goal—returning a ranked appointment—these business capabilities
        # are necessary. Critic review and confirmation are deliberately not planner actions.
        required_actions = {
            PlanAction.FIND_APPOINTMENTS,
            PlanAction.EVALUATE_OPTIONS,
            PlanAction.RANK_OPTIONS,
        }
        missing = required_actions - set(action_ids)
        if missing:
            issues.append(
                "Missing required business actions: "
                + ", ".join(sorted(action.value for action in missing))
            )

        def depends_transitively(
            step_id: str, required_id: str, seen: set[str] | None = None
        ) -> bool:
            seen = seen or set()
            if step_id in seen:
                return False
            seen.add(step_id)
            step = next((item for item in steps if item.step_id == step_id), None)
            if step is None:
                return False
            if required_id in step.depends_on:
                return True
            return any(
                depends_transitively(dep, required_id, seen.copy())
                for dep in step.depends_on
            )

        if not missing:
            find_id = action_ids[PlanAction.FIND_APPOINTMENTS][0]
            evaluate_id = action_ids[PlanAction.EVALUATE_OPTIONS][0]
            rank_id = action_ids[PlanAction.RANK_OPTIONS][0]
            checks = [
                (evaluate_id, find_id, "evaluate_options must depend on find_appointments"),
                (rank_id, evaluate_id, "rank_options must depend on evaluate_options"),
            ]
            for step_id, required_id, message in checks:
                if not depends_transitively(step_id, required_id):
                    issues.append(message + ".")

        return issues

    @staticmethod
    def _ordered_steps(plan: PlannerOutput) -> list[PlanStep]:
        return plan.steps

    @staticmethod
    def _soft_preference_mismatches(plan, decision) -> list[str]:
        selected = decision.selected
        if selected is None:
            return []

        mismatches: list[str] = []
        preferences = plan.preferences
        if preferences.preferred_provider and selected.breakdown.provider_preference == 0:
            mismatches.append(f"preferred provider {preferences.preferred_provider}")
        if preferences.preferred_time_of_day and selected.breakdown.time_preference == 0:
            mismatches.append(f"preferred {preferences.preferred_time_of_day} time")
        return mismatches

    @staticmethod
    def _critic_issues_are_soft_only(issues: list[str], mismatches: list[str]) -> bool:
        if not issues or not mismatches:
            return False

        mismatch_terms: set[str] = set()
        for mismatch in mismatches:
            text = mismatch.lower()
            if "time" in text:
                mismatch_terms.update({"time", "morning", "afternoon", "evening"})
            if "provider" in text:
                mismatch_terms.update({"provider", "doctor", "dr."})

        blocking_terms = {
            "confirmation", "confirm", "does not exist", "invent",
            "valid candidates", "supplied candidates", "highest", "deterministic",
            "wrong slot", "invalid", "hard constraint",
        }
        for issue in issues:
            normalized = issue.lower()
            if any(term in normalized for term in blocking_terms):
                return False
            if not any(term in normalized for term in mismatch_terms):
                return False
        return True

    @staticmethod
    def _fallback_message(decision, mismatches: list[str]) -> str:
        selected = decision.selected
        if selected is None:
            return "No eligible appointments were found."

        slot = selected.slot
        missing = " and ".join(mismatches)
        return (
            f"I could not find an available appointment matching your {missing}. "
            f"The best available option that satisfies the required constraints is "
            f"{slot.provider_name} at {slot.clinic_name} on "
            f"{slot.start_time:%A, %B %d at %I:%M %p}, "
            f"{slot.distance_miles:g} miles away. Would you like me to book this slot?"
        )
    
    @staticmethod
    def _classify_episode_outcome(response: AgentResponse) -> EpisodeOutcome:
        if response.success and response.requires_confirmation:
            return EpisodeOutcome.RECOMMENDATION_READY

        message = response.message.lower()

        if message.startswith("please provide:"):
            return EpisodeOutcome.MISSING_INFORMATION

        if message.startswith("plan failed validation"):
            return EpisodeOutcome.PLAN_REJECTED

        if message.startswith("plan execution failed"):
            return EpisodeOutcome.EXECUTION_FAILED

        if "without producing a recommendation" in message:
            return EpisodeOutcome.NO_RECOMMENDATION

        if "no appointments satisfy" in message:
            return EpisodeOutcome.NO_ELIGIBLE_APPOINTMENTS

        if message.startswith("recommendation failed validation"):
            return EpisodeOutcome.RECOMMENDATION_REJECTED

        return EpisodeOutcome.UNKNOWN_FAILURE

    def confirm(self, patient_id: str, slot_id: str) -> dict:
        return schedule_appointment(patient_id, slot_id).model_dump(mode="json")
