from __future__ import annotations

from datetime import datetime
from math import exp

from app.schemas.models import (
    AppointmentSlot,
    PatientPreferences,
    ScoredOption,
    UtilityBreakdown,
    UtilityDecision,
)


class UtilityEvaluator:
    """Scores valid appointment options; business rules remain outside the LLM."""

    def evaluate(self, slots: list[AppointmentSlot], preferences: PatientPreferences) -> UtilityDecision:
        valid, constraints = self._apply_hard_constraints(slots, preferences)
        scored = [self._score(slot, preferences) for slot in valid]
        scored.sort(key=lambda option: option.utility_score, reverse=True)
        return UtilityDecision(
            goal="Select the best appointment, not merely the first available appointment",
            selected=scored[0] if scored else None,
            alternatives=scored[1:4],
            constraints_applied=constraints,
            requires_confirmation=True,
        )

    def _apply_hard_constraints(
        self,
        slots: list[AppointmentSlot],
        p: PatientPreferences,
    ) -> tuple[list[AppointmentSlot], list[str]]:
        result = [slot for slot in slots if slot.in_network]
        constraints = ["in-network only"]

        if p.max_distance_miles is not None:
            result = [
                slot
                for slot in result
                if slot.distance_miles <= p.max_distance_miles
            ]
            constraints.append(
                f"distance <= {p.max_distance_miles:g} miles"
            )

        if p.earliest_date:
            earliest = datetime.fromisoformat(p.earliest_date)
            result = [
                slot
                for slot in result
                if slot.start_time >= earliest
            ]
            constraints.append(f"on or after {p.earliest_date}")

        if p.latest_date:
            latest = datetime.fromisoformat(p.latest_date)
            result = [
                slot
                for slot in result
                if slot.start_time <= latest
            ]
            constraints.append(f"on or before {p.latest_date}")

        return result, constraints

    def _score(self, slot: AppointmentSlot, p: PatientPreferences) -> ScoredOption:
        now = datetime.now()
        days_away = max((slot.start_time - now).total_seconds() / 86400, 0)
        availability = exp(-days_away / 10) * 100
        distance_scale = p.max_distance_miles or 25.0
        distance = max(
            0.0,
            100 * (1 - slot.distance_miles / distance_scale),
        )

        provider = 100.0 if p.preferred_provider and p.preferred_provider.lower() in slot.provider_name.lower() else (50.0 if not p.preferred_provider else 0.0)
        time_pref = self._time_score(slot.start_time.hour, p.preferred_time_of_day)
        cost = max(0.0, 100 - slot.estimated_cost)

        breakdown = UtilityBreakdown(
            availability=round(availability, 2),
            distance=round(distance, 2),
            provider_preference=round(provider, 2),
            time_preference=round(time_pref, 2),
            cost=round(cost, 2),
        )
        w = self._normalized_weights(p.weights)
        total = (
            availability * w["availability"]
            + distance * w["distance"]
            + provider * w["provider_preference"]
            + time_pref * w["time_preference"]
            + cost * w["cost"]
        )
        explanation = (
            f"{slot.provider_name} at {slot.clinic_name}; {slot.distance_miles:g} miles away, "
            f"${slot.estimated_cost:g} estimated cost, scheduled {slot.start_time:%a %b %d at %I:%M %p}."
        )
        return ScoredOption(slot=slot, utility_score=round(total, 2), breakdown=breakdown, explanation=explanation)

    @staticmethod
    def _normalized_weights(weights) -> dict[str, float]:
        defaults = PatientPreferences().weights.model_dump()
        supplied = weights.model_dump() if hasattr(weights, "model_dump") else dict(weights)
        merged = {
            key: max(float(supplied.get(key, default_value)), 0.0)
            for key, default_value in defaults.items()
        }
        total = sum(merged.values())
        if total == 0:
            return defaults
        return {key: value / total for key, value in merged.items()}

    @staticmethod
    def _time_score(hour: int, preference: str | None) -> float:
        if not preference:
            return 50.0
        preference = preference.lower()
        matches = {
            "morning": 6 <= hour < 12,
            "afternoon": 12 <= hour < 17,
            "evening": 17 <= hour < 21,
        }
        return 100.0 if matches.get(preference, False) else 0.0
