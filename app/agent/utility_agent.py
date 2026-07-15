from __future__ import annotations

from app.agent.utility import UtilityEvaluator
from app.schemas.models import AppointmentSlot, PatientPreferences
from app.tools.appointment_tools import find_appointments, schedule_appointment


class UtilityBasedHealthcareAgent:
    """Legacy deterministic endpoint retained for comparison and unit tests."""

    def __init__(self) -> None:
        self.evaluator = UtilityEvaluator()

    def recommend(self, preferences: PatientPreferences) -> dict:
        result = find_appointments(preferences.preferred_specialty)
        if not result.success:
            return {"success": False, "error": result.error}
        slots = [AppointmentSlot.model_validate(item) for item in result.data]
        decision = self.evaluator.evaluate(slots, preferences)
        return {"success": True, "decision": decision.model_dump(mode="json")}

    def confirm(self, patient_id: str, slot_id: str) -> dict:
        return schedule_appointment(patient_id, slot_id).model_dump(mode="json")
