from datetime import datetime, timedelta

from app.agent.utility import UtilityEvaluator
from app.schemas.models import AppointmentSlot, PatientPreferences


def slot(slot_id: str, provider: str, distance: float, hour: int, days: int, cost: float = 30):
    start = (datetime.now() + timedelta(days=days)).replace(hour=hour, minute=0, second=0, microsecond=0)
    return AppointmentSlot(slot_id=slot_id, provider_id=slot_id, provider_name=provider, specialty="primary care", start_time=start, clinic_name="Clinic", distance_miles=distance, estimated_cost=cost)


def test_prefers_requested_provider_when_weight_is_high():
    preferences = PatientPreferences(preferred_provider="Lee", weights={"availability": 0.1, "distance": 0.1, "provider_preference": 0.7, "time_preference": 0.05, "cost": 0.05})
    decision = UtilityEvaluator().evaluate([slot("1", "Dr. Lee", 15, 9, 5), slot("2", "Dr. Patel", 2, 9, 2)], preferences)
    assert decision.selected.slot.provider_name == "Dr. Lee"


def test_hard_distance_constraint_removes_far_slot():
    preferences = PatientPreferences(max_distance_miles=10)
    decision = UtilityEvaluator().evaluate([slot("1", "Dr. Lee", 20, 9, 1), slot("2", "Dr. Patel", 5, 9, 3)], preferences)
    assert decision.selected.slot.slot_id == "2"
