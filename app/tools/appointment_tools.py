from __future__ import annotations

from app.schemas.models import AppointmentSlot, ToolResult
from app.store import store


def find_appointments(specialty: str) -> ToolResult:
    slots: list[AppointmentSlot] = store.find_slots(specialty)
    return ToolResult(success=True, data=[s.model_dump(mode="json") for s in slots])


def schedule_appointment(patient_id: str, slot_id: str) -> ToolResult:
    try:
        return ToolResult(success=True, data=store.book(patient_id, slot_id))
    except ValueError as exc:
        return ToolResult(success=False, error=str(exc))
