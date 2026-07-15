from __future__ import annotations

from datetime import datetime, timedelta
from threading import Lock
import re

from app.schemas.models import AppointmentSlot


class InMemoryStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._patients: dict[str, dict] = {}
        self._appointments: dict[str, dict] = {}
        self._slots = self._seed_slots()

    @staticmethod
    def _seed_slots() -> list[AppointmentSlot]:
        base = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=2)
        return [
            AppointmentSlot(slot_id="S1", provider_id="P1", provider_name="Dr. Lee", specialty="primary care", start_time=base, clinic_name="North Clinic", distance_miles=3.2, estimated_cost=30),
            AppointmentSlot(slot_id="S2", provider_id="P2", provider_name="Dr. Patel", specialty="primary care", start_time=base + timedelta(days=1, hours=8), clinic_name="Downtown Clinic", distance_miles=10.5, estimated_cost=20),
            AppointmentSlot(slot_id="S3", provider_id="P1", provider_name="Dr. Lee", specialty="primary care", start_time=base + timedelta(days=3, hours=7), clinic_name="North Clinic", distance_miles=3.2, estimated_cost=30),
            AppointmentSlot(slot_id="S4", provider_id="P3", provider_name="Dr. Nguyen", specialty="primary care", start_time=base + timedelta(days=1, hours=4), clinic_name="Eastside Clinic", distance_miles=5.8, estimated_cost=45),
        ]

    @staticmethod
    def _normalize_specialty(value: str) -> str:
        """Normalize user/LLM wording before matching stored specialties."""
        normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
        aliases = {
            "primarycare": "primary care",
            "primary care physician": "primary care",
            "primary care doctor": "primary care",
            "pcp": "primary care",
        }
        compact = normalized.replace(" ", "")
        return aliases.get(normalized, aliases.get(compact, normalized))

    def find_slots(self, specialty: str) -> list[AppointmentSlot]:
        requested_specialty = self._normalize_specialty(specialty)
        with self._lock:
            booked = {a["slot_id"] for a in self._appointments.values()}
            return [
                s
                for s in self._slots
                if self._normalize_specialty(s.specialty) == requested_specialty
                and s.slot_id not in booked
            ]

    def book(self, patient_id: str, slot_id: str) -> dict:
        with self._lock:
            if any(a["slot_id"] == slot_id for a in self._appointments.values()):
                raise ValueError("Slot is no longer available")
            slot = next((s for s in self._slots if s.slot_id == slot_id), None)
            if not slot:
                raise ValueError("Unknown slot")
            appointment_id = f"A{len(self._appointments) + 1}"
            appointment = {"appointment_id": appointment_id, "patient_id": patient_id, "slot_id": slot_id, "slot": slot.model_dump(mode="json")}
            self._appointments[appointment_id] = appointment
            return appointment


store = InMemoryStore()
