from __future__ import annotations

from flask import Flask, jsonify, request
from pydantic import ValidationError

from app.agent.orchestrator import UtilityHealthcareOrchestrator
from app.agent.utility_agent import UtilityBasedHealthcareAgent
from app.schemas.models import PatientPreferences

app = Flask(__name__)
legacy_agent = UtilityBasedHealthcareAgent()
_orchestrator: UtilityHealthcareOrchestrator | None = None


def get_orchestrator() -> UtilityHealthcareOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = UtilityHealthcareOrchestrator()
    return _orchestrator


@app.get("/health")
def health():
    return jsonify({"status": "ok", "agent_type": "LLM utility-based multi-agent"})


@app.post("/agent/chat")
def chat():
    body = request.get_json(force=True) or {}
    message = body.get("message")
    if not message:
        return jsonify({"success": False, "error": "message is required"}), 400
    try:
        response = get_orchestrator().handle(message, include_trace=body.get("include_trace", True))
        return jsonify(response.model_dump(mode="json")), (200 if response.success else 422)
    except RuntimeError as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.post("/agent/recommend")
def recommend():
    """Deterministic comparison endpoint; it does not call a model."""
    try:
        body = request.get_json(force=True) or {}
        preferences = PatientPreferences.model_validate(body.get("preferences", body))
        return jsonify(legacy_agent.recommend(preferences))
    except ValidationError as exc:
        return jsonify({"success": False, "error": "Invalid preferences", "details": exc.errors()}), 400


@app.post("/agent/confirm")
def confirm():
    body = request.get_json(force=True) or {}
    patient_id, slot_id = body.get("patient_id"), body.get("slot_id")
    if not patient_id or not slot_id:
        return jsonify({"success": False, "error": "patient_id and slot_id are required"}), 400
    result = legacy_agent.confirm(patient_id, slot_id)
    return jsonify(result), (200 if result["success"] else 409)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)
