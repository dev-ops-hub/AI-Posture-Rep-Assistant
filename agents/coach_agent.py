from __future__ import annotations

import json
import os

from .models import SessionSummaryPayload

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - depends on local environment
    OpenAI = None


class FitnessCoachAgent:
    def __init__(self, model: str = "gpt-4o-mini", enabled: bool = True) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        self.enabled = enabled and bool(api_key) and OpenAI is not None
        self.model = model
        self.client = OpenAI(api_key=api_key) if self.enabled and OpenAI else None

    @staticmethod
    def calculate_calories(met_value: float, weight_kg: float, duration_seconds: float) -> float:
        duration_hours = max(duration_seconds, 0.0) / 3600.0
        return met_value * weight_kg * duration_hours

    def build_summary(self, payload: SessionSummaryPayload) -> str:
        if not self.enabled or self.client is None:
            return self._fallback_summary(payload)

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "You are a concise fitness coach. Summarize the workout in 3 short sentences.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(payload.to_dict(), indent=2),
                        }
                    ],
                },
            ],
        )
        return (response.output_text or self._fallback_summary(payload)).strip()

    @staticmethod
    def _fallback_summary(payload: SessionSummaryPayload) -> str:
        session = payload.session_summary
        diagnostics = payload.form_audit_diagnostics
        diagnostic_line = diagnostics[-1] if diagnostics else "No major posture faults were captured."
        return (
            f"You completed {session.total_reps} squat reps in {session.duration_seconds:.0f} seconds and burned about "
            f"{session.calculated_calories:.2f} kcal. Posture faults recorded: {session.total_posture_faults}. "
            f"Focus next on this cue: {diagnostic_line}"
        )
