"""Agent 3: Fitness Coach Agent.

Calculates calories burned for a session and generates a short closing
coaching summary via an OpenAI text model (default: ``gpt-4o-mini``). Falls
back to a deterministic templated summary if no API key is configured, the
``openai`` package isn't installed, or the API call fails.
"""

from __future__ import annotations

import json
import os

from .models import SessionSummaryPayload

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - depends on local environment
    OpenAI = None


class FitnessCoachAgent:
    """Computes calorie burn and produces the end-of-session coaching summary."""

    def __init__(self, model: str = "gpt-4o-mini", enabled: bool = True) -> None:
        """Initializes the agent.

        Args:
            model: OpenAI text model used to generate the session summary.
            enabled: Master switch; automatically disabled if ``OPENAI_API_KEY``
                is not set or the ``openai`` package could not be imported.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        self.enabled = enabled and bool(api_key) and OpenAI is not None
        self.model = model
        self.client = OpenAI(api_key=api_key) if self.enabled and OpenAI else None

    @staticmethod
    def calculate_calories(met_value: float, weight_kg: float, duration_seconds: float) -> float:
        """Estimates calories burned using the standard MET formula.

        ``calories = MET * weight_kg * duration_hours``

        Args:
            met_value: Metabolic Equivalent of Task for the exercise performed.
            weight_kg: User's body weight in kilograms.
            duration_seconds: Total session duration in seconds (clamped to >= 0).

        Returns:
            Estimated kilocalories burned during the session.
        """
        duration_hours = max(duration_seconds, 0.0) / 3600.0
        return met_value * weight_kg * duration_hours

    def build_summary(self, payload: SessionSummaryPayload) -> str:
        """Returns a short coaching summary for the given session payload.

        Sends the full session payload (as JSON) to the OpenAI text model
        with a "concise fitness coach" system prompt asking for a 3-sentence
        summary. Falls back to ``_fallback_summary`` if the agent is disabled
        or the model returns an empty response.

        Args:
            payload: Aggregated session stats, user profile, and form-audit
                diagnostics collected during the workout.

        Returns:
            A short (typically 2-3 sentence) coaching summary string.
        """
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
        """Deterministic, templated summary used when the OpenAI call is unavailable."""
        session = payload.session_summary
        diagnostics = payload.form_audit_diagnostics
        diagnostic_line = diagnostics[-1] if diagnostics else "No major posture faults were captured."
        return (
            f"You completed {session.total_reps} squat reps in {session.duration_seconds:.0f} seconds and burned about "
            f"{session.calculated_calories:.2f} kcal. Posture faults recorded: {session.total_posture_faults}. "
            f"Focus next on this cue: {diagnostic_line}"
        )
