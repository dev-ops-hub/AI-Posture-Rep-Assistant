"""Shared data models (dataclasses) used to pass structured payloads between agents.

These classes define the "wire schema" for the multi-agent pipeline:

- ``PostureViolationEvent`` / ``PostureViolationMetrics``: sent from the Vision Agent
  to the Form Audit Agent when a sustained posture fault is detected.
- ``UserProfile`` / ``SessionSummary`` / ``SessionSummaryPayload``: sent from the
  orchestrator (``main.py`` or ``webapp/session_manager.py``) to the Fitness Coach
  Agent at the end of a workout session.

Each payload dataclass exposes a ``to_dict()`` method that produces a JSON-serializable
representation (with floats rounded for readability) suitable for logging or for
inclusion in an OpenAI API request body.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PostureViolationMetrics:
    """Numeric snapshot of a single posture-violation event.

    Attributes:
        spine_angle_deg: Torso lean angle (degrees) away from vertical at the
            moment the violation was flagged.
        knee_angle_deg: Knee angle (degrees) at the same moment, for context.
        violation_duration_sec: How long the spine angle had continuously
            exceeded the posture threshold before this event was raised.
    """

    spine_angle_deg: float
    knee_angle_deg: float
    violation_duration_sec: float


@dataclass(slots=True)
class PostureViolationEvent:
    """Message sent from the Vision Agent to the Form Audit Agent.

    Carries the metrics captured at the moment of a sustained posture fault
    plus a base64-encoded JPEG snapshot of the frame, so the Form Audit Agent
    can request a biomechanical diagnosis from the vision-capable LLM.
    """

    sender: str
    recipient: str
    event_type: str
    exercise: str
    metrics: PostureViolationMetrics
    image_base64: str

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable dict representation of this event."""
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "event_type": self.event_type,
            "exercise": self.exercise,
            "metrics": {
                "spine_angle_deg": round(self.metrics.spine_angle_deg, 2),
                "knee_angle_deg": round(self.metrics.knee_angle_deg, 2),
                "violation_duration_sec": round(self.metrics.violation_duration_sec, 2),
            },
            "image_base64": self.image_base64,
        }


@dataclass(slots=True)
class UserProfile:
    """Basic user context used to personalize calorie math and coaching copy."""

    weight_kg: float
    goal: str


@dataclass(slots=True)
class SessionSummary:
    """Aggregated statistics for one completed workout session."""

    exercise: str
    total_reps: int
    duration_seconds: float
    met_value: float
    calculated_calories: float
    total_posture_faults: int


@dataclass(slots=True)
class SessionSummaryPayload:
    """Message sent to the Fitness Coach Agent at the end of a session.

    Combines the user's profile, the session's aggregated stats, and the list
    of form-audit diagnostic notes collected during the workout so the coach
    agent can generate (or fall back to a templated) closing summary.
    """

    sender: str
    recipient: str
    user_profile: UserProfile
    session_summary: SessionSummary
    form_audit_diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable dict representation of this payload."""
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "user_profile": {
                "weight_kg": self.user_profile.weight_kg,
                "goal": self.user_profile.goal,
            },
            "session_summary": {
                "exercise": self.session_summary.exercise,
                "total_reps": self.session_summary.total_reps,
                "duration_seconds": round(self.session_summary.duration_seconds, 2),
                "met_value": self.session_summary.met_value,
                "calculated_calories": round(self.session_summary.calculated_calories, 2),
                "total_posture_faults": self.session_summary.total_posture_faults,
            },
            "form_audit_diagnostics": self.form_audit_diagnostics,
        }
