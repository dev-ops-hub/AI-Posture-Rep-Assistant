from __future__ import annotations

import os

from .models import PostureViolationEvent

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - depends on local environment
    OpenAI = None


class FormAuditAgent:
    def __init__(self, model: str = "gpt-4o", enabled: bool = True) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        self.enabled = enabled and bool(api_key) and OpenAI is not None
        self.model = model
        self.client = OpenAI(api_key=api_key) if self.enabled and OpenAI else None

    def audit_posture(self, event: PostureViolationEvent) -> str:
        if not self.enabled or self.client is None or not event.image_base64:
            return self._fallback_feedback(event)

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "You are a biomechanics coach. Give one concise squat correction in 1-2 sentences.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Audit this squat frame. Focus on spine alignment and knee tracking. "
                                f"Spine angle: {event.metrics.spine_angle_deg:.1f} deg. "
                                f"Knee angle: {event.metrics.knee_angle_deg:.1f} deg."
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": event.image_base64,
                        },
                    ],
                },
            ],
        )
        return (response.output_text or self._fallback_feedback(event)).strip()

    @staticmethod
    def _fallback_feedback(event: PostureViolationEvent) -> str:
        if event.metrics.spine_angle_deg > 20:
            return "Your torso is folding forward. Brace your core and keep your chest stacked over your hips on the descent."
        return "Your squat is drifting out of position. Slow the rep slightly and keep your spine neutral as you stand up."
